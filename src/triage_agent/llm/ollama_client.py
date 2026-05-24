"""Ollama provider — talks to a local Ollama server via its OpenAI-compatible API.

Ollama exposes /v1/chat/completions on http://localhost:11434/v1, which speaks
the same request/response shape as OpenAI. We use the official `openai` Python
SDK pointed at that local URL — same client, different base_url.

Why a separate class from OpenAIProvider? Two reasons:
  1. Their authentication, defaults, and error semantics differ enough that
     branching inside one class would obscure both.
  2. Keeping them separate makes it easy to swap one without touching the other.

When the duplication starts hurting we'll extract a shared parent — but
premature abstraction is a bigger sin than a little repetition.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from openai import AsyncOpenAI, APIError, APITimeoutError
from openai.types.chat import ChatCompletion

from triage_agent.llm.base import (
    ChatResponse,
    LLMProvider,
    Message,
    TokenUsage,
    ToolCall,
    ToolSchema,
)

logger = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server.

    Construction is cheap — the underlying httpx connection pool is created
    lazily on first request. Always call `await provider.close()` on shutdown.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        timeout_seconds: float = 60.0,
    ) -> None:
        """Build an Ollama client.

        Args:
            model: Ollama model name (e.g. "llama3.2:latest").
            base_url: Ollama's OpenAI-compatible endpoint. Override only
                when running Ollama on a non-default host or port.
            timeout_seconds: Per-request timeout. Local inference can be slow.
        """
        self.model = model
        self._client = AsyncOpenAI(
            api_key="ollama-no-key-needed",  # Ollama ignores this, but the SDK requires non-empty
            base_url=base_url,
            timeout=timeout_seconds,
        )
        logger.info(
            "ollama_provider_initialized",
            model=model,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Send a chat completion request to Ollama and normalize the response."""
        try:
            completion: ChatCompletion = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APITimeoutError as exc:
            logger.warning("ollama_chat_timeout", model=self.model, error=str(exc))
            raise
        except APIError as exc:
            logger.error("ollama_chat_api_error", model=self.model, error=str(exc))
            raise

        response = self._to_chat_response(completion)
        logger.debug(
            "ollama_chat_completed",
            model=self.model,
            finish_reason=response.finish_reason,
            tool_calls=len(response.tool_calls),
            total_tokens=response.usage.total_tokens,
        )
        return response

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.close()
        logger.info("ollama_provider_closed")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _to_chat_response(completion: ChatCompletion) -> ChatResponse:
        """Translate the OpenAI SDK's ChatCompletion into our domain type.

        This is the boundary where provider-specific quirks die — everything
        downstream sees only our clean ChatResponse.
        """
        choice = completion.choices[0]
        message = choice.message

        # Parse tool_calls. The model returns arguments as a JSON STRING;
        # we parse it to a dict so the agent loop doesn't have to.
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for call in message.tool_calls:
                arguments = _parse_tool_arguments(call.function.arguments)
                tool_calls.append(
                    ToolCall(
                        id=call.id,
                        name=call.function.name,
                        arguments=arguments,
                    )
                )

        usage = TokenUsage(
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
            total_tokens=completion.usage.total_tokens if completion.usage else 0,
        )

        return ChatResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            raw=completion.model_dump(),
        )


def _parse_tool_arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse the model's tool arguments into a real dict.

    OpenAI-style APIs return arguments as a JSON string (e.g. '{"x": 1}').
    Defensive: if a future SDK version starts returning a dict directly,
    accept that too.
    """
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("tool_arguments_malformed_json", raw=raw)
        return {}