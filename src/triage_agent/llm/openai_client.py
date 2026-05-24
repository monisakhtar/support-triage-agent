"""OpenAI provider — talks to OpenAI's hosted API.

Structurally identical to OllamaProvider:
  - Same OpenAI SDK
  - Same request/response shape
  - Same translation logic

The real differences are operational:
  - Real API key required (and is a secret)
  - Real money per call (cost matters; we cap max_tokens)
  - Different default base_url (the SDK default points at api.openai.com)
  - More transient errors (rate limits, server hiccups) that may warrant retry

When the third provider arrives we'll extract the shared parts into a common
parent. For now the duplication is honest and easy to read.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
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


class OpenAIProvider(LLMProvider):
    """LLM provider backed by OpenAI's hosted API."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
    ) -> None:
        """Build an OpenAI client.

        Args:
            model: OpenAI model name (e.g. "gpt-4o-mini").
            api_key: OpenAI API key. Never log this.
            base_url: API endpoint. Override for OpenAI-compatible proxies
                (Together AI, Groq, etc.).
            timeout_seconds: Per-request timeout. OpenAI is faster than
                Ollama, so the default is lower.
        """
        if not api_key or api_key.startswith("ollama-"):
            raise ValueError(
                "OpenAIProvider requires a real OpenAI API key. "
                "Set LLM_API_KEY in .env."
            )

        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )
        logger.info(
            "openai_provider_initialized",
            model=model,
            base_url=base_url,
            timeout=timeout_seconds,
            # Note: api_key NEVER appears in logs.
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Send a chat completion request to OpenAI and normalize the response."""
        try:
            completion: ChatCompletion = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError as exc:
            logger.warning("openai_rate_limited", model=self.model, error=str(exc))
            raise
        except APITimeoutError as exc:
            logger.warning("openai_chat_timeout", model=self.model, error=str(exc))
            raise
        except APIError as exc:
            logger.error("openai_chat_api_error", model=self.model, error=str(exc))
            raise

        response = self._to_chat_response(completion)
        logger.debug(
            "openai_chat_completed",
            model=self.model,
            finish_reason=response.finish_reason,
            tool_calls=len(response.tool_calls),
            total_tokens=response.usage.total_tokens,
        )
        return response

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.close()
        logger.info("openai_provider_closed")

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _to_chat_response(completion: ChatCompletion) -> ChatResponse:
        """Translate the OpenAI SDK's ChatCompletion into our domain type."""
        choice = completion.choices[0]
        message = choice.message

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
    """Parse the model's tool arguments into a real dict."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("tool_arguments_malformed_json", raw=raw)
        return {}