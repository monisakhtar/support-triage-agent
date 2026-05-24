"""Provider-agnostic LLM interface.

The agent and orchestration layers depend ONLY on the abstract `LLMProvider`
class defined here. Concrete implementations (Ollama, OpenAI, mocks for tests)
live in sibling modules and conform to this contract.

This is the Dependency Inversion Principle in practice: swap providers via
config, mock in tests, add new providers by implementing the interface —
all without touching the agent code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------
# Every provider speaks the same conversational format: a list of messages,
# each with a role and content. We use plain dicts (typed via TypedDict-ish
# semantics) because the OpenAI SDK already expects this shape — no need to
# build dataclasses and then serialize them.


Role = Literal["system", "user", "assistant", "tool"]
"""Conversational roles. 'tool' messages carry the result of a tool call."""

Message = dict[str, Any]
"""A single chat message. Shape varies slightly by role:

    {"role": "system",    "content": "You are a helpful assistant."}
    {"role": "user",      "content": "What's the weather in Paris?"}
    {"role": "assistant", "content": "...", "tool_calls": [...]}
    {"role": "tool",      "content": "...", "tool_call_id": "call_abc"}
"""

ToolSchema = dict[str, Any]
"""An OpenAI-format tool definition. We don't define a strict shape because
the OpenAI SDK consumes raw dicts — re-modelling them would add no safety."""


# ---------------------------------------------------------------------------
# Response data classes
# ---------------------------------------------------------------------------
# These describe what every provider returns from chat(). Using @dataclass
# keeps them simple, immutable-ish, and free from external dependencies.


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation requested by the model.

    The model says: 'call tool X with these arguments and tell me the result'.
    Our agent loop executes the tool and feeds the result back.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Per-call token accounting. Used for cost tracking and observability."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """What every provider returns from `chat()`.

    Either `content` is non-empty (the model gave a final answer) OR
    `tool_calls` is non-empty (the model wants tool results before continuing).
    They're rarely both meaningful — the agent loop branches on which is set.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The abstract interface
# ---------------------------------------------------------------------------
# Every concrete provider subclasses LLMProvider and implements `chat`.
# That's the only required method. Future additions (streaming, embeddings)
# go here so all providers implement them uniformly.


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement `chat`. Add new providers by creating a new
    subclass and registering it in `factory.py` — agent code stays untouched.
    """

    name: str = "abstract"
    """Short identifier used in logs and error messages."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Send a chat completion request and return the model's response.

        Args:
            messages: Conversation history in OpenAI format.
            tools: Optional tool schemas. If provided, the model may return
                tool_calls instead of a final answer.
            temperature: Sampling temperature. Keep low (0.0-0.2) for tool use.
            max_tokens: Cap on response length. Defends against runaway costs.

        Returns:
            A ChatResponse with either `content` populated (final answer) or
            `tool_calls` populated (model wants tool execution).
        """
        ...

    async def close(self) -> None:
        """Release resources (connection pools, sockets).

        Default no-op. Providers with persistent connections override this.
        """
        return None