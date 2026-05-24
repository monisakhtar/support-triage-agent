"""LLM provider abstraction package.

Re-exports the public API so callers can do:

    from triage_agent.llm import LLMProvider, ChatResponse, ToolCall
"""

from triage_agent.llm.base import (
    ChatResponse,
    LLMProvider,
    Message,
    Role,
    TokenUsage,
    ToolCall,
    ToolSchema,
)

__all__ = [
    "ChatResponse",
    "LLMProvider",
    "Message",
    "Role",
    "TokenUsage",
    "ToolCall",
    "ToolSchema",
]