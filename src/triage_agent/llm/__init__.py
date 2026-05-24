"""LLM provider abstraction package.

Re-exports the public API so callers can do:

    from triage_agent.llm import LLMProvider, OllamaProvider, OpenAIProvider
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
from triage_agent.llm.ollama_client import OllamaProvider
from triage_agent.llm.openai_client import OpenAIProvider

__all__ = [
    "ChatResponse",
    "LLMProvider",
    "Message",
    "OllamaProvider",
    "OpenAIProvider",
    "Role",
    "TokenUsage",
    "ToolCall",
    "ToolSchema",
]