"""LLM provider abstraction package.

The public API for callers:

    from triage_agent.llm import build_llm_provider, LLMProvider

    provider = build_llm_provider()
    response = await provider.chat([{"role": "user", "content": "hi"}])
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
from triage_agent.llm.factory import build_llm_provider
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
    "build_llm_provider",
]