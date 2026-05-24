"""Factory for constructing the configured LLM provider.

This is the *only* place in the codebase that branches on which provider
to use. The rest of the application calls `build_llm_provider()` and
receives an `LLMProvider` — pure abstraction. Adding a new provider means
adding a single `elif` branch here; no other file changes.
"""

from __future__ import annotations

import structlog

from triage_agent.config import LLMProvider, LLMSettings, get_settings
from triage_agent.llm.base import LLMProvider as LLMProviderBase
from triage_agent.llm.ollama_client import OllamaProvider
from triage_agent.llm.openai_client import OpenAIProvider

logger = structlog.get_logger(__name__)


def build_llm_provider(settings: LLMSettings | None = None) -> LLMProviderBase:
    """Construct the LLM provider declared in settings.

    Args:
        settings: Optional override. When None, reads from the global
            Settings singleton. Pass an explicit value in tests so you
            don't depend on environment state.

    Returns:
        A ready-to-use LLM provider matching `settings.provider`.

    Raises:
        ValueError: If `settings.provider` is a value we don't have a
            concrete implementation for. Should be unreachable in
            practice because the Enum already restricts the input.
    """
    if settings is None:
        settings = get_settings().llm

    logger.info(
        "building_llm_provider",
        provider=settings.provider.value,
        model=settings.model,
    )

    if settings.provider is LLMProvider.OLLAMA:
        return OllamaProvider(
            model=settings.model,
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
        )

    if settings.provider is LLMProvider.OPENAI:
        return OpenAIProvider(
            model=settings.model,
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
        )

    # Unreachable: the LLMProvider enum forbids other values. Belt and suspenders.
    raise ValueError(f"Unsupported LLM provider: {settings.provider!r}")