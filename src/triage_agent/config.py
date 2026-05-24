"""Application configuration with provider-agnostic LLM support.

Settings are loaded from environment variables and .env files.
Validation happens at startup, so misconfigured environments
fail fast with clear error messages instead of crashing in
production at runtime.
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers.

    Subclassing `str` lets us compare like a string while keeping
    the type-safety of an Enum: `settings.llm.provider == "ollama"` works.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"


class LLMSettings(BaseSettings):
    """LLM provider configuration.

    Switching from Ollama to OpenAI is purely a config change:
    set LLM_PROVIDER=openai in .env and restart. No code changes.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="Which LLM provider to use",
    )
    model: str = Field(
        default="llama3.2:latest",
        description="Model name to send in API calls",
    )
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible endpoint URL",
    )
    api_key: SecretStr = Field(
        default=SecretStr("ollama-no-key-needed"),
        description="API key (Ollama ignores this, OpenAI requires it)",
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Per-request timeout to prevent hung calls",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; low values for tool-using agents",
    )


class AgentSettings(BaseSettings):
    """Agent loop guardrails.

    These caps prevent runaway loops that burn API credits or hang the server.
    They are enforced regardless of provider.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max think-act cycles before forcing a final answer",
    )
    max_tokens_per_call: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Max tokens per LLM completion",
    )
    daily_budget_usd: float = Field(
        default=0.50,
        ge=0.0,
        description="Soft daily spend cap; agent falls back to Ollama above this",
    )


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "staging", "prod"] = Field(
        default="dev",
        description="Deployment environment",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity",
    )
    log_json: bool = Field(
        default=False,
        description="Emit logs as JSON (true in prod, false locally)",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def uppercase_log_level(cls, v: str) -> str:
        """Accept lowercase log levels and uppercase them.

        Common UX: people write `info` not `INFO`. Be forgiving.
        """
        return v.upper() if isinstance(v, str) else v


class Settings:
    """Container that bundles all sub-configs.

    Not a Pydantic model itself; just a plain class that holds the
    three settings objects so code can do `settings.llm.provider`.
    """

    def __init__(self) -> None:
        self.app = AppSettings()
        self.llm = LLMSettings()
        self.agent = AgentSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance.

    Cached so we read environment variables exactly once per process.
    Use this everywhere instead of constructing Settings() directly.
    """
    return Settings()