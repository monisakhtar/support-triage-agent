"""FastAPI application entry point.

This module defines the ASGI app, its lifecycle, and the top-level routes.
Larger applications split routes into multiple router modules; this one is
small enough to keep everything in one file for now.

Run locally with:
    uv run serve
    # or
    uvicorn triage_agent.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

from triage_agent.config import Settings, get_settings
from triage_agent.llm import LLMProvider, build_llm_provider


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Payload returned by GET /health."""

    status: str
    env: str
    llm_provider: str
    llm_model: str


# ---------------------------------------------------------------------------
# Lifespan: startup & shutdown
# ---------------------------------------------------------------------------
# Build the LLM provider once at startup, store it on app.state, close it
# cleanly on shutdown. Endpoints access it via the get_llm dependency.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """App startup and shutdown hooks."""
    settings = get_settings()
    print(
        f"[startup] env={settings.app.env} "
        f"provider={settings.llm.provider.value} "
        f"model={settings.llm.model}"
    )

    llm = build_llm_provider()
    app.state.llm = llm

    try:
        yield
    finally:
        await llm.close()
        print("[shutdown] bye")


# ---------------------------------------------------------------------------
# The app itself
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Support Triage Agent",
    description="Provider-agnostic agentic AI customer support triage system",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_llm(request: Request) -> LLMProvider:
    """Retrieve the singleton LLM provider stored on app.state."""
    return request.app.state.llm


SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMDep = Annotated[LLMProvider, Depends(get_llm)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Liveness + config sanity probe."""
    return HealthResponse(
        status="ok",
        env=settings.app.env,
        llm_provider=settings.llm.provider.value,
        llm_model=settings.llm.model,
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Friendly landing message for humans."""
    return {
        "service": "support-triage-agent",
        "docs": "/docs",
        "health": "/health",
        "llm_ping": "/llm/ping",
    }


@app.get("/llm/ping")
async def llm_ping(llm: LLMDep) -> dict[str, str]:
    """Sanity-check endpoint: do a one-shot chat with the active LLM."""
    response = await llm.chat(
        messages=[{"role": "user", "content": "Reply with the single word PONG."}],
        max_tokens=10,
    )
    return {
        "provider": llm.name,
        "model": llm.model,
        "reply": response.content.strip(),
        "tokens": str(response.usage.total_tokens),
    }


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Boot the development server."""
    import uvicorn

    uvicorn.run(
        "triage_agent.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    run()