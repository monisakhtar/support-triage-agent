"""FastAPI application entry point.

This module defines the ASGI app, its lifecycle, and the top-level routes.
Larger applications split routes into multiple router modules; this one is
small enough to keep everything in one file for now.

Run locally with:
    uvicorn triage_agent.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from triage_agent.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
# These Pydantic models describe the *shape* of our HTTP responses.
# FastAPI uses them to:
#   - Serialize the return value into JSON
#   - Validate the response matches the declared shape
#   - Generate the OpenAPI schema visible at /docs


class HealthResponse(BaseModel):
    """Payload returned by GET /health."""

    status: str
    env: str
    llm_provider: str
    llm_model: str


# ---------------------------------------------------------------------------
# Lifespan: startup & shutdown
# ---------------------------------------------------------------------------
# An async context manager that runs once at app boot (before `yield`) and
# once at shutdown (after `yield`). This is where we'd open DB pools, warm
# caches, or connect to message brokers. Right now we just read settings to
# fail fast if config is broken.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """App startup and shutdown hooks."""
    settings = get_settings()
    print(f"[startup] env={settings.app.env} provider={settings.llm.provider.value} model={settings.llm.model}")
    yield
    print("[shutdown] bye")


# ---------------------------------------------------------------------------
# The app itself
# ---------------------------------------------------------------------------
# FastAPI() builds the ASGI application. Title and description show up in
# the auto-generated /docs page.


app = FastAPI(
    title="Support Triage Agent",
    description="Provider-agnostic agentic AI customer support triage system",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Typed dependency alias
# ---------------------------------------------------------------------------
# `Annotated[Settings, Depends(get_settings)]` says: "this parameter is of
# type Settings, and FastAPI should obtain it by calling get_settings()."
# Defining the alias once keeps handler signatures readable.

SettingsDep = Annotated[Settings, Depends(get_settings)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Liveness + config sanity probe.

    Returns the current environment and which LLM provider is wired up.
    Useful in CI/CD: if /health responds with the wrong provider, the
    deployment is misconfigured.
    """
    return HealthResponse(
        status="ok",
        env=settings.app.env,
        llm_provider=settings.llm.provider.value,
        llm_model=settings.llm.model,
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Friendly landing message for humans hitting the root URL in a browser."""
    return {
        "service": "support-triage-agent",
        "docs": "/docs",
        "health": "/health",
        "tagline": "agents need async", 
    }