"""Domain models for the triage agent.

These Pydantic models define the inputs the API receives, the outputs the
agent must produce, and the internal state we trace for observability.

Why Pydantic specifically?
- Validation at the API boundary (FastAPI uses these directly)
- Validation of LLM outputs (the agent emits a TriageDecision; if the model
  hallucinates an invalid field, Pydantic rejects it at parse time)
- One source of truth for JSON Schema — the same model generates the schema
  we expose to the LLM and the schema FastAPI uses for /docs

Keeping these in their own module makes them easy to import without dragging
in the agent loop or tools — useful for tests and for documentation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums — closed sets of values
# ---------------------------------------------------------------------------
# We use string-valued Enums so values serialize cleanly to JSON ("high"
# instead of <Urgency.HIGH: 'high'>). Subclassing str makes them comparable
# to plain strings, which the LLM will return.


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


# `Literal` is used here instead of an Enum because the action set is small,
# stable, and unlikely to grow. Literal keeps the JSON schema simpler.
SuggestedAction = Literal["auto_resolve", "draft_reply", "escalate"]


class CustomerTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# ---------------------------------------------------------------------------
# Inputs: the ticket coming in
# ---------------------------------------------------------------------------


class Ticket(BaseModel):
    """A support ticket the agent is asked to triage.

    Validation rules:
        - subject must be 3-200 chars (longer is a copy-paste error)
        - body is bounded too; the model context can't handle infinite text
        - ticket_id and customer_id are strings to allow any ID format
    """

    ticket_id: str = Field(..., min_length=1, max_length=64)
    customer_id: str = Field(..., min_length=1, max_length=64)
    customer_tier: CustomerTier = CustomerTier.FREE
    subject: str = Field(..., min_length=3, max_length=200)
    body: str = Field(..., min_length=1, max_length=10_000)


# ---------------------------------------------------------------------------
# Output: the decision the agent must produce
# ---------------------------------------------------------------------------


class TriageDecision(BaseModel):
    """The structured verdict an agent returns for a ticket.

    The LLM emits this by 'calling' a submit_decision tool whose argument
    schema is exactly this model. The orchestrator extracts the arguments
    and validates them through Pydantic — if validation fails (invalid
    urgency, missing field), we loop again and ask the model to retry.
    """

    urgency: Urgency
    category: Category
    suggested_action: SuggestedAction
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence in this decision (0.0-1.0)",
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=2_000,
        description="Short explanation of why the agent chose this verdict",
    )


# ---------------------------------------------------------------------------
# Internal state / observability
# ---------------------------------------------------------------------------
# These don't go over the wire from the LLM — they're how the orchestrator
# tracks what happened. The API response embeds them so callers can see
# the trace and tune the agent.


class ToolCallRecord(BaseModel):
    """One recorded tool invocation within an agent run."""

    iteration: int
    tool: str
    arguments: dict
    result_preview: str = Field(..., max_length=500)
    duration_ms: float
    error: str | None = None


class AgentTrace(BaseModel):
    """Everything that happened during a single triage run.

    Returned alongside the TriageDecision so callers can audit the agent's
    behavior — which tools fired, how long things took, how many tokens.
    """

    iterations_used: int
    total_tokens: int
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# The combined response the /triage endpoint returns
# ---------------------------------------------------------------------------


class TriageResponse(BaseModel):
    """The full HTTP response from POST /triage."""

    ticket_id: str
    decision: TriageDecision
    trace: AgentTrace