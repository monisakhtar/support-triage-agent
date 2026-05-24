"""High-level triage function — the API the FastAPI endpoint calls.

This file is the domain-specific glue between the generic agent loop and the
ticket-triage problem. It:
  - Builds the system prompt that frames the LLM as a triage specialist
  - Formats the ticket into the initial user message
  - Calls run_agent_loop with the TriageDecision schema
  - Packages the result for the HTTP layer

The loop in loop.py doesn't know what a 'ticket' is. This file does.
"""

from __future__ import annotations

import structlog

from triage_agent.agents.loop import (
    AgentError,
    MaxIterationsExceeded,
    run_agent_loop,
)
from triage_agent.agents.types import (
    AgentTrace,
    Ticket,
    TriageDecision,
    TriageResponse,
)
from triage_agent.llm.base import LLMProvider, Message

logger = structlog.get_logger(__name__)


SYSTEM_PROMPT = """\
You are a customer support triage specialist for a SaaS company.

CRITICAL: You MUST follow this exact sequence for every ticket. Skipping
steps is not allowed.

STEP 1 — Gather context:
  - Call search_kb at least once with a query derived from the ticket subject
    or body. ALWAYS do this, even if you think you know the answer.
  - Call get_account_status with the ticket's customer_id when the ticket
    mentions billing, subscription, charges, refunds, or account access.
  - You may call both tools in the same turn.

STEP 2 — Decide:
  - Only after you've received tool results, call submit_decision with:
      urgency: low | medium | high | critical
      category: billing | technical | account | feature_request | other
      suggested_action: auto_resolve | draft_reply | escalate
      confidence: a float 0.0-1.0 reflecting your certainty AFTER tool results
      reasoning: 1-3 sentences citing what the tools returned

Rules:
  - NEVER call submit_decision in the first turn. You have no information yet.
  - NEVER reply with plain text. Every turn must be a tool call.
  - confidence below 0.5 means escalate (you lack information; ask a human).
  - critical is reserved for security issues, data loss, or revenue-impacting
    outages affecting Enterprise customers.
  - The reasoning field must reference specific facts from search_kb or
    get_account_status results — not generic statements.
"""


def _format_ticket(ticket: Ticket) -> str:
    """Render a Ticket as the first user message."""
    return (
        f"Ticket ID: {ticket.ticket_id}\n"
        f"Customer ID: {ticket.customer_id}\n"
        f"Customer Tier: {ticket.customer_tier.value}\n"
        f"Subject: {ticket.subject}\n\n"
        f"Body:\n{ticket.body}"
    )


async def triage_ticket(
    *,
    ticket: Ticket,
    llm: LLMProvider,
    max_iterations: int = 5,
    max_tokens_per_call: int = 1024,
    temperature: float = 0.1,
) -> TriageResponse:
    """Triage a single ticket and return a structured decision + trace.

    Raises:
        MaxIterationsExceeded: agent didn't converge.
        AgentError: any other agent-side failure.
    """
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_ticket(ticket)},
    ]

    logger.info(
        "triage_started",
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id,
    )

    try:
        result = await run_agent_loop(
            llm=llm,
            initial_messages=messages,
            decision_model=TriageDecision,
            max_iterations=max_iterations,
            max_tokens_per_call=max_tokens_per_call,
            temperature=temperature,
        )
    except AgentError:
        logger.exception("triage_failed", ticket_id=ticket.ticket_id)
        raise

    decision = result.decision
    assert isinstance(decision, TriageDecision), "loop returned wrong model"

    logger.info(
        "triage_finished",
        ticket_id=ticket.ticket_id,
        urgency=decision.urgency.value,
        category=decision.category.value,
        suggested_action=decision.suggested_action,
        iterations=result.trace.iterations_used,
        total_tokens=result.trace.total_tokens,
    )

    return TriageResponse(
        ticket_id=ticket.ticket_id,
        decision=decision,
        trace=result.trace,
    )