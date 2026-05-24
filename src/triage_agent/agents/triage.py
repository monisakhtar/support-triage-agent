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

For each incoming ticket, you must:
  1. Use search_kb when the ticket mentions a topic that might be covered
     by past tickets or policy docs (refunds, billing, login, API limits, etc).
  2. Use get_account_status when billing, subscription, or account state is
     relevant to your verdict.
  3. Once you have enough information, call submit_decision with your final
     verdict. The submit_decision arguments must include:
       - urgency: one of low, medium, high, critical
       - category: one of billing, technical, account, feature_request, other
       - suggested_action: auto_resolve, draft_reply, or escalate
       - confidence: a float between 0.0 and 1.0
       - reasoning: a short explanation (10-2000 characters)

Rules:
  - Never compute or guess. If you don't know, use a tool.
  - Be conservative: when in doubt, prefer escalate over auto_resolve.
  - Critical urgency is reserved for security issues, data loss, or
    revenue-impacting outages affecting Enterprise customers.
  - Always call submit_decision exactly once when ready. Do not reply with
    plain text — every turn should be a tool call.
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