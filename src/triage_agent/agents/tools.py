"""Tool registry for the triage agent.

A "tool" is any Python function the LLM can choose to call. We expose them
in two forms:

  1. A JSON schema (the OpenAI tools format) that gets sent to the LLM.
     The model uses the schema to decide WHICH tool to call and WITH WHAT
     arguments. The LLM never executes anything — it just emits intent.

  2. A Python callable that the orchestrator invokes when the LLM picks
     this tool. The callable does the actual work.

By co-locating schema and implementation in one Tool record, drift is impossible.

For now the tools use static fake data so we can build and test the loop
without a database. Step 25+ will wire these to Postgres + pgvector.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import structlog

from triage_agent.agents.types import TriageDecision

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool record: schema + implementation, bound together
# ---------------------------------------------------------------------------


# Tool implementations are async functions taking a dict of arguments and
# returning a string. Strings keep the contract simple — the LLM just needs
# something it can read in the next turn.
ToolFunc = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class Tool:
    """One tool the agent can call. Schema + implementation bound together."""

    name: str
    description: str
    schema: dict[str, Any]
    func: ToolFunc


# ---------------------------------------------------------------------------
# Tool 1: search_kb — semantic search over the knowledge base
# ---------------------------------------------------------------------------
# Production: query pgvector for similar past tickets and KB articles.
# For now: static lookup table by keyword.

_FAKE_KB: dict[str, list[str]] = {
    "refund": [
        "Refund policy: full refunds within 30 days of purchase.",
        "Past ticket #4421: double-charge case resolved by reversing the duplicate.",
    ],
    "billing": [
        "Stripe Connect powers all our charges; check Stripe dashboard before issuing refund.",
        "Pro tier billed monthly on the day of signup.",
    ],
    "login": [
        "Password reset link goes to the email on the account.",
        "If a user disabled MFA, they need to re-enable it after reset.",
    ],
    "api": [
        "Rate limits: 100 req/min on Free, 1000 req/min on Pro, custom on Enterprise.",
        "API key rotation does not invalidate active sessions.",
    ],
    "downtime": [
        "Status page: status.example.com",
        "Last incident: 2025-12-04, partial outage 14 minutes in us-east-1.",
    ],
}


async def search_kb(args: dict[str, Any]) -> str:
    """Search the knowledge base for articles related to a query.

    Args:
        args: must contain 'query' (str).

    Returns:
        A string with matching KB entries, or a 'no results' notice.
    """
    query = str(args.get("query", "")).lower().strip()
    if not query:
        return "Error: 'query' argument is required and must be non-empty."

    matches: list[str] = []
    for keyword, entries in _FAKE_KB.items():
        if keyword in query:
            matches.extend(entries)

    if not matches:
        return f"No KB entries matched query: {query!r}"

    # Cap output. Long tool results inflate the next prompt's token count.
    return "Knowledge base results:\n" + "\n".join(f"- {m}" for m in matches[:5])


SEARCH_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": (
            "Search the internal knowledge base for articles, past resolved "
            "tickets, and policy documents relevant to a query. Use this "
            "before deciding to escalate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search query, e.g. 'refund', 'login issue'",
                },
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 2: get_account_status — look up a customer's account state
# ---------------------------------------------------------------------------
# Production: query the customers table or call the billing service.
# For now: deterministic fake by hash so different customer_ids look different.

_FAKE_STATUSES = ["active", "past_due", "trial", "cancelled"]


async def get_account_status(args: dict[str, Any]) -> str:
    """Return basic account state for a customer.

    Args:
        args: must contain 'customer_id' (str).

    Returns:
        A short status line the agent can read into the next turn.
    """
    customer_id = str(args.get("customer_id", "")).strip()
    if not customer_id:
        return "Error: 'customer_id' argument is required."

    # Deterministic fake — same customer_id always returns the same status.
    bucket = hash(customer_id) % len(_FAKE_STATUSES)
    status = _FAKE_STATUSES[bucket]
    tier = "pro" if "pro" in customer_id.lower() else "free"

    return (
        f"Customer {customer_id}: status={status}, tier={tier}, "
        f"last_payment=2026-04-12, open_tickets=2"
    )


GET_ACCOUNT_STATUS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_account_status",
        "description": (
            "Look up the current account status for a customer: subscription "
            "state, tier, last payment date, and open ticket count. Use this "
            "when billing or account context matters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer's unique ID",
                },
            },
            "required": ["customer_id"],
        },
    },
}


# ---------------------------------------------------------------------------
# Special "tool" 3: submit_decision — how the agent signals it's done
# ---------------------------------------------------------------------------
# This isn't really a tool — it's a trick for getting structured output.
# When the model calls submit_decision, the arguments ARE the TriageDecision.
# The orchestrator validates them through Pydantic and exits the loop.
#
# Why model it as a tool at all? Because tool-call arguments are the most
# reliable way to get structured JSON out of an LLM. The model has been
# heavily trained to fill out tool schemas correctly.


async def submit_decision(args: dict[str, Any]) -> str:
    """Sentinel tool. The orchestrator handles this specially — never executes here.

    If you see this function actually run, something is wrong with the loop.
    """
    return "ERROR: submit_decision is handled by the orchestrator, not invoked as a tool."


SUBMIT_DECISION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_decision",
        "description": (
            "Call this with your final triage verdict when you have enough "
            "information. After calling this, do not call any more tools."
        ),
        "parameters": TriageDecision.model_json_schema(),
    },
}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# A dict mapping tool name to its bundled record. This is what the agent
# loop consults to (a) know what schemas to send the model and (b) dispatch
# the right Python function when the model asks for one.


TRIAGE_TOOLS: dict[str, Tool] = {
    "search_kb": Tool(
        name="search_kb",
        description=SEARCH_KB_SCHEMA["function"]["description"],
        schema=SEARCH_KB_SCHEMA,
        func=search_kb,
    ),
    "get_account_status": Tool(
        name="get_account_status",
        description=GET_ACCOUNT_STATUS_SCHEMA["function"]["description"],
        schema=GET_ACCOUNT_STATUS_SCHEMA,
        func=get_account_status,
    ),
    "submit_decision": Tool(
        name="submit_decision",
        description=SUBMIT_DECISION_SCHEMA["function"]["description"],
        schema=SUBMIT_DECISION_SCHEMA,
        func=submit_decision,
    ),
}


def tool_schemas() -> list[dict[str, Any]]:
    """Return the list of tool schemas to send the LLM."""
    return [tool.schema for tool in TRIAGE_TOOLS.values()]


async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Invoke a tool by name with the given arguments.

    Returns a string the orchestrator will append to the conversation as a
    'tool' message. Errors are converted into readable strings rather than
    raised — the LLM should see the error and adapt, not crash the run.
    """
    tool = TRIAGE_TOOLS.get(name)
    if tool is None:
        logger.warning("unknown_tool_requested", tool=name)
        return f"Error: unknown tool {name!r}. Available: {list(TRIAGE_TOOLS)}"

    try:
        result = await tool.func(arguments)
        logger.debug(
            "tool_executed",
            tool=name,
            arguments=arguments,
            result_length=len(result),
        )
        return result
    except Exception as exc:
        # Catch broadly here because the orchestrator wants to keep the loop
        # alive on tool failures — give the model a chance to recover.
        logger.exception("tool_execution_failed", tool=name)
        return f"Tool {name} raised an error: {type(exc).__name__}: {exc}"