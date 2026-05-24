"""Streamlit UI for the triage agent.

This is a THIN HTTP CLIENT of the FastAPI backend — exactly like any
other frontend would be. It does not import any application code from
`triage_agent`. It just makes HTTP calls to the running API server.

That's the architectural contract: this file could be replaced with a
React app, a CLI, a Slack bot, anything — and the backend wouldn't know.

Run:
    # Terminal 1
    uv run serve

    # Terminal 2
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import streamlit as st


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("TRIAGE_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 120.0  # Agent calls can be slow on local Ollama


# ---------------------------------------------------------------------------
# Example tickets — for the demo dropdown
# ---------------------------------------------------------------------------

EXAMPLE_TICKETS: dict[str, dict[str, Any]] = {
    "Double charge (billing, high urgency)": {
        "ticket_id": "T-1001",
        "customer_id": "C-pro-42",
        "customer_tier": "pro",
        "subject": "Double charge on my Pro subscription",
        "body": (
            "I was billed twice on April 12 for $49 each. Please refund "
            "the duplicate charge as soon as possible."
        ),
    },
    "Rate limit issue (technical)": {
        "ticket_id": "T-1002",
        "customer_id": "C-pro-77",
        "customer_tier": "pro",
        "subject": "API returning 429 errors despite being under limit",
        "body": (
            "Hello, my requests to /v1/messages are being rate-limited "
            "but my dashboard shows I'm well under the 1000 req/min cap. "
            "Is there a known issue?"
        ),
    },
    "Profile picture (simple, low urgency)": {
        "ticket_id": "T-1003",
        "customer_id": "C-free-99",
        "customer_tier": "free",
        "subject": "How do I change my profile picture?",
        "body": (
            "Hi, I can't find the option to upload a new profile picture "
            "in my settings. Where is it?"
        ),
    },
    "Outage (critical, enterprise)": {
        "ticket_id": "T-1004",
        "customer_id": "C-enterprise-1",
        "customer_tier": "enterprise",
        "subject": "Production is down — all API calls failing",
        "body": (
            "Our entire production environment is failing to authenticate "
            "with your API. This started 8 minutes ago. We have customers "
            "actively losing transactions. Please escalate immediately."
        ),
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers — thin wrappers around the API
# ---------------------------------------------------------------------------


def fetch_health() -> dict[str, Any] | None:
    """Probe the API. Returns the health payload or None if the API is down."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def submit_triage(ticket: dict[str, Any]) -> dict[str, Any]:
    """Submit a ticket to /triage. Raises httpx errors on API failure."""
    response = httpx.post(
        f"{API_BASE_URL}/triage",
        json=ticket,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


URGENCY_COLORS = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🟠",
    "critical": "🔴",
}

ACTION_LABELS = {
    "auto_resolve": "Auto-resolve",
    "draft_reply": "Draft reply",
    "escalate": "Escalate to human",
}


def render_decision(decision: dict[str, Any]) -> None:
    """Render the agent's verdict as a nice card."""
    urgency = decision["urgency"]
    icon = URGENCY_COLORS.get(urgency, "⚪")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Urgency", f"{icon} {urgency.upper()}")
    col2.metric("Category", decision["category"].replace("_", " ").title())
    col3.metric("Action", ACTION_LABELS.get(decision["suggested_action"], decision["suggested_action"]))
    col4.metric("Confidence", f"{decision['confidence']:.0%}")

    st.markdown("**Reasoning**")
    st.info(decision["reasoning"])


def render_trace(trace: dict[str, Any]) -> None:
    """Render the agent's run trace for observability."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Iterations", trace["iterations_used"])
    col2.metric("Total tokens", trace["total_tokens"])
    col3.metric("Tool calls", len(trace.get("tool_calls", [])))

    tool_calls = trace.get("tool_calls", [])
    if tool_calls:
        st.markdown("**Tools used**")
        for i, call in enumerate(tool_calls, 1):
            with st.expander(
                f"#{i} `{call['tool']}` (iteration {call['iteration']}, "
                f"{call['duration_ms']:.1f}ms)"
            ):
                st.markdown("*Arguments:*")
                st.json(call["arguments"])
                st.markdown("*Result preview:*")
                st.code(call["result_preview"], language="text")
                if call.get("error"):
                    st.error(f"Tool error: {call['error']}")
    else:
        st.warning(
            "Agent did not call any tools. This usually means the model "
            "skipped context gathering — common with small local models on "
            "tool-heavy tasks. Try switching to OpenAI in `.env`."
        )

    if trace.get("error"):
        st.error(f"Agent error: {trace['error']}")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Support Triage Agent",
    page_icon="🎫",
    layout="wide",
)

st.title("🎫 Support Triage Agent")
st.caption(
    "Agentic AI demo — submit a ticket, watch the agent gather context "
    "and produce a structured decision."
)


# --- Sidebar: backend status -----------------------------------------------

with st.sidebar:
    st.header("Backend status")

    health = fetch_health()
    if health is None:
        st.error(
            f"Cannot reach API at {API_BASE_URL}.\n\n"
            "Start the backend with:\n```\nuv run serve\n```"
        )
        st.stop()

    st.success("API online")
    st.markdown(f"**Environment:** `{health['env']}`")
    st.markdown(f"**LLM provider:** `{health['llm_provider']}`")
    st.markdown(f"**LLM model:** `{health['llm_model']}`")
    st.divider()
    st.caption(
        "To switch providers, edit `.env` and restart the API "
        "(`Ctrl+C` then `uv run serve`)."
    )


# --- Main: ticket form ------------------------------------------------------

st.subheader("1. Submit a ticket")

# Examples dropdown — picking one fills in the form fields below
example_label = st.selectbox(
    "Load an example",
    options=["(none)"] + list(EXAMPLE_TICKETS.keys()),
    help="Pick an example to pre-fill the form. You can still edit before submitting.",
)

example = EXAMPLE_TICKETS.get(example_label, {})

with st.form("ticket_form"):
    col1, col2, col3 = st.columns([1, 1, 1])
    ticket_id = col1.text_input(
        "Ticket ID", value=example.get("ticket_id", "T-9999")
    )
    customer_id = col2.text_input(
        "Customer ID", value=example.get("customer_id", "C-001")
    )
    customer_tier = col3.selectbox(
        "Customer tier",
        options=["free", "pro", "enterprise"],
        index=["free", "pro", "enterprise"].index(example.get("customer_tier", "free")),
    )

    subject = st.text_input("Subject", value=example.get("subject", ""))
    body = st.text_area(
        "Body",
        value=example.get("body", ""),
        height=150,
    )

    submitted = st.form_submit_button("Triage ticket", type="primary")


# --- Results ----------------------------------------------------------------

if submitted:
    payload = {
        "ticket_id": ticket_id.strip(),
        "customer_id": customer_id.strip(),
        "customer_tier": customer_tier,
        "subject": subject.strip(),
        "body": body.strip(),
    }

    # Client-side sanity checks (server validates too, but fail fast here)
    if len(payload["subject"]) < 3:
        st.error("Subject must be at least 3 characters.")
        st.stop()
    if len(payload["body"]) < 1:
        st.error("Body cannot be empty.")
        st.stop()

    st.subheader("2. Agent thinking…")
    started = time.perf_counter()
    with st.spinner(
        f"Triaging via {health['llm_provider']} ({health['llm_model']})…"
    ):
        try:
            result = submit_triage(payload)
        except httpx.HTTPStatusError as exc:
            st.error(
                f"API returned {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            )
            st.stop()
        except httpx.ConnectError:
            st.error(
                f"Could not connect to {API_BASE_URL}. Is the server running?"
            )
            st.stop()
        except httpx.ReadTimeout:
            st.error(
                f"The agent took longer than {REQUEST_TIMEOUT_SECONDS}s "
                "to respond. Try again, or switch to a faster provider."
            )
            st.stop()
    elapsed = time.perf_counter() - started

    st.subheader("3. Decision")
    render_decision(result["decision"])

    st.subheader("4. Agent trace")
    st.caption(f"Total wall-clock time: {elapsed:.2f}s")
    render_trace(result["trace"])

    with st.expander("Raw API response"):
        st.json(result)