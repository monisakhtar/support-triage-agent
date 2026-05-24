"""The agent loop — think, act, observe, repeat.

This is the orchestrator. Given an LLMProvider and a starting set of messages,
it runs the think-act cycle until either:
  1. The model emits a valid TriageDecision via the submit_decision tool, or
  2. We hit max_iterations (treated as a hard error).

The loop knows nothing about tickets specifically — it just knows how to
shuttle messages between an LLM and a tool registry. That's why this file
talks about 'messages' and 'tools', not 'tickets' and 'triage'. The
domain-specific glue lives in triage.py.

Why split orchestrator from domain? Reuse. The exact same loop could drive a
different agent (refund-bot, ticket-classifier, code-reviewer) with different
tools and a different output schema. Domain rules belong in domain files;
the loop is generic plumbing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, ValidationError

from triage_agent.agents.tools import TRIAGE_TOOLS, execute_tool, tool_schemas
from triage_agent.agents.types import AgentTrace, ToolCallRecord
from triage_agent.llm.base import ChatResponse, LLMProvider, Message, ToolCall

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base class for agent-loop failures."""


class MaxIterationsExceeded(AgentError):
    """The loop ran max_iterations without producing a valid decision."""


class DecisionValidationError(AgentError):
    """The model submitted a decision but its shape was invalid."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """What the loop returns: a validated decision plus an observability trace.

    The decision type is generic — the caller passes its expected Pydantic
    model into run_agent_loop, and we return an instance of that model.
    """

    decision: BaseModel
    trace: AgentTrace


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def run_agent_loop(
    *,
    llm: LLMProvider,
    initial_messages: list[Message],
    decision_model: type[BaseModel],
    decision_tool_name: str = "submit_decision",
    max_iterations: int = 5,
    max_tokens_per_call: int = 1024,
    temperature: float = 0.1,
) -> AgentRunResult:
    """Run the think-act-observe loop until a valid decision is produced.

    Args:
        llm: Any LLMProvider implementation (Ollama, OpenAI, mock).
        initial_messages: Starting conversation. Typically [system, user].
        decision_model: Pydantic model the final decision must conform to.
        decision_tool_name: Name of the tool whose arguments ARE the decision.
        max_iterations: Hard cap on think-act cycles. Defends against runaway loops.
        max_tokens_per_call: Cap per LLM call. Defends against runaway cost.
        temperature: Sampling temperature. Keep low for tool use.

    Returns:
        AgentRunResult with the validated decision and the run's trace.

    Raises:
        MaxIterationsExceeded: Loop ran out of iterations.
        DecisionValidationError: Final attempt produced an invalid decision.
    """
    messages: list[Message] = list(initial_messages)  # copy; don't mutate caller's list
    schemas = tool_schemas()

    trace = AgentTrace(iterations_used=0, total_tokens=0)
    last_validation_error: str | None = None

    logger.info(
        "agent_loop_started",
        max_iterations=max_iterations,
        n_tools=len(schemas),
    )

    for iteration in range(1, max_iterations + 1):
        trace.iterations_used = iteration
        logger.debug("agent_iteration_start", iteration=iteration)

        # ----- 1. Ask the LLM what to do next ------------------------------
        response: ChatResponse = await llm.chat(
            messages=messages,
            tools=schemas,
            temperature=temperature,
            max_tokens=max_tokens_per_call,
        )
        trace.total_tokens += response.usage.total_tokens

        # ----- 2. Did the model decide to submit a final decision? --------
        decision_call = _find_decision_call(response.tool_calls, decision_tool_name)
        if decision_call is not None:
            try:
                decision = decision_model.model_validate(decision_call.arguments)
            except ValidationError as exc:
                # The model meant to commit, but its arguments don't validate.
                # Feed the error back and let it try again next iteration.
                last_validation_error = str(exc)
                logger.warning(
                    "decision_validation_failed",
                    iteration=iteration,
                    error=last_validation_error,
                )
                messages.append(_assistant_with_tool_calls(response))
                messages.append(
                    _tool_result_message(
                        decision_call.id,
                        f"Your decision did not validate: {exc}. "
                        f"Please call {decision_tool_name} again with corrected arguments.",
                    )
                )
                continue

            trace.finished_at = _utcnow()
            logger.info(
                "agent_loop_finished",
                iteration=iteration,
                total_tokens=trace.total_tokens,
                tool_calls=len(trace.tool_calls),
            )
            return AgentRunResult(decision=decision, trace=trace)

        # ----- 3. Other tool calls? Execute them and continue. -----------
        if response.tool_calls:
            messages.append(_assistant_with_tool_calls(response))

            # Run tools in parallel — they're independent, and the model
            # often issues 2-3 in one turn (e.g. search_kb + get_account_status).
            tool_results = await _execute_tools_parallel(
                response.tool_calls, iteration=iteration, trace=trace
            )

            for call, result in zip(response.tool_calls, tool_results):
                messages.append(_tool_result_message(call.id, result))

            continue

        # ----- 4. No tool calls. Nudge the model toward submit_decision. --
        logger.warning(
            "agent_iteration_no_tools",
            iteration=iteration,
            content_preview=response.content[:200],
        )
        messages.append({"role": "assistant", "content": response.content or ""})
        messages.append(
            {
                "role": "user",
                "content": (
                    "You did not call any tool. Either call a tool to gather "
                    f"more information, or call {decision_tool_name} with your "
                    "final verdict. Do not reply with plain text."
                ),
            }
        )

    # Loop exhausted without a decision.
    trace.finished_at = _utcnow()
    trace.error = last_validation_error or "Max iterations reached without a decision."
    logger.error(
        "agent_loop_max_iterations",
        iterations=max_iterations,
        total_tokens=trace.total_tokens,
    )
    raise MaxIterationsExceeded(trace.error)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_decision_call(
    tool_calls: list[ToolCall],
    decision_tool_name: str,
) -> ToolCall | None:
    """Return the decision-tool call if present, else None.

    If the model called multiple tools in one turn AND one of them is the
    decision tool, we treat the decision as authoritative and ignore the
    others. This is a deliberate simplification — production systems might
    want to execute the others first for side effects.
    """
    for call in tool_calls:
        if call.name == decision_tool_name:
            return call
    return None


async def _execute_tools_parallel(
    tool_calls: list[ToolCall],
    *,
    iteration: int,
    trace: AgentTrace,
) -> list[str]:
    """Execute multiple tool calls concurrently and record each in the trace."""

    async def _run_one(call: ToolCall) -> str:
        started = time.perf_counter()
        try:
            result = await execute_tool(call.name, call.arguments)
            error: str | None = None
        except Exception as exc:  # execute_tool catches most; this is belt+suspenders
            result = f"Tool {call.name} crashed: {exc}"
            error = str(exc)
        duration_ms = (time.perf_counter() - started) * 1000

        trace.tool_calls.append(
            ToolCallRecord(
                iteration=iteration,
                tool=call.name,
                arguments=call.arguments,
                result_preview=result[:500],
                duration_ms=round(duration_ms, 2),
                error=error,
            )
        )
        return result

    return await asyncio.gather(*(_run_one(call) for call in tool_calls))


def _assistant_with_tool_calls(response: ChatResponse) -> Message:
    """Build the assistant message representing the model's tool-call turn.

    The conversation needs the assistant message before the tool result
    messages so the API can correlate tool_call_id values.
    """
    return {
        "role": "assistant",
        "content": response.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": _to_arg_string(call.arguments),
                },
            }
            for call in response.tool_calls
        ],
    }


def _tool_result_message(tool_call_id: str, content: str) -> Message:
    """Build a tool-result message, correlated to the assistant's tool call."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def _to_arg_string(arguments: dict) -> str:
    """Serialize tool arguments back to a JSON string for the API.

    Our ToolCall holds arguments as a dict (we parse them at the boundary in
    the provider), but the OpenAI message format expects the string form
    when echoing the assistant message back in conversation history.
    """
    import json

    return json.dumps(arguments)


def _utcnow():
    """Timezone-aware UTC timestamp helper. One import site for swap-friendliness."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)