"""`run_turn()`, the only way into the workflow.

It opens the durable checkpointer and the MCP session, builds the graph (agent/graph.py)
with them, and runs one turn on the thread. A thread paused on an approval is resumed
instead of started: the decision is already in the `approvals` table, written by
application code with `record_approval_decision`, and the graph's `write_gate` reads it.

Tracing comes from Lesson 11 `02-instrument-agent/run.py`: one root span per turn,
`propagate_attributes` with `session_id` = thread id and `user_id` = caller id, and a
flush in `finally`, because Langfuse exports on a background thread and a script that
exits first exports nothing. A resumed turn opens its root span in the trace of the
turn that paused, so the pause and the resume read as one trace.
"""

import uuid
from contextlib import asynccontextmanager

from langchain_core.messages import HumanMessage
from langfuse import get_client, propagate_attributes
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from agent import config
from agent.graph import build_graph
from agent.mcp_tools import list_tool_specs, mcp_session
from agent.model import get_model
from agent.outcome import STATUSES, TurnOutcome, pending_text
from agent.policy import TurnPlan
from agent.structured import ParseFailed

# Each tool round is two graph steps; 6 calls, one rearm and the approval path fit well
# inside this. Hitting it raises GraphRecursionError rather than looping on.
RECURSION_LIMIT = 40


class RequestIdMismatch(ValueError):
    """A paused thread was resumed with a request_id other than the one it paused with."""


def make_models() -> dict:
    """The chat model and the TurnPlan planner. Tests replace this function."""
    return {"model": get_model(),
            "planner": get_model().with_structured_output(TurnPlan, include_raw=True)}


@asynccontextmanager
async def open_checkpointer():
    """The durable checkpointer, injected into build_graph(). A paused thread survives
    the process, and a new process resumes it from the same file."""
    async with AsyncSqliteSaver.from_conn_string(str(config.checkpoint_db())) as saver:
        yield saver


def _resolve_request_id(paused: bool, stored: str | None, passed: str | None) -> str:
    if paused:
        if passed and passed != stored:
            raise RequestIdMismatch(
                f"Thread is paused under request {stored}; got {passed}. A resume keeps "
                "the request_id it paused with.")
        return stored
    return passed or str(uuid.uuid4())


def _outcome(values: dict, interrupted: list, request_id: str, trace_id: str | None) -> TurnOutcome:
    base = {"request_id": request_id, "trace_id": trace_id, "scope": values.get("scope"),
            "intent": values.get("intent"), "visible_tools": values.get("visible_tools", []),
            "tools_called": values.get("turn_tools", [])}
    if interrupted:
        approval = values["approval"]
        return TurnOutcome(answer=pending_text(approval), status="pending",
                           approval_id=approval["approval_id"], **base)
    approval = values.get("approval") or {}
    return TurnOutcome(answer=values["answer"], citations=values.get("citations", []),
                       status=values["status"], approval_id=approval.get("approval_id"), **base)


async def run_turn(message: str, *, caller_id: str, user_group: str, thread_id: str,
                   request_id: str | None = None) -> TurnOutcome:
    langfuse = get_client()
    thread = {"configurable": {"thread_id": thread_id}}
    try:
        async with open_checkpointer() as saver:
            # Read the thread before opening MCP, so a request_id mismatch raises as
            # itself and not wrapped in the MCP session's task-group ExceptionGroup.
            probe = build_graph(None, [], saver, model=None, planner=None)
            snapshot = await probe.aget_state(thread)
            paused = bool(snapshot.next)
            request_id = _resolve_request_id(paused, snapshot.values.get("request_id"), request_id)
            paused_trace = snapshot.values.get("trace_id") if paused else None
            async with mcp_session() as session:
                outcome = await _invoke(
                    build_graph(session, await list_tool_specs(session), saver, **make_models()),
                    thread, message, caller_id=caller_id, user_group=user_group,
                    thread_id=thread_id, request_id=request_id, paused=paused,
                    paused_trace=paused_trace)
    finally:
        langfuse.flush()
    assert outcome.status in STATUSES
    return outcome


async def _invoke(graph, thread: dict, message: str, *, caller_id: str, user_group: str,
                  thread_id: str, request_id: str, paused: bool,
                  paused_trace: str | None) -> TurnOutcome:
    langfuse = get_client()
    configurable = {"thread_id": thread_id, "caller_id": caller_id,
                    "user_group": user_group, "request_id": request_id}
    metadata = {"request_id": request_id, "user_group": user_group,
                "resumed": str(paused).lower()}
    with propagate_attributes(session_id=thread_id, user_id=caller_id,
                              metadata=metadata, trace_name="turn"):
        with langfuse.start_as_current_observation(
                as_type="agent", name="turn", input=message,
                trace_context={"trace_id": paused_trace} if paused_trace else None) as root:
            trace_id = langfuse.get_current_trace_id()
            if paused:
                payload = Command(resume={"message": message},
                                  update={"messages": [HumanMessage(message)]})
            else:
                payload = {"messages": [HumanMessage(message)],
                           "request_id": request_id, "trace_id": trace_id}
            try:
                result = await graph.ainvoke(payload, config={
                    "configurable": configurable, "recursion_limit": RECURSION_LIMIT})
            except ParseFailed as exc:
                outcome = TurnOutcome(answer=f"I could not plan this turn: {exc}",
                                      status="failed", request_id=request_id,
                                      trace_id=trace_id)
            else:
                values = (await graph.aget_state(thread)).values
                outcome = _outcome(values, result.get("__interrupt__") or [],
                                   request_id, trace_id)
            # Set before root.update, so the trace's own output carries the link too.
            # None when Langfuse is not configured (the client is then a no-op).
            outcome.trace_url = langfuse.get_trace_url(trace_id=trace_id)
            # Trace metadata is only settable by propagation; the root span is
            # still current here, so this puts `status` on the trace as well.
            with propagate_attributes(metadata={"request_id": request_id,
                                                "status": outcome.status}):
                root.update(output=outcome.model_dump(), metadata={
                    "status": outcome.status, "request_id": request_id,
                    "scope": outcome.scope, "intent": outcome.intent,
                    "visible_tools": outcome.visible_tools,
                    "tool_sequence": outcome.tools_called,
                    "approval_id": outcome.approval_id})
    return outcome
