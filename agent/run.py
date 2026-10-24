"""`run_turn()`, the only way into the workflow.

Milestone 1 body: retrieve over MCP, answer with `call_model()` from the retrieved
chunks, cite `[source#section]` labels. Task #3 replaces the body with the graph and
keeps the signature, the tracing and the return shape. There is no checkpointer yet.

Tracing comes from Lesson 11 `02-instrument-agent/run.py`: one root span per turn,
`propagate_attributes` with `session_id` = thread id and `user_id` = caller id, and a
flush before returning, because Langfuse exports on a background thread and a script
that exits first exports nothing.
"""

import re
import uuid
from typing import Any

from langchain_core.tools import ToolException
from langfuse import get_client, propagate_attributes

from agent.mcp_tools import call_tool, list_tool_specs, mcp_session
from agent.model import call_model, response_text

SEARCH_TOOLS = ("search_knowledge_base", "search_hr_documents")
STATUSES = ("completed", "blocked", "pending", "needs_human", "failed")
_LABEL = re.compile(r"\[[^\[\]#]+\.md#[^\[\]]+\]")

ANSWER_SYSTEM = (
    "You are the NovaOps assistant. Answer using only the numbered context chunks. "
    "After every sentence that states a fact, cite the chunk it came from with its label "
    "exactly as given, for example [equipment_policy.md#Personal laptops]. Never cite a "
    "label that is not in the context. If chunks disagree, newer or more specific guidance "
    "(a later memo) overrides older or more general guidance (a policy); say which one you "
    "followed. If the context does not answer the question, say so plainly and do not guess. "
    "Answer in at most six sentences, with no preamble."
)


def _resolve_request_id(request_id: str | None) -> str:
    # Task #3 adds the paused-thread case: a resume reads request_id from the checkpoint.
    return request_id or str(uuid.uuid4())


async def _retrieve(session, message: str, configurable: dict) -> tuple[list[dict], list[str], list[str]]:
    """Call each search tool the server lists. Returns the chunks (deduplicated by
    chunk_id, best first), the tools called, and the errors of any call that failed."""
    available = {spec.name for spec in await list_tool_specs(session)}
    chunks: dict[str, dict] = {}
    errors, called = [], []
    for name in SEARCH_TOOLS:
        if name not in available:
            errors.append(f"{name} is not served by the MCP server")
            continue
        called.append(name)
        try:
            result = await call_tool(session, name, {"query": message}, configurable)
        except ToolException as exc:
            errors.append(f"{name}: {exc}")
            continue
        for chunk in result.get("chunks", []):
            chunks.setdefault(chunk["chunk_id"], chunk)
    ranked = sorted(chunks.values(),
                    key=lambda c: c["rerank_score"] if c["rerank_score"] is not None else -1,
                    reverse=True)
    return ranked, called, errors


def _context_block(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[chunk {i}] label: {c['citation']} | {c['doc_type']} | last_updated: "
        f"{c.get('last_updated') or 'unknown'}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def _answer(message: str, chunks: list[dict]) -> str:
    prompt = f"Context:\n{_context_block(chunks)}\n\nQuestion: {message}"
    with get_client().start_as_current_observation(as_type="span", name="answer",
                                                   input={"chunks": len(chunks)}) as span:
        text = response_text(call_model(
            [{"role": "user", "content": [{"text": prompt}]}],
            system=ANSWER_SYSTEM, max_tokens=600, name="answer"))
        span.update(output=text)
    return text


def cited_labels(answer: str, chunks: list[dict]) -> tuple[list[str], list[str]]:
    """(labels the answer cites that a retrieved chunk carries, labels it invented)."""
    known = {c["citation"] for c in chunks}
    used = list(dict.fromkeys(_LABEL.findall(answer)))
    return [u for u in used if u in known], [u for u in used if u not in known]


def _status(chunks: list[dict], errors: list[str], valid: list[str], invented: list[str]) -> str:
    """Set from what the tools returned and what the answer cites, never from model text."""
    if errors and not chunks:
        return "failed"
    if not chunks or not valid or invented:
        return "needs_human"
    return "completed"


async def run_turn(message: str, *, caller_id: str, user_group: str, thread_id: str,
                   request_id: str | None = None) -> dict[str, Any]:
    request_id = _resolve_request_id(request_id)
    configurable = {"thread_id": thread_id, "caller_id": caller_id,
                    "user_group": user_group, "request_id": request_id}
    langfuse = get_client()
    result: dict[str, Any] = {"answer": "", "citations": [], "status": "failed",
                              "request_id": request_id}
    try:
        with propagate_attributes(session_id=thread_id, user_id=caller_id,
                                  metadata={"request_id": request_id, "user_group": user_group},
                                  trace_name="turn"):
            with langfuse.start_as_current_observation(as_type="agent", name="turn",
                                                       input=message) as root:
                async with mcp_session() as session:
                    chunks, called, errors = await _retrieve(session, message, configurable)
                if chunks:
                    answer = _answer(message, chunks)
                else:
                    answer = ("I could not find this in the NovaOps documents. "
                              + " ".join(errors)).strip()
                valid, invented = cited_labels(answer, chunks)
                status = _status(chunks, errors, valid, invented)
                result.update(answer=answer, citations=valid, status=status,
                              trace_id=langfuse.get_current_trace_id())
                # Trace metadata is only settable by propagation; the root span is still
                # current here, so this puts `status` on the trace as well.
                with propagate_attributes(metadata={"request_id": request_id, "status": status}):
                    root.update(output=result, metadata={
                        "status": status, "request_id": request_id,
                        "tool_errors": errors, "uncited_labels": invented,
                        "tool_sequence": called})
        result["trace_url"] = langfuse.get_trace_url(trace_id=result.get("trace_id"))
    finally:
        langfuse.flush()
    assert result["status"] in STATUSES
    return result
