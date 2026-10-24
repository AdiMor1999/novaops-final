"""Caller context: who is calling, from which workflow request.

The agent's tool wrapper (`agent/mcp_tools.py`) sends `caller_id`, `user_group` and
`request_id` in the MCP request `_meta` field, under the key `novaops`. `_meta` is part of
the MCP protocol, not of any tool's input schema, so `list_tools()` never shows these
three and the model cannot set them. Tools trust only this context for identity.

The same `_meta` entry may carry the thread id, the Langfuse trace id and the parent
observation id, so the server's spans (retrieval, rerank) nest under the agent's
`tool:<name>` span, with the same session and user, instead of starting a trace of
their own.
"""

from dataclasses import dataclass

from mcp.server.fastmcp import Context

from mcp_server.db import ToolInputError

META_KEY = "novaops"


@dataclass(frozen=True)
class CallerContext:
    caller_id: str
    user_group: str
    request_id: str
    session_id: str | None = None  # the thread id, for Langfuse only; never for authorization
    trace_id: str | None = None
    parent_observation_id: str | None = None


class MissingCallerContext(ToolInputError):
    """A call without caller context is refused: identity is never optional."""


def from_meta(meta: dict | None) -> CallerContext:
    data = (meta or {}).get(META_KEY) or {}
    missing = [k for k in ("caller_id", "user_group", "request_id") if not data.get(k)]
    if missing:
        raise MissingCallerContext(
            f"Refused: caller context is missing {', '.join(missing)}. This server only "
            "accepts calls made through the NovaOps agent."
        )
    return CallerContext(
        caller_id=str(data["caller_id"]),
        user_group=str(data["user_group"]),
        request_id=str(data["request_id"]),
        session_id=data.get("session_id"),
        trace_id=data.get("trace_id"),
        parent_observation_id=data.get("parent_observation_id"),
    )


def from_request(ctx: Context) -> CallerContext:
    meta = ctx.request_context.meta
    extra = (meta.model_extra or {}) if meta is not None else {}
    return from_meta(extra)
