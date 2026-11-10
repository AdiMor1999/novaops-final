"""The agent's side of the MCP boundary: discover tools, call them with caller context.

Tools are discovered with `list_tools()` and never imported. Every call goes through
`call_tool()`, which:

- reads `caller_id`, `user_group` and `request_id` from the run's config (graph config
  in task #3) and sends them in the MCP request `_meta` under `novaops`. They are not
  tool arguments, so the model never sees, sets or overrides them;
- opens a `tool:<name>` Langfuse span and passes its trace and observation ids in the
  same `_meta`, so server-side spans (retrieval, rerank) nest under it;
- turns an `is_error` result into a `ToolException`, so a refused write or a bad
  argument reaches the model as a message, not as a crash;
- moves the server's heuristic firings (`_heuristics`) out of the result and into the
  turn's record (agent/heuristics.py), so the model never sees them.

`langchain-mcp-adapters` opens the session. Its own tool wrapper is not used, because it
does not forward `_meta` on `call_tool`.
"""

import json
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, ToolException
from langchain_mcp_adapters.sessions import create_session
from langfuse import get_client
from mcp import ClientSession
from mcp.types import CallToolResult

from agent import config, heuristics

CONTEXT_KEYS = ("caller_id", "user_group", "request_id")
META_KEY = "novaops"


class MissingCallerContext(RuntimeError):
    """A tool call was attempted without caller_id, user_group or request_id in config."""


@asynccontextmanager
async def mcp_session():
    connection = {"transport": "streamable_http", "url": config.mcp_server_url()}
    async with create_session(connection) as session:
        await session.initialize()
        yield session


def caller_context(configurable: dict) -> dict:
    missing = [k for k in CONTEXT_KEYS if not configurable.get(k)]
    if missing:
        raise MissingCallerContext(f"config is missing {', '.join(missing)}")
    return {k: configurable[k] for k in CONTEXT_KEYS}


def decode_result(result: CallToolResult) -> Any:
    text = "".join(getattr(block, "text", "") for block in result.content)
    if result.structuredContent is not None and not result.isError:
        return result.structuredContent
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def call_tool(session: ClientSession, name: str, args: dict, configurable: dict) -> Any:
    context = caller_context(configurable)
    langfuse = get_client()
    with langfuse.start_as_current_observation(as_type="tool", name=f"tool:{name}",
                                               input=args) as span:
        meta = {META_KEY: {
            **context,
            "session_id": configurable.get("thread_id"),
            "trace_id": langfuse.get_current_trace_id(),
            "parent_observation_id": langfuse.get_current_observation_id(),
        }}
        result = await session.call_tool(name, args, meta=meta)
        payload = decode_result(result)
        fired = payload.pop(heuristics.SERVER_KEY, []) if isinstance(payload, dict) else []
        for heuristic in fired:
            heuristics.fired(heuristic)
        span.update(output=payload, level="ERROR" if result.isError else "DEFAULT",
                    metadata={"is_error": result.isError, "request_id": context["request_id"],
                              **({"heuristics_fired": fired} if fired else {})})
    if result.isError:
        raise ToolException(payload if isinstance(payload, str) else json.dumps(payload))
    return payload


async def list_tool_specs(session: ClientSession) -> list:
    return (await session.list_tools()).tools


def as_langchain_tools(session: ClientSession, specs: list) -> list[StructuredTool]:
    """LangChain tools for the graph (task #3). The schema is the server's inputSchema
    unchanged; caller context comes from the injected RunnableConfig, not from it."""
    def make(spec) -> StructuredTool:
        async def run(config: RunnableConfig, **kwargs):
            return await call_tool(session, spec.name, kwargs, config.get("configurable", {}))

        return StructuredTool.from_function(
            coroutine=run, name=spec.name, description=spec.description or "",
            args_schema=spec.inputSchema, handle_tool_error=True,
        )
    return [make(spec) for spec in specs]
