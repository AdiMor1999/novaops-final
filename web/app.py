"""The HTTP layer: one static page and a small JSON API around `run_turn()`.

Caller identity works the way spec §6.6 asks of a UI: the page sends the caller id it
shows in its dropdown, this module reads that caller's user group from the employee
row, and both go to `run_turn()` with the page's thread id. Nothing here reads identity
from the message text, picks a scope or a tool, or decides an approval from chat.
"""

import asyncio
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent.run import run_turn
from web import approvals, data, settings

PAGE = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="NovaOps chat", docs_url=None, redoc_url=None)

# A thread keeps the caller it started with. The page enforces this too, but a second
# tab could still send another caller's id on the same thread, so the server holds the
# binding for as long as it runs. Busy threads refuse a second concurrent turn, because
# two turns on one checkpoint would race.
_thread_callers: dict[str, str] = {}
_busy: set[str] = set()


class TurnRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    caller_id: str = Field(min_length=1, max_length=16)
    thread_id: str = Field(min_length=1, max_length=64)


class DecisionRequest(BaseModel):
    caller_id: str = Field(min_length=1, max_length=16)
    decision: Literal["approved", "rejected"]


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health() -> dict:
    host, port = settings.mcp_address()
    return {"mcp": settings.mcp_reachable(), "mcp_address": f"{host}:{port}"}


@app.get("/api/callers")
def callers() -> list[dict]:
    return data.callers()


@app.get("/api/approvals")
def approvals_for(caller_id: str) -> list[dict]:
    return data.open_approvals(caller_id)


@app.post("/api/approvals/{approval_id}/decision")
def decide(approval_id: str, body: DecisionRequest) -> dict:
    try:
        return approvals.decide(approval_id, body.decision, body.caller_id)
    except ValueError as exc:  # ToolInputError / WriteRefused: not the approver, already decided
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _group_of(caller_id: str) -> str:
    try:
        return data.user_group(caller_id)
    except data.UnknownCaller as exc:
        raise HTTPException(status_code=404, detail=f"{caller_id} is not a NovaOps employee.") from exc


def _bind_thread(thread_id: str, caller_id: str) -> None:
    owner = _thread_callers.setdefault(thread_id, caller_id)
    if owner != caller_id:
        raise HTTPException(status_code=409, detail=(
            f"Thread {thread_id} belongs to {owner}. Start a new thread to talk as {caller_id}."))


def _mcp_down() -> HTTPException:
    host, port = settings.mcp_address()
    return HTTPException(status_code=503, detail=(
        f"The MCP server is not reachable at {host}:{port}. Start it with "
        "`python -m mcp_server`, or use `python -m web.demo`, which starts both."))


@app.post("/api/turn")
async def turn(body: TurnRequest) -> dict:
    user_group = _group_of(body.caller_id)
    _bind_thread(body.thread_id, body.caller_id)
    if not settings.mcp_reachable():
        raise _mcp_down()
    if body.thread_id in _busy:
        raise HTTPException(status_code=409, detail="A turn is already running on this thread.")
    _busy.add(body.thread_id)
    try:
        outcome = await run_turn(body.message, caller_id=body.caller_id,
                                 user_group=user_group, thread_id=body.thread_id)
    except BaseExceptionGroup as group:
        # The MCP client runs in a task group, so a server that dies mid-turn surfaces
        # as an ExceptionGroup around httpx's ConnectError. Anything else is re-raised.
        if group.subgroup(httpx.ConnectError) is None:
            raise
        raise _mcp_down() from group
    finally:
        _busy.discard(body.thread_id)
    return outcome.model_dump(mode="json")


async def serve() -> None:
    import uvicorn
    config = uvicorn.Config(app, host=settings.host(), port=settings.port(), log_level="info")
    await uvicorn.Server(config).serve()


def main() -> None:
    print(f"NovaOps chat on http://{settings.host()}:{settings.port()}")
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass  # uvicorn has already shut down; it re-raises the signal it caught

