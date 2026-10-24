"""Replay a golden session turn by turn on one thread, through run_turn().

Starting code: Lesson 10 `evals/runner.py` (`run_session`: one stable thread id, the
turns in order, one record per turn). Changes: the graph is not passed in, because
run_turn() is the only way in; the database is reseeded once before the session (a
session is one case); each turn's request_id is `<session>-t<n>` so traces match turns;
and the record keeps what the checks need: tools, sources, citations, status, the
checklist, the handoff and the trace URL.

    python -m scripts.reseed && python -m mcp_server &      # the server, on MCP_SERVER_URL
    python -m evals.run_session S2-onboarding-maya S9-maya-own-account --out report.json

Exit code 1 if any turn has a problem.
"""

import argparse
import asyncio
import json
import re
import sys
import uuid

from agent import config
from agent.run import run_turn
from evals.session_checks import WRITE_GATE_SESSIONS, binding_checks, project_checks
from evals.tool_matrix import golden_sessions
from mcp_server import db
from scripts.reseed import reseed


def session_caller(session: dict) -> tuple[str, str]:
    """(caller id, user group). S2 and S9 state both. The other sessions name the
    caller only in `persona` ("Sara Ben-David (E004, ..."), so the id comes from there
    and the group from that employee's row, never from a guess."""
    if session.get("caller_id"):
        return session["caller_id"], session["caller_group"]
    found = re.search(r"\((E\d{3})\b", session["persona"])
    if found is None:
        raise ValueError(f"{session['id']} names no caller")
    with db.session() as conn:
        return found[1], db.require_employee(conn, found[1])["user_group_id"]


def turn_record(session_id: str, turn: dict, outcome) -> dict:
    problems = binding_checks(turn, outcome) + project_checks(session_id, turn["n"], outcome)
    return {
        "n": turn["n"],
        "kind": turn.get("kind"),
        "intent": outcome.intent,
        "expected_intent": turn.get("expected_intent"),
        "scope": outcome.scope,
        "status": outcome.status,
        "tools_called": outcome.tools_called,
        "sources": outcome.sources,
        "citations": outcome.citations,
        "checklist": outcome.checklist.model_dump() if outcome.checklist else None,
        "handoff": outcome.handoff,
        "approval_id": outcome.approval_id,
        "approvals": outcome.approvals,
        "access_decision": (outcome.access_decision.model_dump()
                            if outcome.access_decision else None),
        "answer": outcome.answer,
        "trace_url": outcome.trace_url,
        "problems": problems,
    }


async def run_session(session: dict, *, thread_id: str | None = None) -> list[dict]:
    thread_id = thread_id or f"{session['id'].split('-')[0]}-{uuid.uuid4().hex[:8]}"
    caller_id, user_group = session_caller(session)
    with db.session() as conn:
        before = row_counts(conn)
    records = []
    for turn in session["turns"]:
        outcome = await run_turn(turn["user"], caller_id=caller_id,
                                 user_group=user_group, thread_id=thread_id,
                                 request_id=f"{session['id'].split('-')[0]}-t{turn['n']}")
        record = turn_record(session["id"], turn, outcome)
        record["thread_id"] = thread_id
        records.append(record)
        print_turn(record)
    with db.session() as conn:
        after = row_counts(conn)
    if session["id"].startswith(WRITE_GATE_SESSIONS) and after != before:
        # The write-gate sessions reuse T001, AR001, AP001 and AP002: any new row is a
        # duplicate (spec §9 "replaying the case creates no second row").
        records[-1]["problems"].append(f"rows changed over the session: {before} -> {after}")
        print(f"       - rows changed over the session: {before} -> {after}", flush=True)
    records[-1]["row_counts"] = {"before": before, "after": after}
    return records


def row_counts(conn) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("tickets", "access_requests", "approvals")}


def print_turn(record: dict) -> None:
    mark = "ok  " if not record["problems"] else "FAIL"
    print(f"{mark} t{record['n']:<2} {record['intent'] or '-':<20} {record['status']:<11} "
          f"tools={record['tools_called']}", flush=True)
    for problem in record["problems"]:
        print(f"       - {problem}", flush=True)


async def main(ids: list[str], out: str | None) -> int:
    sessions = {s["id"]: s for s in golden_sessions(tuple(ids))}
    report = {}
    for session_id in ids:
        reseed(config.db_path())
        print(f"\n== {session_id}", flush=True)
        report[session_id] = await run_session(sessions[session_id])
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
    failed = sum(bool(r["problems"]) for records in report.values() for r in records)
    print(f"\n{failed} turn(s) with problems")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.sessions, args.out)))
