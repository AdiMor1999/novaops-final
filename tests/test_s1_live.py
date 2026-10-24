"""S1-quick-policy end to end: real Bedrock, real OpenSearch, the real MCP server over
the in-memory transport, four turns on one thread. Skipped without AWS."""

import json
import os
import re
import uuid

import pytest

from agent import config
from agent.run import run_turn
from evals.tool_map import expected_tools_met, forbidden_tools_hit
from tests.agent_stubs import memory_session

pytestmark = pytest.mark.integration


def s1_session() -> dict:
    data = json.loads((config.course_dir() / "GOLDEN-DATASETS.json").read_text(encoding="utf-8"))
    return next(s for s in data["sessions"] if s["id"] == "S1-quick-policy")


def facts_present(required: list[str], answer: str) -> list[str]:
    """Required facts the answer misses. '|' means alternatives, case-insensitive."""
    low = answer.lower()
    return [f for f in required if not any(alt.lower() in low for alt in f.split("|"))]


@pytest.fixture
def live(seeded, monkeypatch):
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("OPENSEARCH_COLLECTION")):
        pytest.skip("no AWS or OpenSearch credentials")
    from mcp_server import opensearch
    if not opensearch.index_exists():
        pytest.skip("novaops-final-kb is not created; run scripts.create_index and scripts.ingest")
    import agent.run
    monkeypatch.setattr(agent.run, "mcp_session", memory_session)


async def test_s1_quick_policy(live):
    session = s1_session()
    thread = f"s1-{uuid.uuid4().hex[:8]}"
    problems = []
    for turn in session["turns"]:
        out = await run_turn(turn["user"], caller_id="E010", user_group="UG_REGULAR",
                             thread_id=thread, request_id=f"S1-turn{turn['n']}")
        print(f"\nturn {turn['n']} intent={out.intent} tools={out.tools_called} "
              f"status={out.status}\n  {out.answer}\n  trace: {out.trace_url}")
        if turn["expected_intent"] and out.intent not in turn["expected_intent"].split("|"):
            problems.append(f"turn {turn['n']}: intent {out.intent}")
        if turn["should_use_tools"] is False and out.tools_called:
            problems.append(f"turn {turn['n']}: called {out.tools_called}, expected none")
        if not expected_tools_met(turn["expected_tools"], out.tools_called):
            problems.append(f"turn {turn['n']}: tools {out.tools_called} miss {turn['expected_tools']}")
        if hit := forbidden_tools_hit(turn["forbidden_tools"], out.tools_called):
            problems.append(f"turn {turn['n']}: forbidden {hit}")
        if missing := facts_present(turn["required_facts"], out.answer):
            problems.append(f"turn {turn['n']}: missing facts {missing}")
        if turn["n"] == 1 and "[equipment_policy.md#Personal laptops]" not in out.citations:
            problems.append(f"turn 1: citations {out.citations}")
    assert problems == []
