"""TurnOutcome.trace_url comes from the Langfuse client, on the outcome and on the trace."""

import os

import pytest
from langfuse._client.span import LangfuseAgent

from agent.run import run_turn
from tests.agent_stubs import ScriptedModel, ScriptedPlanner, call, plan, say, use_stubs

LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")


@pytest.fixture
def langfuse_configured():
    if not all(os.environ.get(k) for k in LANGFUSE_KEYS):
        pytest.skip("Langfuse keys are not set")
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "").lower() == "false":
        pytest.skip("tracing is switched off (LANGFUSE_TRACING_ENABLED=false)")


async def test_trace_url_is_set_on_the_outcome_and_the_root_span(seeded, monkeypatch,
                                                                 langfuse_configured):
    root_outputs = []
    real_update = LangfuseAgent.update

    def spy(self, **kwargs):
        if isinstance(kwargs.get("output"), dict) and "trace_url" in kwargs["output"]:
            root_outputs.append(kwargs["output"])
        return real_update(self, **kwargs)

    monkeypatch.setattr(LangfuseAgent, "update", spy)
    use_stubs(monkeypatch, ScriptedPlanner(plan("policy_question")),
              ScriptedModel(call("list_policies"), say("See the equipment policy.")))
    out = await run_turn("Which policies exist?", caller_id="E010", user_group="UG_REGULAR",
                         thread_id="t-trace-url")

    base = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
    assert out.trace_id and out.trace_url
    assert out.trace_url.startswith(f"{base}/project/")
    assert out.trace_url.endswith(f"/traces/{out.trace_id}")
    assert [o["trace_url"] for o in root_outputs] == [out.trace_url]
