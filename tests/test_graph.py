"""The graph through run_turn(), with scripted models and the real MCP server in memory."""

import pytest

from agent.policy import LOADOUTS, MAX_TOOL_CALLS
from agent.run import run_turn
from agent.structured import ParseFailed, parse_with_repair
from agent.policy import TurnPlan
from tests.agent_stubs import (ScriptedModel, ScriptedPlanner, call, calls, plan, say,
                               use_stubs)

S1_T1 = "What's the NovaOps policy on using a personal laptop for work?"
S3_T1 = ("Rachel Stein in Customer Success says Webex tells her she isn't licensed. "
         "What's actually going on?")


async def test_s1_and_s3_bind_different_tools_without_a_model(seeded, monkeypatch):
    s1_model = ScriptedModel(call("list_policies"), say("See the equipment policy."))
    use_stubs(monkeypatch, ScriptedPlanner(plan("policy_question")), s1_model)
    s1 = await run_turn(S1_T1, caller_id="E010", user_group="UG_REGULAR", thread_id="t-s1")

    s3_model = ScriptedModel(call("check_software_subscription", software="Webex"),
                             say("Webex has 42 active seats on a 40-seat limit. [db:software_subscriptions#SUB001]"))
    use_stubs(monkeypatch, ScriptedPlanner(plan("subscription_review")), s3_model)
    s3 = await run_turn(S3_T1, caller_id="E007", user_group="UG_IT", thread_id="t-s3")

    assert (s1.scope, s1.intent) == ("maya", "policy_question")
    assert (s3.scope, s3.intent) == ("webex", "subscription_review")
    assert s1.visible_tools == LOADOUTS["policy_question"]
    assert s3.visible_tools == LOADOUTS["subscription_review"]
    assert s1.visible_tools != s3.visible_tools
    # What the model was actually bound to, not just what state says.
    assert s1_model.bound[0] == sorted(LOADOUTS["policy_question"])
    assert s3_model.bound[0] == sorted(LOADOUTS["subscription_review"])
    assert s3.tools_called == ["check_software_subscription"] and s3.status == "completed"


async def test_graph_tool_call_carries_caller_context_in_meta(seeded, monkeypatch):
    import mcp_server.server as server
    seen = []
    real = server.from_request

    def spy(ctx):
        caller = real(ctx)
        seen.append(caller)
        return caller

    monkeypatch.setattr(server, "from_request", spy)
    use_stubs(monkeypatch, ScriptedPlanner(plan("employee_lookup")),
              ScriptedModel(call("get_employee", query="E015"), say("Alex Kim is E015.")))
    out = await run_turn("Who is Alex Kim?", caller_id="E007", user_group="UG_IT",
                         thread_id="t-meta", request_id="REQ-META")
    assert out.tools_called == ["get_employee"]
    assert [(c.caller_id, c.user_group, c.request_id, c.session_id) for c in seen] == \
        [("E007", "UG_IT", "REQ-META", "t-meta")]


async def test_tool_calls_are_capped_per_turn(seeded, monkeypatch):
    many = calls(*[("get_employee", {"query": f"E00{i}"}) for i in range(1, 9)])
    use_stubs(monkeypatch, ScriptedPlanner(plan("employee_lookup")),
              ScriptedModel(many, say("Here are the people.")))
    out = await run_turn("List E001 to E008.", caller_id="E007", user_group="UG_IT",
                         thread_id="t-cap")
    assert len(out.tools_called) == MAX_TOOL_CALLS


async def test_a_tool_that_is_not_visible_is_not_run(seeded, monkeypatch, conn):
    before = conn.execute("SELECT COUNT(*) FROM access_requests").fetchone()[0]
    write = call("create_access_request", employee_id="E010", software="Webex",
                 business_justification="calls")
    use_stubs(monkeypatch, ScriptedPlanner(plan("access_request")),
              ScriptedModel(write, call("get_access_requests", employee_id="E010"),
                            say("AR001 is blocked on seats. [db:access_requests#AR001]")))
    out = await run_turn("Give Rachel Webex.", caller_id="E007", user_group="UG_IT",
                         thread_id="t-hidden")
    assert "create_access_request" not in out.tools_called
    assert conn.execute("SELECT COUNT(*) FROM access_requests").fetchone()[0] == before


async def test_rearm_reopens_reads_once_when_the_plan_needed_a_lookup(seeded, monkeypatch):
    model = ScriptedModel(say("A guess with no lookup."),
                          call("get_employee", query="E010"), say("Rachel Stein is E010."))
    use_stubs(monkeypatch, ScriptedPlanner(plan("policy_question")), model)
    out = await run_turn("Who is E010?", caller_id="E007", user_group="UG_IT",
                         thread_id="t-rearm")
    assert "get_employee" not in LOADOUTS["policy_question"]
    assert "get_employee" in model.bound[1] and "create_access_request" not in model.bound[1]
    assert out.tools_called == ["get_employee"]
    assert "guess" not in out.answer


async def test_turn_without_tools_binds_none(seeded, monkeypatch):
    model = ScriptedModel(say("You're welcome."))
    use_stubs(monkeypatch, ScriptedPlanner(plan("other", needs_tools=False)), model)
    out = await run_turn("Thanks!", caller_id="E010", user_group="UG_REGULAR", thread_id="t-none")
    assert model.bound == [[]] and out.visible_tools == [] and out.status == "completed"


async def test_parse_failure_is_repaired_once():
    planner = ScriptedPlanner(None, plan("policy_question"))
    parsed = await parse_with_repair(planner, [], TurnPlan)
    assert parsed.intent == "policy_question" and planner.calls == 2


async def test_parse_failure_after_repair_raises_with_no_default_plan():
    planner = ScriptedPlanner(None, None)
    with pytest.raises(ParseFailed):
        await parse_with_repair(planner, [], TurnPlan)
    assert planner.calls == 2


async def test_run_turn_reports_a_planner_failure_as_failed(seeded, monkeypatch):
    model = ScriptedModel()
    use_stubs(monkeypatch, ScriptedPlanner(None, None), model)
    out = await run_turn("Hello", caller_id="E010", user_group="UG_REGULAR", thread_id="t-pf")
    assert out.status == "failed" and out.intent is None and out.visible_tools == []
    assert model.bound == []  # nothing ran on a made-up plan
