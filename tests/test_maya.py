"""Maya (task #4) through run_turn(), with scripted models and the real MCP server in
memory: the handoff, the subject check, citations, the checklist, distillation and the
tool matrix."""

import pytest
from pydantic import ValidationError

import agent.policy as policy
from agent.memory import ConversationMemory, memory_is_safe
from agent.outcome import cited_labels, uncited_sentences
from agent.ports import FakeWebexPort, McpWebexPort
from agent.run import run_turn
from agent.schemas import EvidenceCitation, OnboardingChecklist, WebexHandoff
from evals import tool_matrix
from tests.agent_stubs import ScriptedModel, ScriptedPlanner, call, plan, say, use_stubs

HR = {"caller_id": "E004", "user_group": "UG_HR"}
LOOKUP = plan("employee_lookup")
SEATS = plan("subscription_review")
FILE_IT = plan("access_request", action_confirmed=True, subject_employee_id="E001",
               system="Webex")
TURN_10 = ("Understood, I'll route it to Tom. Go ahead and file the Webex access request "
           "for her now.")


def setup_replies():
    """Turns 1 and 2 of a shortened S2: Maya's record, then the Webex subscription."""
    return [call("get_employee", query="Maya Cohen"),
            say("Maya Cohen is E001. [db:employees#E001]"),
            call("check_software_subscription", software="Webex"),
            say("Webex has 42 active seats on a 40-seat limit. "
                "[db:software_subscriptions#SUB001]")]


async def two_setup_turns(thread: str):
    await run_turn("Pull up Maya Cohen's record.", thread_id=thread, **HR)
    await run_turn("Is a Webex seat available?", thread_id=thread, **HR)


def approvals(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0]


# The handoff ---------------------------------------------------------------------

async def test_turn_10_hands_off_once_with_no_model_tool_call(seeded, monkeypatch, conn):
    port = FakeWebexPort()
    model = ScriptedModel(*setup_replies())  # nothing scripted for turn 10
    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP, SEATS, FILE_IT, FILE_IT), model, port=port)
    await two_setup_turns("t-handoff")
    calls_before = len(model.seen)

    out = await run_turn(TURN_10, thread_id="t-handoff", request_id="S2-t10", **HR)
    assert out.status == "pending" and out.scope == "maya"
    assert out.tools_called == [] and out.visible_tools == []
    assert len(model.seen) == calls_before  # no model call at all on the handoff turn
    assert len(port.calls) == 1
    handoff = port.calls[0]
    assert (handoff.workflow, handoff.subject_employee_id, handoff.system_id, handoff.software,
            handoff.caller_id, handoff.source_thread_id, handoff.request_id) == \
        ("webex_access", "E001", "S001", "Webex", "E004", "t-handoff", "S2-t10")
    assert handoff.idempotency_key == "handoff:t-handoff:E001:S001"
    assert "pending" in out.answer and "granted" not in out.answer.replace("no access is granted", "")
    assert out.handoff["result"]["status"] == "pending"

    replay = await run_turn(TURN_10, thread_id="t-handoff", request_id="S2-t10-replay", **HR)
    assert replay.status == "pending" and len(port.calls) == 1
    assert replay.handoff == out.handoff


async def test_real_port_persists_one_pending_approval_and_replay_adds_none(seeded, monkeypatch, conn):
    made = []

    def spy_port(session, configurable):
        port = McpWebexPort(session, configurable)
        made.append(port)
        real = port.request_access

        async def counted(handoff):
            port.calls = getattr(port, "calls", 0) + 1
            return await real(handoff)
        port.request_access = counted
        return port

    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP, SEATS, FILE_IT, FILE_IT),
              ScriptedModel(*setup_replies()))
    monkeypatch.setattr("agent.run.make_port", spy_port)
    await two_setup_turns("t-real-port")
    before = approvals(conn)

    out = await run_turn(TURN_10, thread_id="t-real-port", **HR)
    assert out.status == "pending" and out.tools_called == []
    assert approvals(conn) == before + 1
    row = conn.execute("SELECT * FROM approvals WHERE approval_id = ?", (out.approval_id,)).fetchone()
    assert (row["status"], row["request_type"], row["approver_id"], row["request_id"]) == \
        ("pending", "access_request", "E006", "handoff:t-real-port:E001:S001")
    assert conn.execute("SELECT COUNT(*) FROM access_requests WHERE employee_id = 'E001' "
                        "AND system_id = 'S001'").fetchone()[0] == 0  # nothing filed

    await run_turn(TURN_10, thread_id="t-real-port", **HR)
    assert approvals(conn) == before + 1
    assert sum(getattr(p, "calls", 0) for p in made) == 1


async def test_subject_never_looked_up_is_looked_up_by_code_then_handed_off(seeded, monkeypatch,
                                                                         conn):
    """Task #5 replaces PR #4's needs_human: code calls get_employee for the id the
    planner named, and one match routes. E001 is preboarding, so it is a handoff."""
    port = FakeWebexPort()
    use_stubs(monkeypatch, ScriptedPlanner(SEATS, FILE_IT), ScriptedModel(*setup_replies()[2:]),
              port=port)
    await run_turn("Is a Webex seat available?", thread_id="t-lookup", **HR)
    out = await run_turn(TURN_10, thread_id="t-lookup", **HR)  # no get_employee row yet
    assert out.tools_called == ["get_employee"]
    assert out.status == "pending" and len(port.calls) == 1
    assert port.calls[0].subject_employee_id == "E001"


@pytest.mark.parametrize("subject, why", [
    ({"subject_employee_id": "E999"}, "no employee matches"),
    ({"subject_name": "Maya"}, "several employees match: Maya Cohen (E001), Maya Rosen (E020)"),
])
async def test_lookup_with_no_match_or_several_ends_needs_human(seeded, monkeypatch, conn,
                                                                subject, why):
    port = FakeWebexPort()
    before = approvals(conn)
    ask = plan("access_request", action_confirmed=True, system="Webex", **subject)
    use_stubs(monkeypatch, ScriptedPlanner(ask), ScriptedModel(), port=port)
    out = await run_turn(TURN_10, thread_id="t-no-subject", **HR)
    assert out.status == "needs_human" and out.tools_called == ["get_employee"]
    assert why in out.answer and "Nothing was filed" in out.answer
    assert port.calls == [] and out.approval_id is None and approvals(conn) == before


async def test_active_subject_takes_the_webex_approval_path_not_a_handoff(seeded, monkeypatch):
    port = FakeWebexPort()
    rachel = plan("access_request", action_confirmed=True, subject_employee_id="E010",
                  system="Webex")
    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP, rachel),
              ScriptedModel(call("get_employee", query="E010"),
                            say("Rachel Stein is E010. [db:employees#E010]"),
                            call("get_approvals", record_id="AR001"), say("Waits on AP001.")),
              port=port)
    await run_turn("Look up Rachel Stein.", thread_id="t-rachel", **HR)
    out = await run_turn("Go ahead and file it.", thread_id="t-rachel", **HR)
    assert out.scope == "webex" and out.status == "pending" and out.approval_id == "AP001"
    assert port.calls == [] and out.handoff is None


def test_maya_intents_never_route_to_the_approval_node():
    for intent, scope in policy.SCOPE_OF.items():
        if scope == "maya":
            assert not policy.needs_approval(plan(intent, action_confirmed=True))


# Citations -----------------------------------------------------------------------

def test_cited_labels_accepts_database_and_document_labels():
    known = {"[db:onboarding_tasks#OT005]", "[equipment_policy.md#Personal laptops]"}
    valid, invented = cited_labels(
        "OT005 is blocked [db:onboarding_tasks#OT005]. See [equipment_policy.md#Personal laptops]. "
        "Also [db:tickets#T999].", known)
    assert valid == ["[db:onboarding_tasks#OT005]", "[equipment_policy.md#Personal laptops]"]
    assert invented == ["[db:tickets#T999]"]


def test_uncited_sentences_finds_only_fact_sentences_without_a_label():
    answer = ("Maya Cohen starts on 2026-08-01. [db:employees#E001] Her Webex task is blocked "
              "by the seat limit. The manager playbook is not searchable for this caller. "
              "Want me to check Okta?")
    assert uncited_sentences(answer) == ["Her Webex task is blocked by the seat limit."]


async def test_uncited_answer_is_repaired_on_the_first_try(seeded, monkeypatch):
    model = ScriptedModel(call("get_employee", query="E001"),
                          say("Maya Cohen is a Customer Success Manager starting 2026-08-01."),
                          say("Maya Cohen is a Customer Success Manager starting 2026-08-01. "
                              "[db:employees#E001]"))
    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP), model)
    out = await run_turn("Who is Maya Cohen?", thread_id="t-cite-ok", **HR)
    assert out.status == "completed" and out.citations == ["[db:employees#E001]"]
    assert model.bound[-1] == []  # the repair call binds no tools
    assert "[db:employees#E001]" in out.answer


async def test_second_citation_miss_sets_needs_human(seeded, monkeypatch):
    model = ScriptedModel(call("get_employee", query="E001"),
                          say("Maya Cohen is a Customer Success Manager starting soon."),
                          say("Maya Cohen is a Customer Success Manager starting soon."))
    use_stubs(monkeypatch, ScriptedPlanner(LOOKUP), model)
    out = await run_turn("Who is Maya Cohen?", thread_id="t-cite-miss", **HR)
    assert out.status == "needs_human"


async def test_recall_turn_without_tool_results_is_not_checked(seeded, monkeypatch):
    model = ScriptedModel(say("You gave me Israel, hybrid, and a 2026-08-01 start date."))
    use_stubs(monkeypatch, ScriptedPlanner(plan("other", needs_tools=False)), model)
    out = await run_turn("Remind me what I told you.", thread_id="t-recall", **HR)
    assert out.status == "completed" and len(model.seen) == 1


# The checklist -------------------------------------------------------------------

def checklist_dict(**changes) -> dict:
    record = {
        "employee_id": "E001", "employee_name": "Maya Cohen",
        "role": "Customer Success Manager", "start_date": "2026-08-01",
        "items": [
            {"item": "Okta profile", "task_id": "OT002", "status": "pending", "owner": "IT",
             "citations": [{"label": "[db:onboarding_tasks#OT002]"}]},
            {"item": "Webex standard user license", "task_id": "OT005", "status": "blocked",
             "owner": "IT", "blocked_reason": "Webex has 42 active seats on a 40-seat limit.",
             "citations": [{"label": "[db:onboarding_tasks#OT005]"}]},
        ],
        "overall_status": "blocked",
    }
    record.update(changes)
    return record


def onboarding_replies():
    return [call("list_onboarding_tasks", employee_id="E001"),
            say("OT005 Webex is blocked. [db:onboarding_tasks#OT005]")]


async def test_checklist_lists_ot005_blocked_with_citations(seeded, monkeypatch):
    checklist = ScriptedPlanner(checklist_dict())
    use_stubs(monkeypatch, ScriptedPlanner(plan("onboarding_status")),
              ScriptedModel(*onboarding_replies()), checklist=checklist)
    out = await run_turn("What's on her checklist?", thread_id="t-check", **HR)
    assert out.status == "completed" and out.checklist is not None
    webex = next(i for i in out.checklist.items if i.task_id == "OT005")
    assert webex.status == "blocked" and webex.blocked_reason
    assert all(i.citations for i in out.checklist.items)
    assert "OT005" in out.answer  # the prose answer stays


async def test_checklist_parse_failure_is_repaired_once(seeded, monkeypatch):
    bad = checklist_dict(overall_status="on_track")  # contradicts the blocked item
    checklist = ScriptedPlanner(bad, checklist_dict())
    use_stubs(monkeypatch, ScriptedPlanner(plan("onboarding_status")),
              ScriptedModel(*onboarding_replies()), checklist=checklist)
    out = await run_turn("What's on her checklist?", thread_id="t-check-repair", **HR)
    assert out.checklist is not None and checklist.calls == 2


async def test_checklist_second_failure_raises_parse_failed(seeded, monkeypatch):
    invented = checklist_dict()
    invented["items"][0]["citations"] = [{"label": "[db:onboarding_tasks#OT999]"}]
    checklist = ScriptedPlanner(invented, None)
    use_stubs(monkeypatch, ScriptedPlanner(plan("onboarding_status")),
              ScriptedModel(*onboarding_replies()), checklist=checklist)
    out = await run_turn("What's on her checklist?", thread_id="t-check-fail", **HR)
    assert out.status == "failed" and "OnboardingChecklist" in out.answer
    assert checklist.calls == 2


async def test_no_checklist_without_onboarding_rows_this_turn(seeded, monkeypatch):
    checklist = ScriptedPlanner()  # would raise IndexError if called
    use_stubs(monkeypatch, ScriptedPlanner(plan("other", needs_tools=False),
                                           plan("onboarding_status", needs_tools=False)),
              ScriptedModel(say("Hello."), say("Sign the equipment form and finish BambooHR.")),
              checklist=checklist)
    await run_turn("Hi.", thread_id="t-no-check", **HR)
    out = await run_turn("What is mine to do?", thread_id="t-no-check", **HR)
    assert out.checklist is None and checklist.calls == 0


def test_checklist_schema_rejects_blocked_item_without_reason_or_citation():
    no_reason = checklist_dict()
    no_reason["items"][1]["blocked_reason"] = None
    with pytest.raises(ValidationError):
        OnboardingChecklist.model_validate(no_reason)
    no_citation = checklist_dict()
    no_citation["items"][0]["citations"] = []
    with pytest.raises(ValidationError):
        OnboardingChecklist.model_validate(no_citation)
    with pytest.raises(ValidationError):
        EvidenceCitation(label="OT005")


# Distillation --------------------------------------------------------------------

def test_memory_that_drops_a_constraint_is_rejected():
    old = {"active_constraints": ["Q3 SaaS freeze applies"]}
    dropped = ConversationMemory(current_intent="onboarding", important_facts=["E001"])
    kept = ConversationMemory(current_intent="onboarding", important_facts=["E001"],
                              active_constraints=["Q3 SaaS freeze applies"])
    assert not memory_is_safe(old, dropped) and memory_is_safe(old, kept)
    assert not memory_is_safe({}, ConversationMemory(current_intent="x"))


async def test_distillation_runs_past_the_threshold_and_reaches_the_planner(seeded, monkeypatch):
    memory = ConversationMemory(current_intent="Maya Cohen onboarding",
                                important_facts=["Maya Cohen is E001", "starts 2026-08-01"],
                                active_constraints=["Q3 SaaS freeze applies"])
    distiller = ScriptedPlanner(memory)
    lookups = [plan("employee_lookup")] * 4
    replies = []
    for n in range(1, 5):
        replies += [call("get_employee", query=f"E00{n}"), say(f"E00{n} found. [db:employees#E00{n}]")]
    planner = ScriptedPlanner(*lookups, plan("other", needs_tools=False))
    seen = []
    real = planner.ainvoke

    async def spy(messages):
        seen.append(messages[-1].content)
        return await real(messages)
    planner.ainvoke = spy
    use_stubs(monkeypatch, planner, ScriptedModel(*replies, say("Noted.")), distiller=distiller)
    for n in range(1, 5):
        await run_turn(f"Look up E00{n}.", thread_id="t-distil", **HR)
    await run_turn("Thanks.", thread_id="t-distil", **HR)
    assert distiller.calls == 1
    assert "Session memory:" in seen[-1] and "Q3 SaaS freeze applies" in seen[-1]
    assert "Look up E001." not in seen[-1]  # covered by memory, no longer resent


# The tool matrix -----------------------------------------------------------------

def test_tool_matrix_passes():
    assert tool_matrix.check() == []


def test_tool_matrix_catches_an_unreachable_expected_tool(monkeypatch):
    narrowed = dict(policy.LOADOUTS)
    narrowed["subscription_review"] = [t for t in narrowed["subscription_review"]
                                       if t != "search_hr_documents"]
    monkeypatch.setattr(tool_matrix, "LOADOUTS", narrowed)
    problems = tool_matrix.check()
    assert any("S2-onboarding-maya turn 9" in p and "subscription_review" in p for p in problems)


def test_handoff_model_admits_only_the_webex_workflow():
    with pytest.raises(ValidationError):
        WebexHandoff(workflow="direct_write", caller_id="E004", subject_employee_id="E001",
                     system_id="S001", software="Webex", justification="x",
                     source_thread_id="t", request_id="r", idempotency_key="k")


def test_first_turn_that_claims_no_lookup_is_corrected():
    first = policy.checked_plan(plan("policy_question", needs_tools=False), has_history=False)
    assert first.needs_tools is True
    greeting = policy.checked_plan(plan("other", needs_tools=False), has_history=False)
    assert greeting.needs_tools is False
    later = policy.checked_plan(plan("policy_question", needs_tools=False), has_history=True)
    assert later.needs_tools is False
    now = policy.checked_plan(plan("onboarding_status", needs_tools=False), has_history=True,
                              message="And what's actually on her checklist right now?")
    assert now.needs_tools is True
    recall = policy.checked_plan(plan("onboarding_status", needs_tools=False), has_history=True,
                                 message="Remind me which start date I gave you.")
    assert recall.needs_tools is False
    pasted = policy.checked_plan(plan("onboarding_status", needs_tools=False), has_history=True,
                                 message="What here is mine to do?\n\n> Okta is still pending.")
    assert pasted.needs_tools is False


def test_invented_and_user_stated_sentences():
    valid, invented = cited_labels("The freeze applies [constraint]. Seats are 42 [db:assets].",
                                   {"[db:assets]"})
    assert valid == ["[db:assets]"] and invented == ["[constraint]"]
    assert uncited_sentences("As you said, the Q3 SaaS freeze applies to anything new.") == []


def test_access_note_is_added_when_a_non_manager_asks_for_manager_material():
    from langchain_core.messages import HumanMessage, ToolMessage
    from agent.graph import with_access_note
    from mcp_server.retrieval import PLAYBOOK_NOTE
    messages = [HumanMessage("What does the manager's guide say about feedback?"),
                ToolMessage(content=f'{{"access_note": "{PLAYBOOK_NOTE}", "chunks": []}}',
                            tool_call_id="1", name="search_hr_documents")]
    assert with_access_note("The handbook says ...", messages).startswith(PLAYBOOK_NOTE)
    messages[0] = HumanMessage("How do I get promoted?")
    assert with_access_note("The handbook says ...", messages) == "The handbook says ..."


async def test_refined_round_reads_the_checklist_once_when_it_was_skipped(seeded, monkeypatch):
    model = ScriptedModel(call("check_asset_inventory", asset_type="Laptop"),
                          say("A MacBook is reserved for you. [db:assets#A005]"),
                          call("list_onboarding_tasks", employee_id="E001"),
                          say("OT003 prepares your laptop by 2026-07-26 "
                              "[db:onboarding_tasks#OT003], and A005 is reserved "
                              "[db:assets#A005]."))
    use_stubs(monkeypatch, ScriptedPlanner(plan("equipment_request",
                                                subject_employee_id="Maya Cohen")), model)
    out = await run_turn("Am I getting a laptop before the 1st?", caller_id="E001",
                         user_group="UG_REGULAR", thread_id="t-refine")
    assert out.tools_called == ["check_asset_inventory", "list_onboarding_tasks"]
    assert "2026-07-26" in out.answer and "MacBook is reserved for you." not in out.answer
    assert "list_onboarding_tasks for E001" in model.seen[2][0].content
    assert model.bound[2] == ["list_onboarding_tasks"]  # the refined call is forced
    assert all(t not in model.bound[3] for t in policy.NOT_FOR_THE_MODEL)


def test_system_comes_from_the_message_when_the_plan_omits_it():
    from langchain_core.messages import HumanMessage, ToolMessage
    from agent.handoff import system_named_in_message
    rows = '{"rows": [{"system_id": "S001", "system_name": "Webex"}]}'
    messages = [HumanMessage("Seats?"), ToolMessage(content=rows, tool_call_id="1",
                                                    name="check_software_subscription"),
                HumanMessage(TURN_10)]
    assert system_named_in_message(messages) == {"system_id": "S001", "system_name": "Webex"}
    messages[-1] = HumanMessage("Go ahead and file it.")
    assert system_named_in_message(messages) is None
