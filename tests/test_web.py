"""The web chat (task #12): a thin layer over run_turn() and record_approval_decision().

The endpoint tests run the FastAPI app in process. The end-to-end approval test runs the
real run_turn() with the scripted models and the in-memory MCP server from
tests/agent_stubs.py, so the only thing that differs from a live demo is the model.
"""

import ast

import pytest
from fastapi.testclient import TestClient

import web.app
from agent.outcome import TurnOutcome
from tests.agent_stubs import ScriptedModel, ScriptedPlanner, call, plan, say, use_stubs
from tests.conftest import REPO

# What web/ may take from the workflow packages. agent.config only supplies the database
# path and the MCP URL, so the chat reads the same files the agent does.
ALLOWED_IMPORTS = {("agent.run", "run_turn"),
                   ("mcp_server.writes", "record_approval_decision"),
                   ("agent.config", "db_path"), ("agent.config", "mcp_server_url")}


@pytest.fixture
def client(seeded, monkeypatch):
    monkeypatch.setattr(web.app.settings, "mcp_reachable", lambda: True)
    monkeypatch.setattr(web.app, "_thread_callers", {})
    monkeypatch.setattr(web.app, "_busy", set())
    with TestClient(web.app.app) as test_client:
        yield test_client


def _workflow_imports() -> set[tuple[str, str]]:
    found = set()
    for path in (REPO / "web").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {(a.name, "*") for a in node.names
                          if a.name.split(".")[0] in ("agent", "mcp_server")}
            elif isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.split(".")[0] in ("agent", "mcp_server"):
                found |= {(node.module, a.name) for a in node.names}
    return found


def test_web_imports_only_run_turn_the_decision_function_and_config():
    imports = _workflow_imports()
    assert imports <= ALLOWED_IMPORTS, imports - ALLOWED_IMPORTS
    assert ("agent.run", "run_turn") in imports
    assert ("mcp_server.writes", "record_approval_decision") in imports


def test_turn_passes_caller_group_from_the_db_and_thread_unchanged(client, monkeypatch, conn):
    seen = []

    async def fake_run_turn(message, **kwargs):
        seen.append({"message": message, **kwargs})
        return TurnOutcome(answer="ok", status="completed", request_id="REQ-FAKE")

    monkeypatch.setattr(web.app, "run_turn", fake_run_turn)
    group = conn.execute("SELECT user_group_id FROM employees WHERE employee_id = 'E006'"
                         ).fetchone()[0]
    res = client.post("/api/turn", json={"message": "Hello", "caller_id": "E006",
                                         "thread_id": "thread-123", "user_group": "UG_HR"})
    assert res.status_code == 200 and res.json()["status"] == "completed"
    # The page cannot choose the group: an extra field in the body is ignored.
    assert seen == [{"message": "Hello", "caller_id": "E006", "user_group": group,
                     "thread_id": "thread-123"}]


def test_a_thread_keeps_its_caller(client, monkeypatch):
    async def fake_run_turn(message, **kwargs):
        return TurnOutcome(answer="ok", status="completed", request_id="R")

    monkeypatch.setattr(web.app, "run_turn", fake_run_turn)
    assert client.post("/api/turn", json={"message": "a", "caller_id": "E010",
                                          "thread_id": "t1"}).status_code == 200
    res = client.post("/api/turn", json={"message": "b", "caller_id": "E006", "thread_id": "t1"})
    assert res.status_code == 409 and "belongs to E010" in res.json()["detail"]


def test_unknown_caller_and_unreachable_mcp_are_refused_before_run_turn(client, monkeypatch):
    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("run_turn should not be called")

    monkeypatch.setattr(web.app, "run_turn", must_not_run)
    assert client.post("/api/turn", json={"message": "a", "caller_id": "E999",
                                          "thread_id": "t"}).status_code == 404
    monkeypatch.setattr(web.app.settings, "mcp_reachable", lambda: False)
    res = client.post("/api/turn", json={"message": "a", "caller_id": "E010", "thread_id": "t"})
    assert res.status_code == 503 and "python -m mcp_server" in res.json()["detail"]


def test_approvals_panel_lists_only_the_selected_callers_open_approvals(client, conn):
    approvers = [r[0] for r in conn.execute("SELECT DISTINCT approver_id FROM approvals")]
    assert "E006" in approvers
    for approver in approvers + ["E010"]:
        listed = [a["approval_id"] for a in client.get(
            "/api/approvals", params={"caller_id": approver}).json()]
        expected = [r[0] for r in conn.execute(
            "SELECT approval_id FROM approvals WHERE approver_id = ? "
            "AND status IN ('needed', 'pending') ORDER BY created_at, approval_id", (approver,))]
        assert listed == expected, approver
    e006 = client.get("/api/approvals", params={"caller_id": "E006"}).json()
    assert "AP001" in [a["approval_id"] for a in e006]
    assert client.get("/api/approvals", params={"caller_id": "E010"}).json() == []


def test_only_the_named_approver_can_decide_from_the_panel(client, conn):
    res = client.post("/api/approvals/AP001/decision", json={"caller_id": "E007",
                                                             "decision": "approved"})
    assert res.status_code == 409 and "E006" in res.json()["detail"]
    assert conn.execute("SELECT status FROM approvals WHERE approval_id = 'AP001'"
                        ).fetchone()[0] == "needed"


def test_approve_in_the_panel_then_a_message_releases_the_write_once(client, monkeypatch, conn):
    use_stubs(monkeypatch,
              ScriptedPlanner(plan("employee_lookup"),
                              plan("access_request", action_confirmed=True,
                                   subject_employee_id="E010", system="Webex")),
              ScriptedModel(call("get_employee", query="E010"),
                            say("Rachel Stein is E010. [db:employees#E010]"),
                            call("get_approvals", record_id="AR001"), say("Waits on AP001."),
                            call("create_access_request", employee_id="E010", software="Webex",
                                 business_justification="Customer onboarding calls require Webex.")))
    before = conn.execute("SELECT COUNT(*) FROM access_requests").fetchone()[0]

    def turn(message):
        res = client.post("/api/turn", json={"message": message, "caller_id": "E007",
                                             "thread_id": "t-web-approval"})
        assert res.status_code == 200, res.text
        return res.json()

    turn("Look up Rachel Stein.")
    paused = turn("Go ahead and file Webex access for her.")
    assert paused["status"] == "pending" and paused["approval_id"] == "AP001"
    assert "AP001" in [a["approval_id"] for a in client.get(
        "/api/approvals", params={"caller_id": "E006"}).json()]

    decided = client.post("/api/approvals/AP001/decision",
                          json={"caller_id": "E006", "decision": "approved"})
    assert decided.status_code == 200 and decided.json()["status"] == "approved"
    assert "AP001" not in [a["approval_id"] for a in client.get(
        "/api/approvals", params={"caller_id": "E006"}).json()]
    again = client.post("/api/approvals/AP001/decision",
                        json={"caller_id": "E006", "decision": "approved"})
    assert again.status_code == 409

    resumed = turn("E006 approved it in the panel.")
    assert resumed["request_id"] == paused["request_id"]
    assert resumed["tools_called"].count("create_access_request") == 1
    write = [a for a in resumed["access_decision"]["actions_taken"]
             if a["tool"] == "create_access_request"]
    assert [(a["record_id"], a["reused"]) for a in write] == [("AR001", True)]
    assert conn.execute("SELECT COUNT(*) FROM access_requests").fetchone()[0] == before
    assert conn.execute("SELECT COUNT(*) FROM approval_controls WHERE approval_id = 'AP001' "
                        "AND consumed_at IS NOT NULL").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM audit_log WHERE event_type = "
                        "'access_request_reused'").fetchone()[0] == 1
