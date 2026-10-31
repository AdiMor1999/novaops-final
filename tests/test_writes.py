"""Approval-gated writes, idempotency and request_approval, against a reseeded file DB."""

import pytest

from mcp_server import writes
from mcp_server.db import ToolInputError
from mcp_server.writes import WriteRefused
from tests.conftest import caller

CTX = caller("E007", "UG_IT", "REQ-T1")


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def approve_access(conn, employee="E016", software="Figma", ctx=CTX, record="REQ-T1-figma"):
    pending = writes.request_approval(conn, ctx, "access_request", record, "E006",
                                      employee_id=employee, software=software)
    writes.record_approval_decision(pending["approval_id"], "approved", "E006", conn=conn)
    return pending["approval_id"]


def test_gated_write_without_approval_is_refused_and_writes_nothing(conn):
    before = {t: count(conn, t) for t in ("access_requests", "tickets", "audit_log")}
    with pytest.raises(WriteRefused):
        writes.create_access_request(conn, CTX, "E016", "Figma", "Design reviews")
    with pytest.raises(WriteRefused):
        writes.create_ticket(conn, CTX, "E016", "Figma login fails", "SSO loop", "Access", "Figma")
    assert {t: count(conn, t) for t in before} == before


def test_pending_approval_does_not_open_a_write(conn):
    writes.request_approval(conn, CTX, "access_request", "REQ-T1-figma", "E006",
                            employee_id="E016", software="Figma")
    with pytest.raises(WriteRefused):
        writes.create_access_request(conn, CTX, "E016", "Figma", "Design reviews")


def test_same_key_twice_makes_one_row_second_reused(conn):
    approve_access(conn)
    before = count(conn, "access_requests")
    first = writes.create_access_request(conn, CTX, "E016", "Figma", "Design reviews")
    second = writes.create_access_request(conn, CTX, "E016", "Figma", "Design reviews")
    assert first["reused"] is False and first["status"] == "pending_approval"
    assert second["reused"] is True and second["request_id"] == first["request_id"]
    assert count(conn, "access_requests") == before + 1


def test_one_approval_opens_one_write_only(conn):
    pending = writes.request_approval(conn, CTX, "ticket", "REQ-T1-ticket", "E006",
                                      employee_id="E016")
    writes.record_approval_decision(pending["approval_id"], "approved", "E006", conn=conn)
    first = writes.create_ticket(conn, CTX, "E016", "Laptop fan noise", "Loud fan", "Hardware")
    assert first["reused"] is False
    with pytest.raises(WriteRefused):
        writes.create_ticket(conn, CTX, "E016", "Second issue", "Another one", "Hardware")
    replay = writes.create_ticket(conn, CTX, "E016", "Laptop fan noise", "Loud fan", "Hardware")
    assert replay["reused"] is True and replay["ticket_id"] == first["ticket_id"]


def test_approval_is_bound_to_caller_and_request(conn):
    approve_access(conn)
    with pytest.raises(WriteRefused):
        writes.create_access_request(conn, caller("E007", "UG_IT", "REQ-OTHER"),
                                     "E016", "Figma", "Design reviews")
    with pytest.raises(WriteRefused):
        writes.create_access_request(conn, caller("E008", "UG_IT", "REQ-T1"),
                                     "E016", "Figma", "Design reviews")


def test_expired_approval_refused(conn, monkeypatch):
    approve_access(conn)
    monkeypatch.setenv("NOVAOPS_TODAY", "2026-07-20")
    with pytest.raises(WriteRefused):
        writes.create_access_request(conn, CTX, "E016", "Figma", "Design reviews")


def consumed_key(conn, approval_id):
    return conn.execute("SELECT consumed_key FROM approval_controls WHERE approval_id = ?",
                        (approval_id,)).fetchone()[0]


def test_existing_open_request_is_reused_and_still_consumes_the_approval(conn):
    """AR001 (E010, Webex, blocked) is the open request; a write for it returns AR001.
    Returning it is the release, so AP001 is used up by it (task #5)."""
    pending = writes.request_approval(conn, CTX, "access_request", "AR001", "E006")
    assert pending["approval_id"] == "AP001" and pending["reused"] is True
    writes.record_approval_decision("AP001", "approved", "E006", conn=conn)
    before = count(conn, "access_requests")
    result = writes.create_access_request(conn, CTX, "E010", "Webex", "Customer calls")
    assert result["request_id"] == "AR001" and result["reused"] is True
    assert consumed_key(conn, "AP001") == "create_access_request:E010:S001:REQ-T1"
    replay = writes.create_access_request(conn, CTX, "E010", "Webex", "Customer calls")
    assert replay["request_id"] == "AR001" and replay["reused"] is True
    assert count(conn, "access_requests") == before
    assert conn.execute("SELECT COUNT(*) FROM audit_log WHERE event_type = "
                        "'access_request_reused'").fetchone()[0] == 1


def test_a_consumed_approval_does_not_release_another_request(conn):
    """The action key is (employee, system, workflow request), so a new action is a new
    request. AP001, consumed by REQ-T1, opens nothing for REQ-T2, even after REQ-T2
    asks for it: request_approval gives REQ-T2 its own pending row."""
    writes.request_approval(conn, CTX, "access_request", "AR001", "E006")
    writes.record_approval_decision("AP001", "approved", "E006", conn=conn)
    writes.create_access_request(conn, CTX, "E010", "Webex", "Customer calls")
    other = caller("E007", "UG_IT", "REQ-T2")
    writes.request_approval(conn, other, "access_request", "AR001", "E006")
    with pytest.raises(WriteRefused):
        writes.create_access_request(conn, other, "E010", "Webex", "Customer calls")


def test_request_approval_after_release_returns_the_same_consumed_row(conn):
    """Replaying the workflow request after its write finds AP001 again (this_request),
    instead of opening a second approval for an action that already happened."""
    writes.request_approval(conn, CTX, "access_request", "AR001", "E006")
    writes.record_approval_decision("AP001", "approved", "E006", conn=conn)
    writes.create_access_request(conn, CTX, "E010", "Webex", "Customer calls")
    before = count(conn, "approvals")
    again = writes.request_approval(conn, CTX, "access_request", "AR001", "E006")
    assert (again["approval_id"], again["consumed"], again["this_request"]) == ("AP001", 1, True)
    other = writes.request_approval(conn, caller("E007", "UG_IT", "REQ-T2"), "access_request",
                                    "AR001", "E006")
    assert other["approval_id"] != "AP001" and other["this_request"] is True
    assert count(conn, "approvals") == before + 1


def test_request_approval_twice_makes_one_pending_row(conn):
    before = count(conn, "approvals")
    first = writes.request_approval(conn, CTX, "access_request", "REQ-T1-figma", "E006",
                                    employee_id="E016", software="Figma")
    second = writes.request_approval(conn, CTX, "access_request", "REQ-T1-figma", "E006",
                                     employee_id="E016", software="Figma")
    assert first["status"] == "pending" and first["reused"] is False
    assert second["reused"] is True and second["approval_id"] == first["approval_id"]
    assert count(conn, "approvals") == before + 1


def test_decision_only_by_named_approver_and_only_once(conn):
    pending = writes.request_approval(conn, CTX, "access_request", "REQ-T1-figma", "E006",
                                      employee_id="E016", software="Figma")
    with pytest.raises(WriteRefused):
        writes.record_approval_decision(pending["approval_id"], "approved", "E007", conn=conn)
    writes.record_approval_decision(pending["approval_id"], "rejected", "E006", conn=conn)
    with pytest.raises(WriteRefused):
        writes.record_approval_decision(pending["approval_id"], "approved", "E006", conn=conn)


def test_caller_identity_is_checked(conn):
    with pytest.raises(WriteRefused):
        writes.request_approval(conn, caller("E999", "UG_IT"), "access_request", "AR001", "E006")
    with pytest.raises(WriteRefused):
        writes.request_approval(conn, caller("E010", "UG_IT"), "access_request", "AR001", "E006")


def test_bad_arguments_are_tool_errors(conn):
    with pytest.raises(ToolInputError):
        writes.request_approval(conn, CTX, "grant_everything", "AR001", "E006")
    with pytest.raises(ToolInputError):
        writes.create_access_request(conn, CTX, "E016", "NoSuchSystem", "x")
