"""The Webex access case, decided from tool results (spec §9, Milestone 4).

The graph's approval node reads the case from code: the subscription, the subject's
tickets, their access requests and the approval rows of the open request. It then
asks for approval through `request_approval`, which reuses `AP001` (access request,
E006) and `AP002` (subscription expansion, E011) instead of creating rows. The gate
opens on the `access_request` approval only; `AP002` is requested so the trace and
the decision show the durable expansion approval the rubric asks for, and the
Renewal workflow owns the expansion itself.

Everything here is a pure function over tool results, so the rules are testable
without a graph:

- `case_reads()` and `approval_requests()`: what the approval node calls, in order.
- `assemble_decision()`: the `AccessDecision`. The model's validated `DecisionDraft`
  gives the observed facts and next steps. Code gives the status, the blocking
  reason, the reused records and the actions taken, each from a row a tool returned.
- `render_decision()`: the answer text, written by code from the decision, so it
  cannot claim more than the decision holds.

Starting code: none. Lesson 11 homework milestone 2 names the cases (seat-limit
block, reuse of T001, an approval not given yet, a refusal); the rules are ours.
"""

from agent.outcome import tool_results
from agent.ports import WEBEX_APPROVER
from agent.schemas import (AccessDecision, ActionTaken, DecisionDraft, EvidenceCitation,
                           RecordRef)

# Mirrors OPEN_ACCESS_REQUEST and CLOSED_TICKET in mcp_server/writes.py, which decide
# what the server reuses; the agent reports the same rows as reused.
OPEN_ACCESS_REQUEST = ("pending_approval", "blocked", "planned", "waiting_on_manager")
CLOSED_TICKET = ("resolved", "closed", "cancelled")
OPEN_APPROVAL = ("needed", "pending")
WRITE_RESULTS = ("request_approval", "create_access_request", "create_ticket")
# The approval types the node asks for, in order. Only the first one gates a write.
APPROVAL_TYPES = ("access_request", "subscription_expansion")
# Spec §9 step 3, "search the knowledge base and the relevant policies", from code: one
# search for the problem (licence and login articles, the contract's seat limit, the
# cleanup memo) and one for the rule (who approves access). The model may search too;
# these make the evidence the decision rests on independent of what it chose to ask.
CASE_QUERIES = ("{system} access not licensed seat limit",
                "who approves access to {system}: access policy and approval tiers")


def rows_of(messages: list, tool: str) -> list[dict]:
    return [row for result in tool_results(messages, tool) for row in result.get("rows", [])]


def subscription(messages: list, system: str) -> dict | None:
    """The subscription row for `system` (a name or a system id), newest result last."""
    wanted = system.strip().lower()
    found = [row for row in rows_of(messages, "check_software_subscription")
             if wanted in (str(row.get("system_name", "")).lower(),
                           str(row.get("system_id", "")).lower(),
                           str(row.get("vendor_name", "")).lower())]
    return found[-1] if found else None


def system_id(messages: list, system: str) -> str | None:
    """The system id for a name, from the subscription or an access request row."""
    if sub := subscription(messages, system):
        return sub["system_id"]
    wanted = system.strip().lower()
    for row in rows_of(messages, "get_access_requests"):
        if wanted in (str(row.get("system_name", "")).lower(), str(row.get("system_id", "")).lower()):
            return row["system_id"]
    return None


def open_access_request(messages: list, employee_id: str, system_id: str) -> dict | None:
    """The open request the server would reuse for this employee and system (AR001)."""
    found = [row for row in rows_of(messages, "get_access_requests")
             if row.get("employee_id") == employee_id and row.get("system_id") == system_id
             and row.get("status") in OPEN_ACCESS_REQUEST]
    return min(found, key=lambda r: r.get("created_at", "")) if found else None


def open_tickets(messages: list, system_id: str) -> list[dict]:
    """Open tickets about this system from `list_employee_tickets` results (T001)."""
    seen = {}
    for row in rows_of(messages, "list_employee_tickets"):
        if row.get("related_system_id") == system_id and row.get("status") not in CLOSED_TICKET:
            seen[row["ticket_id"]] = row
    return list(seen.values())


def approval_requests(approval_rows: list[dict], record_id: str) -> list[dict]:
    """The `request_approval` arguments for the open access request, one per approval
    type. The approver comes from the existing row of that type (AP001 names E006,
    AP002 names E011). The access approval is always asked for, with the Webex
    approver when no row names one; the expansion only when a row for it exists,
    because no record names its approver otherwise."""
    requests = []
    for request_type in APPROVAL_TYPES:
        typed = [r for r in approval_rows if r.get("request_type") == request_type]
        waiting = [r for r in typed if r.get("status") in OPEN_APPROVAL] or typed
        if waiting:
            approver = waiting[0]["approver_id"]
        elif request_type == "access_request":
            approver = WEBEX_APPROVER
        else:
            continue
        requests.append({"request_type": request_type, "record_id": record_id,
                         "approver_id": approver})
    return requests


def new_request_record(employee_id: str, system_id: str) -> str:
    """The record id an approval is requested for when no access request exists yet.
    `create_access_request` files the row once the approval is decided."""
    return f"access:{employee_id}:{system_id}"


def seat_blocker(sub: dict | None) -> str | None:
    if not sub or not sub.get("over_limit"):
        return None
    return (f"{sub['system_name']} has {sub['active_seats']} active seats against a "
            f"{sub['seat_limit']}-seat limit {sub['citation']}. The subscription is full "
            "for everyone, so no new seat can be assigned until seats are freed or the "
            f"subscription is expanded {sub['citation']}.")


def _citation(row: dict) -> EvidenceCitation:
    return EvidenceCitation(label=row["citation"])


def reused_records(messages: list, access_request: dict | None, system_id: str) -> list[RecordRef]:
    refs = [RecordRef(record_type="ticket", record_id=t["ticket_id"], status=t["status"],
                      citation=_citation(t))
            for t in open_tickets(messages, system_id) if t.get("citation")]
    if access_request and access_request.get("citation"):
        refs.append(RecordRef(record_type="access_request",
                              record_id=access_request["request_id"],
                              status=access_request["status"],
                              citation=_citation(access_request)))
    return refs


def _action_detail(tool: str, result: dict) -> str:
    if tool == "request_approval":
        how = "reused the existing row" if result.get("reused") else "created a new pending row"
        return (f"Asked {result['approver_id']} to approve {result['approval_id']} "
                f"({result['request_type']} for {result['request_id']}): {how}, status "
                f"{result['status']}.")
    record = result.get("request_id") or result.get("ticket_id")
    if result.get("reused"):
        return (f"{tool} ran under {result['approval_id']} and returned the existing "
                f"{record} (status {result['status']}) with reused: true. Nothing new was "
                "filed.")
    return f"{tool} filed {record} with status {result['status']} under {result['approval_id']}."


def actions_taken(messages: list) -> list[ActionTaken]:
    """One entry per write-path tool result that returned a row id, newest state of
    each row. Reads are not actions."""
    actions: dict[tuple, ActionTaken] = {}
    for tool in WRITE_RESULTS:
        for result in tool_results(messages, tool):
            record = result.get("approval_id") if tool == "request_approval" else (
                result.get("request_id") or result.get("ticket_id"))
            if not record or not result.get("citation"):
                continue
            actions[(tool, record)] = ActionTaken(
                tool=tool, record_id=record, reused=bool(result.get("reused")),
                detail=_action_detail(tool, result), citation=_citation(result))
    return list(actions.values())


def write_result(messages: list) -> dict | None:
    results = tool_results(messages, "create_access_request")
    return results[-1] if results else None


def decision_status(gate: dict | None, write: dict | None) -> str:
    """Set by code. No gate yet means the run is pausing on the approval. A rejected
    approval blocks. A released write that returned a blocked request (AR001 under the
    seat limit) is blocked; a filed request is completed. A gate that opened without a
    write result means the write failed or never ran: a person has to look."""
    if gate is None:
        return "pending"
    if gate["state"] == "rejected":
        return "blocked"
    if write is not None:
        return "blocked" if write.get("status") == "blocked" else "completed"
    return "needs_human"


def blocking_reason(gate: dict | None, sub: dict | None,
                    access_request: dict | None, write: dict | None) -> str | None:
    parts = []
    if gate and gate["state"] == "rejected":
        parts.append(gate["reason"])
    if seat := seat_blocker(sub):
        parts.append(seat)
    record = write or access_request
    if record and record.get("status") == "blocked" and record.get("decision_reason"):
        parts.append(f"{record['request_id']} is blocked: {record['decision_reason']} "
                     f"{record['citation']}")
    return " ".join(parts) or None


def assemble_decision(draft: DecisionDraft, *, messages: list, employee_id: str,
                      system: str, system_id: str, gate: dict | None) -> AccessDecision:
    sub = subscription(messages, system_id) or subscription(messages, system)
    access_request = open_access_request(messages, employee_id, system_id)
    write = write_result(messages)
    status = decision_status(gate, write)
    return AccessDecision(
        employee_id=employee_id, system=system, status=status,
        observed_facts=draft.observed_facts,
        reused_records=reused_records(messages, access_request, system_id),
        actions_taken=actions_taken(messages),
        recommended_next_steps=draft.recommended_next_steps,
        blocking_reason=blocking_reason(gate, sub, access_request, write))


def _with_labels(statement: str, labels: list[str]) -> str:
    missing = [label for label in labels if label not in statement]
    return f"{statement} {' '.join(missing)}".strip()


def closing_line(decision: AccessDecision, approval: dict | None, gate: dict | None) -> str:
    if decision.status == "pending" and approval:
        # Cited like every other fact sentence: the approval row says who decides.
        label = f" {approval['citation']}" if approval.get("citation") else ""
        return (f"Nothing is filed and no access is granted until {approval['approver_id']} "
                f"decides on {approval['approval_id']}{label}.")
    if gate and gate["state"] == "rejected":
        return "Nothing was filed, and no access is granted."
    if decision.status == "blocked":
        return "The request is on file but blocked, so no access is granted yet."
    return "Filing a request does not grant access; the request's approver still decides."


def render_decision(decision: AccessDecision, approval: dict | None, gate: dict | None) -> str:
    lines = [f"Status: {decision.status}."]
    if decision.blocking_reason:
        lines.append(f"Blocker: {decision.blocking_reason}")
    lines.append("What the records show:")
    lines += [f"- {_with_labels(f.statement, [c.label for c in f.citations])}"
              for f in decision.observed_facts]
    if decision.reused_records:
        lines.append("Existing records, reused rather than duplicated: " + "; ".join(
            f"{r.record_id} ({r.record_type.replace('_', ' ')}, {r.status}) {r.citation.label}"
            for r in decision.reused_records) + ".")
    lines.append("Actions taken:" if decision.actions_taken else "Actions taken: none.")
    lines += [f"- {a.detail} {a.citation.label}" for a in decision.actions_taken]
    lines.append("Next steps:")
    lines += [f"- {step}" for step in decision.recommended_next_steps]
    lines.append(closing_line(decision, approval, gate))
    return "\n".join(lines)
