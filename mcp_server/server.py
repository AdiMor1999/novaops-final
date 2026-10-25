"""NovaOps MCP server: documents, database and approvals behind one boundary.

Starting code: Lesson 10 `server/server.py` (the ten tools). Changes: the file database,
OpenSearch retrieval in place of FAISS, port 9880, caller context from request `_meta`
on every call (`mcp_server/context.py`), and the tools the workflows need that no
lesson ships:

  documents  list_policies · get_policy · search_knowledge_base · search_hr_documents
  people     get_employee · get_team
  IT         check_software_subscription · check_asset_inventory · list_employee_tickets
  HR         list_onboarding_tasks
  requests   get_access_requests · get_approvals
  approval   request_approval            (pending row only; idempotent; needs no approval)
  writes     create_access_request · create_ticket   (approval-gated, idempotent)

`record_approval_decision` is not here. It lives in `writes.py` and only application
code calls it, so no client of this server can approve anything.

    python -m mcp_server        # serves MCP_SERVER_URL (http://127.0.0.1:9880/mcp)
"""

import logging
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlparse

from langfuse import get_client, propagate_attributes
from mcp.server.fastmcp import Context, FastMCP

from agent import config
from mcp_server import db, retrieval, writes
from mcp_server.context import CallerContext, from_request
from mcp_server.db import ToolInputError
from mcp_server.documents import citation_label, parse_document, split_sections

log = logging.getLogger(__name__)


def _bind() -> tuple[str, int]:
    url = urlparse(config.mcp_server_url())
    return url.hostname or "127.0.0.1", url.port or 9880


_host, _port = _bind()
mcp = FastMCP("novaops-final", host=_host, port=_port)


def _span(stack: ExitStack, caller: CallerContext, name: str, payload: dict):
    """A server-side span inside the agent's trace, with the same session, user and
    request_id. The trace id and parent id come in `_meta` with the caller context;
    without them the server records nothing rather than starting stray traces."""
    if not caller.trace_id:
        return None
    trace_context = {"trace_id": caller.trace_id}
    if caller.parent_observation_id:
        trace_context["parent_span_id"] = caller.parent_observation_id
    stack.enter_context(propagate_attributes(
        session_id=caller.session_id, user_id=caller.caller_id,
        metadata={"request_id": caller.request_id, "user_group": caller.user_group}))
    return stack.enter_context(get_client().start_as_current_observation(
        as_type="span", name=f"server:{name}", input=payload, trace_context=trace_context))


def _serve(ctx: Context, name: str, payload: dict, work: Callable[[CallerContext], dict]) -> dict:
    """Every tool runs through here: caller context first (refused if missing), then a
    span, then the work. A ToolInputError becomes an is_error result for the model; any
    other exception is also returned as is_error by FastMCP, so the server stays up."""
    caller = from_request(ctx)
    with ExitStack() as stack:
        span = _span(stack, caller, name, payload)
        result = work(caller)
        if span is not None:
            span.update(output=result)
    return result


def db_label(table: str, row_id: str | None = None) -> str:
    """The citation label of one database row, e.g. [db:onboarding_tasks#OT005], or of
    a whole result, [db:assets]."""
    return f"[db:{table}#{row_id}]" if row_id else f"[db:{table}]"


def _rows(rows: list[dict], empty: str, cite: tuple[str, str] | None = None, **extra) -> dict:
    """Empty and non-empty results look different on purpose: 'no rows' is an answer.
    With `cite` = (table, id column), every row gets the `citation` label an answer
    cites for a fact taken from it, the same way document chunks carry theirs, and the
    result gets `result_citation` for a fact about the result as a whole (what is
    absent, how many there are)."""
    result_label = {}
    if cite:
        table, key = cite
        rows = [{**row, "citation": db_label(table, row[key])} for row in rows]
        result_label = {"result_citation": db_label(table)}
    result = {"count": len(rows), "rows": rows, **result_label, **extra}
    if not rows:
        result["note"] = empty
    return result


def _db(work: Callable) -> Callable[[CallerContext], dict]:
    def run(caller: CallerContext) -> dict:
        with db.session() as conn:
            return work(conn, caller)
    return run


# Documents ------------------------------------------------------------------------

def _policies_dir() -> Path:
    return config.documents_dir() / "policies"


def _not_a_policy(stem: str) -> str:
    """S9 turn 1: the model asked get_policy for titles-for-support (a handbook page)
    and read "no policy" as "the page does not exist". Say where the file lives, but
    only for collections every caller may search, so the message reveals nothing."""
    for corpus, tool in (("handbook", "search_hr_documents"), ("it_kb", "search_knowledge_base"),
                         ("internal_memos", "search_knowledge_base"),
                         ("contracts", "search_knowledge_base")):
        if (config.documents_dir() / corpus / f"{stem}.md").is_file():
            return (f"{stem}.md is a {corpus} document, not a policy. Its sections come from "
                    f"{tool}; use the chunks it already returned, with their citation labels.")
    return f"No policy named {stem!r}. Call list_policies for valid names."


@mcp.tool()
def list_policies(ctx: Context) -> dict:
    """List every NovaOps company policy: its name, effective date and citation label.

    Call this to find which policy covers a topic, then get_policy for its text.
    """
    def work(_caller):
        found = []
        for path in sorted(_policies_dir().glob("*.md")):
            doc = parse_document(path, "policies")
            found.append({"name": path.stem, "title": doc.title, "source": doc.source,
                          "last_updated": doc.last_updated,
                          "citation": citation_label(doc.source, doc.title)})
        return _rows(found, "No policies found.")
    return _serve(ctx, "list_policies", {}, work)


@mcp.tool()
def get_policy(name: str, ctx: Context) -> dict:
    """Return one NovaOps policy, split into sections, each with its citation label.

    Cite a section with its `citation` value, e.g. [equipment_policy.md#Personal laptops].
    A policy can be overridden by a newer internal memo; check last_updated and search
    for memos when a threshold or rule matters.

    Args:
        name: A policy name from list_policies, e.g. 'equipment_policy'.
    """
    def work(_caller):
        stem = name.removesuffix('.md')
        path = _policies_dir() / f"{stem}.md"
        if not path.is_file():
            raise ToolInputError(_not_a_policy(stem))
        doc = parse_document(path, "policies")
        sections = [{"section": s, "citation": citation_label(doc.source, s), "text": t}
                    for s, t in split_sections(doc)]
        return {"name": path.stem, "title": doc.title, "source": doc.source,
                "doc_type": doc.doc_type, "last_updated": doc.last_updated,
                "sections": sections}
    return _serve(ctx, "get_policy", {"name": name}, work)


@mcp.tool()
def search_knowledge_base(query: str, ctx: Context) -> dict:
    """Search the IT knowledge base, company policies, internal memos and vendor
    contracts.

    Use this for IT how-to and troubleshooting (VPN, MFA, password reset, laptops,
    licenses), for what a policy or memo says, and for contract terms (seat limits,
    renewal dates, annual cost). Returns up to five chunks, most
    relevant first, each with a `citation` label like [vpn_access.md#Resolution] that
    the answer must cite. Memos can be newer than the policy they change: compare
    `last_updated` and `doc_type`; a `precedence_note` says which one is newer.

    Args:
        query: A natural-language description of the problem or question.
    """
    def work(caller):
        return retrieval.search(query, retrieval.KB_CORPORA, caller.caller_id)
    return _serve(ctx, "search_knowledge_base", {"query": query}, work)


@mcp.tool()
def search_hr_documents(query: str, ctx: Context) -> dict:
    """Search HR documents: the employee handbook, employment documents (offer letters,
    addenda, access profiles), policies, internal memos, and, for managers only, the
    manager playbook.

    Use this for onboarding, benefits, what was agreed for a specific person, or a memo
    that changed a rule. Returns up to five chunks, most relevant first, each with a
    `citation` label that the answer must cite. Employment chunks carry
    `subject_employee_id`, the person the document is about; a caller only gets the
    employment documents they may read. When `access_note` is present, the manager
    playbook was not searched for this caller: say so if they asked for it.

    Args:
        query: A natural-language description of what you need to find.
    """
    def work(caller):
        return retrieval.search(query, retrieval.HR_CORPORA, caller.caller_id)
    return _serve(ctx, "search_hr_documents", {"query": query}, work)


# People ---------------------------------------------------------------------------

@mcp.tool()
def get_employee(query: str, ctx: Context) -> dict:
    """Look up NovaOps employees by employee id, email or (partial) name.

    Returns every match, each with a `citation` label such as [db:employees#E001] to
    cite for facts from that row. Several employees share a first name, so check the
    id before acting on a name. A misspelt full name ('rrachel stein') returns the
    employee it is a typo of, with `match: similar`; a different name returns nothing. `systems_allowed_by_role` is what the role entitles someone to,
    not what they have been given.

    Args:
        query: An employee id (e.g. 'E001'), an email, or a name.
    """
    def work(conn, _caller):
        found = db.get_employee(conn, query)
        if found:
            return _rows(found, "", ("employees", "employee_id"))
        close = db.similar_employees(conn, query)
        note = (f"No name contains {query!r} exactly; these employees match words of it. "
                "Check the name and id before acting on one." if close else "")
        return _rows(close, f"No employee matches {query!r}.", ("employees", "employee_id"),
                     **({"match": "similar", "match_note": note} if close else {}))
    return _serve(ctx, "get_employee", {"query": query}, _db(work))


@mcp.tool()
def get_team(manager_id: str, ctx: Context) -> dict:
    """List the direct reports of one manager (employees whose manager_id is this id).

    Use this for questions about "my team" or someone's reports. The manager link in
    the database can be stale (see the audit log and memos).

    Args:
        manager_id: The manager's employee id, e.g. 'E018'.
    """
    return _serve(ctx, "get_team", {"manager_id": manager_id}, _db(
        lambda conn, _c: _rows(db.get_team(conn, manager_id),
                               f"{manager_id} has no direct reports in the database.",
                               ("employees", "employee_id"))))


# IT and HR records ------------------------------------------------------------------

@mcp.tool()
def check_software_subscription(software: str, ctx: Context) -> dict:
    """Check a SaaS subscription: seat limit, seats in use, cost and renewal date.

    Use this when someone cannot get a license (e.g. "Webex says I'm not licensed") to
    see whether the subscription is full. Seat counts are organizational: an over-limit
    subscription blocks new seats for everyone, not one person.

    The database does not record which employee holds a seat. Every result has
    `seat_assignments_recorded: false`; say so rather than inferring seat holders from
    `systems_allowed_by_role`, which is entitlement by role only.

    Args:
        software: The software or vendor name, e.g. 'Webex' or 'Salesforce'.
    """
    return _serve(ctx, "check_software_subscription", {"software": software}, _db(
        lambda conn, _c: _rows(db.check_software_subscription(conn, software, config.today()),
                               f"No subscription matches {software!r}.",
                               ("software_subscriptions", "subscription_id"))))


@mcp.tool()
def check_asset_inventory(ctx: Context, asset_type: str = "", location: str = "") -> dict:
    """Check hardware inventory: what equipment exists, its status and location.

    `status: available` means it can be assigned now. Both arguments are optional
    substring filters; omit them to list everything. For a new joiner, also check
    list_onboarding_tasks: the checklist says when equipment will be prepared.

    Args:
        asset_type: e.g. 'Laptop' or 'Monitor'.
        location: e.g. 'Israel'.
    """
    return _serve(ctx, "check_asset_inventory", {"asset_type": asset_type, "location": location},
                  _db(lambda conn, _c: _rows(db.check_asset_inventory(conn, asset_type, location),
                                             "No assets match these filters.",
                                             ("assets", "asset_id"))))


@mcp.tool()
def list_employee_tickets(employee_id: str, ctx: Context, status: str = "") -> dict:
    """List the IT tickets raised by or for one employee, newest first.

    Check this before suggesting a new ticket: an open ticket for the same problem
    should be reused, not duplicated.

    Args:
        employee_id: The employee id, e.g. 'E010'.
        status: Optional status filter, e.g. 'open'.
    """
    return _serve(ctx, "list_employee_tickets", {"employee_id": employee_id, "status": status},
                  _db(lambda conn, _c: _rows(db.list_employee_tickets(conn, employee_id, status),
                                             f"{employee_id} has no tickets"
                                             f"{' with status ' + status if status else ''}.",
                                             ("tickets", "ticket_id"))))


@mcp.tool()
def list_onboarding_tasks(employee_id: str, ctx: Context) -> dict:
    """List one employee's onboarding checklist tasks, earliest due date first.

    The checklist is the system of record: a task is done only when this says so.
    Each task has a `citation` label such as [db:onboarding_tasks#OT005].
    `blocked` tasks need their blocker explained; `overdue` is computed against today
    (2026-07-01).

    Args:
        employee_id: The employee id, e.g. 'E001'.
    """
    return _serve(ctx, "list_onboarding_tasks", {"employee_id": employee_id}, _db(
        lambda conn, _c: _rows(db.list_onboarding_tasks(conn, employee_id, config.today()),
                               f"{employee_id} has no onboarding tasks.",
                               ("onboarding_tasks", "task_id"))))


@mcp.tool()
def get_access_requests(ctx: Context, employee_id: str = "", access_request_id: str = "") -> dict:
    """Read existing access requests, by employee or by request id.

    Check this before proposing a new access request: an open request for the same
    employee and system (status pending_approval, blocked or planned) is reused, and
    its decision_reason often explains the blocker.

    Args:
        employee_id: e.g. 'E010'. Give this or access_request_id.
        access_request_id: e.g. 'AR001'.
    """
    return _serve(ctx, "get_access_requests",
                  {"employee_id": employee_id, "access_request_id": access_request_id},
                  _db(lambda conn, _c: _rows(db.get_access_requests(conn, employee_id, access_request_id),
                                             "No access requests match.",
                                             ("access_requests", "request_id"))))


@mcp.tool()
def get_approvals(ctx: Context, record_id: str = "", employee_id: str = "") -> dict:
    """Read approval rows: who must approve a request, and whether they have.

    Status `needed` or `pending` means no decision yet; only `approved` lets a write go
    ahead, and only once. Use this to explain what a request is waiting on.

    Args:
        record_id: The record approved, e.g. access request 'AR001'. Give this or employee_id.
        employee_id: e.g. 'E010', for every approval about that employee's requests.
    """
    return _serve(ctx, "get_approvals", {"record_id": record_id, "employee_id": employee_id},
                  _db(lambda conn, _c: _rows(db.get_approvals(conn, record_id, employee_id),
                                             "No approvals match.",
                                             ("approvals", "approval_id"))))


# Approval request and gated writes --------------------------------------------------

def _cited(row: dict, table: str, key: str) -> dict:
    """A write returns the row it wrote or reused, labelled like a read row, so the
    actions an AccessDecision reports cite the row they touched."""
    return {**row, "citation": db_label(table, row[key])}


@mcp.tool()
def request_approval(request_type: str, record_id: str, approver_id: str, ctx: Context,
                     employee_id: str = "", software: str = "") -> dict:
    """Ask a named approver to approve a request. Creates an approval row with status
    `pending`, or returns the open one for the same request (reused: true).

    It does not approve anything and does not file the request. The run that calls it
    ends pending until the approver decides. `this_request: true` means the row is
    bound to the calling workflow request, the only one its approval can release.

    Args:
        request_type: 'access_request', 'subscription_expansion' or 'ticket'.
        record_id: The record the approval is for, e.g. access request 'AR001'.
        approver_id: Employee id of the approver, e.g. 'E006'.
        employee_id: Needed only when record_id is not an existing record.
        software: The system, needed only when record_id is not an existing record.
    """
    args = {"request_type": request_type, "record_id": record_id, "approver_id": approver_id,
            "employee_id": employee_id, "software": software}
    return _serve(ctx, "request_approval", args, _db(
        lambda conn, caller: _cited(writes.request_approval(
            conn, caller, request_type, record_id, approver_id, employee_id, software),
            "approvals", "approval_id")))


@mcp.tool()
def create_access_request(employee_id: str, software: str, business_justification: str,
                          ctx: Context, access_level: str = "standard_user") -> dict:
    """File an access request with status pending_approval. It never grants access.

    WRITES to the database, and only when an approver has approved this exact action
    for this employee and system; otherwise it is refused and nothing is written. If an
    open request already exists it is returned with reused: true.

    Args:
        employee_id: Who the access is for, e.g. 'E010'.
        software: The system, e.g. 'Webex'.
        business_justification: A short reason.
        access_level: Defaults to 'standard_user'.
    """
    args = {"employee_id": employee_id, "software": software,
            "business_justification": business_justification, "access_level": access_level}
    return _serve(ctx, "create_access_request", args, _db(
        lambda conn, caller: _cited(writes.create_access_request(
            conn, caller, employee_id, software, business_justification, access_level),
            "access_requests", "request_id")))


@mcp.tool()
def create_ticket(employee_id: str, subject: str, description: str, category: str,
                  ctx: Context, software: str = "", priority: str = "medium") -> dict:
    """Open an IT ticket with status open.

    WRITES to the database, and only against an approved, unused approval for this
    employee and system; otherwise it is refused and nothing is written. An open ticket
    for the same employee and system is returned with reused: true instead of a copy.

    Args:
        employee_id: Who the ticket is for, e.g. 'E010'.
        subject: One line.
        description: What is wrong.
        category: e.g. 'Access', 'Hardware', 'Security'.
        software: The related system, e.g. 'Webex'. Optional.
        priority: 'low', 'medium' or 'high'.
    """
    args = {"employee_id": employee_id, "subject": subject, "description": description,
            "category": category, "software": software, "priority": priority}
    return _serve(ctx, "create_ticket", args, _db(
        lambda conn, caller: _cited(writes.create_ticket(
            conn, caller, employee_id, subject, description, category, software, priority),
            "tickets", "ticket_id")))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    db.connect().close()  # fail at startup, not on the first call, if reseed never ran
    print(f"NovaOps MCP server on {config.mcp_server_url()} (index {config.opensearch_index()})")
    try:
        mcp.run(transport="streamable-http")
    finally:
        get_client().flush()
