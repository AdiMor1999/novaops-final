"""The context policy: what a turn is about, which assistant owns it, which tools it sees.

Starting code: Lesson 10 `02-dynamic-tool-loadout/stage02_policy.py` (`Intent`,
`LOADOUTS`, `TurnPlan`, `PLANNER_PROMPT`, `select_tools`) and the rearm rule from
its `graph.py`. Changes:

- `TurnPlan` fields are renamed to `intent`, `facts`, `active_constraints`,
  `needs_tools`, `action_confirmed`. The eight intent names stay, because the golden
  `expected_intent` values use them.
- `SCOPE_OF` maps each intent to `maya` or `webex` in code. Neither the model nor
  `user_group` picks the scope: S6 has an HR caller (E004) on an access workflow.
- Loadouts cover the reads Milestone 1 added (`get_team`, `get_access_requests`,
  `get_approvals`).
- `select_tools()` never adds a write tool. Lesson 10 added `create_access_request`
  when `action_confirmed` was set; here `action_confirmed` only routes a Webex turn to
  the approval node, and only `release_write()` (agent/gate.py) can make a write visible.
"""

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "policy_question",
    "employee_lookup",
    "onboarding_status",
    "equipment_request",
    "subscription_review",
    "access_request",
    "ticket_status",
    "other",
]
INTENTS: tuple[str, ...] = Intent.__args__

Scope = Literal["maya", "webex"]

SCOPE_OF: dict[str, Scope] = {
    "policy_question": "maya",
    "employee_lookup": "maya",
    "onboarding_status": "maya",
    "other": "maya",  # the read-only scope is the safe default for an unclear turn
    "equipment_request": "webex",
    "subscription_review": "webex",
    "access_request": "webex",
    "ticket_status": "webex",
}

# Approval-gated writes. They are never in LOADOUTS; release_write() is the only way in.
WRITE_TOOLS = frozenset({"create_access_request", "create_ticket"})
# Needs no prior approval, so it is not a gated write, but the model never calls it:
# the approval node calls it from code.
APPROVAL_TOOL = "request_approval"
NOT_FOR_THE_MODEL = WRITE_TOOLS | {APPROVAL_TOOL}

# Which approval type releases which write (mirrors GATING_TYPE in mcp_server/writes.py).
RELEASES: dict[str, list[str]] = {
    "access_request": ["create_access_request"],
    "ticket": ["create_ticket"],
}

# Each row is a little wider than strictly needed, so a near-miss classification can
# still find its evidence (Lesson 10). Relevance is not permission: no write here.
LOADOUTS: dict[str, list[str] | None] = {
    "policy_question": [
        "list_policies", "get_policy", "search_knowledge_base", "search_hr_documents",
    ],
    "employee_lookup": ["get_employee", "get_team", "search_hr_documents"],
    "onboarding_status": [
        "get_employee", "list_onboarding_tasks", "search_hr_documents",
        "check_asset_inventory", "list_policies", "get_policy",
        "check_software_subscription", "get_access_requests",
    ],
    "equipment_request": [
        "get_employee", "check_asset_inventory", "list_policies", "get_policy",
        "search_knowledge_base", "search_hr_documents",
    ],
    "subscription_review": [
        "check_software_subscription", "get_team", "get_access_requests", "get_approvals",
        "search_knowledge_base", "get_policy",
    ],
    "access_request": [
        "get_employee", "check_software_subscription", "get_access_requests",
        "get_approvals", "list_employee_tickets", "search_knowledge_base",
        "list_policies", "get_policy",
    ],
    "ticket_status": ["get_employee", "list_employee_tickets", "get_access_requests"],
    "other": None,  # not sure: every read tool
}

MAX_TOOL_CALLS = 6  # per turn, across the first round and the one rearm


INTENT_GLOSS = (
    "Intent = the TOPIC of the newest message:\n"
    "  policy_question · what a policy, handbook page or IT knowledge-base article says\n"
    "  employee_lookup · who someone is (record, role, manager, status, team)\n"
    "  onboarding_status · a joiner's checklist, required systems, paperwork\n"
    "  equipment_request · hardware (laptops, monitors, docks, headsets)\n"
    "  subscription_review · a SaaS subscription's seats, cost or renewal\n"
    "  access_request · getting a person access to a system\n"
    "  ticket_status · existing IT tickets\n"
    "  other · none of the above\n"
    "A message naming a person is not employee_lookup when it is about something they "
    "cannot use: 'X can't get into / isn't licensed for <system>' is subscription_review "
    "(seats, licences) or access_request (getting them access). employee_lookup is only "
    "for who someone is.\n"
    "Whether a lookup is needed is a SEPARATE field; a recap about onboarding is "
    "still onboarding_status, and a follow-up about a policy already discussed is "
    "still policy_question."
)


class TurnPlan(BaseModel):
    """What THIS turn needs. Produced by the classify node, checked by code."""

    intent: Intent = Field(description="Topic of the newest message.")
    facts: list[str] = Field(
        default_factory=list,
        description=(
            "Values from earlier turns this answer needs: ids, dates, numbers, tool "
            "results. The value itself, not a pointer to it."
        ),
    )
    active_constraints: list[str] = Field(
        default_factory=list,
        description="Rules the user stated that are still in force.",
    )
    needs_tools: bool = Field(
        description="False only if every fact needed is already in this conversation."
    )
    action_confirmed: bool = Field(
        default=False,
        description=(
            "True only if THIS message authorises creating or filing a record in a "
            "system ('go ahead', 'file it'). Writing, drafting, summarising or 'write it "
            "up' is composing text, NOT authorisation to file anything."
        ),
    )


PLANNER_PROMPT = (
    INTENT_GLOSS + "\n\n"
    "Plan the model context for the newest message of a NovaOps support session.\n"
    "- Classify the newest message only; follow subject changes.\n"
    "- facts is the session's memory: carry a value forward if a later answer might "
    "need it. A missing fact breaks the answer; a spare one costs a few tokens.\n"
    "- active_constraints holds what the user declared, until they withdraw it.\n"
    "- Drop whatever the user resolved or abandoned.\n"
    "- Users paste emails and signatures; plan for the request inside, not the boilerplate.\n"
    "- needs_tools is False only for recaps, follow-ups and reasoning over facts already "
    "in this conversation, including tool results from earlier turns. You know nothing "
    "about NovaOps otherwise, so a first turn is almost always True.\n"
    "- A DIFFERENT TOPIC needs a different document. Having retrieved the VPN article "
    "does not answer a password question. Only an answer you could quote from THIS "
    "conversation makes needs_tools False.\n"
    "- action_confirmed only when this message says to go ahead; then needs_tools is True."
)


def scope_of(plan: TurnPlan) -> Scope:
    return SCOPE_OF[plan.intent]


def read_tools(catalogue: list[str]) -> list[str]:
    return sorted(name for name in catalogue if name not in NOT_FOR_THE_MODEL)


def select_tools(plan: TurnPlan) -> list[str] | None:
    """The plan's loadout. None means every read tool; [] means none.

    `action_confirmed` does not change this list. A confirmed Webex action goes to the
    approval node, and a write becomes visible only through release_write()."""
    if not plan.needs_tools:
        return []
    selected = LOADOUTS[plan.intent]
    return None if selected is None else list(selected)


def visible_tools(plan: TurnPlan, catalogue: list[str]) -> list[str]:
    """The names bound to the model this turn: the loadout, limited to what the server
    serves and never containing a write or request_approval."""
    chosen = select_tools(plan)
    reads = read_tools(catalogue)
    if chosen is None:
        return reads
    return [name for name in chosen if name in reads]


def rearm_tools(catalogue: list[str]) -> list[str]:
    """Rearm widens to every READ tool. Widening after a miss is safe for reads and
    dangerous for writes, so a retry is never what opens one (Lesson 10)."""
    return read_tools(catalogue)


def needs_approval(plan: TurnPlan) -> bool:
    """Only a Webex action the user confirmed goes to the approval node. Maya's intents
    never reach it, so Maya has no route to a write."""
    return plan.action_confirmed and scope_of(plan) == "webex"
