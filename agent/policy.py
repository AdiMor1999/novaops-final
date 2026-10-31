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
- Task #4: `TurnPlan` is the context plan of the Lesson 10 homework. It names the
  subject employee and the system, which code verifies against records already in the
  thread (agent/handoff.py) before a confirmed action goes anywhere. Two loadout rows
  gained a tool the S2/S9 matrix showed missing (evals/tool_matrix.py):
  `subscription_review` gets `search_hr_documents` (S2 turn 9, the freeze memo) and
  `equipment_request` gets `list_onboarding_tasks` (S9 turn 5, OT003).
- Task #5: `TurnPlan.subject_name`, so code can look up a subject the thread has not
  seen yet (agent/handoff.py). `action_confirmed` also covers an employee asking for
  access for themselves (the Rachel scenario of spec §9), and only `access_request`
  reaches the approval node. `subscription_review` gets `list_onboarding_tasks`: a
  manager's seat question is answered from the team's blocked provisioning tasks,
  because seat holders are not recorded (spec §13).
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

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
        "get_employee", "check_asset_inventory", "list_onboarding_tasks", "list_policies",
        "get_policy", "search_knowledge_base", "search_hr_documents",
    ],
    "subscription_review": [
        "check_software_subscription", "get_team", "get_access_requests", "get_approvals",
        "list_onboarding_tasks", "search_knowledge_base", "search_hr_documents", "get_policy",
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
        description=(
            "False only if every fact needed is already in this conversation. True for "
            "any question about the current state (what is 'still' open, 'now', 'after "
            "that') and for any rule, threshold or approval no tool result here states."
        )
    )
    action_confirmed: bool = Field(
        default=False,
        description=(
            "True only if THIS message authorises creating or filing a record in a "
            "system ('go ahead', 'file it'), or is an employee asking to be given access "
            "to a system for themselves ('I need Webex access'). Writing, drafting, "
            "summarising or 'write it up' is composing text, NOT authorisation to file "
            "anything. A message that says not to file is never confirmed."
        ),
    )
    subject_employee_id: str | None = Field(
        default=None,
        description=(
            "Employee id (e.g. E001) of the person the newest message is about, taken "
            "from a record already in this conversation. The person, not the caller. "
            "Null if no record in the conversation identifies them."
        ),
    )
    subject_name: str | None = Field(
        default=None,
        description=(
            "Full name of the person the newest message is about, as the message writes "
            "it ('Rachel Stein'), when no record in the conversation gives their id. "
            "Null when the message is about the caller."
        ),
    )
    system: str | None = Field(
        default=None,
        description="The software system the newest message is about (e.g. Webex), or null.",
    )

    @field_validator("subject_employee_id", "subject_name", "system", mode="before")
    @classmethod
    def _null_words_are_null(cls, value):
        """Nova Lite writes the string "null" (S8 turn 4) where it means no value; code
        would otherwise look up an employee called 'null'."""
        if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "n/a"):
            return None
        return value


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
    "- needs_tools is True when the answer depends on a policy rule, a threshold, an "
    "approval requirement, or the CURRENT state of a checklist, seat count, ticket or "
    "inventory, unless a tool result in this conversation already states exactly that. "
    "A constraint the user stated ('the freeze applies') is not the rule: look up what "
    "the rule requires.\n"
    "- 'still', 'now', 'right now', 'after that', 'currently' ask for the current state: "
    "needs_tools "
    "is True.\n"
    "- A DIFFERENT TOPIC needs a different document. Having retrieved the VPN article "
    "does not answer a password question. Only an answer you could quote from THIS "
    "conversation makes needs_tools False.\n"
    "- action_confirmed only when this message says to go ahead, or an employee asks to "
    "be given access to a system for themselves; then needs_tools is True.\n"
    "- subject_employee_id is the id of the person the request is about (E001), copied "
    "from an employee record in the conversation, or the caller's own id when they ask "
    "about themselves; subject_name is their full name when no record gives the id; "
    "system names the software.\n"
    "- You may be given distilled session memory plus the raw conversation since it was "
    "written. The memory's topic is the session's history, not this turn.\n"
    "- Checklists, seat counts, tickets and inventory in memory are a SNAPSHOT; if the "
    "user asks what the state is NOW, needs_tools is True."
)


# Words that ask for the state of a record as it is now, not as an earlier turn saw it.
CURRENT_STATE = re.compile(r"\b(right now|currently|still|after that|now)\b", re.IGNORECASE)


# An employee asking for access for themselves (spec §9: "I joined Customer Success and
# need Webex access. Webex says my account is not licensed."), and wording that says not
# to file, which always wins (S6/S7/S8 turn 3: "draft ... do not file it").
SELF_ACCESS = re.compile(
    r"\b(i|i'm|i am)\b[^.?!\n]{0,40}\b(need|want|require|request|get|can't|cannot)\b"
    r"[^.?!\n]{0,40}\baccess\b|\bmy account is not licen[cs]ed\b", re.IGNORECASE)
NO_FILING = re.compile(
    r"\b(do not|don't|dont|not)\b[^.?!\n]{0,20}\b(file|submit|create|send|open)\b|"
    r"\bdraft\b|\btext only\b", re.IGNORECASE)


def self_access_request(plan: TurnPlan, request: str, caller_id: str | None) -> bool:
    """An access request the caller makes for themselves: first person, no other person
    named by the planner, no wording against filing."""
    if plan.intent != "access_request" or not caller_id:
        return False
    other = (plan.subject_employee_id not in (None, caller_id)) or bool(plan.subject_name)
    return not other and bool(SELF_ACCESS.search(request)) and not NO_FILING.search(request)


def checked_plan(plan: TurnPlan, has_history: bool, message: str = "",
                 caller_id: str | None = None) -> TurnPlan:
    """Code corrects two planner claims it can disprove. `needs_tools=False` means every
    fact is already in the conversation, which is false:

    - on a first turn, where there is no conversation (any topic but `other`, which
      covers greetings). Nova Lite returned False for S9 turn 1 on every try;
    - when the message asks for the current state ("what's on her checklist right
      now?", S2 turn 3). An earlier tool result is a snapshot, not the state now.
      Only the first paragraph is read: it is the user's own request, and a pasted
      thread after it ("baseline groups are still pending", S9 turn 2) is not.

    Task #5 adds a third: `action_confirmed=False` on an employee's own access request
    (`self_access_request`). Nova Lite left it False on the Rachel scenario although the
    field's description names that case. Setting it routes the turn to the approval
    request, which only reuses or creates a pending row; it opens no write."""
    request = message.strip().split("\n\n")[0]
    if not plan.action_confirmed and self_access_request(plan, request, caller_id):
        plan = plan.model_copy(update={"action_confirmed": True, "needs_tools": True,
                                       "subject_employee_id": caller_id})
    if plan.needs_tools or plan.intent == "other":
        return plan
    if not has_history or CURRENT_STATE.search(request):
        return plan.model_copy(update={"needs_tools": True})
    return plan


# Intents whose answer is about a person's onboarding: the checklist is the system of
# record for what is due and when (list_onboarding_tasks' docstring says so).
# Only equipment: onboarding_status turns reach the checklist on their own once the
# model knows who "I" is, and forcing it on S2 turn 2 (the offer letter) made the
# planner skip the fresh lookup S2 turn 3 needs.
CHECKLIST_INTENTS = frozenset({"equipment_request"})
CHECKLIST_TOOL = "list_onboarding_tasks"


def missing_checklist(plan: TurnPlan, called: list[str]) -> bool:
    """True when a turn about someone's onboarding or equipment looked things up but
    never read the checklist. The graph then runs its one refined round. Nova Lite
    answered S9 turn 5 ("am I getting a laptop before the 1st?") from inventory alone
    on four runs out of four, so the due date (OT003, 2026-07-26) never reached it."""
    return plan.needs_tools and plan.intent in CHECKLIST_INTENTS and CHECKLIST_TOOL not in called


def refine_note(who: str) -> str:
    return (f"Before answering, call {CHECKLIST_TOOL} for {who}: the onboarding checklist "
            "is the system of record for what is due and when. Then answer with both.")


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
    """Only a confirmed access request goes to the approval node. Maya's intents never
    reach it, so Maya has no route to a write; nor do other Webex intents (a confirmed
    equipment turn has no approval type to request)."""
    return plan.action_confirmed and plan.intent == "access_request"
