"""Typed records produced by a forced tool call and validated locally.

Lesson 3's pattern: give the model one tool and force it, so the answer arrives as tool
arguments rather than prose. Two tools: `submit_checklist`, whose schema is
`OnboardingChecklist` (Maya), and `submit_access_decision`, whose schema is the
`DecisionDraft` part of an `AccessDecision` (Webex, task #5). The arguments are
validated by our own Pydantic model plus one check the schema cannot express: every
cited label must be one a tool returned in this conversation. A failure goes through
`parse_with_repair` (agent/structured.py): one repair call, then `ParseFailed`.
"""

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ValidationError

from agent.schemas import DecisionDraft, OnboardingChecklist

CHECKLIST_TOOL = "submit_checklist"
CHECKLIST_DESCRIPTION = (
    "Submit the onboarding checklist. One item per onboarding task, each citing the "
    "label of the evidence it rests on; a blocked item says why."
)

CHECKLIST_PROMPT = (
    "Build the onboarding checklist for the employee in the tool results below and "
    "submit it with submit_checklist.\n"
    "- One item per onboarding task row, with its task_id, status, owner and due_date "
    "copied from the row. Use status 'done' for a completed task.\n"
    "- Every item cites at least one label copied exactly from a `citation` field "
    "below, e.g. [db:onboarding_tasks#OT005]. Never invent a label.\n"
    "- A blocked item gives blocked_reason from the evidence (for example the seat "
    "count of the subscription) and cites that evidence too.\n"
    "- If evidence for an item is missing or conflicting, use status 'unresolved'."
)


def forced_tool(schema: type[BaseModel], name: str, description: str) -> dict:
    """A JSON schema whose `title` becomes the forced tool's name. LangChain's
    conversion dereferences the nested models first, so the model sees one flat
    schema instead of `$defs` references."""
    params = convert_to_openai_tool(schema)["function"]["parameters"]
    return {"title": name, "description": description, **params}


def checklist_tool() -> dict:
    return forced_tool(OnboardingChecklist, CHECKLIST_TOOL, CHECKLIST_DESCRIPTION)


DECISION_TOOL = "submit_access_decision"
DECISION_DESCRIPTION = (
    "Submit what the evidence shows about this access case and what people must do "
    "next. Each fact cites the label of the evidence it rests on."
)

DECISION_PROMPT = (
    "You write the evidence part of an IT access decision. Read the request and the "
    "tool results below, then submit them with submit_access_decision.\n"
    "observed_facts, one sentence each, each citing labels copied exactly from a "
    "`citation` or `result_citation` field below. Cover what the evidence shows on:\n"
    "- whether the employee is eligible for the system, and why (role, department, "
    "systems_allowed_by_role, the access request's decision_reason);\n"
    "- the subscription: seats in use against the seat limit, and that a full "
    "subscription blocks new seats for everyone, not only this person;\n"
    "- the contract: its end or renewal date and how close it is to today;\n"
    "- the tickets and access requests that already exist for this problem;\n"
    "- the approvals: who must approve what, and their status;\n"
    "- what the knowledge base, policies and memos say about this case.\n"
    "recommended_next_steps: what named people or teams must do next. Nothing in "
    "them has happened yet.\n"
    "Never say access was granted or provisioned, that no approval is needed, that "
    "it is only a password reset, or that a new ticket or request was created: this "
    "workflow never grants access, and code reports every action separately. Do not "
    "invent a label or a fact."
)


def decision_tool() -> dict:
    return forced_tool(DecisionDraft, DECISION_TOOL, DECISION_DESCRIPTION)


class Validated:
    """Wraps a `with_structured_output(<tool dict>, include_raw=True)` runnable, whose
    `parsed` is a plain dict, and validates it against `schema` and the known labels.
    It returns the shape parse_with_repair expects: `parsed` is None on any failure and
    `parsing_error` says why, so the repair prompt can quote it."""

    def __init__(self, runnable, schema: type[OnboardingChecklist | DecisionDraft],
                 known_labels: set[str]):
        self.runnable = runnable
        self.schema = schema
        self.known = known_labels

    async def ainvoke(self, messages: list) -> dict:
        result = await self.runnable.ainvoke(messages)
        parsed = result.get("parsed")
        if parsed is None:
            return {**result, "parsed": None}
        try:
            record = self.schema.model_validate(parsed)
        except ValidationError as exc:
            return {**result, "parsed": None, "parsing_error": str(exc)}
        unknown = sorted(record.labels() - self.known)
        if unknown:
            return {**result, "parsed": None,
                    "parsing_error": f"labels no tool returned: {', '.join(unknown)}"}
        return {**result, "parsed": record}
