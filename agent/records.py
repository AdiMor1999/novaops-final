"""Typed records produced by a forced tool call and validated locally.

Lesson 3's pattern: give the model one tool and force it, so the answer arrives as tool
arguments rather than prose. Here the tool is `submit_checklist`, whose schema is
`OnboardingChecklist`. The arguments are validated by our own Pydantic model plus one
check the schema cannot express: every cited label must be one a tool returned in
this conversation. A failure goes through `parse_with_repair` (agent/structured.py):
one repair call, then `ParseFailed`.
"""

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ValidationError

from agent.schemas import OnboardingChecklist

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


class Validated:
    """Wraps a `with_structured_output(<tool dict>, include_raw=True)` runnable, whose
    `parsed` is a plain dict, and validates it against `schema` and the known labels.
    It returns the shape parse_with_repair expects: `parsed` is None on any failure and
    `parsing_error` says why, so the repair prompt can quote it."""

    def __init__(self, runnable, schema: type[OnboardingChecklist], known_labels: set[str]):
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
