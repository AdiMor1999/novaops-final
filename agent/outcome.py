"""What a turn returns, and the code that decides its status and citations.

`TurnOutcome` is the one shape tests, the eval runner and the web chat read. `status`
comes from `turn_status()`, which looks at what the tools and the gate did, never at
the model's wording. `cited_labels()` is Milestone 1's check, now over every label any
tool returned in the conversation, so a follow-up answered from turn 1's policy text
can cite it without calling the tool again.
"""

import json
import re
from typing import Literal

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from agent.policy import TurnPlan

Status = Literal["completed", "blocked", "pending", "needs_human", "failed"]
STATUSES: tuple[str, ...] = Status.__args__

_LABEL = re.compile(r"\[[^\[\]#]+\.md#[^\[\]]+\]")


class TurnOutcome(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    status: Status
    request_id: str
    trace_id: str | None = None
    trace_url: str | None = None
    scope: str | None = None
    intent: str | None = None
    visible_tools: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    approval_id: str | None = None


def known_labels(messages: list) -> set[str]:
    """Every citation label a successful tool result carried."""
    labels: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.status != "error":
            text = message.content if isinstance(message.content, str) else json.dumps(message.content)
            labels.update(_LABEL.findall(text))
    return labels


def cited_labels(answer: str, known: set[str]) -> tuple[list[str], list[str]]:
    """(labels the answer cites that a tool returned, labels it invented)."""
    used = list(dict.fromkeys(_LABEL.findall(answer)))
    return [u for u in used if u in known], [u for u in used if u not in known]


def turn_status(plan: TurnPlan, gate: dict | None, called: list[str], errors: list[str],
                valid: list[str], invented: list[str], documents_this_turn: bool) -> str:
    """Set from the gate, the tool results and the citations.

    - blocked: the approver rejected the action.
    - needs_human: the gate stayed shut for another reason, the answer cites a label no
      tool returned, the plan needed a lookup and none ran, or documents came back and
      the answer cites none of them.
    - failed: every tool call this turn errored.
    - pending is set by run_turn() when the graph pauses on an approval.
    """
    if gate and not gate["open"]:
        return "blocked" if gate["state"] == "rejected" else "needs_human"
    if called and len(errors) >= len(called):
        return "failed"
    if invented:
        return "needs_human"
    if plan.needs_tools and not called:
        return "needs_human"
    if documents_this_turn and not valid:
        return "needs_human"
    return "completed"


def gate_text(gate: dict) -> str:
    """The answer when the gate stays shut. Written by code, so it cannot claim more
    than the approval row says."""
    return f"Nothing was filed. {gate['reason']}"


def pending_text(approval: dict) -> str:
    return (f"Approval {approval['approval_id']} is waiting on {approval['approver_id']}. "
            "Nothing is filed and no access is granted until they decide.")
