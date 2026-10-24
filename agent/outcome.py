"""What a turn returns, and the code that decides its status and citations.

`TurnOutcome` is the one shape tests, the eval runner and the web chat read. `status`
comes from `turn_status()`, which looks at what the tools and the gate did, never at
the model's wording. `cited_labels()` is Milestone 1's check, now over every label any
tool returned in the conversation, so a follow-up answered from turn 1's policy text
can cite it without calling the tool again.

Task #4 adds database labels ([db:onboarding_tasks#OT005]) next to document labels,
`uncited_sentences()` (the strict per-sentence check, reused by the task #6 scorer),
the documents that reached the model this turn, and the checklist and handoff fields.
"""

import json
import re
from typing import Literal

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from agent.policy import TurnPlan
from agent.schemas import LABEL, OnboardingChecklist

Status = Literal["completed", "blocked", "pending", "needs_human", "failed"]
STATUSES: tuple[str, ...] = Status.__args__

_LABEL = LABEL

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_LABELS = re.compile(r"^((?:\[[^\[\]]+\]\s*)+)(.*)$", re.DOTALL)
# Sentences that report missing or restricted evidence state no fact to cite: "the
# manager playbook is not searchable for this caller", "I could not find ...".
_NO_CLAIM = re.compile(
    r"\b(not (searchable|available|recorded|listed|found|included|covered|in the)|"
    r"no (record|information|evidence|document|mention)|could not|couldn't|cannot|can't|"
    r"unable to|(do|does|did) not (have|say|state|mention|cover|show)|"
    r"(don't|doesn't|didn't) (have|say|state|mention|cover|show))\b", re.IGNORECASE)
_NOT_A_SENTENCE = re.compile(r"^(if you|let me know|would you|do you|please|note:)",
                             re.IGNORECASE)
# A fact the user told us is attributed to them, not cited: "as you said, the Q3
# freeze applies". It has no tool result to point at.
_USER_STATED = re.compile(r"\b(you said|you mentioned|you told me|you noted|you gave|"
                          r"you stated|as you (said|mentioned|noted))\b", re.IGNORECASE)
# Anything in square brackets that reads like a label attempt: "[constraint]",
# "[db:assets]". Markdown links ("[text](url)") and checkboxes are not.
_BRACKETED = re.compile(r"\[[^\[\]\n]*[A-Za-z][^\[\]\n]{1,80}\](?!\()")
MIN_CLAIM_WORDS = 5


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
    sources: list[str] = Field(default_factory=list)  # documents that reached the model
    approval_id: str | None = None
    checklist: OnboardingChecklist | None = None
    handoff: dict | None = None  # {"handoff": WebexHandoff, "result": PendingAccessResult}


def tool_results(messages: list, name: str | None = None) -> list[dict]:
    """Decoded results of successful tool calls, optionally of one tool."""
    out = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.status != "error" \
                and (name is None or message.name == name):
            text = message.content if isinstance(message.content, str) else json.dumps(message.content)
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                out.append(decoded)
    return out


def retrieved_sources(messages: list) -> list[str]:
    """`corpus/source` of every document a tool put in front of the model, in order.
    The golden `expected_sources` / `forbidden_sources` are checked against this."""
    found: list[str] = []
    for result in tool_results(messages):
        for chunk in result.get("chunks", []):
            if chunk.get("doc_id"):
                found.append(chunk["doc_id"])
        if result.get("sections") and result.get("source"):
            found.append(f"policies/{result['source']}")
    return list(dict.fromkeys(found))


def sentences(text: str) -> list[str]:
    """Split after . ! ? or a line break. Labels written right after the full stop
    ("... blocked. [db:onboarding_tasks#OT005] Next ...") belong to the sentence
    before them, so they are moved back onto it."""
    out: list[str] = []
    for part in (p.strip(" -*•\t") for p in _SENTENCE_END.split(text)):
        lead = _LEADING_LABELS.match(part)
        if lead and out:
            out[-1] = f"{out[-1]} {lead.group(1).strip()}"
            part = lead.group(2).strip(" -*•\t")
        if part:
            out.append(part)
    return out


def needs_citation(sentence: str) -> bool:
    """A sentence states a fact unless it is a question, a heading, an offer, very
    short, or a report that evidence is missing or restricted."""
    bare = _LABEL.sub("", sentence).strip()
    if len(bare.split()) < MIN_CLAIM_WORDS or bare.endswith(("?", ":")):
        return False
    return not (_NOT_A_SENTENCE.match(bare) or _NO_CLAIM.search(bare)
                or _USER_STATED.search(bare))


def uncited_sentences(answer: str) -> list[str]:
    """Fact-stating sentences that carry no citation label. Applied only on turns with
    tool results: a recall turn answered from what the user said (S2 turn 7) has
    nothing to cite."""
    return [s for s in sentences(answer) if needs_citation(s) and not _LABEL.search(s)]


def known_labels(messages: list) -> set[str]:
    """Every citation label a successful tool result carried."""
    labels: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage) and message.status != "error":
            text = message.content if isinstance(message.content, str) else json.dumps(message.content)
            labels.update(_LABEL.findall(text))
    return labels


def cited_labels(answer: str, known: set[str]) -> tuple[list[str], list[str]]:
    """(labels the answer cites that a tool returned, labels it invented). Invented
    includes made-up bracket labels such as [constraint] that match no label format."""
    used = list(dict.fromkeys(_BRACKETED.findall(answer)))
    return [u for u in used if u in known], [u for u in used if u not in known]


def turn_status(plan: TurnPlan, gate: dict | None, called: list[str], errors: list[str],
                valid: list[str], invented: list[str], documents_this_turn: bool,
                uncited: list[str] | None = None) -> str:
    """Set from the gate, the tool results and the citations.

    - blocked: the approver rejected the action.
    - needs_human: the gate stayed shut for another reason, the answer cites a label no
      tool returned, the plan needed a lookup and none ran, or documents came back and
      the answer cites none of them, or sentences stayed uncited after the one repair.
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
    if uncited:
        return "needs_human"
    return "completed"


def gate_text(gate: dict) -> str:
    """The answer when the gate stays shut. Written by code, so it cannot claim more
    than the approval row says."""
    return f"Nothing was filed. {gate['reason']}"


def pending_text(approval: dict) -> str:
    return (f"Approval {approval['approval_id']} is waiting on {approval['approver_id']}. "
            "Nothing is filed and no access is granted until they decide.")


def handoff_text(handoff: dict, result: dict) -> str:
    """The answer of a handoff turn. Written by code from the port's result, so it
    says pending and cannot claim the request was filed or access granted."""
    approval = (f" Approval {result['approval_id']} is waiting on {result['approver_id']}."
                if result.get("approval_id") else "")
    return (f"I handed the {handoff['software']} access request for "
            f"{handoff['subject_employee_id']} to the Webex access workflow, and it is "
            f"pending.{approval} Nothing is filed and no access is granted until the "
            "approver decides.")
