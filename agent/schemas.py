"""The typed concepts of the Maya workflow (Lesson 10 homework, milestone 3).

- `CallerContext`: who is calling. Built by run_turn() from its arguments, never from
  message text, and checked here before anything runs.
- `EvidenceCitation`: one citation label, a document section or a database row.
- `OnboardingChecklist`: produced by a forced `submit_checklist` call and validated
  locally (agent/records.py); every item cites evidence, every blocked item says why.
- `WebexHandoff` and `PendingAccessResult`: what crosses the `WebexPort` boundary
  (agent/ports.py). Application code builds the handoff; the model never does.

The context plan is `TurnPlan` in agent/policy.py, which names the subject employee
and system a turn is about.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# A document section, [equipment_policy.md#Personal laptops], a database row,
# [db:onboarding_tasks#OT005], or a whole database result, [db:assets], for a fact no
# single row carries ("no headset is in stock in Israel"). The MCP tools return them in
# `citation` and `result_citation` fields.
LABEL = re.compile(r"\[(?:[^\[\]#]+\.md#[^\[\]]+|db:[a-z_]+(?:#[A-Za-z0-9_-]+)?)\]")
DB_LABEL = re.compile(r"^\[db:([a-z_]+)(?:#([A-Za-z0-9_-]+))?\]$")


class CallerContext(BaseModel):
    caller_id: str = Field(pattern=r"^E\d{3}$")
    user_group: str = Field(pattern=r"^UG_[A-Z]+$")


class EvidenceCitation(BaseModel):
    label: str = Field(description="A citation label exactly as a tool returned it.")

    @field_validator("label")
    @classmethod
    def _is_label(cls, value: str) -> str:
        value = value.strip()
        if not LABEL.fullmatch(value):
            raise ValueError(f"{value!r} is not a citation label like "
                             "[source.md#Section] or [db:table#ID]")
        return value

    @property
    def kind(self) -> Literal["document", "db"]:
        return "db" if DB_LABEL.match(self.label) else "document"


ItemStatus = Literal["done", "planned", "pending", "blocked", "unresolved"]


class ChecklistItem(BaseModel):
    item: str = Field(description="What has to happen, e.g. 'Webex standard user license'.")
    task_id: str | None = Field(default=None, description="The onboarding task id, e.g. OT005.")
    status: ItemStatus = Field(description="done, planned, pending, blocked, or unresolved "
                                           "when the evidence is missing or conflicting.")
    owner: str | None = Field(default=None, description="Owning group, e.g. IT or HR.")
    due_date: str | None = None
    blocked_reason: str | None = Field(default=None,
                                       description="Required when status is blocked.")
    citations: list[EvidenceCitation] = Field(
        min_length=1, description="At least one label of the evidence for this item.")

    @model_validator(mode="after")
    def _blocked_has_reason(self):
        if self.status == "blocked" and not (self.blocked_reason or "").strip():
            raise ValueError(f"item {self.item!r} is blocked but gives no blocked_reason")
        return self


class OnboardingChecklist(BaseModel):
    employee_id: str = Field(pattern=r"^E\d{3}$")
    employee_name: str
    role: str | None = None
    start_date: str | None = None
    items: list[ChecklistItem] = Field(min_length=1)
    overall_status: Literal["on_track", "blocked", "unresolved"] = Field(
        description="blocked if any item is blocked; unresolved if any item is unresolved "
                    "and none is blocked; otherwise on_track.")

    @model_validator(mode="after")
    def _overall_matches_items(self):
        statuses = {i.status for i in self.items}
        expected = ("blocked" if "blocked" in statuses
                    else "unresolved" if "unresolved" in statuses else "on_track")
        if self.overall_status != expected:
            raise ValueError(f"overall_status is {self.overall_status!r} but the items make "
                             f"it {expected!r}")
        return self

    def labels(self) -> set[str]:
        return {c.label for i in self.items for c in i.citations}


class WebexHandoff(BaseModel):
    """The request Maya sends to the separate Webex workflow. Built by code."""

    workflow: Literal["webex_access"] = "webex_access"
    caller_id: str
    subject_employee_id: str
    system_id: str
    software: str
    justification: str
    source_thread_id: str
    request_id: str
    idempotency_key: str


class PendingAccessResult(BaseModel):
    """What the Webex side answers. It is always pending: Maya cannot be told that
    access was granted, because filing and granting happen only on the Webex side."""

    status: Literal["pending"] = "pending"
    approval_id: str | None = None
    approver_id: str | None = None
    reused: bool = False
