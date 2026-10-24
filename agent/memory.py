"""Thread-scoped durable memory: distil old history into `ConversationMemory`.

Starting code: Lesson 10 `03-history-distillation/stage03_policy.py`, with its
thresholds (12 fresh messages or ~2,500 tokens) and `memory_is_safe` unchanged.
Changes: the token estimate is inlined (Lesson 10's `evals/tokens.py`, ~4 characters
per token), transcripts use agent/graph.py's renderer, and a distiller parse failure
is repaired once and then recorded instead of silently skipped.

Distillation changes what the planner and the answer model are SENT. It never deletes
a message from state, so a memory that loses something costs tokens next turn, not
evidence. A rejected memory keeps the old memory and the raw history.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

DISTIL_AFTER_MESSAGES = 12
DISTIL_AFTER_TOKENS = 2500
CHARS_PER_TOKEN = 4  # estimate only; Bedrock reports exact usage per call


class ConversationMemory(BaseModel):
    """Five named fields rather than one summary string, so "the Q3 freeze applies"
    has a specific place to live and a specific place to be checked."""

    current_intent: str = Field(description="What the session is about NOW.")
    important_facts: list[str] = Field(
        default_factory=list,
        description="Durable values later turns may need: ids, names, dates, numbers.",
    )
    active_constraints: list[str] = Field(
        default_factory=list,
        description=("Rules the user imposed that are STILL in force. Never drop one unless "
                     "the user withdrew it."),
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="Conclusions reached and actions taken, including things closed.",
    )
    unresolved_items: list[str] = Field(
        default_factory=list,
        description="Open questions and outstanding work. These must survive.",
    )


DISTIL_PROMPT = (
    "You maintain the durable memory of a long support conversation.\n"
    "You are given the memory so far and the conversation since it was last updated. "
    "Return the UPDATED memory.\n\n"
    "Rules:\n"
    "- Carry every still-relevant item forward. This memory replaces the raw messages, "
    "so anything you omit is gone.\n"
    "- Keep concrete values (E001, 2026-08-01, 42 of 40 seats), not descriptions of them.\n"
    "- NEVER silently drop an active constraint. Move it to decisions only if the user "
    "explicitly withdrew or satisfied it.\n"
    "- Move a topic the user closed into decisions, and drop its raw detail.\n"
    "- unresolved_items must keep anything still outstanding.\n"
    "- Be compact. This is a working memory, not minutes."
)


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def fresh_messages(state: dict) -> list:
    return state["messages"][state.get("distilled_upto", 0):]


def needs_distillation(state: dict, render) -> bool:
    """`render` turns messages into the transcript text the planner would see."""
    fresh = fresh_messages(state)
    return (len(fresh) >= DISTIL_AFTER_MESSAGES
            or estimate_tokens(render(fresh)) >= DISTIL_AFTER_TOKENS)


def memory_is_safe(old: dict, new: ConversationMemory) -> bool:
    """Reject empty memory or a result that silently drops active constraints."""
    if not (new.important_facts or new.decisions or new.unresolved_items):
        return False
    return len(new.active_constraints) >= len(old.get("active_constraints", []))


def render_memory(memory: dict | None) -> str:
    if not memory:
        return "(nothing distilled yet)"
    m = ConversationMemory(**memory)

    def bullets(items: list[str]) -> str:
        return "\n".join(f"  - {item}" for item in items) if items else "  - (none)"

    return (
        f"Session is about: {m.current_intent}\n"
        f"Established facts:\n{bullets(m.important_facts)}\n"
        f"Constraints in force:\n{bullets(m.active_constraints)}\n"
        f"Decisions made:\n{bullets(m.decisions)}\n"
        f"Still open:\n{bullets(m.unresolved_items)}"
    )


def distiller_messages(state: dict, render) -> list:
    prompt = (
        f"Memory so far:\n{render_memory(state.get('memory'))}\n\n"
        f"Conversation since then:\n{render(fresh_messages(state))}"
    )
    return [SystemMessage(DISTIL_PROMPT), HumanMessage(prompt)]
