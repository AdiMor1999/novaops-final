"""The five rules added after live runs, each with an off switch and a firing record.

A heuristic is a rule that changes agent behaviour beyond the brief, added because a
live run failed without it. Each one can be switched off for an ablation run with
`NOVAOPS_HEURISTICS_OFF=<comma list>` (empty by default), and each firing is recorded,
so a results file can say how often a rule fires and what changes without it.

| Name | Where | What it does |
| --- | --- | --- |
| `needs_tools_override` | `agent/policy.py` (`checked_plan`) | sets `needs_tools` on a first turn or a "right now" request |
| `self_access_confirmation` | `agent/policy.py` (`checked_plan`) | an employee's own access request counts as confirmed |
| `title_chunk_demotion` | `mcp_server/retrieval.py` (`diverse`) | a header-only chunk gives way to a content chunk of its document |
| `citation_brace_repair` | `agent/schemas.py` (`ObservedFact`) | `{` and `}` in a decision label become brackets, packed labels are split |
| `fuzzy_name_lookup` | `mcp_server/server.py` (`get_employee`) | a misspelt full name returns the employee it is a typo of |

Firings land in a per-turn list that `run_turn()` opens (`collect()`), puts on the root
span's metadata and returns on `TurnOutcome.heuristics`. The two server-side rules run
in the MCP server process: the server adds their names to the tool result under
`SERVER_KEY`, and `agent/mcp_tools.py` moves them into this list and out of the result,
so the model never sees them.
"""

from contextlib import contextmanager
from contextvars import ContextVar

from agent import config

NAMES = ("needs_tools_override", "self_access_confirmation", "title_chunk_demotion",
         "citation_brace_repair", "fuzzy_name_lookup")
SERVER_KEY = "_heuristics"

_fired: ContextVar[list[str] | None] = ContextVar("novaops_heuristics_fired", default=None)


class UnknownHeuristic(ValueError):
    """NOVAOPS_HEURISTICS_OFF names a rule that does not exist (a typo would otherwise
    leave the rule on and the ablation would measure nothing)."""


def switched_off() -> frozenset[str]:
    names = frozenset(n.strip() for n in config.heuristics_off().split(",") if n.strip())
    if unknown := sorted(names - set(NAMES)):
        raise UnknownHeuristic(f"NOVAOPS_HEURISTICS_OFF names {unknown}; known: {NAMES}")
    return names


def enabled(name: str) -> bool:
    assert name in NAMES, name
    return name not in switched_off()


def fired(name: str) -> None:
    """Record one firing in the current turn. Outside a turn (a unit test calling the
    rule directly, or the server process) there is no list and nothing is recorded."""
    assert name in NAMES, name
    found = _fired.get()
    if found is not None:
        found.append(name)


@contextmanager
def collect():
    """Open the turn's firing list. The list object is shared with the tasks LangGraph
    starts for the nodes, because they copy the context and the copy points at it."""
    found: list[str] = []
    token = _fired.set(found)
    try:
        yield found
    finally:
        _fired.reset(token)
