"""The conversation checks the golden dataset leaves to us (its readme: "NOT INCLUDED,
AND YOURS TO WRITE"), as pure functions over one turn's outcome.

- `constraint_carry()`: S4's turn-1 rule ("any SaaS expansion above 7,500 a year needs
  Finance sign-off") still decides turns 5, 8 and 9. Turn 5 names Webex and Figma and
  not Miro as needing Finance; turn 8, the filing turn, still flags Finance for Webex
  (18,400); the wrap-up on turn 9 says which need Finance.
- `closed_branch()`: S3 closes the Rachel/Webex incident on turn 3. From turn 4 no tool
  call is about Webex, Rachel (E010) or her ticket, and the turn-9 summary is about
  Alex Kim only. S2's closed Rachel branch is checked in evals/session_checks.py.
- `citation_problems()`: every fact sentence on a turn that had tool results carries a
  label, and every label the answer uses is one a tool returned. Applied to every turn
  of every session. `citation_split()` names which of the two put a turn in
  `needs_human`.

Each returns a list of problems; empty means the check passed.
"""

import json
import re

from agent.outcome import _BRACKETED, uncited_sentences

FINANCE = re.compile(r"\bfinance\b", re.IGNORECASE)
# A clause that puts Miro (7,200, under the 7,500 bar) on the Finance list: it names
# Miro and Finance and no negation. "Webex and Figma need Finance, Miro does not" splits
# into two clauses and passes.
CLAUSE = re.compile(r"[.;,:\n]")
NEGATION = re.compile(r"\b(not|no|n't|under|below|except|without|neither|nor)\b|n't", re.IGNORECASE)


def miro_on_finance_list(answer: str) -> bool:
    return any(re.search(r"\bmiro\b", c, re.I) and FINANCE.search(c) and not NEGATION.search(c)
               for c in CLAUSE.split(answer))


CLOSED_TOPIC = re.compile(r"\b(webex|rachel|E010|T001|AR001|S001)\b", re.IGNORECASE)


def constraint_carry(session_id: str, n: int, outcome) -> list[str]:
    if not session_id.startswith("S4"):
        return []
    answer = outcome.answer
    problems = []
    if n == 5:
        if not FINANCE.search(answer):
            problems.append("turn 5 does not apply the Finance rule")
        if miro_on_finance_list(answer):
            problems.append("turn 5 puts Miro (7,200, under 7,500) on the Finance list")
    if n == 8 and not FINANCE.search(answer):
        problems.append("turn 8 files for Webex (18,400) without flagging Finance sign-off")
    if n == 9 and not FINANCE.search(answer):
        problems.append("turn 9 wrap-up does not say which need Finance")
    return problems


def closed_branch(session_id: str, n: int, outcome) -> list[str]:
    if not session_id.startswith("S3") or n < 4:
        return []
    problems = [f"turn {n} still works the closed Webex branch: {c['name']}({c['args']})"
                for c in outcome.tool_calls if CLOSED_TOPIC.search(json.dumps(c["args"]))]
    if n == 9 and (hit := CLOSED_TOPIC.search(outcome.answer)):
        problems.append(f"turn 9 summary brings back the closed branch ({hit.group(0)})")
    return problems


def invented_labels(outcome) -> list[str]:
    """Bracketed labels in the answer that no tool returned. `outcome.citations` holds
    the ones that check out (agent/outcome.py `cited_labels`)."""
    return [label for label in dict.fromkeys(_BRACKETED.findall(outcome.answer))
            if label not in outcome.citations]


def had_evidence(outcome) -> bool:
    """The turn put tool results in front of the model."""
    return bool(outcome.tools_called or outcome.sources)


def citation_problems(outcome) -> list[str]:
    problems = []
    if had_evidence(outcome) and (missing := uncited_sentences(outcome.answer)):
        problems.append(f"uncited sentences {missing}")
    if invented := invented_labels(outcome):
        problems.append(f"labels no tool returned {invented}")
    return problems


def citation_split(outcome) -> str | None:
    """Why a needs_human turn failed the citation rule: 'uncited', 'invented', 'both',
    or None when it ended needs_human for another reason (or did not)."""
    if outcome.status != "needs_human":
        return None
    uncited = had_evidence(outcome) and bool(uncited_sentences(outcome.answer))
    invented = bool(invented_labels(outcome))
    if uncited and invented:
        return "both"
    return "uncited" if uncited else "invented" if invented else None


def conversation_checks(session_id: str, n: int, outcome) -> list[str]:
    return (constraint_carry(session_id, n, outcome) + closed_branch(session_id, n, outcome)
            + citation_problems(outcome))
