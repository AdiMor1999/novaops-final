"""PreToolUse hook: refuse agent edits to the files that measure the agent, and any read of
the held-out evaluation inputs.

Adapted from lesson-12-eval-loop-engineering/code/02-coding-loop/.claude/hooks/block_test_edits.py.
That hook blocked every test file, because its loop was not allowed to write tests at all. This
project writes its own tests, so the hook protects a narrower set:

    evals/targets.py            the bar the evals are held to
    tests/fixtures/**           recorded inputs and expected outputs
    .claude/**                  this hook and the permission lists
    GOLDEN-DATASETS.json        the dataset itself, wherever a copy lives
    novaops-course/**, lesson-code/**   read-only reference material

and blocks reading EVALUATION-INPUTS.yaml through Read, Grep or Bash. Development runs on the
golden dataset only; the evaluation inputs run once, in task #11.

LIMITS
------
Shell is an open-ended language and this is pattern matching. A command built from variables
or a Python one-liner can get past it. The hook catches the accidental case and makes the
deliberate one visible. The `permissions.deny` rules in settings.json and diff review are the
layers that hold.

CONTRACT
--------
stdin   one JSON object: {"tool_name": ..., "tool_input": {...}, "cwd": ...}
stdout  JSON with hookSpecificOutput.permissionDecision = "deny" to block
exit 0  even when denying. A non-zero exit means the hook itself failed.
"""

import json
import os
import re
import sys
from pathlib import Path, PurePath

EVAL_INPUTS = "EVALUATION-INPUTS.yaml"
PROTECTED_NAMES = {"GOLDEN-DATASETS.json", EVAL_INPUTS}
READ_ONLY_DIRS = {"novaops-course", "lesson-code"}

# Shell verbs that can create or rewrite a file. Not exhaustive, and cannot be.
WRITE_VERBS = re.compile(
    r"(>>?|\btee\b|\bsed\b\s+-[a-zA-Z]*i|\bcp\b|\bmv\b|\brm\b|\bdd\b|\btruncate\b|\bln\b)"
)
# Path-shaped tokens. Generous on purpose: protected_reason() does the deciding.
PATH_TOKEN = re.compile(r"[\w./~-]+")


def project_root(event: dict) -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or ".").resolve()


def relative_parts(path: str, root: Path) -> tuple[str, ...]:
    """Path parts relative to the project root when the path is inside it, else absolute parts."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve().relative_to(root).parts
    except ValueError:
        return PurePath(os.path.normpath(candidate)).parts


def protected_reason(path: str, root: Path) -> str:
    """Why an edit to this path is refused, or "" if it is allowed."""

    parts = relative_parts(path, root)
    if not parts:
        return ""
    if parts[-1] in PROTECTED_NAMES:
        return f"{parts[-1]} is a course dataset file and is never edited."
    if any(part in READ_ONLY_DIRS for part in parts):
        return "repos/novaops-course and lesson-code/ are read-only reference material."
    if parts[:2] == ("evals", "targets.py"):
        return "evals/targets.py is the bar the evals are held to. Say why it is wrong instead of moving it."
    if parts[:2] == ("tests", "fixtures"):
        return "tests/fixtures/ holds recorded expectations. Fix the code under test, or say which fixture is wrong."
    if parts[0] == ".claude":
        return ".claude/ holds this hook and the permission lists. Only the user changes them."
    return ""


def reads_eval_inputs(path: str) -> bool:
    """True for the evaluation inputs file, or a directory a recursive search would read it from."""

    if not path:
        return False
    target = Path(path)
    return target.name == EVAL_INPUTS or (target.is_dir() and (target / EVAL_INPUTS).exists())


def deny(reason: str) -> int:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"Blocked by guard_protected_paths.py. {reason}",
        }
    }))
    return 0


EVAL_INPUTS_REASON = (
    f"{EVAL_INPUTS} is held out until task #11. Develop against GOLDEN-DATASETS.json."
)


def check_bash(command: str, root: Path) -> str:
    if EVAL_INPUTS in command:
        return EVAL_INPUTS_REASON
    if not WRITE_VERBS.search(command):
        return ""
    for token in PATH_TOKEN.findall(command):
        reason = protected_reason(token, root)
        if reason:
            return reason
    return ""


def main() -> int:
    # Unreadable stdin fails open: a crashing hook would block every tool call in the session,
    # and the deny rules in settings.json still apply.
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input") or {}
    root = project_root(event)

    if tool == "Bash":
        reason = check_bash(str(tool_input.get("command", "")), root)
        return deny(reason) if reason else 0

    # Glob is not checked: it lists file names and never reads contents.
    if tool in ("Read", "Grep"):
        path = tool_input.get("file_path") or tool_input.get("path") or event.get("cwd") or ""
        return deny(EVAL_INPUTS_REASON) if reads_eval_inputs(path) else 0

    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        reason = protected_reason(path, root) if path else ""
        return deny(reason) if reason else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
