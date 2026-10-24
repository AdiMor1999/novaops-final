"""The PreToolUse hook as Daedalus runs it: wired from user settings with --workspace, in a
session folder above the repo checkout. Each case feeds one sample event on stdin; no
protected file is read or written."""

import json
import shutil
import subprocess
import sys

import pytest

from tests.conftest import REPO

HOOK = REPO / ".claude" / "hooks" / "guard_protected_paths.py"


@pytest.fixture
def layout(tmp_path):
    """workspace/worktrees/<task>/<session>/novaops-final, the shape Daedalus creates."""
    workspace = tmp_path / "novaops"
    session = workspace / "worktrees" / "5-task" / "claude-x"
    checkout = session / "novaops-final"
    (checkout / ".claude" / "hooks").mkdir(parents=True)
    shutil.copy(HOOK, checkout / ".claude" / "hooks" / HOOK.name)
    (checkout / "evals").mkdir()
    return workspace, session, checkout


def run_hook(event: dict, workspace=None) -> str:
    args = [sys.executable, str(HOOK)]
    if workspace is not None:
        args += ["--workspace", str(workspace)]
    done = subprocess.run(args, input=json.dumps(event), capture_output=True, text=True,
                          check=True, env={"PATH": "/usr/bin:/bin"})
    if not done.stdout.strip():
        return "allow"
    return json.loads(done.stdout)["hookSpecificOutput"]["permissionDecision"]


def test_bash_naming_the_eval_inputs_is_denied_from_a_session_folder(layout):
    workspace, session, _ = layout
    event = {"tool_name": "Bash", "cwd": str(session),
             "tool_input": {"command": "grep -n W-I EVALUATION-INPUTS.yaml | head -5"}}
    assert run_hook(event, workspace) == "deny"


def test_read_and_grep_of_the_eval_inputs_are_denied(layout):
    workspace, session, _ = layout
    path = str(workspace / "repos" / "novaops-course" / "EVALUATION-INPUTS.yaml")
    assert run_hook({"tool_name": "Read", "cwd": str(session),
                     "tool_input": {"file_path": path}}, workspace) == "deny"
    assert run_hook({"tool_name": "Grep", "cwd": str(session),
                     "tool_input": {"pattern": "W-I", "path": path}}, workspace) == "deny"


def test_edit_inside_a_checkout_below_the_session_folder_is_denied(layout):
    workspace, session, checkout = layout
    for path in (checkout / "evals" / "targets.py",
                 checkout / "tests" / "fixtures" / "case.json",
                 checkout / ".claude" / "settings.json"):
        event = {"tool_name": "Write", "cwd": str(session), "tool_input": {"file_path": str(path)}}
        assert run_hook(event, workspace) == "deny", path


def test_ordinary_edits_and_commands_are_allowed(layout):
    workspace, session, checkout = layout
    assert run_hook({"tool_name": "Write", "cwd": str(session),
                     "tool_input": {"file_path": str(checkout / "agent" / "graph.py")}},
                    workspace) == "allow"
    assert run_hook({"tool_name": "Bash", "cwd": str(checkout),
                     "tool_input": {"command": "python -m pytest -q"}}, workspace) == "allow"


def test_denied_shell_commands_are_refused(layout):
    workspace, session, _ = layout
    for command in ("git push --force origin main", "git reset --hard HEAD~1", "rm -rf build"):
        event = {"tool_name": "Bash", "cwd": str(session), "tool_input": {"command": command}}
        assert run_hook(event, workspace) == "deny", command


def test_the_hook_does_nothing_outside_the_workspace(layout, tmp_path):
    workspace, _, _ = layout
    elsewhere = tmp_path / "other-project"
    elsewhere.mkdir()
    event = {"tool_name": "Bash", "cwd": str(elsewhere),
             "tool_input": {"command": "rm -rf build"}}
    assert run_hook(event, workspace) == "allow"


def test_without_workspace_it_still_guards_the_repo_project(layout):
    _, _, checkout = layout
    event = {"tool_name": "Edit", "cwd": str(checkout),
             "tool_input": {"file_path": "evals/targets.py"}}
    assert run_hook(event) == "deny"
