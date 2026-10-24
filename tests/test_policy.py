"""Intent to scope, loadouts and visibility. No model and no server."""

import pytest

from agent.policy import (APPROVAL_TOOL, INTENTS, LOADOUTS, SCOPE_OF, WRITE_TOOLS,
                          needs_approval, rearm_tools, select_tools, visible_tools)
from tests.agent_stubs import plan

CATALOGUE = sorted([
    "list_policies", "get_policy", "search_knowledge_base", "search_hr_documents",
    "get_employee", "get_team", "check_software_subscription", "check_asset_inventory",
    "list_employee_tickets", "list_onboarding_tasks", "get_access_requests", "get_approvals",
    "request_approval", "create_access_request", "create_ticket",
])


def test_no_loadout_contains_a_write_or_request_approval():
    for intent, names in LOADOUTS.items():
        assert not set(names or []) & (WRITE_TOOLS | {APPROVAL_TOOL}), intent


def test_every_intent_has_a_loadout_and_a_scope():
    assert set(LOADOUTS) == set(INTENTS) == set(SCOPE_OF)
    assert set(SCOPE_OF.values()) == {"maya", "webex"}


@pytest.mark.parametrize("intent", INTENTS)
def test_action_confirmed_never_makes_a_write_visible(intent):
    shown = visible_tools(plan(intent, action_confirmed=True), CATALOGUE)
    assert not set(shown) & (WRITE_TOOLS | {APPROVAL_TOOL})


def test_none_is_every_read_and_empty_is_none():
    assert select_tools(plan("other")) is None
    assert visible_tools(plan("other"), CATALOGUE) == rearm_tools(CATALOGUE)
    assert not set(rearm_tools(CATALOGUE)) & (WRITE_TOOLS | {APPROVAL_TOOL})
    assert visible_tools(plan("policy_question", needs_tools=False), CATALOGUE) == []


def test_s1_and_s3_first_turns_see_different_tools():
    s1 = visible_tools(plan("policy_question"), CATALOGUE)       # E010, personal laptop
    s3 = visible_tools(plan("subscription_review"), CATALOGUE)   # E007, Webex licence
    assert s1 != s3
    assert "check_software_subscription" in s3 and "check_software_subscription" not in s1
    assert SCOPE_OF["policy_question"] == "maya" and SCOPE_OF["subscription_review"] == "webex"


def test_scope_comes_from_intent_not_user_group():
    # S6: an HR caller (E004) on an access workflow is still a Webex turn.
    assert SCOPE_OF[plan("access_request").intent] == "webex"


@pytest.mark.parametrize("intent", [i for i in INTENTS if SCOPE_OF[i] == "maya"])
def test_maya_intents_never_reach_the_approval_node(intent):
    assert needs_approval(plan(intent, action_confirmed=True)) is False


def test_confirmed_webex_action_goes_to_the_approval_node():
    assert needs_approval(plan("access_request", action_confirmed=True))
    assert not needs_approval(plan("access_request"))
