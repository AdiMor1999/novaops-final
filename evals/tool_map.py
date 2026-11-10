"""The one map from the golden dataset's illustrative tool names to ours.

`GOLDEN-DATASETS.json` says `expected_tools`, `optional_tools`, `sequence` and
`arg_checks` describe the reference build's ten Lesson 10 tools and will drift, and that
they should be mapped once, in our own code. This is that place. The S1 integration
test and the task #6 scorer both import it; no other file keeps a copy.

The binding fields (`forbidden_tools`, `should_use_tools`) go through the same map but
are never loosened: a forbidden name maps only to itself.
"""

# Illustrative name -> our tools that do the same job. Our search tools return policy
# sections with the same citation labels as get_policy, so a policy question answered
# through search_knowledge_base meets an expected get_policy (the dataset's own example).
TOOL_MAP: dict[str, frozenset[str]] = {
    "get_policy": frozenset({"get_policy", "search_knowledge_base"}),
    "list_policies": frozenset({"list_policies", "get_policy"}),
    "search_knowledge_base": frozenset({"search_knowledge_base", "get_policy"}),
    "search_hr_documents": frozenset({"search_hr_documents"}),
    "get_employee": frozenset({"get_employee"}),
    "list_employee_tickets": frozenset({"list_employee_tickets"}),
    "check_software_subscription": frozenset({"check_software_subscription"}),
    "check_asset_inventory": frozenset({"check_asset_inventory"}),
    "list_onboarding_tasks": frozenset({"list_onboarding_tasks"}),
    "create_access_request": frozenset({"create_access_request"}),
}


class UnmappedTool(KeyError):
    """A golden tool name this map does not cover. Add it here; do not guess."""


def ours(name: str) -> frozenset[str]:
    """Our tools for one illustrative name. 'a|b' means either alternative."""
    out: set[str] = set()
    for alternative in name.split("|"):
        if alternative not in TOOL_MAP:
            raise UnmappedTool(alternative)
        out |= TOOL_MAP[alternative]
    return frozenset(out)


def expected_tools_met(expected: list[str], called: list[str]) -> bool:
    """Every expected entry is met by at least one call. An empty list means nothing
    is expected, and whether any tool may run is should_use_tools' job."""
    return all(ours(name) & set(called) for name in expected)


def forbidden_tools_hit(forbidden: list[str], called: list[str]) -> list[str]:
    """Calls that a forbidden entry names. Exact names only: a forbidden tool is never
    widened to its look-alikes."""
    names = {alternative for entry in forbidden for alternative in entry.split("|")}
    return [c for c in called if c in names]


# `single_turn_cases.expected_sources` names database tables as well as documents
# ("contracts", "onboarding_tasks"). A table is met when one of our tools that reads it
# ran on the turn. check_software_subscription returns the subscription row joined with
# its contract's dates and cost.
TABLE_TOOLS: dict[str, frozenset[str]] = {
    "contracts": frozenset({"check_software_subscription"}),
    "software_subscriptions": frozenset({"check_software_subscription"}),
    "onboarding_tasks": frozenset({"list_onboarding_tasks"}),
}


def table_read(table: str, called: list[str]) -> bool:
    if table not in TABLE_TOOLS:
        raise UnmappedTool(table)
    return bool(TABLE_TOOLS[table] & set(called))
