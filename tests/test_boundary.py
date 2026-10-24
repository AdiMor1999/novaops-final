"""The MCP boundary, over the real protocol (in-memory transport)."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server.server import mcp

HIDDEN = {"caller_id", "user_group", "request_id"}
META = {"novaops": {"caller_id": "E010", "user_group": "UG_REGULAR", "request_id": "REQ-B1"}}


async def test_record_approval_decision_is_not_a_tool(seeded):
    async with create_connected_server_and_client_session(mcp) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert "record_approval_decision" not in names
    assert {"request_approval", "get_approvals", "get_access_requests", "get_team",
            "create_ticket", "create_access_request"} <= names


async def test_no_schema_exposes_caller_context(seeded):
    async with create_connected_server_and_client_session(mcp) as client:
        tools = (await client.list_tools()).tools
    for tool in tools:
        text = json.dumps(tool.inputSchema)
        assert not HIDDEN & set(tool.inputSchema.get("properties", {})), tool.name
        assert not any(f'"{k}"' in text for k in HIDDEN), tool.name


async def test_call_without_caller_context_is_refused(seeded):
    async with create_connected_server_and_client_session(mcp) as client:
        result = await client.call_tool("get_employee", {"query": "E010"})
    assert result.isError
    assert "caller context" in result.content[0].text


async def test_call_with_context_reads_and_bad_ids_are_errors(seeded):
    async with create_connected_server_and_client_session(mcp) as client:
        ok = await client.call_tool("get_employee", {"query": "E010"}, meta=META)
        bad = await client.call_tool("list_onboarding_tasks", {"employee_id": "E999"}, meta=META)
        empty = await client.call_tool("get_team", {"manager_id": "E010"}, meta=META)
        seats = await client.call_tool("check_software_subscription", {"software": "Webex"}, meta=META)
    assert not ok.isError and json.loads(ok.content[0].text)["rows"][0]["full_name"] == "Rachel Stein"
    assert bad.isError and "get_employee" in bad.content[0].text
    assert json.loads(empty.content[0].text)["count"] == 0 and "note" in json.loads(empty.content[0].text)
    webex = json.loads(seats.content[0].text)["rows"][0]
    assert (webex["active_seats"], webex["seat_limit"]) == (42, 40)
    assert webex["seat_assignments_recorded"] is False


async def test_gated_write_over_mcp_is_refused(seeded):
    async with create_connected_server_and_client_session(mcp) as client:
        result = await client.call_tool(
            "create_access_request",
            {"employee_id": "E010", "software": "Webex", "business_justification": "calls"},
            meta=META)
    assert result.isError and "Refused" in result.content[0].text


async def test_agent_wrapper_injects_context_and_hides_it(seeded):
    from agent.mcp_tools import as_langchain_tools
    async with create_connected_server_and_client_session(mcp) as client:
        specs = (await client.list_tools()).tools
        tools = {t.name: t for t in as_langchain_tools(client, specs)}
        tool = tools["get_employee"]
        assert not HIDDEN & set(tool.tool_call_schema["properties"])
        out = await tool.ainvoke({"query": "E018"}, config={"configurable": META["novaops"]})
    assert "Yael Romano" in str(out)
