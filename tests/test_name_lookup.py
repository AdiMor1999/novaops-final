"""The misspelt-name fallback of get_employee (db.similar_employees). It accepts a typo
of the whole full name and never another person's first or last name."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server import db
from mcp_server.server import mcp

META = {"novaops": {"caller_id": "E010", "user_group": "UG_REGULAR", "request_id": "REQ-N1"}}


def names(conn, query):
    return [r["full_name"] for r in db.similar_employees(conn, query)]


@pytest.mark.parametrize("query", ["rrachel stein", "rrachel steins", "Stein, Rachel"])
def test_a_typo_of_the_full_name_finds_one_employee(conn, query):
    assert names(conn, query) == ["Rachel Stein"]


@pytest.mark.parametrize("query", [
    "rachel stern",      # one changed letter is another surname, not a typo
    "rachel brown",      # the first name alone matches two people and proves nothing
    "rachel steinberg",  # a longer surname
    "tim baker",         # a different first name, same surname
    "danny cohen",
    "maya cohen rosen",  # two employees' names merged
])
def test_a_near_miss_name_matches_nobody(conn, query):
    assert names(conn, query) == []


def test_every_employee_name_matches_only_itself(conn):
    every = [r[0] for r in conn.execute("SELECT full_name FROM employees")]
    assert all(names(conn, name) == [name] for name in every)


async def test_get_employee_returns_no_row_for_a_near_miss(seeded):
    async with create_connected_server_and_client_session(mcp) as client:
        typo = await client.call_tool("get_employee", {"query": "rrachel stein"}, meta=META)
        miss = await client.call_tool("get_employee", {"query": "rachel stern"}, meta=META)
    typo, miss = json.loads(typo.content[0].text), json.loads(miss.content[0].text)
    assert typo["match"] == "similar" and [r["employee_id"] for r in typo["rows"]] == ["E010"]
    assert miss["count"] == 0 and "match" not in miss
