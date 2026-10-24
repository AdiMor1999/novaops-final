from mcp_server.db import is_manager


def test_e018_manages_people_although_ug_regular(seeded):
    assert is_manager("E018", seeded) is True


def test_e001_manages_nobody(seeded):
    assert is_manager("E001", seeded) is False


def test_unknown_caller_and_missing_db_fail_closed(seeded, tmp_path):
    assert is_manager("E999", seeded) is False
    assert is_manager("E018", tmp_path / "missing.db") is False
