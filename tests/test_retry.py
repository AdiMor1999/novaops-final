from botocore.exceptions import ClientError

from agent.model import should_retry


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "Converse")


def test_throttling_is_retried():
    assert should_retry(client_error("ThrottlingException"))


def test_unknown_and_permanent_codes_are_not():
    assert not should_retry(client_error("SomethingNew"))
    assert not should_retry(client_error("ValidationException"))
    assert not should_retry(ValueError("bad"))
