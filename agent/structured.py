"""Structured output with one repair attempt, then a named failure.

Starting code: `parse_with_repair` from Lesson 12
`samples/02-coding-loop/agent-hardened/agent.py`. It replaces Lesson 10's
`result["parsed"] or TurnPlan(current_intent="other", requires_tools=True)`, which
turned every planner parse failure into "other", the widest loadout, without telling
anyone. Here a failure costs one extra model call (the repair, which resends the
whole prompt plus the parse error) and then raises.
"""

from langchain_core.messages import HumanMessage
from pydantic import BaseModel


class ParseFailed(RuntimeError):
    """Structured output that still did not parse after one repair attempt."""

    def __init__(self, schema: str, error: str):
        super().__init__(f"{schema} did not parse after one repair attempt: {error}")
        self.schema = schema
        self.error = error


async def parse_with_repair(model, messages: list, schema: type[BaseModel]) -> BaseModel:
    """`model` is `.with_structured_output(schema, include_raw=True)`, so `ainvoke`
    returns {"raw", "parsed", "parsing_error"} and `parsed` is None on a failure."""
    result = await model.ainvoke(messages)
    if result.get("parsed") is not None:
        return result["parsed"]

    repair = messages + [HumanMessage(
        f"Your previous reply did not match the required {schema.__name__} schema "
        f"({result.get('parsing_error')}). Return ONLY a valid {schema.__name__}."
    )]
    result = await model.ainvoke(repair)
    if result.get("parsed") is not None:
        return result["parsed"]
    raise ParseFailed(schema.__name__, str(result.get("parsing_error")))
