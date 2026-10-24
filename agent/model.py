"""The only Bedrock callers in the repo: `call_model()`, `get_model()` and `embed_text()`.

Retry policy comes from Lesson 12's hardened agent (`should_retry`, `call_with_retry`).
boto's own retries are switched off (`max_attempts: 1`), because boto retrying under
our `should_retry` would multiply the attempts: 4 of ours times boto's default 3 is 12
calls into a throttled endpoint. One layer decides, and it is this file.

The MCP server imports `call_model` (reranking) and `embed_text` (retrieval) from here
too, so a provider swap still touches one file.
"""

import functools
import json
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionError as BotoConnectionError
from botocore.exceptions import ReadTimeoutError
from langchain_aws import ChatBedrockConverse
from langfuse import get_client
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from agent import config

DEFAULT_MAX_TOKENS = 1024  # Converse needs an explicit cap (Lesson 2); callers may lower it.

# Everything outside this set is permanent. Retrying an unknown code spends the whole
# backoff ladder reproducing the same error, and under real throttling it deepens the
# outage. Fail closed.
TRANSIENT_ERROR_CODES = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
    "ModelNotReadyException",
})


def should_retry(exc: BaseException) -> bool:
    """The one place retry policy is decided."""
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code", "") in TRANSIENT_ERROR_CODES
    return isinstance(exc, (BotoConnectionError, ReadTimeoutError, ConnectionError, TimeoutError))


_retrying = retry(
    retry=retry_if_exception(should_retry),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.5, max=8),
    reraise=True,  # the caller sees the original ClientError, not a tenacity wrapper
)


def _boto_config() -> Config:
    return Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=120)


@functools.lru_cache(maxsize=1)
def _bedrock():
    return boto3.client("bedrock-runtime", region_name=config.aws_region(), config=_boto_config())


@_retrying
def _converse(request: dict) -> dict:
    return _bedrock().converse(**request)


def call_model(
    messages: list[dict],
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    name: str = "converse",
) -> dict:
    """One Converse call, recorded as a Langfuse generation named `name`.

    `messages` uses Converse's shape: [{"role": "user", "content": [{"text": ...}]}].
    `tool_choice` names a tool to force (structured output by tool forcing, Lesson 3).
    Returns the raw Converse response; the caller reads `output.message.content`.
    """
    request: dict[str, Any] = {
        "modelId": config.bedrock_model_id(),
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        request["system"] = [{"text": system}]
    if tools:
        request["toolConfig"] = {"tools": tools}
        if tool_choice:
            request["toolConfig"]["toolChoice"] = {"tool": {"name": tool_choice}}

    with get_client().start_as_current_observation(
        as_type="generation",
        name=name,
        model=request["modelId"],
        input={"system": system, "messages": messages},
        model_parameters={"maxTokens": max_tokens, "temperature": temperature},
    ) as generation:
        response = _converse(request)
        usage = response.get("usage", {})
        generation.update(
            output=response["output"]["message"],
            usage_details={
                "input": usage.get("inputTokens", 0),
                "output": usage.get("outputTokens", 0),
            },
            metadata={"stop_reason": response.get("stopReason")},
        )
    return response


def response_text(response: dict) -> str:
    return "".join(
        block.get("text", "") for block in response["output"]["message"]["content"]
    ).strip()


def forced_tool_input(response: dict, tool_name: str) -> dict | None:
    """The input of the forced tool call, or None when the model did not make it."""
    for block in response["output"]["message"]["content"]:
        tool_use = block.get("toolUse")
        if tool_use and tool_use.get("name") == tool_name:
            return tool_use.get("input")
    return None


def get_model(**kwargs) -> ChatBedrockConverse:
    """The LangGraph model (task #3). Same model id, region and retry-off boto config.
    Temperature zero reduces run-to-run drift."""
    return ChatBedrockConverse(
        model=config.bedrock_model_id(),
        region_name=config.aws_region(),
        temperature=0,
        max_tokens=kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS),
        config=_boto_config(),
        **kwargs,
    )


@_retrying
def _invoke_embedding(body: str) -> dict:
    response = _bedrock().invoke_model(modelId=config.embedding_model_id(), body=body)
    return json.loads(response["body"].read())


def embed_text(text: str) -> list[float]:
    """One Titan V2 embedding. normalize=True makes inner product equal cosine
    similarity, which is what the innerproduct HNSW index assumes. Ingest and query
    must use the same model, or the vector geometry is meaningless."""
    body = json.dumps({"inputText": text, "dimensions": config.EMBED_DIM, "normalize": True})
    return _invoke_embedding(body)["embedding"]
