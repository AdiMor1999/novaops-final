# agent/

One LangGraph graph serves both scopes. Starting code:
`lesson-10-context-engineering/code/03-history-distillation/graph.py`, `stage03_policy.py`,
`02-dynamic-tool-loadout/stage02_policy.py` and `00-agent-shared/agent.py`.

## Graph

- The flow is classify (`TurnPlan`) → scope → gather evidence → answer. Classify is the router.
  Do not add a router node above it.
- Intent comes from the newest message, and the loadout is recomputed every turn. Durable facts
  from closed topics live in `ConversationMemory`, not in tool choice.
- History is cut at turn boundaries, never between a tool call and its result.
- At most one refined retrieval round, then answer and say what is missing. `rearm` reopens
  read tools only. A turn cap applies.

## Tools and the write gate

- Tools are discovered from the MCP server with `list_tools()`. Never import a tool function.
- `LOADOUTS` is a dict keyed by intent. `None` means "not sure" and exposes all read tools.
  `[]` means no tools.
- Write tools never appear in `LOADOUTS`. They live in `WRITE_TOOLS`, and `release_write()` is
  the only way to reach one. It opens only when a persisted approval row exists for this
  action, caller and request. `action_confirmed` from the planner affects visibility only and
  never opens the gate.
- `request_approval` is in neither `LOADOUTS` nor `WRITE_TOOLS`, because it needs no prior
  approval. The graph's approval node calls it from code when a Webex action intent has
  `action_confirmed`, then interrupts. The model never calls it directly.
- Maya's intents have no route to `release_write()` or to the approval node.
- The gate in the graph is a convenience, not authorization. The MCP server checks the
  approval again.

## Pause and resume

- Approval waits use `interrupt()`. The checkpointer (`AsyncSqliteSaver`) is injected from
  outside `build_graph()`. Never construct one inside the graph module.
- Application code writes the approval decision to `approvals` before resuming with
  `Command(resume=...)`. The resume continues the same thread and the same trace.

## Outputs

- `OnboardingChecklist`, `AccessDecision` and `Handoff` are Pydantic models produced by a
  forced final tool call and validated locally. One repair attempt, then `ParseFailed`. No
  silent default, and no `result["parsed"] or TurnPlan(...)` fallback.
- `status` is set by code from tool results. `AccessDecision` never says access was granted.
- Every factual claim cites a `[source#section]` label from a retrieved chunk.

## Entry point

```python
run_turn(message, *, caller_id, user_group, thread_id, request_id=None)
```

This is the only way in. Tests, the CLI and the web chat all call it. It attaches the Langfuse
callback, sets `session_id` = `thread_id` and `user_id` = `caller_id`, puts caller context and
`request_id` in graph config, and flushes Langfuse on exit.

`run_turn()` owns `request_id`:

- If the thread is paused on an approval, it reads `request_id` from the checkpoint. A
  different `request_id` passed in is an error, not an override.
- Otherwise it uses the `request_id` the caller passed, or creates one with `uuid4()` when the
  caller passed none. The eval runner passes the case id so traces match cases.
- It stores `request_id` in graph state, so the checkpoint keeps it across the pause and the
  resume.
- It puts `request_id` in trace metadata and the tool wrapper injects it into every MCP call.
  The model never creates, sees or chooses it.
