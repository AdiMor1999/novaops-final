"""The agent graph: one graph, both scopes.

    START ─(fresh history over threshold)→ distil ─┐
      └──────────────────────────────────────────→ classify ─(subject not in thread)→ lookup_subject
                                                      └──────────────────────────────────┴→ scope
    scope ├ confirmed access, subject preboarding → handoff ────────────→ answer
          ├ confirmed access, no subject verified → unverified ─────────→ answer
          └→ gather ⇄ tools                        (≤ MAX_TOOL_CALLS per turn)
               │  ├→ rearm → gather                (once, reads only)
               │  └→ refine → gather               (once: the checklist was never read)
               ├→ request_approval ─(already approved for this request)→ write_gate
               │      └→ decide → await_approval (interrupt)
               │                ↑ pending        │ resume
               │                └── write_gate ←─┘
               │                     ├ approved → gather (the write, forced) → tools → decide → answer
               │                     ├ rejected → decide → answer
               │                     └ otherwise → answer
               └→ cite (one repair) → checklist (onboarding rows only) → answer → END

Starting code: Lesson 10 `02-dynamic-tool-loadout/graph.py` and
`03-history-distillation/graph.py` (plan → select → model ⇄ tools, rearm), the
interrupt shape of Lesson 9 `03-human-in-the-loop/graph.py`, and the callback binding
of Lesson 11 `02-instrument-agent/graph.py`. Changes:

- `plan` and `select` are named `classify` and `scope`, the span names CLAUDE.md
  rule 2 fixes. `scope` also derives the assistant (`maya` or `webex`) from the intent.
- The checkpointer is a parameter. This module never constructs one, so the caller
  injects the durable `AsyncSqliteSaver` (Lesson 10 compiled with an in-memory saver).
- The planner goes through `parse_with_repair`; a second failure raises `ParseFailed`.
- Tools run in our own `tools` node through `as_langchain_tools()` and `call_tool()`,
  not LangGraph's `ToolNode`, so each call leaves exactly one `tool:<name>` span and
  carries caller context in MCP `_meta`. The node also enforces the per-turn cap and
  refuses a call to any tool that is not visible this turn.
- Writes are reachable only through `write_gate`, which opens on a persisted approval
  row (agent/gate.py). `request_approval` is called by the approval node, from code.
- Task #4: `ConversationMemory` distillation (Lesson 10 stage 3, agent/memory.py);
  the subject check and the Maya handoff through an injected `WebexPort`
  (agent/handoff.py, agent/ports.py); the per-sentence citation check with one
  repair call (`cite`); and the forced `submit_checklist` record (agent/records.py).
- Task #5, Webex: `lookup_subject` calls `get_employee` from code when a confirmed
  access request names someone the thread has not looked up. The approval node reads
  the case from code (subscription, tickets, access requests, approvals) and asks for
  the approvals of the open request, reusing AP001 and AP002 (agent/webex.py). `decide`
  builds the `AccessDecision` from a forced `submit_access_decision` draft plus code
  fields, before the pause and again after the released write. When the gate opens,
  the next model call is forced to the released write. Calls made by code go into the
  thread as tool call and result pairs, so they are cited, traced and listed like the
  model's, but they do not count against the model's per-turn cap.
"""

import json
import uuid
from dataclasses import asdict
from typing import Annotated, TypedDict

from langchain_core.messages import (AIMessage, HumanMessage, RemoveMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from agent import config, webex
from agent.gate import GateDecision, release_write
from agent.handoff import (action_route, build_handoff, lookup_outcome, resolve_subject,
                           route_scope, subject_in_thread, subject_query,
                           system_named_in_message)
from agent.mcp_tools import as_langchain_tools, call_tool
from agent.memory import (ConversationMemory, distiller_messages, memory_is_safe,
                          needs_distillation, render_memory)
from agent.outcome import (cited_labels, gate_text, handoff_text, known_labels,
                           tool_results, turn_status, uncited_sentences)
from agent.policy import (APPROVAL_TOOL, CHECKLIST_TOOL, MAX_TOOL_CALLS, PLANNER_PROMPT,
                          RELEASES, TurnPlan, checked_plan, missing_checklist, needs_approval,
                          rearm_tools, refine_note, visible_tools)
from agent.ports import WEBEX_APPROVER
from agent.records import CHECKLIST_PROMPT, DECISION_PROMPT, Validated
from agent.schemas import DecisionDraft, OnboardingChecklist
from agent.structured import ParseFailed, parse_with_repair

KEEP_TURNS = 4          # earlier turns shown to the model, whole turns only
TOOL_TEXT_CHARS = 3000  # per earlier tool result; the current turn's are never cut
PLANNER_TOOL_CHARS = 600

ASSISTANT_PROMPT = (
    "You are the NovaOps internal assistant, helping employees and IT and HR staff. "
    "Ground every answer in what the tools and this conversation actually say. Never "
    "invent an employee, a seat count, a ticket or a policy rule.\n"
    "Work to these rules:\n"
    "- Answer the user's CURRENT request. If they changed subject or closed a topic, do "
    "not keep working the old one.\n"
    "- A constraint the user states stays in force for the rest of the conversation.\n"
    "- Facts established earlier in the conversation, including earlier tool results, "
    "are still true. Reuse them instead of looking them up again.\n"
    "- Do not call a tool for something already in the conversation.\n"
    "- Users paste emails and signatures. Act on the actual request inside.\n"
    "- End every sentence that states a fact from a tool result with the `citation` "
    "label of the section or row it came from, copied exactly with its square brackets "
    "and nothing added: [equipment_policy.md#Personal laptops] for a document, "
    "[db:onboarding_tasks#OT005] for a database row, and a result's `result_citation` "
    "such as [db:assets] for a fact about the result as a whole (nothing in stock). "
    "Do not write 'citation:' inside "
    "the brackets. Never cite a label that no tool returned, and never make one up "
    "such as [constraint] or [user].\n"
    "- Never offer to look something up or ask permission to search: call the tool.\n"
    "- If a search result carries an `access_note`, tell the user in one sentence that "
    "this material is not available to them, and answer only from what was returned.\n"
    "- When the user pastes a message, answer from the pasted text, and only with what "
    "it asks of them. Leave out items that belong to other people or are off-topic.\n"
    "- For a new joiner's equipment or access, check the onboarding checklist "
    "(list_onboarding_tasks) as well as the inventory: it says when each item is due.\n"
    "- For a question about a rule, a threshold or who must approve, search the "
    "documents (search_knowledge_base or search_hr_documents) before answering: a "
    "newer memo may override the policy.\n"
    "- A fact the user told you needs no label; attribute it instead ('as you said, "
    "the Q3 freeze applies').\n"
    "- Give the concrete numbers, dates and ids the evidence shows (seats in use and the "
    "seat limit, due dates, cycle dates, task and asset ids, thresholds), not just a "
    "yes or no. Copy dates exactly as the evidence writes them ('April 1', not 'April'). "
    "A question about whether something happens in time needs its date.\n"
    "- A summary or status write-up opens with who the person is, their start date and "
    "location, copied from the established facts, then the open items and the "
    "constraints in force. Leave out topics the user closed.\n"
    "- If documents disagree, newer or more specific guidance (a later memo) overrides "
    "older or more general guidance (a policy); say which one you followed.\n"
    "- Asked for steps or a process, even as 'can you find the steps?', write the steps "
    "out from the section text that says what to do (for example a troubleshooting "
    "section), with its label. Naming the article, its owner or its date is not an "
    "answer.\n"
    "- If the evidence does not answer the question, say what is missing. Do not guess.\n"
    "- Never say access was granted or a request was filed unless a tool result shows it.\n"
    "- Answer in at most six sentences, with no preamble."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    request_id: str          # owned by run_turn(); kept across pause and resume
    trace_id: str            # the trace a resumed run continues
    plan: dict
    scope: str
    intent: str
    visible_tools: list[str]
    rearmed: bool
    caller: dict             # {"caller_id", "user_group"} from graph config, for prompts
    refined: bool            # the one refined round of this turn has run
    refine: str              # what the refined round asks for, shown to the model
    force: str               # a tool the next gather call must call (the refined round)
    turn_tools: list[str]    # tool calls made this turn, in order, by the model and by code
    cap_offset: int          # calls in turn_tools that do not count against the model's cap
    turn_errors: list[str]
    approval: dict | None    # the request_approval row that gates the write (AP001)
    approvals: list[dict]    # every request_approval row of this case (AP001, AP002)
    case: dict | None        # {employee_id, system, system_id, record_id} of the access case
    gate: dict | None        # the last GateDecision
    subject_note: str        # why lookup_subject verified nobody
    decision: dict | None    # the AccessDecision
    decision_from: str       # id of the user message that opened the access case
    decided: bool            # decide() set this run's answer and status
    answer: str
    citations: list[str]
    status: str
    memory: dict             # ConversationMemory, durable across turns
    distilled_upto: int      # messages before this index are represented by memory
    route: str | None        # action_route() of this turn
    handoffs: dict           # idempotency key -> {"handoff", "result"}; durable
    handoff: dict | None     # this turn's handoff, if any
    draft: str               # the answer after the citation check
    uncited: list[str]       # sentences still uncited after the one repair
    checklist: dict | None


# History projections ----------------------------------------------------------------

def message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in content).strip()
    return str(content)


def turn_starts(messages: list) -> list[int]:
    return [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]


def current_turn(messages: list) -> list:
    starts = turn_starts(messages)
    return list(messages[starts[-1]:]) if starts else list(messages)


def earlier_turns(messages: list, keep: int = KEEP_TURNS) -> list:
    """Whole turns before the current one, so a tool call is never cut from its result."""
    starts = turn_starts(messages)
    if len(starts) < 2:
        return []
    first = starts[max(0, len(starts) - 1 - keep)]
    return list(messages[first:starts[-1]])


def render_transcript(messages: list, tool_chars: int) -> str:
    """Tool traffic as text. Bedrock refuses toolUse blocks in a request that binds no
    tools, so earlier turns' calls and results reach the model as text, not as blocks."""
    lines = []
    for message in messages:
        if isinstance(message, ToolMessage):
            body = message_text(message)
            if len(body) > tool_chars:
                body = body[:tool_chars] + f"... [+{len(body) - tool_chars} chars dropped]"
            lines.append(f"[tool result: {message.name}] {body}")
            continue
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        text = message_text(message)
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            text = (text + " " if text else "") + " ".join(
                f"[called {c['name']}({json.dumps(c['args'])})]" for c in calls)
        lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def render_plan(plan: TurnPlan) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"  - {item}" for item in items) if items else "  - (none)"

    return (
        "\n\n--- CONTEXT FOR THIS TURN ---\n"
        f"User's current intent: {plan.intent}\n"
        f"Established facts you must use:\n{bullets(plan.facts)}\n"
        f"Constraints still in force:\n{bullets(plan.active_constraints)}\n"
        "--- END CONTEXT ---"
    )


def planner_transcript(messages: list) -> str:
    return render_transcript(messages, PLANNER_TOOL_CHARS)


def caller_line(caller: dict | None) -> str:
    """Who "I" and "me" are. Taken from graph config (the session), never from message
    text, so showing it to the model grants nothing: tools still check the caller from
    MCP `_meta`. Without it, S9's "am I getting a laptop?" made the model ask for an id."""
    if not caller:
        return ""
    return (f"The user you are talking to is employee {caller['caller_id']} "
            f"({caller['user_group']}). 'I', 'me' and 'my' refer to them.")


def planner_messages(state: AgentState, caller: dict | None = None) -> list:
    """Before any distillation, the whole history. After it, the memory plus only the
    messages since (Lesson 10 stage 3), which is what bounds the planner's input."""
    covered = state.get("distilled_upto", 0) if state.get("memory") else 0
    history = planner_transcript(state["messages"][covered:-1])
    if covered:
        prompt = (f"Session memory:\n{render_memory(state['memory'])}\n\n"
                  f"Conversation since then:\n{history or '(nothing new)'}\n\n")
    else:
        prompt = f"Conversation so far:\n{history or '(this is the first turn)'}\n\n"
    prompt += f"User's newest message:\n{message_text(state['messages'][-1])}"
    who = caller_line(caller or state.get("caller"))
    return [SystemMessage(PLANNER_PROMPT + (f"\n{who}" if who else "")), HumanMessage(prompt)]


def gather_messages(state: AgentState, *, with_tools: bool) -> list:
    """System prompt, the plan, earlier turns as text, then the current turn. When no
    tools are bound, this turn's tool traffic is folded into the text as well."""
    plan = TurnPlan(**state["plan"])
    # The pinned clock (NOVAOPS_TODAY), so "before the 1st" is compared to a real date.
    system = ASSISTANT_PROMPT + f"\nToday is {config.today().isoformat()}."
    if who := caller_line(state.get("caller")):
        system += f"\n{who}"
    system += render_plan(plan)
    if state.get("memory"):
        system += f"\n\nSession memory:\n{render_memory(state['memory'])}"
    if state.get("refine"):
        system += f"\n\n{state['refine']}"
    earlier = render_transcript(earlier_turns(state["messages"]), TOOL_TEXT_CHARS)
    if earlier:
        system += f"\n\nEarlier in this conversation:\n{earlier}"
    turn = current_turn(state["messages"])
    if with_tools or len(turn) == 1:
        return [SystemMessage(system), *turn]
    gathered = render_transcript(turn[1:], 10 ** 6)
    system += f"\n\nTool results gathered for the current message:\n{gathered}"
    return [SystemMessage(system), turn[0]]


def earlier_result(messages: list, name: str, args: dict) -> dict | None:
    """A successful result of the same call already in `messages`, so code does not
    repeat a read the model made this turn."""
    ids = {c["id"] for m in messages if isinstance(m, AIMessage)
           for c in (m.tool_calls or []) if c["name"] == name and c["args"] == args}
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id in ids \
                and message.status != "error":
            found = tool_results([message])
            if found:
                return found[0]
    return None


def write_note(case: dict, approval: dict) -> str:
    return (f"Approval {approval['approval_id']} for {case['record_id']} is approved. Call "
            f"create_access_request now for employee {case['employee_id']}, software "
            f"{case['system']}, with the business justification from this conversation "
            "(the one drafted, or the existing request's). It records a request; it does "
            "not grant access.")


def decision_messages(window: list) -> list:
    """The forced draft's input: every message since the user opened the access case,
    tool traffic as text."""
    return [SystemMessage(f"{DECISION_PROMPT}\nToday is {config.today().isoformat()}."),
            HumanMessage(f"Conversation and tool results:\n"
                         f"{render_transcript(window, 10 ** 6)}")]


def from_message(messages: list, message_id: str | None) -> list:
    for i, message in enumerate(messages):
        if message.id == message_id:
            return list(messages[i:])
    return current_turn(messages)


# The graph --------------------------------------------------------------------------

def repair_messages(state: AgentState, draft: str, uncited: list[str],
                    invented: list[str]) -> list:
    """The citation repair: the same evidence, the draft, and what to fix, in one user
    message (Bedrock wants user and assistant turns to alternate)."""
    context = gather_messages(state, with_tools=False)
    question = message_text(current_turn(state["messages"])[0])
    problems = ""
    if uncited:
        problems += ("These sentences state facts without a citation label:\n"
                     + "\n".join(f"- {s}" for s in uncited) + "\n")
    if invented:
        problems += f"These labels were returned by no tool: {', '.join(invented)}\n"
    return [context[0], HumanMessage(
        f"{question}\n\nYour draft answer:\n{draft}\n\n{problems}\nRewrite the answer. "
        "End each fact sentence with the label of the tool result it came from, copied "
        "exactly. Attribute facts the user told you ('as you said') instead of labelling "
        "them. If no tool result supports a sentence, remove it. Return only the answer.")]


def checklist_messages(state: AgentState, draft: str) -> list:
    turn = current_turn(state["messages"])
    evidence = render_transcript(turn[1:], 10 ** 6)
    return [SystemMessage(CHECKLIST_PROMPT), HumanMessage(
        f"Request: {message_text(turn[0])}\n\nTool results:\n{evidence}\n\n"
        f"Answer already given:\n{draft}")]


def access_note_this_turn(messages: list) -> str | None:
    for result in tool_results(current_turn(messages)):
        if result.get("access_note"):
            return result["access_note"]
    return None


def with_access_note(text: str, messages: list) -> str:
    """S9 turn 4: a non-manager asks what the manager's guide says. The search ran with
    the playbook filtered out and returned the fixed note; the answer must say so
    rather than answer from look-alike handbook text as if it were the guide. Code adds
    the note when the user asked about manager material and the answer left it out."""
    note = access_note_this_turn(messages)
    asked = "manager" in message_text(current_turn(messages)[0]).lower()
    if note and asked and note.lower() not in text.lower():
        return f"{note} {text}"
    return text


def onboarding_rows_this_turn(state: AgentState) -> bool:
    return any(r.get("rows") for r in tool_results(current_turn(state["messages"]),
                                                   "list_onboarding_tasks"))


def build_graph(session, specs: list, checkpointer, *, model, planner, distiller=None,
                checklist=None, decision=None, port=None):
    """`session` is an open MCP ClientSession, `specs` its list_tools() result.
    `model` is the chat model (bind_tools), `planner` the structured TurnPlan model
    (`with_structured_output(TurnPlan, include_raw=True)`), `distiller` the structured
    ConversationMemory model, `checklist` the forced `submit_checklist` model,
    `decision` the forced `submit_access_decision` model, and `port` the WebexPort for
    this turn. `checkpointer` is injected and must be durable."""
    tools = {t.name: t for t in as_langchain_tools(session, specs)}
    catalogue = sorted(tools)
    bound: dict[tuple, object] = {}

    def model_for(names: list[str]):
        key = tuple(sorted(names))
        if key not in bound:
            bound[key] = model.bind_tools([tools[n] for n in key]) if key else model
        return bound[key]

    def mcp_config(config: RunnableConfig) -> dict:
        # Only `configurable` goes to the tool. The node's callbacks would make the
        # LangChain handler add a second span per call next to `tool:<name>`. They have
        # to be replaced with [], because ensure_config() skips None and would inherit
        # them from the running node's context.
        return {"configurable": config["configurable"], "callbacks": []}

    async def code_call(name: str, args: dict, config: RunnableConfig):
        """A tool call made by code, as a call and result pair for the thread, the
        decoded result (None on an error) and the error text."""
        call_id = f"code_{uuid.uuid4().hex[:16]}"
        request = AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}])
        try:
            result = await call_tool(session, name, args, config["configurable"])
        except ToolException as exc:
            return [request, ToolMessage(content=str(exc), tool_call_id=call_id, name=name,
                                         status="error")], None, str(exc)
        return [request, ToolMessage(content=json.dumps(result), tool_call_id=call_id,
                                     name=name)], result, None

    async def distil(state: AgentState) -> dict:
        """Lesson 10 stage 3. A memory that fails to parse after one repair, or that
        drops an active constraint, is rejected: the old memory and the raw history
        both stay, so the safe failure costs tokens, not facts."""
        try:
            memory = await parse_with_repair(
                distiller, distiller_messages(state, planner_transcript), ConversationMemory)
        except ParseFailed:
            return {}
        if not memory_is_safe(state.get("memory") or {}, memory):
            return {}
        return {"memory": memory.model_dump(), "distilled_upto": len(state["messages"])}

    async def classify(state: AgentState, config: RunnableConfig) -> dict:
        configurable = config["configurable"]
        caller = {"caller_id": configurable["caller_id"],
                  "user_group": configurable["user_group"]}
        plan = await parse_with_repair(planner, planner_messages(state, caller), TurnPlan)
        plan = checked_plan(plan, has_history=len(turn_starts(state["messages"])) > 1,
                            message=message_text(state["messages"][-1]),
                            caller_id=caller["caller_id"])
        return {"plan": plan.model_dump(), "intent": plan.intent, "caller": caller,
                "rearmed": False, "force": "",
                "refined": False, "refine": "",
                "turn_tools": [], "cap_offset": 0, "turn_errors": [], "approval": None,
                "approvals": [], "case": None, "gate": None, "subject_note": "",
                "decision": None, "decision_from": "", "decided": False,
                "answer": "", "citations": [], "status": "", "route": None,
                "handoff": None, "draft": "", "uncited": [], "checklist": None}

    async def lookup_subject(state: AgentState, config: RunnableConfig) -> dict:
        """A confirmed access request for someone the thread never looked up (the
        Rachel scenario; "go ahead" with no lookup turn). Code asks get_employee for
        the id or name the planner named. One match becomes the plan's subject; none
        or several leave it unverified, and scope routes the turn to needs_human."""
        plan = TurnPlan(**state["plan"])
        query = subject_query(plan, state["messages"])
        pair, result, error = await code_call("get_employee", {"query": query}, config)
        update = {"messages": pair, "turn_tools": [*state["turn_tools"], "get_employee"],
                  "cap_offset": state.get("cap_offset", 0) + 1}
        if error:
            update["turn_errors"] = [*state["turn_errors"], f"get_employee: {error}"]
            update["subject_note"] = (f"Looking up {query!r} failed ({error}), so nothing "
                                      "was requested.")
            return update
        employee_id, why = lookup_outcome((result or {}).get("rows", []))
        if employee_id:
            update["plan"] = plan.model_copy(
                update={"subject_employee_id": employee_id}).model_dump()
        else:
            update["subject_note"] = (f"Looking up {query!r}: {why}, so nothing was requested. "
                                      "Say which employee (by id) the access is for.")
        return update

    def scope(state: AgentState) -> dict:
        """A dict lookup, not a model call: the model says what the turn is about, and
        code decides which assistant owns it and what it may touch. A confirmed access
        request is routed on the subject's record (agent/handoff.py)."""
        plan = TurnPlan(**state["plan"])
        route = action_route(plan, state["messages"])
        tools_now = [] if route in ("handoff", "unverified") else visible_tools(plan, catalogue)
        return {"scope": route_scope(plan, route), "intent": plan.intent, "route": route,
                "visible_tools": tools_now}

    async def handoff(state: AgentState, config: RunnableConfig) -> dict:
        """The Maya handoff. No model call and no model tool call: code builds the
        typed handoff and calls the port once per idempotency key. A replay of the
        same request on this thread finds the key in state and does not call again."""
        plan = TurnPlan(**state["plan"])
        built = build_handoff(state["messages"], plan, config["configurable"])
        if isinstance(built, str):
            return {"gate": asdict(GateDecision(False, "unverified", built))}
        done = dict(state.get("handoffs") or {})
        if built.idempotency_key in done:
            return {"handoff": done[built.idempotency_key]}
        try:
            result = await port.request_access(built)
        except ToolException as exc:
            return {"gate": asdict(GateDecision(False, "error", f"Webex handoff: {exc}"))}
        record = {"handoff": built.model_dump(), "result": result.model_dump()}
        done[built.idempotency_key] = record
        return {"handoff": record, "handoffs": done}

    def unverified(state: AgentState) -> dict:
        plan = TurnPlan(**state["plan"])
        named = plan.subject_employee_id or plan.subject_name or "the person named"
        return {"gate": asdict(GateDecision(
            False, "unverified", state.get("subject_note") or
            f"No employee record matches {named}, so nothing was handed off or requested. "
            "Say which employee (by id) the access is for."))}

    def forced(name: str):
        """The model bound to one tool it must call (Converse `toolChoice: tool`)."""
        key = ("force", name)
        if key not in bound:
            bound[key] = model.bind_tools([tools[name]], tool_choice=name)
        return bound[key]

    def model_calls(state: AgentState) -> int:
        return len(state["turn_tools"]) - state.get("cap_offset", 0)

    async def gather(state: AgentState) -> dict:
        names = state["visible_tools"] if model_calls(state) < MAX_TOOL_CALLS else []
        if state.get("force") and names:
            runnable = forced(state["force"])
        else:
            runnable = model_for(names)
        response = await runnable.ainvoke(gather_messages(state, with_tools=bool(names)))
        return {"messages": [response], "force": ""}

    async def run_tools(state: AgentState, config: RunnableConfig) -> dict:
        last = state["messages"][-1]
        called, errors = list(state["turn_tools"]), list(state["turn_errors"])
        offset = state.get("cap_offset", 0)
        replies = []
        for call in last.tool_calls:
            if call["name"] not in state["visible_tools"]:
                replies.append(ToolMessage(
                    content=f"{call['name']} is not available for this request.",
                    tool_call_id=call["id"], name=call["name"], status="error"))
                errors.append(f"{call['name']}: not visible")
                continue
            if len(called) - offset >= MAX_TOOL_CALLS:
                replies.append(ToolMessage(
                    content=f"Not run: the limit of {MAX_TOOL_CALLS} tool calls per "
                            "message is reached. Answer with what you have.",
                    tool_call_id=call["id"], name=call["name"], status="error"))
                continue
            called.append(call["name"])
            reply = await tools[call["name"]].ainvoke(
                {"type": "tool_call", "name": call["name"], "args": call["args"],
                 "id": call["id"]}, config=mcp_config(config))
            if reply.status == "error":
                errors.append(f"{call['name']}: {message_text(reply)}")
            replies.append(reply)
        return {"messages": replies, "turn_tools": called, "turn_errors": errors}

    def rearm(state: AgentState) -> dict:
        """One retry with every read tool, after the model answered a turn the plan said
        needed a lookup without making one. The unshown draft is discarded so the retry
        does not just agree with it (Lesson 10)."""
        return {"visible_tools": rearm_tools(catalogue), "rearmed": True,
                "messages": [RemoveMessage(id=state["messages"][-1].id)]}

    def refine(state: AgentState, config: RunnableConfig) -> dict:
        """The one refined retrieval round (agent/CLAUDE.md): the draft is discarded and
        the next model call is forced to call the checklist tool (asking was not enough:
        Nova Lite skipped it). The subject is resolved to an employee id from the thread.
        Reads only, once per turn."""
        names = list(state["visible_tools"])
        if CHECKLIST_TOOL not in tools:
            return {"refined": True}
        if CHECKLIST_TOOL not in names:
            names.append(CHECKLIST_TOOL)
        plan = TurnPlan(**state["plan"])
        who = resolve_subject(state["messages"], plan.subject_employee_id,
                              config["configurable"]["caller_id"])
        return {"visible_tools": names, "refined": True, "force": CHECKLIST_TOOL,
                "refine": refine_note(who),
                "messages": [RemoveMessage(id=state["messages"][-1].id)]}

    async def request_approval(state: AgentState, config: RunnableConfig) -> dict:
        """The approval transition (spec §9 steps 2 to 6), from code. Read the case:
        the subscription, the subject's tickets and access requests, two knowledge-
        base searches (the problem and the rule), and the approval rows of the open
        request. Then ask for each approval with request_approval, which reuses
        AP001 and AP002 rather than creating rows. Nothing here is chosen by the
        model, and nothing here writes a business record."""
        plan = TurnPlan(**state["plan"])
        messages = state["messages"]
        turn = current_turn(messages)
        out: list = []
        last = messages[-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            # The model's unchecked prose ("I've filed it") never stays in the thread.
            out.append(RemoveMessage(id=last.id))
        seen = list(turn)
        called, errors = list(state["turn_tools"]), list(state["turn_errors"])
        offset = state.get("cap_offset", 0)

        async def run(name: str, args: dict, *, reuse: bool = True) -> dict | None:
            nonlocal offset
            if reuse and (hit := earlier_result(seen, name, args)) is not None:
                return hit
            pair, result, error = await code_call(name, args, config)
            out.extend(pair)
            seen.extend(pair)
            called.append(name)
            offset += 1
            if error:
                errors.append(f"{name}: {error}")
            return result

        def finish(**update) -> dict:
            return {"messages": out, "turn_tools": called, "turn_errors": errors,
                    "cap_offset": offset, "decision_from": turn[0].id, **update}

        def refuse(state_name: str, reason: str) -> dict:
            return finish(gate=asdict(GateDecision(False, state_name, reason)))

        subject = subject_in_thread(messages, plan)
        if subject is None:
            return refuse("unverified", "No verified employee to act for.")
        employee_id = subject["employee_id"]
        system = plan.system or (system_named_in_message(messages) or {}).get("system_name")
        if not system:
            return refuse("missing", "The request names no system, so no approval was requested.")
        await run("check_software_subscription", {"software": system})
        await run("list_employee_tickets", {"employee_id": employee_id})
        await run("get_access_requests", {"employee_id": employee_id})
        for query in webex.CASE_QUERIES:
            await run("search_knowledge_base", {"query": query.format(system=system)})
        system_id = webex.system_id(seen, system)
        if system_id is None:
            return refuse("missing", f"No subscription or access request identifies the system "
                                     f"{system!r}, so no approval was requested.")
        existing = webex.open_access_request(seen, employee_id, system_id)
        if existing:
            record_id = existing["request_id"]
            await run("get_approvals", {"record_id": record_id})
            asks = webex.approval_requests(
                [r for r in webex.rows_of(seen, "get_approvals")
                 if r.get("request_id") == record_id],
                record_id)
        else:
            record_id = webex.new_request_record(employee_id, system_id)
            asks = [{"request_type": "access_request", "record_id": record_id,
                     "approver_id": WEBEX_APPROVER, "employee_id": employee_id,
                     "software": system}]
        rows = [row for ask in asks
                if (row := await run(APPROVAL_TOOL, ask, reuse=False)) is not None]
        gating = next((r for r in rows if r.get("request_type") in RELEASES), None)
        if gating is None:
            return refuse("error", "request_approval failed, so nothing is waiting on an "
                                   f"approver: {'; '.join(errors[-1:])}")
        if not gating.get("this_request"):
            return refuse("missing", f"{gating['approval_id']} is already held by another "
                                     f"request and waits on {gating['approver_id']}; this "
                                     "request cannot use it, and no new approval was created.")
        case = {"employee_id": employee_id, "system": system, "system_id": system_id,
                "record_id": record_id}
        return finish(approval=gating, approvals=rows, case=case)

    def await_approval(state: AgentState) -> dict:
        """The pause. interrupt() returns here when run_turn() resumes the thread. The
        resume value carries no decision: the decision is the approvals row, which
        application code wrote with record_approval_decision, and write_gate reads it."""
        approval = state["approval"]
        interrupt({"approval_id": approval["approval_id"], "record_id": approval["request_id"],
                   "approver_id": approval["approver_id"], "request_id": state["request_id"],
                   "also_requested": [a["approval_id"] for a in state.get("approvals") or []
                                      if a["approval_id"] != approval["approval_id"]]})
        return {}

    async def write_gate(state: AgentState, config: RunnableConfig) -> dict:
        approval = state["approval"]
        # A consumed approval this request already holds is a replay of the same
        # request; the server answers it with the row it returned the first time.
        replay = bool(approval.get("this_request") and approval.get("consumed"))
        decision = await release_write(session, config["configurable"],
                                       record_id=approval["request_id"],
                                       approval_id=approval["approval_id"], replay=replay)
        update = {"gate": asdict(decision), "decided": False}
        if decision.open:
            released = [n for n in decision.released if n in tools]
            update.update(visible_tools=released, force=released[0] if released else "",
                          refine=write_note(state["case"], approval),
                          # The resumed run gets a fresh model budget for the write.
                          cap_offset=len(state["turn_tools"]))
        return update

    async def decide(state: AgentState) -> dict:
        """The AccessDecision. The model drafts the observed facts and next steps through
        the forced submit_access_decision call, validated with every label checked
        against what tools returned (one repair, then ParseFailed). Code adds the
        status, blocking reason, reused records and actions, and writes the answer."""
        case = state["case"]
        window = from_message(state["messages"], state.get("decision_from"))
        known = known_labels(state["messages"])
        draft = await parse_with_repair(Validated(decision, DecisionDraft, known),
                                        decision_messages(window), DecisionDraft)
        record = webex.assemble_decision(draft, messages=window, employee_id=case["employee_id"],
                                         system=case["system"], system_id=case["system_id"],
                                         gate=state.get("gate"))
        text = webex.render_decision(record, state.get("approval"), state.get("gate"))
        valid, _ = cited_labels(text, known)
        return {"decision": record.model_dump(), "answer": text, "status": record.status,
                "citations": valid, "decided": True, "messages": [AIMessage(content=text)]}

    async def cite(state: AgentState) -> dict:
        """Every fact-stating sentence on a turn with tool results must carry a label.
        One repair call (the whole evidence again plus the draft, roughly doubling the
        answer's input tokens) on a miss; a second miss is left for answer() to mark
        needs_human. The repaired answer replaces the draft in the history."""
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and not last.tool_calls):
            return {}
        draft = message_text(last)
        if not known_labels(current_turn(state["messages"])):
            return {"draft": draft}
        _, invented = cited_labels(draft, known_labels(state["messages"]))
        missing = uncited_sentences(draft)
        if not (missing or invented):
            return {"draft": draft}
        repaired = await model_for([]).ainvoke(
            repair_messages(state, draft, missing, invented))
        text = message_text(repaired)
        return {"draft": text, "uncited": uncited_sentences(text),
                "messages": [RemoveMessage(id=last.id), AIMessage(content=text)]}

    async def make_checklist(state: AgentState) -> dict:
        record = await parse_with_repair(
            Validated(checklist, OnboardingChecklist, known_labels(state["messages"])),
            checklist_messages(state, state.get("draft", "")), OnboardingChecklist)
        return {"checklist": record.model_dump()}

    def answer(state: AgentState) -> dict:
        if state.get("decided"):
            return {}  # decide() wrote the answer, status and citations from the decision
        gate = state.get("gate")
        last = state["messages"][-1]
        done = state.get("handoff")
        if gate and not gate["open"]:
            text = gate_text(gate)
        elif done:
            text = handoff_text(done["handoff"], done["result"])
        elif state.get("draft"):
            text = with_access_note(state["draft"], state["messages"])
        elif isinstance(last, AIMessage) and not last.tool_calls:
            text = message_text(last)
        else:
            text = "I could not finish this request within the tool-call limit."
        if done and not (gate and not gate["open"]):
            # The handoff's status is the port's answer, not a guess from the text.
            return {"answer": text, "citations": [], "status": done["result"]["status"]}
        turn = current_turn(state["messages"])
        valid, invented = cited_labels(text, known_labels(state["messages"]))
        status = turn_status(TurnPlan(**state["plan"]), gate, state["turn_tools"],
                             state["turn_errors"], valid, invented,
                             bool(known_labels(turn)), state.get("uncited"))
        return {"answer": text, "citations": valid, "status": status,
                "messages": [AIMessage(content=text)] if text != message_text(last) else []}

    def after_gather(state: AgentState) -> str:
        last = state["messages"][-1]
        plan = TurnPlan(**state["plan"])
        if getattr(last, "tool_calls", None):
            return "tools"
        if plan.needs_tools and not state["rearmed"] and not state["turn_tools"] \
                and not state.get("gate"):
            return "rearm"
        if needs_approval(plan) and state.get("approval") is None and state.get("gate") is None:
            return "request_approval"
        if needs_approval(plan) and state.get("approval") and (state.get("gate") or {}).get("open"):
            return "decide"  # the gate opened but the model answered without the write
        if state["turn_tools"] and not state.get("refined") and not state.get("gate") \
                and len(state["turn_tools"]) < MAX_TOOL_CALLS \
                and missing_checklist(plan, state["turn_tools"]):
            return "refine"
        return "cite"

    def after_tools(state: AgentState) -> str:
        gate = state.get("gate")
        if gate and gate["open"] and state["turn_tools"] \
                and state["turn_tools"][-1] in gate["released"]:
            return "decide"
        return "gather"

    def after_classify(state: AgentState) -> str:
        plan = TurnPlan(**state["plan"])
        return "lookup_subject" if subject_query(plan, state["messages"]) else "scope"

    def after_start(state: AgentState) -> str:
        if distiller is not None and needs_distillation(state, planner_transcript):
            return "distil"
        return "classify"

    def after_scope(state: AgentState) -> str:
        return {"handoff": "handoff", "unverified": "unverified"}.get(state.get("route"),
                                                                      "gather")

    def after_cite(state: AgentState) -> str:
        plan = TurnPlan(**state["plan"])
        if checklist is not None and plan.intent == "onboarding_status" \
                and onboarding_rows_this_turn(state):
            return "checklist"
        return "answer"

    def after_request(state: AgentState) -> str:
        approval = state.get("approval")
        if not approval or state.get("gate"):
            return "answer"
        if approval.get("status") == "approved":
            return "write_gate"  # decided before this run: no pause, release now
        return "decide"

    def after_decide(state: AgentState) -> str:
        return "await_approval" if state["status"] == "pending" else "answer"

    def after_gate(state: AgentState) -> str:
        gate = state["gate"]
        if gate["open"]:
            return "gather"
        if gate["state"] == "pending":
            return "await_approval"
        return "decide" if gate["state"] == "rejected" else "answer"

    graph = StateGraph(AgentState)
    graph.add_node("distil", distil)
    graph.add_node("classify", classify)
    graph.add_node("lookup_subject", lookup_subject)
    graph.add_node("scope", scope)
    graph.add_node("gather", gather)
    graph.add_node("tools", run_tools)
    graph.add_node("rearm", rearm)
    graph.add_node("refine", refine)
    graph.add_node("request_approval", request_approval)
    graph.add_node("await_approval", await_approval)
    graph.add_node("write_gate", write_gate)
    graph.add_node("decide", decide)
    graph.add_node("handoff", handoff)
    graph.add_node("unverified", unverified)
    graph.add_node("cite", cite)
    graph.add_node("checklist", make_checklist)
    graph.add_node("answer", answer)

    graph.add_conditional_edges(START, after_start, ["distil", "classify"])
    graph.add_edge("distil", "classify")
    graph.add_conditional_edges("classify", after_classify, ["lookup_subject", "scope"])
    graph.add_edge("lookup_subject", "scope")
    graph.add_conditional_edges("scope", after_scope, ["handoff", "unverified", "gather"])
    graph.add_edge("handoff", "answer")
    graph.add_edge("unverified", "answer")
    graph.add_conditional_edges("gather", after_gather, ["tools", "rearm", "refine",
                                                         "request_approval", "decide", "cite"])
    graph.add_edge("refine", "gather")
    graph.add_conditional_edges("cite", after_cite, ["checklist", "answer"])
    graph.add_edge("checklist", "answer")
    graph.add_conditional_edges("tools", after_tools, ["gather", "decide"])
    graph.add_edge("rearm", "gather")
    graph.add_conditional_edges("request_approval", after_request,
                                ["decide", "write_gate", "answer"])
    graph.add_conditional_edges("decide", after_decide, ["await_approval", "answer"])
    graph.add_edge("await_approval", "write_gate")
    graph.add_conditional_edges("write_gate", after_gate,
                                ["gather", "await_approval", "decide", "answer"])
    graph.add_edge("answer", END)

    # The Langfuse callback is bound to the compiled graph (Lesson 11), so every node
    # and model call is exported whoever invokes it. CLAUDE.md rule 2: never remove it.
    return graph.compile(checkpointer=checkpointer).with_config(
        {"callbacks": [CallbackHandler()]})
