# Agentic Garbage Collection — Implementation Plan

**Goal**: Fix the 260K-token context-window crash during codebase exploration by giving
the agent a `commit_findings_and_clear_search` tool that saves critical findings to a
persistent scratchpad and triggers LangGraph-level message eviction via `RemoveMessage`.

**Design principle**: Eviction happens at the **LangGraph state level** (removing messages
from the `messages` channel), not at the DA middleware level (trimming content within
messages). This is the only way to actually reduce the context window for subsequent LLM
calls — middleware trimming only compacts message *content* while leaving message *count*
unchanged, so tokenisation overhead from message framing still accumulates.

---

## 1. State Schema Update

### File: `spine/workflow/subgraph_state.py`

Add a `scratchpad` field to `ExplorationSubgraphState`:

```python
class ExplorationSubgraphState(BaseSubgraphState, total=False):
    # ... existing fields ...
    scratchpad: Annotated[str, _op_add]  # ← NEW: working memory accumulator
```

**Why `operator.add`**: String concatenation merges per-round contributions.
Each `commit_findings_and_clear_search` call appends a new entry. The
synthesizer reads the accumulated string.

**Why NOT `WorkflowState`**: The scratchpad is only needed during exploration
(SPECIFY + PLAN). Adding it to the parent `WorkflowState` would pollute every
phase's state. The subgraph state mapper already handles field extraction.

### File: `spine/workflow/compose.py`

Update the state mappers for SPECIFY and PLAN exploration subgraphs to pass
the `scratchpad` field through. The mapper already copies fields like `phase`,
`work_id`, `description` — add `scratchpad` to the copied set.

---

## 2. The Eviction Tool

### New file: `spine/agents/garbage_collector.py`

```python
"""Agentic garbage collection — commit_findings_and_clear_search tool."""

from langchain_core.tools import tool

# ── Anchor string (must be exactly this) ────────────────────────────
EVICTION_ANCHOR = (
    "SUCCESS: Findings saved to the system scratchpad. "
    "SYSTEM ALERT: All previous search history has been successfully "
    "evicted from the context window. You now have a clean context. "
    "Only the scratchpad (below) and this message remain from prior turns. "
    "Use the scratchpad to pick up where you left off."
)

@tool
def commit_findings_and_clear_search(note: str, relevant_code: str) -> str:
    """Save critical findings to a persistent scratchpad and trigger context eviction.

    When you call this tool, the system will:
    1. Append your findings to the persistent scratchpad
    2. Delete ALL prior search history from the context window
    3. Preserve only the essential: scratchpad, this message, and new messages

    Use this tool when you've gathered enough information and your context
    window is getting full. After calling, you start fresh but with all
    your key findings preserved in the scratchpad.
    """
    # The actual state update happens in the LangGraph node — this tool
    # just returns the anchor string. The node checks for this anchor
    # and triggers eviction.
    return EVICTION_ANCHOR
```

### File: `spine/agents/__init__.py` (or inline import)

No changes needed if the tool is imported directly by the exploration subgraph.

**Design decision — why a real tool, not prompt engineering**: The model
must have agency over when eviction happens. If we evict automatically
(e.g., every N turns), we might delete context the model still needs. If we
rely purely on prompt instructions to "tell us when to evict," the model
may hallucinate the anchor string without actually calling the tool. A real
tool forces the model to make an explicit `tool_call` decision, which
triggers the LangGraph state machine. This is the same pattern as `write_file`
— the tool call is the signal.

---

## 3. Boundary-Preserving Eviction Logic

### New file: `spine/agents/garbage_collector.py` (continued)

```python
"""Boundary-preserving eviction logic for agentic garbage collection."""

from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# Not used — we preserve the boundary. See explanation below.

def calculate_safe_eviction(messages: list) -> list[RemoveMessage]:
    """Return RemoveMessage objects that safely evict old search history.

    SAFETY GUARANTEE: Preserves the Boundary AIMessage and ALL sibling
    ToolMessages from the same turn, preventing parallel-tool parser crashes.

    ALGORITHM:
    1. Scan messages backward to find the last ToolMessage whose content
       matches EVICTION_ANCHOR.
    2. From that ToolMessage, scan backward to find the Boundary AIMessage
       (the AIMessage whose tool_calls list includes the commit_findings tool).
    3. Collect ALL ToolMessages whose tool_call_id appears in the Boundary's
       tool_calls list (these are siblings — must be preserved).
    4. Generate RemoveMessage for every AIMessage and ToolMessage that
       occurred BEFORE the Boundary AIMessage.
    5. Do NOT remove: the Boundary, sibling ToolMessages, any SystemMessage,
       or any HumanMessage.

    Returns:
        List of RemoveMessage objects. Empty list = no eviction needed.
    """
    # Step 1: Scan backward for the EVICTION_ANCHOR ToolMessage
    anchor_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage) and msg.content == EVICTION_ANCHOR:
            anchor_idx = i
            break

    if anchor_idx is None:
        return []  # No eviction tool was called

    # Step 2: Scan backward from anchor to find the Boundary AIMessage
    boundary_idx: int | None = None
    for i in range(anchor_idx - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if tc.get("name") == "commit_findings_and_clear_search":
                boundary_idx = i
                break
        if boundary_idx is not None:
            break

    if boundary_idx is None:
        return []  # Should not happen, but be defensive

    # Step 3: Collect IDs of all sibling ToolMessages in the same turn
    preserved_ids: set[str] = set()
    for tc in messages[boundary_idx].tool_calls:
        preserved_ids.add(tc.get("id", ""))

    # Step 4: Generate RemoveMessages for everything before the boundary
    to_remove: list[RemoveMessage] = []
    for i in range(boundary_idx):
        msg = messages[i]
        if isinstance(msg, (AIMessage, ToolMessage)):
            # Don't remove things that might be referenced by preserved siblings
            # (all ToolMessages have unique IDs; AIMessages don't have IDs
            #  that ToolMessages reference — safe to remove)
            if hasattr(msg, 'id') and msg.id:
                to_remove.append(RemoveMessage(id=msg.id))

    return to_remove
```

**Key invariants**:
- The Boundary AIMessage is never removed (it's at `boundary_idx`, and we iterate `range(boundary_idx)`).
- All sibling ToolMessages are at indices > boundary_idx — they're also never removed.
- SystemMessages and HumanMessages are never removed (only AIMessage/ToolMessage checked).
- The anchor ToolMessage itself is preserved (at index > boundary_idx).

**Parallel tool safety**: If the model calls `commit_findings_and_clear_search` AND
some other tools in the same turn (parallel tool calls), all sibling ToolMessages
are preserved because they share the same Boundary AIMessage.

---

## 4. Agent Node Integration

### File: `spine/workflow/subgraphs/exploration_subgraph.py`

Modify the `_explore_node` function (and the synthesizer nodes) to run
`calculate_safe_eviction` before/after agent invocation.

**Strategy — eviction in the explore node**:

The explore node (`_explore_node`) is where context accumulates — each
researcher subagent produces 10-30 ToolMessages for MCP tool calls + file
reads. After the explore node completes and returns its findings, the
subgraph routes back to `_research_manager` for another round. The `messages`
channel carries ALL prior explore nodes' conversation histories.

**Two-phase eviction in the exploration loop**:

#### Phase A: Inject the eviction tool into researcher subagents

In `run_explore_node` (`spine/agents/exploration_agents.py`):

```python
# In run_explore_node(), add commit_findings_and_clear_search to the
# researcher subagent's toolbar:
from spine.agents.garbage_collector import commit_findings_and_clear_search

# Add to extra_tools (the subagent already gets tools from subagent_spec)
extra_tools = list(subagent_spec.get("tools", []))
extra_tools.append(commit_findings_and_clear_search)
```

Also inject the scratchpad into the researcher's prompt:

```python
scratchpad = state.get("scratchpad", "")
if scratchpad:
    prompt += f"\n\n## Working Memory Scratchpad\n{scratchpad}\n"
```

#### Phase B: Eviction in the aggregate/sufficiency router

After the aggregate node completes (all parallel explore nodes have finished),
add an eviction check before routing back to the research manager:

```python
def _sufficiency_router(state: ExplorationSubgraphState) -> Literal["loop", "done"]:
    """Check whether research is sufficient, with garbage collection."""
    # ... existing logic ...

    # NEW: Check for eviction signals in the messages
    messages = state.get("messages", [])
    evictions = calculate_safe_eviction(messages)
    
    if evictions:
        # Extract findings from the latest commit_findings calls
        new_findings = _extract_committed_findings(messages)
        
        return {
            "messages": evictions,  # Triggers LangGraph RemoveMessage
            "scratchpad": _format_scratchpad_entry(new_findings),
        }
    
    return {}  # No eviction needed
```

**Wait — this approach has a subtlety**: The `_sufficiency_router` is a
**conditional edge function** in LangGraph, and *conditional edge functions
cannot return state updates*. They can only return routing strings.

**Corrected design — eviction node after aggregate**:

Add a new node `_eviction_check` between `aggregate` and the sufficiency
router. The edge becomes:

```
aggregate → eviction_check → sufficiency router → research_manager (loop) or synthesize (done)
```

The eviction check node:
1. Scans messages for the EVICTION_ANCHOR
2. If found, returns `{"messages": [RemoveMessage(...)]}` + scratchpad update
3. If not found, returns `{}` (no state changes)

```python
async def _eviction_check_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Evict old search history when commit_findings was called."""
    messages = state.get("messages", [])
    evictions = calculate_safe_eviction(messages)
    
    if not evictions:
        return {}
    
    # Extract committed findings from the evicted messages for the scratchpad
    scratchpad_entry = _extract_scratchpad_from_anchor(messages)
    
    logger.info("Evicting %d messages from exploration context", len(evictions))
    return {
        "messages": evictions,
        "scratchpad": scratchpad_entry,
    }
```

**Graph wiring**:

```python
# In build_exploration_subgraph():
builder.add_node("eviction_check", _eviction_check_node)
builder.add_edge("aggregate", "eviction_check")
builder.add_conditional_edges(
    "eviction_check",
    _sufficiency_router,
    {"loop": "research_manager", "done": "synthesize"},
)
```

### Why inject the tool into researcher subagents (not the exploration subgraph nodes)

The exploration subgraph nodes (`_explore_node`, `_synthesize_*`) use DA agents
internally. Those agents maintain their own internal `messages` list that is
separate from the subgraph's `ExplorationSubgraphState.messages`. The DA agent's
internal messages are what the LLM sees.

The DA agent returns `result["messages"]` which gets merged into the subgraph
state by the node function (e.g., via `return {"messages": result.get("messages", [])}`).

So the tool MUST be in the DA agent's tool surface. The researcher subagent
gets it via `extra_tools`. The synthesizer agent should NOT get it (the
synthesizer writes artifacts, it doesn't search).

---

## 5. System Prompt Update

### File: `spine/agents/subagents.py` — researcher prompt

Add an "Amnesia Warning" section to `SUBAGENT_PROMPTS["researcher"]`:

```python
SUBAGENT_PROMPTS["researcher"] = (
    # ... existing prompt ...
    
    "\n\n## ⚠ Amnesia Warning — Your Context IS Volatile\n"
    "Your context window has a hard limit. As you explore the codebase with\n"
    "MCP tools, previous search results WILL be deleted to make room for new\n"
    "information. You MUST use the `commit_findings_and_clear_search` tool\n"
    "before your context fills up.\n\n"
    "### When to save findings\n"
    "- After every 2-3 rounds of research — before starting the next batch\n"
    "- When you've identified key files, symbols, or patterns worth preserving\n"
    "- Before moving on to a new research topic\n"
    "- If tool results are getting truncated by context limits\n\n"
    "### How to use commit_findings_and_clear_search\n"
    "```\n"
    'commit_findings_and_clear_search(\n'
    '    note="Brief natural-language summary of what you found",\n'
    '    relevant_code="File paths, function names, code snippets, and patterns"\n'
    ')\n'
    "```\n"
    "The system will:\n"
    "1. Append your findings to a persistent scratchpad\n"
    "2. Delete ALL prior search history from the context window\n"
    "3. Only the scratchpad + new messages remain\n\n"
    "### The scratchpad\n"
    "After clearing, the scratchpad is injected back into subsequent turns.\n"
    "The scratchpad IS your working memory — it survives context eviction.\n"
    "Read it at the start of each new turn to pick up where you left off.\n"
)
```

### File: `spine/agents/subagents.py` — also the synthesizer prompts

The SPECIFY and PLAN synthesizer prompts in `exploration_subgraph.py` should
receive the scratchpad as part of the user prompt. Already handled by injecting
`scratchpad` into the findings-based prompt:

```python
# In _synthesize_specify / _synthesize_plan:
scratchpad = state.get("scratchpad", "")
if scratchpad:
    prompt += f"\n\n## Working Memory Scratchpad\n{scratchpad}\n"
```

---

## 6. File-by-File Change Summary

| File | Change | Complexity |
|------|--------|------------|
| `spine/agents/garbage_collector.py` | **New** — `commit_findings_and_clear_search` tool + `calculate_safe_eviction` helper + `EVICTION_ANCHOR` constant | Medium |
| `spine/workflow/subgraph_state.py` | Add `scratchpad: Annotated[str, _op_add]` to `ExplorationSubgraphState` | Trivial |
| `spine/agents/exploration_agents.py` | Inject `commit_findings_and_clear_search` into researcher subagent `extra_tools`; inject scratchpad into prompt | Small |
| `spine/workflow/subgraphs/exploration_subgraph.py` | Add `_eviction_check_node`; wire into graph (aggregate → eviction_check → sufficiency_router); inject scratchpad into synthesizer prompts; scratchpad in state mapper | Medium |
| `spine/workflow/compose.py` | Add `scratchpad` to exploration state mappers | Trivial |
| `spine/agents/subagents.py` | Add "Amnesia Warning" section to `SUBAGENT_PROMPTS["researcher"]` | Small |

### Files NOT modified

| File | Reason |
|------|--------|
| `spine/models/state.py` | `WorkflowState` is the parent — scratchpad lives in subgraph state only |
| `spine/agents/factory.py` | The tool is injected via `extra_tools`, not middleware |
| `spine/agents/context_editing.py` | `ToolOutputTrimmer` is a complementary mechanism (content trimming), not replaced |
| `spine/agents/specify_agent.py` | The specify orchestrator doesn't do exploration — researcher subagents do |
| `spine/agents/plan_agent.py` | Same as above |
| `spine/workflow/subgraph_wrapper.py` | Eviction is inside the subgraph, not the wrapper |

---

## 7. Testing Strategy

### Unit tests (new file: `tests/unit/test_garbage_collector.py`)

1. **`test_calculate_safe_eviction_no_anchor`** — no `EVICTION_ANCHOR` in messages → returns `[]`
2. **`test_calculate_safe_eviction_preserves_boundary`** — Boundary AIMessage and all sibling ToolMessages survive
3. **`test_calculate_safe_eviction_removes_prior`** — Messages before the boundary are removed
4. **`test_calculate_safe_eviction_parallel_tools`** — Parallel tool calls in the same turn are all preserved
5. **`test_calculate_safe_eviction_empty_messages`** — Empty message list → `[]`

### Integration tests

6. **`test_exploration_subgraph_with_gc`** — Full exploration loop with eviction
7. **`test_scratchpad_accumulates_across_rounds`** — Multiple `commit_findings` calls concatenate

---

## 8. Rollout Strategy

1. **Phase 1 — Tool + Eviction Logic**: Ship the tool and `calculate_safe_eviction`. The tool is available but the model may not use it yet (no prompt changes). Observer to confirm it doesn't break anything.

2. **Phase 2 — Node Integration**: Wire the eviction check node into the exploration subgraph. The graph now supports eviction but the model still doesn't know about the tool (unless prompt already mentions it).

3. **Phase 3 — Prompt Update**: Add the "Amnesia Warning" to the researcher prompt. Now the model knows it CAN and SHOULD save findings.

4. **Phase 4 — Monitor & Tune**: Watch LangSmith traces for:
   - `commit_findings_and_clear_search` call frequency
   - Token savings per exploration round
   - Whether the scratchpad is actually read by subsequent turns
   - Residual context window utilization after eviction

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Model calls `commit_findings` too aggressively (every turn) | The prompt says "every 2-3 rounds." If the model over-uses it, add a rate limit in the tool (count calls per phase, refuse after N). |
| Scratchpad grows unboundedly | The scratchpad is `str` with `operator.add`. If it exceeds e.g. 8K chars, the synthesizer prompt may overflow. Add truncation in `_extract_scratchpad_from_anchor` to keep the most recent N chars. |
| Parallel tool safety miss — a sibling ToolMessage is deleted because it references a prior AIMessage | Our algorithm preserves ALL siblings by checking `tool_call_id` against the Boundary's `tool_calls`. The Boundary AIMessage itself is always preserved. Verified by unit test 3. |
| The `before_model` middleware hook trims the anchor message | The `ToolOutputTrimmer` only targets tool results older than `max_full_tool_results=20`. The anchor is the most recent ToolMessage — it won't be trimmed unless the agent produces >20 tool results AFTER the anchor. Unlikely in practice. |
| Researcher subagent calls the tool but the parent graph doesn't process it | The eviction check node runs AFTER every aggregate (which completes all parallel explore nodes). The researcher's messages are merged into the subgraph state by `_explore_node` → the eviction check scans the full subgraph `messages`. |