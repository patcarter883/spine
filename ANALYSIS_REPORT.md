# SPINE Codebase Analysis Report

## 1. EXPLORE → AGGREGATE LOOP

### Files Examined
- `/home/pat/Projects/spine/spine/agents/exploration_agents.py` (17,531 chars)
- `/home/pat/Projects/spine/spine/workflow/subgraphs/exploration_subgraph.py` (19,978 chars)
- `/home/pat/Projects/spine/spine/workflow/subgraph_state.py` (4,609 chars)
- `/home/pat/Projects/spine/spine/agents/subagents.py` (33,442 chars)
- `/home/pat/Projects/spine/spine/workflow/subgraph_wrapper.py` (13,085 chars)
- `/home/pat/Projects/spine/spine/models/types.py` (5,457 chars)

### Exact Flow: Manager → Router → Explore → Aggregate → Synthesize

**Current Architecture (lines 12-16 of exploration_subgraph.py):**

```
START → research_manager
         ↓
research_manager → Send("explore", {topic}) × N  OR  → synthesize
         ↓                              ↘
   explore (parallel via Send API) → aggregate (fan-in)
         ↓                              ↙
   _sufficiency_router → "loop" → research_manager  OR  "done" → synthesize
         ↓
   synthesize (Deep Agent writes spec/plan)
         ↓
   save_artifacts (scans disk, materializes to state)
         ↓
      END
```

### Structured Data Between Manager and Router

**Manager Decision Output (lines 147-153 of exploration_agents.py):**
```python
return {"manager_decision": decision, "topics": topics}
```

The `run_research_manager()` function returns a simple dict with:
- `manager_decision`: "explore" or "done" (JSON parsed from model response)
- `topics`: list of 1-4 research topics to investigate

**Manager Input Context (lines 126-134 of exploration_agents.py):**
```python
context = (
    f"## Work Description\n{description}\n\n"
    f"{spec_section}"
    f"## Round\n{round_num + 1} of max {max_rounds}\n\n"
    f"## Topics Already Explored\n{json.dumps(existing_topics)}\n\n"
    f"## Findings So Far\n{findings_summary}\n\n"
    "Decide: are we done, or do we need more research?"
)
```

### Router Decision Logic

**_research_router (lines 70-89 of exploration_subgraph.py):**
```python
def _research_router(state: ExplorationSubgraphState) -> list[Send] | Literal["synthesize"]:
    decision = state.get("manager_decision", "done")
    topics: list[str] = state.get("topics", [])
    
    if decision == "done" or not topics:
        return "synthesize"  # type: ignore[return-value]
    
    sends = [Send("explore", {"topic": t}) for t in topics]
    return sends
```

The router uses LangGraph's **Send API** to dispatch parallel explore nodes. Each `Send("explore", {"topic": t})` injects the topic into the subgraph state.

### How Explore Agents Are Invoked

Explore nodes are **NOT subgraphs** - they use the existing subagent machinery:

**run_explore_node (lines 192-333 of exploration_agents.py):**
1. Gets the topic from `state.get("topic")` (injected by Send API)
2. Builds a researcher subagent via `build_subagent_spec("researcher", ...)`
3. Creates a minimal agent via `build_phase_agent()` with `skip_filesystem_middleware=True`
4. Invokes with `ainvoke_with_retry()` and extracts findings via `_extract_findings()`

**Key insight:** Explore nodes use `build_phase_agent()` with subagent specs, NOT full subgraphs. They're lightweight agent invocations.

### Synthesis Consumption of Aggregated Results

**_synthesize_plan (_synthesize_specify) (lines 157-354 of exploration_subgraph.py):**

```python
findings_text = _format_findings(findings)  # Formats all accumulated findings
prompt = (
    f"{rework_prefix}Create a detailed technical plan...\n\n"
    f"## Work Description\n{description}\n\n"
    f"## Codebase Research Findings\n{findings_text}\n\n"  # Findings injected here
    ...
)
```

The synthesizer receives all findings accumulated via `operator.add` reducer on the `findings` field.

### State Schema for Exploration

**ExplorationSubgraphState (lines 93-123 of subgraph_state.py):**
```python
class ExplorationSubgraphState(BaseSubgraphState, total=False):
    # Exploration loop control
    research_round: int  # Current round number (0-based)
    max_rounds: int  # Safety valve — max exploration rounds (default 3)
    manager_decision: str  # "explore" | "done" — set by research_manager

    # Accumulated research (operator.add reducer merges per-round findings)
    topics: Annotated[list[str], _op_add]  # Areas being explored this round
    findings: Annotated[list[dict], _op_add]  # ResearchFindings dicts from explore nodes

    # Synthesis output
    agent_response: str  # Final spec/plan text from synthesizer

    # Phase Completion Invariants
    exploration_happened: bool  # True when research rounds executed
    synthesis_completed: bool  # True when synthesizer produced valid output

    # PLAN-specific fields
    spec_path: str
    has_spec: bool
    plan_json: str  # Raw plan.json content (only for PLAN)
    execution_waves: list  # Computed execution waves (only for PLAN)
```

### Gaps Identified

1. **No structured output schema enforcement** - The research manager returns JSON but it's parsed manually with no Pydantic validation
2. **Findings accumulation relies on operator.add** - Works but no deduplication of topics
3. **No explicit state validation** - Invariants are documented but not enforced at runtime

---

## 2. CRITIC AGENT, ROUTER, AND GATES

### Files Examined
- `/home/pat/Projects/spine/spine/workflow/critic_review.py` (11,479 chars)
- `/home/pat/Projects/spine/spine/workflow/artifact_gate.py` (19,234 chars)
- `/home/pat/Projects/spine/spine/critic/agent.py` (5,259 chars)
- `/home/pat/Projects/spine/spine/workflow/subgraphs/critic_subgraph.py` (9,943 chars)
- `/home/pat/Projects/spine/spine/models/enums.py` (1,479 chars)

### Two Tiers of Critic Routing

**Tier 1: Structural Check (lines 56-101 of critic_review.py)**
- Fast, no-LLM check via `structural_critic_check()`
- Checks:
  - Artifacts exist for the reviewed phase
  - Artifact content is non-empty (≥50 chars)
  - Basic structure/length requirements
- Returns `ReviewStatus.PASSED` or `ReviewStatus.NEEDS_REVISION`
- **If structural fails**: Skip agent critic, go straight to rework

**Tier 2: Agent-Based Review (lines 104-196 of critic_review.py)**
- Deep LLM-based review via `agent_critic_check()`
- Builds critic agent via registry, invokes asynchronously
- Checks quality, completeness, actionability
- Catches exceptions and returns `NEEDS_REVISION` (not crash)

**Routing Logic (lines 231-308 of critic_review.py):**
```python
def critic_router(state: WorkflowState) -> str:
    if state.get("status") == "failed":
        return "failed"
    
    # Read last feedback (written by critic)
    last_review = feedback[-1]
    review_status = last_review.get("status")
    
    if review_status == ReviewStatus.PASSED.value:
        return "passed"
    if review_status == ReviewStatus.NEEDS_REVIEW.value:
        return "needs_review"
    
    # NEEDS_REVISION - check retry count
    phase_retries = state.get("retry_count", {}).get(reviewed_phase, 0)
    if phase_retries >= state.get("max_retries", 3):
        return "needs_review"  # Exhausted retries
    return "needs_revision"
```

### Artifact Gate Behavior

**Entry Conditions (lines 300-457 of artifact_gate.py):**
- Basic: `_has_meaningful_artifacts()` - checks state has artifacts ≥50 chars
- Plan→Implement specific: `_check_plan_quality()` validates:
  - `plan.json` exists and is valid JSON
  - `feature_slices` array is non-empty
  - Each slice has required fields: id, title, target_files, execution_requirements, dependencies, acceptance_criteria
- Tasks→Implement specific: `_check_tasks_quality()` validates:
  - `codebase-map.md` exists
  - At least one file path in slice-*.md exists in workspace

**Failure Behavior:**
- Returns `{"status": "needs_review", "feedback": [...]}` 
- Routes to `human_review` interrupt node
- **Never crashes** - all exceptions caught and logged

**Retry Semantics:**
- No built-in retry - gates are structural checks, not transient failures
- Failed gates immediately flag for human review

### Structured Output from Critic

**ReviewFeedback dataclass (lines 74-81 of types.py):**
```python
@dataclass
class ReviewFeedback:
    status: ReviewStatus  # PASSED | NEEDS_REVISION | NEEDS_REVIEW
    tier: str  # "structural" or "agent"
    reason: str
    suggestions: list[str] = field(default_factory=list)
```

The critic writes feedback to `state["feedback"]` as a list (accumulated via `operator.add`).

### Gates in Workflow

**Gate Placement (lines 734-747 of compose.py):**
```python
for (src, dst), required_phase in gate_edges.items():
    gate_name = _gate_node_name(src, dst)
    graph.add_node(
        gate_name,
        make_artifact_gate_node(required_phase, dst),
    )
    graph.add_conditional_edges(
        gate_name,
        artifact_gate_router,
        {
            "proceed": dst,
            "needs_review": "human_review",
        },
    )
```

Current gates:
- `gate_tasks_to_implement` - requires plan artifacts
- `gate_plan_to_implement` - requires structured plan with feature_slices

### Repeated Critic Failure Behavior

When critic returns `NEEDS_REVISION`:
1. `critic_router` checks `state["retry_count"][reviewed_phase]`
2. If retries exhausted (default 3), returns `"needs_review"` 
3. Routes to `human_review` interrupt

**No automatic escalation** - human review is the terminal state for persistent failures.

---

## 3. STRUCTURED DATA HANDOFFS

### Files Examined
- `/home/pat/Projects/spine/spine/models/types.py` (5,457 chars)
- `/home/pat/Projects/spine/spine/models/state.py` (4,694 chars)
- `/home/pat/Projects/spine/spine/agents/subagents.py` (33,442 chars)
- `/home/pat/Projects/spine/spine/agents/artifacts.py` (16,270 chars)
- `/home/pat/Projects/spine/spine/agents/tool_schema_validator.py` (13,618 chars)

### Structured Types for Data Passing

**Pydantic Models (types.py):**
- `WorkUnit` - spawned work item with title, description, priority
- `PlanDecomposition` - list of WorkUnit with validation
- `FeatureSlice` - implementation slice with target_files, dependencies, acceptance_criteria
- `StructuredPlan` - JSON-serializable plan with feature_slices array
- `ReviewFeedback` - critic output (dataclass, not Pydantic)

**TypedDict State Schemas (state.py, subgraph_state.py):**
- `WorkflowState` - main graph state with reducers for accumulation
- `ExplorationSubgraphState` - exploration loop state with operator.add for findings
- `CriticSubgraphState` - critic review state

### Structured Output Usage

**Subagent Response Format (lines 45-82 of subagents.py):**
```python
class ResearchFindings(BaseModel):
    summary: str = Field(description="Concise summary (2-3 paragraphs)")
    patterns: list[str] = Field(description="Notable patterns discovered")
    file_map: dict[str, str] = Field(description="File paths to descriptions")
    dependencies: list[str] = Field(description="Key dependencies found")
```

**Usage in subagent spec (lines 583-594 of subagents.py):**
```python
# Only researcher gets response_format — structured summaries prevent bloating
if name == "researcher" and _supports_forced_tool_choice(model):
    spec["response_format"] = SUBAGENT_RESPONSE_MODELS[name]
```

**Critical limitation:** Response format is skipped for models in `_THINKING_MODEL_PATTERNS` (Qwen3, QwQ, DeepSeek-R1) because they reject `tool_choice="any"`.

### Artifact Materialization

**From Agent Output to Disk (artifacts.py):**
1. Agents write files via `write_file` to `.spine/artifacts/{work_id}/{phase}/`
2. `scan_artifact_dir()` discovers files after agent completes
3. `materialize_artifacts()` writes prior phase artifacts to disk before new phase starts
4. `build_artifact_prompt()` generates paths for agents to read (not full content)

**From Disk to State:**
- `scan_artifact_dir()` → truncated previews stored in `WorkflowState["artifacts"]`
- `_MAX_ARTIFACT_STATE_CHARS = 500` - only previews stored in state

### Current Handoff Mechanism

**Subgraph → Parent State (subgraph_wrapper.py lines 319-348):**
```python
def make_success_result_mapper(phase: str):
    def map_success(subgraph_result, parent_state):
        artifacts = subgraph_result.get("artifacts_output", {})
        # Truncated previews only
        artifact_previews = {name: content[:_MAX_ARTIFACT_STATE_CHARS] ...}
        return {
            "artifacts": {phase: artifact_previews},
            "phase_results": {phase: {"phase": phase, "status": "success", ...}},
        }
    return map_success
```

**Gap:** Handoffs are primarily **unstructured text** (agent_response) that gets saved to disk, then scanned back. No first-class Pydantic model passing between phases.

---

## 4. STATE INVARIANTS

### Files Examined
- `/home/pat/Projects/spine/spine/models/state.py` (4,694 chars)
- `/home/pat/Projects/spine/spine/workflow/subgraph_state.py` (4,609 chars)
- `/home/pat/Projects/spine/spine/docs/state_invariants.md` (5,754 chars)
- `/home/pat/Projects/spine/spine/workflow/compose.py` (38,320 chars)

### Enforced Invariants

**WorkflowState Invariants (lines 100-105 of state.py):**
```python
gap_plan_produced: bool  # True when gap_plan.md was successfully created
exploration_executed: bool  # True when SPECIFY/PLAN research rounds ran
```

**Subgraph State Invariants (subgraph_state.py):**

*ExplorationSubgraphState:*
```python
exploration_happened: bool  # True when research rounds executed (vs. skipped)
synthesis_completed: bool  # True when synthesizer produced valid output
```

*ImplementSubgraphState:*
```python
slices_dispatched: bool  # True when slice-implementers were dispatched
implementation_files_written: bool  # True when code files were created
```

*VerifySubgraphState:*
```python
verification_attempted: bool  # True when verify agent ran (vs. skipped)
verification_passed: bool  # True when verification confirmed passing
```

### Subgraph State Mappers

**State Mapper Pattern (subgraph_wrapper.py):**
```python
def make_subgraph_node(subgraph, phase_name, state_mapper, result_mapper, ...):
    async def subgraph_node(parent_state, config):
        subgraph_input = state_mapper(parent_state, config)  # Parent → Subgraph
        result = await active_subgraph.ainvoke(subgraph_input, ...)
        parent_update = result_mapper(result, parent_state)  # Subgraph → Parent
        return parent_update
```

**Example (from compose.py lines 626-674):**
```python
_STATE_MAPPERS = {
    PhaseName.VERIFY.value: _verify_state_mapper,
    PhaseName.IMPLEMENT.value: _implement_state_mapper,
    ...
}
```

### Validation Hooks

**Current State:**

1. **No runtime invariant validation** - The Documented patterns in `state_invariants.md` are NOT implemented in the code. The invariant fields exist but are not set in the actual result mappers.

2. **Structural validation via artifact_gate.py** - Validates artifact presence/content but doesn't check invariant flags.

3. **Reducer-based accumulation** - State fields with `Annotated` reducers automatically merge:
   - `operator.add` for lists (feedback, spawned_work_ids)
   - `_merge_dicts` for dicts (retry_count, phase_results)
   - `_merge_artifacts` for nested artifact dicts

### Where Invariants Should Be Checked

Per `state_invariants.md`, the integration points are:

1. **GAP_PLAN phase**: Set `gap_plan_produced` in `_gap_plan_result_mapper`
2. **Exploration subgraph**: Set `exploration_happened` and `synthesis_completed` in `_save_exploration_artifacts`
3. **IMPLEMENT phase**: Set `slices_dispatched` and `implementation_files_written` in implement result mapper
4. **VERIFY phase**: Set `verification_attempted` and `verification_passed` in verify result mapper

**Current Status:** These are NOT implemented. The code has the fields but doesn't populate them.

---

## Summary of Key Findings

| Area | Current State | Gap/Issue |
|------|---------------|-----------|
| Explore loop | Fully implemented with Send API parallelism | No Pydantic validation of manager JSON output |
| Structured handoffs | ResearchFindings model exists for researchers | No structured handoff between phases; text→disk→scan |
| Critic routing | Two-tier (structural + agent) working | No automatic retry escalation beyond human review |
| Artifact gates | Structural and quality checks implemented | No built-in retry; immediate human review on failure |
| Invariants | Fields exist in state but not populated | No runtime validation; documented but not implemented |