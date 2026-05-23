# SPINE Codebase Analysis Report

## Section 1: Explore→Aggregate Loop

### How the Manager/Router/Explore Loop Works

**File: `/home/pat/Projects/spine/spine/workflow/subgraphs/exploration_subgraph.py`** (lines 1-547)

The exploration subgraph is a multi-node research loop with the following structure:

```
START → research_manager → [Send(explore) × N] → aggregate → sufficiency_router → [loop OR done]
                                                                 ↓
                                                           synthesize → save_artifacts → END
```

**Key Nodes:**

1. **`research_manager` node** (`_research_manager_node`, lines 52-64):
   - Single LLM call via `run_research_manager()` to decide next topics or done
   - Returns: `{"manager_decision": "explore"|"done", "topics": [...], "research_round": N}`
   - Max rounds safety valve (default 3) prevents infinite loops

2. **`_research_router`** (lines 70-89):
   - Conditional edge function that routes based on manager decision
   - Returns `Send("explore", {"topic": t}) × N` for parallel dispatch via LangGraph Send API
   - Or returns `"synthesize"` string when research is complete

3. **`explore` node** (`_explore_node`, lines 95-109):
   - Invokes a researcher subagent for each topic sent via Send API
   - Uses `build_subagent_spec("researcher", ...)` + `build_phase_agent` machinery
   - Returns `{"findings": [...]}` where findings accumulate via `operator.add`

4. **`aggregate` node** (`_aggregate_node`, lines 115-128):
   - Fan-in point after all parallel explore nodes complete
   - Findings are accumulated via `operator.add` reducer on the `findings` field
   - No manual merging needed - state reducer handles accumulation

5. **`_sufficiency_router`** (lines 134-151):
   - Checks if research is sufficient to proceed to synthesis
   - Returns `"loop"` (research_manager) or `"done"` (synthesize)

### Structured Data Passing Between Manager and Router

**File: `/home/pat/Projects/spine/spine/workflow/subgraph_state.py`** (lines 93-124)

```python
class ExplorationSubgraphState(BaseSubgraphState, total=False):
    research_round: int           # Current round number (0-based)
    max_rounds: int              # Safety valve — max exploration rounds
    manager_decision: str        # "explore" | "done" — set by research_manager
    topics: Annotated[list[str], _op_add]  # Areas being explored
    findings: Annotated[list[dict], _op_add]  # ResearchFindings dicts
    agent_response: str          # Final spec/plan text from synthesizer
    exploration_happened: bool   # True when research rounds executed
    synthesis_completed: bool    # True when synthesizer produced valid output
```

The manager passes `manager_decision` and `topics` to the router, which uses them to either dispatch parallel explore nodes or proceed to synthesis.

### How Explore Agents Are Invoked

**File: `/home/pat/Projects/spine/spine/agents/exploration_agents.py`** (lines 192-333)

The `run_explore_node` function:
1. Builds a researcher subagent via `build_subagent_spec()` + `build_phase_agent()`
2. Injects spec content for PLAN phase (lines 262-293)
3. Uses `ainvoke_with_retry()` for resilient execution
4. Extracts findings from `structured_response` key if available (lines 336-372)

**File: `/home/pat/Projects/spine/spine/agents/subagents.py`** (lines 45-83, 292-296)

```python
class ResearchFindings(BaseModel):
    summary: str = Field(description="Concise summary of findings")
    patterns: list[str] = Field(description="Notable patterns, conventions, idioms")
    file_map: dict[str, str] = Field(description="Important file paths")
    dependencies: list[str] = Field(description="Key dependencies found")

SUBAGENT_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "researcher": ResearchFindings,
    ...
}
```

### Exploration State Schema

**File: `/home/pat/Projects/spine/spine/workflow/subgraph_state.py`** (lines 93-124)

See the `ExplorationSubgraphState` class above. Key fields:
- `research_round`, `max_rounds`, `manager_decision` - loop control
- `topics`, `findings` - accumulated via `operator.add`
- `spec_path`, `has_spec`, `plan_json`, `execution_waves` - PLAN-specific

---

## Section 2: Critic & Gates

### Two-Tier Critic Routing

**File: `/home/pat/Projects/spine/spine/workflow/critic_review.py`** (lines 1-308)

The critic has two tiers:
1. **Structural (fast, no-LLM)**: `structural_critic_check()` (lines 56-101)
   - Checks artifacts exist, aren't empty, have basic structure
   - Returns `{"status": ReviewStatus, "tier": "structural", "reason", "suggestions"}`

2. **Agent (deep, LLM-based)**: `agent_critic_check()` (lines 104-196)
   - Delegates to critic Deep Agent for quality review
   - Exceptions → `NEEDS_REVISION`, not crashes

**File: `/home/pat/Projects/spine/spine/workflow/subgraphs/critic_subgraph.py`** (lines 1-271)

Subgraph flow:
```
START → structural_check → [passed → plan_validation (if PLAN) OR agent_check]
                              [needs_revision/needs_review → END]
                           → plan_validation → [passed → agent_check]
                                                 [needs_revision/needs_review → END]
                         → agent_check → END
```

### Gates

**File: `/home/pat/Projects/spine/spine/workflow/artifact_gate.py`** (lines 1-475)

Gates exist for:
- **plan→implement transition** (lines 300-461):
  - `codebase-map.md` must exist
  - At least one file path in slice-*.md must exist in workspace
  - Checks via `_check_tasks_quality()` (lines 69-170)

- **plan→implement transition** (lines 388-419):
  - `plan.json` must exist and be valid JSON
  - `feature_slices` array must be non-empty
  - Each slice must have required fields (id, title, target_files, etc.)
  - Checks via `_check_plan_quality()` (lines 184-297)

**Gate Node Structure** (lines 300-461):
- Gates are **nodes**, not conditional edge functions
- When gate passes: returns `{"status": "running"}` → routes to next phase
- When gate fails: returns `{"status": "needs_review", "feedback": [...]}` → routes to human_review

### Repeated Critic Failures

**File: `/home/pat/Projects/spine/spine/workflow/critic_review.py`** (lines 274-308)

`_handle_review_outcome()` handles repeated failures:
1. If `PASSED` → return `"passed"`
2. If `NEEDS_REVIEW` → return `"needs_review"` (human review)
3. If `NEEDS_REVISION`:
   - Check `retry_count[reviewed_phase] >= max_retries` (default 3)
   - If exceeded → return `"needs_review"` (escalate to human)
   - If not exceeded → return `"needs_revision"` (rework loop)

**File: `/home/pat/Projects/spine/spine/workflow/compose.py`** (lines 282-327)

`_critic_result_mapper()` increments retry count in state for both `NEEDS_REVISION` and `NEEDS_REVIEW` outcomes.

---

## Section 3: Structured Data Handoffs

### Pydantic BaseModel Classes

**File: `/home/pat/Projects/spine/spine/models/types.py`** (lines 1-176)

```python
@dataclass
class FeatureSlice:
    id: str
    title: str
    target_files: list[str]
    execution_requirements: list[str]
    dependencies: list[str]
    acceptance_criteria: list[str]
    complexity: str = "small"

@dataclass
class StructuredPlan:
    architecture_overview: str
    technology_choices: list[str]
    feature_slices: list[FeatureSlice]
    testing_strategy: str
    risks: list[str]
    codebase_map: dict[str, Any]

class WorkUnit(BaseModel):
    title: str
    description: str
    priority: str = "medium"
    is_critical: bool = False

class PlanDecomposition(BaseModel):
    units: list[WorkUnit]
```

**File: `/home/pat/Projects/spine/spine/agents/subagents.py`** (lines 45-83)

```python
class ResearchFindings(BaseModel):
    summary: str
    patterns: list[str]
    file_map: dict[str, str]
    dependencies: list[str]

class SliceResult(BaseModel):
    status: str  # "implemented" | "partial" | "blocked"
    files_modified: list[str]
    files_created: list[str]
    test_results: str
    issues: list[str]

class CheckItem(BaseModel):
    criterion: str
    passed: bool
    detail: str

class VerificationResult(BaseModel):
    verdict: str  # "VERIFIED" | "NOT_VERIFIED"
    checklist: list[CheckItem]
    gaps: list[str]
    recommendations: list[str]
```

### Are Structured Outputs Currently Used?

**File: `/home/pat/Projects/spine/spine/agents/factory.py`** (lines 235-387)

Yes, via Deep Agents' `response_format` parameter:

```python
def build_phase_agent(..., response_format: Any | None = None, ...):
    ...
    response_format=response_format,  # Passed to create_deep_agent
```

**File: `/home/pat/Projects/spine/spine/agents/subagents.py`** (lines 292-296, 584-594)

Only the **researcher** subagent gets `response_format` (structured summaries):
```python
SUBAGENT_RESPONSE_MODELS = {
    "researcher": ResearchFindings,
    "slice-implementer": SliceResult,
    "slice-verifier": VerificationResult,
}

# Line 584-594: Researcher gets response_format; others skip it
if _supports_forced_tool_choice(model) and name == "researcher":
    spec["response_format"] = SUBAGENT_RESPONSE_MODELS[name]
```

**Note:** Thinking models (Qwen3, QwQ, DeepSeek-R) reject `tool_choice="any"` and skip response_format (lines 409-476).

### How Handoffs Work Today

**File: `/home/pat/Projects/spine/spine/workflow/subgraph_wrapper.py`** (lines 188-309)

Handoffs use `make_subgraph_node()`:
1. **State mapper** (lines 232): Maps parent `WorkflowState` → subgraph input
   - `_specify_state_mapper()`, `_plan_state_mapper()`, etc. (compose.py lines 126-181)
2. **Subgraph invocation** (lines 273-277): Compiled subgraph runs with its own checkpointer
3. **Result mapper** (lines 280): Maps subgraph output → parent state update
   - `make_success_result_mapper()` for standard phases (lines 312-349)
   - Custom mappers for verify, implement, critic (compose.py lines 200-279)

**File: `/home/pat/Projects/spine/spine/agents/artifacts.py`** (lines 100-154, 347-389)

Artifacts are materialized to disk:
- `materialize_artifacts()` (lines 100-154): Writes all prior phase artifacts to `.spine/artifacts/{work_id}/{phase}/`
- `build_inline_artifact_prompt()` (lines 347-389): Provides inline previews for critic
- `build_artifact_prompt()` (lines 229-300): Provides path references (not content) to save tokens

---

## Section 4: State Invariants

### State Fields Today

**File: `/home/pat/Projects/spine/spine/models/state.py`** (lines 60-105)

```python
class WorkflowState(TypedDict, total=False):
    work_id: str
    work_type: str
    description: str
    current_phase: str
    phase_index: int
    retry_count: Annotated[dict, _merge_dicts]
    max_retries: int
    artifacts: Annotated[dict, _merge_artifacts]
    feedback: Annotated[list, operator.add]
    status: str
    prompt_request: dict | None
    critic_reviewing: str
    workspace_root: str
    phase_results: Annotated[dict, _merge_dicts]
    needs_review_phase: str | None
    plan_id: str | None
    spawned_work_ids: Annotated[list[str], operator.add]
    execution_waves: list[list[dict]]
    verify_attempts: int
    gap_plan_produced: bool  # Phase completion invariant
    exploration_executed: bool  # Phase completion invariant
```

**Reducers:**
- `_merge_dicts` (lines 11-18): Shallow merge with right overwriting left
- `_merge_artifacts` (lines 21-43): Deep merge at file level to prevent loss

**File: `/home/pat/Projects/spine/spine/workflow/subgraph_state.py`** (lines 15-123)

Subgraph state schemas extend `BaseSubgraphState` (lines 15-29) with phase-specific fields.

### How Subgraph Mappers Work

**File: `/home/pat/Projects/spine/spine/workflow/compose.py`** (lines 112-327)

State mappers (lines 112-181):
```python
def _base_state_mapper(parent_state, config):
    return {
        "work_id": parent_state.get("work_id", "unknown"),
        "work_type": parent_state.get("work_type", ""),
        "description": parent_state.get("description", ""),
        "workspace_root": parent_state.get("workspace_root", "."),
        "feedback": parent_state.get("feedback", []),
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
    }
```

Result mappers (lines 200-279):
- `make_success_result_mapper()` (subgraph_wrapper.py lines 312-349): Standard success path
- Custom mappers for verify, implement, tasks, specify, plan, critic handle phase-specific logic

### Where Invariants Are Checked

**File: `/home/pat/Projects/spine/spine/models/state.py`** (lines 100-105)

```python
# Phase Completion Invariants (prevent rework misinterpretation)
gap_plan_produced: bool  # True when gap_plan.md was successfully created
exploration_executed: bool  # True when SPECIFY/PLAN research rounds ran
```

**File: `/home/pat/Projects/spine/spine/workflow/subgraph_state.py`** (lines 58-72, 113-117)

```python
# ImplementSubgraphState (lines 58-61)
slices_dispatched: bool  # True when slice-implementers were dispatched
implementation_files_written: bool  # True when code files were created

# VerifySubgraphState (lines 71-72)
verification_attempted: bool  # True when verify agent ran
verification_passed: bool  # True when verification confirmed passing

# ExplorationSubgraphState (lines 113-117)
exploration_happened: bool  # True when research rounds executed
synthesis_completed: bool  # True when synthesizer produced valid output
```

These invariants are set by nodes and checked by result mappers to prevent misinterpreting empty/failed artifacts as intentionally empty work.

---

## Summary

The SPINE codebase implements a sophisticated multi-phase workflow with:

1. **Explore→Aggregate Loop**: Research manager makes decisions, parallel explore nodes investigate topics, aggregate collects findings via reducer, sufficiency router decides loop continuation

2. **Two-Tier Critic**: Fast structural checks before expensive LLM review, with escalation to human review after retry limits

3. **Artifact Gates**: Structural validation ensuring prerequisite artifacts exist before proceeding (plan→implement)

4. **Structured Data Handoffs**: Pydantic models used for researcher and slice results, but only researcher gets `response_format` due to model compatibility issues

5. **State Invariants**: Boolean flags track phase completion to prevent rework misinterpretation, enforced via result mappers