# SPINE Combined Implementation Plan
## Structured Data Handoffs, State Invariants, Prompt Optimization & Fail-Closed Gates

**Version:** 1.0  
**Date:** 2026-05-23  
**Status:** Draft for Review  
**Supersedes:**
- `specify_prompt_optimization_plan.md`
- `plan_synth_optimization_report.md`
- `implement_orch_prompt_plan.md`
- `verify_prompt_analysis_report.md`
- `.hermes/plans/2026-05-23-constrained-decoding-fail-closed-invariants.md`
- `state_invariants.md`

---

## 1. Architecture Context — How the Agent Graph Works Today

### 1.1 Phase Graph Topology

The SPINE workflow is a LangGraph StateGraph with per-phase subgraph isolation. Each phase is wrapped as a `make_subgraph_node()` with its own TypedDict state, SQLite checkpointer, state mapper (parent→subgraph), and result mapper (subgraph→parent).

```
START → SPECIFY → [critic_specify*] → PLAN → critic_plan → [gate] → IMPLEMENT → VERIFY → END
                                                                                ↓
                                                                          GAP_PLAN → IMPLEMENT → VERIFY (loop, max 2 gap cycles)
```

*\* critical_task/critical_reviewed types only*

**Gate:** Only `plan→implement` is gated today. The gate is a **node** (not edge), validates `plan.json` has valid `feature_slices`, and routes to `human_review` on failure.

### 1.2 Explore→Aggregate Loop (SPECIFY + PLAN)

Only SPECIFY and PLAN use the multi-node exploration subgraph (feature-flagged via `_USE_EXPLORATION_SUBGRAPH`):

```
START → research_manager → [Send(explore, {topic}) × N] → aggregate → sufficiency_router
                              ↑____________________________________________↓ (loop)
                              ↓ (done)
                           synthesize → save_artifacts → END
```

**Manager**: Single LLM call (`run_research_manager`). No tools, no agent loop. Returns structured JSON: `{"decision": "explore"|"done", "topics": ["area1", "area2"]}`

**Router** (`_research_router`): Reads `manager_decision` and `topics` from state. On `"explore"`, returns `[Send("explore", {"topic": t}) for t in topics]` — LangGraph executes all in parallel within one super-step. On `"done"`, returns `"synthesize"` string.

**Explore** (`_explore_node`): Builds a researcher subagent via `build_subagent_spec("researcher", ...)` + `build_phase_agent(...)`. Findings accumulate via `operator.add` reducer on `findings: Annotated[list[dict], _op_add]`.

**Aggregate** (`_aggregate_node`): Fan-in point. Findings already merged by reducer. No-op routing checkpoint.

**Sufficiency Router** (`_sufficiency_router`): Checks `manager_decision == "done"` OR `round_num >= max_rounds` → `"done"` (synthesize), else → `"loop"` (research_manager).

**Synthesize** (`_synthesize_specify`/`_synthesize_plan`): Builds the phase agent (specify/plan) with research findings as context. For PLAN, additionally reads `plan.json` from disk and computes `execution_waves`.

### 1.3 Critic System

**Two Tiers:**

1. **Structural** (`structural_critic_check`): No LLM. Checks artifacts exist, are ≥50 chars. Returns `{status, tier: "structural", reason, suggestions}`.

2. **Agent** (`agent_critic_check`): Deep Agent with `ainvoke_with_retry`. Materializes artifacts to disk, provides inline preview + paths. Parses response for PASSED/NEEDS_REVISION/NEEDS_REVIEW keywords. Exceptions → NEEDS_REVISION (fail-safe).

**Routing** (`_handle_review_outcome`):
- PASSED → `"passed"` (proceed to next phase)
- NEEDS_REVIEW → `"needs_review"` (human review)
- NEEDS_REVISION + retries < max (3) → `"needs_revision"` (rework loop)
- NEEDS_REVISION + retries ≥ max → `"needs_review"` (escalate)

**Critic is a subgraph** (`critic_subgraph.py`):
```
START → structural_check → [passed → agent_check → END]
                              [needs_revision/needs_review → END]
```

### 1.4 Current Structured Data Handoffs

| What | Pydantic Model | Gets `response_format`? | Where |
|------|---------------|------------------------|-------|
| Research findings | `ResearchFindings` (summary, patterns, file_map, dependencies) | **YES** (researcher only) | `subagents.py:594` |
| Slice implementation | `SliceResult` (status, files_modified, files_created, test_results, issues) | **NO** | Model defined but skipped |
| Slice verification | `VerificationResult` (verdict, checklist: list[CheckItem], gaps, recommendations) | **NO** | Model defined but skipped |
| Manager decision | Raw JSON `{"decision":"explore"|"done","topics":[...]}` | N/A (single LLM call) | `exploration_agents.py:44` |
| Plan decomposition | `StructuredPlan` dataclass + `FeatureSlice` dataclass | N/A (parsed from disk) | `types.py` |
| Phase results | `PhaseResult` TypedDict | N/A (set by mappers) | `state.py` |

**Key Gap:** SliceResult and VerificationResult have Pydantic models defined but DON'T get `response_format` — they rely on free-text parsing. Only researcher gets structured output, and only after a model compatibility check (`_supports_forced_tool_choice`).

### 1.5 Current State Invariants

| Field | Scope | Set By | Current Status |
|-------|-------|--------|---------------|
| `gap_plan_produced` | Parent `WorkflowState` | `_gap_plan_result_mapper` | ✅ EXISTS |
| `exploration_executed` | Parent `WorkflowState` | (unclear — need to verify) | ✅ EXISTS |
| `slices_dispatched` | `ImplementSubgraphState` | implement save_artifacts | ✅ EXISTS (subgraph-local) |
| `implementation_files_written` | `ImplementSubgraphState` | implement save_artifacts | ✅ EXISTS (subgraph-local) |
| `verification_attempted` | `VerifySubgraphState` | verify save_artifacts | ✅ EXISTS (subgraph-local) |
| `verification_passed` | `VerifySubgraphState` | verify save_artifacts | ✅ EXISTS (subgraph-local) |
| `exploration_happened` | `ExplorationSubgraphState` | save_artifacts | ✅ EXISTS (subgraph-local) |
| `synthesis_completed` | `ExplorationSubgraphState` | save_artifacts | ✅ EXISTS (subgraph-local) |
| `spec_completed` | Parent `WorkflowState` | — | ❌ MISSING |
| `plan_completed` | Parent `WorkflowState` | — | ❌ MISSING |
| `implement_completed` | Parent `WorkflowState` | — | ❌ MISSING |
| `verify_completed` | Parent `WorkflowState` | — | ❌ MISSING |
| `tasks_completed` | Parent `WorkflowState` | — | ❌ MISSING |

**Naming Conflict:** Document 5 uses `<phase>_completed` convention; Document 6 uses `gap_plan_produced`, `exploration_executed`. **Resolution:** Adopt `<phase>_completed` for parent state, keep semantic names for subgraph-local invariants.

---

## 2. Structured Data Handoff Design — All Agents

### 2.1 Principle: Every Agent Handoff Must Be Structured

Following the success of the `ResearchFindings` model with `response_format`, **every phase's output and every subagent's return must use typed Pydantic models with `response_format`** where the inference engine supports it.

### 2.2 SPECIFY Phase

**Data In:** Work description, exploration findings (`ResearchFindings[]`)  
**Data Out:** `specification.md` (prose) + `specification.json` (structured)

**New Pydantic Model:**
```python
class Specification(BaseModel):
    """Structured specification output from SPECIFY phase."""
    title: str
    summary: str  # Executive summary (2-3 sentences)
    objectives: list[str]  # High-level goals
    requirements: list[str]  # Functional requirements
    constraints: list[str]  # Non-functional constraints
    scope_inclusions: list[str]
    scope_exclusions: list[str]
    known_risks: list[str]
```

**Implementation:**
1. Add `Specification` model to `spine/models/types.py`
2. Extend `WriteSpecificationInput` in `specify_tools.py` to accept `specification_json: str | None`
3. Update `write_specification` tool to write both `specification.md` and `specification.json`
4. Update `_synthesize_specify` to include JSON output instruction in prompt
5. Add `specification` field to `SpecifySubgraphState`
6. Update `_specify_result_mapper` to propagate `specification` to parent state

**Agent Handoff:** SPECIFY → PLAN: PLAN receives `specification.json` via state mapper (`spec_path` already exists)

### 2.3 PLAN Phase

**Data In:** `specification.json`, exploration findings (`ResearchFindings[]`)  
**Data Out:** `plan.md` (prose) + `plan.json` (structured — already exists)

**Current state:** `StructuredPlan` dataclass + `FeatureSlice` dataclass already exist. `plan.json` is already written by `write_structured_plan`. **The structured handoff already works** — no new model needed.

**Enhancements:**
1. Add complexity validation to `FeatureSlice`: `complexity` must be one of `["small", "medium", "large"]`
2. Add explicit DAG validation to `write_structured_plan` tool: verify `dependencies` reference valid `slice.id` values
3. Add `min_length` validation to slice fields (`title` ≥ 3 chars, `target_files` non-empty)

**Agent Handoff:** PLAN → IMPLEMENT: IMPLEMENT receives `execution_waves` (pre-computed from `plan.json` by the synthesizer) + `plan.json` on disk. **Already structured.**

### 2.4 IMPLEMENT Phase

**Data In:** `execution_waves` (list of wave dicts), `plan.json` on disk, `gap_plan_path` (optional)  
**Data Out:** `implementation.md` (prose) + source files on disk

**Current state:** Slice-implementer subagents produce results but `SliceResult` model doesn't get `response_format`. The orchestrator uses `write_implementation_report` which already accepts typed slice results.

**Enhancements:**
1. **Enable `response_format=SliceResult`** for slice-implementer subagents (remove the model compatibility skip)
2. Add `_collect_slice_results` to `ImplementSubgraphState` — accumulates slice results as typed objects
3. Update `_implement_result_mapper` to set `implement_completed: True` in parent state
4. Add wave-level status tracking: `wave_status: dict[int, str]` to `ImplementSubgraphState`

**Agent Handoff:** IMPLEMENT → VERIFY: VERIFY receives `implementation.md` + source files on disk. **Already structured** via `write_implementation_report`.

### 2.5 VERIFY Phase

**Data In:** `implementation.md`, `plan.json` on disk, slice files  
**Data Out:** `verification.md` (prose) + structured verification results

**Current state:** `VerificationResult` model exists but doesn't get `response_format`. `write_verification_report` accepts typed `_VerificationResult` objects.

**Enhancements:**
1. **Enable `response_format=VerificationResult`** for slice-verifier subagents
2. Add `verification_findings` list to `VerifySubgraphState` for programmatic access
3. Update `_verify_result_mapper` to set `verification_passed: bool` in parent state
4. Add `verify_completed: bool` to parent `WorkflowState`

**Agent Handoff:** VERIFY → GAP_PLAN (on failure): GAP_PLAN receives structured verification findings. **Critical enhancement** — today it's just `verification.md` markdown.

### 2.6 GAP_PLAN Phase (Highest Priority)

**Data In:** `verification.md`, structured verification failures, `plan.json`, `implementation.md`  
**Data Out:** `gap_plan.md` (currently unstructured) → should be `gap_plan.json` + `gap_plan.md`

**Current state:** Uses generic filesystem tools, no subagents, unstructured output. **Largest gap.**

**New Pydantic Model:**
```python
class FixInstruction(BaseModel):
    """Structured fix instruction for one gap."""
    slice_id: str
    file_path: str
    change_type: Literal["add", "modify", "delete"]
    specific_change: str  # Precise description of what to change
    acceptance_criteria: list[str]
    estimated_complexity: Literal["small", "medium", "large"] = "small"

class GapPlan(BaseModel):
    """Structured gap plan output."""
    verification_summary: str
    gaps_identified: int
    fix_instructions: list[FixInstruction]
    re_verify_slices: list[str]  # Slice IDs that need re-verification
```

**Implementation:**
1. Add `FixInstruction` + `GapPlan` to `spine/models/types.py`
2. Create `spine/agents/gap_plan_tools.py` with:
   - `read_verification_findings` — reads `verification.md` and extracts structured failures
   - `write_structured_gap_plan` — writes `gap_plan.json` + `gap_plan.md`
3. Update `build_gap_plan_agent()` to use purpose-built tools + `skip_filesystem_middleware=True`
4. Add `gap_plan_json` field to `GapPlanSubgraphState`
5. Update `_gap_plan_result_mapper` to set `gap_plan_produced` and propagate structured data
6. Add `gap_plan_completed` to parent `WorkflowState`

**Multi-node pattern consideration:** Could decompose by verification failure slices:
```
analyze_failures → [Send(plan_fix, {slice_id}) × N] → synthesize
```
Defer to future iteration — single agent with structured tools is sufficient first step.

### 2.7 CRITIC Phase

**Data In:** Phase artifacts, inline previews  
**Data Out:** `{status, tier, reason, suggestions}` (already structured)

**Current state:** Critic parses LLM response for PASSED/NEEDS_REVISION/NEEDS_REVIEW keywords. **Robust but imprecise** — relies on string matching.

**Enhancement:**
```python
class CriticReview(BaseModel):
    """Structured critic output."""
    status: Literal["PASSED", "NEEDS_REVISION", "NEEDS_REVIEW"]
    tier: Literal["structural", "agent"]
    reason: str
    suggestions: list[str]
    score: int | None = None  # Optional 1-10 quality score
```

Add `response_format=CriticReview` to critic agent. This eliminates fragile keyword parsing in `_parse_agent_review()`.

### 2.8 Structured Handoff Summary Table

| Handoff | From | To | Data | Format | Status |
|---------|------|----|------|--------|--------|
| Exploration → Synthesis | research_manager | synthesize | `ResearchFindings[]` | Pydantic + `add` reducer | ✅ Done |
| SPECIFY → PLAN | specify subgraph | plan subgraph | `specification.json` | `Specification` Pydantic | ❌ NEW |
| PLAN → IMPLEMENT | plan subgraph | implement subgraph | `execution_waves` | `list[list[FeatureSlice]]` | ✅ Done |
| Slice implement → Orchestrator | slice-implementer | implement orchestrator | `SliceResult` | Pydantic, no response_format | 🔧 Enable response_format |
| IMPLEMENT → VERIFY | implement subgraph | verify subgraph | `implementation.md` + files | Disk artifacts | ✅ Done |
| Slice verify → Orchestrator | slice-verifier | verify orchestrator | `VerificationResult` | Pydantic, no response_format | 🔧 Enable response_format |
| VERIFY → GAP_PLAN | verify subgraph | gap_plan subgraph | `GapPlan` | NEW Pydantic | ❌ NEW |
| GAP_PLAN → IMPLEMENT | gap_plan subgraph | implement subgraph | `gap_plan.json` | NEW Pydantic | ❌ NEW |
| Critic → Router | critic subgraph | critic_router | `CriticReview` | NEW Pydantic | ❌ NEW |
| Any phase → Parent state | subgraph | parent graph | `phase_results` | `PhaseResult` TypedDict | ✅ Done |

---

## 3. State Invariants — Complete Implementation

### 3.1 Parent State (`WorkflowState`) — Phase Completion Flags

Add to `spine/models/state.py`:

```python
class WorkflowState(TypedDict, total=False):
    # ... existing fields ...

    # ── Phase Completion Invariants ──
    # Track whether each phase completed successfully. These prevent
    # re-interpreting empty/failed artifacts as intentionally empty work.
    spec_completed: bool       # SPECIFY produced specification.md
    plan_completed: bool       # PLAN produced plan.md + plan.json
    tasks_completed: bool      # TASKS produced tasks.md + slice-*.md
    implement_completed: bool  # IMPLEMENT wrote source files
    verify_completed: bool     # VERIFY ran and produced verification.md
    critic_specify_completed: bool  # critic_specify ran (critical_task only)
    critic_plan_completed: bool     # critic_plan ran
    gap_plan_completed: bool   # GAP_PLAN produced gap_plan.md
    exploration_executed: bool # SPECIFY/PLAN research rounds ran (existing)
```

### 3.2 Result Mapper Updates

Every result mapper must set its phase's completion flag:

| Result Mapper | Sets Flag | Condition |
|---------------|-----------|-----------|
| `_specify_result_mapper` | `spec_completed: True` | When `phase_status == "success"` |
| `_plan_result_mapper` | `plan_completed: True` | When `phase_status == "success"` |
| `_tasks_result_mapper` | `tasks_completed: True` | When `phase_status == "success"` |
| `_implement_result_mapper` | `implement_completed: True` | When `phase_status == "success"` and `implementation_files_written` |
| `_verify_result_mapper` | `verify_completed: True` | When `phase_status == "success"` |
| `_gap_plan_result_mapper` | `gap_plan_completed: True` | When `phase_status == "success"` |
| `_critic_result_mapper("specify")` | `critic_specify_completed: True` | When critic runs |
| `_critic_result_mapper("plan")` | `critic_plan_completed: True` | When critic runs |
| `_save_exploration_artifacts` | `exploration_executed: True` | When research rounds executed |

### 3.3 Invariant Validation Hook

Add to `spine/workflow/artifact_gate.py`:

```python
def validate_phase_prerequisites(
    state: WorkflowState,
    next_phase: str,
) -> tuple[bool, str]:
    """Validate that prerequisite phases completed before running next_phase."""
    prereqs = {
        PhaseName.PLAN.value: ["spec_completed"],
        PhaseName.IMPLEMENT.value: ["plan_completed", "tasks_completed"],
        PhaseName.VERIFY.value: ["implement_completed"],
        PhaseName.GAP_PLAN.value: ["verify_completed"],
    }
    required = prereqs.get(next_phase, [])
    for flag in required:
        if not state.get(flag, False):
            return False, f"Prerequisite '{flag}' not set — {flag.replace('_completed', '')} phase did not complete"
    return True, ""
```

Call this from `make_artifact_gate_node()` before the existing quality checks.

### 3.4 Subgraph-Local Invariants (Existing — No Changes)

These already exist and are correct:

| Subgraph | Field | Purpose |
|----------|-------|---------|
| `ImplementSubgraphState` | `slices_dispatched` | Track dispatch vs. skip |
| `ImplementSubgraphState` | `implementation_files_written` | Track file creation |
| `VerifySubgraphState` | `verification_attempted` | Track agent execution |
| `VerifySubgraphState` | `verification_passed` | Track outcome |
| `ExplorationSubgraphState` | `exploration_happened` | Track research execution |
| `ExplorationSubgraphState` | `synthesis_completed` | Track synthesis success |

---

## 4. Prompt Optimization — Per-Phase

### 4.1 Common Patterns Across All Phases

All phase prompts share these patterns that should be applied uniformly:

1. **AGENTS.md Removal**: Add all orchestrator phases to `_SKIP_AGENTS_MD` in `spine/agents/skills_resolver.py` (~5K tokens saved per phase):
   ```python
   _SKIP_AGENTS_MD = {
       "specify", "plan", "implement", "verify", "gap_plan", "tasks",
       "critic", "exploration",
   }
   ```

2. **Negative → Positive**: Rewrite all negative constraints ("DON'T do X", "NEVER do Y") as positive directives ("Do Z instead"). This is the single highest-impact prompt change for quantized models.

3. **Remove Hand-Written Tool Docs**: Tool schemas are auto-injected by DA — remove any hand-written tool listings from prompts. The agent already sees the schema.

4. **Remove MCP Tool Lists**: MCP tools auto-register — don't list them in prompts.

5. **Hyper-Literal Instructions**: Replace philosophical prose with numbered, step-by-step workflows using explicit conditionals.

### 4.2 SPECIFY Phase

**File:** `spine/agents/specify_agent.py` — `_build_specify_prompt()`

**Changes (P0):**
- Add SPECIFY to `_SKIP_AGENTS_MD` — ~5K tokens saved in one line
- Remove all negative instructions (30+ instances)
- Remove irrelevant sections: write_todos, skills system, generic task docs, code style
- Remove MCP tool list

**Changes (P1):**
- Restructure to hyper-literal numbered workflow with explicit conditionals
- Reduce researcher tool surface from 17 to 6 essential tools (~800 tokens)

**Changes (P2):**
- Add `Specification` model to types (see §2.2)
- Add structured `specification.json` output instruction

### 4.3 PLAN Phase

**File:** `spine/agents/plan_agent.py` — `_build_plan_prompt()`

**Changes (P0):**
- Remove embedded AGENTS.md content (lines 226-660 of prompt template) — already injected by `profile.py`
- Remove hand-written tool docs — schemas auto-injected
- Convert all 15 negative constraints to positive reframes

**Changes (P1):**
- Simplify `_build_plan_prompt()` to minimal framing
- Filter MCP tools to 4 essentials
- Add explicit DAG validation instruction
- Add explicit complexity criteria (small/medium/large definitions)

### 4.4 IMPLEMENT Phase

**File:** `spine/agents/implement_agent.py` — `_build_wave_orchestrator_prompt()`

**Changes (P0):**
- Remove duplicated prompt sections (Core Behaviour, Interpreter Environment, Tools guidance)
- Replace all negative constraints with positive directives
- Inline complete dispatch patterns (no cross-references)

**Changes (P1):**
- Add `min_length` validation to tool input fields

### 4.5 VERIFY Phase

**File:** `spine/agents/verify_agent.py` — `_build_orchestrator_prompt()`

**Changes (P0):**
- Remove 11 negative prompting instances
- Remove 11 philosophical fluff items

**Changes (P1):**
- Rewrite with positive, step-by-step directives
- Fix JS dispatch pattern (correct tool names, add missing variables)

**Changes (P2):**
- `verify_tools.py` already exists with `ReadVerificationContextTool` and `WriteVerificationReportTool` ✅
- `skip_filesystem_middleware=True` already set ✅

### 4.6 Subagent Prompts

**File:** `spine/agents/subagents.py` — `SUBAGENT_PROMPTS`

- Reduce researcher tool table from 18 to 5 + discovery
- Add min_length validation to slice implementer and verifier instructions

---

## 5. Fail-Closed Gates

### 5.1 Existing Gates

| Gate | Validates | Routes on Fail | Status |
|------|-----------|----------------|--------|
| `gate_plan_to_implement` | `plan.json` has valid `feature_slices` | `human_review` | ✅ EXISTS |

### 5.2 New Fail-Closed Checks

#### 5.2.1 Phase Entry Preconditions

Add to each phase's entry (in the phase node function or state mapper):

```python
# In spine/phases/plan.py (or _plan_state_mapper)
def _check_spec_prerequisite(state: WorkflowState) -> None:
    """Fail-closed: SPECIFY must have produced a specification."""
    if not state.get("spec_completed", False):
        raise CriticalContractFailure(
            "PLAN cannot start: SPECIFY phase did not complete. "
            "spec_completed invariant is False."
        )
    # Also verify on-disk existence
    spec_path = artifact_path(state["work_id"], PhaseName.SPECIFY.value)
    spec_file = Path(state["workspace_root"]) / ".spine" / "artifacts" / state["work_id"] / "specify" / "specification.md"
    if not spec_file.exists():
        raise CriticalContractFailure(
            f"PLAN cannot start: specification.md not found at {spec_file}"
        )
```

#### 5.2.2 Precondition Matrix

| Phase | Prerequisites | Check Type |
|-------|---------------|------------|
| PLAN | `spec_completed == True`, `specification.md` on disk | Invariant + disk |
| IMPLEMENT | `plan_completed == True`, `plan.json` on disk, `execution_waves` non-empty | Invariant + disk + quality |
| VERIFY | `implement_completed == True`, `implementation.md` on disk | Invariant + disk |
| GAP_PLAN | `verify_completed == True`, `verification.md` on disk | Invariant + disk |
| CRITIC_PLAN | `plan_completed == True` | Invariant |
| CRITIC_SPECIFY | `spec_completed == True` | Invariant |

#### 5.2.3 New Exception

In `spine/exceptions.py`:

```python
class CriticalContractFailure(Exception):
    """A hard precondition for a phase was not met.

    Unlike transient errors, this indicates a structural problem
    in the workflow — a phase is attempting to run without its
    prerequisites being satisfied.  The workflow should halt.
    """
```

### 5.3 Gate Wiring

All existing phase→next-phase edges already use `_phase_status_router` which routes `needs_review` to `human_review`. The new precondition checks should raise `CriticalContractFailure` before the phase subgraph runs — this gets caught by the subgraph wrapper and propagated as `status: "failed"`.

---

## 6. Constrained Decoding

### 6.1 Configuration

Add to `spine/config.py`:

```python
@dataclass
class ProviderConfig:
    # ... existing fields ...
    guided_decoding: bool = False  # Enable schema-constrained sampling
```

Add to `_PROVIDER_KEYS` for validation.

### 6.2 Model Wiring

In `spine/agents/helpers.py` — `_build_local_model()`:

When `guided_decoding=True` and a `response_format` Pydantic model is provided, inject `guided_json` into the model constructor:

```python
if guided_decoding and response_format is not None:
    kwargs["guided_json"] = response_format.model_json_schema()
```

### 6.3 Guard Function

```python
def _supports_guided_decoding(model_name: str) -> bool:
    """Check if the inference engine supports guided/constrained decoding."""
    # vLLM supports guided_json for most models
    # Check for thinking models that reject it
    thinking_prefixes = ("qwen3", "qwq", "deepseek-r")
    return not any(model_name.lower().startswith(p) for p in thinking_prefixes)
```

### 6.4 Enable in Config

```yaml
providers:
  phases:
    implement:
      provider: local
      temperature: 0.6
      guided_decoding: true  # NEW
    verify:
      provider: deepseek-v4-pro
      guided_decoding: true  # NEW
```

### 6.5 Integration with `response_format`

Today, only researcher gets `response_format` (and only after model compatibility check). With constrained decoding:

1. **All subagents** with Pydantic `response_format` models get `guided_json` when the provider supports it
2. **Phase orchestrators** with structured output tools get `guided_json` for their tool schemas
3. The `_supports_forced_tool_choice` check is replaced by `_supports_guided_decoding` which is more permissive

---

## 7. Implementation Order

### Milestone 1: Foundation (Low Risk, High Impact)
**Estimated:** 2-3 hours | **All files:** 3-5 changed

| # | Task | Files | Tokens Saved |
|---|------|-------|-------------|
| 1.1 | Add all phases to `_SKIP_AGENTS_MD` | `skills_resolver.py` | ~25K total |
| 1.2 | Add `CriticalContractFailure` exception | `exceptions.py` | — |
| 1.3 | Add `guided_decoding` to config schema | `config.py` | — |
| 1.4 | Add `Specification`, `FixInstruction`, `GapPlan`, `CriticReview` models | `types.py` | — |
| 1.5 | Add all `<phase>_completed` fields to `WorkflowState` | `state.py` | — |

### Milestone 2: State Invariants (Medium Effort)
**Estimated:** 3-4 hours | **All files:** 5-7 changed

| # | Task | Files |
|---|------|-------|
| 2.1 | Update all 8 result mappers to set completion flags | `compose.py` |
| 2.2 | Add `validate_phase_prerequisites()` to artifact gate | `artifact_gate.py` |
| 2.3 | Update `_save_exploration_artifacts` to set invariants | `exploration_subgraph.py` |
| 2.4 | Add invariant consistency check in artifact gate node | `artifact_gate.py` |
| 2.5 | Write tests for invariant propagation | `tests/` |

### Milestone 3: Structured Data Handoffs (Core)
**Estimated:** 5-7 hours | **All files:** 8-12 changed

| # | Task | Files |
|---|------|-------|
| 3.1 | SPECIFY: `specification.json` output + `Specification` model | `specify_tools.py`, `specify_agent.py`, `exploration_subgraph.py` |
| 3.2 | PLAN: Complexity validation + DAG check + min_length | `plan_tools.py`, `types.py` |
| 3.3 | IMPLEMENT: Enable `response_format=SliceResult` | `subagents.py`, `implement_agent.py` |
| 3.4 | VERIFY: Enable `response_format=VerificationResult` | `subagents.py`, `verify_agent.py` |
| 3.5 | GAP_PLAN: Create `gap_plan_tools.py` + `GapPlan` model + structured output | NEW: `gap_plan_tools.py`, `gap_plan_agent.py`, `gap_plan_subgraph.py` |
| 3.6 | CRITIC: Add `response_format=CriticReview` + drop string parsing | `critic/agent.py`, `critic_review.py` |
| 3.7 | Add `verification_findings` to `VerifySubgraphState` | `subgraph_state.py` |
| 3.8 | Add `gap_plan_json` to `GapPlanSubgraphState` | `subgraph_state.py` |

### Milestone 4: Prompt Optimization
**Estimated:** 4-6 hours | **All files:** 5-8 changed

| # | Task | Files |
|---|------|-------|
| 4.1 | SPECIFY: Remove negative instructions, irrelevant sections, MCP tools | `specify_agent.py` |
| 4.2 | PLAN: Remove AGENTS.md duplication, hand-written tool docs, negative constraints | `plan_agent.py` |
| 4.3 | IMPLEMENT: Remove duplicated sections, replace negative constraints | `implement_agent.py` |
| 4.4 | VERIFY: Remove negative prompting + fluff, rewrite step-by-step | `verify_agent.py` |
| 4.5 | Subagents: Reduce researcher tool table, add validation | `subagents.py` |
| 4.6 | Hyper-literal restructuring for SPECIFY and PLAN | `specify_agent.py`, `plan_agent.py` |

### Milestone 5: Fail-Closed Gates
**Estimated:** 3-4 hours | **All files:** 4-6 changed

| # | Task | Files |
|---|------|-------|
| 5.1 | Add `_check_spec_prerequisite` in PLAN entry | `plan.py` or `compose.py` |
| 5.2 | Add `_check_plan_prerequisite` in IMPLEMENT entry | `implement.py` or `compose.py` |
| 5.3 | Add `_check_implement_prerequisite` in VERIFY entry | `verify.py` or `compose.py` |
| 5.4 | Add `_check_verify_prerequisite` in GAP_PLAN entry | `gap_plan.py` or `compose.py` |
| 5.5 | Wire `validate_phase_prerequisites()` into `make_artifact_gate_node()` | `artifact_gate.py` |
| 5.6 | Write tests for fail-closed behavior | `tests/` |

### Milestone 6: Constrained Decoding
**Estimated:** 2-3 hours | **All files:** 3-5 changed

| # | Task | Files |
|---|------|-------|
| 6.1 | Wire `guided_json` in `_build_local_model()` when `response_format` present | `helpers.py` |
| 6.2 | Add `_supports_guided_decoding()` guard | `helpers.py` or `subagents.py` |
| 6.3 | Replace `_supports_forced_tool_choice` with `_supports_guided_decoding` | `subagents.py` |
| 6.4 | Enable `guided_decoding: true` in config for local provider | `.spine/config.yaml` |
| 6.5 | Test structured output with constrained decoding enabled | Manual testing |

### Milestone 7: Integration Testing & Polish
**Estimated:** 2-3 hours

| # | Task |
|---|------|
| 7.1 | End-to-end test: task workflow with all structured handoffs |
| 7.2 | End-to-end test: critical_task workflow with critic structured output |
| 7.3 | End-to-end test: gap-fix loop with structured gap_plan |
| 7.4 | Verify state invariants propagate correctly through all phases |
| 7.5 | Verify fail-closed gates block on missing prerequisites |
| 7.6 | Token budget verification — confirm expected reductions |

---

## 8. Files Changed Summary

| File | Milestones | Change Type |
|------|-----------|-------------|
| `spine/models/state.py` | M1, M2 | Add completion flags |
| `spine/models/types.py` | M1 | Add Specification, FixInstruction, GapPlan, CriticReview |
| `spine/models/enums.py` | — | No changes needed |
| `spine/config.py` | M1, M6 | Add guided_decoding field |
| `spine/exceptions.py` | M1 | Add CriticalContractFailure |
| `spine/workflow/compose.py` | M2, M5 | Update result mappers, add precondition checks |
| `spine/workflow/artifact_gate.py` | M2, M5 | Add invariant validation, wire checks |
| `spine/workflow/subgraph_state.py` | M3 | Add new fields to subgraph states |
| `spine/workflow/critic_review.py` | M3 | Replace string parsing with Pydantic |
| `spine/agents/skills_resolver.py` | M1 | Add all phases to _SKIP_AGENTS_MD |
| `spine/agents/helpers.py` | M6 | Wire guided_json |
| `spine/agents/subagents.py` | M3, M4, M6 | Enable response_format, reduce tool table, update guard |
| `spine/agents/specify_agent.py` | M4 | Prompt optimization |
| `spine/agents/specify_tools.py` | M3 | Add specification.json output |
| `spine/agents/plan_agent.py` | M4 | Prompt optimization |
| `spine/agents/plan_tools.py` | M3 | Add complexity/DAG validation |
| `spine/agents/implement_agent.py` | M4 | Prompt optimization |
| `spine/agents/verify_agent.py` | M4 | Prompt optimization |
| `spine/agents/gap_plan_agent.py` | M3 | Restructure with purpose-built tools |
| `spine/agents/gap_plan_tools.py` | M3 | **NEW FILE** — purpose-built tools |
| `spine/critic/agent.py` | M3 | Add response_format=CriticReview |
| `spine/workflow/subgraphs/exploration_subgraph.py` | M2, M3 | Update invariant setting, specification.json |
| `spine/workflow/subgraphs/gap_plan_subgraph.py` | M3 | Add structured output support |
| `spine/phases/plan.py` | M5 | Add fail-closed spec check |
| `spine/phases/implement.py` | M5 | Add fail-closed waves check |
| `spine/phases/verify.py` | M5 | Add fail-closed implement check |
| `.spine/config.yaml` | M6 | Enable guided_decoding |

**21 files changed, 1 new file created.**

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| `response_format` breaks on thinking models | Medium | Medium | Keep `_supports_guided_decoding` guard; fall back to free-text |
| Structured output increases latency | Low | Low | Guided decoding typically *reduces* latency by eliminating retries |
| State invariant flags get out of sync | Medium | High | Add invariant consistency check in artifact gate; write integration tests |
| Fail-closed gates too aggressive | Low | Medium | Gates check invariants + disk; human_review fallback always available |
| Prompt optimization degrades quality | Medium | High | Test with real workloads before merging; compare spec/plan quality |

---

## 10. Success Criteria

1. **Every agent-to-agent handoff uses a typed Pydantic model** (not free-text parsing)
2. **All 8 `_completed` invariants propagate correctly** through the workflow
3. **Fail-closed gates prevent phases from running without prerequisites** — verified by integration test
4. **Token reduction of ≥25%** across SPECIFY, PLAN, IMPLEMENT, and VERIFY prompts
5. **GAP_PLAN produces structured `gap_plan.json`** with typed `FixInstruction` objects
6. **Critic uses `CriticReview` Pydantic model** — no more string keyword parsing
7. **Zero regression** on existing workflow tests

---

*Plan synthesized from analysis of 6 reference documents, deep codebase inspection of exploration loop, critic system, structured data handoffs, state invariants, and per-phase subgraph architectures. All recommendations aligned with the existing StateGraph topology and LangGraph subgraph isolation model.*
