# Task: Apply the explore/synthesis separation pattern from SPECIFY to the PLAN phase

## Context

SPECIFY was recently refactored to separate **exploration** (codebase research via `research_manager → explore → aggregate` loop) from **synthesis** (writing the spec artifact from accumulated findings). This is implemented as a multi-node LangGraph subgraph in `spine/workflow/subgraphs/exploration_subgraph.py`, gated behind a feature flag `_USE_EXPLORATION_SUBGRAPH` in `spine/workflow/compose.py`.

PLAN currently uses a linear subgraph (`plan_subgraph.py`: `run_agent → save_artifacts`). The PLAN agent's system prompt (`plan_agent.py`) already describes a manual 3-step explore-then-synthesize workflow where the agent dispatches researcher subagents via `eval`/`Promise.allSettled`, then synthesizes into `write_structured_plan`. But this is all inside a single Deep Agent invocation — there's no LangGraph-level separation between exploration and synthesis, no loop with sufficiency checking, and no `research_manager` orchestrating the explore rounds.

The goal is to replace the linear `plan_subgraph.py` with the same exploration subgraph pattern SPECIFY uses, so PLAN gets:
- Multi-round research manager deciding what to investigate
- Parallel researcher subagent dispatch via LangGraph's `Send` API
- Sufficiency gate (loop back or proceed to synthesis)
- Separate synthesis step that reads accumulated findings + the spec and produces plan artifacts

The `build_exploration_subgraph()` function in `exploration_subgraph.py` already has a phase parameter and a stub for `PhaseName.PLAN` that raises `NotImplementedError` (line 330).

## Files to modify

### 1. `spine/workflow/subgraphs/exploration_subgraph.py`

#### a) Add `_synthesize_plan` function (analogous to `_synthesize_specify`, lines 154–233)

This function should:

1. Read from `ExplorationSubgraphState`: `description`, `work_id`, `work_type`, `workspace_root`, `findings`, `retry_count`, `feedback`
2. Build a plan Deep Agent via `build_plan_agent(dict(state), config)` — same as the current `plan_subgraph.py:_run_plan_agent` does
3. Materialize prior artifacts to disk (so the agent can read the spec)
4. Construct a prompt that includes:
   - Rework prefix (if `retry_count > 0`) — same as `_synthesize_specify`
   - Instruction to create a detailed technical plan incorporating the codebase research findings below
   - The spec path instruction: `"The full specification is available on disk at .spine/artifacts/{work_id}/specify/specification.md — read it carefully with read_file"`
   - The formatted research findings (reuse `_format_findings()`)
   - Instruction to call `write_structured_plan` to produce both `plan.md` and `plan.json`
   - Previous review feedback (if rework)
5. Invoke the agent via `ainvoke_with_retry()` with `PhaseName.PLAN` context from `build_context(state, PhaseName.PLAN)`
6. **After the agent completes**: Read `plan.json` from disk (same logic as `plan_subgraph.py` lines 102–127) and compute execution waves via `_compute_waves()` from `plan_subgraph.py` (or inline it)
7. Return: `messages`, `agent_response` (from `extract_response`), plus any PLAN-specific state like `plan_json` (raw string), `execution_waves` (list of wave dicts)

Pitfall: `ExplorationSubgraphState` doesn't have `plan_json` or `execution_waves` fields. Options:
- **Recommended**: Add `plan_json: str` and `execution_waves: list` to `ExplorationSubgraphState` in `subgraph_state.py` (they're harmless for SPECIFY, which will leave them empty)
- Alternative: Store execution waves in a separate key on the returned dict that `_save_exploration_artifacts` forwards

#### b) Wire `_synthesize_plan` into `build_exploration_subgraph()` (line 328–330)

Replace:
```python
elif phase == PhaseName.PLAN.value:
    raise NotImplementedError("PLAN exploration subgraph not yet implemented")
```
with:
```python
elif phase == PhaseName.PLAN.value:
    synthesizer = _synthesize_plan
```

#### c) Update `_save_exploration_artifacts` (lines 268–305)

Currently handles plan.md in the fallback path (line 293: `"plan.md"`), but doesn't handle `plan.json` or execution waves. For PLAN phase, after scanning artifacts, also:
- Check if `plan.json` exists on disk (same pattern as `plan_subgraph.py` lines 238–243)
- Include `plan_json` string in `artifacts_output` if present
- Forward `execution_waves` from state to the output dict so the result mapper picks them up

### 2. `spine/workflow/subgraph_state.py`

Add to `ExplorationSubgraphState`:
```python
plan_json: str  # Raw plan.json content (only used in PLAN phase)
execution_waves: list  # Computed execution waves (only used in PLAN phase)
```

These are safe for SPECIFY — they'll just be empty.

### 3. `spine/workflow/compose.py`

Set the feature flag (line 107):
```python
PhaseName.PLAN.value: True,  # was False
```

The existing `_plan_state_mapper` and `_plan_result_mapper` already handle `execution_waves` forwarding. The state mapper sets `spec_path` and `has_spec` — these fields are on `PlanSubgraphState` but not `ExplorationSubgraphState`. Options:
- **Simplest**: The `_synthesize_plan` function hardcodes `has_spec = True` (all work types now run specify) and derives `spec_path` from `work_id` — doesn't need them from state
- More correct: Have a separate exploration-specific state mapper for PLAN, OR add `spec_path`/`has_spec` to `ExplorationSubgraphState`

Recommended: the simplest approach. Since `has_spec` is always `True` and `spec_path` is deterministic from `work_id`, just derive them in `_synthesize_plan` rather than threading them through state.

## What stays the same

- `research_manager_node`, `_research_router`, `_explore_node`, `_aggregate_node`, `_sufficiency_router` — these are phase-agnostic and work for PLAN exactly as they do for SPECIFY
- `_format_findings` — reused as-is
- The existing `plan_agent.py`, `plan_tools.py`, `plan_subgraph.py` — these continue to work as-is; the exploration subgraph just calls `build_plan_agent()` for synthesis instead of having the agent do the research loop internally
- `_plan_result_mapper` in `compose.py` — already forwards `execution_waves` to parent state

## Acceptance criteria

1. `_synthesize_plan` is implemented with the same structure as `_synthesize_specify` but for plan artifact output
2. `build_exploration_subgraph(phase="plan")` compiles without error
3. `_USE_EXPLORATION_SUBGRAPH["plan"] = True` enables the new path
4. Changing `_USE_EXPLORATION_SUBGRAPH["plan"]` from `True` to `False` falls back to the linear `plan_subgraph` (no regression)
5. Execution waves are computed from `plan.json` and forwarded to parent state
6. Rework/failure paths mirror `_synthesize_specify` behavior