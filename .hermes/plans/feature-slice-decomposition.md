# Plan: Feature-Slice Decomposition

## Problem
Current decomposition creates micro-tasks ("Implement core data models",
"Implement feature module A/B/C") that require the planner to understand the
codebase at a file level. The agent then gets narrow prompts with no context
for making good design decisions. This kills parallelism and quality.

## Solution: Planner defines architectural boundaries, Agent owns implementation

### What changes

1. **New `FeatureSlice` model** — a feature-slice is the unit of work the
   planner produces.  Each slice contains:
   - `id`, `description` (what to build, at feature granularity)
   - `scope` (which modules/directories the agent should work within)
   - `depends_on` (IDs of other slices — the dependency DAG)
   - `agent_role` (coder, test_engineer, reviewer)
   - `acceptance` (what "done" looks like — the gate criteria)

2. **Replace `_get_synthesis_tasks()`** — delete the static task-map.
   The planner's SYNTHESIZE subphase now produces FeatureSlice objects from
   the requirement + analysis + research, NOT from a pre-baked template.

3. **Rewrite SDD IMPLEMENT phase** — instead of creating one TaskNode per
   file ("impl-core", "impl-models"), create one SubPhaseNode per
   FeatureSlice.  Each subphase carries the full slice context (description,
   scope, acceptance) so the agent gets a rich prompt, not a filename.

4. **Rewrite QuickWork IMPLEMENT phase** — same pattern but typically a
   single slice.

5. **Wire AgentProvider into WorkflowEngine** — the engine gets an
   `agent_provider` (or AgentFallbackChain).  When executing an
   implementation subphase, it calls `agent_provider.execute()` with the
   feature-slice description and scope.  The agent decomposes internally.

6. **Verification uses acceptance criteria** — the VERIFY phase checks the
   acceptance criteria from each FeatureSlice, not a generic "run tests".

7. **Remove dead helpers** — `_extract_components`, `_extract_requirements`,
   `_generate_backend_file_structure`, `_generate_frontend_file_structure`,
   `_estimate_complexity` are all heuristics the planner no longer needs.

## Files to create/modify

| File | Action |
|------|--------|
| `spine/models/types.py` | Add `FeatureSlice` dataclass |
| `spine/models/dag.py` | Remove `_get_synthesis_tasks`, `_extract_components`, `_extract_requirements`, `_generate_*_file_structure`, `_estimate_complexity`. Add `synthesize_slices()` that uses LLM to produce FeatureSlices from context. |
| `spine/workflows/engine.py` | Add `agent_provider` parameter to WorkflowEngine. Add `_execute_feature_slice()` that delegates to agent. |
| `spine/workflows/sdd.py` | Rewrite `_build_plan_phase` to produce FeatureSlices. Rewrite `_build_implement_phase` to create one subphase per slice. |
| `spine/workflows/quick_work.py` | Same pattern, single slice. |
| `spine/core/state_machine.py` | Update execution_phase to use feature slices when available. |
| `tests/test_workflow_engine.py` | Update tests for new shape. |
| `tests/test_agent_providers.py` | Add tests for feature-slice execution. |
