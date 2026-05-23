# State Model Invariant Tracking

## Purpose

This document describes the invariant tracking fields added to SPINE state models to prevent rework misinterpretation. These boolean flags explicitly track whether critical phase operations completed successfully, rather than relying on implicit detection methods.

## Problem Statement

Previously, the system relied on implicit detection methods:
- **File existence checks**: "Does `gap_plan.md` exist on disk?"
- **Artifact content parsing**: "Does `verification.md` contain 'VERIFIED'?"
- **Empty state interpretation**: "Empty artifacts means the phase was skipped"

This led to misinterpretation during rework cycles where a failed/skipped phase could be mistaken for intentional empty output.

## Solution: Explicit Invariant Flags

### WorkflowState Invariants

| Field | Type | When to Set | Purpose |
|-------|------|-------------|---------|
| `gap_plan_produced` | `bool` | GAP_PLAN phase success | Distinguishes gap-fix attempt from missing gap plan |
| `exploration_executed` | `bool` | After SPECIFY/PLAN research rounds | Prevents re-exploration on rework loops |

### Subgraph State Invariants

#### ExplorationSubgraphState
| Field | Type | When to Set | Purpose |
|-------|------|-------------|---------|
| `exploration_happened` | `bool` | Research manager returns "done" after rounds | Prevents re-interpretation of empty findings |
| `synthesis_completed` | `bool` | Synthesizer agent succeeds | Confirms output was generated independently of artifacts |

#### ImplementSubgraphState
| Field | Type | When to Set | Purpose |
|-------|------|-------------|---------|
| `slices_dispatched` | `bool` | Slice-implementers dispatched successfully | Tracks dispatch independently from file writes |
| `implementation_files_written` | `bool` | Files detected on disk after agent completes | Confirms actual code changes occurred |

#### VerifySubgraphState
| Field | Type | When to Set | Purpose |
|-------|------|-------------|---------|
| `verification_attempted` | `bool` | Verify agent runs (start of run_verify_agent) | Tracks attempt vs. skip |
| `verification_passed` | `bool` | Verification contains VERIFIED/PASSED | Structured outcome instead of text parsing |

## Usage Patterns

### In Result Mappers

```python
def _gap_plan_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map GapPlanSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.GAP_PLAN.value)(subgraph_result, parent_state)
    
    # Set invariant flag based on actual file production
    disk_artifacts = subgraph_result.get("artifacts_output", {})
    base["gap_plan_produced"] = "gap_plan.md" in disk_artifacts
    
    return base
```

### In Exploration Subgraph

```python
async def _research_manager_node(...) -> dict[str, Any]:
    result = await run_research_manager(dict(state), config)
    round_num = state.get("research_round", 0)
    return {
        **result,
        "research_round": round_num + 1,
        "exploration_happened": True,  # Mark that we attempted research
    }

async def _save_exploration_artifacts(...) -> dict[str, Any]:
    ...
    result: dict[str, Any] = {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
        "synthesis_completed": bool(disk_artifacts),  # True if artifacts exist
    }
    
    if execution_waves:
        result["execution_waves"] = execution_waves
    
    return result
```

### In Verify Result Mapper

```python
def _verify_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    base = make_success_result_mapper(PhaseName.VERIFY.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    
    # Use the invariant flag instead of parsing text
    if subgraph_result.get("verification_passed"):
        final_status = "completed"
    elif phase_status == "needs_review":
        ...
    
    return base
```

## Integration with Existing Code

### Phase State Mappers (compose.py)

Update the state mappers to initialize invariants:

```python
def _gap_plan_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Map parent WorkflowState to GapPlanSubgraphState."""
    work_id = parent_state.get("work_id", "")
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.GAP_PLAN.value,
        "retry_count": 0,
        "verify_path": artifact_path(work_id, PhaseName.VERIFY.value),
        "plan_path": artifact_path(work_id, PhaseName.PLAN.value),
    }
```

### Key Integration Points

1. **GAP_PLAN phase**: Set `gap_plan_produced` in `_gap_plan_result_mapper`
2. **Exploration subgraph**: Set `exploration_happened` and `synthesis_completed` in `_save_exploration_artifacts`
3. **IMPLEMENT phase**: Set `slices_dispatched` and `implementation_files_written` in implement result mapper
4. **VERIFY phase**: Set `verification_attempted` and `verification_passed` in verify result mapper

## Benefits

1. **Explicit semantics**: Clear boolean indicators of what happened
2. **Prevents rework bugs**: System knows whether to re-explore or use existing research
3. **Better debugging**: Log flags show exactly what branches were taken
4. **Test assertions**: Unit tests can verify invariants at phase boundaries
5. **Future-proof**: Adding new invariants doesn't break existing reducer patterns

## Backward Compatibility

All invariant fields use `TypedDict(total=False)`, so existing code that doesn't set them will continue to work. The reducer patterns (using `operator.add` for lists and `_merge_dicts` for dicts) ensure proper accumulation across state transitions.