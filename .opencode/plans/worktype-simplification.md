# SPINE WorkType Simplification Plan

## Summary

Reduce WorkTypes from 8 to 4, rename for clarity, unify the graph structure,
and fix the interrupt/pause mechanism for reviewed (planning) workflows.

## New WorkType Names

| New Name                  | String Value             | Old Equivalent | Critic on Spec? | Pauses for Approval? |
|--------------------------|--------------------------|----------------|-----------------|---------------------|
| `TASK`                   | `"task"`                 | `spec`         | No              | No                  |
| `CRITICAL_TASK`          | `"critical_task"`        | `critical_spec`| Yes             | No                  |
| `REVIEWED_TASK`          | `"reviewed_task"`        | `plan`         | No              | Yes (after critic_plan) |
| `CRITICAL_REVIEWED_TASK` | `"critical_reviewed_task"`| `plan_spec`   | Yes             | Yes (after critic_plan) |

Removed: `quick`, `critical_quick`, `plan_only`, `critical_plan_only`
(`plan_only` was an alias of `plan`; `critical_plan_only` was an alias of `plan_spec`)

## Graph Sequences (all 4 are identical structural sequences)

```python
WORKFLOW_SEQUENCES = {
    "task": [
        ("specify", None),
        ("plan", None),
        ("critic_plan", "plan"),
        ("implement", None),
        ("verify", None),
    ],
    "critical_task": [
        ("specify", None),
        ("critic_specify", "specify"),  # extra critic
        ("plan", None),
        ("critic_plan", "plan"),
        ("implement", None),
        ("verify", None),
    ],
    "reviewed_task": [
        ("specify", None),
        ("plan", None),
        ("critic_plan", "plan"),
        ("implement", None),           # same as task but paused via interrupt_after
        ("verify", None),
    ],
    "critical_reviewed_task": [
        ("specify", None),
        ("critic_specify", "specify"),  # extra critic
        ("plan", None),
        ("critic_plan", "plan"),
        ("implement", None),           # same as critical_task but paused
        ("verify", None),
    ],
}
```

The `reviewed_*` and non-reviewed types share identical phase sequences.
The difference is handled at the **stream level**, not the graph structure.

## Pause Mechanism: `interrupt_after`

### Why `interrupt_after` instead of `interrupt()` in a node

The existing `human_review` node calls `interrupt()` which blocks `astream().__anext__()`
indefinitely. `submit_work()` has a 120s stall timer. When `interrupt()` fires, the stall
timer wins -> work incorrectly marked as "stalled."

Solution: use LangGraph's `interrupt_after` parameter on `graph.astream()`:

```python
# In submit_work() for reviewed_task types:
stream_iter = graph.astream(
    initial_state, thread_config,
    interrupt_after=["critic_plan"],    # stream ends normally after this node
    stream_mode=["updates", "messages"],
    subgraphs=True, version="v2",
)
```

**Behavior:**
- Stream runs SPECIFY -> PLAN -> CRITIC_PLAN
- After CRITIC_PLAN output is yielded, stream raises `StopAsyncIteration` cleanly
- No stall timer issues - the stream completed naturally
- Checkpoint is persisted to SQLite **before** `StopAsyncIteration`
- Dispatcher interprets this as "graph interrupted, not completed"
- Sets work entry status to `"awaiting_approval"`

**On approval (resume):**
```python
from langgraph.types import Command

graph.astream(
    Command(resume=None), thread_config,
    stream_mode=["updates", "messages"],
    subgraphs=True, version="v2",
)
# Continues from IMPLEMENT -> VERIFY
# Uses same thread_id - picks up exact checkpoint
```

### Recovery after process crash

1. Checkpoint is in SQLite (`spine.db`) for the thread_id
2. Work entry is in `work_entries.db` with status `"awaiting_approval"`
3. On restart, if work_entry status is `"awaiting_approval"` and checkpoint exists -> UI shows it
4. If work_entry shows `"running"` but checkpoint is at `critic_plan` -> recovery marks it `"awaiting_approval"`
5. Resume uses `Command(resume=None)` with same thread_id -> no state loss

### submit_work() changes needed

After the streaming loop completes (StopAsyncIteration), detect if work is in an
interrupted state vs. truly completed:

```python
# After the streaming loop:
# Check if the graph was interrupted (interrupt_after) vs. completed
# Option A: Check the checkpointer for whether we're at an interrupt boundary
# Option B: Use LangGraph's get_state() to check the "next" nodes
# Option C: For reviewed task types, always treat completion after
#           critic_plan as "awaiting_approval"

if work_type in ("reviewed_task", "critical_reviewed_task"):
    final_status = "awaiting_approval"
else:
    final_status = result.get("status", "completed")
```

## Files to Change (ordered by dependency)

### 1. `spine/models/enums.py`
- Remove: `QUICK`, `CRITICAL_QUICK`, `PLAN_ONLY`, `CRITICAL_PLAN_ONLY`
- Rename: `SPEC` -> `TASK`, `CRITICAL_SPEC` -> `CRITICAL_TASK`,
  `PLAN` -> `REVIEWED_TASK`, `PLAN_SPEC` -> `CRITICAL_REVIEWED_TASK`

### 2. `spine/models/types.py`
- `WorkSpawnSpec.work_type` default: `WorkType.QUICK` -> `WorkType.TASK`

### 3. `spine/workflow/compose.py`
- Replace `WORKFLOW_SEQUENCES` with 4 entries as shown above
  (reviewed_task and critical_reviewed_task now include implement+verify)
- Update docstrings

### 4. `spine/workflow/studio.py`
- Remove: `quick_graph`, `critical_quick_graph`, `plan_only_graph`, `critical_plan_only_graph`
- Rename: `spec_graph` -> `task_graph`, `critical_spec_graph` -> `critical_task_graph`,
  `plan_graph` -> `reviewed_task_graph`, `plan_spec_graph` -> `critical_reviewed_task_graph`
- Fix stale docstring on `critical_spec_graph` (references non-existent tasks/critic_tasks phases)

### 5. `spine/work/dispatcher.py`
- `submit_work()`:
  - Update docstring work_type values
  - Add `interrupt_after=["critic_plan"]` for reviewed_task/critical_reviewed_task
  - After stream loop: if work_type is reviewed_*, set `final_status = "awaiting_approval"`
  - Remove plan_types tuple override (line 464) - handled by interrupt_after now
- `list_plans()`:
  - `plan_types` -> `("reviewed_task", "critical_reviewed_task")`
- `resume_interrupted_work()`:
  - No changes needed - already uses `Command(resume=...)`
- `approve_plan()` (line ~1830):
  - Update plan_types check -> `("reviewed_task", "critical_reviewed_task")`
  - On approve action: use `resume_interrupted_work()` with `Command(resume=None)` instead of spawning
  - Remove plan decomposition + spawn logic for approve action
  - Keep rejection and revision logic (they call `resume_work()` which re-runs from START)
- `split_work_plan()`:
  - `work_type_override` default: `"quick"` -> `"task"`
- All `plan_types` / `plan_types_check` tuples:
  - `("plan", "plan_spec", "plan_only", "critical_plan_only")` -> `("reviewed_task", "critical_reviewed_task")`

### 6. `spine/work/plan_resolver.py`
- `resolve_plan_to_units()`: default `work_type` -> `"task"`, docstring
- `create_work_spawn_specs()`: `WorkType.QUICK` -> `WorkType.TASK`

### 7. `spine/agents/tasks_agent.py`
- `resolve_tasks_subagents()`:
  - Old: branch on `"quick" in work_type and "critical" not in work_type` -> skip trivial
  - Old: branch on `"quick" in work_type or "spec" in work_type` -> build subagents
  - New: `"task" in work_type` always true for all 4 types -> always build subagents
  - Collapse to just `return build_phase_subagents(phase, state, config)` unconditionally
- `build_tasks_agent()`:
  - `is_quick = "quick" in work_type` -> remove (always false), remove `is_quick`-gated prompt branch

### 8. `spine/phases/implement.py` (line 223)
- `has_spec = "spec" in work_type`
  - Old intent: "did specify phase run?"
  - New reality: all 4 types run specify -> always true
  - Remove the check and its else branch; always load spec artifacts

### 9. `spine/phases/tasks.py` (line 95)
- `has_spec = "spec" in work_type` - same as above, remove conditional

### 10. `spine/cli/__init__.py`
- `--type` choices: `["quick", "critical_quick", "spec", "critical_spec"]`
  -> `["task", "critical_task"]`
- Default: `"spec"` -> `"task"`

### 11. `spine/ui/_pages/work_submit.py`
- Options: `["quick", "critical_quick"]` -> `["task", "critical_task"]`
- Labels:
  - `"task"` -> `"Task (SPECIFY -> PLAN -> CRITIC_PLAN -> IMPLEMENT -> VERIFY)"`
  - `"critical_task"` -> `"Critical Task (SPECIFY -> CRITIC_SPECIFY -> PLAN -> CRITIC_PLAN -> IMPLEMENT -> VERIFY)"`
- Update workflow reference table
- Update docstrings/comments

### 12. `spine/ui/_pages/spec_planning.py`
- Options: `["plan", "plan_spec", "plan_only", "critical_plan_only"]`
  -> `["reviewed_task", "critical_reviewed_task"]`
- Labels:
  - `"reviewed_task"` -> `"Reviewed Task (SPECIFY -> PLAN -> CRITIC_PLAN -> await approval -> IMPLEMENT -> VERIFY)"`
  - `"critical_reviewed_task"` -> `"Critical Reviewed Task (SPECIFY -> CRITIC_SPECIFY -> PLAN -> CRITIC_PLAN -> await approval -> IMPLEMENT -> VERIFY)"`
- Update workflow type reference table
- Remove alias descriptions (plan_only, critical_plan_only)

### 13. `spine/ui/_pages/queue.py`
- `_PHASE_SEQUENCE`: replace old entries with new names
- **Bug fix**: remove stale `tasks` and `critic_tasks` phases from sequences that
  don't exist in actual WORKFLOW_SEQUENCES. All task types go:
  specify -> (critic_specify) -> plan -> critic_plan -> implement -> verify

### 14. `spine/ui/_pages/work_detail.py`
- Fallback work_type: `"quick"` -> `"task"` (lines 107, 246)
- Sequence fallback: `WORKFLOW_SEQUENCES.get("quick", [])` -> `WORKFLOW_SEQUENCES.get("task", [])` (line 108)

### 15. `spine/work/ralph_worker.py`
- Docstring example: `work_type="quick"` -> `work_type="task"`

### 16. `langgraph.json`
```json
{
  "dependencies": ["."],
  "graphs": {
    "task": "spine.workflow.studio:task_graph",
    "critical_task": "spine.workflow.studio:critical_task_graph",
    "reviewed_task": "spine.workflow.studio:reviewed_task_graph",
    "critical_reviewed_task": "spine.workflow.studio:critical_reviewed_task_graph"
  },
  "env": ".env"
}
```

## Test Files to Update

All `"quick"` -> `"task"`, `"spec"` -> `"task"` as appropriate:

1. `tests/unit/test_models.py` - WorkType enum assertions (remove old, add new)
2. `tests/unit/test_workflow_gates.py` - loops: `("quick", "critical_quick", "spec", "critical_spec")` -> `("task", "critical_task")`
3. `tests/unit/test_config.py` - `valid_work_types` -> `["task", "critical_task"]`
4. `tests/unit/test_interrupt_workflow.py` - build_workflow_graph("quick") -> ("task"), same for critical_quick
5. `tests/unit/test_all_subgraphs.py` - all `"quick"` -> `"task"`, graph tests renamed
6. `tests/unit/test_verify_subgraph.py` - `work_type: "quick"` -> `"task"`
7. `tests/unit/test_dispatcher.py` - all `"quick"` -> `"task"`
8. `tests/unit/test_ui_api.py` - all `"quick"` -> `"task"`
9. `tests/unit/test_ralph_worker.py` - all `"quick"` -> `"task"`
10. `tests/unit/test_prompt_efficiency.py` - `"quick"` -> `"task"`
11. `tests/unit/test_specify_plan_tools.py` - `work_type="quick"` -> `"task"`
12. `tests/unit/test_subgraph_wrapper.py` - `"quick"` -> `"task"`
13. `tests/unit/test_dispatcher_streaming.py` - `"quick"` -> `"task"`
14. `tests/test_restart.py` - all `"quick"` -> `"task"`
15. `tests/test_work_ordering.py` - all `"quick"` -> `"task"`

## Implementation Order

1. **Enum + models** (files 1-2) - foundation, everything depends on these
2. **Workflow compose + studio** (files 3-4) - graph structure
3. **Dispatcher** (file 5) - interrupt_after, resume, status handling
4. **Agent builders** (files 7-9) - fix string-matching branches
5. **Plan resolver** (file 6) - spawned item defaults
6. **CLI + UI pages** (files 10-14)
7. **Config files** (files 15-16)
8. **Tests** - fix all references, verify graph builds, verify interrupt works
9. **Lint + typecheck** - `ruff check spine/ tests/`, `ruff format spine/ tests/`, `mypy spine/`