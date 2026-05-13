# SPINE Workflow Rebuild — Implementation Plan

## Architecture

From WORKFLOW_REBUILD.md: composable workflow phases, one phase per file, one agent per file,
composed into workflow graphs by work type.

### Directory Layout

```
spine/
├── __init__.py
├── __main__.py
├── models/
│   ├── __init__.py
│   ├── enums.py          # PhaseName, WorkType, ReviewStatus, TaskStatus
│   ├── types.py          # Task, Artifact, ReviewFeedback, PromptRequest
│   └── state.py          # WorkflowState (TypedDict for LangGraph)
├── config.py             # SpineConfig, load from YAML/env
├── exceptions.py         # Custom exceptions
├── workflow/
│   ├── __init__.py
│   ├── compose.py        # build_workflow_graph(work_type) → StateGraph
│   ├── critic_review.py  # two-tier critic: structural + agent
│   └── registry.py       # Phase registry: name → (phase_fn, agent_fn)
├── phases/
│   ├── __init__.py
│   ├── specify.py         # SPECIFY phase node + graph builder
│   ├── plan.py            # PLAN phase node + graph builder
│   ├── tasks.py            # TASKS phase node + graph builder
│   ├── implement.py        # IMPLEMENT phase node + graph builder
│   └── verify.py           # VERIFY phase node + graph builder
├── agents/
│   ├── __init__.py
│   ├── specify_agent.py   # Deep Agent config for specify
│   ├── plan_agent.py      # Deep Agent config for plan
│   ├── tasks_agent.py     # Deep Agent config for tasks (decomposition)
│   ├── implement_agent.py # Deep Agent config for implement
│   └── verify_agent.py    # Deep Agent config for verify
├── critic/
│   ├── __init__.py
│   └── agent.py           # Critic Deep Agent config
├── work/
│   ├── __init__.py
│   ├── dispatcher.py      # submit_work() — unified entry for CLI, UI, worker
│   └── ralph_worker.py   # RalphLoopWorker — background queue processor
├── persistence/
│   ├── __init__.py
│   ├── checkpoint.py      # SQLite-backed checkpoint store
│   └── artifacts.py       # Artifact storage (.spine/artifacts/)
├── services/
│   ├── __init__.py
│   └── audit_service.py   # Audit logging
├── ui_api/
│   ├── __init__.py
│   └── api.py             # UIApi — sole read/write interface for Streamlit
├── ui/
│   ├── __init__.py
│   ├── app.py             # Streamlit app + navigation
│   ├── pages/
│   │   ├── dashboard.py
│   │   ├── work_submit.py
│   │   ├── work_status.py
│   │   ├── work_history.py
│   │   ├── artifacts.py
│   │   ├── config_view.py
│   │   ├── audit_log.py
│   │   └── human_review.py
│   └── utils.py           # Shared UI helpers
├── cli/
│   ├── __init__.py
│   └── commands.py        # Click CLI: run, status, resume, list, config
```

## Workflow Types → Phase Composition

| Work Type         | Phases                                              |
|-------------------|-----------------------------------------------------|
| quick             | TASKS → IMPLEMENT → VERIFY                          |
| critical_quick    | TASKS → CRITIC → IMPLEMENT → VERIFY                 |
| spec              | SPECIFY → PLAN → CRITIC → TASKS → IMPLEMENT → VERIFY |
| critical_spec     | SPECIFY → CRITIC → PLAN → CRITIC → TASKS → CRITIC → IMPLEMENT → VERIFY |

## State Design (WorkflowState)

```python
class WorkflowState(TypedDict):
    work_id: str                           # Unique work item ID
    work_type: str                          # "quick" | "critical_quick" | "spec" | "critical_spec"
    description: str                        # User's work description
    current_phase: str                      # Current phase name
    phase_index: int                        # Index in the phase sequence
    retry_count: dict[str, int]            # Per-phase retry count (for critic rework)
    max_retries: int                        # Configurable max critic retries
    artifacts: Annotated[dict, merge_dicts] # Phase output artifacts (accumulated)
    feedback: Annotated[list, operator.add] # Review feedback (accumulated)
    status: str                             # "running" | "completed" | "needs_review" | "failed"
    prompt_request: dict | None            # Human input request from a phase
```

## Critic Review Design

Two-tier:
1. **Structural** (fast, no LLM): checks artifacts exist, not empty, basic structure
2. **Agent** (deep, LLM): quality review by critic Deep Agent

If structural fails → rework immediately (skip agent critic).
Agent exceptions → rework, not crash.
Feedback tagged by tier.

## Rework Loop

When critic returns NEEDS_REVISION:
- Increment retry_count for that phase
- If retry_count < max_retries → re-run the phase with feedback
- If retry_count >= max_retries → set status="needs_review", flag for human

## Prompt Request

A phase can emit a prompt_request dict. The workflow engine:
- Sets status="needs_review"
- Stores the request for UI/CLI to surface
- Execution pauses until human provides input

## Dispatcher Design (Zero Duplication)

`submit_work(description, work_type, config)` is the single entry point.
Called by:
- CLI: `spine run "description" --type spec`
- UI: Streamlit submit page
- Worker: RalphLoopWorker dequeues and calls submit_work

All reads go through `UIApi`.
UI pages NEVER import from workflow/ or phases/ directly.
