# SPINE UI: Critical Functions & Features

**Version:** 1.0
**Date:** 2026-05-11
**Status:** Active — Implementation Plan

---

## Architecture

### Core Principle: Zero Duplication

The UI must not duplicate any functional code. Every action the UI performs must
go through the same code path as the CLI. This is enforced at three levels:

1. **Entry points** — Both CLI and UI call `submit_work()` from
   `spine/work/dispatcher.py`. The UI never constructs a `SpineStateMachine`
   directly.

2. **Read layer** — Both CLI and UI read state from the same SQLite checkpoint
   database and `.spine/` file tree. The UI uses `spine/core/ui_api.py` as its
   sole interface to core logic — it never imports from `spine/core/state_machine`
   or `spine/models/` directly.

3. **Write layer** — All write operations (start work, approve gate, update
   config) go through `UIApi` which delegates to the same functions the CLI
   uses. No parallel implementation paths.

```
┌─────────────────────────────────────────────────────┐
│               CLI (click commands)                    │
│               spine/cli/commands/                     │
└──────────────────┬──────────────────────────────────┘
                   │ calls
                   ▼
┌─────────────────────────────────────────────────────┐
│          Shared Backend (dispatcher + ui_api)         │
│  spine/work/dispatcher.py  ←→  spine/core/ui_api.py  │
│          submit_work()  |  run_workflow()              │
└──────────────────┬──────────────────────────────────┘
                   │ calls
                   ▼
┌─────────────────────────────────────────────────────┐
│               UI (Streamlit pages)                    │
│               spine/ui/*.py                           │
│  Calls UIApi exclusively for reads/writes             │
└─────────────────────────────────────────────────────┘
```

### Enforcement Rules

| Rule | Enforcement |
|------|-------------|
| UI pages never import from `spine.core.state_machine` | Code review gate |
| UI pages never construct provider instances directly | `UIApi.start_work()` handles provider resolution |
| UI pages never read checkpoint DB directly | All reads go through `UIApi` or `spine/ui/utils.py` |
| UI pages never write to `.spine/` directly (except via `UIApi`) | `UIApi` is the sole write interface |
| New features in UI must have a CLI equivalent | Feature parity is mandatory |

---

## Feature 1: Dashboard — Summarised Overview

### Current State

The dashboard (`spine/ui/dashboard.py`) lists work items with progress bars and
metrics. It polls the checkpoint DB for updates.

### Required Enhancements

The dashboard must provide a *summarised* overview — a single glance should
reveal the health of all active work, recent outcomes, and pending actions.

#### 1.1 Summary Cards

At the top of the dashboard, show:

| Card | Data Source |
|------|-----------|
| Active Work Items | `UIApi.get_active_work_items()` filtered by phase != COMPLETE/ERROR |
| Pending Review | Work items in `HUMAN_REVIEW` or with `critic_gate_result == "NEEDS_REVISION"` |
| Queue Depth | `UIApi.get_queue_status()` — count of pending tasks in the queue |
| Recent Completions | Last 5 work items with phase == COMPLETE |

#### 1.2 Phase Distribution Chart

A horizontal bar or pie chart showing how many work items are in each phase
(INIT, PLANNING, EXECUTION, VERIFICATION, COMPLETE, ERROR, BLOCKED).

#### 1.3 Quick Actions

- "Start New Work" button (navigates to New Work form)
- "View Queue" button (navigates to Task Queue page)
- "Review Pending" button (navigates to first work item needing review)

### Implementation

- Modify: `spine/ui/dashboard.py` — add summary cards, chart, quick actions
- Add: `UIApi.get_queue_status()` — returns pending/running/failed counts
- Add: `UIApi.get_recent_completions(limit=5)` — last N completed items
- No new dependencies — Streamlit `st.metric` + `st.bar_chart` are sufficient

---

## Feature 2: Task Details — Workflow Outcomes & Artifacts

### Current State

The work detail page (`spine/ui/work_detail.py`) shows phase, sub-phase progress,
agent outputs, and swarm events. It does NOT render markdown artifacts produced
during execution, and it does NOT clearly answer "what was done and why".

### Required Enhancements

#### 2.1 Workflow Outcome Summary

At the top of the work detail page, add a concise outcome block:

```
┌─────────────────────────────────────────────────────┐
│  OUTCOME: ✅ Complete                                │
│  Requirement: "Build auth system"                    │
│  What was done: 5 tasks completed across 3 phases    │
│  Plan: 3 FeatureSlices → all implemented             │
│  Verification: All acceptance criteria passed         │
│  Artifacts: plan.md, spec.md, auth_middleware.py      │
└─────────────────────────────────────────────────────┘
```

This summary is derived from:
- `state["plan"]` — plan tasks and FeatureSlices
- `state["completed_tasks"]` — what finished
- `state["failed_tasks"]` — what didn't
- `state["critic_gate_result"]` — gate status
- Artifacts written to `.spine/artifacts/` and `.spine/spec/`

#### 2.2 Markdown Artifact Rendering

Each task may produce markdown artifacts (plans, specs, verification reports).
These are written to `.spine/artifacts/plans/{work_item_id}.md` and
`.spine/spec/{work_item_id}.md` by the state machine.

The work detail page must:
1. Detect all artifact files for the current work item
2. Render each as a tab with `st.markdown()` (Streamlit renders markdown natively)
3. Show the file path for reference
4. Provide a "Download" button for each artifact

```python
# Pseudocode for artifact rendering
artifacts = UIApi.get_work_item_artifacts(thread_id)
for artifact in artifacts:
    with tab:
        st.markdown(artifact["content"])
        st.download_button("Download", artifact["content"], file_name=artifact["filename"])
```

#### 2.3 FeatureSlice Outcomes

For each FeatureSlice in the plan, show:
- Status: completed / failed / pending
- Scope: which files/directories were affected
- Acceptance criteria: which passed / which failed
- Agent output: the subagent's result summary

### Implementation

- Modify: `spine/ui/work_detail.py` — add outcome summary, artifact tabs, slice outcomes
- Add: `UIApi.get_work_item_artifacts(thread_id)` — reads artifact files from `.spine/`
- Add: `UIApi.get_feature_slice_outcomes(thread_id)` — extracts slice results from plan
- The existing `write_plan_artifact()` and `write_spec_file()` in
  `spine/core/state_machine.py` already write these files — the UI just reads them

---

## Feature 3: Task Queue — Autonomous Ralph Loop Processing

### Current State

`spine/config/queue.py` provides `TaskQueue` with SQLite and Redis backends.
It supports `enqueue()`, `dequeue()`, `acknowledge()`, `fail()`. However:
- No UI for the queue
- No worker loop that dequeues and processes items
- No integration with the Ralph Loop engine (`spine/workflows/engine.py`)

### Big Picture

A user should be able to enqueue a list of tasks that will be worked on
autonomously. When they return, they can review outcomes: completed tasks,
tasks needing input, failed tasks.

#### 3.1 Queue UI Page (`spine/ui/task_queue.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  ⏳ Task Queue                                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─ Summary ────────────────────────────────────────────────┐  │
│  │  Pending: 5    Running: 1    Completed: 12    Failed: 2  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  [+ Enqueue Task]                                               │
│                                                                  │
│  ┌─ Pending ────────────────────────────────────────────────┐  │
│  │  ○ Build auth middleware     enqueued 2m ago              │  │
│  │  ○ Add rate limiting         enqueued 2m ago              │  │
│  │  ○ Write integration tests   enqueued 1m ago              │  │
│  │  ○ Fix CORS issue            enqueued 30s ago             │  │
│  │  ○ Update API docs           enqueued 30s ago             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Running ────────────────────────────────────────────────┐  │
│  │  🟡 Implement user model    started 5m ago               │  │
│  │     Phase: EXECUTION  Progress: ██████░░░░ 60%            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Completed ──────────────────────────────────────────────┐  │
│  │  ✅ Setup project structure  completed 15m ago   [View]   │  │
│  │  ✅ Create database schema   completed 12m ago   [View]   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ Failed ────────────────────────────────────────────────┐  │
│  │  ❌ Deploy to staging        failed 8m ago       [Retry]  │  │
│  │  ❌ Run load test            failed 20m ago      [Retry]  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  [▶ Start Worker]  [⏸ Pause Worker]  [🗑 Clear Completed]     │
└─────────────────────────────────────────────────────────────────┘
```

**Interactions:**
- "Enqueue Task" opens a form (requirement text, priority, method)
- "View" on completed items navigates to work detail page
- "Retry" on failed items re-enqueues with same payload
- "Start Worker" begins the Ralph Loop dequeue-execute loop
- "Pause Worker" stops dequeuing new items (current item finishes)
- "Clear Completed" removes acknowledged items from the display

#### 3.2 Ralph Loop Worker (`spine/work/ralph_worker.py`)

The worker dequeues items from `TaskQueue` and processes them using
`submit_work()` — the same function the CLI uses.

```python
class RalphLoopWorker:
    """Dequeues tasks from the queue and processes them autonomously.
    
    Uses submit_work() for execution — same code path as CLI.
    Runs in a background thread, managed by the UI.
    """

    def __init__(self, queue: TaskQueue, config_path: str = ".spine/config.yaml"):
        self._queue = queue
        self._config_path = config_path
        self._running = False
        self._current_task: QueueTask | None = None

    def start(self) -> None:
        """Start the worker loop in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Stop dequeuing new tasks (current task finishes)."""
        self._running = False

    def _loop(self) -> None:
        while self._running:
            task = self._queue.dequeue(task_types=["spine_work"])
            if task is None:
                time.sleep(2)  # No work available
                continue

            self._current_task = task
            try:
                result = submit_work_from_config(
                    requirement=task.payload["requirement"],
                    config_path=self._config_path,
                    thread_id=task.id,
                    background=False,  # Synchronous within worker thread
                )
                self._queue.acknowledge(task.id, result=result)
            except Exception as e:
                self._queue.fail(task.id, error=str(e))
            finally:
                self._current_task = None

    @property
    def status(self) -> dict:
        """Current worker status for UI display."""
        return {
            "running": self._running,
            "current_task": self._current_task.id if self._current_task else None,
        }
```

#### 3.3 Queue API Extensions

Add to `UIApi`:

| Method | Purpose |
|--------|---------|
| `enqueue_task(requirement, method, priority)` | Enqueue a task via `TaskQueue` |
| `get_queue_status()` | Summary: pending/running/completed/failed counts |
| `get_queue_items(status)` | List items by status |
| `retry_task(task_id)` | Re-enqueue a failed task |
| `clear_completed()` | Remove acknowledged items from display |
| `start_worker()` | Start RalphLoopWorker |
| `pause_worker()` | Pause RalphLoopWorker |
| `get_worker_status()` | Current worker state |

### Implementation

- Create: `spine/ui/task_queue.py` — queue UI page
- Create: `spine/work/ralph_worker.py` — Ralph Loop worker
- Modify: `spine/core/ui_api.py` — add queue methods
- Modify: `spine/ui/utils.py` — add queue helper functions
- Modify: `spine/ui/app.py` — add Task Queue to navigation
- The queue already exists in `spine/config/queue.py` — no new queue code needed

---

## Feature 4: Agent Resources — Review, Edit, Recreate

### Current State

No UI exists for managing agent resources. Currently managed by:
- `AGENTS.md` files in project root (read by Deep Agents `MemoryMiddleware`)
- `.spine/config.yaml` for provider and MCP server config
- `.spine/knowledge/` for constraints and anti-patterns
- Project rules are scattered (`.cursorrules`, `CLAUDE.md`, `.editorconfig`)

### Required

A UI page to review, edit, or recreate all project-wide agent resources:

#### 4.1 Resource Categories

| Category | Files | Description |
|----------|-------|-------------|
| **AGENTS.md** | `AGENTS.md` | Agent memory and instructions read by DA |
| **Project Rules** | `.cursorrules`, `CLAUDE.md`, `.spine/rules/` | Coding style, conventions |
| **MCP Servers** | `.spine/config.yaml` → `providers.tools` | Tool server configurations |
| **Coding Style** | `.editorconfig`, `pyproject.toml [tool.ruff]` | Linting/formatting rules |
| **Knowledge Base** | `.spine/knowledge/constraints.md`, `.spine/knowledge/anti-patterns.md` | Learned constraints |
| **Provider Config** | `.spine/config.yaml` → `providers` | LLM, memory, storage providers |

#### 4.2 UI Layout (`spine/ui/agent_resources.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  🔧 Agent Resources                                             │
├─────────────────────────────────────────────────────────────────┤
│  Tabs: [ AGENTS.md │ Rules │ MCP │ Style │ Knowledge │ Config ]│
│                                                                  │
│  ── AGENTS.md ──────────────────────────────────────────────    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ # Project Agents                                          │  │
│  │                                                           │  │
│  │ ## Architecture                                           │  │
│  │ - This is a Python project using FastAPI                  │  │
│  │ - Tests use pytest with async support                     │  │
│  │ - Code style: ruff formatter, line length 88              │  │
│  │                                                           │  │
│  │ ## Constraints                                            │  │
│  │ - Never modify migration files after creation             │  │
│  │ - All API endpoints must have OpenAPI schemas             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  [💾 Save]  [🔄 Regenerate from project]  [📄 View History]    │
└─────────────────────────────────────────────────────────────────┘
```

**Interactions:**
- Each tab shows a code editor (`st.code_area` or `st.text_area`) with the
  current content of the resource file
- "Save" writes changes to disk (via `UIApi`)
- "Regenerate" uses the LLM to analyze the project and recreate the resource
  (uses `spine/discovery/` modules which already scan the codebase)
- "View History" shows git log for the file

#### 4.3 Regeneration

The "Regenerate from project" button:
1. Reads the current project structure using `spine/discovery/analyzer.py`
2. Generates a new AGENTS.md (or rules file) using the LLM
3. Shows a diff view (old vs new) for review before saving

This reuses the existing `spine/discovery/` infrastructure:
- `analyzer.py` — codebase analysis
- `mapper.py` — project structure mapping
- `reverse_engineer.py` — inference of conventions

### Implementation

- Create: `spine/ui/agent_resources.py` — resources UI page
- Add: `UIApi.get_agent_resources()` — reads all resource files
- Add: `UIApi.save_agent_resource(category, content)` — writes resource file
- Add: `UIApi.regenerate_agent_resource(category)` — triggers regeneration
- Modify: `spine/ui/app.py` — add Agent Resources to navigation

---

## Feature 5: Spec Driven Development — High-Level Management

### Current State

`spine/workflows/sdd.py` implements the full SDD lifecycle:
SPEC → DESIGN → PLAN → IMPLEMENT → REVIEW → VERIFY

The `SDDWorkflow` class orchestrates all 6 phases with the Ralph Loop engine.
Currently, SDD can only be triggered programmatically — no UI exists.

### Required

A UI page to manage the high-level SDD process:

#### 5.1 SDD Dashboard (`spine/ui/sdd.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  📐 Spec Driven Development                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ── Project Lifecycle ──────────────────────────────────────    │
│                                                                  │
│  SPEC ─→ DESIGN ─→ PLAN ─→ IMPLEMENT ─→ REVIEW ─→ VERIFY      │
│  ✅       ✅        ✅      🟡 Running   ○         ○            │
│                                                                  │
│  ── Current Phase: IMPLEMENT ────────────────────────────────   │
│  Progress: ████████░░░░ 65%                                     │
│  Active FeatureSlices:                                           │
│    🟡 auth-middleware (running, 70%)                             │
│    ○ rate-limiter (pending, depends on auth-middleware)          │
│    ○ integration-tests (pending, depends on auth-middleware)     │
│                                                                  │
│  ── Phase Controls ─────────────────────────────────────────    │
│  [⏸ Pause]  [▶ Resume]  [⟲ Restart Phase]  [🗑 Abort Project]  │
│                                                                  │
│  ── Phase Artifacts ─────────────────────────────────────────   │
│  SPEC: [View] spec.md                                            │
│  DESIGN: [View] architecture.md                                  │
│  PLAN: [View] plan.md  FeatureSlices: 3                         │
│                                                                  │
│  ── Start New SDD Project ──────────────────────────────────    │
│  [Start Spec Driven Project →]                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.2 New SDD Project Form

A streamlined form to start a new SDD project:
- Project name
- Requirement description
- LLM provider selection
- Project type (greenfield / brownfield)
- Whether to use worktrees for parallel implementation

This calls `SDDWorkflow.create_project()` then `SDDWorkflow.execute()` —
same code path as programmatic usage.

#### 5.3 Phase Gate Controls

Each SDD phase has exit criteria. The UI shows:
- Which exit criteria are met / unmet
- A "Force Complete" button to manually approve a phase gate
- A "Rollback" button to return to the previous phase

These controls write to gate result files that the `SDDWorkflow` reads —
same mechanism as the existing critic gate approval.

### Implementation

- Create: `spine/ui/sdd.py` — SDD management UI page
- Add: `UIApi.start_sdd_project(name, requirement, ...)` — creates SDDWorkflow and executes
- Add: `UIApi.get_sdd_status(project_id)` — reads SDD hierarchy state
- Add: `UIApi.control_sdd_phase(project_id, action)` — pause/resume/rollback
- Add: `UIApi.force_phase_gate(project_id, phase)` — approve gate manually
- Modify: `spine/ui/app.py` — add SDD to navigation

---

## Implementation Order

### Phase A: Foundation (utils + API)

1. Add queue and artifact helpers to `spine/ui/utils.py`
2. Add new methods to `spine/core/ui_api.py`
3. Create `spine/work/ralph_worker.py`

### Phase B: Enhanced Existing Pages

4. Enhance `spine/ui/work_detail.py` — outcome summary, artifact rendering
5. Enhance `spine/ui/dashboard.py` — summary cards, chart, quick actions

### Phase C: New Pages

6. Create `spine/ui/task_queue.py`
7. Create `spine/ui/agent_resources.py`
8. Create `spine/ui/sdd.py`

### Phase D: Navigation

9. Wire all new pages into `spine/ui/app.py`

---

## File Map

### New Files

| Path | Purpose |
|------|---------|
| `spine/ui/task_queue.py` | Task queue UI page |
| `spine/ui/agent_resources.py` | Agent resources UI page |
| `spine/ui/sdd.py` | SDD management UI page |
| `spine/work/ralph_worker.py` | Ralph Loop worker for queue processing |

### Modified Files

| Path | Changes |
|------|---------|
| `spine/core/ui_api.py` | Add queue, artifact, resource, SDD methods |
| `spine/ui/utils.py` | Add queue, artifact, resource helper functions |
| `spine/ui/dashboard.py` | Add summary cards, chart, quick actions |
| `spine/ui/work_detail.py` | Add outcome summary, artifact tabs, slice outcomes |
| `spine/ui/app.py` | Add Task Queue, Agent Resources, SDD to navigation |

---

## Testing Checklist

- [ ] Dashboard shows summary cards with correct counts
- [ ] Work detail renders markdown artifacts from `.spine/artifacts/`
- [ ] Work detail shows FeatureSlice outcomes
- [ ] Task Queue enqueues items via `TaskQueue`
- [ ] Ralph Loop worker processes items using `submit_work_from_config()`
- [ ] Agent Resources reads/writes AGENTS.md correctly
- [ ] Agent Resources regeneration produces valid content
- [ ] SDD page shows phase lifecycle progress
- [ ] SDD start creates project and begins execution
- [ ] All UI actions use same code paths as CLI (no duplication)
