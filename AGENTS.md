# SPINE - Agent Instructions

## Project Overview

SPINE is a deterministic AI agent harness with a workflow engine and modular provider architecture. It orchestrates AI agents through a structured lifecycle - specify, plan, implement, verify - using **LangGraph StateGraph** for workflow topology and checkpoint persistence, **Deep Agents** for in-process LLM agent loops, and a **Streamlit dashboard** for visibility and human review.

**Language:** Python 3.12+
**Build:** hatchling
**Package:** `spine-harness`
**CLI entry point:** `spine` -> `spine.cli:main`
**Version:** 0.1.0

---

## Quick Reference

```bash
# Install (dev)
uv sync

# Run tests
pytest tests/unit/         # Unit tests
pytest tests/integration/  # Integration tests

# Lint & format
ruff check spine/ tests/
ruff format spine/ tests/

# Type check
mypy spine/

# Run a single test by name
pytest tests/ -k "test_name_goes_here"

# Visualize and debug the graph
langgraph dev
# -> opens https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

---

## Architecture

```
SPINE WORKFLOW ENGINE (LangGraph StateGraph)
  START -> specify -> plan -> [critic_plan] -> implement -> verify -> END
             ^             ^               ^
             |             |               |
          (critic)     (critic)         (gap_fix loop x2)
             |             |               |
          rework <-------+---------------+

WORK TYPES (all 4 share the same phase sequence):
  task:              SPECIFY -> PLAN -> CRITIC_PLAN -> IMPLEMENT -> VERIFY
  critical_task:     SPECIFY -> CRITIC_SPECIFY -> PLAN -> CRITIC_PLAN -> IMPLEMENT -> VERIFY
  reviewed_task:     same as task, pauses for human approval after critic_plan
  critical_reviewed: same as critical_task, pauses after critic_plan

HUMAN REVIEW (LangGraph interrupt() pattern):
  interrupt_after=["critic_plan"] in submit_work() for reviewed work types
  Human responds via UI: rework / approve / abort
  The router maps these back to the phase sequence

GAP-FIX CYCLE:
  When verification fails with verify_attempts < 2:
    STATUS = "needs_gap_fix"
    gap_plan phase generates a plan for what's missing
    -> implement -> verify (loop, up to 2 times)
  After 2 gap attempts exhausted: routes to needs_review

SUBGRAPH ARCHITECTURE:
  Each phase is its own nested LangGraph StateGraph with:
    - Per-phase SQLite checkpoint (isolated DB per phase)
    - State mapper: ParentState -> SubgraphState
    - Result mapper: SubgraphState -> ParentState update
    - Subgraph builders registered in compose.py

ARTIFACT GATES:
  tasks -> implement: checks plan has artifacts (required for implement)
  NOTE: implement -> verify is NOT gated; verify always runs after implement

CRITIC ROUTING (two-tier: structural -> agent):
  passed          -> next phase
  needs_revision  -> rework previous phase (if retries remain, default max=3)
  needs_review    -> END (flag for human review, resumable via UI or interrupt)
```

**LangGraph is the workflow engine.** The StateGraph defines node topology, conditional edges, and checkpoint persistence. Deep Agents handle the LLM work *inside* each phase subgraph. The workflow progression is entirely LangGraph - there is no separate state machine system.

**LangSmith Studio** can visualize and debug the graph. Run `langgraph dev` from the project root (requires `langgraph.json` + `.env` with `LANGSMITH_API_KEY`), then open the Studio URL. Tracing is automatic when `LANGSMITH_TRACING=true` is set in `.env` (loaded by `spine/config.py` on import).

### Subgraph Architecture

Each phase runs inside its own nested LangGraph subgraph with per-phase SQLite checkpoint databases. This provides:

- **Isolation**: A crash or CancelledError in one phase does not corrupt another phase's checkpoints
- **Per-phase state**: Each subgraph has its own TypedDict state schema (e.g., `VerifySubgraphState`, `TasksSubgraphState`) separate from the parent `WorkflowState`
- **State mapping**: State mappers transform parent state into subgraph state on entry; result mappers transform subgraph output back to parent state on exit

Subgraph builders are registered at import time in `compose.py`:

```
build_verify_subgraph    -> spine/workflow/subgraphs/verify_subgraph.py
build_implement_subgraph -> spine/workflow/subgraphs/implement_subgraph.py
build_tasks_subgraph     -> spine/workflow/subgraphs/tasks_subgraph.py
build_specify_subgraph   -> spine/workflow/subgraphs/specify_subgraph.py
build_plan_subgraph      -> spine/workflow/subgraphs/plan_subgraph.py
build_critic_subgraph    -> spine/workflow/subgraphs/critic_subgraph.py
build_exploration_subgraph -> spine/workflow/subgraphs/exploration_subgraph.py
build_gap_plan_subgraph  -> spine/workflow/subgraphs/gap_plan_subgraph.py
```

Feature flags control subgraph rollout independently per phase (`_SUBGRAPH_ENABLED` dict). Exploration subgraph (`research_manager -> explore -> synthesize`) is enabled for both SPECIFY and PLAN (`_USE_EXPLORATION_SUBGRAPH`).

### Key Modules

| Path | Purpose |
|------|---------|
| `spine/workflow/compose.py` | `build_workflow_graph()` - builds the LangGraph StateGraph from WorkType, defines WORKFLOW_SEQUENCES |
| `spine/workflow/critic_review.py` | `critic_router()` - two-tier critic routing (structural + agent) |
| `spine/workflow/artifact_gate.py` | `make_artifact_gate_node()` + `artifact_gate_router()` - gate as a node (not just a conditional edge) |
| `spine/workflow/registry.py` | PhaseRegistry - maps phase names to node functions and agent builders |
| `spine/workflow/studio.py` | Entry points for LangSmith Studio (one compiled graph per WorkType) |
| `spine/workflow/subgraph_wrapper.py` | `make_subgraph_node()` + `make_success_result_mapper()` - wraps subgraphs as parent nodes |
| `spine/workflow/subgraph_state.py` | Per-phase subgraph TypedDict state schemas |
| `spine/workflow/subgraphs/` | Phase subgraph implementations (see list below) |
| `spine/workflow/slice_scheduler.py` | Topological sort for feature slice execution waves via `graphlib.TopologicalSorter` |
| `spine/workflow/phase_progress.py` | `mark_phase_started()` - marks work entries when a phase begins |
| `spine/work/dispatcher.py` | `submit_work()`, `resume_work()`, `get_work_status()`, `list_work()` - unified CLI+UI entry points |
| `spine/work/ralph_worker.py` | RalphLoopWorker - background queue processor (singleton) |
| `spine/work/plan_resolver.py` | Parses plan artifacts into structured work units |
| `spine/phases/specify.py` | SPECIFY phase node function + imports subgraph |
| `spine/phases/plan.py` | PLAN phase node function + imports subgraph |
| `spine/phases/tasks.py` | TASKS phase node function |
| `spine/phases/implement.py` | IMPLEMENT phase node function |
| `spine/phases/verify.py` | VERIFY phase node function |
| `spine/phases/critic.py` | CRITIC phase node function |
| `spine/phases/gap_plan.py` | GAP_PLAN phase node function |
| `spine/agents/factory.py` | `build_phase_agent()` - single entry point for all phase agent construction |
| `spine/agents/helpers.py` | `resolve_model()`, `debug_enabled()`, `extract_response()` - shared utilities |
| `spine/agents/profile.py` | SPINE HarnessProfile - replaces DA base prompt with phase-executor framing |
| `spine/agents/retry.py` | `invoke_with_retry()` - exponential backoff for transient LLM API errors |
| `spine/agents/context.py` | `SpineContext` - typed per-run context that propagates to subagents |
| `spine/agents/artifacts.py` | Artifact materialization to disk + inline preview builder |
| `spine/agents/interpreter.py` | Code interpreter middleware (optional, per-phase) |
| `spine/agents/skills_resolver.py` | `resolve_skills()`, `resolve_memory()` - loads phase-specific skills and memory |
| `spine/agents/backend.py` | `build_backend()` - creates DA backend with CompositeBackend + cross-work memory |
| `spine/agents/debug_callback.py` | LLM debug logging callback (enabled via `--debug-llm` or `SPINE_DEBUG_LLM`) |
| `spine/agents/subagents.py` | SubAgent spec builders for DA `task` tool delegation |
| `spine/agents/tool_schema_validator.py` | Rebound loop middleware - self-corrects when tool args don't match schema |
| `spine/agents/exploration_agents.py` | Lightweight agent functions for the exploration subgraph (research_manager, explore) |
| `spine/agents/specify_agent.py` | SPECIFY phase agent builder |
| `spine/agents/specify_tools.py` | SPECIFY phase purpose-built tools |
| `spine/agents/plan_agent.py` | PLAN phase agent builder |
| `spine/agents/plan_tools.py` | PLAN phase purpose-built tools |
| `spine/agents/tasks_agent.py` | TASKS phase agent builder |
| `spine/agents/tasks_tools.py` | TASKS phase purpose-built tools |
| `spine/agents/implement_agent.py` | IMPLEMENT phase agent builder |
| `spine/agents/implement_tools.py` | IMPLEMENT phase purpose-built tools |
| `spine/agents/verify_agent.py` | VERIFY phase agent builder |
| `spine/agents/gap_plan_agent.py` | GAP_PLAN phase agent builder |
| `spine/agents/context_editing.py` | Structured context editing - surgical file modifications |
| `spine/agents/artifact_validation.py` | Validates artifact quality/formatting |
| `spine/critic/agent.py` | Critic Deep Agent builder (structural + agent tiers) |
| `spine/models/state.py` | `WorkflowState` (TypedDict) with annotated reducers, `PhaseResult` |
| `spine/models/enums.py` | `PhaseName`, `WorkType`, `ReviewStatus`, `TaskStatus` |
| `spine/models/types.py` | `WorkUnit`, `PlanDecomposition`, `WorkSpawnSpec`, `Task`, `Artifact`, `ReviewFeedback` |
| `spine/persistence/checkpoint.py` | `CheckpointStore` - LangGraph SQLite-backed persistence |
| `spine/persistence/artifacts.py` | `ArtifactStore` - file-based artifact storage per work item |
| `spine/services/audit_service.py` | Audit event logging to SQLite |
| `spine/config.py` | `SpineConfig` - loads `.spine/config.yaml` + `.env` on import |
| `spine/exceptions.py` | Custom exceptions: `WorkflowError`, `CriticError`, `TransientAPIError`, etc. |
| `spine/ui_api/api.py` | `UIApi` - sole read/write interface for Streamlit UI |
| `spine/ui/app.py` | Streamlit app entry point with WebSocket push |
| `spine/ui/ws_bus.py` | Async pub/sub event bus for state changes (singleton) |
| `spine/ui/ws_server.py` | WebSocket server for Streamlit client push notifications |
| `spine/ui/ws_component.py` | WebSocket client component for Streamlit pages |
| `spine/ui/_pages/` | 9 UI pages: dashboard, work_submit, work_detail, work_history, human_review, queue, audit_log, config_view, spec_planning |
| `spine/cli/__init__.py` | Click commands: `run`, `status`, `list` (no `resume`/`worker`/`ui` CLI) |
| `spine/mcp/client.py` | MCP tool loader using `langchain-mcp-adapters`, namespaced tools |
| `spine/mcp/tools.py` | MCP tool configuration |
| `spine/mcp/defaults.py` | Default MCP server config |
| `spine/cli/commands/` | Additional CLI command modules |
| `spine/cli/renderers/` | CLI output rendering modules |
| `spine/config/` | Config loading and validation modules |

---

## Coding Conventions

### Style

- **Line length:** 100 characters (ruff configured)
- **Target Python:** 3.12+
- **Formatting:** `ruff format`
- **Linting:** `ruff check`
- **Type annotations:** Use modern syntax (`list[str]` not `List[str]`, `str | None` not `Optional[str]`)
- **Imports:** `from __future__ import annotations` in files with forward references
- **Grouping:** stdlib -> third-party -> relative, separated by blank lines
- **Relative imports:** Use `..` notation for cross-module imports (`from ..models.state import ...`)

### Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `WorkflowState`, `PhaseRegistry` |
| Enums | PascalCase (str, Enum) | `PhaseName.PLAN`, `WorkType.TASK` |
| Functions/Methods | snake_case | `submit_work()`, `build_workflow_graph()` |
| Constants | SCREAMING_SNAKE | `DEFAULT_MAX_RETRIES`, `WORKFLOW_SEQUENCES` |
| Private members | Leading underscore | `_load_dotenv()`, `_handle_review_outcome()` |
| Factory functions | `create_` prefix | `create_deep_agent()`, `make_artifact_gate_fn()` |
| Test classes | `Test` prefix | `TestArtifactGate`, `TestCriticRouter` |

### Docstrings

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Module-level docstrings explaining purpose and key design decisions
- Section headers using `# - Section Name -` comment style (en-dash borders)

### Data Models

- **Dataclasses** for internal types: `@dataclass` with typed fields, `field(default_factory=list)` for mutable defaults
- **TypedDict** for state dicts: `WorkflowState(TypedDict)` used by LangGraph with annotated reducers
- **Pydantic** for API schemas and structured outputs: `WorkUnit`, `PlanDecomposition`, `WorkSpawnSpec`
- **Enum** with `(str, Enum)` base for JSON-serializable enumerations

### Error Handling

- Specific exceptions first, broad `Exception` last
- Custom exceptions: `WorkflowError`, `CriticError`, `MaxRetriesExceeded`, `TransientAPIError`, `PromptRequestError`
- Graceful fallback pattern: catch, log warning, return safe default
- Retry with exponential backoff for transient failures via `invoke_with_retry()`
- Defensive returns: empty list/dict instead of raising on missing data

### Async

- `async def` for all I/O-bound operations (dispatcher, checkpoint store)
- `AsyncIterator[str]` for streaming graph output
- `asyncio.run()` for sync entry points (CLI, worker thread)
- `@pytest.mark.asyncio` for async tests
- Never block the event loop with sync I/O

### Threading

- `threading.Lock()` for shared mutable state (RalphLoopWorker singleton, queue access)
- Daemon threads for background workers
- `threading.Event` for graceful shutdown signaling

---

## Key Design Rules

### 1. Provider Resolution: Config, Not State

Providers must be resolved from `config["configurable"]["providers"]`, never from `WorkflowState`. Storing providers in state causes serialization failures after LangGraph checkpointing. The `_get_providers_from_config()` function is the canonical way to obtain providers inside phase functions.

### 2. Zero Duplication: CLI and UI Share Code Paths

Both CLI commands and Streamlit pages call the same backend functions:
- **Writes:** Both go through `submit_work()` / `resume_work()` in `spine/work/dispatcher.py`
- **Reads:** Both use `UIApi` in `spine/ui_api/api.py`
- UI pages must never import directly from `spine/workflow/` or `spine/phases/`

### 3. One Deep Agent Per Phase

Each SPINE phase constructs its own `create_deep_agent()` via `build_phase_agent()` with phase-specific system prompts, tools, middleware, skills, and memory. This preserves phase isolation while giving each phase the full DA infrastructure. Each phase now has its own dedicated agent module (e.g., `spine/agents/specify_agent.py`, `spine/agents/plan_agent.py`) and purpose-built tool modules.

### 4. Critic Gate Is Structural, Not Prompted

The critic gate enforces that a phase cannot proceed without `PASSED` status. This is enforced in `critic_router()` conditional edge logic, not by prompting the LLM. When retries are exceeded, it routes to `needs_review -> END`.

### 5. Artifact Gates Prevent Empty Progression

Before `implement` runs, an artifact gate checks that `plan` produced artifacts. If the gate fails, it routes to `needs_review -> END` instead of proceeding. There is NO artifact gate between `implement` and `verify` - verify always runs after implement. If implement produced nothing, verify can detect and report that; there is no reason for a human review gate between those two phases.

### 6. SPINE Base Prompt Replaces DA Default

`spine/agents/profile.py` registers `HarnessProfile(base_system_prompt=SPINE_BASE_PROMPT)` for `openrouter`, `openai`, and `anthropic` providers. This replaces the DA `BASE_AGENT_PROMPT` with a phase-executor framing. The prompt assembly order is: `USER` (phase system_prompt) -> `CUSTOM` (SPINE_BASE_PROMPT) -> `SUFFIX` (none).

### 7. OpenRouter Session Tracking

`resolve_model(config, session_id=work_id)` in `spine/agents/helpers.py` returns a pre-built `ChatOpenRouter` instance with `session_id` set when the model is OpenRouter and a work_id is provided. This groups all LLM requests for a work item into a single session on the OpenRouter dashboard.

### 8. Environment Variables via .env

`spine/config.py` calls `load_dotenv()` on import, loading `.env` from the project root. This ensures `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `OPENROUTER_API_KEY`, etc. are available before any LangGraph or Deep Agents code reads them. Uses `override=False` so manually-set env vars take precedence.

### 9. Phase Nodes Must Return Complete State Updates

Every phase node function must return `status` and `prompt_request` in its output dict, even on error paths. Missing fields cause `_update_work_progress()` to record empty status in the audit log and work entries DB. The critic node was a previous offender - all paths now include `"status": "running"` and `"prompt_request": None`.

### 10. Subgraph State Isolation

Each phase runs inside its own nested LangGraph subgraph with per-phase SQLite checkpoints. State mappers transform parent state into subgraph state on entry; result mappers transform subgraph output back to parent state on exit. Never assume data is directly in the parent state - check if the phase uses subgraph state and whether it needs mapping.

### 11. WebSocket Push Notifications

State changes are published to `spine.ui.ws_bus.WSEventBus` (singleton). The `spine.ui.ws_server` WebSocket server fans them to Streamlit clients. UI clients connect via `spine.ui.ws_component` and receive filtered push notifications instead of polling.

### 12. Feature Flag Controlled Rollout

Phase subgraph migration uses `_SUBGRAPH_ENABLED` dict per phase. Exploration subgraph uses `_USE_EXPLORATION_SUBGRAPH` dict. New features should follow this pattern for safe incremental deployment.

### 13. MCP Tools Are Namespaced

MCP tools are loaded via `langchain-mcp-adapters` (not the custom client). All tools are namespaced with the server name to prevent collisions (e.g., `mcp_codebase-index_find_symbol`). Configuration lives in `spine/mcp/`.

### 14. Tool Schema Validation Middleware

`spine/agents/tool_schema_validator.py` implements a rebound loop: when a model generates a tool call with arguments that don't match the registered tool's input schema, the middleware intercepts the call, constructs a precise error message, and retries with corrected arguments. This prevents silent failures from schema mismatches.

---

## Testing

### Framework

- **pytest** with `pytest-asyncio`
- Tests in `tests/` directory at project root
- Class-based organization: `class TestArtifactGate:` with descriptive method names

### Patterns

```python
# Path manipulation for imports (used in most test files)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mocking
from unittest.mock import MagicMock, AsyncMock, patch

# Isolated file system
with tempfile.TemporaryDirectory() as tmpdir:
    ...

# Exception testing
with pytest.raises(ExpectedError):
    ...

# Async tests
@pytest.mark.asyncio
async def test_stream():
    ...
```

### What to Test

- Critic routing: passed / needs_revision / needs_review (with retry exhaustion)
- Artifact gate: proceed when artifacts exist, needs_review when empty/missing/short
- Workflow composition: correct nodes and edges per WorkType
- Phase status propagation: every node returns `status` + `prompt_request`
- Resume: validates needs_review status, rejects unknown/completed work
- Retry logic: transient errors retry, permanent errors raise immediately
- Subgraph state mapping: parent -> subgraph -> parent transformation
- Slice scheduler: topological sort correctness for execution waves
- WebSocket bus: publish/subscribe from sync and async contexts

### What NOT to Test

- Third-party library internals (LangGraph, Deep Agents)
- LLM output quality (that's evaluation, not unit testing)
- Streamlit rendering (test the UIApi methods instead)

---

## Configuration

### `.spine/config.yaml`

```yaml
spine:
  checkpoint_path: .spine/spine.db
  workspace_root: /home/pat/Projects/spine
  interpreter_enabled: true

providers:
  llm:
    - name: default
      type: deepagents-model
      model: openrouter:qwen/qwen3.6-27b
      priority: 1
      enabled: true
```

### `.env` (project root, loaded by `spine/config.py`)

```
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com
LANGSMITH_PROJECT=spine
```

### `langgraph.json` (project root, used by `langgraph dev`)

```json
{
  "dependencies": ["."],
  "graphs": {
    "spec": "spine.workflow.studio:spec_graph",
    "critical_spec": "spine.workflow.studio:critical_spec_graph",
    "quick": "spine.workflow.studio:quick_graph",
    "critical_quick": "spine.workflow.studio:critical_quick_graph"
  },
  "env": ".env"
}
```

### Runtime Data (`.spine/`)

| Path | Purpose |
|------|---------|
| `spine.db` | LangGraph SQLite checkpoints (WAL mode) |
| `audit.db` | Audit event log |
| `queue.db` | Task queue (SQLite backend) |
| `work_entries.db` | Work item tracking (status, phase, result) |
| `config.yaml` | Runtime configuration |
| `artifacts/` | Phase output files (spec, plan, tasks, implement, verify, critic) |
| `state/` | Current work state |
| `events/` | Event data |
| `knowledge/` | Knowledge base files |
| `checkpoints/` | Per-phase subgraph checkpoints (`<work_id>/<phase>.db`) |

---

## Common Workflows

### Adding a New Workflow Phase

1. Add phase name to `PhaseName` enum in `spine/models/enums.py`
2. Create the subgraph in `spine/workflow/subgraphs/new_phase_subgraph.py` with `build_new_phase_subgraph()` function
3. Create the phase node function in `spine/phases/new_phase.py` that delegates to the subgraph
4. Create the phase agent builder in `spine/agents/new_phase_agent.py` using `build_phase_agent()`
5. Register the subgraph builder in `spine/workflow/compose.py` (`register_subgraph_builder`)
6. Add the phase to the appropriate `WORKFLOW_SEQUENCES` entries in `compose.py`
7. If the phase needs an artifact gate before it, add the gate logic in `compose.py`
8. Add to `_SUBGRAPH_ENABLED` dict in `compose.py` for rollout control
9. Write tests in appropriate test files

### Adding a New UI Page

1. Create `spine/ui/_pages/new_page.py` with a `render(api: UIApi)` function
2. Register the page in `spine/ui/pages.py` and `spine/ui/app.py`
3. All data access MUST go through `UIApi` in `spine/ui_api/api.py`
4. Add helper functions to `spine/ui/utils.py` if needed
5. Verify zero-duplication: the same action must work from CLI
6. Test the `UIApi` methods, not the Streamlit rendering

---

## Pitfalls

- **workspace_root is CRITICAL - never break its resolution** - Every Deep Agent's `LocalShellBackend` uses `workspace_root` as its `root_dir`. If this resolves to a wrong path (e.g. `/root`, `/tmp`), agents get `Permission denied` errors and the per-phase checkpointer fails silently. The resolution chain is: `SPINE_WORKSPACE_ROOT` env var -> `.spine/config.yaml` -> `_find_workspace_root()` (CWD walk -> package-dir walk -> CWD fallback). When CWD is wrong (Streamlit, systemd, cron), the package-directory search must find `.spine/`. **Any change to `SpineConfig._find_workspace_root()` or `build_backend()` must be tested from a non-project CWD.**
- **Phase node functions MUST be async** - All phase node functions (`call_specify`, `call_plan`, etc.) must be `async def` and use `ainvoke_with_retry` (not `invoke_with_retry`). Sync nodes run in LangGraph's thread pool, which breaks `asyncio.Lock` objects in the checkpointer (`AsyncSqliteSaver.lock` is bound to the main event loop). When subagents inherit the parent checkpointer through `config["configurable"]`, they encounter `RuntimeError: is bound to a different event loop`. Async nodes stay on the same event loop throughout, avoiding this entirely.
- **Never store providers in WorkflowState** - LangGraph's checkpointer serializes state to SQLite, and provider objects (LLM clients, HTTP sessions) are not serializable. Use `config["configurable"]["providers"]`.
- **Phase nodes must return status and prompt_request** - Every return dict from a phase function must include `"status"` and `"prompt_request"`. Missing these causes empty entries in the audit log and work progress updates. The critic node was a previous offender.
- **Artifact gates use plan artifacts for implement gate** - The gate checks that the plan phase produced artifacts before allowing implement to run. This replaces the old "tasks produced artifacts" check since TASKS phase no longer gates implement in any work type.
- **OpenCode ACP + vLLM returns tiny responses** - Known protocol mismatch. Use `DeepAgentsModelProvider` with `init_chat_model()` directly for local models instead.
- **RalphLoopWorker is a singleton** - Access via `get_worker()`, don't instantiate directly. Thread-safety enforced by `_WORKER_LOCK`.
- **SQLite WAL mode** - Checkpoint DB uses WAL (Write-Ahead Logging). Multiple readers OK, one writer at a time. Don't hold long-running write transactions.
- **TypedDict state keys** - `WorkflowState` is a TypedDict. New keys must be added to the type definition or LangGraph will silently drop them.
- **SpineContext must be a Pydantic BaseModel** - LangGraph's config schema creates a Pydantic field `(SpineContext, None)` for the `context_schema`. When checkpointing, Pydantic tries to serialize this field. A plain dataclass produces `PydanticSerializationUnexpectedValue` warnings because the field type expects `None` but receives a dataclass instance. Using `BaseModel` lets Pydantic serialize it natively.
- **from __future__ import annotations** - Required in files using `str | None` or forward references, but can break Pydantic models at runtime. Use with care in model files.
- **Per-phase checkpoint isolation** - Each phase subgraph writes to its own SQLite database. Do not assume checkpoints are shared across phases. The cache key for checkpointer caching is `{work_id}/{phase}`.
- **State mappers and result mappers must stay in sync** - When you change a phase's subgraph state, ensure both the state mapper (parent -> subgraph) and result mapper (subgraph -> parent) are updated. Mismatches cause silent data loss.
- **Pre-built ChatOpenRouter must apply ProviderProfile kwargs** - When `resolve_model` returns a `ChatOpenRouter` instance (OpenRouter + session_id), the DA `ProviderProfile` factory chain is skipped. If you add new kwargs to the OpenRouter ProviderProfile, verify they're also handled in `_build_openrouter_model()`.
- **.env must be at project root** - `spine/config.py` loads `.env` from `Path.cwd()`. If running from a different directory, the env vars won't be loaded. `langgraph dev` handles this via `langgraph.json`.
- **Resume re-runs the full graph** - `resume_work()` re-invokes the entire StateGraph from START with accumulated state + human feedback. It does NOT resume from the exact checkpoint position. For true mid-graph resume, use LangGraph's `interrupt()` + `Command(resume=...)` pattern instead.
- **Critic node current_phase is always "critic"** - The critic returns `current_phase: PhaseName.CRITIC.value` regardless of which phase it's reviewing (e.g. `critic_plan` node still returns `current_phase: "critic"`). The audit log uses the node name (`critic_plan`) which is correct, but `current_phase` in state is generic.
- **Stall detection uses token-level streaming** - The dispatcher streams the graph with `stream_mode=["updates", "messages"]` and `subgraphs=True`. This means the stall timer (`SPINE_STALL_TIMEOUT`, default 120s) resets on every LLM token, not just on node completions. Only a genuine connection drop triggers a stall - legitimately long agent runs keep the timer alive. Do NOT revert to `stream_mode="values"` or remove `subgraphs=True` without also increasing the stall timeout, or long agent runs will be falsely marked as stalled.
- **GAP_PLAN phase has no retry counting** - The gap plan phase always starts with `retry_count: 0`. Gap fix cycles are controlled by `verify_attempts` in the parent state (max 2), not by the critic retry mechanism.
- **Review work types use interrupt(), not critic routing** - `reviewed_task` and `critical_reviewed_task` share identical phase sequences with their non-reviewed counterparts. The human review pause is handled via `interrupt_after=["critic_plan"]` in `submit_work()`, not via critic routing. The human feedback router (`_make_human_review_router`) maps rework/approve/abort actions back to the phase sequence.
- **Slice scheduler uses graphlib.TopologicalSorter** - Feature slices in the IMPLEMENT phase are grouped into parallel execution waves. Slices with dependencies on other slices are placed in later waves. This is computed by the `tasks_subgraph` and stored as `execution_waves` in parent state.
- **WS_EVENT_BUS singleton is thread-safe** - The WebSocket event bus can be published to from sync code (uses `asyncio.run_coroutine_threadsafe`). But subscribers must run in an async context (the WS server). Events published before the event loop is bound are buffered and flushed automatically.

---

## Dependencies

| Category | Packages |
|----------|----------|
| Core | langgraph, langgraph-checkpoint-sqlite, langgraph-supervisor, deepagents>=0.5.0 |
| LLM | langchain, langchain-openai, langchain-openrouter, openai |
| Data | pydantic>=2.0, pyyaml, sqlite-utils |
| CLI | click, rich |
| UI | streamlit>=1.30, websockets |
| Tracing | langsmith, python-dotenv, langgraph-cli[inmem] |
| Dev | pytest, pytest-asyncio, ruff |
| MCP | langchain-mcp-adapters |
