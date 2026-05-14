# SPINE — Agent Instructions

## Project Overview

SPINE is a deterministic AI agent harness with a workflow engine and modular provider architecture. It orchestrates AI agents through a structured lifecycle — specify, plan, implement, verify — using **LangGraph StateGraph** for workflow topology and checkpoint persistence, **Deep Agents** for in-process LLM agent loops, and a **Streamlit dashboard** for visibility and human review.

**Language:** Python 3.12+  
**Build:** hatchling  
**Package:** `spine-harness`  
**CLI entry point:** `spine` → `spine.cli:main`

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
# → opens https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

---

## Architecture

```
SPINE WORKFLOW ENGINE (LangGraph StateGraph)
  START → specify → plan → [critic_plan] → tasks → implement → verify → END
              ↘ rework ↗            ↘ needs_review → END
  │                                       ↘ artifact_gate → END
  │
  └─ each node delegates to → Deep Agents Runtime
                               ├── SubAgents (from FeatureSlices)
                               ├── Middleware (interpreter, summarization)
                               ├── Skills (phase-specific, RLM)
                               └── Backend (LocalShellBackend)

WORK TYPES:
  quick:           TASKS → IMPLEMENT → VERIFY
  critical_quick:  TASKS → CRITIC_TASKS → IMPLEMENT → VERIFY
  spec:            SPECIFY → PLAN → CRITIC_PLAN → TASKS → IMPLEMENT → VERIFY
  critical_spec:   SPECIFY → CRITIC_SPECIFY → PLAN → CRITIC_PLAN →
                   TASKS → CRITIC_TASKS → IMPLEMENT → VERIFY

ARTIFACT GATES:
  implement → verify:  checks implement has non-empty artifacts (≥50 chars)
  tasks → implement:   checks tasks has non-empty artifacts (≥50 chars)

CRITIC ROUTING (two-tier: structural → agent):
  passed          → next phase
  needs_revision  → rework previous phase (if retries remain, default max=3)
  needs_review    → END (flag for human review, resumable via UI or CLI)

HUMAN REVIEW RESUME:
  rework  → re-runs workflow with human feedback appended
  approve → re-runs workflow with "passed" feedback injected
```

**LangGraph is the workflow engine.** The StateGraph defines node topology, conditional edges, and checkpoint persistence. Deep Agents handle the LLM work *inside* each node function. The workflow progression (which phase runs next, critic routing, rework loops, artifact gates) is entirely LangGraph — there is no separate state machine system.

**LangSmith Studio** can visualize and debug the graph. Run `langgraph dev` from the project root (requires `langgraph.json` + `.env` with `LANGSMITH_API_KEY`), then open the Studio URL. Tracing is automatic when `LANGSMITH_TRACING=true` is set in `.env` (loaded by `spine/config.py` on import).

### Key Modules

| Path | Purpose |
|------|---------|
| `spine/workflow/compose.py` | `build_workflow_graph()` — builds the LangGraph StateGraph from WorkType |
| `spine/workflow/critic_review.py` | Two-tier critic (structural + agent) with `critic_router` |
| `spine/workflow/artifact_gate.py` | Pre-check that prior phase produced artifacts before proceeding |
| `spine/workflow/registry.py` | PhaseRegistry — maps phase names to node functions and agent builders |
| `spine/workflow/studio.py` | Entry points for LangSmith Studio (one compiled graph per WorkType) |
| `spine/work/dispatcher.py` | `submit_work()`, `resume_work()` — unified CLI+UI entry points |
| `spine/work/ralph_worker.py` | RalphLoopWorker — background queue processor (singleton) |
| `spine/phases/specify.py` | SPECIFY phase node + agent builder |
| `spine/phases/plan.py` | PLAN phase node + agent builder |
| `spine/phases/tasks.py` | TASKS phase node — decomposes plan into feature slices |
| `spine/phases/implement.py` | IMPLEMENT phase node — generates code per feature slice |
| `spine/phases/verify.py` | VERIFY phase node — confirms implementation meets requirements |
| `spine/phases/critic.py` | CRITIC phase node — two-tier review (structural + agent) |
| `spine/agents/factory.py` | `build_phase_agent()` — single entry point for all phase agent construction |
| `spine/agents/helpers.py` | `resolve_model()`, `debug_enabled()`, `extract_response()` — shared utilities |
| `spine/agents/profile.py` | SPINE HarnessProfile — replaces DA base prompt with phase-executor framing |
| `spine/agents/retry.py` | `invoke_with_retry()` — exponential backoff for transient LLM API errors |
| `spine/agents/context.py` | `SpineContext` — typed per-run context that propagates to subagents |
| `spine/agents/artifacts.py` | Artifact materialization to disk + inline preview builder |
| `spine/agents/interpreter.py` | Code interpreter middleware (optional, per-phase) |
| `spine/agents/skills_resolver.py` | `resolve_skills()`, `resolve_memory()` — loads phase-specific skills and memory |
| `spine/agents/backend.py` | `build_backend()` — creates DA backend for agent tool access |
| `spine/agents/backend_memory.py` | Cross-work memory via LangGraph InMemoryStore |
| `spine/agents/debug_callback.py` | LLM debug logging callback (enabled via `--debug-llm` or `SPINE_DEBUG_LLM`) |
| `spine/agents/subagents.py` | SubAgent spec builders for DA `task` tool delegation |
| `spine/critic/agent.py` | Critic Deep Agent builder |
| `spine/models/state.py` | `WorkflowState` (TypedDict) — state schema for the StateGraph |
| `spine/models/enums.py` | `PhaseName`, `WorkType`, `ReviewStatus`, `TaskStatus` |
| `spine/models/types.py` | `Task`, `Artifact`, `ReviewFeedback`, `PromptRequest` dataclasses |
| `spine/persistence/checkpoint.py` | `CheckpointStore` — LangGraph SQLite-backed persistence |
| `spine/persistence/artifacts.py` | `ArtifactStore` — file-based artifact storage per work item |
| `spine/services/audit_service.py` | Audit event logging to SQLite |
| `spine/config.py` | `SpineConfig` — loads `.spine/config.yaml` + `.env` on import |
| `spine/exceptions.py` | Custom exceptions: `WorkflowError`, `CriticError`, `TransientAPIError`, etc. |
| `spine/ui_api/api.py` | UIApi — sole read/write interface for Streamlit UI |
| `spine/ui/app.py` | Streamlit app entry point with WebSocket push |
| `spine/ui/_pages/` | 8 UI pages: dashboard, work_submit, work_detail, work_history, human_review, queue, audit_log, config_view |
| `spine/cli/__init__.py` | Click commands: `run`, `status`, `list`, `resume`, `worker`, `ui` |

### Empty directories (planned/future, no .py files)

`spine/adapters/`, `spine/swarm/`, `spine/hive/`, `spine/providers/`, `spine/middleware/`, `spine/discovery/`, `spine/git/`, `spine/github/`, `spine/jobs/`, `spine/prompts/`, `spine/core/`, `spine/utils/`, `spine/workflows/`, `spine/skills/`

---

## Coding Conventions

### Style

- **Line length:** 100 characters (ruff configured)
- **Target Python:** 3.12+
- **Formatting:** `ruff format`
- **Linting:** `ruff check`
- **Type annotations:** Use modern syntax (`list[str]` not `List[str]`, `str | None` not `Optional[str]`)
- **Imports:** `from __future__ import annotations` in files with forward references
- **Grouping:** stdlib → third-party → relative, separated by blank lines
- **Relative imports:** Use `..` notation for cross-module imports (`from ..models.state import ...`)

### Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `WorkflowState`, `PhaseRegistry` |
| Enums | PascalCase (str, Enum) | `PhaseName.PLAN`, `WorkType.SPEC` |
| Functions/Methods | snake_case | `submit_work()`, `build_workflow_graph()` |
| Constants | SCREAMING_SNAKE | `DEFAULT_MAX_RETRIES`, `WORKFLOW_SEQUENCES` |
| Private members | Leading underscore | `_load_dotenv()`, `_handle_review_outcome()` |
| Factory functions | `create_` prefix | `create_deep_agent()`, `make_artifact_gate_fn()` |
| Test classes | `Test` prefix | `TestArtifactGate`, `TestCriticRouter` |

### Docstrings

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Module-level docstrings explaining purpose and key design decisions
- Section headers using `# ── Section Name ──` comment style (en-dash borders)

### Data Models

- **Dataclasses** for internal types: `@dataclass` with typed fields, `field(default_factory=list)` for mutable defaults
- **TypedDict** for state dicts: `WorkflowState(TypedDict)` used by LangGraph with annotated reducers
- **Pydantic** for API schemas (if backend added)
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
- **Writes**: Both go through `submit_work()` / `resume_work()` in `spine/work/dispatcher.py`
- **Reads**: Both use `UIApi` in `spine/ui_api/api.py`
- UI pages must never import directly from `spine/workflow/` or `spine/phases/`

### 3. One Deep Agent Per Phase

Each SPINE phase constructs its own `create_deep_agent()` via `build_phase_agent()` with phase-specific system prompts, tools, middleware, skills, and memory. This preserves phase isolation while giving each phase the full DA infrastructure.

### 4. Critic Gate Is Structural, Not Prompted

The critic gate enforces that a phase cannot proceed without `PASSED` status. This is enforced in `critic_router()` conditional edge logic, not by prompting the LLM. When retries are exceeded, it routes to `needs_review → END`.

### 5. Artifact Gates Prevent Empty Progression

Before `verify` runs, an artifact gate checks that `implement` produced non-empty artifacts (≥50 chars). Before `implement` runs, it checks `tasks`. If the gate fails, it routes to `needs_review → END` instead of proceeding. This prevents the workflow from reaching verify with no implementation.

### 6. SPINE Base Prompt Replaces DA Default

`spine/agents/profile.py` registers `HarnessProfile(base_system_prompt=SPINE_BASE_PROMPT)` for `openrouter`, `openai`, and `anthropic` providers. This replaces the DA `BASE_AGENT_PROMPT` with a phase-executor framing. The prompt assembly order is: `USER` (phase system_prompt) → `CUSTOM` (SPINE_BASE_PROMPT) → `SUFFIX` (none).

### 7. OpenRouter Session Tracking

`resolve_model(config, session_id=work_id)` in `spine/agents/helpers.py` returns a pre-built `ChatOpenRouter` instance with `session_id` set when the model is OpenRouter and a work_id is provided. This groups all LLM requests for a work item into a single session on the OpenRouter dashboard.

### 8. Environment Variables via .env

`spine/config.py` calls `load_dotenv()` on import, loading `.env` from the project root. This ensures `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `OPENROUTER_API_KEY`, etc. are available before any LangGraph or Deep Agents code reads them. Uses `override=False` so manually-set env vars take precedence.

### 9. Phase Nodes Must Return Complete State Updates

Every phase node function must return `status` and `prompt_request` in its output dict, even on error paths. Missing fields cause `_update_work_progress()` to record empty status in the audit log and work entries DB. The critic node was a previous offender — all paths now include `"status": "running"` and `"prompt_request": None`.

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

---

## Common Workflows

### Adding a New Workflow Phase

1. Add phase name to `PhaseName` enum in `spine/models/enums.py`
2. Create the phase module in `spine/phases/new_phase.py` with a `call_new_phase()` function that returns `{artifacts, current_phase, status, prompt_request}`
3. Self-register in the registry at module bottom: `_registry.register(name=PhaseName.NEW_PHASE.value, call_fn=call_new_phase, build_agent_fn=build_new_agent)`
4. Add the phase to the appropriate `WORKFLOW_SEQUENCES` in `spine/workflow/compose.py`
5. If the phase needs an artifact gate before it, add to `ARTIFACT_GATE_MAP` in compose.py
6. Create the agent builder in `spine/agents/new_phase_agent.py` using `build_phase_agent()`
7. Add an entry point in `spine/workflow/studio.py` if needed for Studio
8. Write tests in `tests/unit/test_workflow_gates.py`

### Adding a New UI Page

1. Create `spine/ui/_pages/new_page.py` with a `render(api: UIApi)` function
2. Register the page in `spine/ui/pages.py` and `spine/ui/app.py`
3. All data access MUST go through `UIApi` in `spine/ui_api/api.py`
4. Add helper functions to `spine/ui/utils.py` if needed
5. Verify zero-duplication: the same action must work from CLI
6. Test the `UIApi` methods, not the Streamlit rendering

---

## Pitfalls

- **Never store providers in `WorkflowState`** — LangGraph's checkpointer serializes state to SQLite, and provider objects (LLM clients, HTTP sessions) are not serializable. Use `config["configurable"]["providers"]`.
- **Phase nodes must return `status` and `prompt_request`** — Every return dict from a phase function must include `"status"` and `"prompt_request"`. Missing these causes empty entries in the audit log and work progress updates. The critic node was a previous offender.
- **Artifact gates use ≥50 char threshold** — Artifacts shorter than 50 characters are treated as empty by the artifact gate. This avoids false positives from stub content but means very short valid artifacts would be rejected.
- **OpenCode ACP + vLLM returns tiny responses** — Known protocol mismatch. Use `DeepAgentsModelProvider` with `init_chat_model()` directly for local models instead.
- **RalphLoopWorker is a singleton** — Access via `get_worker()`, don't instantiate directly. Thread-safety enforced by `_WORKER_LOCK`.
- **SQLite WAL mode** — Checkpoint DB uses WAL (Write-Ahead Logging). Multiple readers OK, one writer at a time. Don't hold long-running write transactions.
- **TypedDict state keys** — `WorkflowState` is a TypedDict. New keys must be added to the type definition or LangGraph will silently drop them.
- **`from __future__ import annotations`** — Required in files using `str | None` or forward references, but can break Pydantic models at runtime. Use with care in model files.
- **Pre-built ChatOpenRouter must apply ProviderProfile kwargs** — When `resolve_model` returns a `ChatOpenRouter` instance (OpenRouter + session_id), the DA `ProviderProfile` factory chain is skipped. If you add new kwargs to the OpenRouter ProviderProfile, verify they're also handled in `_build_openrouter_model()`.
- **`.env` must be at project root** — `spine/config.py` loads `.env` from `Path.cwd()`. If running from a different directory, the env vars won't be loaded. `langgraph dev` handles this via `langgraph.json`.
- **Resume re-runs the full graph** — `resume_work()` re-invokes the entire StateGraph from START with accumulated state + human feedback. It does NOT resume from the exact checkpoint position. For true mid-graph resume, use LangGraph's `interrupt()` + `Command(resume=...)` pattern instead.
- **Critic node `current_phase` is always `"critic"`** — The critic returns `current_phase: PhaseName.CRITIC.value` regardless of which phase it's reviewing (e.g. `critic_plan` node still returns `current_phase: "critic"`). The audit log uses the node name (`critic_plan`) which is correct, but `current_phase` in state is generic.

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
