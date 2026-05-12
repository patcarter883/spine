# SPINE — Agent Instructions

## Project Overview

SPINE is a deterministic AI agent harness with a state machine workflow engine, DAG-based parallel execution, and modular provider architecture. It orchestrates AI agents through a structured lifecycle — planning, execution, verification — using LangGraph's state machine and checkpointing, Deep Agents for in-process agent loops, and a Streamlit dashboard for visibility.

**Language:** Python 3.11+  
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
pytest tests/e2e/           # End-to-end tests

# Lint & format
ruff check spine/ tests/
ruff format spine/ tests/

# Type check
mypy spine/

# Run a single test by name
pytest tests/ -k "test_name_goes_here"
```

---

## Architecture

```
SPINE STATE MACHINE (LangGraph StateGraph)
  INIT → PLANNING → EXECUTION → VERIFICATION → COMPLETE
         ↘ REWORK ↗   ↗ BLOCKED / ERROR / HUMAN_REVIEW
  │
  └─ delegates to → Deep Agents Runtime (one create_deep_agent() per phase)
                      ├── SubAgents (from FeatureSlices)
                      ├── Middleware (critic gate, step limit, message queue)
                      └── Backends (StateBackend, LocalShellBackend)
```

### Key Modules

| Path | Purpose |
|------|---------|
| `spine/core/state_machine.py` | LangGraph StateGraph, phase transitions, `should_continue()` routing |
| `spine/core/persistence.py` | GitWorkflow, Checkpoint, SQLite-backed state |
| `spine/core/ui_api.py` | UIApi — sole read/write interface for Streamlit UI |
| `spine/models/types.py` | SpineState (TypedDict), Task, SubPhase, Phase, FeatureSlice |
| `spine/models/enums.py` | PhaseName, StateStatus, SubPhaseStatus, ErrorState |
| `spine/models/dag.py` | `synthesize_slices()` — heuristic and agent-based FeatureSlice decomposition |
| `spine/adapters/da_phase_adapter.py` | Bridge: SPINE phases → `create_deep_agent()` instances |
| `spine/middleware/` | CriticGateMiddleware, StepLimitMiddleware, MessageQueueMiddleware |
| `spine/providers/base.py` | Provider ABC, ProviderRegistry, ProviderFallbackChain, PluginLoader |
| `spine/providers/llm.py` | LLMProvider — async generate/stream with TTFB timeout |
| `spine/providers/deepagents_model.py` | DeepAgentsModelProvider — `init_chat_model()` for local models |
| `spine/providers/agents.py` | OpenCodeAgentProvider, CodexAgentProvider, ClaudeCodeAgentProvider |
| `spine/work/dispatcher.py` | `submit_work()` — unified CLI+UI entry point |
| `spine/work/ralph_worker.py` | RalphLoopWorker — background queue processor (singleton) |
| `spine/workflows/engine.py` | Ralph Loop engine integration |
| `spine/workflows/sdd.py` | SDDWorkflow — SPEC→DESIGN→PLAN→IMPLEMENT→REVIEW→VERIFY |
| `spine/workflows/quick_work.py` | QuickWork — simplified plan→implement→verify |
| `spine/swarm/` | Swarm agents, gates, supervisor, mail system |
| `spine/hive/` | Hive memory + reservations (shared agent memory) |
| `spine/ui/` | Streamlit dashboard (8 pages + utils, zero-duplication) |
| `spine/cli/` | Click commands, Rich renderers |
| `backend/` | FastAPI REST API (work, status, audit routes) |
| `spine/services/audit_service.py` | Audit logging |
| `spine/discovery/` | Codebase analyzer, mapper, reverse-engineer |

---

## Coding Conventions

### Style

- **Line length:** 100 characters (ruff + black configured)
- **Target Python:** 3.11+
- **Formatting:** `ruff format` (replaces black in practice)
- **Linting:** `ruff check`
- **Type annotations:** Use modern syntax (`list[str]` not `List[str]`, `str | None` not `Optional[str]`)
- **Imports:** `from __future__ import annotations` in files with forward references
- **Grouping:** stdlib → third-party → relative, separated by blank lines
- **Relative imports:** Use `..` notation for cross-module imports (`from ..core.state_machine import ...`)

### Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `FeatureSlice`, `SpineStateMachine` |
| Enums | PascalCase (str, Enum) | `PhaseName.PLANNING` |
| Functions/Methods | snake_case | `submit_work()`, `_evaluate_entry_conditions()` |
| Constants | SCREAMING_SNAKE | `DEFAULT_TIMEOUT`, `TTFB_TIMEOUT`, `PHASE_ICONS` |
| Private members | Leading underscore | `_run_critic()`, `_current_task` |
| Factory functions | `create_` prefix | `create_deep_agent()`, `create_provider()` |
| Test classes | `Test` prefix | `TestParallelExecution`, `TestProviderFallbackChain` |

### Docstrings

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Module-level docstrings explaining purpose and key design decisions
- Section headers using `# ── Section Name ──` comment style (en-dash borders)

### Data Models

- **Dataclasses** for internal types: `@dataclass` with typed fields, `field(default_factory=list)` for mutable defaults
- **TypedDict** for state dicts: `SpineState(TypedDict)` used by LangGraph
- **Pydantic** for API schemas (backend only)
- **Enum** with `(str, Enum)` base for JSON-serializable enumerations

### Error Handling

- Specific exceptions first, broad `Exception` last
- Custom exceptions: `GateEnforcementError`, `TTFBTimeoutError`, `InvalidAgentRoleError`, `ConflictRequiresHuman`
- Graceful fallback pattern: catch, log warning, return safe default
- Retry with exponential backoff for transient failures (persistence, network)
- Defensive returns: empty list/dict instead of raising on missing data

### Async

- `async def` for all provider methods that do I/O
- `AsyncIterator[str]` for streaming
- `asyncio.run_in_executor()` to bridge sync code into async context
- `@pytest.mark.asyncio` for async tests
- Never block the event loop with sync I/O

### Threading

- `threading.Lock()` for shared mutable state (RalphLoopWorker singleton, queue access)
- Daemon threads for background workers
- `threading.Event` for graceful shutdown signaling

---

## Key Design Rules

### 1. Provider Resolution: Config, Not State

Providers must be resolved from `config["configurable"]["providers"]`, never from `SpineState`. Storing providers in state causes serialization failures after LangGraph checkpointing. The `_get_providers_from_config()` function is the canonical way to obtain providers inside phase functions.

### 2. Zero Duplication: CLI and UI Share Code Paths

Both CLI commands and Streamlit pages call the same backend functions:
- **Writes**: Both go through `submit_work()` in `spine/work/dispatcher.py`
- **Reads**: Both use `UIApi` in `spine/core/ui_api.py`
- UI pages must never import directly from `spine.core.state_machine` or `spine.models/`

### 3. One Deep Agent Per Phase

Each SPINE phase (PLANNING, EXECUTION, VERIFICATION) constructs its own `create_deep_agent()` with phase-specific system prompts, tools, and middleware. This preserves phase isolation while giving each phase the full DA infrastructure.

### 4. FeatureSlices → SubAgents

During EXECUTION, each FeatureSlice from the plan becomes a `SubAgent` spec. The DA `task` tool handles delegation with isolated context windows. Dependencies between slices are respected by the DAG executor.

### 5. Critic Gate Is Structural, Not Prompted

The critic gate enforces that PLANNING cannot transition to EXECUTION without `APPROVED` status. This is enforced in `should_continue()` routing logic, not by prompting the LLM.

### 6. Wave-Based DAG Execution

FeatureSlices with no pending dependencies execute in parallel via Deep Agents' SubAgent `task` tool. The DA runtime handles dependency ordering and concurrent delegation.

---

## Testing

### Framework

- **pytest** with `pytest-asyncio`
- Tests in `tests/` directory at project root
- Class-based organization: `class TestParallelExecution:` with descriptive method names

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

- Phase transitions and routing logic in `should_continue()`
- DAG execution: wave ordering, dependency resolution, parallel execution
- Provider fallback chains and conflict resolution
- Critic gate enforcement (APPROVED/NEEDS_REVISION routing)
- Sub-phase state tracking (status, retry, error count)
- Entry/exit condition evaluation
- Serialization/deserialization of checkpoints
- FeatureSlice synthesis from plans

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

providers:
  llm:
    - name: local-vllm
      type: deepagents-model
      model: "openai:Qwen/Qwen3-32B"
      base_url: "http://localhost:8000/v1"
      api_key: "dummy"
      temperature: 0.3
      max_tokens: 8192
      priority: 0
      enabled: true

  agent:
    - name: default
      type: opencode
      mode: acp
      model: openrouter/minimax/minimax-m2.7
      auto_approve: true

queue:
  backend: sqlite
  # redis_url: "redis://localhost:6379/0"  # only for redis backend
```

### Runtime Data (`.spine/`)

| Path | Purpose |
|------|---------|
| `spine.db` | LangGraph SQLite checkpoints |
| `queue.db` | Task queue (SQLite backend) |
| `work_entries.db` | Work item tracking |
| `jobs.db` | Job execution records |
| `state/current_work.json` | Current active work item |
| `spec/{thread_id}.md` | Spec artifacts per work item |
| `artifacts/plans/{thread_id}.json` | Plan artifacts per work item |

---

## Common Workflows

### Adding a New Provider

1. Create a new class in `spine/providers/` extending `Provider` (ABC in `base.py`)
2. Implement `configure()`, `validate()`, `name` property, `enabled` property
3. Register the provider type in `ProviderType` enum (`base.py`)
4. Add loader logic in `PluginLoader` or auto-discovery
5. Add tests in `tests/test_providers_base.py` or a new test file
6. Update `config.yaml` documentation in README

### Adding a New Workflow Phase

1. Add phase name to `PhaseName` enum in `spine/models/enums.py`
2. Add the phase function in `spine/core/state_machine.py` (follow existing pattern)
3. Add routing logic in `should_continue()` or create a new conditional edge
4. Build the DA adapter config in `spine/adapters/da_phase_adapter.py`
5. Add entry/exit conditions in `_evaluate_entry_conditions()` / `_evaluate_exit_conditions()`
6. Write tests for the new transition paths

### Adding a New UI Page

1. Create `spine/ui/new_page.py` with a `render()` function
2. Add page to navigation in `spine/ui/app.py`
3. All data access MUST go through `UIApi` in `spine/core/ui_api.py`
4. Add helper functions to `spine/ui/utils.py` if needed
5. Verify zero-duplication: the same action must work from CLI
6. Test the `UIApi` methods, not the Streamlit rendering

---

## Pitfalls

- **Never store providers in `SpineState`** — LangGraph's checkpointer serializes state to SQLite, and provider objects (LLM clients, HTTP sessions) are not serializable. Use `config["configurable"]["providers"]`.
- **SubPhase uses `field(default_factory=...)` for mutable defaults** — Never use `[]` or `{}` as default values in dataclasses. Use `field(default_factory=list)`.
- **OpenCode ACP + vLLM returns tiny responses** — This is a known protocol mismatch. Use `DeepAgentsModelProvider` with `init_chat_model()` directly for local models instead.
- **RalphLoopWorker is a singleton** — Access via `get_worker()`, don't instantiate directly. Thread-safety enforced by `_WORKER_LOCK`.
- **SQLite WAL mode** — Checkpoint DB uses WAL (Write-Ahead Logging). Multiple readers OK, one writer at a time. Don't hold long-running write transactions.
- **TypedDict state keys** — `SpineState` is a TypedDict. New keys must be added to the type definition or LangGraph will silently drop them.
- **Error threshold on sub-phases** — Default `max_errors=3`. A sub-phase is marked FATAL after exceeding this, not at the threshold. Off-by-one: `error_count > max_errors`.
- **`from __future__ import annotations`** — Required in files using `str | None` or forward references, but can break Pydantic models at runtime. Use with care in model files.

---

## Dependencies

| Category | Packages |
|----------|----------|
| Core | langgraph, langgraph-checkpoint-sqlite, langgraph-supervisor, deepagents>=0.5.0 |
| LLM | langchain, langchain-openai, langchain-openrouter, openai |
| Data | pydantic>=2.0, pyyaml, sqlalchemy>=2.0, alembic, sqlite-utils |
| CLI | click, rich |
| UI | streamlit>=1.30 |
| API | fastapi, httpx |
| Dev | pytest, pytest-asyncio, ruff, mypy |
