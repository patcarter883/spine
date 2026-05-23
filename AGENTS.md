# SPINE - Agent Instructions

## Project Overview
SPINE is a deterministic AI agent harness with a workflow engine and modular provider architecture. It orchestrates AI agents through a structured lifecycle - specify, plan, implement, verify - using **LangGraph StateGraph** for workflow topology and checkpoint persistence, **Deep Agents** for in-process LLM agent loops, and a **Streamlit dashboard** for visibility and human review.

**Language:** Python 3.12+
**Build:** hatchling
**Package:** `spine-harness`
**CLI entry point:** `spine` -> `spine.cli:main`
**Version:** 0.1.0

---

## Work Types
*Task*: SPECIFY -> PLAN -> CRITIC_PLAN -> IMPLEMENT -> VERIFY
*Critical Task*: SPECIFY -> CRITIC_SPECIFY -> PLAN -> CRITIC_PLAN -> VERIFY
*Reviewed Task*: Same as Task but pauses after CRITIC_PLAN for approval
*Critical Reviewed Task*: Same as Critical Task but pauses after CRITIC_PLAN for approval

(Note: The CLI, `spine run`, automatically chooses the appropriate work type based on `--type` flag: `task` or `critical_task`.)

---

## Configuration
```yaml
spine:
  checkpoint_path: .spine/spine.db
  workspace_root: /home/pat/Projects/spine
  interpreter_enabled: true

# MCP Servers
# Model Context Protocol servers for external tool integration.
# mcp-codebase-index provides 18 structural codebase query tools
# (symbol lookup, dependency analysis, change impact assessment).
mcp_servers:
  codebase-index:
    transport: stdio
    command: mcp-codebase-index
    args: []
    env:
      PROJECT_ROOT: /home/pat/Projects/spine

providers:
  llm:
    - name: glm
      type: deepagents-model
      model: openrouter:z-ai/glm-5.1
      enabled: true
    - name: qwen3.7-max
      type: deepagents-model
      model: openrouter:qwen/qwen3.7-max
      enabled: true
    - name: deepseek-v4-flash
      type: deepagents-model
      model: openrouter:deepseek/deepseek-v4-flash
      enabled: true
    - name: deepseek-v4-pro
      type: deepagents-model
      model: openrouter:deepseek/deepseek-v4-pro
      enabled: true
    - name: laguna-m1
      type: deepagents-model
      model: openrouter:poolside/laguna-m.1:free
      enabled: true
    - name: laguna-xs2
      type: deepagents-model
      model: openrouter:poolside/laguna-xs.2:free
      enabled: true
    - name: gemini-3.5-flash
      type: deepagents-model
      model: openrouter:google/gemini-3.5-flash
      enabled: true
    - name: local
      type: deepagents-model
      model: openai:model
      base_url: "http://localhost:8000/v1"
      api_key: "vllm"
      enabled: true
    - name: pat
      type: deepagents-model
      model: openai:model
      base_url: "http://10.50.1.51:8000/v1"
      api_key: "vllm"
      enabled: true

  phases:
    # Heavy reasoning -> frontier model (via reference)
    specify:
      provider: deepseek-v4-pro
    plan:
      provider: deepseek-v4-pro
    critic:
      provider: deepseek-v4-pro
    exploration:
      provider: deepseek-v4-pro
    # Implementation -> local vLLM, but colder temperature for code
    implement:
      provider: local
      temperature: 0.6           # freezes just the temp override

    # Slice implementer subagent gets the same vLLM but even faster
    implement/subagents/slice-implementer:
      provider: local

    # Verification -> deepseek-v4-pro for quick validation
    verify:
      provider: deepseek-v4-pro
```

## Provider Resolution
Providers are configured in `.spine/config.yaml` and resolved at runtime:
- Per-phase providers override default settings
- Subagent providers can be specified independently
- Direct keys (base_url, temperature) take priority over inherited settings

## Key Concepts
- **LangGraph is the core engine**: StateGraph handles workflow progression with atomic checkpoints per phase
- **Providers resolved from config**: Never store providers in state for serialization compatibility
- **Four work types**: Standard task workflow with optional critical tier and review gates
- **Subagent decomposition**: Feature slices in IMPLEMENT phase handled by parallel subagents
- **Gate mechanisms**: Artifact gates prevent empty progression, critic routing enforces structural quality
- **Core phases**: `specify`, `plan`, `critic`, `exploration`, `implement`, `verify`, `gap_plan`

## CLI Commands
* `spine run "description"` - Start new work item (use `--type task` or `--type critical_task`)
* `spine status <work_id>` - Check work status
* `spine list` - View active work items
* `spine resume <work_id>` - Resume paused work items (after human review)
* `spine restart <work_id>` - Restart stalled or failed work
* `spine worker` - Start the RalphLoopWorker background processor
* `spine ui` - Launch Streamlit dashboard (localhost:8501)

## Current Provider Setup
- **DeepSeek-V4-PRO** for heavy reasoning across `specify`, `plan`, `critic`, and `exploration` phases
- **Local vLLM** (`openai:model` with `vllm` key) for implementation (code generation)
- **Additional LLMs**: `glm`, `qwen3.7-max`, `deepseek-v4-flash`, `laguna-m1`, `laguna-xs2`, `gemini-3.5-flash`, `pat` for specialized tasks or overrides

---

### Notes
- For users wanting to experiment with different providers, edit the `providers` section in `.spine/config.yaml`. The CLI automatically reloads the config on restart.
- The `exploration` phase is used internally for the new subgraph-based exploration engine; adding it here provides visibility into the underlying workflow.
- The `verify` phase defaults to `deepseek-v4-pro` for quick validation; adjust temperature in the config if you require more deterministic outputs.

### Subgraph Architecture

Each phase runs inside its own nested LangGraph subgraph with per-phase SQLite checkpoints. This provides:

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

### Key Modules

| Path | Purpose |
|------|---------|
| `spine/workflow/compose.py` | `build_workflow_graph()` - builds the LangGraph StateGraph from WorkType, defines WORKFLOW_SEQUENCES |
| `spine/workflow/critic_review.py` | `critic_router()` - two-tier critic routing (structural + agent) |
| `spine/workflow/artifact_gate.py` | `make_artifact_gate_node()` + `artifact_gate_router()` - gate as a node (not just a conditional edge) |
| `spine/workflow/registry.py` | PhaseRegistry - maps phase names to node functions and agent builders |
| `spine/workflow/subgraph_wrapper.py` | `make_subgraph_node()` + `make_success_result_mapper()` - wraps subgraphs as parent nodes |
| `spine/workflow/subgraph_state.py` | Per-phase subgraph TypedDict state schemas |
| `spine/workflow/subgraphs/` | Phase subgraph implementations |
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
| `spine/ui/ws_bus.py` | Async pub/sub event bus for state changes (singleton) |
| `spine/ui/ws_server.py` | WebSocket server for Streamlit client push notifications |
| `spine/ui/ws_component.py` | WebSocket client component for Streamlit pages |
| `spine/ui/app.py` | Streamlit app entry point with WebSocket push |
| `spine/ui/_pages/` | 9 UI pages: dashboard, work_submit, work_detail, work_history, human_review, queue, audit_log, config_view, spec_planning |
| `spine/cli/__init__.py` | Click commands: `run`, `status`, `list`, `resume`, `restart`, `worker`, `ui` |
| `spine/mcp/client.py` | MCP tool loader using `langchain-mcp-adapters`, namespaced tools |
| `spine/mcp/tools.py` | MCP tool configuration |
| `spine/mcp/defaults.py` | Default MCP server config |

---

### Design Principles

1. **LangGraph is the workflow engine.** The StateGraph defines node topology, conditional edges, and checkpoint persistence. Deep Agents handle the LLM work *inside* each phase subgraph. The workflow progression is entirely LangGraph - there is no separate state machine system.

2. **Zero Duplication: CLI and UI Share Code Paths** - Both CLI commands and Streamlit pages call the same backend functions:
   - **Writes:** Both go through `submit_work()` / `resume_work()` in `spine/work/dispatcher.py`
   - **Reads:** Both use `UIApi` in `spine/ui_api/api.py`
   - UI pages must never import directly from `spine/workflow/` or `spine/phases/`

3. **One Deep Agent Per Phase** - Each SPINE phase constructs its own `create_deep_agent()` via `build_phase_agent()` with phase-specific system prompts, tools, middleware, skills, and memory. This preserves phase isolation while giving each phase the full DA infrastructure. Each phase now has its own dedicated agent module (e.g., `spine/agents/specify_agent.py`, `spine/agents/plan_agent.py`) and purpose-built tool modules.

4. **Critic Gate Is Structural, Not Prompted** - The critic gate enforces that a phase cannot proceed without `PASSED` status. This is enforced in `critic_router()` conditional edge logic, not by prompting the LLM. When retries are exceeded, it routes to `needs_review -> END`.

5. **Artifact Gates Prevent Empty Progression** - Before `implement` runs, an artifact gate checks that `plan` produced artifacts. If the gate fails, it routes to `needs_review -> END` instead of proceeding. There is NO artifact gate between `implement` and `verify` - verify always runs after implement.

6. **SPINE Base Prompt Replaces DA Default** - `spine/agents/profile.py` registers `HarnessProfile(base_system_prompt=SPINE_BASE_PROMPT)` for `openrouter`, `openai`, and `anthropic` providers. This replaces the DA `BASE_AGENT_PROMPT` with a phase-executor framing. The prompt assembly order is: `USER` (phase system_prompt) -> `CUSTOM` (SPINE_BASE_PROMPT) -> `SUFFIX` (none).

7. **OpenRouter Session Tracking** - When using OpenRouter with session tracking, the session_id is passed through to enable continuity across long-running workflows.

8. **Phase Nodes Must Return Complete State Updates** - Every phase node function must return `status` and `prompt_request` in its output dict, even on error paths. Missing fields cause `_update_work_progress()` to record empty status in the audit log and work entries DB.

9. **Subgraph State Isolation** - Each phase runs inside its own nested LangGraph subgraph with per-phase SQLite checkpoints. State mappers transform parent state into subgraph state on entry; result mappers transform subgraph output back to parent state on exit. Never assume data is directly in the parent state.

10. **WebSocket Push Notifications** - State changes are published to `spine.ui.ws_bus.WSEventBus` (singleton). The `spine.ui.ws_server` WebSocket server fans them to Streamlit clients. UI clients connect via `spine.ui.ws_component` and receive filtered push notifications instead of polling.

11. **Feature Flag Controlled Rollout** - Phase subgraph migration uses `_SUBGRAPH_ENABLED` dict per phase. Exploration subgraph uses `_USE_EXPLORATION_SUBGRAPH` dict.

12. **MCP Tools Are Namespaced** - MCP tools are loaded via `langchain-mcp-adapters` (not the custom client). All tools are namespaced with the server name to prevent collisions (e.g., `mcp_codebase-index_find_symbol`).

---

### Pitfalls

- **workspace_root is CRITICAL - never break its resolution** - Every Deep Agent's `LocalShellBackend` uses `workspace_root` as its `root_dir`. If this resolves to a wrong path (e.g. `/root`, `/tmp`), agents get `Permission denied` errors and the per-phase checkpointer fails silently.

- **Phase node functions MUST be async** - All phase node functions (`call_specify`, `call_plan`, etc.) must be `async def` and use `ainvoke_with_retry` (not `invoke_with_retry`). Sync nodes run in LangGraph's thread pool, which breaks `asyncio.Lock` objects in the checkpointer.

- **Never store providers in WorkflowState** - LangGraph's checkpointer serializes state to SQLite, and provider objects (LLM clients, HTTP sessions) are not serializable. Use `config["configurable"]["providers"]`.

- **Phase nodes must return status and prompt_request** - Every return dict from a phase function must include `"status"` and `"prompt_request"`. Missing these causes empty entries in the audit log and work progress updates.

- **Per-phase checkpoint isolation** - Each phase subgraph writes to its own SQLite database. Do not assume checkpoints are shared across phases.

- **State mappers and result mappers must stay in sync** - When you change a phase's subgraph state, ensure both the state mapper (parent -> subgraph) and result mapper (subgraph -> parent) are updated.

- **TypedDict state keys** - `WorkflowState` is a TypedDict. New keys must be added to the type definition or LangGraph will silently drop them.

- **SpineContext must be a Pydantic BaseModel** - LangGraph's config schema creates a Pydantic field `(SpineContext, None)` for the `context_schema`. Using `BaseModel` lets Pydantic serialize it natively.

- **RalphLoopWorker is a singleton** - Access via `get_worker()`, don't instantiate directly. Thread-safety enforced by `_WORKER_LOCK`.

---

### Code Style

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

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Module-level docstrings explaining purpose and key design decisions
- Section headers using `# - Section Name -` comment style

---

### Testing

Run tests with:
```bash
pytest tests/
```

Tests are organized by what they verify:
- Unit tests: Single unit in isolation (`tests/unit/`)
- Integration tests: Multiple units together (`tests/integration/`)

Key test patterns:
- Async: Use `@pytest.mark.asyncio`
- Mock: Use `unittest.mock` for external deps
- State propagation: Every node returns `status` + `prompt_request`
- Resume: validates needs_review status, rejects unknown/completed work
- Retry logic: transient errors retry, permanent errors raise immediately
- Subgraph state mapping: parent -> subgraph -> parent transformation
- Slice scheduler: topological sort correctness for execution waves
- WebSocket bus: publish/subscribe from sync and async contexts

What NOT to Test:
- LangGraph internals (StateGraph, checkpointing) - trust the library
- LLM provider API responses - mock at provider boundary
- Streamlit rendering - test the API, not the UI

---

### Directory Layout

```
spine/
├── agents/           # Deep Agent builders, tools, context, backend
├── cli/              # Click commands and rendering
├── config/           # Config loading and validation
├── critic/           # Critic agent and review logic
├── mcp/              # MCP tool loading via langchain-mcp-adapters
├── models/           # State, enums, types
├── phases/           # Phase node functions
├── persistence/      # Checkpoint and artifact storage
├── services/         # Audit logging
├── ui/               # Streamlit dashboard and WebSocket
├── ui_api/           # UI backend API
├── workflow/         # StateGraph composition, subgraphs, registry
│   └── subgraphs/    # Per-phase subgraph implementations
├── work/             # Dispatcher and worker
└── exceptions.py     # Custom exceptions
```

`.spine/` directory (not tracked):
| Path | Purpose |
|------|---------|
| `config.yaml` | Runtime configuration |
| `spine.db` | Work items, queue entries, audit log |
| `artifacts/` | Artifact storage per work item |
| `checkpoints/` | Per-phase subgraph checkpoints (`<work_id>/<phase>.db`) |
| `state/` | Current work state |
| `events/` | Event data |
| `knowledge/` | Knowledge base files |

---

### Dependencies

| Category | Packages |
|----------|----------|
| Core | langchain>=1.0, langchain-openai>=1.2, langchain-openrouter>=0.2, langgraph>=0.4, deepagents[quickjs]>=0.5 |
| Persistence | langgraph-checkpoint-sqlite>=3, sqlite-utils, aiosqlite, sqlalchemy |
| UI | streamlit>=1.30, websockets |
| Tracing | langsmith, python-dotenv, langgraph-cli[inmem] |
| Dev | pytest, pytest-asyncio, ruff |
| MCP | langchain-mcp-adapters, mcp-codebase-index |

---

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SPINE_WORKSPACE_ROOT` | Override workspace root detection | `.spine/` dir walk |
| `SPINE_DEBUG_LLM` | Log all LLM messages to console | unset |
| `SPINE_STALL_TIMEOUT` | Stall detection timeout (seconds) | 120 |
| `SPINE_MAX_CRITIC_RETRIES` | Max critic retry attempts | 3 |
| `SPINE_WORK_TYPE` | Default work type | `task` |

---

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

---

### Adding a New UI Page

Add a new file in `spine/ui/_pages/` that imports only from `spine/ui_api/api.py`. Never import from `spine/workflow/` or `spine/phases/` in a UI page.

---

### Git Hooks (Optional)

Install pre-commit hooks for automated checks:
```bash
pip install pre-commit
pre-commit install
```

This runs `ruff check` and `ruff format --check` on staged files before commit.

---

### License

MIT - see LICENSE file.