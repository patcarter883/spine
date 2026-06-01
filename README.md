# SPINE

**A deterministic AI agent harness with a state machine workflow engine.**

SPINE orchestrates AI agents through a structured lifecycle — specify, plan, implement, verify — using [LangGraph](https://github.com/langchain-ai/langgraph) for workflow topology and checkpoint persistence, [Deep Agents](https://github.com/deepagents/deepagents) for in-process LLM agent loops, and a [Streamlit](https://streamlit.io) dashboard for visibility and human review.

```
SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY
```

Each phase runs as a nested LangGraph subgraph with isolated SQLite checkpoints, so workflows can be interrupted and resumed at any point without losing progress.

---

## Features

- **State machine workflow** — Four work types with configurable critic gates and optional human review pauses
- **Per-phase subgraphs** — Each phase is an isolated LangGraph subgraph; crashes in one phase never corrupt another
- **Feature slice decomposition** — IMPLEMENT phase decomposes work into parallelisable slices with dependency edges
- **Multi-provider LLM routing** — Route different phases to different models (frontier for reasoning, local vLLM for code generation)
- **MCP tool integration** — Load external tools via the Model Context Protocol using `langchain-mcp-adapters`
- **Streamlit dashboard** — 9-page UI sharing the same code paths as the CLI (zero duplication)
- **Background worker** — `spine worker` runs a queue processor that autonomously dequeues and executes tasks
- **Artifact gates** — Prevent empty phase progression; a missing plan blocks the implement phase
- **Critic gates** — Structural quality checks enforced in conditional edge logic, not just prompts
- **Resumable** — SQLite-backed checkpoints per phase mean any interrupted workflow can be resumed

---

## Quick Start

```bash
# Install
pip install spine-harness

# Scaffold config in your project directory
spine init

# Start a work item
spine run "Build a REST API with authentication"

# Check status
spine status <work_id>

# Or launch the dashboard
spine ui
```

---

## Work Types

| Type | Phases |
|------|--------|
| `task` | SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY |
| `critical_task` | SPECIFY → CRITIC_SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY |
| `reviewed_task` | Same as `task`, pauses after CRITIC_PLAN for human approval |
| `critical_reviewed_task` | Same as `critical_task`, pauses after CRITIC_PLAN for human approval |

```bash
spine run "Add OAuth2 support" --type critical_task
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `spine init [path]` | Scaffold `.spine/` and a baseline `config.yaml` |
| `spine run "description"` | Start a new work item (`--type task` or `--type critical_task`) |
| `spine status <work_id>` | Show current workflow status |
| `spine list` | List active work items |
| `spine resume <work_id>` | Resume a paused work item (after human review) |
| `spine restart <work_id>` | Restart stalled or failed work |
| `spine worker` | Start the background queue processor |
| `spine ui` | Launch the Streamlit dashboard (localhost:8501) |

---

## Configuration

After `spine init`, edit `.spine/config.yaml`:

```yaml
spine:
  checkpoint_path: .spine/spine.db
  workspace_root: /path/to/your/project

providers:
  llm:
    - name: frontier
      type: deepagents-model
      model: openrouter:deepseek/deepseek-v4-pro
      enabled: true

    - name: local-vllm
      type: deepagents-model
      model: openai:model
      base_url: "http://localhost:8000/v1"
      api_key: "vllm"
      enabled: true

  # Per-phase provider overrides (most specific wins)
  phases:
    specify:
      provider: frontier
    plan:
      provider: frontier
    critic:
      provider: frontier
    implement:
      provider: local-vllm
      temperature: 0.6
    implement/subagents/slice-implementer:
      provider: local-vllm
    verify:
      provider: frontier
```

**Provider resolution order:**
1. `providers.phases.<phase>/subagents/<name>` — subagent override
2. `providers.phases.<phase>` — per-phase override
3. First enabled entry in `providers.llm[]`
4. `SPINE_MODEL` environment variable

### Using a local vLLM instance

For cost-effective code generation, run a local model for the IMPLEMENT phase:

```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --quantization awq --max-model-len 8192
```

Then point `local-vllm` in your config at `http://localhost:8000/v1`.

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SPINE_WORKSPACE_ROOT` | Override workspace root detection | `.spine/` dir walk |
| `SPINE_DEBUG_LLM` | Log all LLM messages to console | unset |
| `SPINE_STALL_TIMEOUT` | Stall detection timeout (seconds) | 120 |
| `SPINE_MAX_CRITIC_RETRIES` | Max critic retry attempts | 3 |
| `SPINE_WORK_TYPE` | Default work type | `task` |

---

## Architecture

### Workflow engine

LangGraph is the sole workflow engine. The `StateGraph` defines node topology, conditional edges, and SQLite-backed checkpoint persistence. Deep Agents handle LLM execution *inside* each phase subgraph.

```
build_workflow_graph()        # spine/workflow/compose.py
  └── phase subgraphs
        ├── specify_subgraph
        ├── plan_subgraph
        ├── critic_subgraph
        ├── exploration_subgraph
        ├── implement_subgraph
        ├── gap_plan_subgraph
        └── verify_subgraph
```

Each subgraph has its own TypedDict state schema, state mapper (parent → subgraph), and result mapper (subgraph → parent). Per-phase checkpoint databases are written to `.spine/checkpoints/<work_id>/<phase>.db`.

### Zero-duplication UI/CLI

Both the CLI and Streamlit UI call the same backend:

- **Writes:** `submit_work()` / `resume_work()` in `spine/work/dispatcher.py`
- **Reads:** `UIApi` in `spine/ui_api/api.py`

UI pages never import directly from `spine/workflow/` or `spine/phases/`.

### Feature slice decomposition

During IMPLEMENT, the plan is decomposed into feature slices with dependency edges. `slice_scheduler.py` uses `graphlib.TopologicalSorter` to determine execution waves; independent slices run in parallel via the LangGraph Send API.

### Critic gates

The critic gate is structural, not prompted. `critic_router()` inspects `PhaseResult.status`; if status is not `PASSED` after the maximum retries, the workflow routes to `needs_review → END`. The LLM cannot talk its way past this gate.

---

## File Structure

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

.spine/               # Runtime data (not tracked in git)
├── config.yaml       # Provider and workspace configuration
├── spine.db          # Work items, queue entries, audit log
├── artifacts/        # Artifact storage per work item
└── checkpoints/      # Per-phase SQLite checkpoints
```

---

## Development

```bash
# Clone and install in editable mode
git clone https://github.com/your-org/spine.git
cd spine
pip install -e ".[dev]"

# Run tests
pytest tests/unit/
pytest tests/integration/

# Lint
ruff check spine/
```

---

## Dependencies

| Category | Packages |
|----------|----------|
| Workflow | `langgraph>=0.4`, `langgraph-checkpoint-sqlite>=3` |
| LLM | `langchain>=1.0`, `langchain-openai>=1.2`, `langchain-openrouter>=0.2`, `deepagents>=0.5` |
| Persistence | `sqlite-utils`, `aiosqlite`, `sqlalchemy`, `alembic` |
| UI | `streamlit>=1.30`, `websockets` |
| Tools | `langchain-mcp-adapters`, `mcp>=1.0`, `mcp-codebase-index` |
| Indexing | `tree-sitter`, `sqlite-vec` |
| Dev | `pytest`, `pytest-asyncio`, `ruff`, `mypy` |

---

## Acknowledgements

SPINE builds on and draws inspiration from several excellent projects:

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — the state machine and checkpointing backbone that makes resumable, deterministic workflows possible
- **[LangChain](https://github.com/langchain-ai/langchain)** — the foundational toolchain for LLM provider abstraction and tool integration
- **[Deep Agents](https://github.com/deepagents/deepagents)** — the in-process agent loop library powering each phase's LLM execution
- **[Streamlit](https://streamlit.io)** — rapid dashboard development without a separate frontend stack
- **[LangSmith](https://smith.langchain.com)** — tracing and observability for agent runs
- **[mcp-codebase-index](https://github.com/mcp-codebase-index/mcp-codebase-index)** — structural codebase query tools exposed via MCP

The following projects were direct inspirations for SPINE's design:

- **[workspine](https://github.com/PatrickSys/workspine)** — pioneered the pattern of persisting plans, decisions, and verification artifacts to a `.planning/` directory so that structured deliverables survive across agent sessions and runtimes; SPINE's artifact store and phase checkpoint model owes a great deal to this approach
- **[smallcode](https://github.com/Doorman11991/smallcode)** — a terminal-native coding agent built for local small LLMs (8B–35B) with context budget management, forgiving tool-call parsing, and patch-first editing; influenced SPINE's thinking around local vLLM routing and context engineering for the implement phase
- **[learnship](https://github.com/FavioVazquez/learnship)** — a spec-driven development harness for multiple AI coding assistants with structured phase loops and persistent memory; shaped the philosophy behind SPINE's sequential phase discipline and critic gate design

---

## License

MIT — see [LICENSE](LICENSE).
