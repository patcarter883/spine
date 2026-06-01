# SPINE

**Deterministic AI agent harness with a state machine workflow engine.** DAG-based parallel execution, and modular provider architecture.

SPINE orchestrates AI agents through a structured lifecycle - from planning through execution to verification - using LangGraph's state machine and checkpointing, Deep Agents for in-process agent loops, and a Streamlit dashboard for full visibility.

## Quick Start

```bash
# Install from PyPI
pip install spine-harness

# Initialize in your project directory
spine init

# Start a new work item
spine run "Build a REST API with authentication"

# Or launch the Streamlit dashboard
spine ui
```

## Features

- **State Machine Workflow** - Structured lifecycle: SPECIFY -> PLAN -> IMPLEMENT -> VERIFY, with checkpoints and resume support.
- **Work Types** - Four workflow types: `task`, `critical_task`, `reviewed_task`, `critical_reviewed_task` with critic gates and optional human review.
- **Deep Agent Architecture** - Deep Agents (DA) as the sole execution path, providing structured subagent delegation and critic gates per phase.
- **FeatureSlice Decomposition** - IMPLEMENT phase decomposes into parallelizable feature slices with dependency edges.
- **Streamlit Dashboard** - 9-page UI: Dashboard, New Work, Work Detail, Work History, Human Review, Task Queue, Audit Log, Config View, Spec Planning. Zero-duplication architecture - UI and CLI share the same code paths.
- **Ralph Loop Worker** - Background queue processor that autonomously dequeues and executes tasks.
- **Multi-Provider Support** - LLM providers via DeepAgentsModelProvider (vLLM, OpenRouter).
- **Subgraph Architecture** - Each phase runs inside its own nested LangGraph subgraph with per-phase SQLite checkpoints.

## CLI Commands

| Command | Description |
|---------|-------------|
| `spine init [path]` | Scaffold `.spine/` and a baseline `config.yaml` (use `--tech-stack` and `--force`) |
| `spine run "description"` | Start new work item (use `--type task` or `--type critical_task`) |
| `spine status <work_id>` | Show current workflow status |
| `spine list` | List active work items |
| `spine resume <work_id>` | Resume a paused work item |
| `spine restart <work_id>` | Restart stalled or failed work |
| `spine worker` | Start RalphLoopWorker background processor |
| `spine ui` | Start Streamlit dashboard (localhost:8501) |

Options: `--type`, `--config`, `--debug-llm`

## Architecture

### Workflow Phases

```
TASK:       SPECIFY -> PLAN -> CRITIC_PLAN -> IMPLEMENT -> VERIFY
CRITICAL:   SPECIFY -> CRITIC_SPECIFY -> PLAN -> CRITIC_PLAN -> IMPLEMENT -> VERIFY
REVIEWED:   Same as above but pauses after CRITIC_PLAN for human approval
```

Each phase is a LangGraph node with SQLite-backed checkpoints. The state machine persists between interruptions and can be resumed at any point.

### Subgraph Architecture

Each phase runs inside its own nested LangGraph subgraph with:
- Per-phase SQLite checkpoint databases (isolated per phase)
- State mappers: ParentState -> SubgraphState
- Result mappers: SubgraphState -> ParentState update
- Feature flags (`_SUBGRAPH_ENABLED`) for controlled rollout

### Provider System

Providers are configured in `.spine/config.yaml` and resolved at runtime:

```yaml
providers:
  # Named provider entries
  llm:
    - name: deepseek-v4-pro
      type: deepagents-model
      model: openrouter:deepseek/deepseek-v4-pro
      enabled: true

    - name: local-vllm
      type: deepagents-model
      model: openai:model
      base_url: "http://localhost:8000/v1"
      api_key: "vllm"
      enabled: true

    - name: local-tiny
      type: deepagents-model
      model: openai:model
      base_url: "http://localhost:8001/v1"
      api_key: "vllm"
      temperature: 0.1
      enabled: true

  # Per-phase and subagent overrides
  phases:
    specify:
      provider: deepseek-v4-pro
    plan:
      provider: deepseek-v4-pro
    critic:
      provider: deepseek-v4-pro
    exploration:
      provider: deepseek-v4-pro

    implement:
      provider: local-vllm
      temperature: 0.6

    implement/subagents/slice-implementer:
      provider: local-vllm

    verify:
      provider: deepseek-v4-pro
```

**Resolution order (most specific wins):**

1. `providers.phases.<phase>/subagents/<name>` - subagent override
2. `providers.phases.<phase>` - per-phase override
3. First enabled entry in `providers.llm[]`
4. `SPINE_MODEL` environment variable

## File Structure

```
spine/
├── spine/                       # Main package
│   ├── agents/                  # Deep Agent builders, tools, context, backend
│   ├── cli/                     # Click commands
│   ├── config/                  # Config loading and validation
│   ├── critic/                  # Critic agent and review logic
│   ├── mcp/                     # MCP tool loading via langchain-mcp-adapters
│   ├── models/                  # State, enums, types
│   ├── phases/                  # Phase node functions
│   ├── persistence/             # Checkpoint and artifact storage
│   ├── services/                # Audit logging
│   ├── ui/                      # Streamlit dashboard and WebSocket
│   ├── ui_api/                  # UI backend API
│   ├── workflow/                # StateGraph composition, subgraphs, registry
│   │   └── subgraphs/           # Per-phase subgraph implementations
│   ├── work/                    # Dispatcher and worker
│   └── exceptions.py            # Custom exceptions
├── tests/
│   ├── unit/
│   └── integration/
├── .spine/                      # Runtime data (config, checkpoints, DB)
├── pyproject.toml
└── README.md
```

## Configuration

After installation, create a `.spine/` directory with:

- `config.yaml` - Provider configuration and workspace settings
- `spine.db` - SQLite checkpoint database for state machine persistence
- `artifacts/` - Artifact storage per work item
- `checkpoints/` - Per-phase subgraph checkpoints

### Setting up vLLM + Deep Agents

For best results, run a local vLLM instance:

```bash
vllm serve Qwen3.6-35B-A3B --quantization awq --max-model-len 8192
```

Then configure `config.yaml` with the Deep Agents provider type pointing to your vLLM endpoint.

## Dependencies

| Category | Packages |
|----------|----------|
| Core | langchain>=1.0, langchain-openai>=1.2, langchain-openrouter>=0.2, langgraph>=0.4, deepagents[quickjs]>=0.5 |
| Persistence | langgraph-checkpoint-sqlite>=3, sqlite-utils, aiosqlite, sqlalchemy |
| UI | streamlit>=1.30, websockets |
| Tracing | langsmith, python-dotenv, langgraph-cli[inmem] |
| Dev | pytest, pytest-asyncio, ruff |
| MCP | langchain-mcp-adapters, mcp-codebase-index |

## Testing

```bash
pytest tests/unit/        # Unit tests
pytest tests/integration/  # Integration tests
```

## License

MIT - see LICENSE file.