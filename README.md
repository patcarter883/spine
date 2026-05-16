# SPINE

**Deterministic AI agent harness with a state machine workflow engine.**DAG-based parallel execution, and modular provider architecture.**

SPINE orchestrates AI agents through a structured lifecycle — from planning through execution to verification — using LangGraph's state machine and checkpointing, Deep Agents for in-process agent loops, and a Streamlit dashboard for full visibility.

## Quick Start

```bash
# Install from PyPI
pip install spine-harness

# Initialize in your project directory
spine init

# Start a new work item
spine work "Build a REST API with authentication"

# Or launch the Streamlit dashboard
spine ui
```

## Features

- **State Machine Workflow** — Structured lifecycle: INIT → PLANNING → EXECUTION → VERIFICATION → COMPLETE, with checkpoints and resume support.
- **Deep Agent Architecture** — Deep Agents (DA) as the sole execution path, providing structured subagent delegation, critic gates, and context compaction per phase.
- **Deep Agents Integration** — `create_deep_agent()` powered phases: planning, execution (with subagent decomposition), and verification. Providers resolved through LangGraph config to avoid serialization issues.
- **FeatureSlice Decomposition** — `synthesize_slices()` decomposes requirements into parallelizable feature slices with dependency edges, used by both planning and SDD workflows.
- **Streamlit Dashboard** — 8-page UI: Dashboard, New Work, Work Detail, Task Queue, Agent Resources, Spec-Driven Development (SDD), Providers, Settings. Zero-duplication architecture — UI and CLI share the same code paths.
- **Ralph Loop Worker** — Background queue processor that autonomously dequeues and executes tasks.
- **Spec-Driven Development (SDD)** — Write specs upfront, let the agents plan and execute against them.
- **Multi-Provider Support** — LLM providers (vLLM, OpenAI, OpenRouter, Ollama), agent executors (OpenCode, Codex, Claude Code), notification providers (Discord, Slack, Email).
- **GitHub Integration** — Issue resolution, worktree management, and PR handling.
- **Pattern Learning** — Captures and reuses patterns across work items.
- **Hive Memory** — Shared memory system across agents.
- **SQLite Persistence** — LangGraph checkpoints + SQLite tracking DB for durable state.
- **FastAPI Backend** — REST API layer with work, status, and audit endpoints.

## CLI Commands

| Command | Description |
|---------|-------------|
| `spine work "requirement"` | Start new work item with streaming progress |
| `spine resume` | Resume a previously interrupted work item |
| `spine status` | Show current workflow status |
| `spine init` | Initialize SPINE in current directory |
| `spine ui` | Start Streamlit dashboard (localhost:8501) |
| `spine plugins` | List available provider plugins |
| `spine resolve-conflict` | Resolve provider conflicts with strategies |
| `spine notify` | Send notifications via Discord, Slack, or email |

Options: `--thread-id`, `--checkpoint`, `--config`, `--debug-prompts`

## Architecture

### Workflow Phases

```
INIT → PLANNING → EXECUTION → VERIFICATION → COMPLETE
```

Each phase is a LangGraph node with SQLite-backed checkpoints. The state machine persists between interruptions and can be resumed at any point.

### Agent Execution Paths

**Deep Agents Path (Primary):**
- One compiled Deep Agent per phase (planning, execution, verification)
- FeatureSlices from the planning phase map to SubAgents for parallel execution
- Middleware hooks (after_model, before_model) for cross-cutting concerns
- Providers injected via LangGraph config (not state) to avoid serialization

### Provider System

Providers are configured in `.spine/config.yaml` and resolved at runtime:

```yaml
providers:
  llm:
    - name: local-vllm
      type: deepagents-model       # For DA path
      config:
        model: "openai:model"
        base_url: "http://localhost:8000/v1"
        api_key: "dummy"

  agent:
    - name: default
      type: opencode               # OpenCode, codex, claude-code
      config:
        mode: acp
        model: openrouter/minimax/minimax-m2.7
        auto_approve: true
```

#### Per-Phase and Per-Subagent Models

SPINE supports assigning different LLM models to individual workflow phases and
subagents.  This lets you route heavy reasoning (planning, specification) to
large frontier models while using faster/cheaper models for implementation and
verification — and even finer-grained subagent-level overrides.

The model for every phase agent and subagent is resolved through the same
hierarchy in `.spine/config.yaml`:

**Resolution order (most specific wins):**

1. `providers.phases.<phase>/subagents/<name>.model` — subagent override
2. `providers.phases.<phase>.model` — per-phase override
3. `providers.llm[].model` — default provider model
4. `SPINE_MODEL` environment variable

If none of the above are set, `spine` errors with a clear diagnostic.

**Subagent keys use a path syntax:**  `phase/subagents/name`.  When a subagent
key is unknown, resolution falls back to its parent phase key; when the phase
itself is unknown, it falls back to the default provider.

**Phases and their subagents:**

| Phase      | Subagents                |
|------------|--------------------------|
| `specify`  | `researcher`             |
| `plan`     | *(none)*                 |
| `tasks`    | `researcher`             |
| `implement`| `slice-implementer`      |
| `verify`   | `slice-verifier`         |
| `critic`   | *(none)*                 |

**Example configuration:**

```yaml
providers:
  # ── Named provider entries ─────────────────────────────────────────
  llm:
    - name: frontier
      type: deepagents-model
      model: openrouter:z-ai/glm-4.5-air:free
      request_timeout: 600
      enabled: true

    - name: local-vllm
      type: deepagents-model
      model: openai:qwen3.6-local
      base_url: http://localhost:8000/v1
      api_key: vllm
      temperature: 0.7
      enabled: true

    - name: local-tiny
      type: deepagents-model
      model: openai:dolphin3.0-local
      base_url: http://localhost:8001/v1
      api_key: vllm
      temperature: 0.1
      enabled: true

  # ── Per-phase and subagent overrides ───────────────────────────────
  phases:
    # Heavy reasoning → frontier model (via reference)
    specify:
      provider: frontier
    plan:
      provider: frontier

    # Implementation → local vLLM, but colder temperature for code
    implement:
      provider: local-vllm
      temperature: 0.3           # freezes just the temp override

    # Slice implementer subagent gets the same vLLM but even faster
    implement/subagents/slice-implementer:
      model: openai:codestral-local
      provider: local-tiny

    # Verification → cheap local model with custom endpoint
    verify:
      model: openai:tiny-model
      base_url: http://other:8000/v1       # dedicated server
      api_key: other-key
      request_timeout: 120                 # short timeout for verify
```

**Provider resolution for ``base_url``, ``api_key``, and other settings** follows
the same hierarchy as model resolution but with an extra dimension — you can
either *inherit* a named provider's settings or *inline* the values directly:

1. Phase config's direct provider keys (``base_url``, ``temperature``, etc.)
   — always take priority
2. Phase config's ``provider`` reference — look up ``providers.llm[name]``
   and inherit all its settings
3. First enabled entry in ``providers.llm[]``

Direct keys are applied *on top of* the inherited provider, so you can
reference a provider and then override just one or two fields:

```yaml
# Big provider defined once...
llm:
  - name: local-vllm
    model: openai:qwen3.6
    base_url: http://localhost:8000/v1
    api_key: vllm
    temperature: 0.7

# ...then referenced with per-phase tweaks:
phases:
  implement:
    provider: local-vllm
    temperature: 0.3   # colder for code generation
  verify:
    provider: local-vllm
    temperature: 0.1   # near-deterministic for tests
```

Only the settings change — each subagent still inherits the parent phase's
system prompt, tool restrictions, and skill configuration.

### File Structure

```
spine/
├── spine/                       # Main package
│   ├── core/                    # State machine, hierarchy, persistence, learning
│   ├── ui/                      # Streamlit dashboard (8 pages + utils)
│   ├── work/                    # Dispatcher, Ralph Loop worker
│   ├── adapters/                # Deep Agents phase adapters
│   ├── middleware/               # Critic gate, step limit, message queue
│   ├── providers/               # LLM, agent, memory, storage providers
│   ├── swarm/                   # Agents, gates, supervisor, mail
│   ├── workflows/               # SDD, Quick Work workflows
│   ├── cli/                     # Click commands
│   ├── github/                  # GitHub client, issue resolver
│   ├── git/                     # Worktree manager, PR handler
│   ├── jobs/                    # Task worker
│   ├── discovery/               # Code analysis tools
│   ├── hive/                    # Hive memory system
│   ├── config/                  # Queue configuration
│   ├── models/                  # Types, enums, DAG executor
│   └── prompts/                 # Prompt builders and templates
├── backend/                     # FastAPI REST API
│   ├── main.py
│   ├── routes/                  # work, status, audit
│   └── schemas/                 # Pydantic models
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── .spine/                      # Runtime data (checkpoints, config, DB)
├── pyproject.toml
└── README.md
```

## Configuration

After `spine init`, a `.spine/` directory is created with:

- `config.yaml` — Provider configuration, queue settings, agent resources
- `spine.db` — SQLite checkpoint database for state machine persistence
- `queue.db` — SQLite task queue backend

### Setting up vLLM + Deep Agents

For best results, run a local vLLM instance:

```bash
vllm serve Qwen3.6-35B-A3B --quantization awq --max-model-len 8192
```

Then configure `config.yaml` with the Deep Agents provider type pointing to your vLLM endpoint.

### Queue Backends

The queue supports both Redis and SQLite backends. Configure in `config.yaml`:

```yaml
queue:
  backend: sqlite   # or redis
  redis_url: "redis://localhost:6379/0"  # only for redis backend
```

## Dependencies

- **Core:** LangGraph, LangGraph Checkpoint SQLite, LangGraph Supervisor, Deep Agents
- **LLM:** OpenAI SDK (for provider abstraction)
- **Data:** Pydantic, PyYAML, SQLAlchemy, Alembic
- **CLI:** Click, Rich
- **UI:** Streamlit
- **API:** FastAPI, HTTPX, SQLite-Utils

## Testing

```bash
pytest tests/unit/        # Unit tests
pytest tests/integration/  # Integration tests
pytest tests/e2e/          # End-to-end tests
```

## Future Plans

- [ ] Enhanced spec-driven development workflows
- [ ] Multi-repo and monorepo support
- [ ] Real-time collaboration and multi-agent orchestration
- [ ] Extended provider integrations (more LLM backends, custom providers)
- [ ] Performance optimizations for long-running workflows
- [ ] API documentation generation
- [ ] Plugin system for custom providers and middleware

## License

This project is proprietary. All rights reserved.
