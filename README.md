# SPINE

**A deterministic AI agent harness with a state machine workflow engine.**

SPINE orchestrates AI agents through a structured lifecycle — specify, plan, implement, verify — using [LangGraph](https://github.com/langchain-ai/langgraph) for workflow topology and checkpoint persistence, [Deep Agents](https://github.com/deepagents/deepagents) for in-process LLM agent loops, and a [Streamlit](https://streamlit.io) dashboard for visibility and human review.

```
SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY
```

Each phase runs as a nested LangGraph subgraph with isolated SQLite checkpoints, so workflows can be interrupted and resumed at any point without losing progress.

---

## Project Goals

Beyond being a useful harness, SPINE is an experiment in **AI building its own tools**.
Two goals shaped the project:

1. **Self-development feasibility** — Can LLMs meaningfully build and modify the very
   tools and harnesses they run inside? SPINE was written **exclusively** by AI agents,
   intentionally, to find out.
2. **Local-first inference** — Can a serious agent workflow run on ~30B-class models
   served locally, not just frontier APIs? (See [Models](#models).)

### An AI-built harness

The codebase was developed start-to-finish by AI coding agents. It **bootstrapped**
from general-purpose tools — early scaffolding and the first working phases were
built with [OpenCode](https://github.com/opencode-ai/opencode) and
[Claude Code](https://github.com/anthropics/claude-code) driving the edits.

As the harness matured, the project crossed an inflection point: SPINE became capable
enough to **work on itself**. Specifications, plans, implementation slices, and
verification for new SPINE features can now be run *through SPINE* — the harness
specifying, building, and checking its own next iteration. The bootstrap tools remain
useful for ad-hoc work, but SPINE is now a participant in its own development.

This makes the repository both the product and the proof: a deterministic agent
harness whose own git history is evidence that LLM-driven, self-improving tool
development is feasible today.

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

---

## Models

SPINE relies on **three distinct model types**. The general LLM is mandatory; the
embedding model is required for the RAG/recall features; the reranker is optional
and off by default. All three can be served from a single OpenAI-compatible
endpoint (e.g. one vLLM or llama.cpp server) or mixed across providers.

> **Quick local setup:** For the easiest path to local inference, try
> [**Lemonade Server**](https://github.com/lemonade-sdk/lemonade) — it serves LLM,
> embedding, and reranking models behind one OpenAI-compatible endpoint with minimal
> configuration, so you can point all three `providers.*` blocks below at a single
> `base_url`.

| Type | Required? | Config block | Purpose |
|------|-----------|--------------|---------|
| **General LLM** | Yes | `providers.llm[]` | Drives every phase agent (specify/plan/critic/implement/verify), subagents, classification, and summarization |
| **Embedding model** | For RAG | `providers.embedding[]` | Vectorises code/symbols for `spine index` and powers the dense channel of hybrid recall |
| **Reranking model** | Optional | `providers.reranker[]` | Cross-encoder that re-scores recall candidates; off unless `reranker_provider` is set |

### General LLM

> **Local inference is a first-class goal.** SPINE was designed to run a full
> workflow on **~30B-class models served locally** — not just frontier APIs. Recent
> testing confirms this works end-to-end: a single local 30B-class model can drive
> every phase, including tool calling and structured artifact output.

The general LLM runs the agent loop in every phase. Two capabilities matter most:

- **Tool / function calling** — phases drive work through tools (`write_file`,
  `edit_file`, `write_plan`, MCP codebase-index tools, …), some with non-trivial
  argument schemas, so robust function-calling is the single most important
  capability.
- **Structured output** — set `guided_decoding: true` for models that need
  grammar-constrained decoding to emit valid structured artifacts. Frontier models
  with native structured output (e.g. `deepseek-v4-pro`) can leave it `false`.

A context window of **at least 32K** is recommended; the IMPLEMENT phase compactor
is tuned for a 60K threshold (see `token_compaction` in config).

**Models exercised during development & testing:**

| Model | How served | Notes |
|-------|------------|-------|
| `deepseek/deepseek-v4-pro` | OpenRouter | Strongest all-rounder; native structured output (`guided_decoding: false`). Good escalation target for the researcher subagent. |
| `deepseek/deepseek-v4-flash` | OpenRouter | Cheaper/faster; solid default for most phases. |
| `z-ai/glm-5.1` | OpenRouter | Capable general model; reliable tool calling. |
| `google/gemini-3.5-flash` | OpenRouter | Fast, cheap; good for classification/summarization phases. |
| `poolside/laguna-m.1` | OpenRouter (free tier) | Useful for low-cost runs. |
| **`poolside/laguna-xs.2`** | OpenRouter (free tier) / local | **Recommended local model** — the most reliable small model tested for the IMPLEMENT phase; good tool-calling and structured output on modest hardware. |

> Reliability varies between models — some local models we evaluated produced
> inconsistent tool calls or structured output. Stick to a tested model; `laguna-xs.2`
> was the most reliable small local model in our runs.

> **Routing tip:** a single capable model can run every phase. To optimise for cost
> or latency you can mix — e.g. a frontier model for reasoning-heavy phases
> (specify/plan/critic/verify) and a local model such as `laguna-xs.2` for IMPLEMENT.
> If a particular subagent (e.g. the researcher's MCP-heavy tool calls) underperforms
> on your chosen model, escalate just that step via the
> `specify/subagents/researcher` provider override.

### Embedding model

Required for `spine index` and the dense (vector) channel of hybrid recall. The
embedding dimension is **auto-probed** from the live model at index time — no `dim`
setting needed. After swapping embedding models, rebuild with `spine index --wipe`.

Asymmetric models need `query_prefix` / `document_prefix` set so queries and
documents are encoded into the same space.

**Tested:** [`nomic-embed-text-v2-moe`](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)
(768-dim), served via vLLM/llama.cpp. It is asymmetric and **requires** the
prefixes:

```yaml
providers:
  embedding:
    - name: local-embeddings
      type: openai-embedding
      model: nomic-embed-text-v2-moe-GGUF
      base_url: http://localhost:8000/v1
      api_key: vllm
      query_prefix: "search_query: "
      document_prefix: "search_document: "
```

> Hybrid recall fuses BM25 (lexical) and the vector channel via RRF. On the
> code-symbol benchmark (`tests/recall_eval`), lexical BM25 dominated, so the dense
> channel is kept at a small hedge weight (`rrf_vector_weight: 0.2`). Pick an
> embedding model that fits your corpus and **measure with `tests/recall_eval`**
> before trusting it.

### Reranking model

Optional and **off by default**. A cross-encoder re-scores the top recall
candidates for higher precision. To enable, serve a reranker behind a
Cohere/Jina/vLLM-compatible `/rerank` (or llama.cpp `/reranking`) endpoint, add it
under `providers.reranker[]`, and set `reranker_provider` plus `rerank_pool` in the
top-level `spine:` block.

**Tested:** [`bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
(GGUF), served via llama.cpp:

```yaml
spine:
  reranker_provider: bge-reranker
  rerank_pool: 50

providers:
  reranker:
    - name: bge-reranker
      model: bge-reranker-v2-m3-GGUF
      base_url: http://localhost:8000/v1
      api_key: vllm
      rerank_path: "/reranking"   # omit to auto-probe /rerank then /reranking
```

> Always validate a reranker against `tests/recall_eval` — it adds latency, so only
> keep it if it measurably improves MRR/recall on your corpus.

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
