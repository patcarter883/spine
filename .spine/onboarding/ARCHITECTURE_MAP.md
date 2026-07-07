# Architecture Map

## Module: spine.agents

`spine.agents` (57 files under `spine/agents/` plus `spine/agents/tools/`, ~25K lines) implements every LLM-driven executor in the SPINE workflow: the phase agents (specify, plan, implement, verify, critic, and related review/gap-plan phases), their delegated subagents, and the tool surfaces bound to each. It is the largest module because each phase needs its own tightly-scoped prompt, tool whitelist, and structured-output schema, and because cross-cutting context engineering (prompt caching, token-budget eviction, artifact materialization, model/backend resolution) is implemented once here and shared rather than duplicated per phase. The central entry point is `build_phase_agent` in `spine/agents/factory.py`; parallel work below the phase agent (researcher, slice-implementer, slice-verifier) is dispatched via LangGraph's `Send` API from the workflow's per-phase subgraph routers, not from inside the agent itself — phase agents deliberately carry no `eval` interpreter or subagent-dispatch tool. Verified imports show `spine.agents` depends on `spine.config`, `spine.models`, `spine.mcp`, `spine.work`, `spine.workflow`, `spine.persistence`, and `spine.ui_api`; it is in turn imported by `spine.phases.*`, `spine.workflow.*` (subgraphs, compose, artifact_gate), `spine.critic`, `spine.adversarial`, `spine.project`, `spine.cli`, and `spine.ui`.

### build_phase_agent (`spine/agents/factory.py`)

Single entry point for constructing every phase agent. Resolves the model via `helpers.resolve_model`, builds the backend, assembles the system prompt (phase prompt + onboarding doc reference + cross-run experience + base prompt), and returns a `create_agent` graph. Assembles a fixed middleware stack — `FilesystemMiddleware` (or a purpose-built tool replacement when `skip_filesystem_middleware=True`), `ReadCacheMiddleware`, `PatchToolCallsMiddleware`, `ToolSchemaValidator`, `SearchLoopGuard`, `TokenBudgetCompactor`/`DynamicCompletionCapMiddleware`, and `StaticPrefixCacheMiddleware`/`AnthropicPromptCachingMiddleware` — in a documented order.

### Subagent definitions (`spine/agents/subagents.py`)

Defines the three subagent specs dispatched by phase subgraph routers: `researcher` (read-only investigation for SPECIFY/PLAN, response schema `ResearchFindings`), `slice-implementer` (IMPLEMENT, tool-restricted to `read_edit_lint` only, schema `SliceResult`), and `slice-verifier` (VERIFY, restricted to `read_file`/`run_checks`, schema `VerificationResult`, with an alternate no-tool "evidence-then-judge" prompt path gated by `verify_evidence_then_judge`). `build_subagent_spec` resolves model, memory, and tool list per subagent into the dict `SubAgent` spec Deep Agents expects.

### Structural decomposer (`spine/agents/decomposer.py`)

`run_decomposer` splits a specification (PLAN mode), a failed slice plus traceback (FALLBACK mode), or a multi-file slice (PER_FILE mode) into `FeatureSliceSchema` units via one structured LLM call, populating an `edit_plan` of `EditHint` anchors (file/symbol/mode/action) so implementers can apply edits directly. `enrich_slice` fills in `edit_plan` for an already-decomposed slice from indexed symbols, and `_scrub_phantom_symbols` strips any symbol/reference not found in the local codebase index before it reaches an implementer.

### ReadEditLintTool (`spine/agents/tools/read_edit_lint.py`)

The slice-implementer's only filesystem tool: a ranged/line-numbered read plus four mutually exclusive edit modes (exact `old_str`/`new_str`, `full_replace`, batched `edits`, and line-range replacement), plus an `ast_edit` mode anchored by symbol name across Python/PHP/TS/TSX. Edits are staged in memory and syntax-checked; a failing check leaves disk untouched so the model can retry in-loop.

### CodebaseQueryTool (`spine/agents/tools/codebase_query.py`)

Collapses five separate MCP-backed lookups (`find_symbol`, `get_source`, `get_dependencies`, `get_dependents`, `search`) into one tool with a single `action` enum, with a local SQLite fallback for PHP or when the MCP server is unavailable — built to eliminate a malformed-argument failure class seen when models were given five separate tool schemas.

### SpineContext (`spine/agents/context.py`)

The Pydantic context schema (`work_id`, `phase`, `workspace_root`, `retry_count`, `critic_feedback`, `artifact_paths`, `read_cache`) passed to every phase agent invocation via `context=`. `build_context` derives it from `WorkflowState` per invocation; tools read it through `ToolRuntime.context` instead of the prompt, and it propagates automatically to subagents.

### Model/backend resolution (`spine/agents/helpers.py`)

`resolve_model`/`resolve_chat_model` centralize per-phase model-override resolution against `SpineConfig`, hand-building provider-specific instances (`ChatOpenRouter` with `session_id`, or a manually wired `ChatOpenAI` for local `base_url` servers) because the generic `init_chat_model` string path silently drops required kwargs. Also owns structured-output plumbing (`bind_structured_output`, the `_SelfHealingStructured` method-demotion wrapper for OpenRouter routing 404s) and `extract_response`'s leaked-chain-of-thought detection.

### Artifact materialization (`spine/agents/artifacts.py`)

`materialize_artifacts`/`materialize_phase_artifacts` write prior-phase outputs to `.spine/artifacts/{work_id}/{phase}/` on disk so later phases reference them by path rather than having their full content inlined into the prompt — the mechanism that keeps later-phase prompts from growing unbounded.

**Notes:**
- This overview is grounded in 8 of the module's ~57 files read in depth (factory.py, helpers.py, subagents.py, decomposer.py, context.py, artifacts.py, plus partial reads of classification.py, profile.py, retry.py, tool_forcing.py, and two tools/ files); phase-specific agent files not traced here (plan_agent.py, verify_agent.py, implement_agent.py, gap_plan_agent.py, project_review/verify_agent.py, experience.py, garbage_collector.py, evidence_compression.py, researcher_supervisor.py, and others) were seen only in the file listing.
- The 651-symbol / 103-class / 374-function / 174-method counts come from the supplied plan fragment's metadata, not an independent recount.

## Module: spine.cli

The `spine.cli` package (`spine/cli/__init__.py`, a single flat module of 21 Click command functions) is the entry point for the `spine` executable. It defines the top-level `click.group()` `main`, which configures logging and stores the `--verbose` flag on the Click context, and hangs every subcommand off it: `run` (submit a work item and invoke `spine.work.dispatcher.submit_work`), `index` (build the RAG vector store via `spine.persistence.vector_store.VectorStore` and `spine.workflow.workers.vector_indexer.run_indexing_job`), `status_cmd`/`list_cmd`/`resume`/`restart` (query and drive `spine.work.dispatcher`), `worker` (start `RalphLoopWorker` via `spine.work.ralph_worker`), `ui` (shell out to `streamlit run` against `spine/ui/app.py`), `export` (render a work item through `spine.workflow.export`), and `init` (scaffold a new project via `spine.work.onboarding.init.init_workspace`). A nested `project` group (`create`/`add`/`show`/`list`/`verify`/`review`) manages multi-work-item project envelopes through `spine.persistence.project_store.ProjectStore`, `spine.project.aggregator`, `spine.project.project_verifier`, and `spine.project.project_reviewer`. A nested `experience` group (`list`/`clear`) inspects and prunes the cross-run distilled-lesson store via `spine.agents.experience.experience_store_for`.

Every command loads configuration first via `SpineConfig.load(path=config_path)` (imported from `spine.config`), defaulting `--config` to `.spine/config.yaml`, then defers to the relevant subsystem module — most imports are done lazily inside the command bodies rather than at module top level, presumably to keep CLI startup fast. Output goes through a shared `rich.console.Console` instance (`console`) using `Table` and `Panel` for formatted terminal output.

### main (`spine/cli/__init__.py`)

The root Click group and CLI entry point; calls `configure_logging` (from `spine.log`) and seeds `ctx.obj["verbose"]`.

### run, index, export (`spine/cli/__init__.py`)

The three commands most explicitly documented at the fragment level: `run` submits/creates a work item (asyncio-driving `submit_work`), `index` performs incremental tree-sitter-based symbol extraction and embedding into the vector store, and `export` writes a work item's specification/plan/research/prompts to Markdown or JSON.

**Notes:**
- Depends on (per the supplied dependency edges) `spine.agents`, `spine.models`, `spine.persistence`, `spine.project`, `spine.ui`, `spine.work`, and `spine.workflow` — this module is a thin orchestration layer with no business logic of its own.
- Command bodies do local `import` of dispatcher/persistence/project modules rather than importing at module top; this is read directly from the source, not inferred.

## Module: spine.config.py

`spine/config.py` is a single ~1,370-line module that is the canonical source of runtime configuration for the whole SPINE agent system. Its centerpiece is the `SpineConfig` dataclass, loaded via the classmethod `SpineConfig.load(path=".spine/config.yaml")`, which reads YAML, applies environment-variable overrides, and falls back to hardcoded defaults for any missing key (e.g. `checkpoint_path=".spine/spine.db"`, `artifact_path=".spine/artifacts"`, `max_critic_retries=2`). `load()` also resolves `workspace_root` (auto-detecting upward from CWD or the package directory for a `.spine/` marker via the static helper `_find_workspace_root`), parses `mcp_servers`, and builds the two companion dataclasses `ConvergenceConfig` (researcher convergence thresholds and supervisor max-cycle budgets for the specify/plan phases) and `TokenCompactionConfig` (token-budget compaction thresholds for phase agents), via module-level parser functions `_parse_convergence_config` and `_parse_token_compaction_config`.

Beyond loading, `SpineConfig` exposes the model/provider-resolution surface the rest of the system calls into: `resolve_model(phase, escalation_level)` walks a documented precedence order (explicit phase model → named provider lookup → path-prefix walk for subagent overrides → escalation-ladder entry → first enabled provider → `SPINE_MODEL` env var → `ValueError`); `resolve_active_provider()` and `_lookup_provider_by_name()` look up entries in `providers.llm[]`; `resolve_provider_config()` merges provider-level and phase-level overrides for keys like `base_url`, `temperature`, and `max_tokens`; `resolve_embedding_provider()`/`resolve_reranker_provider()` do the equivalent lookup for `providers.embedding[]`/`providers.reranker[]`. `ensure_dirs()` creates the checkpoint/artifact/queue/project/experience directories on disk. The module also has side effects on import: `_load_dotenv()` loads a project-root `.env` file, and `_disable_global_tracing()` forces LangSmith/LangChain tracing env vars off unless `SPINE_TRACE_ALL` is set.

### SpineConfig (`spine/config.py`)

The dataclass holding every runtime setting (paths, retry limits, provider dicts, RAG/recall settings, `ephemeral_pod`/`ephemeral_pods` blocks, etc.) plus the resolution methods described above.

### ConvergenceConfig / TokenCompactionConfig (`spine/config.py`)

Small companion dataclasses for researcher-loop convergence steering and token-compaction behavior, parsed from the `convergence:` and `token_compaction:` YAML blocks respectively, each overridable by matching `SPINE_*` env vars.

**Notes:**
- The listed dependency edge (`spine.config.py` → `spine.agents`) reflects one real, narrow coupling actually observed in source: `_resolve_model_profile` does a local `from spine.agents.helpers import _extract_model_name` to normalize a model-spec string against `model_profiles` keys.
- This module was skimmed rather than read line-by-line given its size (1,373 lines); the summary above covers the public class and the methods whose bodies were directly inspected (`load`, `resolve_model`, `resolve_active_provider`, `resolve_embedding_provider`, `resolve_reranker_provider`, `ensure_dirs`). Other private helpers (`_escalation_entry`, `_escalation_entry_for`, `_resolve_model_profile`) were read but not exhaustively cross-checked against every caller.

## Module: spine.exceptions.py

`spine/exceptions.py` is a small (113-line), fully self-contained module defining SPINE's exception hierarchy — no imports beyond `from __future__ import annotations`, and no dependencies on any other spine submodule. Every exception ultimately subclasses the root `SpineError(Exception)`. From it hang two second-tier bases: `WorkflowError` (errors in workflow execution/phase transition) and `GitOrchestratorError` (base for the transactional git-sandbox orchestrator). `WorkflowError` has one subclass, `MaxRetriesExceeded(phase, retries)`, raised when a phase exceeds its configured critic retry limit. `GitOrchestratorError` has two subclasses: `SandboxPreparationError` (worktree/branch creation failed on a dirty tree or failed git command) and `ValidationError(gate_name, command, output)` (a pipeline validation gate failed, carrying the gate name, command, and captured output) and `MergeError` (a fast-forward merge of the verified patch branch failed, typically because main advanced).

Directly under `SpineError` sit several flat, single-purpose exceptions: `CriticError` (critic review execution errors), `PromptRequestError` (human prompt-request handling errors), `AgentUnavailableError` (no agent provider available for phase execution), `ConfigurationError` (invalid/missing configuration), `TransientAPIError(original)` (used internally by `invoke_with_retry()` to mark a retryable 5xx/429/provider error — not normally raised to callers since retries exhaust first), and `CriticalContractFailure(phase, reason, carryover=None)` — the most elaborate of the group, signaling a broken phase precondition/invariant that requires human intervention, and optionally carrying a `carryover` dict of subgraph state (e.g. exploration findings) to seed a wrapper's structural retry so it doesn't repeat completed work.

### SpineError (`spine/exceptions.py`)

The root of the hierarchy; a bare `Exception` subclass with only a docstring, everything else in the module descends from it (directly or via `WorkflowError`/`GitOrchestratorError`).

### CriticalContractFailure (`spine/exceptions.py`)

The only exception with meaningfully custom `__init__` behavior beyond message formatting: stores `phase`, `reason`, and `carryover` (defaulting to `{}`), documented in-source as letting a failed synthesis retry skip re-running an intact exploration phase.

**Notes:**
- 13 classes total in the file (confirmed by direct count: `SpineError`, `WorkflowError`, `CriticError`, `MaxRetriesExceeded`, `PromptRequestError`, `AgentUnavailableError`, `ConfigurationError`, `TransientAPIError`, `CriticalContractFailure`, `GitOrchestratorError`, `SandboxPreparationError`, `ValidationError`, `MergeError`), matching the fragment's "13 classes/interfaces" count.
- The supplied fragment listed zero dependency edges for this module, and the source confirms it: the file has no `import` of any other spine module.

## Module: spine.git

`spine.git` (20 symbols: 2 classes/interfaces, 2 functions, 16 methods) is Spine's transactional git-sandbox layer. It gives any code-producing workflow run an isolated, revertible execution boundary: prepare a throwaway git worktree, run the workflow against it, optionally gate the result through a configurable validation pipeline, and only then land it on `main` — or roll everything back. The two public classes are `SpineGitOrchestrator` (`spine/git/orchestrator.py`) and `WorktreeSandbox` (`spine/git/sandbox.py`); the module's other 18 symbols are their supporting methods and two module-level functions (`load_gate_config`, `work_type_writes_code`).

`SpineGitOrchestrator` owns the low-level git mechanics: `prepare_sandbox()` creates a branch (`spine/patch-<token>`) and worktree off `main_branch`, after `ensure_tree_clean()` confirms `git status --porcelain` is empty; `run_validation_pipeline()` runs each configured gate (lint/typecheck/test) in order, resolving `.venv/`-prefixed commands against the *master* tree since worktrees don't copy virtualenvs; `commit_and_merge()` fast-forward merges (raising `MergeError` on conflict); `rollback_workspace()` force-removes the worktree/branch and hard-resets the master tree. Configuration comes from `spine-gate.yaml` via `load_gate_config()`, falling back to built-in defaults. Its `execute_transactional_run()` method is explicitly documented as deprecated in-code — it would nest a second sandbox on top of the one `WorktreeSandbox` already creates.

`WorktreeSandbox` is the thin, mandatory policy layer `spine.work.dispatcher` actually calls on every `submit_work`/`resume_work`/`restart_work`. `work_type_writes_code()` checks `spine.workflow.compose.WORKFLOW_SEQUENCES` for an IMPLEMENT phase; if present, `enter()` builds a `SpineGitOrchestrator` and returns a `SpineConfig` copy with `workspace_root` swapped to the new worktree. `finalize(status)` merges when `status == "completed"`, otherwise rolls back; `abort()` force-rolls-back on an unhandled exception. Non-code work types make every method a no-op.

### SpineGitOrchestrator (`spine/git/orchestrator.py`)

Implements `prepare_sandbox`, `run_validation_pipeline`, `commit_and_merge`, `rollback_workspace`, `execute_transactional_run` (deprecated), and `status`. Reads `spine-gate.yaml` (git strategy, branch prefix, validation commands, `require_successful_phases`).

### WorktreeSandbox (`spine/git/sandbox.py`)

Implements `enter`, `preflight`, `finalize`, `abort`. `active` is set from `work_type_writes_code(work_type)` at construction; every method is a no-op when `active` is False, so callers can wrap all runs uniformly regardless of work type.

**Notes:**
- The supplied dependency edges list only `spine.git → spine.models` and `spine.git → spine.workflow`, but `orchestrator.py`'s own deprecated `execute_transactional_run()` imports `submit_work` from `spine.work.dispatcher` — so a reverse, legacy-path dependency on `spine.work` exists in code even though it isn't in the edge list. The live, primary direction is `spine.work → spine.git` via `WorktreeSandbox`.
- `_DEFAULT_GATE_CONFIG`'s lint/typecheck/test commands (`ruff`, `mypy`, `pytest`) are this repo's specific defaults, not a generic property of the module.

## Module: spine.infra

`spine/infra/` (its `__init__.py` is a one-line docstring: "SPINE infrastructure helpers (ephemeral compute, etc.)") contains a single substantial file, `spine/infra/ephemeral_pod.py`, which manages the full lifecycle of remote GPU pods used to serve an LLM for the duration of one `spine` run. The design intent, stated in the module docstring, is that a run only pays for GPU while actually executing: pods are brought up at the start of a dispatcher entry point and torn down in a `finally` covering success, exception, and `KeyboardInterrupt`. Configuration comes from `.spine/config.yaml`'s `ephemeral_pod:` (legacy single-pod, always boots when enabled) or `ephemeral_pods:` (list of named pods, each scoped to the phases it serves) blocks, parsed into `EphemeralPodConfig` dataclass instances by `parse_pods(config)`. Which phases a pod serves is intersected against the phases a run will actually execute (computed by `executed_phases_for_run`/`lane_phase_set`, using the `PLANNING_PHASES`/`EXECUTION_PHASES` frozensets) via `select_pods`, so e.g. a `reviewed_task` run (planning lane only, ending at the human-approval gate) never boots an implement/verify pod.

Booting is backend-specific: `_boot_runpod` (RunPod is currently the only implemented backend) calls the `runpod` SDK's `create_pod` with kwargs built by `_create_kwargs`/`_docker_args` (supporting both a vLLM and an SGLang serving engine, chosen by `EphemeralPodConfig.engine`), retries across a `gpu_fallback` ladder on capacity errors (`_gpu_attempts`), polls `GET /v1/models` via `_http_ok` until the server answers, and publishes the live URL into an env var (`SPINE_POD_BASE_URL` for the legacy single pod, or `SPINE_POD_URL__<NAME>` for a named pod, per `EphemeralPodConfig.url_env`) rather than through the config object — the docstring explains this is because provider resolution reloads `SpineConfig` from disk every phase. Lifecycle is tracked by a process-wide reference-counted registry (`_PODS`, `_REFCOUNT`, guarded by `_LOCK`) via `acquire()`/`release()`, so nested dispatcher entry points (e.g. resume → restart → `_run_workflow_graph`) share one pod and only the outermost caller's release tears it down. The `with_ephemeral_pod` decorator wraps async dispatcher entry points to acquire/release automatically. `_RunPodLease` wraps one booted pod plus an optional watchdog task (`max_lifetime_s`) that force-terminates it if a run hangs.

### EphemeralPodConfig (`spine/infra/ephemeral_pod.py`)

Dataclass parsed from one `ephemeral_pod`/`ephemeral_pods[]` entry: backend, engine (`vllm` or `sglang`), GPU spec, image, startup timeout, `on_failure` policy (`abort` or `local_fallback`), and `gpu_fallback` ladder. `_from_raw` derives `model` from the bound `providers.llm[]` entry (`binds_provider`) when not set explicitly.

### acquire / release (`spine/infra/ephemeral_pod.py`)

The reference-counted entry points dispatcher code calls to get a `_Lease`; `acquire` boots only pods not already up for this run and rolls back anything it itself booted on failure, `release` tears every pod down once the last outstanding lease is released.

### PodStartupError (`spine/infra/ephemeral_pod.py`)

`RuntimeError` subclass raised when a pod can't be brought up and its policy is `abort` (as opposed to falling back to `fallback_base_url`).

**Notes:**
- RunPod is the only backend with an implementation in this file (`_import_runpod`, `_boot_runpod`); the config's `backend` field exists but no alternate backend branch was found in source.
- The fragment listed zero dependency edges for `spine.infra`; the source shows a `TYPE_CHECKING`-only import of `spine.config.SpineConfig` and a runtime local import of the same in `with_ephemeral_pod`'s wrapper — a thin, one-directional coupling, not a two-way dependency.

## Module: spine.mcp

`spine.mcp` (`spine/mcp/`) is the adapter layer that connects SPINE's agents to external Model Context Protocol servers. The entire implementation lives in `spine/mcp/client.py` — `defaults.py` and `tools.py` are now empty stub files whose only content is a comment pointing to `client.py`, left in place for import-path compatibility. The module wraps LangChain's `MultiServerMCPClient` (from `langchain-mcp-adapters`): `_get_client` lazily builds and caches one client per unique server-config hash, and `get_mcp_tools` is the public entry point that converts SPINE's server config dicts into adapter-compatible dicts (`_convert_server_config`), fetches tools via `_run_async` (a sync/async bridge that spins up a thread pool when already inside a running event loop, since SPINE's subagents may call MCP tools from different LangGraph event loops), and namespaces every returned tool with a `mcp_{server_name}_` prefix (`_namespace_tool`) to avoid collisions across servers.

A second responsibility is defensive post-processing of the codebase-index MCP server's results, since raw tool output was observed dumping up to ~190 KB into an agent transcript in one call. `_wrap_for_postprocessing` monkey-patches any tool named `mcp_codebase-index_*` so its `_run`/`_arun` output passes through `_post_process_result`, which chains `_strip_excluded_paths` (drops results under `.spine/artifacts/`, `.spine/checkpoints/`, `.spine/spine.db`, or any dot-folder segment, via `_is_excluded_path` / `_line_starts_with_excluded_path`) and `_cap_result` (enforces `SEARCH_CODEBASE_MAX_HITS` = 50 lines and `SEARCH_CODEBASE_MAX_BYTES` = 8192 bytes, appending a truncation note that nudges the agent toward narrower queries or structural tools like `find_symbol`/`get_dependencies`).

Dependency-wise, `spine.mcp.client` imports `spine.agents.symbol_cache` (calling `register_cacheable_server` so each configured server's deterministic read-only tools opt into cross-branch result caching). In the other direction, the only in-repo consumer found is `spine/agents/tools/codebase_query.py`, which imports `get_mcp_tools` and `_is_excluded_path` from `spine.mcp.client`.

### get_mcp_tools (`spine/mcp/client.py`)

The module's public entry point. Given a dict of server_name → {command, args, env, ...}, forces every server's `PROJECT_ROOT` env var to match SPINE's `workspace_root` (overriding any user-configured value), fetches and namespaces tools, and wraps codebase-index tools for post-processing. Returns an empty list if `server_configs` is falsy, and an empty list (with a logged warning) if the underlying client raises.

### _get_client / _run_async (`spine/mcp/client.py`)

`_get_client` memoizes a single `MultiServerMCPClient` keyed by a hash of the server configs, rebuilding it only when the config changes. `_run_async` executes the resulting coroutine synchronously — using `asyncio.run` directly, or offloading to a `ThreadPoolExecutor` (60s timeout) when a loop is already running.

### _cap_result / _strip_excluded_paths (`spine/mcp/client.py`)

Enforce the size/hit ceilings and excluded-path filtering described above. `_strip_excluded_paths` handles both the JSON-array result shape (`[{"file", "line_number", "content"}, ...]`) and a plain-text, one-match-per-line shape.

**Notes:**
- `spine/mcp/defaults.py` and `spine/mcp/tools.py` contain no code — both are single-line "Removed" stubs; all logic is in `client.py`.
- The fragment listed only 8 of the module's 13 functions (e.g. `get_mcp_tools` and `_wrap_for_postprocessing` were not in the supplied fragment); this section adds them from the source file, which was read in full.
- Only one in-repo importer of `spine.mcp` was found (`spine/agents/tools/codebase_query.py`); this reflects an in-repo grep, not a claim about all possible callers.

## Module: spine.models

`spine.models` (`spine/models/`) is the shared type layer: the Pydantic models, dataclasses, `TypedDict`s, and enums that every other part of SPINE — workflow graph, agents, persistence, CLI, UI API — imports rather than redefining. It has no behavior of its own beyond serialization helpers; its job is to give the rest of the codebase one set of vocabulary for tasks, artifacts, reviews, plans, and workflow state. A repo-wide check found 57 other files importing from `spine.models`, consistent with it sitting at the bottom of the dependency graph alongside `spine.persistence` (which itself imports `ExperienceLesson` and `ProjectSpec` from `spine.models.types`).

`spine/models/enums.py` defines `PhaseName` (the LangGraph node identifiers: `specify`, `plan`, `tasks`, `implement`, `verify`, `critic`, `adversarial`, `gap_plan`, `project_verify`, `project_review`), `WorkType` (`task` / `critical_task` / `reviewed_task` / `critical_reviewed_task`, each implying a different phase chain — critical types insert an adversarial red-team stage after plan critique, reviewed types pause for human approval before implement), `ReviewStatus`, and `TaskStatus` (including a `CANCELLED` terminal state used by the Ralph worker's queue loop).

`spine/models/types.py` is the larger file, split into four groups: legacy dataclasses (`Task`, `Artifact`, `ReviewFeedback`, `PromptRequest`); slice-planning models (`FeatureSlice` — a single implementation slice with target files, dependencies, `reference_symbols`/`provides` for cross-slice API contracts, and an `edit_plan`; and `StructuredPlan`, its container); specification/gap-planning Pydantic models (`Specification`, with a `hard_boundaries` field enforced deterministically as a scope gate; `FixInstruction`, `GapPlan`); and the project/milestone layer (`RequirementRef`, `RoadmapPhase`, `Roadmap`, `ProjectSpec` — persisted by `ProjectStore`). `CriticReview` and `ExperienceLesson` round out the file — `CriticReview`'s docstring notes its field order is deliberately `reason` before `status` because it's a guided-decoding schema and reasoning fields must precede verdict fields; `ExperienceLesson` is the record type `ExperienceStore` persists.

`spine/models/state.py` defines `PhaseResult` (a `TypedDict` summarizing one phase subgraph's output) and `WorkflowState`, the full LangGraph `StateGraph` state schema, along with the custom reducers (`_merge_dicts`, `_append_capped_feedback`, `_merge_read_cache`, `_merge_artifacts`) that LangGraph uses to combine per-node updates into a single accumulating state across the run.

### FeatureSlice / StructuredPlan (`spine/models/types.py`)

`FeatureSlice` is a dataclass describing one topologically-orderable unit of implementation work (target files, `dependencies`, `acceptance_criteria`, `reference_symbols`, `provides`, `edit_plan`, `reference_only_files`); `StructuredPlan` is the ordered container of slices that replaces a prose `plan.md`.

### CriticReview (`spine/models/types.py`)

Structured critic-verdict schema (`reason`, `status`, `tier`, `suggestions`, `cited_exclusions`, `score`, `blocker_category`) returned by critic phases.

### WorkflowState (`spine/models/state.py`)

The `TypedDict` state schema for SPINE's LangGraph `StateGraph`, covering phase status/retry tracking, artifacts, feedback, verify/gap-plan bookkeeping, adversarial-review tracking, and human-review interrupt fields.

**Notes:**
- `FeatureSlice.from_dict` silently drops unknown keys, an explicit forward-compatibility choice so older consumers don't break on newer plan payloads.
- `blocker_category="spec_contradiction"` is reserved for cases the critic judges unfixable by reworking the phase under review — it signals a spec-amendment escalation rather than another rework cycle.
- `human_feedback` must be declared as a state channel for LangGraph to commit it at all; the module's own comment records a past bug (trace 019f1628) where an undeclared channel caused every human-review resume to silently default to "abort".

## Module: spine.persistence

`spine.persistence` (`spine/persistence/`) is SPINE's storage layer: every place the workflow engine needs something to survive past a single process invocation goes through one of its five store classes. Two physical backends are in play — plain files under `.spine/` for documents, and a shared SQLite database (`.spine/spine.db`) for checkpoints and the vector index — plus an Alembic migration scaffold (`spine/persistence/migrations/`, one revision so far: `001_create_vector_store.py`) targeting that same database via `SPINE_DB_PATH`.

`ArtifactStore` (`spine/persistence/artifacts.py`) writes workflow-phase output under `.spine/artifacts/{work_id}/{phase}/{name}`, pairing every file with a `.meta.json` sidecar (phase, size, timestamp) so `list_artifacts` can discover both sidecar-tracked files and "orphan" files the agent wrote directly via `write_file`, backfilling sidecars for the latter. `save_artifact` refuses to overwrite a longer on-disk file with a shorter one unless `overwrite_shorter=True`, guarding against truncated 500-char state previews clobbering full agent-written files. Consumers: `spine/work/dispatcher.py`, `spine/workflow/export.py`, `spine/workflow/subgraphs/exploration_subgraph.py`, `spine/ui_api/api.py`, and `spine/work/onboarding/engine.py` (which persists the onboarding docs this very task produces).

`CheckpointStore` (`checkpoint.py`) wraps LangGraph's `AsyncSqliteSaver` over `.spine/spine.db`, one `thread_id` per work item; `get_state`/`delete_state` are async and let read errors propagate rather than masking a locked database as "no checkpoint". Used by `spine/work/dispatcher.py`, `spine/work/ralph_worker.py`, `spine/git/orchestrator.py`, `spine/project/aggregator.py`, `spine/project/project_reviewer.py`, `spine/project/project_verifier.py`, and `spine/ui_api/api.py`.

`ExperienceStore` (`experience_store.py`) and `ProjectStore` (`project_store.py`) are both file-backed, `fcntl`-locked, read-modify-write JSON/JSONL stores (`.spine/experience/lessons.jsonl` and `.spine/project/{project_id}/spec.json` respectively), atomic via tmp-file-then-`os.replace`. `ExperienceStore` persists `ExperienceLesson` records (capped at 12 per phase, highest-salience kept) consumed by `spine/agents/experience.py`. `ProjectStore` persists `ProjectSpec` project/roadmap documents, consumed by the CLI (`spine/cli/__init__.py`), `spine/ui_api/api.py`, `spine/project/project_reviewer.py`, and `spine/project/project_verifier.py`.

`VectorStore` (`vector_store.py`) also lives in `.spine/spine.db` (opened via the same `checkpoint_path` config value as `CheckpointStore`) and layers a `sqlite-vec` `vec0` virtual table (cosine similarity) with an FTS5 `symbol_fts` table (Porter-stemmed BM25) over a `symbol_metadata` table. `search_hybrid` fuses both via reciprocal rank fusion (`_RRF_K = 60`); the module note says the local embedding model produces a near-random vector space for code, so BM25 does most of the real recall work. It also tracks a `symbol_edges` table for languages the AST indexer can't otherwise resolve (PHP), and an `indexed_files` ledger (content hash per file) enabling incremental re-indexing. Consumers: `spine/agents/tools/recall_tool.py`, `spine/workflow/workers/vector_indexer.py`, `spine/work/onboarding/analyzer.py`, and `spine/cli/__init__.py`.

### ArtifactStore (`spine/persistence/artifacts.py`)

Filesystem persistence for phase artifacts, with metadata sidecars and orphan-file detection/repair in `list_artifacts`. `save_artifact`'s "don't overwrite a longer file" guard is a deliberate anti-truncation measure, not a generic versioning scheme — it compares only lengths, not content or timestamps.

### CheckpointStore (`spine/persistence/checkpoint.py`)

Async LangGraph checkpoint persistence via `AsyncSqliteSaver`, one thread per work item, backed by `.spine/spine.db`. `list_checkpoints` is a stub — its docstring and the code both flag it as "a simplified synchronous wrapper" that returns an empty list; it does not actually enumerate checkpoints.

### ExperienceStore (`spine/persistence/experience_store.py`)

File-backed store for `ExperienceLesson` records distilled from critic/adversarial feedback, replayed into future prompts for the same phase. Deduplication keys off `ExperienceLesson.dedup_basis` (frozen pre-generalization text), not the user-visible `lesson` string, specifically so a later LLM paraphrase pass can't defeat cross-run dedup.

### ProjectStore (`spine/persistence/project_store.py`)

File-backed persistence for `ProjectSpec` (project/milestone) documents, with per-project `fcntl` locking around membership mutations. The store is the documented source of truth for project membership; the `project_id` column elsewhere is explicitly called out as only a denormalized reverse-lookup.

### VectorStore (`spine/persistence/vector_store.py`)

SQLite-backed (`sqlite-vec` + FTS5) hybrid vector/lexical search over indexed code symbols, plus a dependency-edge table and an incremental-index ledger. `insert` raises `ValueError` if the embedding width doesn't match the table's configured dimension — a model swap requires `spine index --wipe`, and `ensure_schema` will refuse to silently drop a populated table of the wrong dimension.

**Notes:**
- No dependency edges were supplied in the fragment for this module in one pass; the source-level couplings above were confirmed directly against each file.

## Module: spine.phases

`spine.phases` (spine/phases/) holds one file per SPINE workflow phase — `specify.py`, `plan.py`, `implement.py`, `verify.py`, `critic.py`, `adversarial.py`, `gap_plan.py` — plus an empty `__init__.py`. Each file defines an async `call_<phase>` node function (where applicable) and a `build_<phase>_agent`-style factory, then self-registers both with `spine.workflow.registry.PhaseRegistry` at import time via a module-level `_registry = get_registry(); _registry.register(...)` call. This registration is how `spine.workflow.registry.get_registry()` populates itself: `_import_phase_modules()` in `registry.py` imports all seven of these modules purely for that side effect.

Per the module docstrings read directly, this is currently a **legacy fallback path**: `implement.py` and `verify.py` state outright that their `call_implement`/`call_verify` functions are "kept as a fallback for when the `_SUBGRAPH_ENABLED` feature flag is turned off," and this was confirmed in `spine/workflow/compose.py` — that flag dict currently has every migrated phase (VERIFY, IMPLEMENT, TASKS, SPECIFY, PLAN, CRITIC, GAP_PLAN) set to `True` — so in the live graph, phases run via the `spine.workflow.subgraphs.*` builders instead, and these `call_*` functions are not on the hot path today. The `build_agent_fn` half of each registration is still meaningful, though for CRITIC and ADVERSARIAL it is a one-line delegation (`_build_critic_agent` calls `spine.critic.agent.build_critic_agent`; `_build_adversarial_agent` calls `spine.adversarial.agent.build_adversarial_agent`) rather than logic housed in `spine.phases` itself. `critic.py` additionally holds real validation logic used by its own legacy `call_critic`: `_load_plan_json` reads `plan.json` from state artifacts or disk, `_has_cycle` does a DFS cycle check, and `_validate_plan_structure` combines them to check `feature_slices` presence, field completeness, dependency integrity, and cycles. `plan.py`'s `_read_plan_json` and `_compute_waves` mirror this for the PLAN phase, converting slice dicts to `FeatureSlice` objects and calling `spine.workflow.slice_scheduler.compute_execution_waves`. `specify.py`'s `_early_commitment` classifies the incoming task and retrieves relevant code chunks via `RecallTool` before the specify agent runs.

### call_plan / _compute_waves (`spine/phases/plan.py`)

Legacy PLAN node: reads `plan.json`, computes execution waves, and returns a `needs_review` status with actionable feedback if wave validation fails (cycles or missing dependencies).

### call_critic / _validate_plan_structure (`spine/phases/critic.py`)

Legacy CRITIC node combining structural checks (artifacts exist/non-empty) with plan-structure validation for the PLAN-reviewing case.

### _build_adversarial_agent (`spine/phases/adversarial.py`)

Thin delegation to `spine.adversarial.agent.build_adversarial_agent`; `adversarial.py` registers no `call_fn` at all, only `build_agent_fn` — consistent with ADVERSARIAL running exclusively as a subgraph node in production.

**Notes:**
- All 8 example symbols in the supplied fragment were verified against the actual source; none were invented.
- Could not verify from the code alone whether any deployment currently flips a `_SUBGRAPH_ENABLED` flag to `False` in practice — this reports what the checked-in default dict in `compose.py` shows, not runtime configuration.

## Module: spine.project

`spine.project` (`spine/project/`) implements SPINE's project-level (multi-work-item) review and verification pipeline — the layer that runs after individual work items have each gone through their own VERIFY phase. It has three files: `aggregator.py`, `project_verifier.py`, and `project_reviewer.py`, each with a distinct job in the same overall flow: compute deterministic coverage, then run LLM-judged integration verification per roadmap phase, then run one holistic adversarial review of the whole project.

`aggregator.py` is explicitly documented as read-only, deterministic, and LLM-free. Its `aggregate_project_coverage` walks a `ProjectSpec`'s member work_ids, reads each member's latest checkpointed state via `CheckpointStore`, and computes per-requirement coverage using exact, normalized (`strip().casefold()`) string matching between the project's requirements and each member's declared `specification_json` requirements — deliberately no fuzzy/semantic matching. A requirement is "satisfied" only if every covering member also has `verification_passed is True`; helpers `_normalize`, `_member_requirements`, and `_member_passed` implement these primitives.

`project_verifier.py` (`ProjectVerifyState`, `_build_project_verify_graph`) builds a LangGraph graph that fans out via `Send` to one `phase_verifier` node per roadmap phase (`_phase_verify_router`), running an integration-verification agent (`build_project_verify_agent`) per phase to check cross-member integration gaps beyond the aggregator's string coverage, then fans in through `_synthesize_verify_node` (rolls per-phase verdicts VERIFIED/PARTIAL/FAILED into an overall verdict) and `_save_verify_result_node`, which persists `.spine/project/{id}/project_verification.json` via `ProjectStore`.

`project_reviewer.py` (`ProjectReviewState`, `_build_project_review_graph`) runs a single-pass adversarial red-team review over the entire completed project (`START → review_directive → adversarial_agent → save_review_result → END`), producing `.spine/project/{id}/project_review.json`. `_adversarial_agent_node` builds the review agent via `build_project_review_agent`, invokes it with `ainvoke_with_retry`, and reads the agent-written result file back from disk, falling back to a synthetic `NEEDS_REVIEW` document if the agent failed to write one.

The public entry points `run_project_verify` and `run_project_review` are called from `spine/cli/__init__.py` and `spine/ui_api/api.py`, confirming `spine.project` is invoked from both the CLI and the UI API layer. Both entry points, plus `aggregate_project_coverage`, also depend on `spine.persistence` (`ProjectStore`, `CheckpointStore`) for reading specs/state and persisting results, and on `spine.agents` (`build_project_review_agent`, `build_project_verify_agent`, `run_plan_node`, `ainvoke_with_retry`, `SpineContext`) to actually run the underlying LLM agents.

### aggregate_project_coverage (`spine/project/aggregator.py`)

Async function computing deterministic per-requirement and per-phase coverage for a project, returning `project_id`, `total_members`, `members_with_state`, `verified_members`, a `requirements` list (each with `status`/`covering`/`verified`), a `summary` count dict, and a `phases` rollup (`complete`/`in_progress`/`pending`).

### run_project_verify (`spine/project/project_verifier.py`)

Entry point that loads the `ProjectSpec` and trimmed member checkpoint state, computes aggregator coverage, runs the per-phase verifier graph, and returns the persisted `project_verification.json` document (or an error dict if the project isn't found or the graph invocation fails).

### run_project_review (`spine/project/project_reviewer.py`)

Entry point that loads the spec, aggregator coverage, member states, and any prior `project_verification.json`, runs the adversarial review graph, and returns the persisted `project_review.json` document.

**Notes:**
- The fragment reported "17 symbols (2 classes, 15 functions)" in one place and "13 functions" counts differ slightly between fragments for this module; the higher/directly-counted figure was treated as authoritative.
- `_adversarial_agent_node`, `_build_project_review_graph`, and `_build_project_verify_graph` had empty `summary` fields in the fragment — descriptions above come from reading `project_reviewer.py` and `project_verifier.py` directly, not the fragment text.
- `edges` in the fragment name `spine.agents` and `spine.persistence` as dependencies; both are corroborated by the actual imports in the three files.

## Module: spine.ui

`spine.ui` is the Streamlit-based operator console for SPINE. Its entry point is `spine/ui/app.py`, which configures the page (`st.set_page_config`), instantiates a single `UIApi` in `st.session_state`, starts the WebSocket push server (`start_ws_server()`), runs one-time cleanup (`api.reconcile_orphaned_entries_once()`), ensures the background worker loop is alive (`api.ensure_worker_running()`), and wires up Streamlit's `st.navigation`/`st.Page` multipage routing to eleven page-render functions (`_dashboard`, `_submit_work`, `_queue`, `_work_detail`, `_work_history`, `_config`, `_audit_log`, `_human_review`, `_spec_planning_render`, `_onboarding`, `_projects`, `_experience`). Each page module lives under `spine/ui/_pages/` (e.g. `dashboard.py`, `work_detail.py`, `queue.py`, `config_view.py`) and is imported lazily inside its `_*` wrapper function, then rendered by calling `render(api)`.

Real-time updates flow through `spine/ui/ws_bus.py`: the `Event` dataclass (`to_json`) and the `WSEventBus` class provide an async, thread-safe pub/sub bus (`publish`, `publish_sync`, `subscribe`, `unsubscribe`, `subscriber_count`), with a `get_bus()` singleton and pending-event buffering for events published before an event loop is bound (`set_loop`, `_flush_pending`). `spine/ui/ws_server.py` runs this bus behind an actual WebSocket endpoint (`run_server`, `_client_handler`, default port 8765 via `DEFAULT_WS_PORT`/`SPINE_WS_PORT`), started idempotently in a daemon thread by `start_ws_server()` (guards against double-binding via `_port_in_use`). `spine/ui/ws_component.py` embeds a tiny iframe (`render_ws_client`) that shows a green/red connection dot in the sidebar — Streamlit pages themselves poll for data using `@st.fragment(run_every=...)` rather than relying on WS-triggered reruns. `spine/ui/pages.py` holds a small page registry (`register`/`get`/`all_pages`) so pages can call `st.switch_page` on each other, and `spine/ui/utils.py` supplies shared formatting helpers (`status_icon`, `format_timestamp`, `format_duration`, `normalize_artifacts`, `create_work_link`, `navigate_to_work`).

### app.py (`spine/ui/app.py`)

Top-level Streamlit script and page registry; the only module that touches `UIApi()` construction directly and boots the WS server and worker loop.

### ws_bus.py / ws_server.py (`spine/ui/ws_bus.py`, `spine/ui/ws_server.py`)

The push-notification layer: `WSEventBus`/`Event` model and buffer state changes; `run_server`/`start_ws_server`/`_client_handler` expose them over a raw WebSocket that Streamlit's browser-side iframe (`ws_component.py`) connects to.

### _pages/ (`spine/ui/_pages/*.py`)

Thirteen page modules (dashboard, work_submit, queue, work_detail, work_history, config_view, audit_log, human_review, spec_planning, onboarding, projects, experience, plus `__init__.py`) — each takes the shared `UIApi` instance and renders one screen; `work_detail.py`'s `_execution_duration` is an example helper that reads `api.get_audit_log` and calls `format_duration`.

**Notes:**
- The supplied fragment's dependency edge lists `spine.ui -> spine.workflow`; the code read directly shows `spine.ui` importing `spine.ui_api` (the `UIApi` facade) directly, and it is `UIApi` (not `spine.ui`) that reaches into `spine.work.dispatcher`/`spine.work.ralph_worker`. A direct `spine.ui -> spine.workflow` coupling could still exist inside individual `_pages/*.py` files not fully read line-by-line (e.g. `spec_planning.py`, `onboarding.py`).
- `app.py`'s docstring and comments explicitly forbid UI pages from importing workflow/phases/dispatcher code directly — `UIApi` is meant to be the sole seam.

## Module: spine.ui_api

`spine.ui_api` is a single-file, single-class package: `spine/ui_api/api.py` defines `UIApi`, exported via `spine/ui_api/__init__.py`. Its own docstring states the contract plainly: "UI pages MUST use UIApi for all data access. Never import directly from workflow/, phases/, or work.dispatcher. This maintains the zero-duplication principle: CLI and UI share the same backend code paths." `UIApi.__init__` builds its dependencies once — `SpineConfig.load()`, `ArtifactStore(base_path=config.artifact_path)`, `AuditService` (pointed at `audit.db` next to the queue path), and a `ProjectStore` — and every method is a thin, mostly-synchronous wrapper around lower-level modules: `spine.work.dispatcher` (`get_work_status`, `list_work`, `submit_work`), `spine.work.ralph_worker` (`get_worker`), `spine.project.aggregator` (`aggregate_project_coverage`), and `spine.persistence.project_store`/`artifacts`.

The ~70 methods fall into clear groups: project CRUD (`get_project`, `list_projects`, `create_project`, `update_project`, `delete_project`, `get_project_members`, `get_project_coverage`, `run_project_verify`/`run_project_review`); work-item access (`get_work`, `list_work`, `enqueue_work`, `enqueue_onboarding`, async `submit_work` for blocking CLI-parity calls); artifacts and audit (`get_artifacts`, `read_artifact`, `read_onboarding_doc`, `get_feedback`, `get_critic_review`, `get_audit_log`); configuration and providers, which read/write `.spine/config.yaml` directly via `yaml.safe_load`/`yaml.dump` (`get_config`, `update_mcp_server`, `remove_mcp_server`, `_save_config`, `add_llm_provider`, `update_llm_provider`, `remove_llm_provider`, `set_phase_provider`); MCP connectivity testing using `MultiServerMCPClient` (`test_mcp_connection`); and queue/worker orchestration (`get_worker_status`, `ensure_worker_running`, `get_queue_overview`, `_active_jobs`/`_build_active_job` which merge the worker's `queue` table rows with `work_entries` so restarted/resumed off-queue jobs still show as active, `resume_work`, `restart_work`, `restart_from_phase`, `stop_work`, `reset_stuck_items`, `reconcile_orphaned_entries`/`reconcile_orphaned_entries_once`) plus planning-session helpers (`list_planning_sessions`, `get_planning_detail`, `approve_plan`).

### UIApi (`spine/ui_api/api.py`)

The sole facade class; roughly 1550 lines, one `__init__` and ~69 public/private methods, no other classes or module-level functions in the package.

### _artifact_store_for (`spine/ui_api/api.py`)

Resolves which `ArtifactStore` base path applies to a given work item — onboarding jobs targeting an external repo write artifacts under that repo's own `.spine/artifacts`, recorded as `workspace_root` in the work entry's result payload, rather than under SPINE's own configured `artifact_path`.

**Notes:**
- Confirmed dependency edges from the fragment (`spine.agents`, `spine.models`, `spine.persistence`, `spine.project`, `spine.work`, `spine.workflow`) are broader than what was directly observed in the code actually read (`spine.config`, `spine.models.enums`, `spine.persistence.artifacts`, `spine.services.audit_service`, `spine.work.dispatcher`, `spine.work.ralph_worker`, plus lazy imports of `spine.persistence.project_store` and `spine.project.aggregator`). Not every one of the ~1551 lines was read, so imports of `spine.agents` or direct `spine.workflow` usage likely exist further down (e.g. around `resume_work`/`restart_from_phase`/`get_restart_phases`) but were not personally verified.
- Config writes (`update_mcp_server`, `remove_mcp_server`, `_save_config`, provider CRUD) hardcode the path `.spine/config.yaml` relative to the process's working directory rather than deriving it from `self._config`.

## Module: spine.work

`spine.work` (182 symbols: 18 classes/interfaces, 116 functions, 48 methods) is Spine's execution and dispatch layer, plus the repo-onboarding analysis-and-doc-synthesis subsystem it hosts under `spine/work/onboarding/`. The single entry point is `submit_work` in `spine/work/dispatcher.py` — the CLI (`spine run`), the Streamlit UI submit page, and the background queue worker all route through it. `submit_work` inserts a row into the SQLite `work_entries` table (`.spine/work_entries.db`), builds the LangGraph workflow graph for the given `work_type` via `spine.workflow.compose.build_workflow_graph`, streams phase-by-phase updates back into that row, and persists artifacts through `spine.persistence.artifacts.ArtifactStore`. `dispatcher.py` also exposes `resume_work`, `resume_interrupted_work`, and `restart_work`, which rebuild the same graph from a saved checkpoint (`spine.persistence.checkpoint.CheckpointStore`) to continue a paused or `needs_review` run.

One `work_type` — `"onboarding"` — bypasses the LangGraph phase machinery entirely: `submit_work` special-cases it and calls `spine.work.onboarding.engine.run_onboarding`, which analyses (or scaffolds, for greenfield) a target repo and synthesises four markdown documents. This is itself a separate `StateGraph` (`onboarding_graph.build_onboarding_graph`), deliberately unregistered in the main `WORKFLOW_SEQUENCES` so it needs no critic/rework/human-review wiring.

For code-producing `work_type`s (any whose phase sequence includes IMPLEMENT, per `spine.git.sandbox.work_type_writes_code`), `submit_work` wraps the run in a `spine.git.WorktreeSandbox`. `dispatcher.py` and `ralph_worker.py` depend on `spine.persistence`, `spine.models.enums`, `spine.ui.ws_bus` (WebSocket progress push), and `spine.workflow.compose`.

### submit_work (`spine/work/dispatcher.py`)

Unified entry point taking a description/work_type; creates the work entry, enters the worktree sandbox, streams the compiled workflow graph (`stream_mode=["updates","messages"]`), derives a terminal status via `_derive_final_status`, then merges or rolls back the sandbox accordingly.

### RalphLoopWorker (`spine/work/ralph_worker.py`)

Threading-based singleton (`get_worker()`) owning its own SQLite `queue` table (distinct from `work_entries`, keyed by `config.queue_path`). `dequeue()` atomically claims the next pending row via a single `UPDATE … WHERE status='pending' … RETURNING`; its daemon-thread `_loop()` calls `submit_work` per item.

### RepoAnalyzer (`spine/work/onboarding/analyzer.py`)

Compiles a `RepoManifest` from a workspace using only semantic signals — file discovery via `codebase_query.list_files` and AST byte-slicing via `extract_symbols` — never raw line reads. Each `ModuleBoundary` records at most 8 ranked "key symbols" (classes/interfaces first) as *representative* examples, not the module's full symbol list; `spine.work` itself has 182 symbols even though any one boundary's `key_symbols` array is capped at 8.

### Two-tier synthesis hierarchy (`spine/work/onboarding/synthesis_nodes.py`)

A `doc_manager` (Tier A) plans document sections from a compact `manifest_index`; a `section_worker` (Tier B) fills exactly one bounded fragment per LLM call via `resolve_fragment_from_dict`; a deterministic `assemble_docs`/`aggregate_synthesis` (Tier C) concatenates and writes the four `.md` files. No single LLM call ever receives the whole manifest.

**Notes:**
- `RepoManifest`'s per-module `key_symbols` cap (8) is a real, code-verified limit on how many *example* symbols each module boundary carries in the manifest — it is not evidence that a module itself only has 8 symbols.
- The onboarding graph is intentionally excluded from `spine.workflow`'s `WORKFLOW_SEQUENCES`/subgraph registry; it is a parallel execution path, not a phase of the main workflow graph.

## Module: spine.workflow

`spine.workflow` (spine/workflow/) is the orchestration layer for SPINE's LangGraph agent pipeline. Its job is to turn a `WorkType` into a compiled `StateGraph`: `compose.py`'s `build_workflow_graph` selects a phase sequence (e.g. `task`: SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY; `critical_task` additionally inserts ADVERSARIAL_PLAN between critic and implement), wires each phase in as a node, and adds conditional edges for critic/adversarial rework loops and a human-review interrupt. `subgraph_state.py` defines the isolated state schema for every phase — `BaseSubgraphState` holds fields shared by all phases (`phase`, `artifacts_output`, `phase_status`, a deduplicated `read_cache`), and per-phase TypedDicts (`SpecifySubgraphState`, `PlanSubgraphState`, `ImplementSubgraphState`, `VerifySubgraphState`, `CriticSubgraphState`, `AdversarialSubgraphState`, `GapPlanSubgraphState`, `ExplorationSubgraphState`) extend it with phase-specific channels, some using custom reducers (`_slice_list_reducer`, `_bool_or`) so parallel `Send` branches can update shared lists/flags without LangGraph's `InvalidUpdateError`. `registry.py` provides `PhaseRegistry`/`PhaseDefinition`, a name→node-function lookup that `spine.phases` modules populate via `register()` at import time.

Each phase actually runs as one of the builders in `spine/workflow/subgraphs/` (e.g. `plan_subgraph.build_plan_subgraph`, `implement_subgraph.build_implement_subgraph`), wrapped into a parent-graph node by `subgraph_wrapper.make_subgraph_node`, which maps `WorkflowState` into the subgraph's state, invokes it with its own checkpointer/timeout, and maps the result back. `compose.py`'s `_SUBGRAPH_ENABLED` dict shows VERIFY, IMPLEMENT, TASKS, SPECIFY, PLAN, CRITIC, and GAP_PLAN are all currently flagged `True`, so the `registry.PhaseRegistry` call_fn path is a dormant fallback today, exercised only when a flag is toggled off. Supporting modules implement quality gates and scheduling used by those subgraphs: `artifact_gate.py` (structural pre-check before tasks→implement), `critic_review.py`/`critic_convergence.py`/`plan_reference_gate.py`/`plan_score.py` (critic-tier checks, rework-repeat detection, a deterministic dangling-reference gate, and best-of-N plan scoring), `slice_scheduler.py` (`compute_execution_waves`, topological wave grouping for parallel IMPLEMENT/VERIFY dispatch), `verify_snapshot.py` (best-state ratchet across gap-fix cycles), `phase_progress.py` (work-entry progress tracking), `export.py` (markdown export of a work item for external review) and `studio.py` (LangSmith Studio graph entry points). `workers/vector_indexer.py` is a background job that chunks the codebase and embeds it for vector search.

### PhaseRegistry / PhaseDefinition (`spine/workflow/registry.py`)

`PhaseRegistry` stores `PhaseDefinition`s (name, `call_fn`, `build_agent_fn`, optional `subgraph_node_fn`, description) keyed by phase name; `get_registry()` is a module-level singleton that, on first call, imports all seven `spine.phases.*` modules so they self-register.

### Subgraph state schemas (`spine/workflow/subgraph_state.py`)

`BaseSubgraphState` plus eight per-phase TypedDicts isolate each phase's working state (and DA message history) from the parent `WorkflowState`; `ImplementSubgraphState` and `VerifySubgraphState` carry the most machinery (dispatch counters, slice lists with custom reducers) to support Send-API fan-out/fan-in.

### build_workflow_graph (`spine/workflow/compose.py`)

Builds the per-`WorkType` `StateGraph`, choosing per-phase between a subgraph node (default) and the legacy registry `call_fn` node depending on `_SUBGRAPH_ENABLED`.

**Notes:**
- Only 8 of 287 symbols were supplied as examples in the source fragment; this description covers module responsibilities and the files actually read, not an exhaustive symbol inventory.
- The dependency relationship with `spine.phases` is bidirectional in practice: `registry.py` imports `spine.phases.*` to trigger registration, while every `spine.phases` module imports `spine.workflow.registry.get_registry()` to register itself, and `spine.phases.critic` also imports `spine.workflow.critic_review`.

## Module: tests.conftest.py

`tests/conftest.py` is the shared pytest configuration for the whole suite — every test module under `tests/` picks up its fixtures automatically. Its job is test-suite hygiene: keep the suite from corrupting the real repository or leaking real API traffic, and give every other test module a consistent set of `SpineConfig`-backed fixtures instead of ad hoc doubles.

Three autouse fixtures do the protective work. `_guard_repo_git_head` snapshots the real repo's symbolic HEAD before each test (via `_repo_head`, a `git symbolic-ref` call) and, if a test flipped it (e.g. by driving `SpineGitOrchestrator` without overriding `master_dir`), re-points HEAD and fails that test by name. `_no_langsmith_tracing` deletes `LANGSMITH_API_KEY`, `LANGCHAIN_API_KEY`, and `SPINE_TRACE_ALL` from the environment so `work_run_tracing()` can't opt dispatcher/onboarding tests into real LangSmith tracing. `_preserve_lazy_imported_modules` snapshots and restores `sys.modules` entries for `spine.workflow.compose` and `spine.persistence.checkpoint`, which some dispatcher/resume tests mock by assigning-then-popping — without this, a later test that imports one of those modules at load time sees it missing.

The rest are ordinary fixtures: `temp_dir`/`test_config`/`async_test_config` build a `SpineConfig` pointed at a tmp directory; `sample_task`/`sample_artifact`/`sample_review_feedback`/`sample_prompt_request` construct real `spine.models.types` objects (`Task`, `Artifact`, `ReviewFeedback`, `PromptRequest`); `mock_openai_response`, `sample_work_description`, and `sample_work_config` supply canned data; `event_loop` hands back a fresh asyncio loop for async tests.

### _guard_repo_git_head (`tests/conftest.py`)

Autouse fixture that fails any test which leaves the real repo's git HEAD pointed somewhere different than it started, and repairs HEAD via `symbolic-ref` so the developer's checkout isn't left on the wrong branch.

**Notes:**
- Depends on `spine.config.SpineConfig` and `spine.models.types` (`Task`, `Artifact`, `ReviewFeedback`, `PromptRequest`) — every other test module depends on this file implicitly through pytest's conftest auto-discovery, not via import.

## Module: tests.fixtures

`tests/fixtures/` holds four small, static, non-Python source files — `sample.c`, `sample.cpp`, `sample.php`, `sample.ts` — used as input data for tests of the multi-language symbol-extraction layer (AST parsing / codebase indexing). These are not test files themselves; they're fixtures other tests read and parse.

Each file defines a small, deliberately parallel set of constructs: `sample.c` has a `Point` struct, a `Value` union, a `Color` enum, a free `add()` function, and a `static` `format_point()` helper. `sample.cpp` wraps similar shapes in a `geo` namespace: `enum class Color`, `struct Point`, and a `Greeter` class whose `shout()` method is defined out-of-line, plus free functions `dot()`, `pick()`, and `make_counter()`. `sample.php` defines a top-level `greet()` function, a `Greeter` class with a constructor and `say()` method, and a `Speaker` interface. `sample.ts` mirrors that shape with an exported `greet()` function, a `shout` arrow-function constant, an exported `Greeter` class, and a `Speaker` interface.

### sample.cpp (`tests/fixtures/sample.cpp`)

C++ fixture: namespace `geo` containing `enum class Color`, `struct Point`, and class `Greeter` (constructor, `greet()` inline, `shout()` defined out-of-line), plus free functions `dot`, `pick`, `make_counter`.

**Notes:**
- All four fixtures repeat a `Greeter`/`Speaker`/`Color` naming pattern across languages, which is what lets extraction tests assert the same claim (e.g. "class found", "interface found") across every supported language from one shared vocabulary.

## Module: tests.integration

`tests/integration/` contains end-to-end tests that exercise multiple SPINE phases or subsystems together, mostly without a live LLM. Six files: `test_structured_io_flow.py` (the largest, ~31KB) validates the JSON artifact contracts that flow through SPECIFY → PLAN → IMPLEMENT → VERIFY — `Specification`, `FeatureSlice`, `StructuredPlan`, `GapPlan`, `FixInstruction`, `CriticReview` schemas, `ArtifactStore` round-trips, the full artifact chain, and workflow graph topology via `spine.workflow.compose.build_workflow_graph`/`_plan_state_mapper`/`_verify_state_mapper`; it degrades gracefully (stubs `langgraph.checkpoint.sqlite` or skips) when optional dependencies are missing. `test_plan_to_implement_flow.py` builds a 3-slice `plan.json` (models, config, api — api depends on both others) and asserts `spine.workflow.slice_scheduler.compute_execution_waves` produces the correct 2-wave dispatch order. `test_mcp_integration.py`'s `TestRealMCPCodebaseIndex` spins up the actual `mcp-codebase-index` server over stdio via `spine.mcp.client.get_mcp_tools` against a temp sample project and asserts all 18 expected tools are present and namespaced (`mcp_test-idx_*`). `test_git_lifecycle.py` drives `SpineGitOrchestrator` against a real throwaway git repo in `tmp_path` (`prepare_sandbox` → `run_validation_pipeline` → `commit_and_merge`), skipped if `git` isn't on `PATH`. `test_prompt_assembly.py` confirms the `deepagents` harness profile composes `phase_prompt + SPINE_BASE_PROMPT`, not the upstream conversational default. `test_exploration_subgraph.py` checks `build_exploration_subgraph` compiles for SPECIFY and PLAN phases, rejects unknown phases, and that `ExplorationSubgraphState` fields/reducers behave as expected.

### test_plan_to_implement_flow.py (`tests/integration/test_plan_to_implement_flow.py`)

Builds a plan with dependency topology `models, config → api` and asserts `compute_execution_waves` schedules `models`/`config` in wave 0 and `api` in wave 1 — the scheduler's core dependency-respecting contract.

**Notes:**
- Depends on `spine.agents`, `spine.mcp`, `spine.models`, `spine.workflow` per the module graph; confirmed directly by imports of `spine.workflow.compose`, `spine.workflow.slice_scheduler`, `spine.mcp.client`, `spine.agents.profile`, and `spine.workflow.subgraphs.exploration_subgraph`.

## Module: tests.recall_eval

`tests/recall_eval/` is not a pytest suite — its own README calls it "a live script, not a CI test." It's a standalone eval harness for the retrieval quality of `RecallTool`/`VectorStore`, meant to be run manually before/after a change to indexing, embedding, or reranking.

`run_eval.py` loads a golden set of `(query -> expected files/symbols)` pairs from `golden.jsonl` (`_load_golden`), runs each through the real `RecallTool._arun` concurrently under an `asyncio.Semaphore` (`_run_one`), and scores results with `_first_hit_rank` and `_set_recall_at`. `_aggregate` rolls per-query results into `hit@5`/`hit@10`/`hit@20`, `mrr`, `miss@{k}` (the "recall ceiling" — misses no reranker downstream can fix), and `set_recall@20`; `_by_source` breaks the same metrics down by `curated` vs. `mined` query source. `_print_compare` diffs a run's metrics against a previously saved baseline JSON. `build_golden.py` is the dev tool that grows the golden set: it mines `.spine/artifacts/*/*/research_log.json` (gitignored, local-only) for `ResearchFinding` `topic`/`file_map` pairs, cleans the topic via `_clean_topic`, keeps only concrete indexable production files via `_concrete_prod_files` (capped at 6 files/topic), and writes candidates to `golden_mined.jsonl` for manual review — it deliberately never overwrites the committed `golden.jsonl`. The directory also holds several saved report snapshots (`baseline.json`, `nomic_final.json`, `phase2_final.json`, `rerank_on.json`) — comparison points captured from different embedding/reranking configurations over time.

### run_eval.py (`tests/recall_eval/run_eval.py`)

Async CLI harness: `python tests/recall_eval/run_eval.py [--retrieve-k N] [--label L] [--out path] [--compare baseline.json]`; requires a populated vector store (`spine index`) and a reachable embedding endpoint — it is not runnable offline.

**Notes:**
- Depends on `spine.agents` (`RecallTool`) and `spine.config` (`SpineConfig.load` supplies the default db path when `--db` isn't given).

## Module: tests.test_core.py

`tests/test_core.py` is a small, single-purpose test file: `TestFormatDuration` exercises `format_duration` from `src.utils.helpers` (note the `src/` import path — this helper lives outside the `spine` package). It covers seconds-only, minutes-and-seconds, hours-only, hours-and-minutes, and day-scale formatting; the zero and negative-duration edge cases (negative clamps to `"0s"`); float-value truncation; a very-large (30-day) duration; and exact minute/hour/day boundaries. There are no fixtures beyond a `sys.path` insertion to make the repo root importable.

### TestFormatDuration (`tests/test_core.py`)

Boundary-value test class for `format_duration`, e.g. `format_duration(90061) == "1d 1h 1m 1s"`, `format_duration(-5) == "0s"`, `format_duration(45.9) == "45s"`.

**Notes:**
- Imports `format_duration` from `src.utils.helpers`, not from the `spine` package — distinct import root from most of the rest of the suite.

## Module: tests.test_plan_score.py

`tests/test_plan_score.py` tests `spine.workflow.plan_score.score_plan`, the function that scores candidate plans for the PLAN phase's best-of-N selection. A `_slice()` helper builds minimal feature-slice dicts (`id`, `title`, `target_files`, `dependencies`).

The tests establish the scoring contract: a clean 3-slice plan (with `s2` depending on `s1`) scores `total == 1.0` and is schedulable; a 2-slice dependency cycle (`s1↔s2`) is unschedulable and scores `0.0`; an empty `feature_slices` list scores `0.0`; slices with no `target_files` are schedulable but penalised (`target_files_score == 0.0`, `total < 0.5`); 250 duplicate target files in one slice crush `dup_penalty_score` below `0.05`; and `test_coverage_neutral_when_spec_names_no_files` checks that `coverage_score` stays neutral (`1.0`) when the spec prose doesn't name any file paths at all. The single most important test, `test_clean_plan_outranks_every_pathology`, asserts the clean plan's score beats each pathological plan (empty target files, duplicate explosion, dependency cycle) — the selector's core contract.

### test_clean_plan_outranks_every_pathology (`tests/test_plan_score.py`)

Asserts `score_plan` ranks a well-formed, schedulable, dependency-respecting plan above three specific pathological plans (empty target files, 250x duplicate target file, and a 2-slice dependency cycle).

**Notes:**
- Only imports `spine.workflow.plan_score.score_plan`; no mocking or fixtures beyond the local `_slice()` helper.

## Module: tests.test_restart.py

`tests/test_restart.py` covers `restart_work()`, `reset_stuck_items()`, and the UI-facing wrappers around both. An autouse `_reset_worker_singleton` fixture resets `spine.work.ralph_worker._WORKER_INSTANCE` before and after each test so tests don't share a worker bound to another test's config. `tmp_config` builds a `SpineConfig` backed by a freshly `git init`'d tmp repo (so the code-producing restart path's worktree preflight passes); `work_db`/`queue_db` create isolated `sqlite_utils` tables.

`TestRestartWork` checks: completed and failed work items raise `ValueError` on restart; running, stalled, needs_review, and cancelled items are all accepted and re-run the workflow graph (via a mocked `spine.work.dispatcher._run_workflow_graph`); restarting a nonexistent id raises `ValueError`; and — the two artifact-handling tests — `clear_artifacts=False` (the default) leaves on-disk artifact files in place while `clear_artifacts=True` removes them before the graph re-runs. `TestResetStuckItems` verifies `RalphLoopWorker.reset_stuck_items()` moves `running` queue rows back to `pending` (checked for 0/1/3-row cases) while leaving already-`pending` rows untouched. `TestUIApiRestart`/`TestUIApiResetStuck` check that `UIApi.restart_work()` returns immediately with `status="running"`/`action="restart"` (fire-and-forget) and that `UIApi.reset_stuck_items()` delegates to the worker. `TestStatusColorCSS` checks `spine.ui.utils.status_color_css` maps each known status to a specific CSS color name and falls back to `"white"` for an unknown status.

### restart_work (`spine/work/dispatcher.py`, tested from `tests/test_restart.py`)

Validates a work item's current status (rejecting `completed`/`failed`), then re-invokes the workflow graph; `clear_artifacts` controls whether on-disk artifact files from a prior run are wiped first.

**Notes:**
- Depends on `spine.ui`, `spine.ui_api`, `spine.work` per the module graph — confirmed by direct imports of `spine.work.dispatcher`, `spine.work.ralph_worker.RalphLoopWorker`, `spine.ui_api.UIApi`, and `spine.ui.utils.status_color_css`.

## Module: tests.test_work_ordering.py

`tests/test_work_ordering.py` tests the ordering guarantees of `spine.work.dispatcher.list_work()` against a fresh sqlite `work_entries` table (built via the module's own `_get_work_db`). `TestListWorkOrdering` provides `_make_config` (isolated `SpineConfig`/queue path) and `_insert_entries` helpers, then asserts: default calls return items strictly newest-`created_at`-first; a `status="completed"` filter preserves that same newest-first order; `limit=N` returns only the N newest; rows with identical `created_at` break ties by rowid, with the higher (more recently inserted) rowid sorting first; rows with `created_at IS NULL` sort to the very end, after all dated rows; and an empty table returns `[]`.

### test_list_work_null_created_at (`tests/test_work_ordering.py`)

Inserts one row with `created_at=None` alongside two dated rows and asserts the NULL row lands last — `list_work` must not let a missing timestamp float to the top or crash the ordering.

**Notes:**
- No `edges` were reported in the fragment for this module, but the file directly imports `spine.config.SpineConfig` and `spine.work.dispatcher` (`_get_work_db`, `list_work`).

## Module: tests.unit

`tests/unit/` is by far the largest part of the test suite — 135 test files, one per unit of behavior, covering nearly every subsystem under `spine/`. Sampled files show it spans: the onboarding-doc engine and its graph/manifest/scaffold/synthesis pipeline (`test_onboarding_engine.py`, `test_onboarding_graph.py`, `test_onboarding_manifest.py`, `test_onboarding_synthesis_plan.py`, `test_onboarding_synthesis_tools.py`, and others); the PLAN/critic/verify/implement subgraphs and their gates and guards (`test_critic_convergence.py`, `test_critic_escalation.py`, `test_verify_subgraph.py`, `test_implement_dispatch_gating.py`, `test_plan_reference_gate.py`); the exploration/research loop (`test_explore_supervisor_loop.py`, `test_research_manager_rework.py`, `test_researcher_supervisor.py`); the dispatcher and Ralph worker (`test_dispatcher.py`, `test_dispatcher_streaming.py`, `test_ralph_worker.py`); git orchestration (`test_git_orchestrator.py`); MCP tooling (`test_mcp_client.py`, `test_mcp_tools.py`, `test_codebase_query_tool.py`); ephemeral GPU pod lifecycle (`test_ephemeral_pod.py`); token/context budgeting (`test_token_budget.py`, `test_token_compactor.py`, `test_context_engineering.py`); the read/edit/lint tool loop (`test_read_edit_lint.py` and its skeleton/cache/write-breaker variants); AST/symbol extraction (`test_ast_extract.py`, `test_symbol_cache.py`); and the UI/API surface (`test_ui_api.py`, `test_queue_status.py`).

A pattern recurring across the sampled files: heavy use of small `Fake*` test doubles in place of real LLM or network calls — `FakeConfig`, `FakeGraph`/`FakeDB`/`FakeCkptStore`, `FakeChatOpenAI`, `FakeModel`. `test_slice_scheduler.py` shows a second pattern: loading a target module directly via `importlib.util.spec_from_file_location` to sidestep `spine/workflow/__init__.py`'s heavier imports (LangGraph, deep agents) — described in-file as "the project convention."

### test_ephemeral_pod.py (`tests/unit/test_ephemeral_pod.py`)

Exercises `spine.infra.ephemeral_pod` — config parsing, the reference-counted `acquire`/`release` singleton, failure policies, the `with_ephemeral_pod` decorator, and `env:`-prefixed provider-value expansion — entirely without the optional `runpod` SDK or a real cloud call, by monkeypatching the boot step.

**Notes:**
- Depends on `spine.agents`, `spine.critic`, `spine.git`, `spine.mcp`, `spine.models`, `spine.persistence`, `spine.phases`, `spine.project`, `spine.ui_api`, `spine.work`, `spine.workflow` — i.e., effectively every top-level `spine` subpackage, consistent with this being the catch-all unit-test package for the whole engine.

## Module: spine.services

`spine/services/` currently contains one concern: audit logging. `spine/services/audit_service.py` defines `AuditService`, a small wrapper around a SQLite database (via `sqlite_utils.Database`, tuned through `spine.persistence.sqlite_tuning.tune_connection`) that records workflow events. `__init__(db_path=".spine/audit.db")` creates the parent directory and calls `_ensure_table()`, which creates an `audit_events` table (`id`, `work_id`, `event_type`, `phase`, `details`, `timestamp`) if absent. `_get_db()` opens a fresh tuned connection on each call (documented as "safe across threads" — no connection is held as instance state). `log_event(work_id, event_type, phase, details=None)` inserts a row with `details` JSON-serialized and an ISO-8601 timestamp. `query_events(work_id=None, event_type=None, limit=100)` builds an optional `WHERE work_id = ? AND event_type = ?` filter, orders by `timestamp DESC`, and JSON-decodes `details` back into a dict for each returned row.

The package's `__init__.py` re-exports `AuditService` as the sole public symbol (`__all__ = ["AuditService"]`), so this module is essentially a single class with no other moving parts — no functions or other classes exist in the package today.

### AuditService (`spine/services/audit_service.py`)

The only class in the module: opens a fresh SQLite connection per operation, ensures its own schema on construction, and exposes `log_event`/`query_events` as its entire public API.

**Notes:**
- No dependency edges were supplied in the fragment for this module, and the only real import outside the standard library / `sqlite_utils` is `spine.persistence.sqlite_tuning`.
- Nothing in the read source indicates other modules import `AuditService` — this is unconfirmed either way, not evidence of no callers.

## Module: spine.adversarial

`spine.adversarial` runs a full red-team review of an already-critic-approved `plan.json` against `specification.json`, for critical work types. `build_adversarial_agent` (`spine/adversarial/agent.py`) constructs a no-tool Deep Agent (`allowed_tools=[]`, `skip_filesystem_middleware=True`) whose system prompt instructs it to assume the plan is flawed and hunt for failure modes, hidden assumptions, coverage gaps, and unsafe sequencing, classifying each finding as autonomously-fixable (→ `NEEDS_REVISION`, looped back to PLAN) or human-judgement-required (→ `NEEDS_REVIEW`). It reuses the `CriticReview` response model so the critic's parser and escalation guards apply unchanged.

`agent_adversarial_check` and `run_once` (`spine/adversarial/review.py`) drive execution: they materialize artifacts, build the review prompt via `spine.workflow.critic_review`'s shared `_build_review_prompt` (imported lazily to dodge a circular import through package init), invoke the agent with retry, then run it through `_validate_scope_claim` and `_corroborate_terminal_verdict` before tagging the verdict `tier="adversarial"`.

**Notes:**
- Depends on `spine.agents` (`build_phase_agent`, `ainvoke_with_retry`, `build_context`, `materialize_artifacts`) and `spine.workflow` (`critic_review` internals, `registry`); raises `spine.exceptions.CriticalContractFailure` if the plan's structured state field is missing.
- On any non-contract exception it fails safe to `NEEDS_REVISION` rather than crashing.

## Module: alembic.env.py

`alembic/env.py` is the standard Alembic migration environment script, not an importable library module. `run_migrations_offline` configures the migration context with a literal `sqlalchemy.url` and runs migrations inside a transaction with no live DB connection (for generating SQL without a database). `run_migrations_online` builds a real engine via `engine_from_config` with `NullPool`, connects, and runs migrations against that live connection. `target_metadata` is bound to `spine.models.work_entry.Base.metadata`, so Alembic's autogenerate diffing is scoped to that one model. Mode selection (`context.is_offline_mode()`) happens at import time, at the bottom of the file.

**Notes:**
- Only real code dependency is `spine.models.work_entry.Base`; otherwise it only touches `alembic.context` and `sqlalchemy`.
- No inbound edges: it's a CLI entrypoint invoked by the Alembic tool itself, not imported by application code.

## Module: alembic.versions

The migrations directory currently holds exactly one revision: `c69967ea0727_create_work_entries_table.py`, with `down_revision = None` — this is the initial migration. `upgrade()` creates the `work_entries` table (`id`, `thread_id`, `timestamp`, `action`, `payload` columns, a unique constraint on `(thread_id, timestamp)`, and an index on `thread_id`). `downgrade()` reverses it (drop index, then drop table). The file is unmodified Alembic-autogenerated boilerplate ("please adjust!" comments still present).

**Notes:**
- This is a one-migration directory today; "alembic.versions" as a concept is currently just this single script.
- Consumed only via the Alembic CLI through `alembic/env.py`; no other module imports migration files directly.

## Module: spine.observability.py

`spine/observability.py` scopes LangSmith tracing to individual work-task runs instead of tracing the whole process. Per its docstring, tracing is disabled process-wide by default (`spine.config._disable_global_tracing`) specifically so codebase indexing, the test suite, and onboarding analysis stay untraced. `work_run_tracing(work_id, work_type)` is a contextmanager that no-ops when `SPINE_TRACE_ALL` is set or no LangSmith API key is present, and otherwise wraps the block in `langchain_core.tracers.context.tracing_v2_enabled` with `work_id`/`work_type` tags and a project name from `LANGSMITH_PROJECT`/`LANGCHAIN_PROJECT` (default `"spine"`). `traced_astream(stream, work_id, work_type)` wraps an async `graph.astream()` iterator with that same context manager for the life of iteration, including cancellation on stall-timeout.

**Notes:**
- Contextvar-scoped rather than `os.environ`-based, so it's async/concurrency-safe — a traced work run and an untraced concurrent indexing job in the same process don't cross-contaminate.
- No inbound/outbound edges were reported in the fragment; it's a narrow, self-contained utility.

## Module: tests.smoke

`tests/smoke/smoke_specify_recall.py` is a single manual diagnostic script (not a pytest-discovered test — no assertions, run via `if __name__ == "__main__"`) that exercises the SPECIFY phase's recall (vector-store) tool end-to-end. `_build_state()` builds a minimal `WorkflowState` for a hard-coded "add a --verbose flag" task. `main()` calls `RecallTool` standalone, injects the retrieved chunks into the prompt, builds the SPECIFY Deep Agent (`build_specify_agent`) with MCP tools and memory middleware patched out to fit the model's context budget, invokes it via `ainvoke_with_retry`, and inspects the resulting tool-call trace to confirm whether the model actually called `recall`.

**Notes:**
- Its own docstring states the full production tool surface (18 MCP tools + AGENTS.md) exceeds 200k tokens of context and "needs its own fix" — this script is a deliberately reduced repro, not a general smoke-test entrypoint.
- The `tests/smoke` directory has no `__init__.py`.

## Module: spine.critic

`spine/critic/agent.py` builds the Deep Agent for the CRITIC phase — the quality gate every phase output passes through. `build_critic_agent` assembles a system prompt requiring the reviewer to judge completeness, correctness, clarity, and actionability from the inlined JSON payload alone (`allowed_tools=[]`, `skip_filesystem_middleware=True`), while explicitly excluding schema conformance (owned by a separate deterministic validator) to avoid non-convergent rework. When the reviewed phase is PLAN, extra instructions are appended for structured `plan.json` validation (slice coherence, dependency-graph acyclicity, requirement coverage, granularity, proportionality/scope-creep); for SPECIFY, instructions require every spec requirement to trace back to the original user description. Output is constrained to the `CriticReview` response format, capped at `_CRITIC_COMPLETION_CAP = 12000` tokens — raised from an earlier 6000 after a well-behaved model legitimately truncated a multi-slice PLAN review at that ceiling.

**Notes:**
- Built on `spine.agents.factory.build_phase_agent`; uses `spine.models.enums.PhaseName`, `spine.models.types.CriticReview`, `spine.agents.prompt_snippets.SCOPE_EXCLUSION_CITATION_RULE`, and `spine.workflow.critic_review._get_reviewed_phase`.
- Only one real symbol lives here (`build_critic_agent`); prompt-building, parsing, and escalation-guard logic actually live in `spine.workflow.critic_review`, which `spine.adversarial.review` also reuses.

## Module: spine.log.py

`spine/log.py` is a single-function module: `configure_logging(verbose: bool = False)`. When `verbose` is true it sets the root logger to `DEBUG` with a timestamped/leveled format; otherwise `WARNING` with a terse format. It force-reconfigures by removing any existing handlers on the root logger before calling `logging.basicConfig(..., stream=sys.stdout, force=True)` — intended to be called once from a CLI entrypoint wiring up a `--verbose` flag.

**Notes:**
- No dependencies beyond the standard library (`logging`, `sys`); no edges reported in the fragment.
- This is genuinely a one-function module — there is no more to describe without padding.

## Module: __other_modules__

`__other_modules__` is the onboarding manifest's catch-all bucket for smaller or otherwise uncategorized files that didn't warrant an individually-documented module entry. The supplied fragment for it contains no modules and no edges, so there is nothing concrete to report here beyond its role as a tail bucket; it isn't a real module with its own responsibility, execution path, or dependencies.
