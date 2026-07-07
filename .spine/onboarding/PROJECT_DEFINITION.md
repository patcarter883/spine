# Project Definition

## Domain: spine.agents

`spine.agents` exists because SPINE runs coding work as a deterministic, unattended multi-agent pipeline rather than a single interactive chat session: something has to actually write the code, verify it, decompose the work when it's too large to land in one pass, and do all of this reliably across a dozen different LLM providers (Anthropic, OpenRouter-routed cloud models, and self-hosted OpenAI-compatible servers) — each of which fails in its own particular way. Without a dedicated agents domain, every phase (specify, plan, implement, verify) would reimplement its own model resolution, tool wiring, and failure handling; instead this domain is the single place that turns an abstract workflow phase into a concrete, correctly-scoped agent, and the single place that absorbs the model-specific failure modes so the rest of SPINE doesn't have to know about them.

### Phase execution, not orchestration

This domain does not decide when a phase runs, what happens after it finishes, or how retries are scheduled — that responsibility belongs to `spine.workflow` (subgraph routers, LangGraph `Send`-based dispatch). `spine.agents` only builds and configures the agent that runs once invoked, handing back either a compiled agent graph or a subagent spec; the boundary is intentionally one-directional (`spine.agents` depends on `spine.workflow`'s types, not the other way around).

### Reliability across heterogeneous model backends

A substantial share of this domain's code (`helpers.py`, `retry.py`, `tool_forcing.py`) exists purely to make identical phase logic behave correctly whether the underlying model is Anthropic, an OpenRouter-routed model, or a local vLLM/llama.cpp server — each rejects requests, drops credentials, or exhausts its output budget differently. This is what lets SPINE assign a different, cheaper model to each phase without every phase author re-solving provider quirks.

### Delegation boundary between orchestrator and worker agents

Phase agents for IMPLEMENT and VERIFY do not touch source code directly; they dispatch to leaf subagents (researcher, slice-implementer, slice-verifier) that are each scoped to one feature slice with a deliberately narrow tool surface. This split exists because giving one agent broad filesystem and search access led it to "research" the codebase instead of making the one edit its slice required.

### Structural decomposition as a formal step

Splitting oversized or context-overflowing work into smaller slices (`decomposer.py`) is treated as its own LLM-assisted step, not something an implementer agent is expected to improvise — because leaving it improvised was a repeatable source of failed or abandoned edits.

**Notes:**
- `spine.critic` and `spine.adversarial` are separate packages that consume `spine.agents` rather than living inside it; their own domain boundaries were not verified as part of this pass.
- The 651-symbol figure is from the supplied module-role metadata and was not independently recounted.

## Domain: spine.cli

`spine.cli` is the command-line surface of SPINE — "Deterministic AI Agent Harness," per its own `--help` text. Its purpose is to give an operator (human or scripted caller) a single `spine` binary for the full work-item lifecycle: submitting a task (`spine run`), checking on it (`spine status`, `spine list`), unblocking it after a human-review pause (`spine resume`), recovering a stuck run (`spine restart`), and exporting its artifacts for external analysis (`spine export`). It also owns two adjacent operational concerns: bootstrapping a brand-new SPINE project in a target directory (`spine init`, scaffolding `.spine/skills/`, `.spine/artifacts/`, and a baseline config) and building/serving the retrieval index that SPINE's agents use for codebase context (`spine index`, `spine ui`, `spine worker`).

A second layer of the domain is multi-work-item coordination: the `project` subgroup lets a user define a "project" — a roadmap spanning many individual work items — create it, add members, inspect deterministic requirement-coverage rollups, and run holistic integration verification/adversarial review across the whole project rather than per-item. A third, smaller layer is the `experience` subgroup, which exposes the cross-run "learned lesson" store (list/clear distilled lessons by phase) for inspection and pruning. Per the supplied module role, the entire domain is implemented by the one `spine.cli` module (21 functions, 0 classes) — it is a pure command-dispatch layer with no domain logic of its own, delegating to `spine.work`, `spine.project`, `spine.persistence`, `spine.agents`, `spine.ui`, and `spine.workflow`.

**Notes:**
- Domain boundary is inferred from the single supplied module role plus the source read in `spine/cli/__init__.py`; no other modules were named as part of this domain in the fragment.

## Domain: spine.config.py

The `spine.config.py` domain is configuration management: giving every other part of SPINE — the CLI, the agent phases, the ephemeral-pod infrastructure, the persistence layer — one consistent, environment-aware place to ask "what model do I use," "where do checkpoints live," and "which provider backs this phase." Its business purpose is to let an operator describe an entire run's behavior declaratively in `.spine/config.yaml` (provider credentials and models, retry limits, RAG/recall tuning, ephemeral GPU-pod policy, token-compaction and researcher-convergence thresholds) while still allowing environment variables to override any individual setting at deploy time, and while degrading gracefully to sensible defaults when a key is simply absent.

A second, load-bearing objective of this domain is portability across how SPINE gets launched: because the same code can run from a CLI invocation in a project root, from Streamlit (which may change CWD to somewhere like `/tmp` or `$HOME`), or from a background worker, the module goes out of its way to auto-detect the true workspace root and locate `.env`/`config.yaml` by walking upward from either the CWD or the installed package directory. Per the supplied module role, this whole domain is one file (`spine/config.py`, 19 symbols: 3 classes, 4 functions, 12 methods) — there is no separate "config domain" package, just this single, load-bearing module.

**Notes:**
- Domain scope reflects the one module role supplied; broader claims about how many other modules read `SpineConfig` are not made here since the fragment supplied no consumer edges for this direction (only the outgoing `spine.config.py → spine.agents` edge was given).

## Domain: spine.exceptions.py

The `spine.exceptions.py` domain exists to give SPINE's workflow engine a shared, structured vocabulary for what can go wrong and how the system should react. Its business purpose isn't user-facing directly — it underpins reliability: distinguishing errors that should trigger a bounded retry (`TransientAPIError`, consumed by an `invoke_with_retry()` helper elsewhere), errors that require escalating to a human (`CriticalContractFailure`, `MaxRetriesExceeded`), and errors specific to the git-sandbox orchestration that isolates a work item's changes (`SandboxPreparationError`, `ValidationError`, `MergeError` under `GitOrchestratorError`). By centralizing this hierarchy in one file with no dependencies of its own, any other module in SPINE can raise or catch a precisely-typed error without a circular import risk.

Per the supplied module role, this domain is implemented entirely by the single `spine/exceptions.py` file (17 symbols: 13 classes, 4 methods) — there is no separate exceptions package, and the fragment listed no dependency edges, consistent with the module having zero imports from the rest of spine.

**Notes:**
- The domain's real-world value — enabling structural retry via `CriticalContractFailure.carryover` and graceful degradation via `TransientAPIError` — is stated directly in the source docstrings, not inferred.

## Domain: spine.git

`spine.git` exists to answer one question safely: when an autonomous agent edits this repository's own code, how is a failed or rejected run kept from corrupting the working tree? Its purpose is transactional isolation for code-producing work — every run whose phase sequence performs an IMPLEMENT step executes against a disposable git worktree, never the live tree, and is fast-forward merged to `main` only once the run's terminal status confirms success (`"completed"`). Any other outcome — `needs_review`, `stalled`, `failed`, `cancelled` — triggers a full rollback: worktree removal, branch deletion, and a hard reset-and-clean of the master directory.

The domain's boundary is narrow and deliberate: it owns git mechanics (branch/worktree lifecycle, fast-forward merge, rollback) and an optional shell-command validation gate (lint/typecheck/test), but it does not decide *what* code to write or *when* a run counts as done. Those decisions belong to `spine.workflow` (phase/graph topology) and `spine.work` (dispatch and status derivation) respectively; `spine.git` only consumes their outputs — a `work_type` string and a terminal status string — to decide merge versus rollback.

Modules implementing this domain: `spine.git` (20 symbols: 2 classes/interfaces — `SpineGitOrchestrator`, `WorktreeSandbox` — 2 functions, 16 methods) is the sole module.

**Notes:**
- The repo-wide tech stack recorded alongside this domain (python, c, cpp, php, typescript) is the whole project's stack, not something specific to `spine.git` — nothing in the module ties individual languages to the git operations themselves; the commands it shells out to (ruff/mypy/pytest) are Python-tooling specific by convention, not by any language-detection logic in the module.

## Domain: spine.infra

The `spine.infra` domain is cost- and lifecycle-management for the compute SPINE's agent phases run against. Its business purpose is straightforward: LLM inference for planning/implementation phases is often served by a rented GPU pod (currently RunPod) rather than a hosted API, and renting a GPU by the hour makes "leave it running just in case" expensive. This domain automates the tradeoff — boot the specific pod(s) a run's phases actually need, right when the run starts, and guarantee teardown (success, exception, or Ctrl-C) so idle GPU time is never billed. It further supports splitting work across a "planning lane" (specify/plan/tasks/critic/adversarial) and an "execution lane" (implement/verify), which the project's human-approval-gated workflow already runs as separate `spine` invocations — letting an operator point a cheaper/smaller model at planning and a stronger coding model at implementation, each on its own pod, without paying for both simultaneously.

Per the supplied module role, this domain is implemented by 30 symbols (4 classes, 18 functions, 8 methods), essentially all of it living in `spine/infra/ephemeral_pod.py` (the package's `__init__.py` is a docstring only). The domain also absorbs the operational nuance of GPU capacity — a `gpu_fallback` ladder retries a boot request across alternative GPU types/counts when the primary spec is unavailable — and of provider-config integration: a pod is meant to back one `providers.llm[]` entry (via `binds_provider`), so once it's live, existing phase-routing config in `.spine/config.yaml` picks it up with no further code change.

**Notes:**
- No dependency edges were supplied for this domain in the fragment; the source-level coupling to `spine.config.SpineConfig` (for typing and for a runtime default-config load inside the `with_ephemeral_pod` decorator) was confirmed directly in `ephemeral_pod.py`.
- "Currently RunPod" is a direct reading of the source (`_import_runpod`, `_boot_runpod`) — the config schema has a generic `backend` field but no second backend implementation was found in this file.

## Domain: spine.mcp

The `spine.mcp` domain exists to let SPINE's agents pull in tools from third-party Model Context Protocol (MCP) servers — most notably a codebase-indexing server — without every agent needing to know how to speak the MCP session/transport protocol itself. Its job is narrowly scoped: given SPINE's own server configuration format, produce ready-to-use LangChain `BaseTool` objects that agents can bind and call like any other tool.

Two business concerns motivate the domain beyond simple protocol translation. First, correctness of context: MCP servers are launched as separate processes and don't inherently know which project SPINE is operating on, so this layer forces every server's `PROJECT_ROOT` environment variable to match SPINE's actual workspace root, overriding any stale or mistyped value a user's config might carry — a silent `PROJECT_ROOT` mismatch would otherwise make every codebase query return results for the wrong project. Second, cost/safety of agent context: MCP tools (especially codebase search) can return very large payloads that would blow out an agent's context window, so the domain enforces hard byte/line ceilings and strips out paths under SPINE's own scratch/checkpoint directories so agents don't mistake their own prior artifacts for project source.

The domain is implemented entirely by the single module `spine.mcp` (`spine/mcp`), which wraps LangChain's `MultiServerMCPClient`. There is no separate business-logic module beyond this adapter — the domain's scope begins and ends at "load and sanitize MCP tools for agent use."

**Notes:**
- The supplied fragment gave only the module's symbol count ("13 functions, 0 classes") and tech stack, not behavioral detail; the purpose/motivation described above is drawn from reading `spine/mcp/client.py` directly, since the fragment alone is too sparse to write a grounded purpose statement from.
- The `tech_stack` field in the fragment (`python, c, cpp, php, typescript`) appears to be a repo-wide list, not specific to `spine.mcp` — the module itself is pure Python; no C/C++/PHP/TypeScript files exist under `spine/mcp/`.

## Domain: spine.models

Every phase of a SPINE workflow — and every tool outside the workflow that inspects or reports on one (the CLI, the UI API, the persistence layer, the queue worker) — needs to agree on what a "task," a "plan slice," a "critic verdict," or "the current state of a run" actually looks like. `spine.models` (`spine/models/`, 28 symbols: 21 classes/interfaces, 4 functions, 3 methods) is where that shared vocabulary lives, and nothing else. It defines the enums that route workflow control flow (`PhaseName`, `WorkType`, `ReviewStatus`, `TaskStatus`), the Pydantic schemas that structure LLM output for guided decoding (`Specification`, `CriticReview`, `GapPlan`, `FixInstruction`), the dataclasses that describe a plan's implementation units (`FeatureSlice`, `StructuredPlan`), the project/roadmap layer (`ProjectSpec`, `RoadmapPhase`, `RequirementRef`), and the `WorkflowState` `TypedDict` (with its LangGraph reducers) that is the actual state object flowing through the graph.

Because it sits underneath the rest of the system with no dependencies of its own beyond Pydantic and the standard library, `spine.models` functions as the project's contract layer: changing a field here has ripple effects across the roughly 57 other files that import from it. Its purpose isn't to do anything — it has no I/O, no side effects — but to make sure "plan," "review," and "project" mean the same thing to the planner, the critic, the persistence layer, and the human reviewing a paused work item.

## Domain: spine.persistence

SPINE runs autonomous coding-agent workflows (specify → plan → implement → verify, with critic/adversarial/gap-fix loops) as long-lived, resumable processes — a single work item can span retries, human-review pauses, and process restarts. The `spine.persistence` domain exists to make that possible: it is SPINE's entire durability boundary. Everything the workflow produces or needs to recall across a process boundary — phase artifacts, LangGraph checkpoint state, distilled cross-run lessons, project/roadmap membership, and the indexed codebase used for retrieval-augmented context — is written and read through this one domain, implemented as a single module, `spine.persistence` (`spine/persistence/`, 65 symbols: 5 store classes plus their methods).

Concretely this buys SPINE three things a stateless agent loop couldn't have: crash-safety (atomic tmp-file-then-`os.replace` writes for JSON documents, so a crash mid-write never corrupts `spec.json` or the experience ledger), resumability (LangGraph checkpoints in `.spine/spine.db` let a work item pick up exactly where it left off, including after a human-review interrupt), and grounded generation (the vector store's hybrid BM25+embedding search lets agents recall real code symbols instead of hallucinating them, which is directly relevant to why this documentation effort exists). Concurrency safety — `fcntl` locking on the file-backed stores — matters because SPINE's own CLI, UI API, and background queue worker are separate OS processes that can legitimately touch the same on-disk state at once.

## Domain: spine.phases

`spine.phases` implements the per-phase business logic registration point for SPINE's task pipeline — one module per pipeline stage (specify, plan, implement, verify, critic, adversarial, gap_plan), each responsible for producing that stage's artifact (e.g. `plan.md`/`plan.json`, `verification.md`/`verification.json`) or, in the legacy path, for the actual node function LangGraph runs. Its purpose in the domain model is to be the seam between "a workflow phase exists as a name" (`spine.workflow.registry.PhaseRegistry`) and "a workflow phase does something" (an agent builder and, historically, a call function) — every phase module supplies both halves for exactly one phase, keeping phase-specific logic (plan-JSON parsing, cycle detection, early task classification) out of the generic orchestration code in `spine.workflow`.

Based on the source read directly, `spine.phases`' domain boundary has narrowed over time as its modules migrated from being the direct execution path to being fallback/registration shims, with real dispatch logic (parallel Send-based fan-out, best-of-N scoring, structural gates) now living in `spine.workflow.subgraphs`. The domain today is best understood as "phase self-registration plus a handful of surviving structural helpers" (JSON-based plan validation, wave computation, cycle detection) rather than the primary place where phase work happens; a new contributor should look to `spine.workflow.subgraphs` and `spine.agents` for the live implementation of what each phase actually does, and to `spine.phases` for how a phase name gets wired to its (fallback) agent/call functions.

**Notes:**
- The supplied fragment for this domain contained only the module's symbol-count role string (16 functions, 0 classes) with no additional domain modules — this description reflects that `spine.phases` is a single-module domain, grounded in the source files under `spine/phases/`.

## Domain: spine.project

The `spine.project` domain answers a question that per-work-item verification can't: does the codebase, taken as a whole, actually satisfy the project's requirements, once multiple independently-implemented work items are combined? SPINE decomposes larger efforts into a `ProjectSpec` — requirements grouped into roadmap phases, each phase backed by one or more member work items. Each member work item gets its own VERIFY phase in isolation, but nothing checks that the pieces integrate, that a requirement nominally "covered" by two different members isn't secretly only half-implemented, or that cross-cutting concerns (shared files, hidden assumptions between members) were handled.

`spine.project` closes that gap in three deliberately layered passes, all implemented in `spine/project/`. The cheapest and only deterministic pass, `aggregator.py`, computes exact-string requirement coverage across member specs with no LLM involved — a fast, auditable "is this even claimed to be covered" check. `project_verifier.py` then spends LLM judgment per roadmap phase to catch cross-member integration gaps the aggregator's string matching can't see (e.g. requirements that are covered on paper but not truly wired together). Finally `project_reviewer.py` runs a single holistic adversarial ("red-team") review of the completed project, looking for hidden assumptions, unimplemented edge cases, and contradictions between what different members built.

Results are persisted to `.spine/project/{id}/project_verification.json` and `.spine/project/{id}/project_review.json` respectively, so downstream tooling (the CLI and the UI API, both of which call into this domain) can present a project's overall health without re-running the checks.

**Notes:**
- The supplied fragment for this domain gave only a symbol count, not behavior; the purpose statement above is grounded in reading all three files under `spine/project/`.
- The `tech_stack` list in the fragment looks repo-wide rather than domain-specific — `spine/project/` contains only Python.

## Domain: spine.services

The `spine.services` domain is workflow auditability: giving SPINE a durable, queryable record of what happened during a run — which events fired, for which work item, in which phase, and when — independent of the transient in-memory/checkpoint state the workflow engine itself uses. Its business purpose is operational visibility and debugging: when a run behaves unexpectedly, the audit trail in `.spine/audit.db` (a SQLite database) lets an operator reconstruct the sequence of `phase_start`/`phase_complete`/`critic_review`-style events after the fact, filtered by work item or event type.

Per the supplied module role, this domain currently comprises 6 symbols (1 class, 5 methods) and is implemented entirely by `AuditService` in `spine/services/audit_service.py` — a narrowly-scoped logging service, not a broader "services layer" with multiple concerns. It is deliberately minimal: table creation is idempotent (`_ensure_table` no-ops if the table exists), and each operation opens its own tuned connection rather than holding shared state, which keeps it safe to call from multiple threads.

**Notes:**
- The fragment supplied no dependency edges for this domain; the one real cross-module dependency confirmed in source is `spine.persistence.sqlite_tuning.tune_connection`, used to tune each opened SQLite connection.
- Whether other spine domains (workflow engine, CLI, UI) currently call into `AuditService` was not established from the files read for this task — flagged as unverified rather than assumed.

## Domain: spine.ui

This domain is the human-facing control surface for SPINE: a Streamlit dashboard that lets an operator submit new work, watch it execute, inspect artifacts and audit trails, manage configuration/providers, review flagged work, and drive project onboarding — without ever touching the backend workflow engine directly. Its purpose is operational visibility and control, not business logic: every data read or mutation is delegated to the `spine.ui_api` facade, so this domain's own code is limited to page composition, navigation, live-update plumbing, and display formatting.

Structurally the domain is just `spine/ui` (per the supplied module role: 97 symbols — 2 classes, 85 functions, 10 methods), which in practice splits into an app shell/navigation layer (`app.py`, `pages.py`), a real-time notification layer (`ws_bus.py`, `ws_server.py`, `ws_component.py`), formatting utilities (`utils.py`), and a set of independent page renderers under `_pages/` (dashboard, work submission, queue monitoring, work detail/history, config, audit log, human review, spec/planning, onboarding, projects, learned-experience). Each page is a self-contained render function that takes the shared API instance and displays or mutates one slice of system state.

**Notes:**
- The fragment's `tech_stack` list (python, c, cpp, php, typescript) is tagged at the whole-repository level, not specific to this domain; every file actually read under `spine/ui/` is pure Python using the Streamlit framework. No evidence of C/C++/PHP/TypeScript code was found within `spine/ui` itself.
- The fragment supplies no explicit "purpose" text for this domain beyond the module role string — the purpose described above is inferred from reading `app.py`'s docstring and the page set, not asserted from a provided description field.

## Domain: spine.ui_api

This domain exists to enforce a single architectural rule: the UI and the CLI must run through the same backend code, never diverge into parallel logic. `spine.ui_api` implements that rule as one class, `UIApi`, which is the sole read/write gateway Streamlit pages are permitted to use — its own module docstring states pages must never import `workflow/`, `phases/`, or `work.dispatcher` directly. Functionally, the domain's scope is everything an operator console needs: projects, work items and their lifecycle (submit, enqueue, resume, restart, stop), artifacts and audit history, LLM/MCP provider configuration, and the background queue/worker's live status.

Per the supplied module role, this is a narrow, single-class domain (70 symbols: 1 class, 0 module-level functions, 69 methods) — there is no multi-module structure to speak of; all responsibility is concentrated in `spine/ui_api/api.py`. That concentration is deliberate: it gives the UI exactly one seam to depend on, so backend refactors in the workflow, dispatcher, or persistence layers only require updating this facade rather than every page.

**Notes:**
- The fragment's `tech_stack` tag (python, c, cpp, php, typescript) is repo-wide, not domain-specific; the code read in `spine/ui_api/api.py` is pure Python.
- It could not be verified from the supplied fragment alone why the domain is scoped exactly to `spine/ui_api` rather than merged with `spine.ui`; the separation is inferred from the "facade" docstring and the fact that `spine/ui_api` has no dependency back on `spine.ui`.

## Domain: spine.work

`spine.work` is the domain that turns a submitted work item into a finished, tracked result. Its core objective is a single, uniform entry point — `submit_work` — shared by the CLI, the Streamlit UI, and the background queue worker, so a task always follows the same lifecycle: create a SQLite-tracked entry, run the appropriate LangGraph workflow (or, for onboarding, a dedicated non-graph engine), stream progress back to the UI, persist artifacts, and land on a terminal status (`completed`, `needs_review`, `stalled`, `cancelled`, or `failed`) that downstream consumers — notably `spine.git`'s merge-or-rollback decision — act on.

A second objective lives inside this same domain: onboarding a repository (this one included) by analysing its structure and generating four documents — `PROJECT_DEFINITION`, `CODING_GUIDELINES`, `ARCHITECTURE_MAP`, `SPINE_ASSISTANCE_REQUIREMENTS` — for new contributors and for Spine's own agents. That objective is why this document exists: `spine.work.onboarding`'s synthesis pipeline is the mechanism that would ordinarily produce text like this automatically, working from bounded slices of a `RepoManifest` (built by `RepoAnalyzer`) so no single LLM call ever needs the whole codebase at once.

The domain's boundary excludes the git isolation mechanics (delegated to `spine.git`), the phase/graph topology itself (delegated to `spine.workflow`), and checkpoint/artifact storage internals (delegated to `spine.persistence`) — `spine.work` orchestrates calls into all three rather than reimplementing them.

Modules implementing this domain: `spine.work` (182 symbols: 18 classes/interfaces, 116 functions, 48 methods) is the sole module.

**Notes:**
- 182 is the module's actual total symbol count. The fragment used to ground this document sampled only a handful of representative symbols (`RepoManifest`, `RalphLoopWorker`, `RepoAnalyzer`, `ReadRepoManifestTool`, etc.) — that sampling is an artifact of how the source data for this document was assembled, not evidence that `spine.work` is small or that its surface is limited to those symbols.

## Domain: spine.workflow

`spine.workflow` implements SPINE's workflow-orchestration domain: it is responsible for defining and compiling the LangGraph state machines that drive an autonomous coding work item from specification through verification. Its business purpose is to make each phase's execution deterministic and observable — every phase gets an isolated state schema, a compiled subgraph with its own checkpointer, and gate/scoring logic that decides whether output is good enough to proceed, needs rework, or must escalate to a human. This is the layer where "how a task actually gets processed" is decided: which phases run for a given `WorkType`, how rework loops (critic, adversarial, gap-fix) are bounded and detected as converging or stagnating, and how parallel work (multiple feature slices implemented or verified at once) is scheduled and merged back together.

The domain boundary is orchestration and control-flow, not task content generation — the actual LLM agents that write specs, plans, code, and reviews live in `spine.agents` (and `spine.critic`/`spine.adversarial` for review agents), which `spine.workflow` invokes but does not implement. `spine.workflow` also stops short of persistence and process-management concerns, which belong to `spine.persistence` and `spine.work` respectively; it depends on both for artifact storage and work-item bookkeeping. Within its own boundary, `spine.workflow` owns the quality-control logic that keeps rework loops honest: deterministic, LLM-free checks (`plan_reference_gate.py`'s dangling-symbol gate, `critic_convergence.py`'s repeat-verdict detection, `verify_snapshot.py`'s best-state ratchet) exist specifically to catch cases where an LLM-driven rework cycle would otherwise loop indefinitely, regress previously-passing work, or stagnate without a clear signal — a recurring failure mode documented in the module's own docstrings via specific incident traces.

**Notes:**
- The supplied fragment gave only the module's symbol-count role string, not additional domain modules to cross-reference — this description is grounded in the source files read directly under `spine/workflow/`.

## Domain: tests.conftest.py

This domain is test-harness hygiene: making sure the suite can exercise the real `spine` package — including code paths that shell out to git and touch environment-derived config — without side effects that outlive the test run. A test that mutates the developer's real git HEAD or emits a real LangSmith trace during a normal `pytest` run would be a correctness bug in the *test infrastructure*, not just noise, since it corrupts the environment other tests (and the developer) depend on.

The secondary purpose is reducing duplication: `SpineConfig` construction with isolated `checkpoint_path`/`artifact_path`/`queue_path`, plus canned `Task`/`Artifact`/`ReviewFeedback`/`PromptRequest` instances, are needed by many test modules, so they live here once rather than being re-implemented per file.

## Domain: tests.fixtures

This domain exists because SPINE's codebase-indexing tooling has to parse more than Python — the product supports multi-language repositories. These fixtures give the AST-extraction and MCP codebase-index tests concrete, minimal, human-readable source in C, C++, PHP, and TypeScript so extraction correctness (classes, enums, interfaces, methods, free functions) can be verified per language without depending on a real third-party codebase.

## Domain: tests.integration

This domain validates that SPINE's phases actually compose end-to-end rather than just passing in isolation: that JSON artifacts produced by one phase are valid input for the next, that dependency-aware scheduling produces a correct wave order for parallel/serial implement dispatch, that the MCP codebase-index server the agents rely on for retrieval is real and reachable (not just mocked), that git sandboxing and merge-back are safe against a real repository, and that the LLM-facing system prompt assembly matches SPINE's own harness profile rather than the upstream `deepagents` conversational default.

## Domain: tests.recall_eval

This domain exists to make retrieval-quality changes measurable rather than judged by feel. Because `miss@N` is a hard ceiling — no downstream reranking step can surface a result the retriever never returned — the harness treats it as the primary number to drive toward zero, with `hit@k` and `MRR` as secondary signals for whether a good result lands in the window the researcher actually reads, and in what order. The golden set separates `curated` (sharp, hand-written, single/dual-target) queries from `mined` (realistic multi-file) queries so both stay visible in the report.

## Domain: tests.test_core.py

This domain covers human-readable duration formatting — the small utility that turns a raw second count into a display string like `"1h 30m"`. It matters because it's a display-facing function: a bug here (e.g. mishandling negative durations or boundary values) would show up directly as a malformed or nonsensical elapsed-time string wherever the UI reports how long something took.

## Domain: tests.test_plan_score.py

This domain is plan-quality scoring for automatic best-of-N plan selection: when the PLAN phase generates multiple candidate plans, `score_plan` must reliably rank a coherent, schedulable, spec-covering plan above degenerate ones (cyclic dependencies, empty or duplicated target files) so the harness can pick a winner without a human in the loop.

## Domain: tests.test_restart.py

This domain is operational recovery for the work queue: giving an operator a safe way to restart work that got stuck, cancelled, or sent to review, and to bulk-reset queue rows stuck in `running` after an unclean shutdown — without silently destroying artifacts unless explicitly asked to. It also covers the small piece of UI vocabulary (status → color) operators read at a glance on the dashboard.

## Domain: tests.test_work_ordering.py

This domain is the work-queue dashboard's list-ordering contract. Operators scanning the queue need items in a stable, predictable newest-first order even under filtering, row limits, concurrent inserts with identical timestamps, or rows with a missing `created_at` — a wrong order here would show operators stale work as if it were current, or vice versa.

## Domain: tests.unit

This domain is correctness of the SPINE agent-harness engine itself: the multi-phase (SPECIFY/PLAN/IMPLEMENT/VERIFY) workflow graph and its critics, gates, and convergence guards; the onboarding-documentation generation pipeline this very task is producing content for; dispatcher/worker/queue mechanics; git sandboxing; MCP-based codebase retrieval; token/context budget enforcement; and ephemeral GPU compute provisioning. Because the product *is* an LLM-driven multi-phase work harness, this is where most of the engineering — and most of the test count — concentrates; the consistent use of fake doubles instead of live models keeps that large surface testable without real LLM calls or cloud spend.

## Supporting Domains

Several modules in this codebase are single-purpose utilities or infrastructure glue rather than core workflow logic — each does one narrow job, has few (often one or two) symbols, and mostly sits at the edges of the dependency graph rather than the center. They're grouped here because none warrants its own extended treatment, but a new contributor still needs to know they exist and what each is for.

### alembic.env.py

The Alembic migration runner script (`alembic/env.py`, 2 functions). Configures the migration context against `spine.models.work_entry.Base.metadata` and runs migrations either offline (SQL generation, no DB connection) or online (live engine connection), selected automatically based on how Alembic was invoked.

### alembic.versions

The migrations directory (`alembic/versions`, 2 functions total). Currently contains exactly one revision, which creates the `work_entries` table — the persistence backing for SPINE's work-item tracking.

### spine.adversarial

A red-team reviewer (`spine/adversarial`, 3 functions) that runs after the critic approves a plan, for critical work types: it actively looks for ways the plan will fail rather than confirming it looks reasonable, then routes fixable findings back to planning or escalates ambiguous ones to a human.

### spine.critic

The quality-review gate (`spine/critic`, 1 function) that every phase's output passes through before the workflow proceeds, judging semantic quality (completeness, correctness, clarity, actionability) while deliberately staying out of schema-validation territory.

### spine.log.py

A one-function logging bootstrap (`spine/log.py`) that lets the CLI toggle between quiet (`WARNING`) and verbose (`DEBUG`) console logging.

### spine.observability.py

LangSmith tracing control (`spine/observability.py`, 2 functions). Tracing is off by default project-wide; this module is the opt-in mechanism that scopes tracing to a single work-task run so indexing/tests/ad-hoc usage never get traced by accident.

### src.utils

A small standalone helper package (`src/utils`, 1 function: `format_duration` in `src/utils/helpers.py`) that formats a duration in seconds into a human-readable string (e.g. "1h 30m"). Verified to exist at the repo root. It appears to be a separate, duplicate implementation from the `format_duration` actually used in production UI code (`spine/ui/utils.py`); `src/utils/helpers.py` is exercised only by `tests/test_core.py`.

### tests.smoke

A manual smoke-test script (`tests/smoke`, 2 functions) that drives the SPECIFY phase's recall tool against a live agent build to confirm retrieval-augmented context is actually being used, with MCP tools and memory patched out to fit within a token budget.

**Notes:**
- `src.utils` DOES resolve to a real module in this repo (`src/utils/helpers.py`) — verified directly, not a hallucinated or missing entry.
- `alembic.versions` currently has only one migration file; it is not yet a large migration history.
- `tests.smoke` contains one script that is a manual diagnostic tool, not part of the automated pytest suite.
