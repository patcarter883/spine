# Coding Guidelines

## Logging Conventions

Every module that logs instantiates its own logger at import time with `logger = logging.getLogger(__name__)` immediately after the import block, then calls `logger.info`/`.warning`/`.error`/`.exception` on that module-level instance — never the root logger and never a hardcoded name. This appears in 94 files across `spine/`, including `spine/agents/decomposer.py`, `spine/agents/backend.py`, and `spine/phases/gap_plan.py`. Root logger setup itself is centralized in a single place, `spine.log.configure_logging`, which sets level and format for the whole process from a CLI verbose flag — individual modules must never call `logging.basicConfig` themselves.

For messages with dynamic content, prefer lazy `%s`-style formatting (`logger.warning("workspace_root %s is not writable", resolved_root)`) over eagerly-interpolated f-strings, so string formatting is skipped when the level is disabled; this is the majority style (115 call sites) though f-string usage is a real, tolerated minority (59 sites, e.g. `spine/phases/plan.py`). When logging inside an `except` block for a caught error you still need visibility into, use `logger.exception(...)` (no need to pass the exception object) rather than `logger.error(str(e))`.

### Module-level logger declaration (`spine/agents/backend.py`)

Declares `logger = logging.getLogger(__name__)` once, right after the `import logging` line and before any other module code, so every function in the file logs through the same named logger.

**Notes:**
- `spine/config.py` deviates from this pattern: instead of a module-level logger, `SpineConfig.load()` does an inline `import logging` and calls `logging.getLogger(__name__).warning(...)` directly inside the method body. Treat this as a one-off, not a second accepted style — new code should use a module-level logger.

## Data Model Conventions

Immutable, structural data — records produced once by an analysis pass and then only read — is modeled as `@dataclass(frozen=True)`, not as plain dicts or mutable dataclasses. `spine/work/onboarding/manifest.py` defines `SymbolRef`, `ModuleBoundary`, `DependencyEdge`, `PatternFinding`, and `RepoManifest` this way; `spine/agents/tools/ast_extract.py` defines `Edge` and `Symbol` the same way; `spine/agents/synthesis_budget.py` does it for `SynthesisBudget` and `EvidenceAllocation`. `spine/work/onboarding/analyzer.py`'s `RepoAnalyzer.analyze` builds and returns exactly these frozen types, confirming the pattern is used for real cross-boundary data, not just as decoration.

By contrast, objects that represent live, mutable runtime state — configuration loaded from disk and then mutated by callers, or long-lived service state — use plain `@dataclass` (no `frozen=True`). `spine/config.py`'s `SpineConfig`, `ConvergenceConfig`, and `TokenCompactionConfig` are all plain dataclasses; tests routinely mutate a `SpineConfig` instance's fields directly after construction (`tests/conftest.py`'s `test_config` fixture sets `config.checkpoint_path`, `config.artifact_path`, etc. post-construction). The rule: if a value is computed once and passed downstream as a fact, freeze it; if it's a stateful object that callers configure after construction, don't.

### Frozen analysis output (`spine/work/onboarding/manifest.py`)

`RepoManifest` and its four constituent frozen dataclasses are the immutable output contract of the repo-analysis pipeline — nothing downstream mutates them.

**Notes:**
- Neither `RepoAnalyzer` (a plain class) nor `SpineConfig` (an unfrozen dataclass) is itself frozen — only the data model types they produce/consume are.

## Naming Conventions

Nearly every function and method signature carries full type hints, including parameter types and an explicit return type — even `-> None` for pure side-effecting functions. An AST scan of `spine/` found 1,356 non-dunder function/method definitions, of which 1,310 (96.6%) had an explicit return annotation; the exceptions cluster in a handful of middleware hook signatures (`spine/agents/tool_forcing.py`, `spine/agents/context_editing.py`) whose types are dictated by an external framework's callback protocol. Even boilerplate migration code follows this: `alembic/env.py`'s `run_migrations_offline() -> None` and `run_migrations_online() -> None` are annotated despite being one-off scripts.

Identifier casing follows standard Python convention consistently rather than by accident: classes are `PascalCase` (`PhaseName`, `StructuredPlan`, `RoadmapPhase` in `spine/models/*.py`), functions and variables are `snake_case`, and module-private helpers not meant for import elsewhere are prefixed with a single underscore (`_bind_capped`, `_scrub_phantom_symbols`, `_normalize_per_file_slices` in `spine/agents/decomposer.py`). No `def` in the codebase uses non-snake_case naming.

### Fully-annotated private helper (`spine/agents/decomposer.py`)

`_build_reference_signatures_block` and its siblings are underscore-prefixed (signaling internal-only use) while still fully typing every parameter and return value — the two conventions (privacy marker, complete typing) are applied together, not traded off against each other.

**Notes:**
- A small number of framework-integration callbacks (`before_model`, `wrap_model_call`, `wrap_tool_call` families) skip return annotations because their signature is fixed by an external middleware protocol — treat these as accepted exceptions, not evidence the rule is optional elsewhere.

## Error Handling Conventions

The codebase defines a custom exception hierarchy in `spine/exceptions.py` rooted at `SpineError`, with purpose-specific subclasses (`CriticalContractFailure`, `MaxRetriesExceeded`, `TransientAPIError`, `ConfigurationError`, `GitOrchestratorError` and its children). The load-bearing pattern built on top of it, seen in `spine/adversarial/review.py`'s `agent_adversarial_check`, `spine/workflow/subgraphs/specify_subgraph.py`, and `spine/workflow/subgraphs/exploration_subgraph.py`, is a two-tier catch: `except CriticalContractFailure: raise` re-propagates workflow-contract violations untouched (they signal the workflow cannot safely continue), while a following `except Exception as e:` catches everything else, logs it with `logger.error(..., exc_info=True)` or `logger.exception(...)`, and returns a safe fallback result (e.g. a `NEEDS_REVISION` status dict) instead of crashing the phase.

For narrower, expected failure modes, functions catch a specific exception and degrade gracefully rather than propagating: `spine/agents/_tokens.py`'s `count_tokens` wraps the `tiktoken` import and encode call in a bare `except Exception`, falling back to a character-count heuristic so callers never handle `ImportError`; `spine/agents/artifacts.py`'s `scan_artifact_dir` catches `OSError` per-file and logs a warning rather than aborting the whole directory scan. Across the codebase there are zero bare `except:` clauses — always catch a named exception type (even if that type is `Exception`).

### Contract-vs-recoverable split (`spine/adversarial/review.py`)

`agent_adversarial_check` re-raises `CriticalContractFailure` unchanged but converts any other exception into a `NEEDS_REVISION` verdict dict — the calling graph is never allowed to crash on an adversarial-review failure, but a broken invariant (missing structured plan payload) is never silently swallowed either.

**Notes:**
- `TransientAPIError` in `spine/exceptions.py` is explicitly documented as "not raised to callers by default" — it exists for internal classification inside `invoke_with_retry`, not as a general-purpose error type to raise from new code.

## Testing Conventions

Tests live under `tests/`, split into `tests/unit/`, `tests/integration/`, `tests/smoke/`, and `tests/recall_eval/` by scope, and use plain pytest — 137 files contain `test_*` functions and/or `Test*` classes (e.g. `tests/unit/test_read_cache_dedup.py`'s `TestMultimodalDedupe` and `TestTurnBudgetGuard`), with `pytest-asyncio` (`@pytest.mark.asyncio`) for coroutine tests. Shared fixtures live in `tests/conftest.py` (`temp_dir`, `test_config`, `sample_task`) rather than being redefined per file. Integration tests favor structural assertions over live LLM calls where possible — `tests/integration/test_exploration_subgraph.py` builds and compiles a subgraph and asserts it's non-`None`, explicitly documenting that it tests "graph topology, state schema, and edge routing only," not live model behavior.

`tests/conftest.py` also carries a real safety mechanism worth calling out as a convention in its own right: an autouse fixture (`_guard_repo_git_head`) snapshots the real repo's git HEAD before every test and fails loudly (restoring HEAD) if a test mutated it — because `SpineGitOrchestrator` defaults its working directory to `os.getcwd()`, an unguarded test can flip the developer's actual checkout. Any new test that exercises git-sandbox code must point `master_dir` at a temp repo rather than relying on this guard to catch the mistake after the fact.

### Fixture named like a test (`tests/conftest.py`)

`test_config` is a `@pytest.fixture`, not a test function, despite the `test_` prefix — this is safe because pytest only auto-collects `test_*` functions from test modules (files named `test_*.py`), not from `conftest.py`.

**Notes:**
- Don't copy the `test_config`-style naming into an actual `tests/unit/test_*.py` file — there, a bare `test_`-prefixed function *is* collected and run as a test, fixture or not.

## Config Conventions

Runtime configuration is centralized behind `SpineConfig.load()` in `spine/config.py` — 107 call sites across `spine/` (agents, phases, workflow, CLI, UI API) resolve config through this one classmethod rather than reading `.spine/config.yaml` or environment variables directly. `SpineConfig` exposes typed resolver methods for the pieces callers actually need — `resolve_model`, `resolve_active_provider`, `resolve_provider_config`, `resolve_embedding_provider`, `resolve_reranker_provider` — so consumers ask for "the model for this phase" rather than re-implementing YAML-plus-env-var precedence themselves. New code that needs configuration should add a resolver method to `SpineConfig` rather than reading `.spine/config.yaml` inline.

Two documented exceptions exist and should not be treated as violations to copy indiscriminately. First, `spine/git/orchestrator.py` reads a separate, unrelated YAML file (a git validation-gate config) directly via `yaml.safe_load` — this is a distinct config surface from `SpineConfig`, not a workaround of it. Second, `spine/ui_api/api.py` reads and rewrites the raw `.spine/config.yaml` directly with `yaml.safe_load`/manual dict edits in at least five handlers (adding/removing MCP servers, adding/updating/removing LLM providers, setting a phase provider) — this bypasses `SpineConfig` because `SpineConfig.load()` is read-only and has no write-back path; any future write-capable config API should add mutation support to `SpineConfig` itself rather than growing more ad hoc YAML-rewrite call sites in the UI layer.

### Centralized resolution (`spine/agents/decomposer.py`)

`run_decomposer` and its callers obtain model and provider settings through `SpineConfig.load()` plus its `resolve_*` methods rather than reading environment variables or YAML inline.

**Notes:**
- Treat direct `yaml.safe_load` of `.spine/config.yaml` outside `spine/config.py` (as seen repeatedly in `spine/ui_api/api.py`) as a known gap, not a pattern to extend — it exists only because the read-write UI use case isn't yet served by `SpineConfig`.
