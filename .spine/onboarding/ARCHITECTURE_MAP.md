# Architecture Map

## Module: spine.agents

**Responsibility:** Core agent orchestration, decomposition, and tool-based code analysis within the Spine system.

**Key Symbols:**
- `AstExtractSymbolInput`, `AstExtractSymbolTool` (spine/agents/tools/ast_extract_symbol.py)
- `CodebaseQueryInput`, `CodebaseQueryTool` (spine/agents/tools/codebase_query.py)
- `FindReplaceEdit` (spine/agents/tools/read_edit_lint.py)
- `CheckItem`, `DecompositionResult`, `FeatureSliceSchema` (spine/agents/subagents.py, spine/agents/decomposer.py)

**Dependencies:**y depends on `spine.mcp`, `spine.models`, `spine.work`, and `spine.workflow`.

**Dependents:** None listed.

## Module: spine.cli

The `spine.cli` module provides the command-line interface entry point and subcommands for the spine application. It exposes all CLI functionality through functions defined in `spine/cli/__init__.py` and acts as the orchestration layer that delegates work to core domain modules.

### Responsibility

- **CLI Entry Point**: Exposes `main()` as the top-level command-line entry function.
- **Subcommand Implementation**: Provides 15 additional functions for specific operations:
  - Export functionality: `export`\n  - Project management: `project`, `project_add`, `project_create`
  - Task/workflow operations: `list_cmd`
  - Initialization: `init`, `index`

### Key Symbols

All symbols are located in `spine/cli/__init__.py`:

| Symbol | Type | Description |
|--------|------|-------------|
| `export` | function | Handles export operations
| `index` | function | Handles indexing operations
| `init` | function | Initializes new projects
| `list_cmd` | function | Lists tasks/projects
| `main` | function | Primary CLI entry point
| `project` | function | Project-related operations
| `project_add` | function | Adds projects
| `project_create` | function | Creates new projects |

### Dependencies

The module depends on the following core modules:
- `spine.agents` - Agent management operations
- `spine.models` - Data models and structures
- `spine.persistence` - Data persistence layer
- `spine.project` - Project domain logic
- `spine.work` - Work/task management
- `spine.workflow` - Workflow orchestration

### Reverse Dependencies

No modules depend on `spine.cli`; it is a leaf module in the dependency graph and serves as the topmost-is-section CLI facade

### Data Flow

```
[User Input] → main() → [delegate to specific subcommandsn] → [domain modules] → [persistence/models] → [return results]
```

The CLI module receives user commands through Click decorators or argparse, validates inputs, and delegates all business logic to the appropriate domain modules (`spine.project`, `spine.work`, etc.). Results are formatted and returned to the user.

### Notes

- The module is isolated to functions only (0 classes/interfaces)
- All functions are defined in a single file: `spine/cli/__init__.py`
- Serves as an orchestration layer rather than implementing business logic directly:is

```markdown
### Module: `.config.py

**Responsibility:** The `spine.config.py` module centralizes configuration management for the Spine application. It defines data structures for runtime settings, handles loading and parsing of configuration sources, and provides utilities for discovering the workspace context.

**Key Symbols:**

Classes/Interfaces:
- `ConvergenceConfig` (`spine/config.py`) – Defines parameters for numerical convergence behavior.
- `SpineConfig` (`spine/config.py`) – Main configuration container aggregating application-wide settings.
- `TokenCompactionConfig` (`spine/config.py`) – Encapsulates rules for token compaction strategies.

Functions:
- `_disable_global_tracing()` (`spine/config.py`) – Disables tracing facilities globally.
- `_load_dotenv()` (`spine/config.py`) – Loads environment variables from `.env` files.
- `_parse_convergence_config()` (`spine/config.py`) – Parses convergence-related configuration data.
- `_lookup_provider_by_name()` (`spine/config.py`) – Resolves a provider instance by its registered name.

Methods:
- `_find_workspace_root()` (`spine/config.py`) – Locates the root directory of the current workspace.
- *(Additional methods inferred from symbol count but not detailed in fragment.)*

**Dependencies & Dependents:**

The module currently shows no explicit external dependencies (`edges: []` in the fragment). It likely serves as a foundational component, potentially imported or extended by other modules that require structured configuration access. Its outputs (config objects) are consumed downstream to control behaviors such as convergence algorithms, token handling, and provider resolution.

---
```

## Module: spine.exceptions.py

**File:** `spine/exceptions.py`

**Responsibility:** Defines a collection of custom exception classes for the Spine framework, providing structured error handling across the system.

**Key Symbols:**

- `AgentUnavailableError` *(class)*
- `ConfigurationError` *(class)*
- `CriticError` *(class)*
- `CriticalContractFailure` *(class)*
- `GitOrchestratorError` *(class)*
- `MaxRetriesExceeded` *(class)*
- `MergeError` *(class)*
- `PromptRequestError` *(class)*

**Dependencies:**

No upstream dependencies recorded (no `edges` defined inn
**Dependents:**

No immediate dependents listed in the provided fragment.

## Module: spine.git

**Path:** `spine/git`

**Responsibility:**

The `spine.git` module provides Git orchestration and sandboxed
workspace sandboxing isolation utilities. It contains the
`SpineGitOrchestrator` class responsible for managing Git-based
workflows with support for phase prerequisites and shell execution,
and the `WorktreeSandbox` class for safe, isolated git worktree
operations.

**Key Symbols:**

* `SpineGitOrchestrator` (class) — Core orchestrator for Git
  operations
* `WorktreeSandbox` (class) — Isolated
  sandbox for isolated Git worktrees
* `SpineGitOrchestrator.__init__` (method) — Initializes the
  orchestrator
* `WorktreeSandbox.__init__` (method) — Initializes the
  sandbox
* `SpineGitOrchestrator._check_phase_prerequisites` (method) —
  Validatesn
  prerequisites
* `SpineGitOrchestrator._execute_shell` (method) — Executes
  shell commands
* `SpineGitOrchestrator._resolve_validation_command` (method) —
 lid resolve validation commands
* `WorktreeSandbox.abort` (method) — Aborts the
  sandbox operation

**Dependencies & Dependents:**

* * on:*
  * `spine.models`
  * `spine.workflow`
* Is depended on by: None directly listed

**Data:**

Total: 18 symbols (2 classes, 2 functions, 14 methods).

## Module: spine.mcp

- **Responsibility:** MCP client-side utilities for tool invocation, result processing, and server configuration management.
- **Key Symbols:**
  - `_cap_result()` & `_post_process_result()` – Result aggregation and post-processing.
  - `_convert_server_config()` – Server configuration transformation.
  - `_get_client()` – Asynchronous MCP client initialization.
  - `_line_starts_with_excluded_path()` & `_strip_excluded_paths()` – Path filtering.
  - `_namespace_tool()` – Tool namespace prefixing
  - `_run_async()` – Async execution wrapper.
- **Dependencies:** No dependent modules listed.
- **Depended On By:** No outgoing dependencies declared in the provided fragment.

## Module: spine.models

The `spine.models` module serves as the foundational type and enum definitions layer, housing domain objects, configuration specifications, and state representations used throughout the system.

### Responsibility
- **Type Definitions**: Defines core domain classes and interfaces that model artifacts, reviews, project specifications, and planning constructs.
- **Enumeration **Enums**: Provides standardized enumerations for system phases and states.
- **State Representations**: Represents phase results and execution states for downstream processing.

### Key Symbols

| Symbol | Type | File | Purpose |
| --- | --- | --- | --- |
| Artifact | --- | --- | --- |
| Artifact | --- | --- | --- |
| Artifact | class | spine | Represents transactional or deliverable entities within workflows. |
| CriticReview | class | het | Captures | Captures feedback or evaluation data structures. |
| FeatureSlice | class | .geo | Defines bounded segments or modules of functionality. |
| FixInstruction | class | .geo | Encodes corrective actions or patches. |
| GapPlan | class | .geo | Specifies remediation | Represents phased | /state.py | Models the conclusion and output of a processing phase.
| ProjectSpec | class | .geo | Declares input requirements or desired outcomes for projects.
| PhaseName | class | .enums | Enumerates the named stages of system execution.

### Dependencies
- **Undefined** (leaf module with edges: [])
  - No inbound or outbound dependencies documented in the provided fragment.

### Depended-On By
- **Undocumented**
  - Modules consuming `spine.models` types are not listed in this fragment.

## Module: spine.persistence

### Responsibility
The `spine.persistence` module provides data persistence and storage abstractions. It defines a set of store classes that serve as interfaces for persisting |*|

## Module: spine.phases

The `spine.phases` module orchestrates workflow phases through a collection of functions organized around four primary files. Its main responsibility is coordinating the lifecycle of a workflow instance, from initial specification through planning, execution planning, and validation (critic stage).

### Key Symbols

Located | Symbol | File | Responsibility |
|---------|------|---------------|
| `_build_critic_agent` | `spine/phases/critic.py` | Constructs the critic agent instance for validation. |
| `_compute_waves` | `spine/phases/plan.py` | Computes execution waves for plan scheduling. |
| `_early_commitment` | `spine/phases/specify.py` | Handles early commitment logic during specification. |
| `_has_cycle` | `spine/phases/critic.py` | Detects cycles in the workflow dependency graph. |
| `_load_plan_json` | `spine/phases/critic.py` | Loads plan configuration from a JSON file. |
| `_read_plan_json` | `spine/phases/plan.py` | Reads and parses plan JSON data. |
| `_validate_plan_structure` | `spine/phases/critic.py` | Validates the structural integrity of a plan. |
| `call_critic` | `spine/phases/critic.py` | Invokes |

### Dependencies

- **Depends on**:
  - `spine.critic`: Provides validation and agent functionality.
  - `spine.workflow`: Supplies core workflow orchestration primitives.

- **Is depended on by**: No dependent modules are specified in this fragment.

## Module: spine.ui

**Responsibility**

The `spine.ui` module provides a web-based user interface layer for the Spine workflow system. It exposes HTTP endpoints, WebSocket event buses, and dashboard pages that allow users to monitor and interact with workflow executions.

**Key Symbols**

- **Classes (2)**
  - `Event` (spine/ui/ws_bus.py) – Data class representing a UI event.
  - `WSEventBus` (spine/ui/ws_bus.py) – WebSocket-backed event bus for real-time UI updates.

- **Functions & Methods (76)**
  - Application & configuration helpers
    - `_audit_log()` – Records UI audit trail.
    - `_config()` – Builds UI configuration objects.
    - `_dashboard()` – Renders the main dashboard page.
  - WebSocket handlers
    - `_client_handler()` – Handles new WebSocket client connections.
  - Workflow detail pages
    - `_execution_duration()` – Displays execution duration on work‑detail pages.
  - Additional UI utilities not fully listed but contribute to the total of 66 functions and 10 methods.

**Dependencies**

- ` on: `spine.workflow` (monitors workflow state and execution).

## Module: spine.ui_api

### Responsibility
`spine.ui_api` provides the core UI API layer for the application.

### Key Symbols
- **`UIApi`** (class)   plated in `spine/ui_api/api.py`.
- **`__init__`** (method) — Constructor for `UIApi`.
- **`_active_jobs`** (method) — Retrieves active jobs.
- **`_artifact_store_for`** (method) — Manages artifact storage per context.
- **`_build_active_job`** (method) — Constructs active job representations.
- **`_discover`** (method) — Handles discovery logic.
- **`_finalise_ghost_entries`** (method) — Finalizes placeholder entries.
- **`_mark_running`** (method) — Marks jobs or tasks as running.

### Dependencies
- **` on: `spine.persistence`
- Depends on: `spine.project`
- Depends on: `spine.work`
- Depends on: `spine.workflow`

### Depended On By
- No downstream dependencies are recorded in the provided fragment.

## Module: spine.work

**Path:** `spine/work`
**Symbols:** 166 total (15 classes/interfaces, 103 functions, 48 methods)

### Responsibility
The `spine.work` module orchestrates repository analysis workflows, including dependency analysis, boundary detection, and state management for code exploration and synthesis.

### Key Symbols

- **`DependencyEdge`** *(spine/work/onboarding/manifest.py)*:://DependencyEdge/class)* Represents a directed dependency relationship between modules.
- **`ModuleBoundary`** *(spine/work/onboarding/manifest.py://ModuleBoundary/class)* Defines the extent and composition of a module in the codebase.
- **`OnboardingGraphState`** *(spine/work/onboarding/onboarding_state.py):(://OnboardingGraphState/class)* Manages the runtime state of the onboarding graph traversal.
- **`PatternFinding`** *(spine/work/onboarding/manifest.py):(//PatternFinding/class)* Captures Initially empty summary; likely captures recurring structural patterns during analysis.
- **`RalphLoopWorker`** *(spine/work/ralph_worker.py):(//RalphLoopWorker/class)* Implements a worker class that executes continuous processing loops.
- **`ReadRepoManifestTool`** *(spine/work/onboarding/synthesis_tools.py):(//ReadRepoManifestTool/class)* Tool to read and parse repository manifest data.
- **`RepoAnalyzer`** *(spine/work/onboarding/analyzer.py):(//RepoAnalyzer/class)* Analyzes Analyzes Core component responsible for analyzing repository structure and relationships.
- **`RepoManifest`** *(spine/work/onboarding/manifest.py):(//RepoManifest/class)* Central data structure representing the manifest of a repository being onboarded.

### Dependencies

`spine.work` **depends on**:
- `spine.agents`
- `spine.git`
- `spine.mcp`
- `spine.models`
- `spine.persistence`
- `spine.ui`
- `spine.workflow`

### Depended On By

No reverse dependencies provided in this fragment.

### Data Flow

The workflow begins when [`RepoAnalyzer`] ingests a repository and produces a [`RepoManifest`]. This manifest is consumed by [`RalphLoopWorker`] which manages processing iterations. State changes are tracked by [`OnboardingGraphState`], while [`DependencyEdge`] and [`ModuleBoundary`] provide structural insights used in [`PatternFinding`].

---

*Generated by semantic-analyzerlement extractor]*

## Module: spine `spine.workflow`

### Responsibility

The `spine.workflow` module serves as the workflow orchestration layer, coordinating multi-stage processes through phase-based execution and hierarchical state management.

### Key Symbols

**State Classes** (`spine/workflow/subgraph_state.py`):
- `BaseSubgraphState` – foundational state container for workflow subgraphs
- `CriticSubgraphState` – evaluation and quality assurance phase state
- `ExplorationSubgraphState` – discovery and research phase state
- `GapPlanSubgraphState` – gap analysis and planning state
- `ImplementSubgraphState` – execution and implementation phase state
- `PlanSubgraphState` – high-level planning state

**Registry Classes** (`spine/workflow/registry.py`):
- `PhaseDefinition` – declarative phase configuration and metadata
- `PhaseRegistry` – centralized phase definitions and lifecycle management

### Dependencies

- `spine.agents` – provides agent interfaces for decision-making within workflow phases
- `spine.mcp` – supplies multi-channel protocol capabilities for external communications
- `spine.persistence` – handles state serialization and durable storage
- `spine.phases` – defines phase lifecycle contracts and execution semantics
- `spine.work` – manages work item creation and tracking

### Depended On By

No modules in the current fragment depend on `spine.workflow`.

### Execution Flow

Workflow execution follows a registry-driven, phase-based model where `PhaseRegistry` orchestrates state transitions across specialized `SubgraphState` implementations, each corresponding to a distinct phase of the overall process.

## Module: tests.conftest.py

**Path:** `tests/conftest.py`

**Role:** Test configuration and fixtures provider. This module centralizes pytest fixtures and test utilities used across the test suite to ensure consistent setup and reduce duplication.

**Symbolcount:** 11 symbols (0 classes/interfaces, 11 functions, 0 methods)

### Key Symbols

| Symbol | Type | Description |
|--------|------|-------------|
| `async_test_config` | function | Pytest fixture providing asynchronous test configuration settings. |
| `event_loop` | function | Custom event loop fixture for async test execution. |
| `mock_openai_response` | function | Fixture that mocks OpenAI API responses for isolated testing. |
| `sample_artifact` | function | Fixture |
| `sample_prompt_request` | function | Returns a sample prompt request object for testing. |
| `sample_review_feedback` | function | Provides sample review feedback data for test cases. |
| `sample_task` | function | Generates a sample task object for use in tests. |
| `sample_work_config` | function | Supplies supplying |

## Module: tests.integration

**Responsibility:** Integration testing for end-to-end workflows and cross-module interactions.

**Location:** `tests/integration`

**Key Symbols:**
- `tests/integration/test_structured_io_flow.py`
  - `TestArtifactStoreRoundTrip`
  - `TestCriticReviewSchema`
  - `TestFeatureSliceSchema`
  - `TestFullArtifactChain`
  - `TestGapPlanSchema`
- `tests/integration/test_plan_to_implement_flow.py`
  - `TestPlanToImplementFlow`
- `tests/integration/test_prompt_assembly.py`
  - `TestPromptAssemblyIntegration`
- `tests/integration/test_mcp_integration.py`
  - `TestRealMCPCodebaseIndex`

**Dependencies (used by):**
- `spine.agents`
- `spine.mcp`
- `spine.models`
- `spine.workflow`

**Dependency Direction:** The `tests.integration` module depends on (imports/tests against) `spine.*`, but is not depended depended on by any other module listed in the fragment.

**Summary:** Contains 115 symbols (22 classes/interfaces, 32 functions, 61 methods) focused on validating integrated behavior across the ` agent framework.

## Module: tests.recall_eval

**Responsibility**
The `tests.recall_eval` module provides utilities for running and managing recall evaluation tests. It aggregates results, processes golden datasets, and computes metrics like first-hit rank.

**Key Symbols**
| Symbol | File | Type | Description |
|--------|------|------|-------------|
| `_aggregate` | tests/recall_eval/run_eval.py | function | Aggregates |
| `_by_source` | tests/recall_eval/run_eval.py | function | :\n| `_clean_topic` | tests/recall_eval/build_golden.py | function | Cleansn| `_concrete_prod_files` | tests/recall_eval/build_golden.py | function | :
| `_first_hit_rank` | tests/recall_eval/run_eval.py | function | :
| `_load_golden` | tests/recall_eval/run_eval.py | function | :
| `_main_async` | tests/recall_eval/run_eval.py | function | :
| `_print_compare` | tests/recall_eval/run_eval.py | function | :

**Dependencies**
- Depends on: `spine.agents`

**Dependents**
- None listed in the supplied fragment.

## Module: tests.test_core.py

This module is responsible for unit testing the `format_duration` functionality. It contains 13 symbols (1 class and 12 methods) following the `unittest` framework pattern.

### Key Symbols

- **TestFormatDuration** (`class`): Test class containing all test cases for duration formatting
- **test_days_only_minutes_seconds** (`method`): Tests formatting of durations with days, minutes, and seconds
- **test_exact_day** (`method`): Tests formatting of exact day durations
- **test_exact_hour** (`method`): Tests formatting of exact hour durations
- **test_exact_minute** (`method`): Tests formatting of exact minute durations
- **test_float_values** (`method`): Tests handling of float input values
- **test_hours_and_minutes** (`method`): Tests formatting of durations with hours and minutes
- **test_hours_only** (`method`): Tests formatting of hour-only durations

### Dependencies

The module imports symbols from:
- `core` (likely `src/core.py` or similar): Contains the `format_duration` function being tested

### Depended On By

No modules depend on `tests.test_core.py`. It is a leaf module in the dependency graph, serving as a Terminal test module providing no symbols to other modules.

```markdown
### `tests.test_restart.py`

**Responsibility:**
This test module verifies reset and restart functionality, UI API interactions, status color CSS, and work-related operations. It contains 5 test classes with 17 methods and 3 utility functions.

**Key Symbols:**
- **Classes (Test Suites):**
  - `TestResetStuckItems` – Tests for resetting stuck items
  - `TestRestartWork` – Tests for restart work functionality
  - `TestStatusColorCSS` – Tests for status color styling
  - `TestUIApiResetStuck` – UI API tests for resetting stuck items
  - `TestUIApiRestart` – UI API tests for restart actions
- **Functions:**
  - `queue_db` – Utility function for database queue operations
- **Methods:**
  - `mock_graph` (2 overloads) – Mock implementations for graph-related testing

**Dependencies:**
- `spine.ui` – Dependent on UI components and interfaces
- `spine.ui_api` – Relies on UI API functionality for testing
- `spine.work` – Requires work module functionality for test validation

**Depended On By:**
*This None (leaf module in test hierarchy)
```
}

## Module: tests.test_work_ordering.py

### Responsibility
This module contains unit tests for work ordering functionality within a list work feature. It serves as a test suite to validate the ordering behavior of work entries under various conditions.

### Key Symbols
- **Class**: `TestListWorkOrdering` - Main test class containing test methods
- **Methods**:
  - `_insert_entries` - Helper method for inserting test entries
  - `_make_config` - Helper method for creating test configurations
  - `test_list_work_default_order` - Tests default ordering behavior
  - `test_list_work_empty` - Tests behavior with empty work list
  - `test_list_work_filtered_order` - Tests ordering with filtered entries
  - `test_list_work_limit` - Tests ordering with limit constraints
  - `test_list_work_null_created_at` - Tests handling of null created_at fields

### Dependencies
This test module depends on:
- The work ordering implementation being tested (implied `list_work` functionality)

### Depended On By
No modules are documented as depending on this test module in the provided fragment.

### Data Flow
Test methods use helper methods (`_insert_entries`, `_make_config`) to set up test data and configurations, then validate the ordering behavior of the system under test.

## Module: tests.unit

### Responsibility
The `tests.unit` module contains the project's unit test suite, encompassing with 1748 symbols across 264 classes/interfaces, 509 functions, and 975 methods. It validates the behavior of core components in isolation using test doubles and mock implementations.

### Key Symbols
| File | Symbol | Type | Purpose |
|------|--------|------|---------|
| `tests/unit/test_explore_summarise_split.py` | `FakeBadRequest` | class | Mock class for simulating bad request errors in explore/summarise tests |
| `tests/unit/test_trace_019e77a7_fixes.py` | `FakeModel` | class | Test double representing a model for trace correction tests |
| `tests/unit/test_context_editing.py` | `FakeRequest` | class | Mock request object for context editing unit tests |
| `tests/unit/test_context_integration.py` | `FakeRequest` | class | Mock request object for context integration tests |

### Dependencies
The `tests.unit` module depends on:
- `spine.agents`\n- `spine.critic

- `spine.git`\n- `spine.models`
- `spine.phases`
- `spine.project`
- `spine.ui_api`
- `spine.work`
- `spine.workflow`

This module is a leaf node in the architecture—it does not expose any symbols that other modules depend on.

## Module: spine.services

**Responsibility:**

The `spine.services` module encapsulates service-layer components responsible for audit-related operations. It provides a centralized interface for logging and querying audit events, ensuring the underlying persistence concerns from the rest of the system.

**Key Symbols:**

*   **`AuditService`** (`spine/services/audit_service.py`): The primary class within this module. It serves as the main entry point for audit functionalities.
    *   **`__init__`** (`spine/services/audit_service.py`): Constructor for initializing the `AuditService`.
    *   **`_ensure_table`** (`spine/services/audit_service.py`): A private method likely responsible for ensuring the existence of the audit log table in the database.
    *   **`_get_db`** (`spine/services/audit_service.py`): A private method for obtaining a database connection or handle.
    *   **`log_event`** (`spine/services/audit_service.py`): A public method to record a new audit event.
    *   **`query_events`** (`spine/services/audit_service.py`): A public method to retrieve or search audit events.

**Dependencies & Dependents:**

Based on the provided fragment, there are no explicit dependency edges defined for this module. Therefore, its dependencies on other internal modules and what depends on it cannot be determined from this specific data.

## Module: tests.fixtures

**Responsibility:** Test fixtures providing sample code for testing TypeScript analysis tools.

**Key Symbols:**
- `Greeter` (class) - Defined in `tests/fixtures/sample.ts`
- `Speaker` (interface) - Defined in `tests/fixtures/sample.ts`
- `greet` (function) - Defined in `tests/fixtures/sample.ts`
- `shout` (function) - Defined in `tests/fixtures/sample.ts`
- `constructor` (method) - Belongs to `Greeter` class in `tests/fixtures/sample.ts`
- `say` (method) - Belongs to `Speaker` interface in `tests/fixtures/sample.ts`

**Dependencies:** No incoming or outgoing module dependencies found in the fragment.

## Module: spine.project

**Path:** `spine/project`

**Responsibility:** Provides project-level coverage aggregation functionality.

**Key Symbols:**
- `_member_passed` *(function)* — Defined in `spine/project/aggregator.py`
- `_member_requirements` *(function)* — Defined in `spine/project/aggregator.py`
- `_normalize` *(function)* — Defined in `spine/project/aggregator.py`
- `aggregate_project_coverage` *(function)* — Defined in `spine/project/aggregator.py`

**Dependencies:**
- `spine.persistence` (depends_on)

**Depended On By:**
- *(No downstream dependencies listed in the fragment)*

## Module: scratch.test_explore_flash.py

### Responsibility

```
scratch.test_explore_flash.py
```
is a test module in the scratch area. Per the fragment it exports **3 functions** and has **no classes or methods**. The module serves as an exploration or utility script for flash-related functionality.

### Key Symbols

| Symbol | Type | Summary |
|--------|------|---------|
| [`_patched_resolve_model`](#)
| function | *(no summary provided)* |
| [`_patched_resolve_provider`](#)
| function | *(no summary provided)* |
| [`main`](#)
| function | *(no summary provided)* |

All symbols reside in the Python file located at:
```
scratch/test_explore_flash.py
```

### Dependencies

The module **depends on**:
- **`spine.agents`**

It does not appear as a dependency of any other module in the captured graph.

### Notes

As this module is located under `scratch/`, it likely represents experimental or temporary code rather than production logic. Any reliance on it should be done cautiouslyibly, given its transient nature.

## Module: alembic.env.py

### Responsibility
`alembic/env.py` serves as the Alembic migration configuration module, exposing two primary entry-point functions that orchestrate database schema migrations.

### Key Symbols
- `run_migrations_offline` *(function)*: Configures migrations to run without an active database connection, typically generating SQL scripts.
- `run_migrations_online` *(function)*: Configures migrations to run with a live database connection, applying changes directly.

### Dependencies / Depended-On By
No dependency information is available in the supplied fragment (`edges`: []).

## Module: alembic.versions

### Responsibility
The `alembic.versions` module contains Alembic database migration scripts. It defines schema upgrade and downgrade operations for database versions.

### Key Symbols
- **`upgrade`** (function) - Located in `alembic/versions/c69967ea0727_create_work_entries_table.py`. Handles applying database schema changes.
- **`downgrade`** (function) - Located in `alembic/versions/c69967ea0727_create_work_entries_table.py`. Handles reverting database schema changes.

### Dependencies
No direct dependencies on other modules are documented in the supplied fragment. The edges array indicates no explicit dependency relationships with other modules within the provided data.

---

## Module: `.observability.py

### Responsibility

The `spine.observability.py` module provides observability utilities, specifically tracing functions designed to instrument and monitor streaming operations and work execution.

### Key Symbols

- **`traced_astream`** (`function`): A function that wraps asynchronous streaming operations with tracing capabilities, enabling observability to observe and record the lifecycle of data streams.
- **`work_run_tracing`** (`function`): A function that provides tracing instrumentation for work-run contexts, allowing monitoring and debugging of work execution flows.

### Dependencies

This module is a leaf module with **no outgoing dependencies** (edges: []). It does not import or depend on other modules within the architecture. As a provider of observability functionality, it may be imported by other modules requiring tracing capabilities.

### Dependents

While this module has no declared dependents in the current architecture graph, its functions (`traced_astream`, `work_run_tracing`) are designed to be consumed by other modules that require distributed tracing or observability to expose observability features for higher-level operations.

## Module: tests.smoke

### Responsibility
The `tests.smoke` module serves as a smoke test suite, containing test functions that validate basic functionality. The module is implemented in Python and contains two primary functions used for testing purposes.

### Key Symbols
- `tests/smoke/smoke_specify_recall.py`: `_build_state` (function) - Internal helper function for constructing test state
- `tests/smoke/smoke_specify_recall.py`: `main` (function) - Entry point for test execution

### Dependencies
- No known dependencies on other modules (edges: [])

### Depended On By
- No known dependent modules

This is a standalone test module with no documented inter-module dependencies.

## Module: spine.critic

### Responsibility
The `spine.critic` module is responsible for constructing critic agents used in reinforcement learning evaluation. It provides infrastructure for building and configuring critic models that assess policy quality.

### Key Symbols
- `build_critic_agent` (function, `spine/critic/agent.py`): Factory function for instantiating critic agents with specified configurations and model architectures.

### Dependencies
No dependencies on other modules are documented in the supplied fragment.

### Dependents
No modules that depend on `spine.critic` are documented in the supplied fragment.

## Module: spine.log.py

### Responsibility
The `spine.log.py` module is responsible for configuring logging settings. It provides a single function to set up logging configuration for the application.

### Key Symbols
- **`configure_logging`** (function): The primary function in this module used to configure logging parameters. Defined in `spine/log.py`.

### Dependencies and Interactions
- This module contains no classes or methods.
- The fragment indicates no inter-module dependencies (`edges: []`), suggesting this module operates independently or its dependencies are not explicitly mapped in the provided data.

---

## Module: src.utils

### Responsibility
The `src.utils` module provides utility functions that support common operations across the application. It contains helper functions that can be reused by multiple components without introducing tight coupling.

### Key Symbols
- **`format_duration`** *(function)*: Located in `src/utils/helpers.py`)*
  A helper function designed to format duration values, likely converting time durations into human-readable strings (e.g., "5m 30s" or "1h 20m")). No additional classes or interfaces are defined in this module.

### Dependencies
Based on the provided data, `src.utils` has no explicit dependencies on other modules. This suggests it is a leaf module that does not import or rely on symbols from other parts of the codebase.

### Dependents
No modules explicitly depend on `src.utils` within the current codebase representation. However, given its utility nature, it may be imported by other modules not listed in this fragment.

---
