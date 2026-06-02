# Architecture Map

## Module: spine.agents

**Role:** Core agent orchestration and decomposition logic (353 symbols: 63 classes/interfaces, 200 functions, 90 methods).

**Key Symbols:**
- `spine/agents/tools/ast_extract_symbol.py`: `AstExtractSymbolInput`, `AstExtractSymbolTool`
- `spine/agents/subagents.py`: `CheckItem`
- `spine/agents/tools/codebase_query.py`: `CodebaseQueryInput`, `CodebaseQueryTool`
- `spine/agents/decomposer.py`: `DecompositionResult`, `FeatureSliceSchema`
- `spine/agents/tools/read_edit_lint.py`: `FindReplaceEdit`

**Dependencies:**
- `spine.mcp`
- `spine.models`
- `spine.work`
- `spine.workflow`

**Depended On By:** None explicitly listed in this fragment.

## Module: spine.work

**Responsibility:** The `spine.work` module orchestrates repository analysis, onboarding workflows, and worker-based processing loops. It serves as the central coordination layer for transforming code analysis into actionable architectural insights.

**Key Symbols:**

- `DependencyEdge`, `ModuleBoundary`, `PatternFinding`, `RepoManifest` (classes in `spine/work/onboarding/manifest.py`) — Core data structures representing repository boundaries, dependencies, and analysis results.
- `OnboardingGraphState` (class in `spine/work/onboarding/onboarding_state.py`) — Manages state during repository onboarding processes.
- `RalphLoopWorker` (class in `spine/work/ralph_worker.py`) — Implements a worker loop for continuous processing tasks.
- `ReadRepoManifestTool` (class in `spine/work/onboarding/synthesis_tools.py`) — Tool for reading and synthesizing repository manifests.
- `RepoAnalyzer` (class in `spine/work/onboarding/analyzer.py`) — Main repositories to extract architectural patterns.

**Dependencies:**

`spine.work` depends on:
- `spine.agents`n- `spine.git`
- `spine.mcp`
- `spine.models`
- `spine.persistence`
- `spine.ui`
- `spine.workflow`

**Dependents:**

No incoming edges in the provided fragment, indicating this module is not depended on by other modules in the current view.

## Module: spine.workflow

The `spine.workflow` module (1 at `spine/workflow`) is responsible for orchestrating workflow state management and phase coordination within the task execution system. It contains 188 symbols total, including 12 classes/interfaces and 162 functions.

### Key Symbols

**State Management Classes (`spine/workflow/subgraph_state.py`):**
- `BaseSubgraphState` - Base class for subgraph state implementations
- `CriticSubgraphState` - State management for critic evaluation workflows
- `ExplorationSubgraphState` - State handling for exploration-phase workflows
- `GapPlanSubgraphState` - State coordination for gap planning operations
- `PlanSubgraphState` - Planning phase state management
- `ImplementSubgraphState` - Implementation phase state handling

**Registry Classes (`spine/workflow/registry.py`):**
- `PhaseDefinition` - Defines phase configuration and behavior
- `PhaseRegistry` - Central registry for phase management and lookup

### Dependencies

The `spine.workflow` module depends on:
- `spine.agents` - For agent integration in workflow execution
- `spine.mcp` - For tool and capability discovery
- `spine.persistence` - For state persistence operations
- `spine.phases` - For phase definition references
- `spine.work` - For work item integration

### Downstream Consumers

Other modules depend on `spine.workflow` for workflow orchestration and state management services.

## Module: spine.persistence

The `spine.persistence` module provides persistent storage abstractions for spine
base operations.

### Responsibility

This module defines store classes that encapsulate persistence concerns,
allowing other components to interact with data through consistent interfaces
without managing storage details directly.

### Key Symbols

| Symbol | Type | File | Summary |
|--------|------|------|---------|
| `Store | class | spine/persistence/artifacts.py | Manages storage and retrieval of artifacts. |
| CheckpointStore | class | spine/persistence/checkpoint.py | Manages checkpoint data persistence. |
| ProjectStore | class | spine/persistence/project_store.py | Handles project-level storage operations. |
| VectorStore | class | spine/persistence/vector_store.py | Provides vector-based data storage. |
| `__init__` (ArtifactStore) | method | spine/persistence/artifacts.py | Initializes the artifact store. |
| `__init__` (CheckpointStore) | method | spine/persistence/checkpoint.py | Initializes the checkpoint store. |
| `__init__` (ProjectStore) | method | spine/persistence/project_store.py | Initializes the project store. |
| `__init__` (VectorStore) | method | spine/persistence/vector_store.py | Initializes the vector store. |

### Dependencies

No upstream or downstream module dependencies are defined in the current fragment.

## Module: spine.models

### Responsibility

`spine.models` defines core domain data structures and enumerations used across the spine application. It centralizes type definitions for artifacts, workflow phases, feature slices, fix instructions, gap plans, and project specifications. The module supports consistent modeling of phases, reviews, and state transitions.

### Key Symbols

| Symbol | File | Type | Description
|--------|------|------|----------|
| Artifact | `spine/models/types.py` |  | Core type definitions including:\n| CriticReview | `spine/models/types.py` | class | Represents a review provided by a critic
| FeatureSlice | `spine/models/types.py` | class | Defines a segment of features to be processed
| FixInstruction | `spine/models/types.py` | class | Describes how to fix a given issue
| GapPlan | `spine/models/types.py` | class | Represents a plan to address gaps
| PhaseName | `spine/models/enums.py` | class | Enumeration of valid phase names
| PhaseResult | `spine/models/state.py` | class | Captures |
| ProjectSpec | `spine/models/types.py` | class | Specification for a project

### Dependencies

This module currently has no declared outgoing dependencies (edges) within the fragment.

## Module: spine.ui

**Responsibility:** The `spine.ui` module provides a web-based user interface layer, encompassing with key symbols split across multiple files in the `spine/ui/` directory.

**Key Symbols:**

*   **Classes/Interfaces (2):**
    *   `Event` (class): Defined in `spine/ui/ws_bus.py`.
    *   `WSEventBus` (class): Defined in `spine/ui/ws_bus.py`.
*   **Functions (66):** Core logic functions including `_audit_log` (`spine/ui/app.py`), `_config` (`spine/ui/app.py`), `_dashboard` (`spine/ui/app.py`), and `_execution_duration` (`spine/ui/_pages/work_detail.py`).
*   **Methods (10):** Primary methods such as `__init__` (`spine/ui/ws_bus.py`).

**Dependencies & Dependents:**

*   **Depends On:** `spine.workflowworkflow`
*   **(Implied) Depended On By:** Other parts of the application framework that require a UI layer to interact with backend workflows.

## Module: spine.ui_api

### Responsibility
The `spine.ui_api` module provides the public interface for UI-driven interactions with Spine's core domain logic. It orchestrates project, work, and workflow operations while abstracting persistence concerns through the `spine.persistence` layer. The module centralizes UI-specific job lifecycle management, artifact store resolution, and workflow discovery.

### Key Symbols
- **Class:** [`UIApi`](spine/ui_api/api.py) - Primary entry point for UI-bound API operations.
- **Key Methods:**
  - `__init__` - Initializes the API with dependencies on persistence and domain services.
  - `_active_jobs` - Manages and retrieves currently running or active jobs.
  - `_artifact_store_for` - Resolves the appropriate artifact storage backend.
  - `_build_active_job` - Constructs job representations for UI consumption.
  - `_discover` - Discovers available workflows or projects.
  - `_finalise_ghost_entries` - Handles cleanup of placeholder/null entries.
  - `_mark_running` - Updates job state markers.

### Dependencies
The module **depends on**:
- `spine.persistence` - For data access and job storage.
- `spine.project` - For project-related operations.
- `spine.work` - For work and job management.
- `spine.workflow` - For workflow discovery and execution logic.

This module is typically invoked by UI layers or external integrators who need a stable, high-level API to interact with the Spine backend.

## Module: spine.git

**Responsibility**: The `spine.git` module orchestrates Git workspace operations and manages isolated worktree sandboxes for spine workflow execution. `SpineGitOrchestrator` coordinates high-level git commands and fixture resolution, while `WorktreeSandbox` provides isolated git worktree environments.

**Key Symbols**:
- `SpineGitOrchestrator` *(class)* — Main orchestrator for git operations [spine/git/orchestrator.py]
- `WorktreeSandbox` *(class)* — Isolated git worktree manager [spine/git/sandbox.py]
- Methods: `__init__`, `_check_phase_prerequisites`, `_execute_shell`, `_resolve_validation_command` *(SpineGitOrchestrator)*
- Methods: `__init__`, `abort` *(WorktreeSandbox)*

**Dependencies**:
- Depends `spine.git` → `spine.models`
-   `spine.git` → `spine.workflow`

**Depended On By**: No modules listed as depending on `spine.git` in the provided fragment.

## Module: spine.phases

**Responsibility:** The `spine.phases` module orchestrates high-level workflow phases, coordinating plan validation, critic evaluation, and wave computation through a set of focused utility functions.

**Key Symbols:**

- From `spine/phases/critic.py`:
  - `_build_critic_agent()`
  - `_has_cycle()`
  - `_load_plan_json()`
  - `_validate_plan_structure()`
  - `call_critic()`
- From `spine/phases/plan.py`:
  - `_compute_waves()`
  - `_read_plan_json()`
- From `spine/phases/specify.py`:
  - `_early_commitment()`

**Dependencies:**

The module depends on:
- `spine.critic`
```
- `spine.workflow`n

**Dependents:**

No directly consuming modules are specified in this fragment.

---

## Module: spine.phases

**Responsibility:** The `spine.phases` module orchestrates high-level workflow phases, coordinating plan validation, critic evaluation, and wave computation through a set of focused utility functions.

**Key Symbols:**

- From `spine/phases/critic.py`:
  - `_build_critic_agent()`
  - `_has_cycle()`
  - `_load_plan_json()`
  - `_validate_plan_structure()`
  - `call_critic()`
- From `spine/phases/plan.py`:
  - `_compute_waves()`
  - `_read_plan_json()`
- From `spine/phases/specify.py`:
  - `_early_commitment()`

**Dependencies:**

The module depends on:
- `spine.critic`
- `spine.workflow`

**Dependents:**

No directly consuming modules are specified in this fragment.

---

## Module: spine.config.py

### Responsibility
The `spine.config.py` module manages configuration data structures and initialization logic for the spine runtime. It defines configuration classes (`ConvergenceConfig`, `SpineConfig`, `TokenCompactionConfig`) and provides helper functions for discovering workspace roots, loading environment files, and parsing convergence settings.

### Key Symbols
- **Classes/Interfaces**:
  - `ConvergenceConfig` (class): Holds convergence-related configuration parameters.
  - `SpineConfig` (class): Main configuration container for the spine runtime.
  - `TokenCompactionConfig` (class): Encapsulates token compaction settings.

- **Functions**:
  - `_disable_global_tracing()`: Disables tracing at the global level.
  - `_load_dotenv()`: Loads environment variables from `.env` files.
  - `_parse_convergence_config()`: Parses convergence configuration from raw input.

- **Methods**:
  - `_find_workspace_root()`: Discovers the root directory of the current workspace.
  - `_lookup_provider_by_name()`: Resolves a provider instance by its name.
  - (Additional methods may exist but are not detailed in the fragment.)

### Dependencies
This module does not explicitly depend on other modules within the spine bricks-js-sdk repository, as indicated by the empty `edges` array in the fragment.

### Depended On By
No modules are listed as depending on `spine.config.py` in the supplied fragment.

## Module: `.exceptions.py

### Responsibility
The `spine.exceptions.py` module defines custom exception classes used throughout the Spine framework to signal various error conditions. These exceptions serve as standardized error types for handling failures in agents, configuration, git orchestration, merging, and other critical operations.

### Key Symbols
| Symbol | Type | Description |
|--------|------|-------------|
| `AgentUnavailableError` | Class | Signals when an agent becomes unavailable during execution. |
| `ConfigurationError` | Class | Indicates issues with system or component configuration. |
| `CriticError` | Class | Used to signal errors from critique `Critic` components. |
| `CriticalContractFailure` | Class | Denotes a severe breakdown in contract-based interactions. |
| `GitOrchestratorError` | Class | Exception for errors in git-related operations managed by the orchestrator. |
| `MaxRetriesExceeded` | Class | Thrown when a retryable operation surpasses its allowed retry limit. |
| `MergeError` | Class | Occurs during conflicts or issues in merging processes.
| `PromptRequestError` | Class | Signals problems related to prompt request handling. |

### Dependencies
As an exception base module, `spine.exceptions.py` does not import from other application modules. However, its exceptions are **consum and raised by multiple modules** across the system:

- **Consumers**: All modules that handle errors from `Agent`s, `GitOrchestrator`, `Contract` interactions, and configuration validators import these symbols to throw standardized exceptions.
- **No upstream Dependencies**: This module does not depend on other internal modules for its functionality; it defines foundational types.

### Data Flow
Exceptions defined here propagate upward through the call stack. Modules that raise these exceptions rely on higher-level error handlers or top-level scripts to catch and process them, enabling consistent error management across the Spine runtime.

*Note: The current fragment lists 17 symbols (13 classes/interfaces including the 8 noted above plus `SpineError`, `ContractError`, `InvalidError`, `RetryableError`, `GitError`, `MergeConflictError`, `NoValidPromptError`, and `BaseError`), but only 8 are explicitly named in `key_symbols`. The remaining symbols follow similar patterns of use as described.*

## Module: spine.mcp

### Responsibility

The `spine.mcp` module provides a client-side implementation for managing Model Context Protocol (MCP) interactions within the spine Spine framework. It encapsulates utilities and functions required for connecting to MCP servers, processing tool results, and preparing requests for execution. This module operates as a support layer, enabling lower-level communication protocols while exposing high-level interfaces for tool use.

### Key Symbols

All symbols defined in this module reside in [`spine/mcp/client.py`](spine/mcp/client.py). They are Python functions designed for internal support rather than direct external invocation.

| Symbol | Description |
|--------|-------------|
| `_cap_result` | Captures and structures the final result before returning it from an MCP operation. |
| `_convert_server_config` | Transforms raw server configuration into a format suitable for internal client usage. |
| `_get_client` | Retrieves or instantiatesates a configured MCP client instance, potentially managing its lifecycle. |
| `_line_starts_with_excluded_path` | Checks if a given line begins with a path that should be excluded from processing. |
| `_namespace_tool` | Assigns |

## Module: spine.cli

**Responsibility:** CLI command implementation for the.

**Key Symbols:**
- `export`, `index`, `init`, `list_cmd`, `main`, `project`, `project_add`, `project_create`: Functions implementing CLI commands.

**Dependencies:** `spine.cliensely on `spine.agents`, `spine.models`, `spine.persistence`, `spine.project`, `spine.work`, and `spine.workflow`.

**Deped Upon By:** None (leaf module).

## Module: spine.project

### Responsibility 

`spine.project` orchestrates project-level coverage aggregation. It consumes persistence data, applies consolidation\nmember requirement rules, normalizes findings, and produces consolidated aggregate project coverage reports.

### Key Symbols 

| Symbol | Location | Type | Summary |
|--------|----------|------|---------|
| `_member_passed` | `spine/project/aggregator.py` | function | |
| `_member_requirements` | `spine/project/aggregator.py` | function | |
| `_normalize` | `spine/project/aggregator.py` | function | |
| `aggregate_project_coverage` | `spine/project/aggregator.py` | function | |

### Dependencies 

* **Depends on:** [`spine.persistence`](#module-spine.persistence)

### Dependents 

* *No explicit dependents listed in the fragment.*

## Module: spine.services

### Responsibility

The `spine.services` module provides audit logging capabilities for the Spine framework. It encapsulates all functionality related to recording, storing, and querying audit events through a unified service interface.

### Key Symbols

| Symbol | Type | Location | Summary |
|--------|------|----------|---------|
| `AuditService` | Class | `spine/services/audit_service.py` | Main service class for audit logging operations |
| `__init__` | Method | `spine/services/audit_service.py` | Initializes the audit service |
| `_ensure_table` | Method | `spine/services/audit_service.py` | Internal method for database table setup |
| `_get_db` | Method | `spine/services/audit_service.py` | Internal method for database connection management |
| `log_event` | Method | `spine/services/audit_service.py` | Records audit events to the database |
| `query_events` | Method | `spine/services/audit_service.py` | Queries and retrieves stored audit events |

### Dependencies

*No incoming or outgoing dependencies documented in this module's entry*

### Module: spine

**`tests.unit`** (`tests/unit`)

#### Responsibility
The `tests.unit` module contains unit tests for the codebase. It comprises 1786 symbols (267 classes/interfaces, 527 functions, 992 methods). Key test files include:

- `test_explore_summarise_split.py`n- `test_trace_019e77a7_fixes.py`
- `test_context_editing.py`
- `test_context_integration.py`

#### Key Symbols

| Symbol | Type | File Path |
|--------|------|-----------|
| `FakeBadRequest` | class | `tests/unit/test_explore_summarise_split.py` |
| `FakeModel` | class | `tests/unit/test_trace_019e77a7_fixes.py` |
| `FakeRequest` | class | `tests/unit/test_context_editing.py`, `tests/unit/test_context_integration.py` |

#### Dependencies

The `tests.unit` module depends on the following modules:

- `spine.agents`
- `spine.critic`
- `spine.git`
- `spine.models`
- `spine.phases`
- `spine.project`
- `spine.ui_api`
- `spine.work`
- `spine.workflow`

## Module: tests.integration

The `tests.integration` module contains integration tests verifying end-to-end functionality across multiple ` components. It is located at `tests/integration` and encompasses 115 symbols: 22 classes/interfaces, 32 functions, and 61 methods.

### Responsibility
Integration testing layer that validates cross-module workflows, data flows, and schema consistency. Key test classes verify artifact store round-trips, schema compliance (eReview, FeatureSlice, GapPlan), prompt assembly, and MCP-based indexing.

### Key Symbols
| Symbol | File | Type | Description |
|--------|------|------|-------------|
| TestArtifactStoreRoundTrip | tests/integration/test_structured_io_flow.py | class | Validates artifact persistence and retrieval. |
| TestCriticReviewSchema | tests/integration/test_structured_io_flow.py | class | Tests critic review schema structure. |
| TestFeatureSliceSchema | tests/integration/test_structured_io_flow.py | class | Tests feature slice schema structure. |
| TestFullArtifactChain | tests/integration/test_structured_io_flow.py | class | End-to-end artifact chain validation. |
| TestGapPlanSchema | tests/integration/test_structured_io_flow.py | class | Tests gap plan schema structure. |
| TestPlanToImplementFlow | tests/integration/test_plan_to_implement_flow.py | class | Validates plan-to-implement workflow. |
| TestPromptAssemblyIntegration | tests/integration/test_prompt_assembly.py | class | Integration test for prompt assembly. |
| TestRealMCPCodebaseIndex | tests/integration/test_mcp_integration.py | class | Tests real MCP codebase indexing. |

### Dependencies
- Depends on: `spine.agents`, `spine.mcp`, `spine.models`, `spine.workflow`
- Is depended on by: None (terminal test module)

### Execution Paths
1. Artifact I/O flow: `TestArtifactStoreRoundTrip` → `TestFullArtifactChain`
2. Schema validation flow: Individual schema test classes validate models from `spine.models`
3. Workflow integration: `TestPlanToImplementFlow` exercises logic from `spine.workflow`
4. Prompt integration: `TestPromptAssemblyIntegration` uses components from `spine.agents`
5. MCP indexing: `TestRealMCPCodebaseIndex` validates `spine.mcp` functionality

No inbound dependencies; serves as the verification layer validating the integrated behavior of dependent modules.

## Module: tests.conftest.py

**Path:** `tests/conftest.py`

**Responsibility:** Provides pytest fixtures and configuration utilities for test setup, including async test configuration, event loop management, and sample data generation for artifacts, prompts, task reviews, and work configurations.

**Key Symbols:**
- `async_test_config` *(function)*: Configures asynchronous testing environments.
- `event_loop` *(function)*: Manages pytest event loop setup for async tests.
- `mock_openai_response` *(function)*: Mocks generates mocked y mock responses for OpenAI API interactions.
- `sample_artifact` *(function)*: Produces test artifact data fixtures.
- `sample_prompt_request` *(function)*: Creates sample prompt request objects for testing.
- `sample_review_feedback` *(function)*: Generates sample feedback on reviewed tasks.
- `sample_task` *(function)*: Suppliesates task objects for test scenarios.
- `sample_work_config` *(function)*: Supplies configuration objects for work-related tests.

**Dependencies & Usage:**
No upstream dependencies detected (edges list is empty). Consumed indirectly by test modules importing fixtures from this conftest.

## Module: tests.fixtures

- **Path**: `tests/fixtures`
- **Responsibility**: Test fixtures providing sample code for testing purposes
- **Key Symbols**:
  - `Greeter` (class) - defined in `tests/fixtures/sample.ts`
  - `Speaker` (interface) - defined in `tests/fixtures/sample.ts`
  - `constructor` (method) - part of `Greeter` class
  - `greet` (function) - standalone greeting function
  - `say` (method) - instance method
  - `shout` (function) - utility function
- **Dependencies**: No dependencies on other modules
- **Usage**: No other modules depend on tests.fixtures

## Module: tests.test_core.py

- **Responsibility:** `tests/test_core.py` provides unit tests for duration formatting functionality, specifically validating the `format_duration` function with various input scenarios.
- **Key Symbols:**
  - `TestFormatDuration` (class) — Test container for duration formatting test cases.
  - `test_days_only_minutes_seconds`, `test_exact_day`, `test_exact_hour`, `test_exact_minute`, `test_float_values`, `test_hours_and_minutes`, `test_hours_only` (methods) — Individual test methods covering edge cases and typical usage.
- **Dependencies / Depended On:**
  - No explicit module dependencies or reverse dependencies are listed in this fragment; the test module likely imports and tests an internal duration formatting function (e.g., from a `core` module).

## Module: tests.test_restart.py

### Responsibility

The `tests/test_restart.py` module is responsible for testing restart-related functionality, including:

- Reset of stuck items
- Restarting work processes
- Status color CSS handling
- UI API reset and restart operations

### Key Symbols

**Classes (5):**
- `TestResetStuckItems` – Test class for resetting stuck items
- `TestRestartWork` – Test class for restart work functionality
- `TestStatusColorCSS` – Test class for status color CSS
- `TestUIApiResetStuck` – Test class for UI API reset stuck operations
- `TestUIApiRestart` – Test class for UI API restart operations

**Functions (3):**
- `mock_graph()` – Factory/utilityven from
- `queue_db` – Database queue database

**Methods (17):**
- `mock_graph()` – Mock graph factory method

### Dependencies

This module depends on:
- `spine.ui`
- `spine.ui_api`
- `spine.work`

### Depended On By

No modules are listed as depending on `tests.test_restart.py`.

## Module: tests.test_work_ordering.py

**Responsibility**


This module contains unit tests for verifying work ordering behavior in list operations. It focuses on testing default ordering, empty result handling, filtered orderings with limit constraints, and edge cases such as null values in ordering fields.


**Key Symbols**

* **TestListWorkOrdering** (class): Test container housing all work ordering test cases.
* **_insert_entries** (method): Helper for setting test data entries.
* **_make_config** (method): Factory method for constructing test configurations.
* **test_list_work_default_order** (method): Validates that list operations return results in the expected default order.
* **test_list_work_empty** (method): Confirms correct handling of empty query results.
* **test_list_work_filtered_order** (method): Tests ordering behavior when additional filters are applied.
* **test_list_work_limit** (method): Ensures limit clauses correctly constrain result sets without breaking offset calculations.
* **test_list_work_null_created_at** (method): Checks proper ordering when created_at values are null.


**Dependencies & Impact**

*No explicit incoming or outgoing module dependencies are defined in the provided fragment. The edge data indicates no direct relationships or dependencies with other modules.*

* *File Path*: `tests/test_work_ordering.py`
* *Language*: Python
* *Symbol Count*: 9 total (1 class/interface, 0 functions, 8 methods)'}]<|observation_ consequence|>[ERROR: Error parsing the generated content: Parsing latexN unexpected character '''. Detected||Got result: [2,

## Module: tests.recall_eval

**Responsibility:** Evaluates
The `tests.recall_eval` module orchestrates recall evaluation for synthetic data. It constructs golden datasets for retrieval tasks and executes end-to-end evaluation pipelines to measure retrieval effectiveness.

**Key Symbols:**

| Symbol | File | Type |
|--------|------|------|
| `_aggregate` | `tests/recall_eval/run_eval.py` | Function |
| `_by_source` | `tests/recall_eval/run_eval.py` | Function |
| `_clean_topic` | `tests/recall_eval/build_golden.py` | Function |
| `_concrete_prod_files` | `tests/recall_eval/build_golden.py` | Function |
| `_first_hit_rank` | `tests/recall_eval/run_eval.py` | Function |
| `_load_golden` | `tests/recall_eval/run_eval.py` | Function |
| `_main_async` | `tests/recall_eval/run_eval.py` | Function |
| `_print_compare` | `tests/recall_eval/run_eval.py` | Function |

**Dependencies:**

*   on `spine.agents`: Used for agent-based retrieval evaluation infrastructure.

**Dependents:**

* No other modules depend on `tests.recall_eval` according to the current dependency graph.

## Module: tests.smoke

- **Responsibility**: Smoke testing framework for validating recall configuration; provides state construction and entry point for verification workflows.
- **Key Symbols**: 
  - `main` (`tests/smoke/smoke_specify_recall.py`) – Primary entry function for smoke test execution.
  - `_build_state` (`tests/smoke/smoke_specify_recall.py`) – Internal helper for constructing test state.
- **Dependencies**: None (module is self-contained; edge list is empty).
- **Depended On By**: None specified.

Module: src.utils
===================

Responsibility
--------------
The `src.utils` module provides utility functions to support common operations across the application. It is located at `src/utils` and contains helper logic that can be reused by other modules.

Key Symbols
-----------
- **format_duration** *(function)* Defined in `src/utils/helpers.py`. Formats duration values for consistent display or logging purposes.

Dependencies & Dependents
--------------------------
No module dependencies or dependent modules are recorded for `src.utils` in the current analysis.

## Module: alembic.env.py

### Responsibility

The `alembic/env.py` module serves as the Alembic migration environment configuration. It defines how database migrations are executed—either in offline mode (generating SQL scripts without connecting to the database) or online mode (applying migrations directly to a live database).

### Key Symbols

| Symbol | Type | Description |
|--------|------|-------------|
| `run_migrations_offline` | function | Configures and runs migrations in 'offline' mode, where Alembic outputs SQL to a file or stdout without executing it against a database connection. |
| `run_migrations_online` | function | Configures and runs migrations in 'online' mode, establishing a connection to the database and applying migrations directly. |

### Dependencies

No incoming or outgoing module dependencies are documented in the provided fragment.

### Execution Flow

1. **Offline Path:**
   - `run_migrations_offline` is invoked.
   - Alembic|lavor` context is configured without a live database connection.
   - Migration scripts are generated as raw SQL and written to output (file/stdout).

2. **Online Path:**
   - `run_migrations_online` is invoked.
   - An engine is created using configuration from `alembic.ini`.
   - A connection is established, and migration commands are executed against the database.

Both functions typically share common setup logic (e.g., `context.configure`) but differ in how they supply the connection/informal to Alembic's migration context.

## Module: alembic.versions

### Responsibility
The `alembic.versions` module contains Alembic database migration scripts. It is responsible for defining schema evolution operations, specifically the `upgrade` and `downgrade` functions that modify the database schema when migrations are applied or rolled back.

### Key Symbols
- `upgrade`: Function in `alembic/versions/c69967ea0727_create_work_entries_table.py` that applies the migration to create the work entries table.
- `downgrade`: Function in `alembic/versions/c69967ea0727_create_work_entries_table.py` that reverts the migration by dropping the work entries table.

### Dependencies
This module is a on by:
- The Alembic command-line tool, which loads and executes these migration functions during `alembic upgrade` and `alembic downgrade` operations.

No explicit dependencies are shown in the provided fragment, indicating this module operates as a terminal node in the dependency graph.

## Module: scratch.test_explore_flash.py

**Path:** [`scratch/test_explore_flash.py`scratch/test_explore_flash.py)  
**Symbols:** 3 functions, 0 classes/interfaces, 0 methods

### Responsibility
This test module that patches resolution helpers and orchestrates a main workflow. Intended used for Flash exploration or testing.

### Key Symbols
| Symbol | Type | Summary |
|--------|------|---------|
| `_patched_resolve_model` | function | No summary available |
| `_patched_resolve_provider` | function | No summary available |
| `main` | function | No summary available |

### Dependencies & Dependents
- **Depends on:** `spine.agents`eady (src: scratch.test_explore_flash.py ↔ dst: spine.agents, kind: depends_on)**
- **Depend Depended on by:** None known from supplied fragment.

---

> The module contains three functions (`_patched_resolve_model`, `_patched_resolve_provider`, `main`) and shows a single hard dependency edge toward `spine.agents` originating from `scratch.test_explore_flash.py`. No downstream dependents were detected in the provided architecture data.

## Module: spine  

**File:** `spine/observability.py`

### Responsibility
This module provides observability utilities, likely focused on tracing and monitoring execution flows within the Spine framework. It contains two core functions designed to instrument and trace operations, particularly those involving streaming or asynchronous work runs.

### Key Symbols
- `traced_astream`: Function likely responsible for tracing asynchronous streaming operations.
- `work_run_tracing`: Function likely used to add tracing instrumentation around work execution.

### Dependencies
Currently, there are no explicit dependencies listed in the provided data for this module.

### Depended On By
There are no modules listed as depending on `spine.observability.py` in the provided dataset.
