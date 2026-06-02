# Architecture Map

## Module: spine.agents

**Responsibility:** spine
`spine.agents` orchestrates agent-based workflows for code understanding, decomposition, and transformation. It provides tools and sub-agents for AST symbol extraction, codebase querying, and structured editing, plus decomposer logic for feature slicing.

**Key Symbols:**

- **AST Extraction:** `AstExtractSymbolInput`, `AstExtractSymbolTool` (spine/agents/tools/ast_extract_symbol.py)
- **Codebase Query:** `CodebaseQueryInput`, `CodebaseQueryTool` (spine/agents/tools/codebase_query.py)
- **Editing:** `FindReplaceEdit` (spine/agents/tools/read_edit_lint.py)
- **Decomposition:** `DecompositionResult`, `FeatureSliceSchema` (spine/agents/decomposer.py)
- **Subagents:** `CheckItem` (spine/agents/subagents.py)

**Dependencies:**
- `spine.mcp`n- `spine.models`
- `spine.work`
- `spine.workflow`

**Depended On By:** *(No inbound edges in fragment)*

**Primary Data Flow:**
Agent requests → MCP protocol → codebase queries → AST extraction → decomposition → structured output (schemas, edits). Output flows back through workflow orchestration layers.

## Module: spine.workflow

The `spine.workflow` module is responsible for orchestrating multi-phase workflows, managing state transitions, and coordinating the execution of modular workflow components. It serves as the central coordination layer that integrates agent behaviors, tools, phases, and persistent state management.

### Responsibilities

- **Workflow Orchestration**: Manages the lifecycle of multi-phase workflows from planning to completion.
- **State Management**: Handles subgraph state transitions and maintains workflow context through `BaseSubgraphState` and related specialized state classes.
- **Phase Registry Management**: Provides centralized registration and lookup of workflow phases via `PhaseRegistry` and `PhaseDefinition`.
- **Cross-Cutting Concerns**: Integrates agent systems, tools (MCP), persistence, and work phases into cohesive workflows.

### Key Symbols

#### Core State Classes (`spine/workflow/subgraph_state.py`)

- `BaseSubgraphState`: Foundational state management for workflow subgraphs
- `CriticSubgraphState`: Specialized state for evaluation/critique phases
- `ExplorationSubgraphState`: State handling exploration workflows
- `GapPlanSubgraphState`: Manages gap planning phase state
- `ImplementSubgraphState`: Handles implementation phase state management
- `PlanSubgraphState`: Coordinates planning phase state transitions

#### Registry Classes (`spine/workflow/registry.py`)

- `PhaseDefinition`: Defines registration metadata and configuration for workflow phases
- `PhaseRegistry`: Central registry for phase discovery and lifecycle management

### Dependencies

The `spine.workflow` module depends on the following modules:

- `spine.agents`: Provides agent interfaces and behaviors for workflow execution
- `spine.mcp`: Supplies tool integrations and MCP (Machine Communication Protocol) capabilities
- `spine.persistence`: Offers state persistence and data storage mechanisms
- `spine.phases`: Delivers modular phase implementations for workflow composition
- `spine.work`: Integrates with work context and task management systems

### Reverse Dependencies

Other modules depend on `spine.workflow` for:

- Workflow orchestration and phase management capabilities
- Centralized state and registry infrastructure
- Coordination between agents, tools, and persistent storage

## Module: spine.work

**Path:** `spine/work`

**Symbols:** 171 total (15 classes/interfaces, 108 functions, 48 methods)

### Responsibility

The `spine.work` module orchestrates repository onboarding
analysis, dependency tracking, and workflow execution within the Spine system.
It serves as the central work interface for processing code repositories,
managing module boundaries, and driving synthesis tasks through specialized
components like the RalphLoopWorker.

### Key Symbols

| Symbol | Type | File | Purpose |
|--------|------|------|---------|
| `RalphLoopWorker` | class | `spine/work/ralph_worker.py` | Main worker loop for executing synthesis tasks |
| `RepoManifest` | class | `spine/work/onboarding/manifest.py` | Represents the structure and dependencies of a repository |
| `OnboardingGraphState` | class | `spine/work/onboarding/onboarding_state.py` | Manages state during the repository onboarding process |
| `RepoAnalyzer` | class | `spine/work/onboarding/analyzer.py` | Analyzes repository structure and extracts patterns |
| `DependencyEdge` | class | `spine/work/onboarding/manifest.py` | Models dependency relationships between modules |
| `ModuleBoundary` | class | `spine/work/onboarding/manifest.py` | Defines boundaries of modules within a repository |
| `PatternFinding` | class | `spine/work/onboarding/manifest.py` | Captures identified patterns in repository analysis |
| `ReadRepoManifestTool` | class | `spine/work/onboarding/synthesis_tools.py` | Tool for reading and parsing repository manifests |

### Dependencies

`spine.work` depends on:
- `spine.agents` - Agent infrastructure for decision making
- `spine.git` - Git repository operations
- `spine.mcp` - Model context protocol integration
- `spine.models` - Core data models
- `spine.persistence` - Data persistence mechanisms
- `spine.ui` - User interface components
- `spine.workflow` - Workflow orchestration primitives

### Depended On By

Currently no modules are documented as depending on `spine.work` within this fragment.

### Data Flow

1. Repository analysis triggered via `RepoAnalyzer`
2. Manifest construction using `RepoManifest`, `DependencyEdge`, `ModuleBoundary`
3. State persistence via `OnboardingGraphState`
4. Synthesis task execution through `RalphLoopWorker`
5. Tool integration via `ReadRepoManifestTool`}

## Module: tests.unit

**Responsibility:**
The `tests.unit` module is responsible for providing unit test infrastructure and test cases for the codebase. It contains 1767 symbols including 264 classes/interfaces, 527 functions, and 976 methods.

**Key Symbols:**

- `FakeBadRequest` (class) – `tests/unit/test_explore_summarise_split.py`
- `FakeModel` (class) – `tests/unit/test_trace_019e77a7_fixes.py`
- `FakeRequest` (class) – `tests/unit/test_context_editing.py`n- Additional `FakeRequest` instances – `tests/unit/test_context_integration.py`

**Dependencies:**

The `tests.unit` module depends on:
- `spine.agents`
- `spine.critic`
- `spine.git`
- `spine.models`
- `spine.phases`
- `spine.project`
- `spine.ui_api`
- `spine.work`
- `spine.workflow`

**Dependency Direction:**

`tests.unit` is a consumer module; it depends on the above spinekeletal
 os packages for testing purposes, but none of those packages depend on `tests.unit`.

## Module: spine.persistence

The `spine.persistence` module provides storage abstractions for managing persistent data artifacts, checkpoints, projects, and vector representations. It defines four primary store classes that encapsulate data persistence concerns.

### Responsibility
The module is responsible for:/storage operations, providing a consistent interface for persisting and retrieving application state across different storage backends.

### Key Symbols

**Classes/Interfaces (4):**
- `ArtifactStore` (spine/persistence/artifacts.py)
- `CheckpointStore` (spine/persistence/checkpoint.py)
- `ProjectStore` (spine/persistence/project_store.py)
- `VectorStore` (spine/persistence/vector_store.py)

**Methods (35 total):**
Each class implements an `__init__` method for initialization, along with additional methods (not detailed in this fragment) that handle data operations.

### Dependencies
The module contains no documented inter-module dependencies within the provided fragment (edges array is empty). All classes follow a similar initialization pattern with `__init__` methods defined in their respective files.

### Depended Upon By
No dependent modules are documented in the provided fragment.

## Module: spine.ui

**spine.ui** (`spine/ui`) is a UI module comprising 78 symbols (2 classes/interfaces, 66 functions, 10 methods).\nb) and is responsible for providing user-facing interfaces and event-driven communication within the system.

### Key Symbols

| Symbol | Type | File | Summary |
|--------|------|------|---------|
| `Event` | class | `spine/ui/ws_bus.py` | |
| `WSEventBus` | class | `spine/ui/ws_bus.py` | |
| `__init__` | method | `spine/ui/ws_bus.py` | |
| `_audit_log` | function | `spine/ui/app.py` | |
| `_client_handler` | function | `spine/ui/ws_server.py` | |
| `_config` | function | `spine/ui/app.py` | |
| `_dashboard` | function | `spine/ui/app.py` | |
| `_execution_duration` | function | `spine/ui/_pages/work_detail.py` | |

### Dependencies

- **sp on**: `spine.workflow`n
### Dependents

_None dont appear to be explicitly listed in the fragment as depending on `spine.ui`.

## Module: tests.integration

**Path:** `tests/integration`
**Responsibility:** Provides integration-level tests that validate end-to-end workflows and cross-module interactions.

**Key Symbols:**
- `TestArtifactStoreRoundTrip` (tests/integration/test_structured_io_flow.py)
- `TestCriticReviewSchema` (tests/integration/test_structured_io_flow.py)
- `TestFeatureSliceSchema` (tests/integration/test_structured_io_flow.py)
- `TestFullArtifactChain` (tests/integration/test_structured_io_flow.py)
- `TestGapPlanSchema` (tests/integration/test_structured_io_flow.py)
- `TestPlanToImplementFlow` (tests/integration/test_plan_to_implement_flow.py)
- `TestPromptAssemblyIntegration` (tests/integration/test_prompt_assembly.py)
- `TestRealMCPCodebaseIndex` (tests/integration/test_mcp_integration.py)

**Dependencies:** This module depends on `spine.agents`, `spine.mcp`, `spine.models`, and `spine.workflow`.

**Depended On By:** No modules list this module as a dependency.

## Module: spine.models

The `spine.models` module serves as the core data modeling layer, defining domain types and structures used throughout the Spine codebase. It provides a foundation of classes, interfaces, and enums that represent key domain concepts.

### Key Symbols

**Types (spine/models/types.py):**
- `Artifact` - Represents a tangible
- `CriticReview` - Represents a critiques
- `FeatureSlice` - Represents a
n- `FixInstruction` - Represents a :
- `GapPlan` - Represents a 
- `ProjectSpec` - Represents a :

**Enums (spine/models/enums.py):**
- `PhaseName` - Enumeration for phase identifiers

**State (spine/models/state.py):**
- `PhaseResult` - Represents the result of a phase execution

### Responsibilities

This module is responsible for:
- Defining core domain models used across the application
- Providing structured data types for artifacts, reviews, and project specifications
- Enumerating standardized phase names for consistent referencing
- Modeling state and results from phase executions

### Dependencies

No direct module dependencies are specified in the current fragment. The `spine.models` module appears to be a foundational layer that other modules may depend upon for type definitions and domain models.

### Dependents

The module has no recorded dependents in the current edge data, suggesting it may serve as a leaf module in the dependency graph, or its dependent relationships are not captured in this fragment.

## Module: spine.git

### Responsibility
The `spine.git` module manages Git-based workflow orchestration and isolated worktree operations within the Spine system.

### Key Symbols
- **Classes/Interfaces:**
  - `SpineGitOrchestrator` (spine/git/orchestrator.py): Core orchestrator class
  - `WorktreeSandbox` (spine/git/sandbox.py): Manages sandbox environment class

- **Functions:**
  - `__init__` methods in both orchestrator.py and sandbox.py

- **Key Methods:**
  - `_check_phase_prerequisites`: Validates phase requirements
  - `_execute_shell`: Executes shell commands
  - `_resolve_validation_command`: Resolves validation commands
  - `abort`: Handles sandbox abort operations

### Dependencies
- **Depends on:**
  - `spine.models`
  - `spine.workflow`

### Usage
This module serves as an internal implementation layer, providing Git orchestration and sandbox management capabilities.

## Module: spine.config.py

### Responsibility
The `spine.config.py` module centralizes configuration management for the Spine runtime, handling initialization, environment loading, and provider resolution for optional integrations.

### Key Symbols
- **Classes:** `ConvergenceConfig`, `SpineConfig`, `TokenCompactionConfig`
- **Functions:** `_disable_global_tracing`, `_load_dotenv`, `_parse_convergence_config`
- **Methods:** `_find_workspace_root`, `_lookup_provider_by_name`

### Dependencies & Dependents
- **No upstream dependencies** — This module does not import or use other internal modules (edges count: 0).\n- **No downstream dependents** — No other modules in the provided fragment reference this module (edges count: 0)

### Module: spine.exceptions.py

#### Responsibility
The `spine.exceptions.py` module centralizes error definitions for the Spine framework, providing specialized exception classes for different subsystem failures. It serves as the foundational error-handling layer, enabling consistent exception types across the codebase.

#### Key Symbols
- **Agent Failures**: `AgentUnavailableError`
- **Configuration Issues**: `ConfigurationError`
- **Criticique System Errors**: `CriticiсError`
- **Contract Violations**: `CriticalContractFailure`
- **Git Operations**: `GitOrchestratorError`
`, `MergeError`
- **Retry Logic**: `MaxRetriesExceeded`
- **Prompt Processing**: `PromptRequestError`

#### Dependencies
- **Is depended on by**: All Spine modules that handle agent coordination, configuration, critique workflows, contract validation, git operations, retry mechanisms, and prompt rendering.
- **Dependencies**: None (base module with no upstream dependencies).
)

## Module: spine.phases

**Role:** Orchestration layer for workflow processing phases. Contains 14 functions (`spine/phases/*.py`) implementing discrete pipeline stages for critic evaluation, planning, and specification.

**Key Symbols:**
- `call_critic()` (critic.py): Main entry point for critic invocation
- `_build_critic_agent()`: Constructs critic instances
- `_has_cycle()`: Detects cyclic dependencies
- `_validate_plan_structure()`: Validates plan JSON schema
- `_early_commitment()`: Handles specification phase
- `_compute_waves()`: Calculates execution waves in planning
- `_load_plan_json()` / `_read_plan_json()`: JSON plan IO

**Dependencies:**/MSn on `spine.critic` (validation logic) and `spine.workflow` (orchestration context).

**Depended On by:** (none specified) — likely invoked by higher-level pipeline coordinator.

## Module: spine.cli

### Responsibility
`spine.cli` is the command-line interface layer for the spine application. It exposes a set of callable functions that constitute the CLI entrypoints, handling user interactions and orchestrating business logic to other modules.

### Key Symbols
All symbols are defined in `spine/cli/__init__.py`:

| Symbol | Type | Description |
|--------|------|-------------|
| `export` | function | Exports data or project output |
| `index` | function | Indexes project or workflow data |
| `init` | function | Initializes a new spine |-
project or configuration
| `list_cmd` | function | Lists available projects, workflows, or items |
| `main` | function | Main CLI entrypoint that dispatches to subcommands |
| `project` | function | High-level project operations dispatcher |
| `project_add` | function | Adds a project or item to a project |
| `project_create` | function | Creates a new project |

*Note: Symbol summaries were not populated in the source fragment.*

### Dependencies
`spine.cli` depends on the following modules:
- `spine.agents`: Provides agent-related functionality for CLI operations
- `spine.models`: Supplies data models used in CLI commands
- `spine.persistence`: Handles data persistence operations invoked by CLI
- `spine.project`: Core project management logic
- `spine.work`: Work item handling for CLI workflows
- `spine.workflow`: Workflow orchestration called from CLI commands

### Consumed By
The modules that depend on `spine.cli` are not detailed in the provided fragment; this section would be expanded if reverse dependency data were available.

## Module: spine.mcp

**Path:** `spine/mcp`

**Responsibility:**
The `spine.mcp` module provides a client-side implementation for MCP (Model Context Protocol) communication. It encapsulates all functionality related to connecting to MCP servers, executing tools, and processing results. This module serves as the primary interface between the spine framework and external MCP-compatible services, handling server configuration conversion, connection management, tool namespacing, and result post-processing to filter excluded paths.

**Key Symbols:**
All functions reside in `spine/mcp/client.py`:
- `_cap_result`: Wraps and processes the final result from an MCP tool execution.
- `_convert_server_config`: Transforms internal server server configuration into MCP-compatible format.
- `_get_client`: Retrieves or initializes the MCP client instance for communication.
- `_line_starts_with_excluded_path`: Checks if a log line begins with a path that should be excluded from results.
- `_namespace_tool`: Applies namespacing to tool names to avoid collisions and clarify origin.
- `_post_process_result`: Filters result content to remove lines referencing excluded paths.
- `_run_async`: Executes asynchronous operations within the MCP client context.
- `_strip_excluded_paths`: Removes sections from results that match excluded path patterns.

**Dependencies:**
The module depends on external MCP libraries for protocol handling and communicates with server implementations (referenced via edges in broader architecture). No explicit incoming edges are defined in the provided fragment, indicating it may be a terminal or leaf module in the current context.

**Depend | Done.

## Module: spine.project

### Responsibility
The `spine.project` module centralizes project-level aggregation logic. It collects and normalizesizes coverage data across an entire project, normalizing and combining inputs from lower-level modules into a unified reporting structure.

### Key Symbols
All symbols are defined in `spine/project/aggregator.py`:

- **`_member_passed`** *(function)* – Internal helper that evaluates whether a single project member satisfies passing criteria.
- **`_member_requirements`** *(function)* – Computes the set of requirements applicable to an individual project member.
 showcased in the module’s primary export.
- **`_normalize`** *(function)* – Standardizes raw project coverage data into a consistent format for downstream processing.
- **`aggregate_project_coverage`** *(function)* – Main entry point that orchestrates the aggregation of coverage metrics for the whole project.

### Dependencies
- **Depends on**: `spine.persistence`aw, persisted coverage data from the database layer.

### Consumed By
No downstream modules consume this module directly; it serves as a leaf supplier of aggregated project coverage intelligence.

## Module: tests.conftest.py

### Responsibility
The `tests.conftest.py` module provides pytest fixtures and configuration helpers for test setup. It defines utility functions that generate sample data and mock configurations used across the test suite.

### Key Symbols
| Symbol | Type | Description |
|--------|------|-------------|
| `async_test_config` | function | Pytest fixture providing async test configuration. |
| `event_loop` | function | Pytest fixture for managing the asyncio event loop in tests. |
| `mock_openai_response` | function | Fixture to mock OpenAI API responses during testing. |
| `sample_artifact` | function | Fixture generating sample artifact data for tests. |
| `sample_prompt_request` | function | Fixture providing sample prompt request data. |
| `sample_review_feedback` | function | Fixture supplying sample review feedback data. |
| `sample_task` | function | Fixture returning sample task data for test scenarios. |
| `sample_work_config` | function | Fixture yielding sample work configuration data. |

### Dependencies
- **Consumes**: No incoming dependencies identified (leaf module in the test infrastructure).  
- **Provides**: These fixtures are imported and used by test modules in the `tests/` directory (e.g., `test_*.py` files) that require pre-configured test data or async environments. |

### Data Flow
Fixturesn/a — This is a pytest fixture module; data flows from these fixtures into test functions as needed.

## Module: tests.fixtures

**Path:** `tests/fixtures`

**Responsibility:** Test fixture module providing sample classes, interfaces, and helper functions for testing purposes.

**Key Symbols:**

- **Greeter** (`tests/fixtures/sample.ts:Greeter`) — Class for greeting functionality
- **Speaker** (`tests/fixtures/sample.ts:Speaker`) — Interface defining speaker contract
- **constructor** (`tests/fixtures/sample.ts:constructor`) — Method for initializing Greeter instances
- **greet** (`tests/fixtures/sample.ts:greet`) — Function for generating greetings
- **say** (`tests/fixtures/sample.ts:say`) — Method for speech output
- **shout** (`tests/fixtures/sample.ts:shout`) — Function for shouting messages

**Dependencies:** No dependent modules found
**Dependencyed By:** No modules found

- `TestFormatDuration.test_exact_extensions   :class:` tests/test_core.py: Tests the `format_duration` function with only minutes and seconds values. - `TestFormatDuration.test_exact_day` :class:` tests/test_core.py: Tests the `format_duration` function with exactly one day. - `TestFormatDuration.test_exact_hour` :class:` tests/test_core.py: Tests the `format_duration` function with exactly one hour. - `TestFormatDuration.test_exact_minute` :class:` tests/test_core.py: Tests the `format_duration` function with exactly one minute. - `TestFormatDuration.test_float_values` :class:` tests/test_core.py: Tests the `format_duration` function with float input values. - `TestFormatDuration.test_hours_and_minutes` :class:` tests/test_core.py: Tests the `format_duration` function with hours and minutes values. - `TestFormatDuration.test_hours_only` :class:` tests/test_core.py: Tests the `format_duration` function with only hours values.

## Module: tests/lodash.py

- `TestFormatDuration` : 3 symbols (1 classes/interfaces, 0 functions, 2 methods)
- **Key symbols**:
  - `TestFormatDuration` :class:` tests/lodash.py: The test class for the `format_duration` function.
  - `test_empty_string` :class:` tests/lodash.py: Tests the `format_duration` function with an empty string input.
  - `test_invalid_hournegative_number` :class:` tests/lodash.py: Tests the `format_duration` function with a negative number.

## Module: tests.test_helpers.py

- :class:`

## Module: tests/lodash.py

- `format_duration` : 9 symbols (1 classes/interfaces, 0 functions, 8 methods)
- **Key symbols**:
  - `format_duration` :function:` src/lodash.py: Converts a duration in seconds into a human-readable string.
  - `TestFormatDuration` :class:` src/lodash.py: Test class for the `format_duration` function.
  - `TestFormatDuration.test_basic` :method:` src/lodash.py: Tests a basic case of the `format_duration` function.
  - `TestFormatDuration.test_days_only` :method:` src/lodash.py: Tests the `format_duration` function with only days.
  - `TestFormatDuration.test_hours_only` :method:` src/lodash.py: Tests the `format_duration` function with only hours.
  - `TestFormatDuration.test_minutes_only` :method:` src/lodash.py: Tests the `format_duration` function with only minutes.
  - `TestFormatDuration.test_with_seconds` :method:` src/lodash.py: Tests the `format_duration` function with seconds included.
  - `TestFormatDuration.test_zero` :method:` src/lodash.py: Tests the `format_duration` function with zero input.

## Module: tests.test_l.py

- `TestHelpers` : 3 symbols (1 classes/interfaces, 0 functions, 2 methods)
- **Key symbols**:
  - `TestHelpers` :class:` tests/test_helpers.py: Test suite for helper functions.
  - `test_capitalize` :method:` tests/test_helpers.py: Tests the `capitalize` helper function.
  - `test_merge_dicts` :method:` tests/test_helpers.py: Tests the `merge_dicts` helper function.

## Module: tests.helpersom APUs  .py

- `TestsOmgebraltion` : 6 symbols (1 classes/interfaces, 0 functions, 5 methods)
- **Key symbols**:
  - `TestsOmgebraltion` :class:` tests/testomgebraltion.py: Test suite for the `Omgebraltion` class.
  - `test_angular_momentum` :method:` tests/testomgebraltion.py: Tests the angular momentum property.
  - `test_cubic_displacement` :method:` tests/testomgebraltion.py: Tests the cubic displacement method.
  - `test_cubic_tidimum` :method:` tests/testomgebraltion.py: Tests the cubic tidimum method.
  - `test_cubic_triple_alpha` :method:` tests/testomgebraltion.py: Tests the cubic triple alpha method.
  - `test_gravitational_potential` :method:` tests/testomgebraltion.py: Tests the gravitational potential method.

## Module: tests.test_physics.py

- `TestPhysics` : 4 symbols (1 classes/interfaces, 0 functions, 3 methods)
- **Key symbols**:
  - `TestPhysics` :class:` tests/test_physics.py: Tests the `Physics` class.
  - `test_acceleration` :method:` tests/test_physics.py: Tests the acceleration calculation.
  - `test_velocity` :method:` tests/test_physics.py: Tests the velocity calculation.

## Module: tests.test_utils.py

- :class:`

## Module: physics/physics.py

- `Physics` : 4 symbols (1 classes/interfaces, 0 functions, 3 methods)
- **Key symbols**:
  - `Physics` :class:` src/physics/physics.py: Main physics class containing various physics calculations.
  - `acceleration` :method:` src/physics/physics.py: Calculates acceleration using velocity and time.
  - `velocity` :method:` src/physics/physics.py: Calculates velocity using distance and time.

## Module: utils/utils.py

- `capitalize` : 3 symbols (1 classes/interfaces, 0 functions, 2 methods)
- **Key symbols**:
  - `capitalize ` :class:` src/utils/helpers.py: Utility class for helper functions.
  - `capitalize` :method:` src/utils/helpers.py: Capitalizes the first letter of a string.
  - `merge_dicts` :method:` src/utils/helpers.py: Merges two dictionaries together.

## Module: students/student.py

- :class:`

## Module: students/managerntity.py

- `Entity` : 1 symbols (1 classes/interfaces, 0 functions, 0 methods)
- **Key symbols**:

## Module: omgebraltion.py

- `Omgebraltion` : 6 symbols (1 classes/interfaces, 0 functions, 5 methods)
- **Key symbols**:
  - `Omgebraltion` :class:` src/omgebraltion.py: Main class for performing omgebraltion calculations.
  - `angular_momentum` :method:` src/omgebraltion.py: Calculates the angular momentum.
  - `cubic_displacement` :method:` src/omgrebraltion.py: Calculates cubic displacement.
  - `cubic_tidimum` :method:` src/omgrebraltion.py: Calculates cubic tidimum.
  - `cubic_triple_alpha` :method:` src/omgrebraltion.py: Calculates cubic triple alpha.
  - `gravitational_potential` :method:` src/omgrebraltion.py: Calculates the gravitational potential.

## Module: students/student.py

- :class:`

## Module: tests.test_restart.py

**Responsibility:** Test module for restart and reset functionality, covering stuck items reset, work restart, UI API reset/stuck operations, and status color CSS.

**Key Symbols:**
- `TestResetStuckItems` (class): Test cases for resetting stuck items
- `TestRestartWork` (class): Test cases for work restart functionality
- `TestStatusColorCSS` (class): Test cases for status color CSS validation
- `TestUIApiResetStuck` (class): Test cases for UI API stuck item reset
- `TestUIApiRestart` (class): Test cases for UI API restart operations
- `mock_graph` (method): Mock graph utility used across multiple test cases
- `queue_db` (function): Test fixture for database queue setup

**Dependencies:**
- **Depends on:**
  - `spine.ui`: Core UI component testing
  - `spine.ui_api`: UI API functionality testing
  - `spine.work`: Work-related functionality testing

**Depended by:** No modules are documented as depending on this test module.

## Module: tests.test_work_ordering.py

- **Path:** `tests/test_work_ordering.py`
- **Responsibility:** Validates *No summary provided in the fragment.*

### Key Symbols
- **Class:** `TestListWorkOrdering`n- **Methods:**
  - `_insert_entries`
  - `_make_config`
  - `test_list_work_default_order`
  - `test_list_work_empty`
  - `test_list_work_filtered_order`
  - `test_list_work_limit`
  - `test_list_work_null_created_at`

### Dependencies
- **This module has no dependencies** (edges: []).

## Module: tests.recall_eval

The `tests.recall_eval` module is a test evaluation framework responsible for running recall evaluation assessments. It consists entirely of utility functions (no classes or interfaces) across two primary files.

### Key Symbols

**tests/recall_eval/run_eval.py**

*   `_aggregate()`: Aggregates evaluation results.
*   `_by_source()`: Groups results by data source.
*   `_first_hit_rank()`: Calculates the rank of the first successful hit.
*   `_load_golden()`: Loads the ground truth dataset.
*   `_main_async()`: Main asynchronous entry point for the evaluation run.
*   `_print_compare()`: Prints a comparison of results.

**tests/recall_eval/build_golden.py**

*   `_clean_topic()`: Sanitizes topic identifiers.
*   `_concrete_prod_files()`: Resolves concrete product file paths.

### Dependencies

*   **Depends on**: `spine.agents`n
### Depended On By

*   No direct dependents are listed in the provided fragment.

## Module: tests.smoke

**Responsibility**

The `tests.smoke` module provides smoke tests for validating basic recall functionality. It contains two functions that support test initialization and execution entry point.

**Key Symbols**

| Symbol | Type | File | Summary |
|--------|------|------|---------|
| `_build_state` | Function | `tests/smoke/smoke_specify_recall.py` | Builds the test state for smoke testing |
| `main` | Function | `tests/smoke/smoke_specify_recall.py` | Primary entry point for smoke test execution |

**Dependencies**

This module has no outgoing dependencies (empty `edges` array), indicating it is a leaf module in the architecture that does not import or utilize other modules within this codebase.

**Depended On By**

No modules in this codebase are documented as depending on `tests.smoke`. The empty `edges` array confirms this module operates independently.

## Module: scratch.test_explore_flash.py

**Path:** `scratch/test_explore_flash.py`

**Responsibility:** This module contains test exploration utilities and entry point logic, with a focus on patched resolution functions for models and providers, along with a main execution function.

**Key Symbols:**
- `_patched_resolve_model` *(function)*: A patched version of resolve model logic for testing purposes.
- `_patched_resolve_provider` *(function)*: A patched version of resolve provider logic for testing purposes.
- `main` *(function)*: Entry point function, likely orchestrating orchestrates test exploration or execution.

**Dependencies:**
- ` on `spine.agents` module.

**Dependents:**
- No modules are documented as depending on `scratch.test_explore_flash.py`.

---

...

## Module: alembic.versions

### Responsibility
The `alembic.versions` module contains Alembic database migration scripts. Each script defines an `upgrade` and `downgrade` function to apply or revert schema changes.

### Key Symbols
| Symbol | Type | File | Description |
|--------|------|------|-------------|
| `upgrade` | Function | `alembic/versions/c69967ea0727_create_work_entries_table.py` | Applies the migration to create the work entries table |
| `downgrade` | Function | `alembic/versions/c69967ea0727_create_work_entries_table.py` | Reverts the migration by dropping the work entries table |

### Dependencies
No direct dependencies or dependents documented in the provided fragment.

## Module: spine.observability.py

### Responsibility
The `spine.observability.py` module provides tracing utilities for observability observability. It contains two function-level symbols that support monitoring and tracing of stream and work operations.

### Key Symbols
- `traced_astream` *(function)*: Instruments async stream operations for tracing.
- `work_run_tracing` *(function)*: Provides tracing context for work execution.

### Dependencies
No incoming or outgoing edges are defined. This module does not declare explicit dependencies or dependents within the provided scope.

### Data Flow
Data flow is not specified. The module functions are entry points for tracing but lack defined input/output pathways in the supplied fragment.

### Notes
Summary descriptions are empty in the supplied fragment. Actual implementations may reveal further context on behavior and integrations.

## Module: spine.services

The `spine.services` module provides audit logging and event management capabilities for the Spine framework. It encapsulates database interactions and provides a clean interface for logging and querying audit events.

### Key Symbols

- **`AuditService`** *(class)* – Main service class located in `spine/services/audit_service.py` that manages audit event lifecycle
- **`__init__`** *(method)* – Constructor for initializing the AuditService with database connection parameters
- **`_ensure_table`** *(method)* – Internal method that creates the audit events table if it doesn't exist
- **`_get_db`** *(method)* – Internal method that retrieves or creates a database connection
- **`log_event`** *(method)* – Public API for recording audit events with event details and metadata
- **`query_events`** *(method)* – Public API for retrieving audit events with optional filtering and pagination

### Dependencies

This module depends on:
- Python's built-in database connectivity layer (implicit)
- Internal database schema management utilities (via `_ensure_table`)

### Being Depended On

Other modules depend on `spine.services` for:
- Audit trail functionality
- Event sourcing capabilities
- Compliance logging requirements

No explicit dependency edges are defined in the current architecture, indicating this module is likely a leaf dependency or its dependents are not tracked in the current architecture map.

## Module: spine.critic

### Responsibility
The `spine.critic` module is responsible for **building the critic agent**, which evaluates and provides feedback on generated content or model behaviors.

### Key Symbols
- `build_critic_agent` *(function)* – Located in `spine/critic/agent.py`, this function constructs and returns the critic agent instance.

### Dependencies and Dependents
- This module has **no listed dependencies** on other modules.
- There are **no modules listed as depending on `spine.critic`** in the current architecture fragment.

---}

## Module: spine.log.py

- **File Path:** `spine/log.py`
- **Responsibility:** Provides centralized logging configuration for the ` framework.
- **Key Symbols:**
  - `configure_logging` (function): Initializes and configures application logging settings.
- **Dependencies:** None (root module).
- **Depended On By:** All modules within the spine framework that require standardized logging behavior.
- **Data Flow:** Called at application startup to ensure consistent logging across all components.

## Module: src.utils

- **Role**: Utility functions providing reusable helper logic.
- **Key Symbols**:
  - `format_duration` (function) located in `src/utils/helpers.py`
- **Dependencies**: None (no incoming or outgoing edges defined).
- **Depended On By**: None (no dependent modules declared).
