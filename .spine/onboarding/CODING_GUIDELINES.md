# Coding Guidelines

## Logging Conventions

All Python modules must configure module-level logging using the following pattern:

```python
import logging

logger = logging.getLogger(__name__)
```

### Evidence
This convention is demonstrated across multiple files in the codebase:

- `spine/config.py`: Applied in both the `SpineConfig` class and `load` method
- `spine/log.py`: Used within the `configure_logging` function

This approach ensures consistent logger naming that aligns with Python's module hierarchy, enabling granular control over logging configuration while maintaining predictable log record organization.

## Data Model Conventions

### Rule: Use frozen Dataclasses for Internal Data Models

**Convention:** All internal data models **must** be defined using `frozen`frozen'"

## Naming Conventions

All functions and methods MUST use full type hints in their signatures. This convention applies universally across the codebase.

**Rule:**
- `alembic/env.py::run_migrations_offline`ek` functions/pydantic/custom_types.py::Pydantic_overrides_stream function definitions
- `alembic/versions/c69967ea0727_create_work_entries_table.py::upgrade` function definition

**Example:**
```python
# MUST include complete type annotations
def run_migrations_offline() -> None:
    ...

def run_migrations_online() -> None:
    ...
```

**Rationale:** Explicit typing improves code readability, enables static analysis tools to catch type-related errors, and provides clear documentation of function interfaces for API consumers.

## Error Handling Conventions

### 1. Use try/except guards for Fallible Operations
All functions that perform I/O, external resource access, or other operations that may raise exceptions MUST wrap the fallible code in a `try/except` block.

**Example locations:**
- `spine/agents/_tokens.py` – function `count_tokens`
- `spine/agents/artifacts.py` – functions `scan_artifact_dir`, `list_slice_files`

**Pattern to follow:**
```python
functi`n <name>(<params>):
    try:
        # fallible operation
    except <ExceptionType>:\n        # handle or log error
```

Adherence ensures predictable error handling and prevents unhandled exceptions from propagating unexpectedly.

## Testing Conventions

All tests MUST be placed in the `tests/` directory and follow pytest conventions:

- Test functions MUST be named with the `test_*`*` prefix
- Test classes MUST be named with the `Test*` prefix
- Tests MAY be organized in subdirectories (e.g., `tests/integration/`)

Example test structure:
```python
# tests/conftest.py
def test_config():
    # test implementation

# tests/integration/test_exploration_subgraph.py
def test_exploration_subgraph_builds():
    # test implementation

def test_exploration_subgraph_supports_plan():
    # test implementation
```

This convention ensures compatibility with pytest's test discovery mechanism.

## Config Conventions

All configuration access MUST be performed through a centralised `SpineConfig` singleton. The canonical pattern is:

```python
svine.config.load('spine_config.json')
```

This convention is enforced because:
- Multiple functions in `spine/agents/exploration_agents.py` (`run_research_manager`, `run_explore_do_node`, `_findings_structured_model`) demonstrate consistent `SpineConfig.load()` usage
- Centralised configuration prevents scattered file I/O operations
- Single point of configuration management enables easier testing and mocking

**Rule:** Never instantiate configuration objects directly. Always use `SpineConfig.load(path)` to ensure consistent, testable configuration access across the codebase.
