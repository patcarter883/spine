# Coding Guidelines

## Logging Conventions

All modules MUST obtain a logger instance using `logging.getLogger(__name__)` to ensure proper hierarchical logging and log correlation with module names.

```python
import logging

logger = logging.getLogger(__name__)
```

This pattern is established in:
- `spine/config.py` - Used by both `SpineConfig` class and `load` method
- `spine/log.py` - Used by `configure_logging` function

This convention ensures loggersn logs are properly namespaced under their originating module.

## Data Model Conventions

All internal data models must be implemented as **frozen `dataclass` decorators**. This ensures immutability, clear structure, and automatic method generation (__init__, __repr__, etc.) for data-holding classes.

### Rule
Use Python's `@dataclass(frozen=True)` when defining classes solely for holding data within the codebase.

### Example
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RepoAnalyzer:
    name: str
    path: str
```

This pattern enforces type safety, prevents accidental mutation, and clearly separates data models from behavioral classes.

## Naming Conventions

### Functions and Methods
All function and method signatures **must** include complete type hints for parameters and return values. Examples from the codebase:

- `run_migrations_offline()` — `alembic/env.py`
- `run_migrations_online()` — `alembic/env.py`
- `upgrade()` — `alembic/versions/c69967ea0727_create_work_entries_table.py`

These functions demonstrate the requirement for explicit typing in signatures.

### Module-Level Symbols
Public and private symbols named at module level (e.g., functions, classes) follow standard Python naming: lowercase with underscores for functions and methods, CapWords for classes.

*(No evidence was found for constants, variables, or other naming patterns beyond the function type hint convention stated above.)*

---

## Error Handling Conventions

### Rule: Guard Fallible Operations with try/except

Wrap all operations that can raise exceptions in a `try/except` block. This pattern is demonstrated in:

- `spine/agents/_tokens.py` - [`count_tokens`](()(function)`](code-spine/agents/_tokens.py)
- `spine/agents/artifacts.py` - [`scan_artifact_dir()(function)`](code-spine/agents/artifacts.py)
- `spine/agents/artifacts.py` - [`list_slice_files()(function)`](code-spine/agents/artifacts.py)

Apply this guard pattern to any I/O, parsing, or external-call operations where failure is possible.

**Example Template:**
```python
def example_function():
    try:
        # fallible operation here
        pass
    except Exception as e:
        # handle error appropriately
        pass
```

*Do Policy:* All contributions MUST adopt this defensive structure when invoking potentially failing operations.

---

Type hints MUST be explicit

*Formal utilizes PEP 484 type annotations.

## Testing Conventions

All tests MUST follow these naming and placement conventions:

1. **Function Names**: Test functions MUST be prefixed with `test_` and placed in the `tests/` directory.
2. **Class Names**: Test classes MUST follow the `Test*` naming pattern (e.g., `TestExplorer`) and also reside in `tests/`.
3. **Example Evidence**:
   - `tests/conftest.py` contains `test_config`n   - `tests/integration/test_exploration_subgraph.py` contains `test_exploration_subgraph_builds`

## Config Conventions

All configuration access must be performed through the centralized `SpineConfig.load()` method. Contributors must not instantiate or import configuration objects directly. The `SpineConfig.load()` method provides the single, authoritative source for accessing configuration throughout the codebase.

**Evidence
This convention is evidenced
verified by examining the following functions in `spine/agents/exploration_agents.py`:
- `run_research_manager`
- `run_explore_do_node`
- `_findings_structured_model`

All configuration access within these functions consistently uses `SpineConfig.load()` rather than direct instantiation or alternative configuration patterns.

**Enforcement:**nModule/common.rst: Failure to use `SpineConfig.load()` will result in inconsistent configuration states and potential runtime errors.
