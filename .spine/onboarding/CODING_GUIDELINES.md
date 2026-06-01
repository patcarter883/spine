# Coding Guidelines

## Logging Conventions

### Use module-level loggers.getLogger(__name__)

All Python modules MUST obtain a logger using `logging.getLogger(__name__)` at the module level. This pattern ensures log records are organized organized under the module's fully-qualified name in the logger tree.

**Evidence:**
- `spine/config.py`: The `SpineConfig` class and `load` method both operate within a module that follows this convention
- `spine/log.py`: The `configure_logging` function establishes this as the project-wide standard

**Example:**
```python
import logging

logger = logging.getLogger(__name__)

class MyClass:
    def my_method(self):
        logger.info("Processing started")
```

**Rationale:**
This convention enables granular control over logging levels per module and integrates seamlessly with the project's centralized `configure_logging` setup in `spine/log.py`.

## Data Model Conventions

### Use frozen dataclasses for internal data models

**Rule**: Define all internal data models as Python `dataclass`es decorated with `@dataclass(frozen=True)`.

**Evidence**: The `RepoAnalyzer` class in `spine/work/onboarding/analyzer.py` demonstrates this convention.

**Rationale**: Frozen dataclasses provide immutability, automatic `__eq__`, `__repr__`, and type-safe attribute definitions while being lightweight and Python.

**Example**:
```python
def __post_init__(self):\n):
    # Validation logic if needed
```

**Note**: This convention applies to internal data structures only. API contracts and external interfaces may follow different serialization patterns.

## Naming Conventions

All functions and methods MUST have fully typed signatures. Use Python type hints for all parameters and return values.

```python
# Example from alembic/env.py
from typing import Any

def run_migrations_offline() -> None:
    """Run migrations offline."""
    ...

def run_migrations_online() -> None:
    """Run migrations online."""
    ...

def upgrade() -> None:
    """Upgrade schema."""
    ...
```

Key rules:
- Use snake_case for function and method names
- Include complete type annotations for all parameters
- Use `-> None` for functions that don't return a value
- Follow PEP 8 naming conventions

Evidence: Functions `run_migrations_offline`, `run_migrations_online`, and `upgrade` in `alembic/env.py` and migration files all follow this pattern with full type hints.

## Error Handling Conventions

### Rule: Use try/except guards for fallible operations

Wrap all fallible operations in `try/except` blocks to gracefully handle exceptions and prevent unhandled crashes. This pattern is consistently applied across the codebase in functions that interact with external systems or perform potentially unsafe operations. Examples can be found in:

- `spine/agents/_tokens.py` - function `count_tokens`
- `spine/agents/artifacts.py` - function `scan_artifact_dir`
- `spine/agents/artifacts.py` - function `list_slice_files`

When implementing new functions that perform file I/O, network operations, or other potentially exception-raising activities, ensure you follow this same pattern by wrapping the fallible code in appropriate error handling blocks.

## Testing Conventions

All tests **must** follow the established pytest patterns:

1. **Function naming**: Test functions **must** be prefixed with `test_`. Example: `test_exploration_subgraph_builds`\n2. **Class naming**: Test classes **must** use `Test*` PascalCase naming (e.g., `TestExplorer`).
3. **Location**: Place tests in the `tests/` directory, using subdirectories like `tests/integration/` to organize by scope.
4. **Structure**: Place `conftest.py` files in relevant test directories to scope fixtures appropriately.

Adhere strictly to these conventions to maintain consistency with existing tests like `test_expl` and `test_exploration_subgraph_supports_plan`.

## Config Conventions

All configuration access must be performed through the centralized `SpineConfig.load()` method. Never instantiate or access configuration values directly.

**Rule:**

The use of `SpineConfig.load()` is consistently applied across the codebase, including:

* `run_research_manager` function in spine(':spine/agents/exploration_agents.py
* `run_explore_do_node()` function in ':spine/agents/exploration_agents.py
* `_findings_structured_model()` function in ':spine/agents/exploration_agents.py

This centralized approach ensures uniform configuration management and prevents ad-hoc instantiation or scattering of configuration logic throughout the codebase.

**Rule:**

Distribute
Contributors must route all configuration retrieval through `SpineConfig.load()`, treating it as the sole entry point for configuration access.
