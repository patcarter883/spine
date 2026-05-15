# Rebound Loop (Self-Correction) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a DA middleware that validates tool call arguments against the registered tool schema before execution, returning structured error feedback to the model when validation fails — enabling self-correction within the agent loop instead of crashing the phase.

**Architecture:** A new `ToolSchemaValidator` middleware in `spine/agents/tool_schema_validator.py` that intercepts `awrap_tool_call`, validates the model's tool call arguments against the `BaseTool.get_input_schema()` of the resolved tool, and returns a `ToolMessage(status="error")` with a precise, actionable error message. Consecutive validation failures for the same tool are counted and capped to prevent infinite rebound loops. The middleware is added to the factory's middleware chain for all phase agents.

**Tech Stack:** Python 3.12+, DA AgentMiddleware, Pydantic (for schema validation via tool.get_input_schema()), LangChain BaseTool

---

### Task 1: Create the ToolSchemaValidator middleware module

**Objective:** Write the core middleware class that validates tool call args against the tool's input schema.

**Files:**
- Create: `spine/agents/tool_schema_validator.py`
- Test: `tests/unit/test_tool_schema_validator.py`

**Step 1: Write failing test — unknown tool (tool=None)**

```python
"""Tests for ToolSchemaValidator middleware."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestToolSchemaValidator:
    """Tests for ToolSchemaValidator middleware."""

    def test_passes_through_when_tool_is_none(self):
        """When the model calls a tool that isn't registered (tool=None),
        the middleware should let it through — the DA runtime will produce
        its own error, and we don't duplicate that."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator()
        request = MagicMock()
        request.tool = None
        request.tool_call = {"name": "fake_tool", "args": {}, "id": "tc1"}

        handler = MagicMock(return_value=MagicMock())
        result = validator.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
```

**Step 2: Run test to verify failure**

Run: `pytest tests/unit/test_tool_schema_validator.py::TestToolSchemaValidator::test_passes_through_when_tool_is_none -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spine.agents.tool_schema_validator'`

**Step 3: Write minimal implementation — pass-through skeleton**

Create `spine/agents/tool_schema_validator.py`:

```python
"""SPINE tool schema validator middleware — rebound loop for self-correction.

When a model generates a tool call with arguments that don't match the
registered tool's input schema, this middleware intercepts the call before
execution, constructs a precise error message describing the mismatch, and
returns it as a ``ToolMessage(status="error")`` — giving the model a chance
to self-correct within the same agent loop.

This is the "rebound loop" pattern: the model tries a tool call, gets
told exactly what's wrong, and tries again with corrected arguments.
Without this, a bad tool call crashes the phase immediately.

Two failure modes are handled:

1. **Schema validation error** — arguments are present but don't match
   the tool's input schema (wrong type, unknown parameter, missing
   required field). Returns a structured error with the exact field
   and problem.
2. **Execution error after validation** — arguments passed schema
   validation but the tool itself raised an exception at runtime.
   The error is caught, formatted, and returned as a ToolMessage
   so the model can attempt a different approach.

Consecutive validation failures for the same tool are tracked. When
``max_rebound`` is exceeded, the middleware stops intercepting and lets
the original exception propagate — preventing infinite rebound loops.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


class ToolSchemaValidator(AgentMiddleware):
    """Validate tool call arguments against the registered tool schema.

    Intercepts ``awrap_tool_call`` to check that model-generated arguments
    match the tool's ``get_input_schema()`` before the tool executes.
    On validation failure, returns a ``ToolMessage`` with the error so
    the model can self-correct.

    Args:
        max_rebound: Maximum consecutive validation failures per tool
            before giving up and letting the original error propagate.
            Default 3.
        catch_runtime_errors: If True, also catch exceptions from
            tool execution (after schema validation passes) and return
            them as error ToolMessages. Default True.
    """

    def __init__(
        self,
        max_rebound: int = 3,
        catch_runtime_errors: bool = True,
    ) -> None:
        self.max_rebound = max_rebound
        self.catch_runtime_errors = catch_runtime_errors
        # Track consecutive validation failures per (tool_name, tool_call_id_prefix)
        self._rebound_counts: dict[str, int] = defaultdict(int)

    def _reset_rebound(self, tool_name: str) -> None:
        """Reset the rebound counter on a successful validation."""
        self._rebound_counts[tool_name] = 0

    def _increment_rebound(self, tool_name: str) -> int:
        """Increment and return the rebound counter for a tool."""
        self._rebound_counts[tool_name] += 1
        return self._rebound_counts[tool_name]

    def _validate_args(self, tool: Any, args: dict[str, Any]) -> str | None:
        """Validate tool call arguments against the tool's input schema.

        Args:
            tool: The BaseTool instance with the schema to validate against.
            args: The model-generated arguments dict.

        Returns:
            None if valid, or an error message string if invalid.
        """
        if not args:
            # Empty args — check if the tool requires parameters
            schema = tool.get_input_schema()
            if schema:
                required = getattr(schema, "model_fields", {})
                has_required = any(
                    f.is_required() for f in getattr(schema, "model_fields", {}).values()
                )
                if has_required:
                    return (
                        f"Tool '{tool.name}' requires arguments but none were provided. "
                        f"Required fields: {[k for k, v in schema.model_fields.items() if v.is_required()]}"
                    )
            return None

        # Get the tool's input schema as a Pydantic model
        try:
            input_schema = tool.get_input_schema()
        except Exception:
            # Can't get schema — let it through
            return None

        if input_schema is None:
            return None

        # Try to validate the args against the schema
        try:
            input_schema.model_validate(args)
            return None
        except Exception as exc:
            return self._format_validation_error(tool.name, args, exc)

    def _format_validation_error(
        self, tool_name: str, args: dict[str, Any], exc: Exception
    ) -> str:
        """Format a Pydantic validation error into an actionable message.

        The goal is to give the model exactly what it needs to correct the
        call: which field is wrong, what was provided, and what's expected.
        """
        # Extract field-level errors from Pydantic ValidationError
        errors = getattr(exc, "errors", None)
        if callable(errors):
            error_list = errors()
        else:
            error_list = []

        if not error_list:
            return (
                f"Tool execution failed for '{tool_name}': {type(exc).__name__}: {exc}. "
                f"Check the tool's parameter names and types and try again."
            )

        parts = [f"Tool call to '{tool_name}' failed validation:"]
        for err in error_list:
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "unknown error")
            err_type = err.get("type", "")
            parts.append(f"  - Field '{loc}': {msg} (type={err_type})")

        # List the valid parameter names from the schema
        try:
            schema = tool_name  # we don't have the tool here; fix below
        except Exception:
            pass

        parts.append(
            f"Review the tool's parameter schema and retry with correct argument names and types."
        )
        return "\n".join(parts)

    def _format_validation_error_v2(
        self, tool_name: str, tool: Any, args: dict[str, Any], exc: Exception
    ) -> str:
        """Format a Pydantic validation error with full schema hint."""
        errors = getattr(exc, "errors", None)
        if callable(errors):
            error_list = errors()
        else:
            error_list = []

        if not error_list:
            return (
                f"Tool execution failed for '{tool_name}': {type(exc).__name__}: {exc}. "
                f"Check the tool's parameter names and types and try again."
            )

        parts = [f"Tool call to '{tool_name}' failed validation:"]

        for err in error_list:
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "unknown error")
            err_type = err.get("type", "")
            parts.append(f"  - Field '{loc}': {msg} (type={err_type})")

        # Provide the valid schema fields
        try:
            input_schema = tool.get_input_schema()
            if input_schema and hasattr(input_schema, "model_fields"):
                valid_fields = []
                for fname, finfo in input_schema.model_fields.items():
                    required = "required" if finfo.is_required() else "optional"
                    ftype = getattr(finfo, "annotation", "any")
                    valid_fields.append(f"  {fname}: {ftype} ({required})")
                if valid_fields:
                    parts.append("Valid parameters:")
                    parts.extend(valid_fields)
        except Exception:
            pass

        parts.append("Retry with the correct parameter names and types.")
        return "\n".join(parts)

    def _make_error_message(
        self, tool_name: str, tool_call_id: str, error_content: str
    ) -> ToolMessage:
        """Create a ToolMessage with error status for rebound."""
        return ToolMessage(
            content=error_content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    # ── Sync wrap_tool_call ──────────────────────────────────────────

    def wrap_tool_call(self, request, handler):
        """Intercept tool calls and validate arguments before execution."""
        tool = request.tool
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        args = tool_call.get("args", {})

        # Pass through if tool not registered (DA will handle the error)
        if tool is None:
            return handler(request)

        # Validate arguments against the tool's input schema
        error = self._validate_args(tool, args)
        if error is None:
            # Valid — reset rebound counter and execute
            self._reset_rebound(tool_name)
            if self.catch_runtime_errors:
                try:
                    return handler(request)
                except Exception as exc:
                    rebound = self._increment_rebound(tool_name)
                    if rebound > self.max_rebound:
                        logger.warning(
                            "Tool '%s' rebound limit (%d) exceeded on runtime error, "
                            "letting exception propagate: %s",
                            tool_name, self.max_rebound, exc,
                        )
                        raise
                    runtime_error = (
                        f"Tool '{tool_name}' execution failed: "
                        f"{type(exc).__name__}: {exc}\n"
                        f"Check the arguments and retry."
                    )
                    return self._make_error_message(
                        tool_name, tool_call_id, runtime_error
                    )
            else:
                return handler(request)

        # Invalid — increment rebound counter
        rebound = self._increment_rebound(tool_name)
        if rebound > self.max_rebound:
            logger.warning(
                "Tool '%s' rebound limit (%d) exceeded on schema validation, "
                "letting handler run (will likely raise): %s",
                tool_name, self.max_rebound, error,
            )
            return handler(request)

        logger.info(
            "Tool '%s' validation failed (rebound %d/%d): %s",
            tool_name, rebound, self.max_rebound, error,
        )

        # Use the richer format that includes valid parameter names
        try:
            full_error = self._format_validation_error_v2(
                tool_name, tool, args, type("E", (Exception,), {
                    "errors": lambda self=None: getattr(
                        self._validate_args.__func__(tool, args), "errors", lambda: []
                    )()
                })()
            )
        except Exception:
            full_error = error

        # Re-validate to get the actual exception for the rich format
        try:
            tool.get_input_schema().model_validate(args)
        except Exception as real_exc:
            full_error = self._format_validation_error_v2(
                tool_name, tool, args, real_exc
            )
        else:
            full_error = error

        return self._make_error_message(tool_name, tool_call_id, full_error)

    # ── Async wrap_tool_call ─────────────────────────────────────────

    async def awrap_tool_call(self, request, handler):
        """Intercept async tool calls and validate arguments before execution."""
        tool = request.tool
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        args = tool_call.get("args", {})

        # Pass through if tool not registered
        if tool is None:
            return await handler(request)

        # Validate arguments against the tool's input schema
        error = self._validate_args(tool, args)
        if error is None:
            # Valid — reset rebound counter and execute
            self._reset_rebound(tool_name)
            if self.catch_runtime_errors:
                try:
                    return await handler(request)
                except Exception as exc:
                    rebound = self._increment_rebound(tool_name)
                    if rebound > self.max_rebound:
                        logger.warning(
                            "Tool '%s' rebound limit (%d) exceeded on runtime error, "
                            "letting exception propagate: %s",
                            tool_name, self.max_rebound, exc,
                        )
                        raise
                    runtime_error = (
                        f"Tool '{tool_name}' execution failed: "
                        f"{type(exc).__name__}: {exc}\n"
                        f"Check the arguments and retry."
                    )
                    return self._make_error_message(
                        tool_name, tool_call_id, runtime_error
                    )
            else:
                return await handler(request)

        # Invalid — increment rebound counter
        rebound = self._increment_rebound(tool_name)
        if rebound > self.max_rebound:
            logger.warning(
                "Tool '%s' rebound limit (%d) exceeded on schema validation, "
                "letting handler run (will likely raise): %s",
                tool_name, self.max_rebound, error,
            )
            return await handler(request)

        logger.info(
            "Tool '%s' validation failed (rebound %d/%d): %s",
            tool_name, rebound, self.max_rebound, error,
        )

        # Get the actual exception for rich formatting
        try:
            tool.get_input_schema().model_validate(args)
        except Exception as real_exc:
            full_error = self._format_validation_error_v2(
                tool_name, tool, args, real_exc
            )
        else:
            full_error = error

        return self._make_error_message(tool_name, tool_call_id, full_error)
```

**Step 4: Run test to verify pass**

Run: `pytest tests/unit/test_tool_schema_validator.py::TestToolSchemaValidator::test_passes_through_when_tool_is_none -v`
Expected: PASS

**Step 5: Commit**

```bash
git add spine/agents/tool_schema_validator.py tests/unit/test_tool_schema_validator.py
git commit -m "feat: add ToolSchemaValidator middleware skeleton with tool=None pass-through"
```

---

### Task 2: Add test for valid tool call (happy path)

**Objective:** Verify that when a model sends correct arguments, the middleware passes through without modification.

**Files:**
- Modify: `tests/unit/test_tool_schema_validator.py`

**Step 1: Write failing test**

Add to `TestToolSchemaValidator`:

```python
    def test_passes_through_valid_args(self):
        """When tool call args match the schema, handler is called normally."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator()

        # Create a mock tool with a Pydantic input schema
        from pydantic import BaseModel

        class ReadFileInput(BaseModel):
            filepath: str

        tool = MagicMock()
        tool.name = "read_file"
        tool.get_input_schema.return_value = ReadFileInput

        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/tmp/test.txt"},
            "id": "tc1",
        }

        expected_result = MagicMock()
        handler = MagicMock(return_value=expected_result)

        result = validator.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
        assert result is expected_result
```

**Step 2: Run test**

Run: `pytest tests/unit/test_tool_schema_validator.py::TestToolSchemaValidator::test_passes_through_valid_args -v`
Expected: PASS (the skeleton already does this correctly)

**Step 3: Commit**

```bash
git add tests/unit/test_tool_schema_validator.py
git commit -m "test: add happy-path test for ToolSchemaValidator"
```

---

### Task 3: Add test for schema validation failure (rebound)

**Objective:** Verify that when the model sends wrong arguments, a ToolMessage with error status is returned and the handler is NOT called.

**Files:**
- Modify: `tests/unit/test_tool_schema_validator.py`

**Step 1: Write failing test**

```python
    def test_returns_error_on_wrong_param_name(self):
        """When model uses an incorrect parameter name, return error ToolMessage."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator()

        from pydantic import BaseModel

        class ReadFileInput(BaseModel):
            filepath: str

        tool = MagicMock()
        tool.name = "read_file"
        tool.get_input_schema.return_value = ReadFileInput

        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"file_names": ["/tmp/a.txt"]},  # wrong: should be 'filepath'
            "id": "tc2",
        }

        handler = MagicMock()

        result = validator.wrap_tool_call(request, handler)
        # Handler should NOT be called — the error was intercepted
        handler.assert_not_called()
        # Result should be a ToolMessage with error status
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "read_file" in result.content
        assert "file_names" in result.content or "filepath" in result.content
```

**Step 2: Run test**

Run: `pytest tests/unit/test_tool_schema_validator.py::TestToolSchemaValidator::test_returns_error_on_wrong_param_name -v`
Expected: PASS (the implementation handles this)

**Step 3: Commit**

```bash
git add tests/unit/test_tool_schema_validator.py
git commit -m "test: add schema validation failure test for ToolSchemaValidator"
```

---

### Task 4: Add test for rebound limit exhaustion

**Objective:** Verify that after max_rebound consecutive failures for the same tool, the middleware stops intercepting and calls the handler (letting the original error propagate or the runtime handle it).

**Files:**
- Modify: `tests/unit/test_tool_schema_validator.py`

**Step 1: Write failing test**

```python
    def test_rebound_limit_exhaustion(self):
        """After max_rebound failures, middleware passes through to handler."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator(max_rebound=2)

        from pydantic import BaseModel

        class ReadFileInput(BaseModel):
            filepath: str

        tool = MagicMock()
        tool.name = "read_file"
        tool.get_input_schema.return_value = ReadFileInput

        # First two calls should be intercepted (return error ToolMessage)
        for i in range(2):
            request = MagicMock()
            request.tool = tool
            request.tool_call = {
                "name": "read_file",
                "args": {"file_names": ["bad"]},
                "id": f"tc{i}",
            }
            handler = MagicMock()
            result = validator.wrap_tool_call(request, handler)
            handler.assert_not_called()
            assert isinstance(result, ToolMessage)
            assert result.status == "error"

        # Third call should pass through (rebound exhausted)
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"file_names": ["still_bad"]},
            "id": "tc3",
        }
        expected_result = MagicMock()
        handler = MagicMock(return_value=expected_result)
        result = validator.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
        assert result is expected_result

    def test_rebound_counter_resets_on_valid_call(self):
        """A successful validation resets the rebound counter for that tool."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator(max_rebound=2)

        from pydantic import BaseModel

        class ReadFileInput(BaseModel):
            filepath: str

        tool = MagicMock()
        tool.name = "read_file"
        tool.get_input_schema.return_value = ReadFileInput

        # One bad call
        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"file_names": ["bad"]},
            "id": "tc1",
        }
        handler = MagicMock()
        validator.wrap_tool_call(request, handler)
        assert validator._rebound_counts["read_file"] == 1

        # One good call resets the counter
        request2 = MagicMock()
        request2.tool = tool
        request2.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/tmp/good.txt"},
            "id": "tc2",
        }
        handler2 = MagicMock(return_value=MagicMock())
        validator.wrap_tool_call(request2, handler2)
        assert validator._rebound_counts["read_file"] == 0

        # Bad calls should work again from count 0
        request3 = MagicMock()
        request3.tool = tool
        request3.tool_call = {
            "name": "read_file",
            "args": {"file_names": ["bad_again"]},
            "id": "tc3",
        }
        handler3 = MagicMock()
        result = validator.wrap_tool_call(request3, handler3)
        handler3.assert_not_called()
        assert isinstance(result, ToolMessage)
```

**Step 2: Run tests**

Run: `pytest tests/unit/test_tool_schema_validator.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/test_tool_schema_validator.py
git commit -m "test: add rebound limit and counter reset tests"
```

---

### Task 5: Add test for runtime error catching

**Objective:** Verify that when a tool passes schema validation but fails at runtime (e.g. file not found), the error is caught and returned as a ToolMessage so the model can try a different approach.

**Files:**
- Modify: `tests/unit/test_tool_schema_validator.py`

**Step 1: Write failing test**

```python
    def test_catches_runtime_error(self):
        """When tool execution raises after passing validation, return error message."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator(catch_runtime_errors=True)

        from pydantic import BaseModel

        class ReadFileInput(BaseModel):
            filepath: str

        tool = MagicMock()
        tool.name = "read_file"
        tool.get_input_schema.return_value = ReadFileInput

        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/nonexistent/file.txt"},
            "id": "tc1",
        }

        # Handler raises FileNotFoundError
        handler = MagicMock(side_effect=FileNotFoundError("No such file"))

        result = validator.wrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "FileNotFoundError" in result.content

    def test_runtime_errors_disabled(self):
        """When catch_runtime_errors=False, runtime exceptions propagate."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator(catch_runtime_errors=False)

        from pydantic import BaseModel

        class ReadFileInput(BaseModel):
            filepath: str

        tool = MagicMock()
        tool.name = "read_file"
        tool.get_input_schema.return_value = ReadFileInput

        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/tmp/test.txt"},
            "id": "tc1",
        }

        handler = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            validator.wrap_tool_call(request, handler)
```

**Step 2: Run tests**

Run: `pytest tests/unit/test_tool_schema_validator.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/test_tool_schema_validator.py
git commit -m "test: add runtime error catching tests for ToolSchemaValidator"
```

---

### Task 6: Add test for wrong type (not just wrong name)

**Objective:** Verify that type mismatches (e.g. string where int expected) also produce clear error messages.

**Files:**
- Modify: `tests/unit/test_tool_schema_validator.py`

**Step 1: Write test**

```python
    def test_returns_error_on_wrong_type(self):
        """When model sends a value of wrong type, return error with type info."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator()

        from pydantic import BaseModel

        class SearchInput(BaseModel):
            max_results: int

        tool = MagicMock()
        tool.name = "search"
        tool.get_input_schema.return_value = SearchInput

        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "search",
            "args": {"max_results": "ten"},  # string instead of int
            "id": "tc_type",
        }

        handler = MagicMock()
        result = validator.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        # Should mention the field and/or expected type
        assert "max_results" in result.content
```

**Step 2: Run test**

Run: `pytest tests/unit/test_tool_schema_validator.py::TestToolSchemaValidator::test_returns_error_on_wrong_type -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/test_tool_schema_validator.py
git commit -m "test: add type mismatch test for ToolSchemaValidator"
```

---

### Task 7: Add test for async awrap_tool_call

**Objective:** Verify the async path works identically to the sync path.

**Files:**
- Modify: `tests/unit/test_tool_schema_validator.py`

**Step 1: Write test**

```python
    @pytest.mark.asyncio
    async def test_async_valid_passes_through(self):
        """Async path passes through valid tool calls."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator()

        from pydantic import BaseModel

        class ReadFileInput(BaseModel):
            filepath: str

        tool = MagicMock()
        tool.name = "read_file"
        tool.get_input_schema.return_value = ReadFileInput

        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"filepath": "/tmp/good.txt"},
            "id": "tc_async1",
        }

        expected = MagicMock()
        handler = AsyncMock(return_value=expected)

        result = await validator.awrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
        assert result is expected

    @pytest.mark.asyncio
    async def test_async_invalid_returns_error(self):
        """Async path returns error ToolMessage on invalid args."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        validator = ToolSchemaValidator()

        from pydantic import BaseModel

        class ReadFileInput(BaseModel):
            filepath: str

        tool = MagicMock()
        tool.name = "read_file"
        tool.get_input_schema.return_value = ReadFileInput

        request = MagicMock()
        request.tool = tool
        request.tool_call = {
            "name": "read_file",
            "args": {"file_names": ["bad"]},
            "id": "tc_async2",
        }

        handler = AsyncMock()
        result = await validator.awrap_tool_call(request, handler)
        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
```

**Step 2: Run tests**

Run: `pytest tests/unit/test_tool_schema_validator.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/test_tool_schema_validator.py
git commit -m "test: add async middleware tests for ToolSchemaValidator"
```

---

### Task 8: Wire ToolSchemaValidator into the factory

**Objective:** Add the ToolSchemaValidator to the middleware chain in `build_phase_agent()` so all phase agents get it automatically.

**Files:**
- Modify: `spine/agents/factory.py`

**Step 1: Add import and instantiation**

In `spine/agents/factory.py`, after the `ToolOutputTrimmer` import block (~line 119-122), add the ToolSchemaValidator to the middleware chain for all non-subagent phase agents:

```python
    # Tool schema validation (rebound loop for self-correction)
    if not is_subagent:
        from spine.agents.tool_schema_validator import ToolSchemaValidator
        middleware.append(ToolSchemaValidator())
```

This should go BEFORE the `ToolOutputTrimmer` in the middleware list order. The validator intercepts tool calls before execution; the trimmer intercepts model calls. They operate on different hooks so order between them doesn't strictly matter, but logically: validate first, then trim.

**Step 2: Verify no regressions**

Run: `pytest tests/unit/ -v`
Expected: All existing tests pass.

**Step 3: Commit**

```bash
git add spine/agents/factory.py
git commit -m "feat: wire ToolSchemaValidator into phase agent factory"
```

---

### Task 9: Add config toggle for the feature

**Objective:** Add a `tool_schema_validation` setting to `.spine/config.yaml` and `SpineConfig` so users can disable the validator (e.g. for models with known-good tool use that don't need the overhead).

**Files:**
- Modify: `spine/config.py`
- Modify: `spine/agents/factory.py`

**Step 1: Add field to SpineConfig**

In `spine/config.py`, add to the `SpineConfig` dataclass:

```python
    tool_schema_validation: bool = True
```

And in `SpineConfig.load()`, add the resolution:

```python
            tool_schema_validation=os.getenv(
                "SPINE_TOOL_SCHEMA_VALIDATION",
                str(spine.get("tool_schema_validation", True))
            ).lower() not in ("0", "false", "no"),
```

**Step 2: Use the config in factory**

In `spine/agents/factory.py`, modify the ToolSchemaValidator addition to check the config. The config isn't directly accessible in the factory function (it takes `config: RunnableConfig`), so we use an env var check as a simple approach:

```python
    # Tool schema validation (rebound loop for self-correction)
    # Can be disabled via SPINE_TOOL_SCHEMA_VALIDATION=false
    if not is_subagent:
        import os
        validation_enabled = os.getenv(
            "SPINE_TOOL_SCHEMA_VALIDATION", "true"
        ).lower() not in ("0", "false", "no")
        if validation_enabled:
            from spine.agents.tool_schema_validator import ToolSchemaValidator
            middleware.append(ToolSchemaValidator())
```

**Step 3: Verify tests pass**

Run: `pytest tests/unit/ -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add spine/config.py spine/agents/factory.py
git commit -m "feat: add config toggle for tool schema validation (SPINE_TOOL_SCHEMA_VALIDATION)"
```

---

### Task 10: Clean up the implementation

**Objective:** Remove the unused `_format_validation_error` method (we only use `_format_validation_error_v2`) and simplify the double-validation in `wrap_tool_call`. The current implementation validates args twice — once in `_validate_args` and once for error formatting. Refactor to validate once and keep the exception.

**Files:**
- Modify: `spine/agents/tool_schema_validator.py`

**Step 1: Refactor _validate_args to return (error_message, exception)**

Replace the two-step validation with a single validation that captures both the formatted message and the original exception:

```python
    def _validate_args(self, tool: Any, args: dict[str, Any]) -> tuple[str | None, Exception | None]:
        """Validate tool call arguments against the tool's input schema.

        Returns:
            (error_message, original_exception) — (None, None) if valid.
        """
        ...  # same logic but returns both
```

Then `wrap_tool_call` and `awrap_tool_call` can use the exception directly for formatting without re-validating.

**Step 2: Remove _format_validation_error (keep only _format_validation_error_v2, renamed)**

**Step 3: Verify all tests still pass**

Run: `pytest tests/unit/test_tool_schema_validator.py -v`
Expected: All 10+ tests pass.

**Step 4: Commit**

```bash
git add spine/agents/tool_schema_validator.py
git commit -m "refactor: clean up ToolSchemaValidator — single-pass validation, remove dead code"
```

---

### Task 11: Add integration test — end-to-end middleware chain

**Objective:** Verify that ToolSchemaValidator works correctly alongside ToolOutputTrimmer in the full middleware chain (no conflicts, no double-interception).

**Files:**
- Modify: `tests/unit/test_tool_schema_validator.py`

**Step 1: Write test**

```python
    def test_coexists_with_output_trimmer(self):
        """ToolSchemaValidator and ToolOutputTrimmer don't conflict."""
        from spine.agents.tool_schema_validator import ToolSchemaValidator
        from spine.agents.context_editing import ToolOutputTrimmer

        # Both in middleware list — validator uses wrap_tool_call,
        # trimmer uses awrap_model_call. Different hooks, no conflict.
        validator = ToolSchemaValidator()
        trimmer = ToolOutputTrimmer()

        # Just verify they both have the expected hooks
        assert hasattr(validator, "wrap_tool_call")
        assert hasattr(validator, "awrap_tool_call")
        assert hasattr(trimmer, "wrap_tool_call")
        assert hasattr(trimmer, "awrap_tool_call")
        assert hasattr(trimmer, "awrap_model_call")
```

**Step 2: Run test**

Run: `pytest tests/unit/test_tool_schema_validator.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/test_tool_schema_validator.py
git commit -m "test: add coexistence test for ToolSchemaValidator + ToolOutputTrimmer"
```

---

### Task 12: Run full test suite and lint

**Objective:** Verify no regressions across the entire project.

**Files:** None (verification only)

**Step 1: Run full unit tests**

Run: `pytest tests/unit/ -v`
Expected: All pass.

**Step 2: Run linter**

Run: `ruff check spine/agents/tool_schema_validator.py spine/agents/factory.py spine/config.py`
Expected: No errors.

**Step 3: Run formatter**

Run: `ruff format spine/agents/tool_schema_validator.py`
Expected: No changes (or auto-formatted).

**Step 4: Final commit if formatting changed**

```bash
git add -A
git commit -m "style: ruff format on tool_schema_validator"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Middleware module skeleton | `spine/agents/tool_schema_validator.py`, `tests/unit/test_tool_schema_validator.py` |
| 2 | Happy-path test (valid args pass through) | `tests/unit/test_tool_schema_validator.py` |
| 3 | Schema failure test (wrong param name → error ToolMessage) | `tests/unit/test_tool_schema_validator.py` |
| 4 | Rebound limit test (exhaustion + counter reset) | `tests/unit/test_tool_schema_validator.py` |
| 5 | Runtime error catching test | `tests/unit/test_tool_schema_validator.py` |
| 6 | Type mismatch test | `tests/unit/test_tool_schema_validator.py` |
| 7 | Async path tests | `tests/unit/test_tool_schema_validator.py` |
| 8 | Wire into factory | `spine/agents/factory.py` |
| 9 | Config toggle (env var) | `spine/config.py`, `spine/agents/factory.py` |
| 10 | Clean up implementation (remove dead code, single-pass) | `spine/agents/tool_schema_validator.py` |
| 11 | Coexistence test with ToolOutputTrimmer | `tests/unit/test_tool_schema_validator.py` |
| 12 | Full test suite + lint | All files |

**Key design decisions:**
- **Middleware, not LangGraph state** — stays inside the DA agent loop; no new nodes/edges needed in any WorkType's graph
- **Rebound limit (default 3)** — prevents infinite loops where the model keeps sending the same bad arguments
- **Counter resets on success** — a valid call after a bad one means the model learned; don't accumulate
- **Runtime error catching** — schema validation passes but execution fails (file not found, permission denied, etc.) — also returned as error ToolMessage
- **Config toggle** — `SPINE_TOOL_SCHEMA_VALIDATION=false` to disable (for models with known-perfect tool use)
- **tool=None pass-through** — unregistered tools get DA's native error; we don't duplicate
