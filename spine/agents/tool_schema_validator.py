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
        # Track consecutive validation failures per tool name
        self._rebound_counts: dict[str, int] = defaultdict(int)

    def _reset_rebound(self, tool_name: str) -> None:
        """Reset the rebound counter on a successful validation."""
        self._rebound_counts[tool_name] = 0

    def _increment_rebound(self, tool_name: str) -> int:
        """Increment and return the rebound counter for a tool."""
        self._rebound_counts[tool_name] += 1
        return self._rebound_counts[tool_name]

    def _validate_args(
        self, tool: Any, args: dict[str, Any]
    ) -> tuple[str | None, Exception | None]:
        """Validate tool call arguments against the tool's input schema.

        Args:
            tool: The BaseTool instance with the schema to validate against.
            args: The model-generated arguments dict.

        Returns:
            (error_message, original_exception) — (None, None) if valid.
        """
        if not args:
            # Empty args — check if the tool requires parameters
            try:
                input_schema = tool.get_input_schema()
            except Exception:
                return None, None

            if input_schema and hasattr(input_schema, "model_fields"):
                required_fields = [
                    k for k, v in input_schema.model_fields.items() if v.is_required()
                ]
                if required_fields:
                    return (
                        f"Tool '{tool.name}' requires arguments but none were provided. "
                        f"Required fields: {required_fields}",
                        None,
                    )
            return None, None

        # Get the tool's input schema as a Pydantic model
        try:
            input_schema = tool.get_input_schema()
        except Exception:
            # Can't get schema — let it through
            return None, None

        if input_schema is None:
            return None, None

        # Try to validate the args against the schema
        try:
            input_schema.model_validate(args)
            return None, None
        except Exception as exc:
            return self._format_validation_error(tool, args, exc), exc

    def _format_validation_error(self, tool: Any, args: dict[str, Any], exc: Exception) -> str:
        """Format a Pydantic validation error into an actionable message.

        The goal is to give the model exactly what it needs to correct the
        call: which field is wrong, what was provided, and what's expected.
        """
        tool_name = tool.name

        # Extract field-level errors from Pydantic ValidationError
        errors_fn = getattr(exc, "errors", None)
        error_list = errors_fn() if callable(errors_fn) else []

        if not error_list:
            return (
                f"Tool call to '{tool_name}' failed: "
                f"{type(exc).__name__}: {exc}. "
                f"Check the tool's parameter names and types and try again."
            )

        parts = [f"Tool call to '{tool_name}' failed validation:"]

        for err in error_list:
            loc = " -> ".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "unknown error")
            err_type = err.get("type", "")
            parts.append(f"  - Field '{loc}': {msg} (type={err_type})")

        # Provide the valid schema fields so the model knows what to use
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

    def _format_runtime_error(self, tool_name: str, exc: Exception) -> str:
        """Format a runtime tool execution error."""
        return (
            f"Tool '{tool_name}' execution failed: "
            f"{type(exc).__name__}: {exc}\n"
            f"Check the arguments and retry."
        )

    def _make_error_message(
        self, tool_name: str, tool_call_id: str | None, error_content: str
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
        error, _exc = self._validate_args(tool, args)
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
                            tool_name,
                            self.max_rebound,
                            exc,
                        )
                        raise
                    logger.info(
                        "Tool '%s' runtime error (rebound %d/%d): %s",
                        tool_name,
                        rebound,
                        self.max_rebound,
                        exc,
                    )
                    return self._make_error_message(
                        tool_name,
                        tool_call_id,
                        self._format_runtime_error(tool_name, exc),
                    )
            else:
                return handler(request)

        # Invalid — increment rebound counter
        rebound = self._increment_rebound(tool_name)
        if rebound > self.max_rebound:
            logger.warning(
                "Tool '%s' rebound limit (%d) exceeded on schema validation, "
                "letting handler run (will likely raise): %s",
                tool_name,
                self.max_rebound,
                error,
            )
            return handler(request)

        logger.info(
            "Tool '%s' validation failed (rebound %d/%d): %s",
            tool_name,
            rebound,
            self.max_rebound,
            error,
        )

        return self._make_error_message(tool_name, tool_call_id, error)

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
        error, _exc = self._validate_args(tool, args)
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
                            tool_name,
                            self.max_rebound,
                            exc,
                        )
                        raise
                    logger.info(
                        "Tool '%s' runtime error (rebound %d/%d): %s",
                        tool_name,
                        rebound,
                        self.max_rebound,
                        exc,
                    )
                    return self._make_error_message(
                        tool_name,
                        tool_call_id,
                        self._format_runtime_error(tool_name, exc),
                    )
            else:
                return await handler(request)

        # Invalid — increment rebound counter
        rebound = self._increment_rebound(tool_name)
        if rebound > self.max_rebound:
            logger.warning(
                "Tool '%s' rebound limit (%d) exceeded on schema validation, "
                "letting handler run (will likely raise): %s",
                tool_name,
                self.max_rebound,
                error,
            )
            return await handler(request)

        logger.info(
            "Tool '%s' validation failed (rebound %d/%d): %s",
            tool_name,
            rebound,
            self.max_rebound,
            error,
        )

        return self._make_error_message(tool_name, tool_call_id, error)
