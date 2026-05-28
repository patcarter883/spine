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

import json
import logging
import typing
from collections import defaultdict
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

_MAX_EXC_CHARS = 300
_MAX_VALIDATION_ERRORS = 5
_MAX_SCHEMA_FIELDS = 10


def _summarize_exception(exc: Exception) -> str:
    """Render an exception as a single short line — no traceback.

    Some tool runtime errors arrive with embedded tracebacks or multi-line
    Pydantic dumps in ``str(exc)``. The model gains nothing from those —
    they bloat the conversation and previously ended up serialized into
    ResearchFindings.summary verbatim. Strip to the first non-empty line
    and cap at ``_MAX_EXC_CHARS``.
    """
    raw = str(exc) or exc.__class__.__name__
    first_line = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Traceback") or stripped.startswith('File "'):
            continue
        first_line = stripped
        break
    if not first_line:
        first_line = raw.strip().replace("\n", " ")
    if len(first_line) > _MAX_EXC_CHARS:
        first_line = first_line[: _MAX_EXC_CHARS - 1] + "…"
    return f"{type(exc).__name__}: {first_line}"


def _resolve_schema_fields(tool: Any) -> list[tuple[str, Any, bool]]:
    """Return ``[(field_name, annotation_or_type, required), ...]`` for *tool*.

    Handles two shapes:

    * **Pydantic BaseModel** with real fields — read directly from
      ``model_fields``. Used by SPINE-native tools.
    * **RootModel wrapping a JSON Schema dict** — produced by
      langchain-mcp-adapter when MCP servers ship raw JSON Schema as
      ``args_schema``. ``model_fields`` for these is just
      ``{'root': ...}``, which leaks an irrelevant ``root`` placeholder
      to the model on schema errors. Unwrap by reading ``tool.args_schema``
      directly and parsing the JSON Schema ``properties``/``required``.

    Returns an empty list if the schema can't be resolved.
    """
    raw = getattr(tool, "args_schema", None)
    if isinstance(raw, dict) and isinstance(raw.get("properties"), dict):
        required = set(raw.get("required") or [])
        out: list[tuple[str, Any, bool]] = []
        for fname, fspec in raw["properties"].items():
            ftype = fspec.get("type", "any") if isinstance(fspec, dict) else "any"
            out.append((fname, ftype, fname in required))
        return out

    try:
        input_schema = tool.get_input_schema()
    except Exception:
        return []
    if not input_schema or not hasattr(input_schema, "model_fields"):
        return []
    fields = input_schema.model_fields
    # RootModel wrappers expose a single "root" field whose annotation is
    # not the user-facing schema. Treat as unresolvable.
    if set(fields.keys()) == {"root"}:
        return []
    return [
        (fname, getattr(finfo, "annotation", "any"), finfo.is_required())
        for fname, finfo in fields.items()
    ]


def _placeholder_for_annotation(annotation: Any, field_name: str) -> Any:
    """Best-effort placeholder value for a Pydantic field annotation.

    Used to synthesise an example tool call. Conservative — anything we
    can't recognise becomes a ``"<field_name>"`` string placeholder so
    the example is at least JSON-valid.
    """
    # JSON Schema type strings (from MCP-shaped tools via _resolve_schema_fields).
    if isinstance(annotation, str):
        return {
            "string": f"<{field_name}>",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "array": [],
            "object": {},
        }.get(annotation, f"<{field_name}>")
    origin = typing.get_origin(annotation)
    if origin is None:
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is bool:
            return False
        if annotation is str:
            return f"<{field_name}>"
        if annotation is list:
            return []
        if annotation is dict:
            return {}
        return f"<{field_name}>"
    if origin in (list, tuple, set, frozenset):
        return []
    if origin is dict:
        return {}
    # Optional[X] / Union[X, None] — recurse on the first non-None arg.
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if args:
        return _placeholder_for_annotation(args[0], field_name)
    return f"<{field_name}>"


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
        max_schema_rebound: int = 2,
    ) -> None:
        self.max_rebound = max_rebound
        # Schema-violation rebound is tighter than runtime-error rebound:
        # a model that has seen the schema with an example and still gets
        # the call shape wrong is hallucinating, not retrying productively.
        # Let LangGraph surface the underlying error after this many tries.
        self.max_schema_rebound = max_schema_rebound
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
            # Empty args — check if the tool requires parameters. Use the
            # JSON-Schema-aware resolver so MCP tools (which wrap real
            # fields under a single RootModel ``root`` field) report the
            # user-facing field names, not the internal wrapper.
            fields = _resolve_schema_fields(tool)
            required_fields = [name for (name, _t, req) in fields if req]
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

    # ── Hardened guards (markup / whitespace) ─────────────────────────

    _MARKUP_TOKENS: tuple[str, ...] = (
        "<tool_call>", "</tool_call>",
        "<arg_value>", "</arg_value>",
        "<tool_response>", "</tool_response>",
    )

    @classmethod
    def _markup_in(cls, value: str) -> str | None:
        for tok in cls._MARKUP_TOKENS:
            if tok in value:
                return tok
        return None

    @classmethod
    def _classify_validation(
        cls, args: dict[str, Any], exc: Exception | None
    ) -> list[str]:
        """Return short telemetry tags describing what went wrong.

        Tags are non-exclusive: a single failed call can have any combination
        of ``unknown_keys``, ``whitespace_value``, ``markup_leak``, and
        ``pydantic_other`` (catch-all for cases the more-specific tags don't
        cover). Used by the rebound logger to build a per-tool failure
        histogram across runs.
        """
        tags: list[str] = []
        if exc is not None:
            errors_fn = getattr(exc, "errors", None)
            error_list = errors_fn() if callable(errors_fn) else []
            if any(e.get("type") == "extra_forbidden" for e in error_list):
                tags.append("unknown_keys")
        for v in (args or {}).values():
            if not isinstance(v, str):
                continue
            if v != "" and not v.strip():
                tags.append("whitespace_value")
            if cls._markup_in(v):
                tags.append("markup_leak")
        if not tags:
            tags.append("pydantic_other")
        # De-duplicate while preserving order.
        seen: set[str] = set()
        return [t for t in tags if not (t in seen or seen.add(t))]

    def _scan_value_guards(
        self, args: dict[str, Any]
    ) -> tuple[list[str], list[str], list[str]]:
        """Inspect top-level string args for whitespace-only or markup-leak values.

        Returns three lists of (already-formatted) callout strings:
        whitespace_callouts, markup_callouts, classifications.
        Classifications are short tags used for telemetry.
        """
        ws_callouts: list[str] = []
        markup_callouts: list[str] = []
        tags: list[str] = []
        for k, v in args.items():
            if not isinstance(v, str):
                continue
            if v != "" and not v.strip():
                ws_callouts.append(
                    f"  - Field '{k}': value is whitespace-only ({v!r}). "
                    f"Pass a clean identifier like 'MyClass.my_method' or "
                    f"'my_function'."
                )
                tags.append("whitespace_value")
            tok = self._markup_in(v)
            if tok is not None:
                markup_callouts.append(
                    f"  - Field '{k}': value contains tool-call markup "
                    f"({tok!r}). Pass only the raw identifier / regex; do "
                    f"not echo back the tool-call envelope."
                )
                tags.append("markup_leak")
        return ws_callouts, markup_callouts, tags

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

        # ── Invented-key callout (Pydantic 'extra_forbidden') ──
        # The default Pydantic message says 'Extra inputs are not permitted',
        # which the model frequently misreads as advice to add more inputs.
        # Lead with an explicit "you used a key that doesn't exist" line and
        # the list of valid parameters so the correction is unambiguous.
        invented_keys = [
            str(err.get("loc", [""])[0])
            for err in error_list
            if err.get("type") == "extra_forbidden" and err.get("loc")
        ]
        telemetry_tags: list[str] = []
        if invented_keys:
            valid_field_names = [
                fname for (fname, _t, _req) in (_resolve_schema_fields(tool) or [])
            ]
            parts.append(
                f"You used unknown parameter(s) {invented_keys!r}. "
                f"Valid parameters: {valid_field_names}. "
                f"Replace the unknown keys with valid ones — do not add extra "
                f"fields and do not invent new keys."
            )
            telemetry_tags.append("unknown_keys")

        # ── Whitespace / markup-leak guards on supplied string values ──
        ws_callouts, markup_callouts, value_tags = self._scan_value_guards(args)
        if ws_callouts:
            parts.append("Value guard — whitespace-only string fields:")
            parts.extend(ws_callouts)
        if markup_callouts:
            parts.append("Value guard — tool-call markup leaked into a field:")
            parts.extend(markup_callouts)
        telemetry_tags.extend(value_tags)

        for err in error_list[:_MAX_VALIDATION_ERRORS]:
            loc = " -> ".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "unknown error")
            err_type = err.get("type", "")
            parts.append(f"  - Field '{loc}': {msg} (type={err_type})")
        if len(error_list) > _MAX_VALIDATION_ERRORS:
            parts.append(f"  … {len(error_list) - _MAX_VALIDATION_ERRORS} more")

        # Provide the valid schema fields so the model knows what to use.
        # Uses the JSON-Schema-aware resolver so MCP-wrapped tools surface
        # real field names rather than the RootModel ``root`` wrapper.
        fields = _resolve_schema_fields(tool)
        if fields:
            valid_fields = []
            for fname, ftype, required in fields[:_MAX_SCHEMA_FIELDS]:
                required_str = "required" if required else "optional"
                valid_fields.append(f"  {fname}: {ftype} ({required_str})")
            parts.append("Valid parameters:")
            parts.extend(valid_fields)
            if len(fields) > _MAX_SCHEMA_FIELDS:
                parts.append(f"  … {len(fields) - _MAX_SCHEMA_FIELDS} more")

        # Synthesise a concrete example call from the schema. Field-level
        # error lists are abstract; an example call shows the model the
        # exact JSON shape and tends to short-circuit hallucinated shapes.
        example = self._synthesize_example_call(tool)
        if example:
            parts.append("Example valid call:")
            parts.append(f"  {example}")

        parts.append("Retry with the correct parameter names and types.")
        return "\n".join(parts)

    def _synthesize_example_call(self, tool: Any) -> str | None:
        """Build a JSON example of a valid call from the tool's schema.

        Emits a placeholder value for each required field based on its
        annotation: ``str`` → ``"<field_name>"``, ``int``/``float`` → 0,
        ``bool`` → false, list/dict → [] / {}. Works for both Pydantic
        BaseModels and JSON-Schema-dict args_schemas (MCP tools).
        Returns ``None`` if no schema is available or there are no
        required fields.
        """
        fields = _resolve_schema_fields(tool)
        if not fields:
            return None
        example: dict[str, Any] = {}
        for fname, ftype, required in fields:
            if not required:
                continue
            example[fname] = _placeholder_for_annotation(ftype, fname)
        if not example:
            return None
        try:
            return json.dumps(example)
        except Exception:
            return None

    def _format_runtime_error(self, tool_name: str, exc: Exception) -> str:
        """Format a runtime tool execution error.

        Keeps the output terse and free of tracebacks. The model only needs
        the failure reason and a hint to retry — embedding a full
        ``str(exc)`` (which often contains a multi-line traceback or schema
        dump) just pollutes the conversation and ends up serialized into
        findings.
        """
        return (
            f"Tool '{tool_name}' execution failed: "
            f"{_summarize_exception(exc)}. "
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
        tags = self._classify_validation(args, _exc)
        if rebound > self.max_schema_rebound:
            logger.warning(
                "Tool '%s' schema rebound limit (%d) exceeded [tags=%s], "
                "letting handler run (will likely raise): %s",
                tool_name,
                self.max_schema_rebound,
                tags,
                error,
            )
            return handler(request)

        logger.warning(
            "Tool '%s' validation failed (rebound %d/%d) [tags=%s]",
            tool_name,
            rebound,
            self.max_schema_rebound,
            tags,
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

        # Pre-validation: edit_file with empty old_string is always an error
        if tool_name == "edit_file":
            old_string = args.get("old_string", "")
            if not old_string:
                rebound = self._increment_rebound(tool_name)
                if rebound > self.max_rebound:
                    return await handler(request)
                return self._make_error_message(
                    tool_name,
                    tool_call_id,
                    "edit_file: old_string cannot be empty — it matches every "
                    "location in the file (2308+ matches observed in production). "
                    "Use write_file instead if you want to replace the entire file "
                    "content.",
                )

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
        tags = self._classify_validation(args, _exc)
        if rebound > self.max_schema_rebound:
            logger.warning(
                "Tool '%s' schema rebound limit (%d) exceeded [tags=%s], "
                "letting handler run (will likely raise): %s",
                tool_name,
                self.max_schema_rebound,
                tags,
                error,
            )
            return await handler(request)

        logger.warning(
            "Tool '%s' validation failed (rebound %d/%d) [tags=%s]",
            tool_name,
            rebound,
            self.max_schema_rebound,
            tags,
        )

        return self._make_error_message(tool_name, tool_call_id, error)
