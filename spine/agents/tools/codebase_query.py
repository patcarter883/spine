"""Consolidated codebase-query tool for the researcher subagent.

Replaces the five separately-bound ``mcp_codebase-index_*`` tools the
researcher previously had access to with a single ``BaseTool`` exposing
an ``action`` enum. The model picks one of five actions, fills the one
or two argument fields the action needs, and the tool dispatches to the
matching MCP backend.

Why this exists
---------------
Trace ``019e6cc4-f57d-7652-a718-15d04278ad5c`` showed the local researcher
model emitting malformed MCP args at a high rate: invented argument keys
(``maxfile_patterns``, ``queries`` instead of ``pattern``), whitespace-only
``name`` values (``"test\\n"``), and tool-call markup leaking into regex
patterns (``"spine.*__init__|arg_value>\\n</tool_call>\\n"``). Branches
exhausted the 50-step recursion cap before producing findings; 23/23 real
research branches failed.

Collapsing five tool schemas into one (with one ``action`` decision plus a
small, named arg set) removes the wrong-key class of failures entirely,
and the markup/whitespace guards short-circuit the obvious garbage at
validator level before it reaches the MCP server.

This module is the ONLY sanctioned entry point to codebase indexing:
no other production code may invoke raw ``mcp_codebase-index_*`` tools
or import :mod:`spine.agents.tools.codebase_query_local` directly. Agents
get :class:`CodebaseQueryTool`; programmatic callers (e.g. the onboarding
analyzer) use :func:`list_files`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


# Markup substrings that must never appear inside model-supplied strings.
# These indicate the model has spilled its own tool-call XML into a value.
_FORBIDDEN_MARKUP_TOKENS: tuple[str, ...] = (
    "<tool_call>",
    "</tool_call>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
)


CodebaseQueryAction = Literal[
    "find_symbol",
    "get_source",
    "get_dependencies",
    "get_dependents",
    "search",
]


# Literal null-word placeholders a small model emits for an optional field it
# means to leave empty (e.g. name="None"). Compared case-insensitively after
# stripping; see :func:`_blank_to_none`. Deliberately excludes the empty /
# whitespace-only string so a blank value on the field an action actually USES
# still gets its specific "whitespace-only" validation message.
_NULLISH_TOKENS: frozenset[str] = frozenset({"none", "null"})


# action → backing MCP tool name on the codebase-index server.
_ACTION_TO_MCP: dict[str, str] = {
    "find_symbol":      "mcp_codebase-index_find_symbol",
    "get_source":       "mcp_codebase-index_get_function_source",
    "get_dependencies": "mcp_codebase-index_get_dependencies",
    "get_dependents":   "mcp_codebase-index_get_dependents",
    "search":           "mcp_codebase-index_search_codebase",
}


# "No results" phrasings the MCP server uses; checked only on short
# responses so a real result mentioning these words is never swallowed.
_EMPTY_RESULT_MARKERS: tuple[str, ...] = (
    "not found", "no results", "no matches", "no match", "no symbols",
)


def _looks_empty(result: Any) -> bool:
    """True when an MCP response carries no usable content.

    mcp-codebase-index has no PHP analyzer, so PHP queries come back
    empty / "not found" rather than erroring — that is the signal to try
    the local index instead.
    """
    if result is None:
        return True
    text = result if isinstance(result, str) else str(result)
    stripped = text.strip()
    if not stripped or stripped in ("[]", "{}", "null"):
        return True
    if len(stripped) < 200:
        lowered = stripped.lower()
        return any(marker in lowered for marker in _EMPTY_RESULT_MARKERS)
    return False


def _parse_tool_result(result: Any) -> list[Any]:
    """Normalise an MCP tool result to a list of entries.

    Mirrors ``VectorIndexer._parse_tool_result`` — handles the LangChain
    ``[{"type": "text", "text": "[...json...]"}]`` envelope, plain JSON
    strings, and ``{"items": [...]}``-style dicts.
    """
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict) and "text" in result[0]:
            text = result[0]["text"]
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                return [text] if text else []
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            return [result] if result else []
    if isinstance(result, dict):
        for key in ("items", "results", "symbols", "functions", "classes", "files"):
            if key in result:
                return result[key]
        return [result]
    return []


async def list_files(
    workspace_root: str,
    mcp_servers: dict[str, dict[str, Any]] | None,
    *,
    extensions: frozenset[str],
    skip_dirs: frozenset[str] = frozenset(),
) -> list[str]:
    """List repo-relative source files matching *extensions*.

    Index-first with a per-extension local supplement: the MCP server's
    ``list_files`` only covers languages it can analyze (no PHP), so any
    requested extension with zero MCP hits is filled in from a local walk
    instead of being silently dropped. Dot-folder paths are excluded from
    every return path.
    """
    from spine.agents.tools.codebase_query_local import local_list_files
    from spine.mcp.client import _is_excluded_path, get_mcp_tools

    mcp_files: list[str] = []
    if mcp_servers:
        try:
            tools = get_mcp_tools(
                mcp_servers,
                cache_key=f"list-files-{workspace_root}",
                workspace_root=workspace_root,
            )
            tool = next(
                (t for t in tools if t.name == "mcp_codebase-index_list_files"), None
            )
            if tool is not None:
                raw = await tool.ainvoke({"root": workspace_root})
                mcp_files = [
                    f
                    for f in _parse_tool_result(raw)
                    if isinstance(f, str)
                    and os.path.splitext(f)[1].lower() in extensions
                    and not _is_excluded_path(f.replace("\\", "/"))
                    and not any(
                        part in skip_dirs for part in f.replace("\\", "/").split("/")
                    )
                ]
            else:
                logger.info("codebase_query.list_files: MCP list_files unavailable")
        except Exception as exc:
            logger.info("codebase_query.list_files: index discovery failed: %s", exc)

    covered = {os.path.splitext(f)[1].lower() for f in mcp_files}
    uncovered = extensions - covered
    local_files: list[str] = []
    if uncovered:
        local_files = local_list_files(workspace_root, frozenset(uncovered), skip_dirs)
    return sorted({f.replace("\\", "/") for f in mcp_files}
                  | {f.replace("\\", "/") for f in local_files})


class CodebaseQueryInput(BaseModel):
    """Single schema covering every action of :class:`CodebaseQueryTool`.

    Validation rules (enforced in :meth:`CodebaseQueryTool._arun`):

    * ``find_symbol`` / ``get_source`` / ``get_dependencies`` /
      ``get_dependents`` all require ``name`` (a non-whitespace identifier).
    * ``search`` requires ``pattern`` (a non-whitespace regex).
    * ``name`` and ``pattern`` are mutually exclusive — supply only the
      one the chosen action needs.
    * Neither field may contain tool-call markup substrings.
    """

    action: CodebaseQueryAction = Field(
        description=(
            "One of: 'find_symbol' (locate a symbol's definition), "
            "'get_source' (full source body of a function/method/class), "
            "'get_dependencies' (what a symbol calls/uses), "
            "'get_dependents' (what calls/uses a symbol), "
            "'search' (regex search across the codebase)."
        ),
    )
    name: str | None = Field(
        default=None,
        description=(
            "Symbol identifier (function, class, or method name). REQUIRED "
            "for find_symbol, get_source, get_dependencies, get_dependents. "
            "For action='search', OMIT this field entirely — do not pass the "
            "string 'None' or 'null'. Must be a clean identifier — no "
            "whitespace, no module prefix, no parentheses. Do not pass any "
            "other arguments (e.g. file_hint) — they are ignored."
        ),
    )
    pattern: str | None = Field(
        default=None,
        description=(
            "Regex pattern for the 'search' action. REQUIRED for search; "
            "must not be used with other actions. Output is capped to "
            "~8 KB / 50 hits — refine with anchors/file-globs rather than "
            "retrying naively."
        ),
    )
    max_results: int = Field(
        default=20,
        description="Result-count cap for the 'search' action (default 20).",
    )


def _reject_markup(value: str, field: str) -> None:
    """Raise ``ToolException`` if *value* contains tool-call markup."""
    for token in _FORBIDDEN_MARKUP_TOKENS:
        if token in value:
            raise ToolException(
                f"codebase_query: '{field}' contains tool-call markup ({token!r}). "
                f"Pass only the raw identifier / regex; do not echo back the "
                f"<tool_call> envelope."
            )


def _normalise_name(value: str | None, action: str = "") -> str:
    """Strip whitespace from a symbol name and validate it is non-empty."""
    if value is None:
        raise ToolException(
            f"codebase_query: 'name' is required for action={action!r}. "
            f"Retry with name='<symbol_identifier>' (e.g. name='MyClass.my_method')."
        )
    stripped = value.strip()
    if not stripped:
        raise ToolException(
            "codebase_query: 'name' is whitespace-only. Pass a clean symbol "
            "identifier like 'MyClass.my_method' or 'my_function'."
        )
    _reject_markup(stripped, "name")
    return stripped


def _normalise_pattern(value: str | None) -> str:
    """Validate the regex pattern for the 'search' action."""
    if value is None:
        raise ToolException(
            "codebase_query: 'pattern' is required for action='search'. "
            "Retry with pattern='<regex>' (e.g. pattern='def my_function')."
        )
    if not value.strip():
        raise ToolException(
            "codebase_query: 'pattern' is empty or whitespace-only."
        )
    _reject_markup(value, "pattern")
    return value


def _blank_to_none(value: Any) -> Any:
    """Coerce a null-ish placeholder string to ``None``.

    Small local models routinely fill an optional field they mean to OMIT
    with the literal string ``"None"`` / ``"null"`` instead of leaving it
    out. Trace ``019ed870`` sank an otherwise valid
    ``action='search'`` because the model emitted ``name="None"`` next to a
    well-formed ``pattern`` — the mutual-exclusivity guard saw a non-``None``
    ``name`` and rejected the call, and the agent retried the identical args.
    Mapping these placeholders to ``None`` lets the guard treat the field as
    absent. Non-string values pass through untouched.
    """
    if not isinstance(value, str):
        return value
    return None if value.strip().lower() in _NULLISH_TOKENS else value


def resolve_backing_call(
    action: str,
    name: str | None,
    pattern: str | None,
    max_results: int = 20,
) -> tuple[str, dict[str, Any]]:
    """Map facade args → ``(backing MCP tool name, backing args)``.

    Single source of truth for both dispatch (``CodebaseQueryTool``) and
    cache keying (:func:`canonical_backing_call`) so the two can never
    diverge. Raises ``ToolException`` on invalid input.
    """
    # Treat null-ish placeholder strings ("None"/"null"/"") as field-absent so
    # a stray placeholder on the unused field doesn't trip mutual exclusivity.
    name = _blank_to_none(name)
    pattern = _blank_to_none(pattern)
    if action == "search":
        if name is not None:
            raise ToolException(
                "codebase_query: 'name' must not be supplied for action='search'; "
                "use 'pattern' instead."
            )
        return _ACTION_TO_MCP[action], {
            "pattern": _normalise_pattern(pattern),
            "max_results": max(1, int(max_results)),
        }
    # All other actions take a symbol name.
    if action not in _ACTION_TO_MCP:
        raise ToolException(
            f"codebase_query: unknown action {action!r}. "
            f"Available actions: {sorted(_ACTION_TO_MCP)}."
        )
    if pattern is not None:
        raise ToolException(
            f"codebase_query: 'pattern' must not be supplied for action={action!r}; "
            f"use 'name' instead."
        )
    return _ACTION_TO_MCP[action], {"name": _normalise_name(name, action)}


def canonical_backing_call(args: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Resolve raw facade args to the backing MCP call for dedupe keying.

    ``symbol_cache`` fingerprints raw tool args, so superficially different
    facade calls — e.g. a hallucinated ``file_hint`` key (silently dropped
    by Pydantic ``extra="ignore"``) or an explicit default ``max_results`` —
    produced distinct cache keys for the *identical* backing MCP call
    (trace 019eafac: ``get_source(render)`` fetched 3×). Keying on the
    resolved backing call coalesces every variant, and also shares entries
    with any path that still issues raw ``mcp_codebase-index_*`` calls.

    Returns ``None`` when the args don't validate — the caller falls back
    to raw-key behaviour (error responses are never worth memoising under
    a canonical key anyway).
    """
    try:
        return resolve_backing_call(
            args["action"],
            args.get("name"),
            args.get("pattern"),
            int(args.get("max_results", 20)),
        )
    except Exception:
        return None


class CodebaseQueryTool(BaseTool):
    """One tool, five actions — wraps the codebase-index MCP backend.

    Construction loads the MCP tool set once (via
    :func:`spine.mcp.client.get_mcp_tools`) and dispatches by action.
    Failures during MCP loading are surfaced at first call time rather
    than at construction so the rest of the researcher's tool list can
    still be built when the index is unavailable.
    """

    name: str = "codebase_query"
    description: str = (
        "Structural codebase navigation. ONE tool covering every kind of "
        "lookup the researcher needs:\n"
        "  - action='find_symbol' + name='X'        → locate symbol X (file, line, type)\n"
        "  - action='get_source' + name='X'         → full source of function/method/class X\n"
        "  - action='get_dependencies' + name='X'   → what X calls/uses\n"
        "  - action='get_dependents' + name='X'     → what calls/uses X\n"
        "  - action='search' + pattern='regex'      → regex search across the codebase\n"
        "Pick the action FIRST, then fill the one argument it needs. Use "
        "'name' for the four symbol actions and 'pattern' for 'search' — "
        "they are mutually exclusive. Sub-millisecond latency; use this "
        "before reading whole files. Covers Python/TS/Go/Rust via the "
        "structural index and PHP via the local symbol index."
    )
    args_schema: ArgsSchema | None = CodebaseQueryInput

    workspace_root: str = ""
    db_path: str = ".spine/spine.db"

    # Lazily-loaded MCP tool map: { mcp_tool_name → BaseTool }.
    _tool_map: dict[str, BaseTool] | None = PrivateAttr(default=None)
    _server_configs: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        *,
        workspace_root: str,
        mcp_servers: dict[str, dict[str, Any]],
        db_path: str = ".spine/spine.db",
    ):
        super().__init__(workspace_root=workspace_root, db_path=db_path)
        # Store the server config; the actual tool load happens on first use.
        self._server_configs = mcp_servers or {}

    # ── Lazy backend loader ─────────────────────────────────────────────

    def _ensure_loaded(self) -> dict[str, BaseTool]:
        if self._tool_map is not None:
            return self._tool_map
        from spine.mcp.client import get_mcp_tools

        try:
            tools = get_mcp_tools(
                self._server_configs,
                cache_key=f"codebase-query-{self.workspace_root}",
                workspace_root=self.workspace_root,
            )
        except Exception as exc:
            logger.warning("codebase_query: MCP tool load failed: %s", exc)
            self._tool_map = {}
            return self._tool_map
        mapping = {t.name: t for t in tools}
        self._tool_map = mapping
        missing = [m for m in _ACTION_TO_MCP.values() if m not in mapping]
        if missing:
            logger.warning(
                "codebase_query: backing MCP tools missing: %s (available: %s)",
                missing, sorted(mapping.keys())[:10],
            )
        return mapping

    # ── Action validation + dispatch ────────────────────────────────────

    def _resolve_args(
        self, action: str, name: str | None, pattern: str | None, max_results: int,
    ) -> tuple[str, dict[str, Any]]:
        return resolve_backing_call(action, name, pattern, max_results)

    # NOTE: backing tools are wrapped as ``langchain_core.tools.StructuredTool``
    # by the MCP adapter. Their ``_run`` / ``_arun`` signatures require a
    # ``config: RunnableConfig`` keyword-only argument (it carries tags,
    # metadata, callbacks). Direct calls without ``config`` raise the
    # ``TypeError: ... missing 1 required keyword-only argument: 'config'``
    # observed in trace 019e6d27. We dispatch via the public ``.invoke`` /
    # ``.ainvoke`` API instead — it handles config plumbing and is the
    # documented surface for invoking a BaseTool by argument dict.

    # ── Local-index fallback ────────────────────────────────────────────
    #
    # mcp-codebase-index cannot analyze PHP, but the Phase 1 vector index
    # (symbol_metadata / symbol_fts / symbol_edges in .spine/spine.db)
    # covers it. Three triggers, in order: a symbol the local index knows
    # only as PHP skips MCP entirely; an unloaded MCP backend falls back
    # instead of raising; an empty MCP result falls back before being
    # returned. Fallback responses carry "source": "local_index".

    def _local_fallback(self, action: str, args: dict[str, Any]) -> str | None:
        from spine.agents.tools import codebase_query_local as local

        try:
            if action == "find_symbol":
                return local.local_find_symbol(self.db_path, args["name"])
            if action == "get_source":
                return local.local_get_source(
                    self.db_path, self.workspace_root, args["name"]
                )
            if action == "search":
                return local.local_search(
                    self.db_path, args["pattern"], args["max_results"]
                )
            if action == "get_dependencies":
                return local.local_dependencies(self.db_path, args["name"], "dependencies")
            if action == "get_dependents":
                return local.local_dependencies(self.db_path, args["name"], "dependents")
        except Exception:
            logger.debug("codebase_query: local fallback failed", exc_info=True)
        return None

    def _php_short_circuit(self, action: str, args: dict[str, Any]) -> str | None:
        """Serve PHP-only symbols locally without touching MCP."""
        if action == "search":
            return None
        from spine.agents.tools.codebase_query_local import lookup_local_langs

        if lookup_local_langs(self.db_path, args["name"]) == {"php"}:
            return self._local_fallback(action, args)
        return None

    def _finish(self, action: str, args: dict[str, Any], result: Any) -> str:
        """Apply the empty-result fallback to an MCP response."""
        if _looks_empty(result):
            local = self._local_fallback(action, args)
            if local is not None:
                return local
        return result

    def _backing_tool_or_fallback(
        self, action: str, backing_name: str, args: dict[str, Any]
    ) -> BaseTool | str:
        tool = self._ensure_loaded().get(backing_name)
        if tool is not None:
            return tool
        local = self._local_fallback(action, args)
        if local is not None:
            return local
        raise ToolException(
            f"codebase_query: backing tool {backing_name!r} is not loaded "
            f"(MCP server unavailable?) and the local index has no match. "
            f"Available actions: {sorted(_ACTION_TO_MCP)}."
        )

    def _run(
        self,
        action: str,
        name: str | None = None,
        pattern: str | None = None,
        max_results: int = 20,
    ) -> str:
        backing_name, backing_args = self._resolve_args(
            action, name, pattern, max_results,
        )
        local = self._php_short_circuit(action, backing_args)
        if local is not None:
            return local
        tool = self._backing_tool_or_fallback(action, backing_name, backing_args)
        if isinstance(tool, str):
            return tool
        return self._finish(action, backing_args, tool.invoke(backing_args))

    async def _arun(
        self,
        action: str,
        name: str | None = None,
        pattern: str | None = None,
        max_results: int = 20,
    ) -> str:
        backing_name, backing_args = self._resolve_args(
            action, name, pattern, max_results,
        )
        local = self._php_short_circuit(action, backing_args)
        if local is not None:
            return local
        tool = self._backing_tool_or_fallback(action, backing_name, backing_args)
        if isinstance(tool, str):
            return tool
        return self._finish(action, backing_args, await tool.ainvoke(backing_args))
