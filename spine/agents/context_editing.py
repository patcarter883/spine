"""SPINE context editing middleware — read cache and tool output trimming.

DA's built-in SummarizationMiddleware was removed (2026-05) — it was doing
more harm than good, compressing context at inconvenient times and losing
critical working state.

The ReadCacheMiddleware replaces summarization: before every read_file call
it checks a short-term cache in SpineContext. If the file was already read
this phase, it returns a compact metadata summary instead of full content.
The cache is shared across subagents, preventing re-read amnesia without
changing the eviction strategy. ToolOutputTrimmer provides LRU-style
eviction markers for old tool outputs.

Strategy: When tool result count exceeds `max_full_tool_results`, replace
old tool call results with a structured metadata placeholder that preserves
the key information an agent needs without the full content. Additionally,
trim large arguments in AI messages that correspond to evicted tool results
(e.g., write_file content, edit_file old/new strings).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from spine.agents import symbol_cache

# Tools (beyond read_file) whose results we deduplicate when called with
# identical arguments inside a single phase / subagent invocation. Read-only
# lookup tools are safe to short-circuit; anything that mutates state must
# never appear here. The MCP codebase-index tools are dispatched under the
# ``mcp_codebase-index_*`` prefix; rather than enumerating each one, the
# middleware also dedupes any tool whose name starts with ``mcp_``.
_DEDUPED_TOOLS: frozenset[str] = frozenset({
    "search_codebase",
    "ast_extract_symbol",
    "glob",
    "grep",
})


def _is_dedupable(tool_name: str) -> bool:
    """Return True if repeated calls to *tool_name* with the same args should
    return a cached placeholder instead of re-executing.

    Covers explicit read-only research tools plus every MCP tool. MCP tools
    are by convention named ``mcp_<server>_<tool>`` (e.g. the codebase-index
    server exposes ``mcp_codebase-index_find_symbol``); they are all
    read-only lookups in the SPINE configuration.
    """
    if tool_name in _DEDUPED_TOOLS:
        return True
    return tool_name.startswith("mcp_")


def _args_fingerprint(args: dict) -> str:
    """Stable short hash of a tool call's arguments for cache keying."""
    try:
        canonical = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        canonical = repr(sorted(args.items()))
    return hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol extraction helpers
# ---------------------------------------------------------------------------

_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")
_CLASS_RE = re.compile(r"^\s*class\s+(\w+)")


def _extract_symbols(content: str, max_symbols: int = 5) -> list[str]:
    """Extract top-level function/class names from file content."""
    symbols: list[str] = []
    for line in content.splitlines():
        m = _DEF_RE.match(line)
        if m:
            symbols.append(f"def {m.group(1)}")
            if len(symbols) >= max_symbols:
                break
        m = _CLASS_RE.match(line)
        if m:
            symbols.append(f"class {m.group(1)}")
            if len(symbols) >= max_symbols:
                break
    return symbols


def _count_lines(content: str) -> int:
    """Count the number of lines in *content*."""
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)


# ---------------------------------------------------------------------------
# ReadCacheMiddleware
# ---------------------------------------------------------------------------


class ReadCacheMiddleware(AgentMiddleware):
    """Short-circuit re-issued read-only tool calls.

    Two cache paths share ``SpineContext.read_cache``:

    1. ``read_file`` — keyed by file path; the cached placeholder carries
       a symbol summary so the model can still see what the file contains
       without paying the re-read cost.
    2. Other read-only lookup tools (``search_codebase``, ``ast_extract_symbol``,
       ``glob``, ``grep``, and every ``mcp_*`` tool) — keyed by a hash of the
       call's arguments.  On a hit the placeholder tells the model the
       query was already issued so it should not re-run it verbatim.

    The cache is shared across subagents via ``SpineContext`` propagation.
    It is intentionally never invalidated within a phase — duplicate calls
    inside one explore loop are the bug we are guarding against; tools that
    mutate state (write_file, edit_file, execute, …) are not on the dedupe
    list and run untouched.
    """

    async def awrap_tool_call(self, request, handler):
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {}) or {}

        ctx = request.runtime.context
        if ctx is None:
            return await handler(request)

        # ── read_file path: keyed by file_path, stores symbol summary ──
        if tool_name == "read_file":
            file_path = args.get("file_path", "")
            if not file_path:
                return await handler(request)

            ctx.read_cache_turn += 1
            turn = ctx.read_cache_turn

            cache: dict = ctx.read_cache
            if file_path in cache:
                entry = cache[file_path]
                return ToolMessage(
                    content=(
                        f"[cached: {file_path} "
                        f"({entry['n_lines']} lines) — "
                        f"read at turn {entry['turn']} — "
                        f"{entry['symbols']}]"
                    ),
                    tool_call_id=request.tool_call["id"],
                    name="read_file",
                )

            result = await handler(request)

            content: str = ""
            if isinstance(result, ToolMessage) and isinstance(result.content, str):
                content = result.content
            elif hasattr(result, "content") and isinstance(result.content, str):
                content = result.content

            if content:
                n_lines = _count_lines(content)
                symbols = _extract_symbols(content)
                cache[file_path] = {
                    "n_lines": n_lines,
                    "symbols": ", ".join(symbols) if symbols else "no symbols",
                    "turn": turn,
                }

            return result

        # ── generic lookup-tool dedupe (search_codebase, MCP, …) ──
        if _is_dedupable(tool_name):
            ctx.read_cache_turn += 1
            turn = ctx.read_cache_turn
            cache = ctx.read_cache
            key = f"call::{tool_name}::{_args_fingerprint(args)}"

            # ── Cross-branch dedupe via symbol_cache ──────────────────
            # Deterministic codebase-index lookups are memoised at the
            # work_id level so sibling explore branches running in the
            # same super-step share results instead of each re-issuing
            # the same MCP call. The per-branch SpineContext cache below
            # still serves intra-branch repeat-call detection.
            if symbol_cache.is_cacheable(tool_name) and getattr(ctx, "work_id", ""):
                async def _fetch() -> str | None:
                    res = await handler(request)
                    if isinstance(res, ToolMessage) and isinstance(res.content, str):
                        return res.content
                    if hasattr(res, "content") and isinstance(res.content, str):
                        return res.content
                    return None

                cached_content = await symbol_cache.get_or_fetch(
                    ctx.work_id, tool_name, args, _fetch,
                )
                if cached_content is not None:
                    if key not in cache:
                        cache[key] = {
                            "summary": f"{_count_lines(cached_content)} lines",
                            "turn": turn,
                        }
                    return ToolMessage(
                        content=cached_content,
                        tool_call_id=request.tool_call["id"],
                        name=tool_name,
                    )
                # Fall through: non-string payload from the handler —
                # let the regular path run so the model still sees it.

            if key in cache:
                entry = cache[key]
                return ToolMessage(
                    content=(
                        f"[cached: {tool_name} — issued at turn {entry['turn']} — "
                        f"{entry['summary']}. Re-issue avoided; consult the "
                        f"earlier result above or change the arguments.]"
                    ),
                    tool_call_id=request.tool_call["id"],
                    name=tool_name,
                )

            result = await handler(request)

            content = ""
            if isinstance(result, ToolMessage) and isinstance(result.content, str):
                content = result.content
            elif hasattr(result, "content") and isinstance(result.content, str):
                content = result.content

            if content:
                summary = f"{_count_lines(content)} lines"
                cache[key] = {"summary": summary, "turn": turn}

            return result

        return await handler(request)

    def wrap_tool_call(self, request, handler):
        # SPINE uses async invoke; sync pass-through only.
        return handler(request)


# ---------------------------------------------------------------------------
# ToolOutputTrimmer
# ---------------------------------------------------------------------------


class ToolOutputTrimmer(AgentMiddleware):
    """Trims old tool outputs from the conversation to keep context lean.

    Replaces old tool result content with a structured metadata placeholder
    when the tool result count exceeds the threshold. Also trims large
    arguments in AI messages whose corresponding tool results were evicted.

    Design: treats context as L1 cache. Evicted content lives in the
    offloaded conversation history (swap) and can be paged back via
    read_file if needed.

    Only intercepts model calls (``awrap_model_call``) — tool call
    wrapping is passed through unchanged.
    """

    def __init__(
        self,
        max_full_tool_results: int = 20,
    ) -> None:
        self.max_full_tool_results = max_full_tool_results

    # ── Metadata extraction ──────────────────────────────────────────────

    def _extract_metadata(self, content: str, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Produce a structured placeholder from a tool result.

        The placeholder encodes enough information for the agent to know
        *what* was read/written without needing to re-execute the tool.
        """
        if tool_name == "read_file":
            path = tool_args.get("file_path", "?")
            n_lines = _count_lines(content)
            symbols = _extract_symbols(content)
            sym_str = ", ".join(symbols) if symbols else "no symbols"
            return f"[read: {path} ({n_lines} lines) — {sym_str}]"

        if tool_name == "execute":
            cmd = tool_args.get("command", "?")
            # Try to find exit code in the content
            last_line = content.rstrip().rsplit("\n", 1)[-1] if content.strip() else ""
            # Look for common exit-code patterns
            exit_match = re.search(r"(?:exit code|Exit code|exit_status)[:\s]*(\d+)", content)
            if exit_match:
                return f"[exec: {cmd} — exit code {exit_match.group(1)}]"
            if last_line and len(last_line) < 120:
                return f"[exec: {cmd} — {last_line}]"
            return f"[exec: {cmd}]"

        if tool_name == "grep":
            pattern = tool_args.get("pattern", "?")
            path = tool_args.get("path", ".")
            # Count matching files and lines
            file_matches = re.findall(r"^([^:\n]+):", content, re.MULTILINE)
            n_files = len(set(file_matches))
            n_lines = _count_lines(content)
            return f"[grep: '{pattern}' in {path} — {n_files} files, {n_lines} lines]"

        if tool_name == "write_file":
            path = tool_args.get("file_path", "?")
            return f"[written: {path}]"

        if tool_name == "edit_file":
            path = tool_args.get("file_path", "?")
            new_str = tool_args.get("new_string", "")
            preview = new_str[:60]
            if len(new_str) > 60:
                preview += "..."
            return f"[edited: {path} → {preview}]"

        if tool_name == "glob":
            pattern = tool_args.get("pattern", "?")
            n_files = _count_lines(content)
            return f"[glob: '{pattern}' — {n_files} files]"

        if tool_name == "ls":
            path = tool_args.get("path", ".")
            entries = [line for line in content.splitlines() if line.strip()]
            return f"[ls: {path} — {len(entries)} entries]"

        # Default fallback
        hint = content[:80].split("\n")[0]
        if len(hint) == 80 and len(content) > 80:
            hint += "..."
        return f"[evicted({tool_name}): {hint}]"

    # ── AI message argument trimming ─────────────────────────────────────

    def _trim_ai_args(self, messages: list, evicted_ids: set[str]) -> list:
        """Trim tool_call arguments in AI messages for evicted tool results.

        For write_file / edit_file calls whose ToolMessage was evicted,
        replace the large argument values with compact summaries.
        """
        trimmed = list(messages)
        for i, msg in enumerate(trimmed):
            if not isinstance(msg, AIMessage):
                continue
            if not msg.tool_calls:
                continue
            new_calls: list[dict] | None = None
            for j, tc in enumerate(msg.tool_calls):
                tc_id = tc.get("id", "")
                if tc_id not in evicted_ids:
                    continue
                name = tc.get("name", "")
                args = tc.get("args", {})
                new_args = self._trim_args(name, args)
                if new_args is not args:
                    if new_calls is None:
                        new_calls = [dict(c) for c in msg.tool_calls]
                    new_calls[j] = {**new_calls[j], "args": new_args}
            if new_calls is not None:
                trimmed[i] = AIMessage(
                    content=msg.content,
                    tool_calls=new_calls,
                )
        return trimmed

    @staticmethod
    def _trim_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Return trimmed args for a single tool call, or *args* unchanged."""
        if name == "write_file":
            content_val = args.get("content", "")
            path = args.get("file_path", "?")
            if isinstance(content_val, str) and len(content_val) > 100:
                return {**args, "content": f"[{len(content_val)} chars written to {path}]"}
            return args

        if name == "edit_file":
            new_args = dict(args)
            path = args.get("file_path", "?")
            old_str = args.get("old_string", "")
            new_str = args.get("new_string", "")
            if isinstance(old_str, str) and len(old_str) > 100:
                new_args["old_string"] = f"[{len(old_str)} chars from {path}]"
            if isinstance(new_str, str) and len(new_str) > 100:
                new_args["new_string"] = f"[{len(new_str)} chars → {path}]"
            return new_args

        # Don't trim read_file args — they're small
        return args

    # ── Build tool_call_id → (name, args) map ───────────────────────────

    @staticmethod
    def _build_tool_call_map(messages: list) -> dict[str, tuple[str, dict]]:
        """Map tool_call_id → (tool_name, tool_args) from AI messages."""
        call_map: dict[str, tuple[str, dict]] = {}
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    call_map[tc_id] = (tc.get("name", ""), tc.get("args", {}))
        return call_map

    # ── Main model-call hook ─────────────────────────────────────────────

    async def awrap_model_call(self, request, handler):
        """Trim old tool results and AI args before each model call."""
        messages = request.messages

        # Build tool_call_id → (name, args) map from AI messages
        call_map = self._build_tool_call_map(messages)

        # Count tool results in the message list
        tool_result_indices: list[int] = []
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage):
                tool_result_indices.append(i)

        # If within budget, pass through unchanged
        if len(tool_result_indices) <= self.max_full_tool_results:
            return await handler(request)

        # Trim old results — keep the last N in full
        trim_count = len(tool_result_indices) - self.max_full_tool_results
        evicted_ids: set[str] = set()
        trimmed_messages = list(messages)

        for idx in tool_result_indices[:trim_count]:
            msg = trimmed_messages[idx]
            assert isinstance(msg, ToolMessage)
            tc_id = msg.tool_call_id
            name = msg.name or ""
            args: dict = {}
            if tc_id and tc_id in call_map:
                name, args = call_map[tc_id]

            content = msg.content if isinstance(msg.content, str) else ""
            metadata = self._extract_metadata(content, name, args)

            evicted_ids.add(tc_id or "")
            try:
                trimmed_messages[idx] = ToolMessage(
                    content=metadata,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            except Exception:
                pass

        # Trim AI message args for evicted tool_call_ids
        if evicted_ids:
            trimmed_messages = self._trim_ai_args(trimmed_messages, evicted_ids)

        return await handler(request.override(messages=trimmed_messages))

    # ── Pass-through tool call wrapping ────────────────────────────────
    # This middleware only intercepts model calls. Tool calls must still
    # be defined so DA's factory check (``m.__class__.wrap_tool_call is
    # not AgentMiddleware.wrap_tool_call``) passes without AttributeError.

    def wrap_tool_call(self, request, handler):
        """Pass-through: no tool-call interception needed."""
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        """Pass-through: no tool-call interception needed."""
        return await handler(request)
