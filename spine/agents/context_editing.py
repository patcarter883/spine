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
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from spine.agents import symbol_cache
from spine.agents._tokens import count_tokens
from spine.agents.synthesis_budget import window_hard_ceiling

# Tools (beyond read_file) whose results we deduplicate when called with
# identical arguments inside a single phase / subagent invocation. Read-only
# lookup tools are safe to short-circuit; anything that mutates state must
# never appear here. The MCP codebase-index tools are dispatched under the
# ``mcp_codebase-index_*`` prefix; rather than enumerating each one, the
# middleware also dedupes any tool whose name starts with ``mcp_``.
# ``codebase_query`` is the facade that replaced the raw mcp_ surface for
# subagents (commit b2f60ac) — it must be listed explicitly or the
# middleware skips it entirely.
_DEDUPED_TOOLS: frozenset[str] = frozenset({
    "codebase_query",
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


def _content_sha(content: str) -> str:
    """Short SHA of a tool result string — used in dedupe sentinels so the
    model can recognise "this is the exact same response I already have"
    rather than treating a placeholder as a new opaque result."""
    if not content:
        return "empty"
    return hashlib.sha1(
        content.encode("utf-8", errors="replace"), usedforsecurity=False
    ).hexdigest()[:7]

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


def _stringify_content(content: Any) -> str:
    """Coerce LangChain message content into a plain string.

    Tool results do not always arrive as a bare ``str``. The
    ``codebase_query`` facade and the ``mcp_codebase-index_*`` tools return
    *multimodal* content — a list of blocks like
    ``[{"type": "text", "text": "..."}]``. Eviction and caching previously
    discarded any non-``str`` content (``... if isinstance(c, str) else ""``),
    which produced empty placeholders such as ``[evicted(codebase_query): ]``
    and forced agents to re-issue structural lookups they had already paid
    for (trace 019ed3b8: a single subagent re-read the same 3 files 48×).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _summarize_code_locations(content: str, max_locs: int = 4) -> str:
    """Summarize a ``codebase_query`` result as ``file:line`` references.

    The structural lookup tools return JSON — either a single object
    (``find_symbol`` / ``get_source``) or a list (``search``). We surface the
    file/line of each hit so the evicted placeholder still tells the agent
    *where* the symbol lives, instead of an opaque stub it has to re-query.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        hint = content.strip().split("\n", 1)[0][:80]
        return hint or "no results"
    items = data if isinstance(data, list) else [data]
    if not items:
        return "no results"
    locs: list[str] = []
    for item in items[:max_locs]:
        if not isinstance(item, dict):
            continue
        path = item.get("file") or item.get("file_path") or "?"
        line = item.get("line") or item.get("start_line")
        locs.append(f"{path}:{line}" if line else str(path))
    if not locs:
        return "no results"
    extra = f" (+{len(items) - max_locs} more)" if len(items) > max_locs else ""
    return f"{len(items)} hit(s): " + ", ".join(locs) + extra


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
            if isinstance(result, ToolMessage):
                content = _stringify_content(result.content)
            elif hasattr(result, "content"):
                content = _stringify_content(result.content)

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
                # Capture the handler result in the closure so the middleware
                # can return it directly when _fetch decides not to memoise
                # (error responses, non-string payloads). Avoids invoking the
                # handler a second time on the fall-through path.
                captured: dict[str, Any] = {}

                async def _fetch() -> str | None:
                    res = await handler(request)
                    captured["result"] = res
                    # Don't memoise error responses: schema-validation or
                    # runtime errors are call-shape-specific, not codebase
                    # facts. Caching them would make a future sibling see
                    # an "ALREADY FETCHED" sentinel wrapping a stale error.
                    if isinstance(res, ToolMessage) and getattr(res, "status", None) == "error":
                        return None
                    # codebase_query and the mcp_codebase-index_* tools return
                    # MULTIMODAL content — a list of {type:text,...} blocks, not
                    # a bare str. The previous `isinstance(content, str)` guards
                    # therefore returned None for every facade lookup, so
                    # get_or_fetch never populated an entry: each call re-ran the
                    # handler AND fell through to the early return below, skipping
                    # the per-branch cache write too. Net effect — the dedupe was
                    # inert for the exact tools it was built for (trace 019ed413:
                    # one slice issued `codebase_query find_symbol UIApi` 4× with
                    # zero cache hits). Coerce via _stringify_content (the same
                    # helper the eviction path uses) so the result is memoisable;
                    # treat an empty string as a non-result so we don't cache "".
                    if isinstance(res, ToolMessage):
                        return _stringify_content(res.content) or None
                    if hasattr(res, "content"):
                        return _stringify_content(res.content) or None
                    return None

                cached_content, hit_count = await symbol_cache.get_or_fetch(
                    ctx.work_id, tool_name, args, _fetch,
                )
                if cached_content is not None:
                    content_sha = _content_sha(cached_content)
                    if key not in cache:
                        cache[key] = {
                            "summary": f"{_count_lines(cached_content)} lines",
                            "turn": turn,
                            "content_sha": content_sha,
                        }
                    if hit_count > 0:
                        payload = (
                            f"[ALREADY FETCHED via sibling branch — "
                            f"turn {turn}, hit #{hit_count}, content_sha={content_sha}]\n"
                            f"{cached_content}"
                        )
                        return ToolMessage(
                            content=payload,
                            tool_call_id=request.tool_call["id"],
                            name=tool_name,
                        )
                    # First fetch for this work_id: the result is now memoised for
                    # future hits, but return the original handler result verbatim
                    # so the immediate consumer keeps full (possibly multimodal)
                    # fidelity rather than the stringified copy.
                    if "result" in captured:
                        return captured["result"]
                    return ToolMessage(
                        content=cached_content,
                        tool_call_id=request.tool_call["id"],
                        name=tool_name,
                    )
                # Uncacheable result (error or non-string payload) — return
                # the captured handler result verbatim. Skip the per-branch
                # cache write below too: caching an error keyed by args would
                # mask a corrected retry from the same branch.
                if "result" in captured:
                    return captured["result"]

            if key in cache:
                entry = cache[key]
                sha = entry.get("content_sha", "?")
                return ToolMessage(
                    content=(
                        f"[cached: {tool_name} — issued at turn {entry['turn']} — "
                        f"{entry['summary']} — content_sha={sha}. Re-issue avoided; "
                        f"the response is above in this conversation. Change "
                        f"arguments or stop retrying.]"
                    ),
                    tool_call_id=request.tool_call["id"],
                    name=tool_name,
                )

            result = await handler(request)

            content = ""
            if isinstance(result, ToolMessage):
                content = _stringify_content(result.content)
            elif hasattr(result, "content"):
                content = _stringify_content(result.content)

            if content:
                summary = f"{_count_lines(content)} lines"
                cache[key] = {
                    "summary": summary,
                    "turn": turn,
                    "content_sha": _content_sha(content),
                }

            return result

        return await handler(request)

    def wrap_tool_call(self, request, handler):
        # SPINE uses async invoke; sync pass-through only.
        return handler(request)


# ---------------------------------------------------------------------------
# Tool-output eviction helpers (shared by ToolOutputTrimmer and TokenBudgetCompactor)
# ---------------------------------------------------------------------------


# Tools whose ToolMessage outputs must NEVER be replaced by a metadata
# placeholder. These carry information the agent needs verbatim to keep
# working on the current slice (write/edit acknowledgements, structured
# artifact bodies, verification reports). Trimming them caused the
# regression that retired the old ToolOutputTrimmer.
DEFAULT_PRESERVED_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "read_edit_lint",
    "write_specification",
    "write_plan",
    "write_tasks",
    "write_verification_report",
})


def extract_metadata(content: str, tool_name: str, tool_args: dict[str, Any]) -> str:
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
        last_line = content.rstrip().rsplit("\n", 1)[-1] if content.strip() else ""
        exit_match = re.search(r"(?:exit code|Exit code|exit_status)[:\s]*(\d+)", content)
        if exit_match:
            return f"[exec: {cmd} — exit code {exit_match.group(1)}]"
        if last_line and len(last_line) < 120:
            return f"[exec: {cmd} — {last_line}]"
        return f"[exec: {cmd}]"

    if tool_name == "grep":
        pattern = tool_args.get("pattern", "?")
        path = tool_args.get("path", ".")
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

    if tool_name == "codebase_query" or tool_name.startswith("mcp_codebase-index"):
        action = tool_args.get("action") or tool_name.rsplit("_", 1)[-1]
        target = tool_args.get("name") or tool_args.get("pattern") or "?"
        locs = _summarize_code_locations(content)
        return f"[codebase_query {action} '{target}' → {locs}]"

    hint = content[:80].split("\n")[0]
    if len(hint) == 80 and len(content) > 80:
        hint += "..."
    return f"[evicted({tool_name}): {hint}]"


# ---------------------------------------------------------------------------
# Search-loop detection (SearchLoopGuard)
# ---------------------------------------------------------------------------

# Tools whose results represent a *search*. A run of empty results from these
# means the agent is hunting for something that does not exist and keeps
# rewording the query — the spin observed in trace 019ed3b8 where ~7
# near-duplicate reranker/recall searches each returned ``[]``.
_SEARCH_TOOLS: frozenset[str] = frozenset({
    "codebase_query",
    "search_codebase",
    "grep",
    "glob",
})


def _is_search_tool(name: str) -> bool:
    """Return True if *name* is a code/text search tool (incl. MCP index)."""
    return name in _SEARCH_TOOLS or name.startswith("mcp_codebase-index")


def _is_empty_search_result(content: str) -> bool:
    """Return True if a search/lookup result found nothing.

    Handles both fresh results (``[]`` / ``{}`` / empty) and the eviction
    placeholders produced by :func:`extract_metadata` (``→ no results``,
    ``0 files`` for grep, ``0 hit`` for codebase_query).
    """
    s = content.strip()
    if not s:
        return True
    if s in ("[]", "{}", "null", "no results"):
        return True
    return "→ no results" in s or "0 files" in s or "0 hit" in s


def trim_tool_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
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

    return args


def trim_ai_args(messages: list, evicted_ids: set[str]) -> list:
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
            new_args = trim_tool_args(name, args)
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


def build_tool_call_map(messages: list) -> dict[str, tuple[str, dict]]:
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


def _estimate_message_tokens(messages: list) -> int:
    """Estimate prompt-token cost of *messages* via the shared tokenizer."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            # Multimodal content blocks: count any text parts we can find.
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content") or ""
                    if isinstance(text, str):
                        total += count_tokens(text)
                elif isinstance(block, str):
                    total += count_tokens(block)
        # tool_calls args also live in the wire prompt; account for them.
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            try:
                total += count_tokens(json.dumps(tool_calls, default=str))
            except Exception:
                pass
    return total


# ---------------------------------------------------------------------------
# ToolOutputTrimmer (count-based eviction — used by researcher subagent)
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

    # Back-compat shims so existing tests that reach into the instance
    # methods keep working after the helpers were lifted to module scope.
    def _extract_metadata(
        self, content: str, tool_name: str, tool_args: dict[str, Any]
    ) -> str:
        return extract_metadata(content, tool_name, tool_args)

    def _trim_ai_args(self, messages: list, evicted_ids: set[str]) -> list:
        return trim_ai_args(messages, evicted_ids)

    @staticmethod
    def _trim_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return trim_tool_args(name, args)

    @staticmethod
    def _build_tool_call_map(messages: list) -> dict[str, tuple[str, dict]]:
        return build_tool_call_map(messages)

    async def awrap_model_call(self, request, handler):
        """Trim old tool results and AI args before each model call."""
        messages = request.messages
        call_map = build_tool_call_map(messages)

        tool_result_indices = [
            i for i, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        if len(tool_result_indices) <= self.max_full_tool_results:
            return await handler(request)

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

            content = _stringify_content(msg.content)
            metadata = extract_metadata(content, name, args)

            evicted_ids.add(tc_id or "")
            try:
                trimmed_messages[idx] = ToolMessage(
                    content=metadata,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            except Exception:
                pass

        if evicted_ids:
            trimmed_messages = trim_ai_args(trimmed_messages, evicted_ids)

        return await handler(request.override(messages=trimmed_messages))

    # Pass-throughs so DA's factory check keeps the middleware bound.
    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


# ---------------------------------------------------------------------------
# TokenBudgetCompactor (token-threshold eviction — used by phase agents)
# ---------------------------------------------------------------------------


class TokenBudgetCompactor(AgentMiddleware):
    """Token-threshold compaction with hardened preservation rules.

    The retired ``ToolOutputTrimmer`` was removed in 2026-05 because it
    aggressively evicted tool results during implement/verify phases and
    cost the agent its working state. This middleware keeps the same
    metadata-placeholder approach but only fires when prompt tokens exceed
    ``threshold_tokens`` AND applies three preservation rules so the
    regression cannot recur:

      1. Keep the most recent ``keep_recent`` ToolMessages verbatim.
      2. Keep any ToolMessage whose source tool is in ``preserved_tools``
         (defaults to write/edit and the SPINE artifact-writer family).
      3. Anything else older than the preservation window is replaced by
         the structured metadata placeholder produced by
         :func:`extract_metadata`, and the corresponding ``AIMessage``
         ``tool_calls`` args are compacted via :func:`trim_ai_args`.
    """

    def __init__(
        self,
        threshold_tokens: int,
        keep_recent: int = 6,
        preserved_tools: frozenset[str] = DEFAULT_PRESERVED_TOOLS,
        *,
        window: int = 0,
        overhead: int = 4000,
    ) -> None:
        self.threshold_tokens = threshold_tokens
        self.keep_recent = keep_recent
        self.preserved_tools = preserved_tools
        # Hard pre-send guard: when the provider declares a finite context
        # window, no single prompt may exceed it. window<=0 (cloud/legacy
        # providers) disables the guard entirely — behaviour is unchanged there.
        self.window = window
        self.overhead = overhead

    async def awrap_model_call(self, request, handler):
        messages = request.messages

        total_tokens = _estimate_message_tokens(messages)

        normal_active = (
            self.threshold_tokens > 0 and total_tokens >= self.threshold_tokens
        )
        call_map = build_tool_call_map(messages)
        tool_result_indices = [
            i for i, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        evicted_ids: set[str] = set()
        trimmed_messages = list(messages)

        # ── Tier 1: normal threshold eviction ────────────────────────────
        # Evict everything older than the last `keep_recent`, EXCEPT
        # preserved-tool outputs which are always kept verbatim.
        if normal_active and len(tool_result_indices) > self.keep_recent:
            for idx in tool_result_indices[: -self.keep_recent]:
                msg = trimmed_messages[idx]
                assert isinstance(msg, ToolMessage)
                tc_id = msg.tool_call_id or ""
                name = msg.name or ""
                args: dict = {}
                if tc_id and tc_id in call_map:
                    name, args = call_map[tc_id]

                if name in self.preserved_tools:
                    continue

                content = _stringify_content(msg.content)
                metadata = extract_metadata(content, name, args)
                evicted_ids.add(tc_id)
                try:
                    trimmed_messages[idx] = ToolMessage(
                        content=metadata,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
                except Exception:
                    pass

        # ── Tier 2: hard window guard (finite-window providers only) ──────
        # If, after tier 1, the prompt still exceeds the window ceiling, the
        # preserved + keep_recent tail alone is too big to send. Escalate:
        # evict the OLDEST tool results regardless of preserved/recent status,
        # oldest-first, until we're under the ceiling — but never the single
        # most-recent ToolMessage (the agent's live working result) and never
        # an as-yet-unacknowledged write/edit (its body is the only record of
        # what was written). Degrading a re-read file body to its `[read: …]`
        # metadata is recoverable; sending an over-window prompt is not.
        hard = window_hard_ceiling(self.window, self.overhead)
        if hard:
            cur = _estimate_message_tokens(trimmed_messages)
            if cur > hard and len(tool_result_indices) > 1:
                for idx in tool_result_indices[:-1]:  # keep the most-recent verbatim
                    if cur <= hard:
                        break
                    msg = trimmed_messages[idx]
                    if not isinstance(msg, ToolMessage):
                        continue
                    tc_id = msg.tool_call_id or ""
                    if tc_id in evicted_ids:
                        continue
                    name = msg.name or ""
                    args = {}
                    if tc_id and tc_id in call_map:
                        name, args = call_map[tc_id]
                    if name in ("write_file", "edit_file"):
                        continue
                    before = _estimate_message_tokens([msg])
                    metadata = extract_metadata(_stringify_content(msg.content), name, args)
                    try:
                        trimmed_messages[idx] = ToolMessage(
                            content=metadata,
                            tool_call_id=msg.tool_call_id,
                            name=msg.name,
                        )
                    except Exception:
                        continue
                    evicted_ids.add(tc_id)
                    cur -= max(0, before - count_tokens(metadata))
                if cur > hard:
                    logger.warning(
                        "TokenBudgetCompactor HARD GUARD: prompt still %d > ceiling "
                        "%d after escalation (window=%d) — preserved tail exceeds window",
                        cur, hard, self.window,
                    )

        if not evicted_ids:
            return await handler(request)

        trimmed_messages = trim_ai_args(trimmed_messages, evicted_ids)
        new_tokens = _estimate_message_tokens(trimmed_messages)
        logger.info(
            "TokenBudgetCompactor: %d→%d tokens, evicted %d msgs",
            total_tokens, new_tokens, len(evicted_ids),
        )
        return await handler(request.override(messages=trimmed_messages))

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


# ---------------------------------------------------------------------------
# DynamicCompletionCapMiddleware (per-turn completion clamp for finite windows)
# ---------------------------------------------------------------------------


class DynamicCompletionCapMiddleware(AgentMiddleware):
    """Lower ``max_tokens`` each turn so prompt + completion stays in-window.

    The slice-implementer reserves a *static* completion cap when the agent is
    built, but its prompt grows turn by turn as it reads files. Once
    ``prompt + cap`` exceeds the model's context window the provider rejects
    the request outright — trace 019ece87 spun the fallback decomposer 75 times
    because every slice touching ``spine/ui_api/api.py`` (a 1200-line file) sent
    a ~27K prompt + 12K reserved cap against a 32K window and 400'd with
    "Context size has been exceeded".

    This middleware measures the prompt before each model call and clamps the
    model's ``max_tokens`` down to whatever room is left under the window. It
    only ever *lowers* the bound cap (eviction, not this, is what reclaims
    prompt room) and is a no-op for providers without a declared
    ``context_window``. It is added INNER of :class:`TokenBudgetCompactor` so it
    measures the already-trimmed message list.
    """

    def __init__(
        self,
        *,
        window: int,
        overhead: int = 2000,
        floor: int = 512,
    ) -> None:
        self.window = window
        self.overhead = overhead
        self.floor = floor

    @staticmethod
    def _bound_cap(model: Any) -> int | None:
        """Current completion ceiling bound on the model, if any."""
        for attr in ("max_tokens", "max_completion_tokens"):
            val = getattr(model, attr, None)
            if val:
                return int(val)
        return None

    @staticmethod
    def _apply_cap(model: Any, cap: int) -> Any:
        """Return a copy of ``model`` with ``cap`` as its completion ceiling.

        Mirrors :func:`spine.agents.helpers.cap_completion_tokens` without the
        import (ChatOpenAI stores the value as ``max_tokens``; updating the
        alias alone leaves the wire field untouched).
        """
        if getattr(model, "max_tokens", None) is not None:
            return model.model_copy(update={"max_tokens": cap})
        return model.model_copy(update={"max_completion_tokens": cap})

    async def awrap_model_call(self, request, handler):
        if self.window <= 0:
            return await handler(request)

        from spine.agents.synthesis_budget import window_aware_completion_cap

        messages = list(request.messages)
        sys_msg = getattr(request, "system_message", None)
        if sys_msg is not None:
            messages = [sys_msg, *messages]
        prompt_tokens = _estimate_message_tokens(messages)

        current = self._bound_cap(request.model)
        # When the model is uncapped, the window itself is the ceiling.
        base_cap = current if current else self.window
        cap = window_aware_completion_cap(
            window=self.window,
            prompt_tokens=prompt_tokens,
            base_cap=base_cap,
            overhead=self.overhead,
            floor=self.floor,
        )
        if current is None or cap < current:
            logger.info(
                "DynamicCompletionCap: prompt=%d window=%d cap %s→%d",
                prompt_tokens, self.window,
                current if current is not None else "uncapped", cap,
            )
            request = request.override(model=self._apply_cap(request.model, cap))
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


# ---------------------------------------------------------------------------
# ResearcherConvergenceMiddleware (push researcher to finalise findings)
# ---------------------------------------------------------------------------


class ResearcherConvergenceMiddleware(AgentMiddleware):
    """Steer the researcher subagent toward emitting a final ResearchFindings
    message before the LangGraph ``recursion_limit`` is hit.

    Researcher convergence == an ``AIMessage`` with no ``tool_calls``; that
    is what :func:`spine.agents.exploration_agents._finalize_research_findings`
    consumes. The salvage pathway added in commit ``80c9a66`` kicks in only
    *after* a ``GraphRecursionError``, so up to ``recursion_limit`` (50)
    turns can be wasted on breadth-first wandering first. This middleware
    counts tool calls emitted so far and intervenes in two stages:

      * ``>= soft_threshold``: append a nudging ``SystemMessage`` asking the
        model to begin synthesising findings.
      * ``>= hard_threshold``: append a stronger forcing reminder AND drop
        the tool bindings (``request.override(tools=[])``) so the model has
        no choice but to emit a final non-tool-calling message.
    """

    def __init__(
        self,
        soft_threshold: int = 25,
        hard_threshold: int = 40,
        recursion_limit: int = 50,
    ) -> None:
        if hard_threshold < soft_threshold:
            raise ValueError("hard_threshold must be >= soft_threshold")
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.recursion_limit = recursion_limit

    @staticmethod
    def _count_tool_calls(messages: list) -> int:
        n = 0
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                n += len(msg.tool_calls)
        return n

    async def awrap_model_call(self, request, handler):
        messages = request.messages
        n = self._count_tool_calls(messages)

        if n < self.soft_threshold:
            return await handler(request)

        if n >= self.hard_threshold:
            reminder = (
                f"[CONVERGENCE FORCING] You have made {n} of {self.recursion_limit} "
                f"allotted tool calls. STOP calling tools NOW. Your next message MUST be "
                f"your final ResearchFindings markdown report — patterns, file_map, "
                f"dependencies, summary — with NO tool_calls. If you call another tool, "
                f"your partial findings will be auto-salvaged and may be lower quality."
            )
            logger.info(
                "ResearcherConvergence: forcing (n=%d ≥ hard=%d, tools dropped)",
                n, self.hard_threshold,
            )
            return await handler(
                request.override(
                    messages=[*messages, SystemMessage(content=reminder)],
                    tools=[],
                )
            )

        nudge = (
            f"[CONVERGENCE NUDGE] {n} of {self.recursion_limit} tool calls used. "
            f"Begin synthesising your ResearchFindings now. Only call further tools "
            f"if absolutely required to fill a critical gap; otherwise emit your final "
            f"markdown report with no tool_calls."
        )
        logger.info(
            "ResearcherConvergence: nudging (n=%d ≥ soft=%d)",
            n, self.soft_threshold,
        )
        return await handler(
            request.override(messages=[*messages, SystemMessage(content=nudge)])
        )

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


# ---------------------------------------------------------------------------
# SearchLoopGuard (zero-result search spin breaker)
# ---------------------------------------------------------------------------


class SearchLoopGuard(AgentMiddleware):
    """Break consecutive zero-result search loops.

    A subagent that keeps rewording a search for a symbol/section that does
    not exist will burn turns — and, because the whole history is re-sent each
    turn, tokens — chasing it. In trace 019ed3b8 a single ``implement``
    subagent issued ~7 near-duplicate reranker/recall searches that each
    returned ``[]`` before accepting the section didn't exist and creating it.

    After ``threshold`` consecutive empty search results this middleware
    appends a ``SystemMessage`` telling the model to stop re-searching and act
    on what it already knows (create the thing, or proceed) — without dropping
    tools, so legitimate non-search work still runs. The streak resets the
    moment any search returns a hit.
    """

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold

    @staticmethod
    def _trailing_empty_streak(messages: list) -> int:
        """Count the trailing run of consecutive empty search results."""
        call_map = build_tool_call_map(messages)
        streak = 0
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            name = msg.name or ""
            if not name and msg.tool_call_id in call_map:
                name = call_map[msg.tool_call_id][0]
            if not _is_search_tool(name):
                continue
            content = _stringify_content(msg.content)
            streak = streak + 1 if _is_empty_search_result(content) else 0
        return streak

    async def awrap_model_call(self, request, handler):
        messages = request.messages
        streak = self._trailing_empty_streak(messages)
        if streak < self.threshold:
            return await handler(request)
        reminder = (
            f"[SEARCH LOOP GUARD] Your last {streak} searches returned no "
            f"results. Stop re-searching with reworded queries — what you are "
            f"looking for most likely does not exist yet. Either create it, or "
            f"proceed using what you already have. Do NOT issue another "
            f"search/codebase_query for the same thing."
        )
        logger.info("SearchLoopGuard: intervening (empty streak=%d)", streak)
        return await handler(
            request.override(messages=[*messages, SystemMessage(content=reminder)])
        )

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


class TurnBudgetGuard(AgentMiddleware):
    """Bound a single agent's model-turn count with an escalating nudge.

    Per-call token cost is clamped elsewhere (completion caps + the
    compaction threshold that bounds the prompt), but nothing bounds the
    NUMBER of turns. Because the full history is re-sent every turn, total
    input scales linearly with turn count: a slice-implementer that keeps
    re-querying and re-editing can grind dozens of turns at the compaction
    ceiling and burn ~1M input tokens on a small edit (trace 019ed413: 69
    model turns ≈ 954K input for a config-UI change).

    After ``threshold`` model turns this middleware appends a SystemMessage
    directing the model to converge — finish the edit, report status, stop
    exploring. Tools stay bound, so a genuinely long slice can still finish;
    the directive escalates each turn past the threshold so a model that
    ignores the first nudge gets an increasingly firm one. The recursion
    limit remains the ultimate backstop.
    """

    def __init__(self, threshold: int = 30) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold

    @staticmethod
    def _count_model_turns(messages: list) -> int:
        """Count assistant turns already taken in this invocation."""
        return sum(1 for m in messages if isinstance(m, AIMessage))

    async def awrap_model_call(self, request, handler):
        turns = self._count_model_turns(request.messages)
        if turns < self.threshold:
            return await handler(request)
        over = turns - self.threshold + 1
        reminder = (
            f"[TURN BUDGET GUARD] You have taken {turns} turns on this task "
            f"(soft budget {self.threshold}). Converge NOW: make the remaining "
            f"edits you are confident about, then return your final result with "
            f"a status. Do NOT issue more exploratory codebase_query/search "
            f"calls — act on what you already have. If something genuinely "
            f"blocks completion, report it as blocked rather than continuing to "
            f"loop."
        )
        if over > 1:
            reminder += (
                f" This is reminder #{over}; further looping is wasting tokens "
                f"with no progress — finish or report blocked on this turn."
            )
        logger.info("TurnBudgetGuard: intervening (turns=%d, over=%d)", turns, over)
        return await handler(
            request.override(messages=[*request.messages, SystemMessage(content=reminder)])
        )

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)
