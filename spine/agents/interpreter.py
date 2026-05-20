"""SPINE interpreter factory — creates CodeInterpreterMiddleware per phase.

The Recursive Language Model (RLM) pattern uses an interpreter as a
programmable workspace inside the agent loop. Instead of routing every
intermediate result through the model context, the agent writes code to:

1. **Inspect large inputs** — search, chunk, and filter codebases that exceed
   the model's context window (the primary finding of arXiv:2512.24601).
2. **Orchestrate subagents** — call ``tools.task(...)`` from code via
   programmatic tool calling (PTC), enabling loops, parallel batches,
   try/catch, and conditional logic without token overhead.
3. **Transform structured data** — sort, group, validate, score, or aggregate
   results deterministically before returning a compact synthesis to the
   model.

Language-agnostic design
------------------------
SPINE targets TypeScript, PHP, and Python projects (among others). The
interpreter runs **QuickJS** (JavaScript), which is the native language of
the DA ``CodeInterpreterMiddleware``. This is by design:

- TypeScript projects: QuickJS is native. The model writes JS to inspect/parse
  TS source directly.
- PHP projects: Neither JS nor Python is "native" to PHP, but JS is closer in
  syntax (C-derived, curly-brace, similar string/array semantics). The model
  can write JS to ``find()``/``chunk()`` through PHP source effectively.
- Python projects: QuickJS is the odd one out, but for orchestration (not
  execution), this matters less — the interpreter calls tools via PTC, and
  the tools themselves (shell, filesystem) handle language-specific work.

The interpreter is the **orchestration brain**, not the **execution hands**.
Filesystem writes, shell commands, and test runs still go through the backend
(``LocalShellBackend``) and are language-agnostic.

QuickJS environment
-------------------
The eval tool runs inside **QuickJS**, a server-side JS sandbox. Key globals:

- ``globalThis`` — the global object (NOT ``window`` — that's browser-only)
- ``globalThis.tools`` — PTC-accessible DA tools (based on allowlist).
  Tool names are **camelCase** (``tools.readFile``, ``tools.writeFile``,
  ``tools.editFile``, ``tools.task``), while argument keys retain the
  original **snake_case** from the tool schema (``{file_path: '...'}``,
  not ``{filePath: '...'}``). Return values are native JS types —
  ``readFile`` returns a string, not an object with ``.content``.
- ``globalThis.console`` — captured console output
- ``globalThis.context`` — seeded by SPINE phase prompts with work_id, phase,
  artifact_dir

Using ``window.*`` in eval code causes ``ReferenceError: window is not defined``.
All prompts and skills must use ``globalThis.*`` instead.

PTC allowlists per phase
------------------------
Each SPINE phase gets a different set of tools exposed in the interpreter
via the ``ptc`` (programmatic tool calling) allowlist:

- SPECIFY: ``task``, ``read_file``, ``grep``, ``glob``, ``ls``, ``write_file``, ``edit_file``
- TASKS:   ``task``, ``read_file``, ``grep``, ``glob``, ``ls``, ``write_file``, ``edit_file``
- IMPLEMENT: ``task``, ``read_file``, ``grep``, ``glob``, ``ls``, ``write_file``, ``edit_file``
- VERIFY:  ``task``, ``read_file``, ``grep``, ``glob``, ``ls``, ``write_file``, ``edit_file``
- CRITIC:  (none) — the critic reviews output, it doesn't orchestrate.

When using PTC to call the ``task`` tool, use the named subagent for the
current phase (e.g. ``researcher`` for SPECIFY, ``slice-implementer`` for
IMPLEMENT, ``slice-verifier`` for VERIFY) instead of ``general-purpose``.
The named subagents have tailored system prompts, tool restrictions, and
structured response formats that produce better results.

Call ``build_interpreter_middleware(phase_name)`` to get the right config.
"""

from __future__ import annotations

import logging
from typing import Any

from spine.models.enums import PhaseName

logger = logging.getLogger(__name__)

# ── PTC allowlists per phase ────────────────────────────────────────────
# Maps SPINE phase names to the list of DA tools that the interpreter
# is allowed to call programmatically. Each phase exposes "task" for
# read_file removed from PTC allowlist (2026-05). Under virtual_mode=True,
# tools.readFile() returns null for project .py source files — the agent
# must fall back to the native read_file tool, wasting a turn. Reserve eval
# for orchestration (ls, glob, grep, subagent dispatch) and use native
# read_file for source code content with batch reads (≥3 files per turn).

_PTC_ALLOWLISTS: dict[str, list[str | Any]] = {
    PhaseName.SPECIFY.value: ["task", "grep", "glob", "ls", "write_file", "edit_file"],
    PhaseName.TASKS.value: ["task", "grep", "glob", "ls", "write_file", "edit_file"],
    PhaseName.IMPLEMENT.value: ["task", "grep", "glob", "ls", "write_file", "edit_file"],
    PhaseName.VERIFY.value: ["task", "grep", "glob", "ls", "write_file", "edit_file"],
    # CRITIC and PLAN don't need PTC — they review/plan, not orchestrate.
}


# ── Interpreter defaults ───────────────────────────────────────────────

_DEFAULT_MEMORY_LIMIT = 64 * 1024 * 1024  # 64 MB QuickJS heap
_DEFAULT_TIMEOUT = 10.0  # seconds per eval
_DEFAULT_MAX_PTC_CALLS = 256
_DEFAULT_MAX_RESULT_CHARS = 4000  # compact synthesis for RLM — keeps context lean
_DEFAULT_SNAPSHOT_BETWEEN_TURNS = True


def build_interpreter_middleware(
    phase_name: str,
    *,
    memory_limit: int = _DEFAULT_MEMORY_LIMIT,
    timeout: float = _DEFAULT_TIMEOUT,
    max_ptc_calls: int | None = _DEFAULT_MAX_PTC_CALLS,
    max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
    snapshot_between_turns: bool = _DEFAULT_SNAPSHOT_BETWEEN_TURNS,
) -> Any:
    """Build a CodeInterpreterMiddleware configured for a SPINE phase.

    Creates a QuickJS interpreter with PTC enabled for phases that need
    subagent orchestration (SPECIFY, TASKS, IMPLEMENT, VERIFY). Phases
    without PTC (PLAN, CRITIC) get an interpreter for data transformation
    only — no tool calling from code.

    The middleware is a DA ``AgentMiddleware`` instance, ready to pass to
    ``create_deep_agent(middleware=[...])``.

    Args:
        phase_name: SPINE phase name (e.g. ``"specify"``, ``"implement"``).
        memory_limit: QuickJS heap limit in bytes (default 64 MB).
        timeout: Per-eval timeout in seconds (default 10).
        max_ptc_calls: Max ``tools.*`` calls per eval (default 256).
            Set ``None`` for unlimited (trusted environments only).
        max_result_chars: Max chars returned from eval + stdout (default 8000).
        snapshot_between_turns: Persist interpreter state across turns
            (default True). Pairs with LangGraph checkpointing.

    Returns:
        A ``CodeInterpreterMiddleware`` instance.

    Raises:
        ImportError: If ``langchain-quickjs`` is not installed.
    """
    try:
        from langchain_quickjs import CodeInterpreterMiddleware
    except ImportError:
        raise ImportError(
            "langchain-quickjs is required for interpreter support. "
            "Install with: uv add 'deepagents[quickjs]' or pip install langchain-quickjs"
        )

    ptc_allowlist = _PTC_ALLOWLISTS.get(phase_name)
    if ptc_allowlist is None:
        logger.debug(
            "Phase %r has no PTC allowlist — interpreter will be data-only",
            phase_name,
        )

    class SpineInterpreterMiddleware(CodeInterpreterMiddleware):
        """Strips redundant TypeScript schemas injected by langchain-quickjs."""

        async def before_model(
            self, messages: list[Any], *, tools: list[Any], **kwargs: Any
        ) -> dict[str, Any]:
            result = await super().before_model(messages, tools=tools, **kwargs)
            out_msgs = result.get("messages", messages)

            import re

            for m in out_msgs:
                if hasattr(m, "content") and isinstance(m.content, str):
                    if "API Reference" in m.content and "tools namespace" in m.content:
                        # Scrub the TypeScript definitions section. It starts with the header
                        # and ends exactly at the last typescript block closure in that section.
                        m.content = re.sub(
                            r"### API Reference — `tools` namespace.*?(```typescript.*?```\s*)+",
                            "### PTC Note\nTools are pre-bound on `globalThis.tools` using camelCase (e.g. `tools.readFile`). Return values are native JS. Do not json parse.",
                            m.content,
                            flags=re.DOTALL,
                        )

            return result

    middleware = SpineInterpreterMiddleware(
        memory_limit=memory_limit,
        timeout=timeout,
        max_ptc_calls=max_ptc_calls,
        tool_name="eval",
        max_result_chars=max_result_chars,
        capture_console=True,
        ptc=ptc_allowlist,
        snapshot_between_turns=snapshot_between_turns,
    )

    logger.debug(
        "Built interpreter middleware for phase %r (ptc=%s)",
        phase_name,
        ptc_allowlist,
    )

    return middleware


def interpreter_enabled() -> bool:
    """Check if interpreter support is available and enabled.

    Returns True if all of:
    1. ``langchain-quickjs`` is importable
    2. The interpreter feature flag is set — either via the
       ``SPINE_INTERPRETER`` environment variable or the
       ``interpreter_enabled`` key in ``.spine/config.yaml``

    Disabled by default so existing workflows are unaffected.
    """
    import os

    # Env var takes priority, then config file, then default off
    env_val = os.getenv("SPINE_INTERPRETER", "").strip().lower()
    if env_val:
        enabled = env_val in ("1", "true", "yes")
    else:
        try:
            from spine.config import SpineConfig

            enabled = SpineConfig.load().interpreter_enabled
        except Exception:
            enabled = False

    if not enabled:
        return False

    try:
        import langchain_quickjs  # noqa: F401

        return True
    except ImportError:
        logger.warning(
            "Interpreter is enabled but langchain-quickjs is not installed. "
            "Install with: uv add langchain-quickjs"
        )
        return False
