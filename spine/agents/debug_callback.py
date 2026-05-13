"""SPINE LLM debug callback — logs chat model messages to the console.

Enabled by setting the ``SPINE_DEBUG_LLM`` environment variable (the
``spine ui --debug-llm`` flag does this automatically).

Logs:
- **on_chat_model_start**: the serialized messages being sent
- **on_chat_model_end**: the response content received
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def is_debug_llm_enabled() -> bool:
    """Return True if LLM debug logging is enabled."""
    return os.getenv("SPINE_DEBUG_LLM", "").strip().lower() in ("1", "true", "yes")


class LLMDebugCallback(BaseCallbackHandler):
    """Callback handler that logs chat model request/response to stderr.

    Each LLM call produces two log lines:
      ┌─ LLM → <model_name>  (N messages, <timestamp>)
      │   <messages summary>
      └─ LLM ← <model_name>  (<elapsed>s)
          <response summary>

    Attach via ``config={"callbacks": [LLMDebugCallback()]}`` when invoking
    an agent, or call :func:`install_global` to attach to all calls.
    """

    def __init__(self, out: Any | None = None) -> None:
        self._out = out or sys.stderr
        self._start_times: dict[int, float] = {}

    def _print(self, msg: str) -> None:
        print(msg, file=self._out, flush=True)

    # ── Chat model lifecycle ──

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        model_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        if isinstance(model_name, list):
            model_name = model_name[-1] if model_name else "unknown"
        self._start_times[id(run_id) if run_id else 0] = time.monotonic()

        msg_count = sum(len(m) for m in messages) if messages else 0
        self._print(f"┌─ LLM → {model_name}  ({msg_count} messages)")

        # Summarise each message (truncate long content)
        for batch in messages:
            for m in batch:
                role = getattr(m, "type", getattr(m, "role", "?"))
                content = getattr(m, "content", str(m))
                # Handle content that is a list (multimodal)
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                preview = content[:300].replace("\n", "↵")
                if len(content) > 300:
                    preview += "…"
                self._print(f"│   [{role}] {preview}")

    def on_chat_model_end(
        self,
        response: LLMResult,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._start_times.pop(id(run_id) if run_id else 0, None)
        elapsed = f"{time.monotonic() - start:.2f}s" if start else "??s"

        generations = response.generations
        model_name = ""
        if response.llm_output and isinstance(response.llm_output, dict):
            model_name = response.llm_output.get("model_name", "")

        self._print(f"└─ LLM ← {model_name}  ({elapsed})")

        for gen_list in generations:
            for gen in gen_list:
                text = gen.text
                if not text and hasattr(gen, "message"):
                    text = getattr(gen.message, "content", "")
                if isinstance(text, list):
                    text = json.dumps(text, ensure_ascii=False)
                preview = str(text)[:500].replace("\n", "↵")
                if len(str(text)) > 500:
                    preview += "…"
                self._print(f"    {preview}")

    def on_chat_model_error(
        self,
        error: BaseException,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._start_times.pop(id(run_id) if run_id else 0, None)
        self._print(f"└─ LLM ✗ ERROR: {error}")

    # ── Tool lifecycle (brief) ──

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "tool")
        preview = str(input_str)[:200].replace("\n", "↵")
        self._print(f"┌─ TOOL → {tool_name}: {preview}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        preview = str(output)[:200].replace("\n", "↵")
        self._print(f"└─ TOOL ← {preview}")


# ── Global install helpers ──

_installed: bool = False


def install_global() -> None:
    """Install the debug callback globally so it applies to all LLM calls.

    Uses ``langchain_core.globals.set_debug(True)`` which is the standard
    LangChain mechanism for logging all chat model I/O (prompts, completions,
    tool calls) to stderr.

    Also sets ``langchain_core.globals.set_verbose(True)`` for additional
    agent-level output.

    Safe to call multiple times — only installs once.
    """
    global _installed
    if _installed:
        return
    _installed = True
    try:
        from langchain_core.globals import set_debug, set_verbose

        set_debug(True)
        set_verbose(True)
        print(
            "[SPINE] LLM debug logging enabled — all chat model I/O will appear on stderr",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"[SPINE] Failed to install LLM debug logging: {e}", file=sys.stderr)
