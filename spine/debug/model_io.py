"""Model I/O debug logger for SPINE Deep Agents.

Wraps a LangChain BaseChatModel to log every invoke() call's input messages
and output response to ``.spine/debug/model_io/`` as timestamped JSON files.

Enable by setting the environment variable ``SPINE_DEBUG_MODEL_IO=1`` or by
passing ``debug_model_io=True`` in the provider config.

Logged files:
    .spine/debug/model_io/YYYYMMDD_HHMMSS_{call_id}_in.json   — input messages
    .spine/debug/model_io/YYYYMMDD_HHMMSS_{call_id}_out.json  — output response

Each file contains a ``_meta`` key with:
    - timestamp: ISO 8601 UTC
    - call_id: unique identifier for this invocation
    - phase: SPINE phase name (if available from thread-local context)
    - model: model identifier (if available)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

logger = logging.getLogger(__name__)

# Thread-local storage for current phase context
_context = threading.local()


def set_debug_phase(phase: str) -> None:
    """Set the current phase name in thread-local storage.

    Called by the state machine phase functions so that model I/O
    logs can be tagged with the originating phase.
    """
    _context.phase = phase


def get_debug_phase() -> str:
    """Get the current phase name from thread-local storage."""
    return getattr(_context, "phase", "")


def is_debug_enabled() -> bool:
    """Check if model I/O debugging is enabled.

    Enabled when SPINE_DEBUG_MODEL_IO=1 or the provider config
    has debug_model_io=True.
    """
    return os.environ.get("SPINE_DEBUG_MODEL_IO", "").strip() in ("1", "true", "yes")


class ModelIOLogger(BaseChatModel):
    """Debug wrapper that logs all model invocations to disk.

    Delegates actual LLM calls to the wrapped model.  Intercepts
    invoke() / ainvoke() to write input and output JSON files.

    This is NOT a full proxy — it only wraps the methods that SPINE
    uses (invoke, ainvoke).  Other BaseChatModel methods fall through
    to the underlying model via __getattr__.

    Usage::

        real_model = init_chat_model("openrouter:anthropic/claude-sonnet-4-5", ...)
        debug_model = ModelIOLogger.wrap(real_model)
        # Use debug_model everywhere instead of real_model
    """

    def __init__(self, wrapped: BaseChatModel, debug_dir: Optional[str] = None) -> None:
        super().__init__()
        self._wrapped = wrapped
        self._debug_dir = debug_dir or os.path.join(".spine", "debug", "model_io")
        self._call_counter = 0
        self._lock = threading.Lock()
        os.makedirs(self._debug_dir, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────

    @classmethod
    def wrap(cls, model: BaseChatModel, debug_dir: Optional[str] = None) -> "ModelIOLogger":
        """Wrap a BaseChatModel with debug logging.

        If SPINE_DEBUG_MODEL_IO is not enabled, returns the original
        model unchanged (zero overhead).
        """
        if not is_debug_enabled():
            return model  # type: ignore[return-value]
        if isinstance(model, cls):
            return model  # Already wrapped
        return cls(model, debug_dir=debug_dir)

    @classmethod
    def wrap_if_enabled(
        cls,
        model: BaseChatModel,
        debug_dir: Optional[str] = None,
    ) -> BaseChatModel:
        """Wrap a model only if debug logging is enabled.

        Always returns a BaseChatModel (typed correctly for callers).
        """
        if not is_debug_enabled():
            return model
        if isinstance(model, cls):
            return model
        return cls(model, debug_dir=debug_dir)

    # ── Core invoke interception ─────────────────────────────────────

    def invoke(
        self,
        input: Any,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the wrapped model and log I/O."""
        call_id = self._next_call_id()
        phase = get_debug_phase()
        timestamp = datetime.now(timezone.utc)

        # Log input
        self._write_io(
            call_id=call_id,
            timestamp=timestamp,
            suffix="in",
            data=self._serialize_messages(input),
            phase=phase,
        )

        # Call the real model
        result = self._wrapped.invoke(input, config=config, **kwargs)

        # Log output
        self._write_io(
            call_id=call_id,
            timestamp=timestamp,
            suffix="out",
            data=self._serialize_response(result),
            phase=phase,
        )

        return result

    async def ainvoke(
        self,
        input: Any,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Async invoke the wrapped model and log I/O."""
        call_id = self._next_call_id()
        phase = get_debug_phase()
        timestamp = datetime.now(timezone.utc)

        self._write_io(
            call_id=call_id,
            timestamp=timestamp,
            suffix="in",
            data=self._serialize_messages(input),
            phase=phase,
        )

        result = await self._wrapped.ainvoke(input, config=config, **kwargs)

        self._write_io(
            call_id=call_id,
            timestamp=timestamp,
            suffix="out",
            data=self._serialize_response(result),
            phase=phase,
        )

        return result

    # ── BaseChatModel abstract methods ───────────────────────────────

    @property
    def _llm_type(self) -> str:
        return f"debug-logger({getattr(self._wrapped, '_llm_type', 'unknown')})"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Delegate _generate to the wrapped model."""
        return self._wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    # ── Proxy passthrough ────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the wrapped model."""
        return getattr(self._wrapped, name)

    # ── Internal helpers ─────────────────────────────────────────────

    def _next_call_id(self) -> str:
        with self._lock:
            self._call_counter += 1
            return f"{self._call_counter:04d}"

    def _write_io(
        self,
        call_id: str,
        timestamp: datetime,
        suffix: str,
        data: Any,
        phase: str,
    ) -> None:
        """Write a model I/O record to disk."""
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts_str}_{call_id}_{suffix}.json"
        filepath = os.path.join(self._debug_dir, filename)

        record = {
            "_meta": {
                "timestamp": timestamp.isoformat(),
                "call_id": call_id,
                "direction": suffix,
                "phase": phase or "unknown",
                "model": getattr(self._wrapped, "_llm_type", "unknown"),
            },
            "data": data,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.warning("Failed to write model I/O debug log %s: %s", filepath, e)

    @staticmethod
    def _serialize_messages(input: Any) -> Any:
        """Serialize model input (messages) to a JSON-compatible structure."""
        if isinstance(input, str):
            return [{"role": "user", "content": input}]
        if isinstance(input, list):
            result = []
            for msg in input:
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    # BaseMessage subclass
                    result.append({
                        "role": getattr(msg, "type", "unknown"),
                        "name": getattr(msg, "name", None),
                        "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                    })
                elif isinstance(msg, dict):
                    result.append(msg)
                else:
                    result.append(str(msg))
            return result
        return str(input)

    @staticmethod
    def _serialize_response(response: Any) -> Any:
        """Serialize model output to a JSON-compatible structure."""
        if hasattr(response, "content"):
            # AIMessage or similar
            return {
                "role": getattr(response, "type", "ai"),
                "content": response.content if isinstance(response.content, str) else str(response.content),
                "tool_calls": getattr(response, "tool_calls", None),
                "usage_metadata": getattr(response, "usage_metadata", None),
                "id": getattr(response, "id", None),
            }
        if isinstance(response, dict):
            return response
        return str(response)


__all__ = ["ModelIOLogger", "is_debug_enabled", "set_debug_phase", "get_debug_phase"]
