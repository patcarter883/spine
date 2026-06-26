"""LLM error diagnosis on command failure — a one-line hint appended in place.

When an ``execute`` (shell) tool call fails, a small model is much better at
telling a weak agent *why* than the raw stderr is — "the package isn't
installed", "you're in the wrong directory", "that flag doesn't exist". This
middleware watches ``execute`` results, and on failure makes one cheap, capped,
reasoning-suppressed LLM call to produce a single corrective line, which it
appends to the ToolMessage as ``[diagnosis] …``.

It is **append-only**: the tool's status, exit code, and output are never
changed, so nothing downstream (gates, retries, the agent's own logic) is
affected — the agent just gets one extra line of advice.

Opt-in via ``error_diagnosis`` config / ``SPINE_ERROR_DIAGNOSIS`` (default off).
This is the one adaptation that injects a non-deterministic LLM result into a
checkpointed ToolMessage, so replay can differ — hence it stays gated. Its tokens
also bypass the per-work_id budget tracker in ``retry.py``; keep that in mind if
enabling it on a tight budget.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

# Matches a non-zero exit code in execute output (the tool reports failure as
# "[Command failed with exit code N]" with status still "success"). Mirrors the
# exit-code probe in context_editing.extract_metadata.
_EXIT_RE = re.compile(r"(?:exit code|Exit code|exit_status)[:\s]*(\d+)")
_MAX_OUTPUT_CHARS = 2000
_DIAGNOSIS_CAP_TOKENS = 256


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b if isinstance(b, str) else str(b.get("text", ""))
            for b in content
            if isinstance(b, (str, dict))
        )
    return str(content or "")


class ExecuteErrorDiagnoser(AgentMiddleware):
    """Append a one-line LLM diagnosis to failed ``execute`` tool results."""

    @staticmethod
    def _is_execute_failure(tool_name: str, result: Any) -> bool:
        if tool_name != "execute" or not isinstance(result, ToolMessage):
            return False
        if getattr(result, "status", None) == "error":
            return True
        text = _stringify(result.content)
        m = _EXIT_RE.search(text)
        if m and m.group(1) != "0":
            return True
        return "command failed" in text.lower()

    async def _diagnose(self, command: str, output: str) -> str:
        """Return a one-line fix suggestion, or '' if the call fails."""
        from spine.agents.helpers import (
            cap_completion_tokens,
            disable_streaming,
            resolve_chat_model,
            suppress_reasoning,
        )

        # config=None → resolve from .spine/config.yaml; route a cheap model via
        # providers.phases.error_diagnosis if desired.
        model = resolve_chat_model(None, phase="error_diagnosis")
        model = cap_completion_tokens(
            suppress_reasoning(disable_streaming(model)), _DIAGNOSIS_CAP_TOKENS
        )
        prompt = (
            "A shell command failed. In ONE short line, state the most likely "
            "cause and the concrete fix. No preamble, no markdown.\n\n"
            f"Command:\n{command}\n\nOutput:\n{output[:_MAX_OUTPUT_CHARS]}"
        )
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        line = _stringify(getattr(resp, "content", "")).strip()
        # Collapse to the first non-empty line — the model may still ramble.
        for candidate in line.splitlines():
            if candidate.strip():
                return candidate.strip()[:300]
        return ""

    @staticmethod
    def _append(result: ToolMessage, diagnosis: str) -> ToolMessage:
        new_content = f"{_stringify(result.content)}\n[diagnosis] {diagnosis}"
        return result.model_copy(update={"content": new_content})

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        try:
            tool_call = request.tool_call
            if self._is_execute_failure(tool_call.get("name", ""), result):
                command = str((tool_call.get("args") or {}).get("command", ""))
                diagnosis = await self._diagnose(command, _stringify(result.content))
                if diagnosis:
                    logger.info("ExecuteErrorDiagnoser: %s", diagnosis)
                    return self._append(result, diagnosis)
        except Exception:  # noqa: BLE001 — diagnosis is best-effort, never fatal
            logger.debug("ExecuteErrorDiagnoser: skipped after error", exc_info=True)
        return result
