"""Forgiving tool-call parsing — recover tool calls emitted as plain text.

Frontier providers return tool calls as native ``AIMessage.tool_calls``. Small
local models (8B–30B), especially quantized or Hermes-templated ones, frequently
emit the call as *text* in the message body instead — e.g.::

    <tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>

or a fenced ```json block, or a bare leading ``{"name": ..., "arguments": ...}``.
When that happens the parsed ``AIMessage.tool_calls`` is empty, so the call never
reaches :class:`~spine.agents.tool_schema_validator.ToolSchemaValidator` (which
runs at the tool layer) nor DeepAgents' ``PatchToolCallsMiddleware`` (which
repairs *history*). The turn ends with no tool executed and the agent stalls.

This middleware catches the case at the **model layer**: it runs the model, and
*only* when the returned ``AIMessage`` has empty ``tool_calls`` does it scan the
content for a strict tool-call envelope. On a confident match it populates proper
``tool_calls`` and strips the envelope from the content; otherwise it returns the
response untouched.

Scope is deliberately narrow — **format extraction only**. Parameter-name and
type repair is left to ``ToolSchemaValidator``'s rebound loop (this middleware is
registered just before it, so a recovered call still gets validated). Parsing is
deterministic, so checkpoint replay is unaffected.

Opt-in via ``SPINE_TOOL_CALL_NORMALIZE`` (default off) — see
``factory._add_spine_middleware``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

# Strict envelopes, tried in priority order. Hermes ``<tool_call>`` is the
# strongest signal; fenced blocks and bare objects need an explicit args-like key
# to qualify, so ordinary prose / example JSON is not mistaken for a call.
_HERMES_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCED_RE = re.compile(
    r"```(?:json|yaml|yml|tool_call)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Tool names are identifiers — reject anything with whitespace so a prose line
# like ``name: John Smith`` can never be read as a tool call.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")

# Keys a model might use for the call name / arguments, in preference order.
_NAME_KEYS = ("name", "tool", "tool_name", "function")
_ARG_KEYS = ("arguments", "args", "parameters", "params", "input")


def _stringify_content(content: Any) -> str:
    """Flatten message content to text — handles LangChain multimodal blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _parse_payload(text: str) -> Any:
    """Parse a candidate payload as JSON, falling back to YAML. None on failure."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return None


def _coerce_args(raw: Any) -> dict[str, Any] | None:
    """Coerce an arguments value to a dict; None signals 'not a valid call'."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            v = json.loads(s)
        except Exception:
            return None
        if isinstance(v, dict):
            return v
    return None


def _obj_to_call(obj: Any, *, require_args_key: bool) -> dict[str, Any] | None:
    """Turn a parsed object into a ``{"name", "args"}`` call, or None.

    ``require_args_key`` is False only for the Hermes ``<tool_call>`` path — the
    envelope itself is strong enough evidence that a name without an explicit
    arguments key (a no-arg call) is real. Fenced/bare candidates must carry an
    args-like key to qualify, which keeps incidental JSON out.
    """
    if not isinstance(obj, dict):
        return None
    name: Any = None
    for key in _NAME_KEYS:
        if isinstance(obj.get(key), str) and obj[key].strip():
            name = obj[key].strip()
            break
    if not name or not _TOOL_NAME_RE.match(name):
        return None
    for key in _ARG_KEYS:
        if key in obj:
            args = _coerce_args(obj[key])
            if args is None:
                return None
            return {"name": name, "args": args}
    if require_args_key:
        return None
    return {"name": name, "args": {}}


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Return *text* with the given (start, end) spans removed."""
    out: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def extract_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Extract tool calls from a text body. Returns (calls, content_without_them).

    Tries Hermes envelopes, then fenced blocks, then a single bare object —
    stopping at the first layer that yields at least one valid call.
    """
    if not text:
        return [], text

    # 1. Hermes <tool_call>…</tool_call> envelopes (may be several).
    hermes = list(_HERMES_RE.finditer(text))
    if hermes:
        calls: list[dict[str, Any]] = []
        for m in hermes:
            call = _obj_to_call(_parse_payload(m.group(1)), require_args_key=False)
            if call:
                calls.append(call)
        if calls:
            return calls, _HERMES_RE.sub("", text).strip()

    # 2. Fenced code blocks that parse to a tool-call object.
    fenced = list(_FENCED_RE.finditer(text))
    if fenced:
        calls = []
        spans: list[tuple[int, int]] = []
        for m in fenced:
            call = _obj_to_call(_parse_payload(m.group(1)), require_args_key=True)
            if call:
                calls.append(call)
                spans.append(m.span())
        if calls:
            return calls, _remove_spans(text, spans).strip()

    # 3. The whole body is a single bare tool-call object.
    call = _obj_to_call(_parse_payload(text), require_args_key=True)
    if call:
        return [call], ""

    return [], text


class ToolCallNormalizer(AgentMiddleware):
    """Recover text-emitted tool calls into native ``AIMessage.tool_calls``.

    Runs the model, then — only when the response carries no native tool calls —
    looks for a strict tool-call envelope in the content and promotes it. A
    no-op for models that emit native tool calls (the common path).
    """

    def _normalize_response(self, response: Any) -> None:
        messages = getattr(response, "result", None)
        if not isinstance(messages, list):
            return
        # The model's output is the trailing AIMessage; only it can carry a
        # freshly text-emitted call.
        idx = next(
            (i for i in range(len(messages) - 1, -1, -1)
             if isinstance(messages[i], AIMessage)),
            None,
        )
        if idx is None:
            return
        msg = messages[idx]
        if msg.tool_calls:  # already has native calls — leave untouched
            return
        text = _stringify_content(msg.content)
        # Cheap pre-filter: bail unless something envelope-shaped is present.
        if "<tool_call>" not in text.lower() and "```" not in text and not text.lstrip().startswith("{"):
            return
        calls, stripped = extract_tool_calls(text)
        if not calls:
            return
        tool_calls = [
            {
                "name": c["name"],
                "args": c["args"],
                "id": f"spine_tc_{uuid.uuid4().hex[:22]}",
                "type": "tool_call",
            }
            for c in calls
        ]
        messages[idx] = msg.model_copy(
            update={"content": stripped, "tool_calls": tool_calls}
        )
        logger.info(
            "ToolCallNormalizer: recovered %d text-emitted tool call(s): %s",
            len(tool_calls),
            [c["name"] for c in tool_calls],
        )

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        try:
            self._normalize_response(response)
        except Exception:  # noqa: BLE001 — never let normalization break a turn
            logger.debug("ToolCallNormalizer: pass-through after error", exc_info=True)
        return response

    def wrap_model_call(self, request, handler):
        response = handler(request)
        try:
            self._normalize_response(response)
        except Exception:  # noqa: BLE001
            logger.debug("ToolCallNormalizer: pass-through after error", exc_info=True)
        return response
