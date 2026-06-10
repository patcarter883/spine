"""Middleware that forces a structured-output tool call to actually fire.

Some quantized / smaller local models (e.g. Gemma Q4 on llama.cpp) reliably
emit a final ``write_specification`` payload as a fenced ```json text block
instead of *calling* the tool.  The structured-output contract then fails
because ``specification.json`` never lands on disk (observed in trace
``019eaa90-2b02-7761-938b-69dd575f2cf6``: the synthesizer printed the spec
as text with ``finish_reason="stop"`` and no tool call).

:class:`ForceToolUntilCalledMiddleware` constrains decoding so the model
cannot answer in prose while the required tool is still outstanding:

- While ``final_tool`` has not yet been called *successfully*, set
  ``tool_choice`` so the provider forbids a bare text turn — the model is
  forced to call *some* tool.  ``"any"`` is the value LangChain itself uses
  to force structured-output tools (see ``langchain/agents/factory.py``),
  and is the most broadly supported across providers.
- Optionally, once a ``gate_tool`` has been called, pin ``tool_choice`` to
  ``final_tool`` specifically so the very next turn is the structured write.
- Once ``final_tool`` has reported success, release the constraint (leave
  ``tool_choice`` untouched) so the agent loop can emit a normal terminal
  message and finish.  Without this release the forced-call loop would never
  terminate — the model would be compelled to keep calling tools forever.

``tool_choice`` flows straight into ``model.bind_tools(tool_choice=...)``
(see ``langchain/agents/factory.py``), which every OpenAI-style provider
honors natively: OpenAI, vLLM's OpenAI server, OpenRouter, and llama.cpp's
OpenAI-compatible server.  Anthropic uses a different ``tool_choice``
grammar, so the middleware is a deliberate no-op for non-OpenAI-style
models (Claude's tool-calling reliability does not need this anyway).

One wire-format caveat: ``bind_tools`` maps ``"any"`` onto the *string*
``tool_choice="required"``, but a tool *name* onto the OpenAI *object* form
``{"type": "function", "function": {"name": ...}}``.  llama.cpp's server
only parses string values ("none"/"auto"/"required") and discards the
object form with::

    Wrong type supplied for parameter 'tool_choice'. Expected 'string',
    using default value

i.e. the pin silently degrades to "auto" — exactly the prose-stall the
middleware exists to prevent.  Local servers (detected via a custom
``base_url`` on ChatOpenAI) therefore get ``"any"`` instead of a named
pin; "required" still forbids a bare text turn, which is the property
that matters.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from spine.agents.helpers import _is_openai_style_model

logger = logging.getLogger(__name__)

# Sentinel meaning "leave request.tool_choice untouched" (distinct from the
# valid tool_choice value ``None``/"auto").
_RELEASE = object()

# LangChain's canonical "force the model to call some tool" value. ChatOpenAI /
# ChatOpenRouter bind_tools() map this onto the OpenAI ``tool_choice="required"``
# wire value; vLLM and llama.cpp honor it identically.
_FORCE_ANY = "any"


def _named_pin_supported(model: Any) -> bool:
    """True when pinning ``tool_choice`` to a named tool is safe on the wire.

    A named pin serializes as the OpenAI object form
    ``{"type": "function", "function": {"name": ...}}``.  llama.cpp's server
    only accepts string ``tool_choice`` values and silently falls back to the
    default ("auto") on the object form, defeating the pin entirely.  Local
    OpenAI-compatible servers are recognised by a custom ``base_url`` on
    ChatOpenAI (cloud OpenAI leaves it unset); vLLM does support the object
    form, but demoting it to "required" only loosens *which* tool is forced,
    never whether one is — a safe trade for not having to fingerprint the
    server behind the URL.
    """
    return not getattr(model, "openai_api_base", None)


class ForceToolUntilCalledMiddleware(AgentMiddleware):
    """Force tool use until a required structured-output tool has fired.

    Args:
        final_tool: Name of the tool that satisfies the phase contract
            (e.g. ``"write_specification"``). Forcing is released once this
            tool reports success.
        gate_tool: Optional name of a tool expected to run *before*
            ``final_tool`` (e.g. ``"read_work_context"``). Once it has been
            called, ``tool_choice`` is pinned to ``final_tool`` for maximal
            reliability. Leave ``None`` to only ever force "any" tool — the
            right choice when the agent legitimately uses several tools
            (e.g. ``recall``) between the gate and the final write.
        success_marker: Substring that marks a *successful* ``final_tool``
            result. The write tools return a validation/error string (which
            lacks this marker) when args are bad, so a failed write keeps the
            constraint on and lets the model self-correct in the same loop.
    """

    def __init__(
        self,
        final_tool: str,
        gate_tool: str | None = None,
        success_marker: str = "written to",
    ) -> None:
        self.final_tool = final_tool
        self.gate_tool = gate_tool
        self.success_marker = success_marker

    # ── decision helpers ────────────────────────────────────────────────

    def _final_tool_succeeded(self, messages: list[Any]) -> bool:
        """True once ``final_tool`` was called and returned a success result."""
        final_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    if tc.get("name") == self.final_tool and tc.get("id"):
                        final_ids.add(tc["id"])
        if not final_ids:
            return False
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.tool_call_id in final_ids:
                if getattr(msg, "status", None) == "error":
                    continue
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                # A validation rejection (e.g. "VALIDATION_ERROR: …") lacks the
                # success marker — treat it as not-yet-done so forcing persists.
                if self.success_marker and self.success_marker not in content:
                    continue
                return True
        return False

    def _gate_tool_called(self, messages: list[Any]) -> bool:
        if not self.gate_tool:
            return False
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    if tc.get("name") == self.gate_tool:
                        return True
        return False

    def _decide(self, request: Any) -> Any:
        """Return the tool_choice to force, or ``_RELEASE`` to leave as-is."""
        model = getattr(request, "model", None)
        if not _is_openai_style_model(model):
            return _RELEASE
        messages = list(getattr(request, "messages", None) or [])
        if self._final_tool_succeeded(messages):
            return _RELEASE
        if self._gate_tool_called(messages) and _named_pin_supported(model):
            return self.final_tool  # pin the structured write specifically
        return _FORCE_ANY

    def _apply(self, request: Any) -> Any:
        choice = self._decide(request)
        if choice is _RELEASE:
            return request
        return request.override(tool_choice=choice)

    # ── middleware hooks (sync + async) ─────────────────────────────────

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))
