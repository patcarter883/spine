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

A named ``tool_choice`` pin is only as good as the model's willingness to
honor it.  Weak / quantized free models routinely *ignore* the pin and
re-call whatever tool they like — most damagingly the ``gate_tool``, which
they call over and over while the heavy evidence prompt rides along in
every request (trace ``019ec913``: ``poolside/laguna-xs.2:free`` ignored
the ``write_specification`` pin and re-called ``read_work_context`` 17
times per attempt, re-sending a ~32K-token findings blob each turn — 95%
of the trace's 814K input tokens).  Pinning ``tool_choice`` cannot stop a
model that disregards it.  So once the gate tool has fired, this middleware
*removes it from the request's tool surface*: a tool the model can no
longer see is a tool it cannot re-call, no matter how it treats
``tool_choice``.  With the gate gone the only remaining tool is the final
write, so even a plain ``tool_choice="any"`` (the local-server fallback)
resolves to it.  This is the structural backstop the ``tool_choice`` pin
alone is not.
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


def _tool_name(tool: Any) -> str:
    """Extract a tool's name from a BaseTool or an OpenAI tool dict.

    ``request.tools`` is ``list[BaseTool | dict]``; dicts arrive either as the
    OpenAI function envelope ``{"type": "function", "function": {"name": …}}``
    or a flat ``{"name": …}``. Returns ``""`` for unrecognised shapes so the
    caller simply keeps the tool (never drops one it can't identify).
    """
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict):
            return fn.get("name", "") or ""
        return tool.get("name", "") or ""
    return getattr(tool, "name", "") or ""


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
        failure_prefixes: Prefixes that mark a *failed* ``final_tool`` result.
            The write tools return ``"VALIDATION_ERROR: …"`` / ``"ERROR: …"``
            strings when args are bad or the write fails; those keep the
            constraint on so the model can self-correct in the same loop.
            Any other non-error result releases the forcing — matching
            failure is far less fragile than matching a success phrase
            (trace 019eb43f: the plan tool's success message lacked the old
            ``"written to"`` marker, so forcing never released and the
            synthesizer rewrote the plan every turn until context overflow).
    """

    def __init__(
        self,
        final_tool: str,
        gate_tool: str | None = None,
        failure_prefixes: tuple[str, ...] = ("VALIDATION_ERROR", "ERROR"),
    ) -> None:
        self.final_tool = final_tool
        self.gate_tool = gate_tool
        self.failure_prefixes = failure_prefixes

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
                # A rejection ("VALIDATION_ERROR: …" / "ERROR: …") means the
                # write didn't land — keep forcing so the model self-corrects.
                if content.lstrip().startswith(self.failure_prefixes):
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

    def _gate_dropped_tools(self, request: Any) -> list[Any] | None:
        """Tool list with ``gate_tool`` removed, or ``None`` to leave as-is.

        Returns the filtered list only when the gate genuinely needs dropping:
        an OpenAI-style model, a configured ``gate_tool`` that has already been
        called, the final write not yet successful, and the gate still present
        in the surfaced tools. ``None`` everywhere else — so the no-gate
        orchestrator variant (which legitimately re-uses ``recall``) and
        non-OpenAI models are untouched, exactly like ``_decide``.
        """
        if not self.gate_tool:
            return None
        if not _is_openai_style_model(getattr(request, "model", None)):
            return None
        messages = list(getattr(request, "messages", None) or [])
        if self._final_tool_succeeded(messages):
            return None
        if not self._gate_tool_called(messages):
            return None
        tools = list(getattr(request, "tools", None) or [])
        kept = [t for t in tools if _tool_name(t) != self.gate_tool]
        if len(kept) == len(tools):
            return None
        return kept

    def _apply(self, request: Any) -> Any:
        # A single override carries both changes: the structural gate-drop and
        # the tool_choice pin. (Chaining two .override() calls is correct on the
        # real ModelRequest but needlessly fragile against partial test fakes.)
        overrides: dict[str, Any] = {}
        kept_tools = self._gate_dropped_tools(request)
        if kept_tools is not None:
            overrides["tools"] = kept_tools
        choice = self._decide(request)
        if choice is not _RELEASE:
            overrides["tool_choice"] = choice
        if not overrides:
            return request
        return request.override(**overrides)

    # ── middleware hooks (sync + async) ─────────────────────────────────

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))
