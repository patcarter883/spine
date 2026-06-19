"""SPINE agent helpers — shared utilities for all agent builders and phases.

Every agent builder (specify, plan, implement, verify, critic) and every
phase function had identical copies of ``_resolve_model``,
``_debug_enabled``, and ``_extract_response``.  Consolidate them here
to eliminate duplication and ensure consistent behavior.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from contextlib import contextmanager
from typing import Any, Iterator, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_StructuredT = TypeVar("_StructuredT", bound=BaseModel)


@contextmanager
def suppress_parsed_serializer_warning() -> Iterator[None]:
    """Silence the benign ``parsed`` Pydantic serialization warning.

    ``with_structured_output`` round-trips through provider response models
    (e.g. OpenAI's ``ParsedChatCompletionMessage``) whose ``parsed`` field is a
    generic typed ``Optional[...]``; serialising one carrying a structured
    instance makes pydantic-core emit::

        UserWarning: Pydantic serializer warnings:
          PydanticSerializationUnexpectedValue(Expected `none` - serialized value
          may not be as expected [field_name='parsed', input_value=...,
          input_type=...])

    It is cosmetic — the parsed object is still correct — but it floods logs
    during onboarding synthesis (one per section worker / manager call).

    The ``message`` regex MUST account for the warning text being multi-line:
    ``warnings.filterwarnings`` matches it with ``re.match`` (anchored at the
    start) and ``.`` does not cross newlines by default, so a naive
    ``.*PydanticSerializationUnexpectedValue.*parsed`` never matches (it can't
    get past the first ``\\n``). We anchor on the real first line and enable
    DOTALL via ``(?s)`` so ``.*`` reaches the ``parsed`` field on line two — a
    targeted filter that leaves every other serializer warning visible.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?s)Pydantic serializer warnings.*parsed",
            category=UserWarning,
        )
        yield


def resolve_model(
    config: RunnableConfig | None,
    session_id: str | None = None,
    phase: str | None = None,
) -> str | BaseChatModel:
    """Resolve the LLM model identifier from config or SpineConfig.

    Supports per-phase and per-subagent model overrides.  When ``phase`` is
    provided, checks ``SpineConfig.providers.phases.<phase>.model`` before
    falling back to the default provider resolution.

    When the resolved model is an OpenRouter model and a ``session_id`` is
    provided, returns a pre-built :class:`ChatOpenRouter` instance with
    ``session_id`` set — this lets OpenRouter group all requests for a work
    item into a single session on the dashboard.

    When the resolved model starts with ``openai:`` and the provider config
    includes ``base_url`` (i.e. a local/OpenAI-compatible server), returns a
    pre-built :class:`ChatOpenAI` with ``base_url`` and ``api_key`` wired
    in — without this, ``init_chat_model()`` creates a default that looks
    for ``OPENAI_API_KEY`` in the environment, causing a "missing
    credentials" error.

    Otherwise returns the model string for Deep Agents' built-in resolution.

    Args:
        config: LangGraph runtime config (may contain ``configurable.model``).
        session_id: Optional session identifier (typically the work_id) to
            pass to OpenRouter for request grouping. Ignored for non-OpenRouter
            providers.
        phase: Optional phase or phase/subagent path for model override
            resolution (e.g. ``"implement"`` or
            ``"implement/subagents/slice-implementer"``).

    Returns:
        A model string like ``"openrouter:z-ai/glm-4.5-air:free"`` or a
        pre-built ``BaseChatModel`` instance when extra config is needed.
    """
    model_spec = _model_spec_from_config(config, phase=phase)

    # Only build a pre-built model when the provider needs extra kwargs
    # (base_url, api_key, session_id, etc.) that the string-based
    # init_chat_model path would silently drop.
    if session_id and model_spec.startswith("openrouter:"):
        return _build_openrouter_model(model_spec, session_id, phase=phase)

    # For local/OpenAI-compatible servers with custom base_url + api_key,
    # we must build a ChatOpenAI instance ourselves — otherwise
    # init_chat_model() creates one that falls back to OPENAI_API_KEY
    # env var (which isn't set for local servers), producing a
    # "missing credentials" error.
    #
    # Only do this when the model spec came from the active provider
    # (i.e. matches what's in config.yaml).  If the caller explicitly
    # set config["configurable"]["model"] to a different provider
    # (e.g. "openai:gpt-4o-mini" for cloud OpenAI), we must NOT
    # apply the local server's base_url/api_key to it.
    if model_spec.startswith("openai:"):
        provider_cfg = _active_provider_config(phase=phase)
        if (
            provider_cfg
            and provider_cfg.get("base_url")
            and provider_cfg.get("model") == model_spec
        ):
            return _build_local_model(model_spec, provider_cfg)

    return model_spec


def resolve_chat_model(
    config: RunnableConfig | None,
    session_id: str | None = None,
    phase: str | None = None,
) -> BaseChatModel:
    """Resolve a ready-to-invoke :class:`BaseChatModel` from config.

    Thin wrapper over :func:`resolve_model` that always returns a built model
    instance: when ``resolve_model`` returns a string spec (the common case
    where no provider-specific kwargs are needed), it is coerced via
    ``init_chat_model``. This consolidates the ``resolve_model`` + ``if
    isinstance(model, str): init_chat_model(model)`` block that every bare-LLM
    call site (onboarding doc-manager / section-worker, research manager) had
    copied verbatim.

    Args:
        config: LangGraph runtime config (may contain ``configurable.model``).
        session_id: Optional session identifier (typically the work_id) passed
            through to :func:`resolve_model` for OpenRouter request grouping.
        phase: Optional phase or phase/subagent path for model override
            resolution.

    Returns:
        A ``BaseChatModel`` instance ready for ``.with_structured_output`` or
        ``.ainvoke``.
    """
    model = resolve_model(config, session_id=session_id, phase=phase)
    if isinstance(model, str):
        from langchain.chat_models import init_chat_model

        provider_cfg = _active_provider_config(phase=phase)
        cap_kwargs: dict[str, Any] = {}
        if provider_cfg:
            _apply_concurrency_cap(cap_kwargs, provider_cfg)
            # ── Enable streaming + usage reporting (parity with the local /
            # OpenRouter builders) ───────────────────────────────────────
            # The string path routes through init_chat_model, which would
            # otherwise drop these. Without stream_usage the streamed final
            # chunk carries no token counts, producing 0-token spans in
            # LangSmith and starving the per-work_id budget tracker (trace
            # 019ec965: 6/52 explore_do spans reported no usage). streaming
            # keeps the stall timer alive on slow local models. Both are
            # default-on; opt out with providers.llm[].stream_usage: false.
            cap_kwargs.setdefault("streaming", True)
            if provider_cfg.get("stream_usage") is not False:
                cap_kwargs.setdefault("stream_usage", True)
        return init_chat_model(model, **cap_kwargs) if cap_kwargs else init_chat_model(model)
    return model


def cap_completion_tokens(model: BaseChatModel, cap: int) -> BaseChatModel:
    """Return a model_copy with the completion-token cap on the correct field.

    ``ChatOpenAI`` stores the cap as ``max_tokens`` (with ``max_completion_tokens``
    as a Pydantic alias).  ``model_copy(update={"max_completion_tokens": n})``
    silently leaves the underlying ``max_tokens`` field unchanged, so the API
    call still uses the original value.  We detect which field is actually set
    and update that one — matching the defensive pattern already used in
    ``exploration_agents._cap_findings_model``.
    """
    if getattr(model, "max_tokens", None) is not None:
        return model.model_copy(update={"max_tokens": cap})
    return model.model_copy(update={"max_completion_tokens": cap})


def suppress_reasoning(model: BaseChatModel) -> BaseChatModel:
    """Return a model_copy with thinking disabled — no-op for non-local models.

    Mechanical structured-output calls (supervisor directives, findings
    extraction) gain nothing from chain-of-thought, but on local thinking
    models (e.g. Qwen3.6) the reasoning channel consumes the tight
    ``cap_completion_tokens`` budget before any JSON is emitted, raising
    ``LengthFinishReasonError`` (trace 019eafac). We send the same two
    suppression levers as :func:`_build_local_model` (see its
    ``reasoning: false`` block for why both are needed): ``reasoning_budget:
    0`` for budget-honouring models and ``enable_thinking: false`` for
    template-gated ones.

    Scoped to ``ChatOpenAI`` instances pointing at a non-OpenAI ``base_url``
    (local/OpenAI-compatible servers). ChatOpenRouter is not a ChatOpenAI
    subclass and real OpenAI endpoints reject unknown extra_body keys —
    both pass through unchanged. Merges into any existing ``extra_body``
    rather than replacing it, so a provider-level ``reasoning: false``
    config stays intact (idempotent). Fails open on any error.
    """
    try:
        from langchain_openai import ChatOpenAI

        if not isinstance(model, ChatOpenAI):
            return model
        base_url = getattr(model, "openai_api_base", None)
        if not base_url or "api.openai.com" in str(base_url):
            return model
        extra = dict(getattr(model, "extra_body", None) or {})
        template_kwargs = dict(extra.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = False
        extra["reasoning_budget"] = 0
        extra["chat_template_kwargs"] = template_kwargs
        return model.model_copy(update={"extra_body": extra})
    except Exception:
        logger.debug("suppress_reasoning: copy failed — using model as-is", exc_info=True)
        return model


def disable_streaming(model: BaseChatModel) -> BaseChatModel:
    """Return a model_copy with streaming off — for single-shot structured calls.

    The model builders force ``streaming=True`` so LangGraph's
    ``stream_mode=["messages"]`` keeps the stall timer alive during long agent
    loops (see ``_build_local_model`` / ``_build_openrouter_model``). But a
    one-shot ``ainvoke()`` of a tiny structured decision (e.g. the
    ``research_manager``'s ``explore``/``done`` choice) is not an agent loop: it
    gains nothing from token streaming, and on local llama.cpp/vLLM backends a
    mid-stream SSE break surfaces as ``openai.APIError('An error occurred during
    streaming')`` that aborts the whole call at zero tokens (trace 019ecdea).
    Routing it through the non-streaming completions endpoint removes that
    failure mode — and the final response still carries usage, so token
    accounting is unaffected. Fails open on any error.
    """
    try:
        return model.model_copy(update={"streaming": False})
    except Exception:
        logger.debug("disable_streaming: copy failed — using model as-is", exc_info=True)
        return model


def _is_openai_style_model(model: Any) -> bool:
    """True when ``model``'s ``with_structured_output`` supports json_schema.

    Covers ``ChatOpenAI`` (and local OpenAI-compatible vLLM via ``ChatOpenAI``)
    and ``langchain_openrouter.ChatOpenRouter``.  NB: ChatOpenRouter is *not* a
    ChatOpenAI subclass — it extends ``BaseChatModel`` directly — but exposes
    the same ``method="json_schema"`` structured-output path, so it must be
    checked explicitly.  Returns ``False`` for ``ChatAnthropic``, string specs,
    and test fakes.
    """
    candidates: list[type] = []
    try:
        from langchain_openai import ChatOpenAI

        candidates.append(ChatOpenAI)
    except ImportError:
        pass
    try:
        from langchain_openrouter import ChatOpenRouter

        candidates.append(ChatOpenRouter)
    except ImportError:
        pass
    return bool(candidates) and isinstance(model, tuple(candidates))


def bind_structured_output(model: Any, schema: type[BaseModel]) -> Any:
    """Bind ``schema`` for structured output, avoiding forced ``tool_choice``.

    ``model.with_structured_output(schema)`` defaults to
    ``method="function_calling"`` on OpenAI-compatible models, which forces a
    ``tool_choice`` value to make the model call the extraction function.
    Combined with our ``require_parameters=True`` OpenRouter setting (see
    :func:`_build_openrouter_model`), any model whose endpoints don't support
    that value is rejected up-front with::

        NotFoundResponseError: No endpoints found that support the
        provided 'tool_choice' value.

    Because every bare ``with_structured_output`` callsite wraps its invoke in
    a ``try/except`` fallback, that 404 doesn't crash the run — it silently
    degrades each planner/decomposer/supervisor call to its floor (empty
    directive, terminating directive, skeleton plan).

    For OpenAI-style models we therefore request ``method="json_schema"``,
    which uses the native ``response_format`` path and sends no ``tool_choice``
    at all.  Other providers (e.g. ``ChatAnthropic``, which has no
    ``response_format`` and handles forced tool choice fine) keep their default
    method.  ``method="json_schema"`` is selected only at bind time; if a model
    later rejects ``response_format`` json_schema at invoke time, the callsite's
    existing ``try/except`` still degrades gracefully.

    The json_schema default has its own mirror-image failure (trace 019eaf1f):
    OpenRouter gates ``response_format: json_schema`` behind the separate
    ``structured_outputs`` endpoint capability, and some models (e.g.
    minimax/minimax-m3) advertise ``tools``/``tool_choice`` but NOT
    ``structured_outputs`` — so with ``require_parameters=True`` every call
    404s before reaching the provider.  For ChatOpenRouter models we therefore
    consult the model's endpoint capabilities (cached, fail-open) and pick the
    method its endpoints can actually serve.

    The capability probe alone is not sufficient: ``supported_parameters``
    only says an endpoint accepts a parameter, not which *values* it accepts.
    minimax-m3 advertises ``tool_choice`` but rejects a forced named function
    (trace 019eaf2a: 404 "No endpoints found that support the provided
    'tool_choice' value"), which is exactly what ``method="function_calling"``
    sends.  ChatOpenRouter bindings are therefore returned as a
    :class:`_SelfHealingStructured` wrapper that catches routing 404s at
    invoke time, demotes the method down the ladder (json_schema →
    function_calling → json_mode), updates the per-model cache so concurrent
    and future binds skip the dead method, and retries the same input.
    """
    if not _is_openai_style_model(model):
        return model.with_structured_output(schema)
    if type(model).__name__ == "ChatOpenRouter":
        model_name = getattr(model, "model_name", None)
        if model_name:
            method = _openrouter_structured_method(model_name)
            return _SelfHealingStructured(model, schema, method)
    return model.with_structured_output(schema, method="json_schema")


# Structured-output methods in order of preference. Demotion on a routing
# 404 steps one place right; past the end there is nothing left to try and
# the error propagates to the callsite's existing fallback.
_STRUCTURED_METHOD_LADDER = ("json_schema", "function_calling", "json_mode")


def _is_structured_routing_404(exc: BaseException) -> bool:
    """True for OpenRouter's up-front "No endpoints found" routing rejection.

    With ``require_parameters=True`` OpenRouter rejects a request before it
    reaches any provider when no endpoint supports the parameters (or
    parameter *values*) sent — always HTTP 404 with a message starting
    "No endpoints found". Matched structurally (status_code attribute +
    message text) rather than by exception class so we don't need a hard
    dependency on ``langchain_openrouter``'s exception hierarchy.
    """
    return getattr(exc, "status_code", None) == 404 and "No endpoints found" in str(exc)


def _demoted_method(method: str) -> str | None:
    """Next structured-output method down the ladder, or None at the bottom."""
    try:
        idx = _STRUCTURED_METHOD_LADDER.index(method)
    except ValueError:
        return None
    if idx + 1 >= len(_STRUCTURED_METHOD_LADDER):
        return None
    return _STRUCTURED_METHOD_LADDER[idx + 1]


class _SelfHealingStructured:
    """Structured-output binding that demotes its method on routing 404s.

    Endpoint capability listings can't reveal value-level support (trace
    019eaf2a: minimax-m3 advertises ``tool_choice`` but rejects a forced
    named function), so the only reliable detection point is the 404 at
    invoke time.  On each ``ainvoke`` this wrapper binds with its current
    method, and when OpenRouter rejects routing it demotes one ladder step,
    records the demotion in ``_structured_method_cache`` (so other in-flight
    workers and future binds for the same model skip straight past the dead
    method), and retries the same input.  At the bottom of the ladder the
    error propagates unchanged, preserving every callsite's existing
    ``try/except`` degradation path.

    Only ``ainvoke`` is exposed: every structured callsite in SPINE invokes
    through ``await structured.ainvoke(...)`` (directly or via
    :func:`ainvoke_structured_with_retry`).
    """

    def __init__(self, model: Any, schema: type[BaseModel], method: str) -> None:
        self._model = model
        self._schema = schema
        self._method = method
        self._model_name: str = getattr(model, "model_name", "") or ""

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        # Adopt a demotion another worker already paid a 404 to discover.
        if self._model_name:
            cached = _structured_method_cache.get(self._model_name)
            if cached:
                self._method = cached
        while True:
            bound = self._model.with_structured_output(self._schema, method=self._method)
            try:
                return await bound.ainvoke(input, config, **kwargs)
            except Exception as exc:
                next_method = (
                    _demoted_method(self._method)
                    if _is_structured_routing_404(exc)
                    else None
                )
                if next_method is None:
                    raise
                logger.warning(
                    "%s: structured-output method=%s rejected by OpenRouter "
                    "routing (404); demoting to method=%s and retrying",
                    self._model_name or type(self._model).__name__,
                    self._method,
                    next_method,
                )
                self._method = next_method
                if self._model_name:
                    _structured_method_cache[self._model_name] = next_method


_OPENROUTER_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"

# model name -> chosen with_structured_output method. Populated lazily; also
# caches the fail-open default on lookup errors so a flaky network costs at
# most one blocked bind per model per process.
_structured_method_cache: dict[str, str] = {}


def _openrouter_structured_method(model_name: str) -> str:
    """Pick the ``with_structured_output`` method this model's endpoints serve.

    OpenRouter's per-model endpoint listing advertises ``supported_parameters``
    per endpoint.  With ``require_parameters=True`` a request succeeds if ANY
    endpoint supports every parameter sent, so we check the union across
    endpoints, preferring the most reliable method available:

    1. ``json_schema`` — needs the ``structured_outputs`` capability.
    2. ``function_calling`` — needs ``tool_choice`` (forced tool call).
    3. ``json_mode`` — needs only plain ``response_format`` (json_object);
       weakest guarantee, but better than a guaranteed 404.

    Fails open to ``json_schema`` (the previous unconditional behavior) when
    the lookup errors or returns no endpoints. The result — including the
    fail-open default — is cached per model for the process lifetime.
    """
    cached = _structured_method_cache.get(model_name)
    if cached is not None:
        return cached
    method = "json_schema"
    try:
        import httpx

        resp = httpx.get(
            _OPENROUTER_ENDPOINTS_URL.format(model=model_name), timeout=5.0
        )
        resp.raise_for_status()
        endpoints = (resp.json().get("data") or {}).get("endpoints") or []
        if endpoints:
            supported: set[str] = set()
            for ep in endpoints:
                supported.update(ep.get("supported_parameters") or [])
            if "structured_outputs" not in supported:
                if "tool_choice" in supported:
                    method = "function_calling"
                elif "response_format" in supported:
                    method = "json_mode"
                logger.info(
                    "%s: endpoints lack structured_outputs support; using "
                    "structured-output method=%s",
                    model_name,
                    method,
                )
    except Exception as exc:  # noqa: BLE001 — fail open to the static default
        logger.debug(
            "OpenRouter endpoint lookup failed for %s (%s); defaulting to "
            "json_schema",
            model_name,
            exc,
        )
    _structured_method_cache[model_name] = method
    return method


# Substring of the ValueError LangChain's ``_oai_structured_outputs_parser``
# raises when an OpenAI-style model returns finish_reason='stop' with empty
# content and no parsed object — i.e. the json_schema response carried neither a
# ``parsed`` field nor a ``refusal``. Matched on the message because LangChain
# raises a bare ``ValueError`` for this case (no dedicated exception type).
_EMPTY_STRUCTURED_PARSE_MARKER = "does not have a 'parsed' field"


def is_empty_structured_parse(exc: BaseException) -> bool:
    """True for the transient json_schema empty-parse ``ValueError``.

    Distinct from ``openai.LengthFinishReasonError`` (the token-cap truncation
    case, which is a separate exception type left for each caller's own salvage
    path). This covers the failure where a vLLM-served local model emits an
    empty completion under guided decoding and produces no structured object;
    the run finishes with ``finish_reason='stop'`` so it is *not* a length
    issue and is almost always recoverable on a re-invoke.
    """
    return isinstance(exc, ValueError) and _EMPTY_STRUCTURED_PARSE_MARKER in str(exc)


def _retryable_api_error(exc: BaseException) -> bool:
    """True when ``exc`` is a transient OpenAI-compatible transport failure.

    Targets the failure modes that resolve on a retry of the *same* request:

    - bare ``openai.APIError`` — what local llama.cpp/vLLM backends raise on a
      mid-stream SSE break (``"An error occurred during streaming"``, trace
      019ecdea); it has no ``status_code`` so it's not an ``APIStatusError``;
    - ``APIConnectionError`` / ``APITimeoutError`` — dropped or stalled sockets;
    - 5xx and 429 ``APIStatusError`` — server-side / rate-limit hiccups.

    Deterministic client errors (400/401/404/422 ``BadRequestError`` etc.) are
    *not* retryable — re-sending the identical request just fails again — so
    they return ``False`` and propagate to the caller's fallback unchanged.
    Returns ``False`` for any non-OpenAI exception (and if ``openai`` can't be
    imported), so the caller's existing handling is never widened.
    """
    try:
        from openai import APIError, APIStatusError
    except Exception:
        return False
    if not isinstance(exc, APIError):
        return False
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is None or status >= 500 or status == 429
    return True


async def ainvoke_structured_with_retry(
    structured_model: Any,
    messages: list[Any],
    *,
    retries: int = 1,
    api_retries: int = 2,
    api_backoff: float = 0.5,
    label: str = "structured-output",
) -> Any:
    """Invoke a structured-output model, retrying transient empty parses.

    Some json_schema-mode models occasionally return an empty completion with no
    parsed object, which LangChain surfaces as a ``ValueError`` (see
    :func:`is_empty_structured_parse`). The failure is non-deterministic, so we
    retry up to ``retries`` times, appending a short nudge so a deterministic
    decoder doesn't simply reproduce the empty result.

    Transient transport failures — a mid-stream SSE break on a local backend
    (``openai.APIError``), dropped sockets, 5xx/429 — are retried separately up
    to ``api_retries`` times with exponential backoff and *no* nudge (the
    request was well-formed; only the connection broke). See
    :func:`_retryable_api_error` for exactly which errors qualify; deterministic
    4xx client errors are never retried.

    Every other exception — including ``LengthFinishReasonError`` — propagates
    immediately, and the empty-parse ``ValueError`` re-raises once ``retries``
    are exhausted, so each caller's existing ``try/except`` fallback still runs
    unchanged (empty/terminating directive, sentinel, salvage, …).

    Args:
        structured_model: A model already bound via ``with_structured_output``.
        messages: Messages to invoke with; never mutated.
        retries: Max extra attempts after the first on empty parse (default 1 →
            up to 2 calls).
        api_retries: Max extra attempts on a transient API/transport error
            (default 2). Set to 0 to disable transport retries.
        api_backoff: Base seconds for exponential backoff between transport
            retries (delay = ``api_backoff * 2**(n-1)``).
        label: Identifier used in the retry log line.

    Returns:
        Whatever ``structured_model.ainvoke`` returns on the first success.
    """
    nudge = HumanMessage(
        content=(
            "Your previous response was empty. Respond with ONLY the JSON "
            "object for the required schema — no prose, no preamble."
        )
    )
    # Local import avoids any import-order coupling; retry.py is standalone.
    from spine.agents import retry as _retry

    attempt = 0
    api_attempt = 0
    while True:
        invoke_messages = messages if attempt == 0 else [*messages, nudge]
        try:
            result = await structured_model.ainvoke(invoke_messages)
            # Reachable endpoint → clear the connection-failure circuit breaker.
            _retry.reset_conn_breaker()
            return result
        except ValueError as exc:
            if attempt >= retries or not is_empty_structured_parse(exc):
                raise
            attempt += 1
            logger.warning(
                "%s: empty structured parse — retrying (%d/%d)",
                label,
                attempt,
                retries,
            )
        except Exception as exc:
            # Run-level breaker: this structured path is the dominant offender
            # under a down local server (trace 019ece87: decomposer 12684 calls).
            # Trip BEFORE the per-call retry branch so ServerUnreachable aborts
            # the run rather than being absorbed by api_retries.
            _retry._trip_breaker_if_unreachable(exc, prefix="", phase_label=label)
            if api_attempt >= api_retries or not _retryable_api_error(exc):
                raise
            api_attempt += 1
            delay = api_backoff * (2 ** (api_attempt - 1))
            logger.warning(
                "%s: transient API error %s — retrying (%d/%d) after %.1fs",
                label,
                type(exc).__name__,
                api_attempt,
                api_retries,
                delay,
            )
            await asyncio.sleep(delay)


def coerce_structured_output(
    response: Any,
    schema: type[_StructuredT],
) -> _StructuredT | None:
    """Coerce a ``with_structured_output`` response into a ``schema`` instance.

    Handles the three response shapes seen across LangChain/provider versions:

    1. the Pydantic ``schema`` instance directly;
    2. an ``AIMessage`` carrying the parsed model on ``.parsed`` (which is
       reset to ``None`` afterwards to avoid the
       ``PydanticSerializationUnexpectedValue`` warning on re-serialisation);
    3. a plain ``dict`` validated through ``schema.model_validate``.

    Returns ``None`` when the shape is unrecognised or a dict fails validation
    so callers can fall back to a deterministic floor (e.g. the skeleton plan
    or a generic error result) — it never raises.

    Args:
        response: The raw return value of ``model.with_structured_output(...)
            .ainvoke(...)``.
        schema: The expected Pydantic model class.

    Returns:
        A ``schema`` instance, or ``None`` if the response can't be coerced.
    """
    if isinstance(response, schema):
        return response
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        try:
            response.parsed = None  # prevent Pydantic serialization warning
        except Exception:  # noqa: BLE001 - response may be immutable
            pass
        return parsed
    if isinstance(response, dict):
        try:
            return schema.model_validate(response)
        except Exception:  # noqa: BLE001 - malformed dict → caller falls back
            return None
    return None


def _model_spec_from_config(config: RunnableConfig | None, phase: str | None = None) -> str:
    """Extract the model spec string from config or SpineConfig.

    Checks ``config["configurable"]["model"]`` first, then delegates to
    ``SpineConfig.resolve_model(phase=phase)`` which handles per-phase
    overrides and the default provider resolution.

    Args:
        config: LangGraph runtime config.
        phase: Optional phase or phase/subagent path for override resolution.

    Returns:
        A model spec string like ``"openrouter:z-ai/glm-4.5-air:free"``.
    """
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_model(phase=phase)


def _active_provider_config(phase: str | None = None) -> dict[str, Any] | None:
    """Return the full provider config dict for a given phase.

    Delegates to :meth:`SpineConfig.resolve_provider_config` so that
    per-phase overrides (``base_url``, ``api_key``, ``provider``
    references, etc.) are applied before returning.

    Args:
        phase: Optional phase or phase/subagent path for policy resolution.

    Returns:
        The merged provider config dict, or ``None`` if no provider found.
    """
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_provider_config(phase=phase)


def _build_openrouter_model(
    model_spec: str,
    session_id: str,
    phase: str | None = None,
) -> BaseChatModel:
    """Build a ChatOpenRouter instance with session_id set.

    Applies the DA ProviderProfile for OpenRouter (app_url, app_title,
    openrouter_provider defaults) before constructing the model, so we
    don't lose the attribution headers and Azure-ignore rule that the
    string-based ``init_chat_model`` path would normally provide.

    Sets a default ``request_timeout`` of 300 seconds (5 minutes) to
    prevent hung connections from blocking the workflow indefinitely.
    Provider config can override via ``providers.llm[].request_timeout``
    or a per-phase ``request_timeout`` field.

    Args:
        model_spec: Full model spec like ``"openrouter:z-ai/glm-4.5-air:free"``.
        session_id: Work item ID for OpenRouter request grouping.
        phase: Optional phase path for provider config resolution
            (e.g. ``"implement"`` or ``"implement/subagents/slice-implementer"``).

    Returns:
        A configured ``ChatOpenRouter`` instance.
    """
    from deepagents.profiles.provider import apply_provider_profile

    from langchain_openrouter import ChatOpenRouter

    # Strip the "openrouter:" prefix to get the raw model name
    model_name = model_spec.removeprefix("openrouter:")

    # OpenRouter limits session_id to 128 characters
    truncated_session_id = session_id[:128]

    # Apply DA ProviderProfile kwargs (app_url, app_title,
    # openrouter_provider, etc.) so we don't lose defaults that the
    # string-based init_chat_model path would normally inject.
    profile_kwargs = apply_provider_profile(model_spec)

    # ── Resolve request_timeout ──────────────────────────────────────
    # Default: 300s (5 min).  Provider config can override this via
    # providers.llm[].request_timeout.  Without a timeout, hung
    # connections (e.g. OpenRouter dropping mid-stream) can block
    # the workflow for 30+ minutes waiting for OS-level TCP timeouts.
    # Note: ChatOpenRouter expects milliseconds, not seconds.
    timeout_ms = _resolve_timeout_from_config(default=300, phase=phase) * 1000

    # ── Resolve max_completion_tokens ────────────────────────────────
    # When max_completion_tokens is not set, reasoning models (e.g.
    # DeepSeek-v4-flash) can consume their entire output budget on
    # chain-of-thought tokens, leaving the visible content truncated
    # mid-generation.  Setting an explicit limit ensures the model
    # allocates enough output budget to produce complete artifacts.
    #
    # Provider config can override via providers.llm[].max_completion_tokens
    # or providers.llm[].max_tokens.  max_completion_tokens is preferred
    # (it includes reasoning tokens in the budget, giving the model full
    # control over allocation).
    provider_cfg = _active_provider_config(phase=phase) or {}
    max_completion_tokens = provider_cfg.get("max_completion_tokens")
    max_tokens = provider_cfg.get("max_tokens")
    # Fall back to the global SpineConfig.max_completion_tokens when the
    # provider hasn't set its own cap. Prevents finite-window providers
    # from being asked to allocate the entire remaining context as output
    # budget. See SpineConfig.max_completion_tokens for the trace context.
    if max_completion_tokens is None and max_tokens is None:
        from spine.config import SpineConfig as _SpineConfig
        global_cap = _SpineConfig.load().max_completion_tokens
        if global_cap and global_cap > 0:
            max_completion_tokens = global_cap

    # Merge OpenRouter provider preferences. require_parameters=True makes
    # OpenRouter reject the request up-front if the chosen model doesn't
    # support every parameter we send (notably response_format/json_schema),
    # instead of silently dropping them and returning unstructured text.
    provider_prefs: dict[str, Any] = dict(profile_kwargs.pop("openrouter_provider", {}) or {})
    provider_prefs.setdefault("require_parameters", True)

    model_kwargs: dict[str, Any] = {
        "model": model_name,
        "session_id": truncated_session_id,
        "request_timeout": timeout_ms,
        "openrouter_provider": provider_prefs,
        **profile_kwargs,
    }
    # ── Explicitly pass api_key from environment or provider config ──
    # ChatOpenRouter validates OPENROUTER_API_KEY in os.environ on
    # construction.  Worker threads/subprocesses may not inherit the
    # parent shell's env vars, so we pass the key explicitly instead of
    # relying on the implicit check.  Prefer the provider config's
    # api_key field, then fall back to the environment variable.
    api_key = provider_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        model_kwargs["api_key"] = api_key
    # Send the resolved cap as `max_tokens`, NOT `max_completion_tokens`.
    # OpenRouter routes across many provider endpoints, the majority of which
    # advertise `max_tokens` but NOT `max_completion_tokens` in their
    # supported_parameters (e.g. every deepseek-v4-flash endpoint).  With
    # require_parameters=True (set above), sending `max_completion_tokens`
    # makes OpenRouter reject the request up-front with HTTP 404 "No endpoints
    # found that can handle the requested parameters" — even though the model
    # itself is fine.  `max_tokens` is the portable field OpenRouter normalises
    # for every provider, so collapse both config fields onto it.
    cap = max_completion_tokens if max_completion_tokens is not None else max_tokens
    if cap is not None:
        model_kwargs["max_tokens"] = int(cap)

    # ── Enable streaming for stall detection ─────────────────────────
    # ChatOpenRouter defaults to streaming=False.  Without streaming,
    # no on_llm_new_token callbacks fire during generation, so LangGraph's
    # stream_mode=["messages"] yields nothing — and the stall detector
    # falsely fires on slow models (reasoning models, long outputs).
    model_kwargs.setdefault("streaming", True)

    # ── Enable usage_metadata on streamed responses ─────────────────
    # ChatOpenRouter inherits ChatOpenAI; stream_usage=True causes it
    # to send stream_options: {"include_usage": true} on the request,
    # which makes the final stream chunk carry the token counts.
    # Without it, LangSmith records 0 tokens on every call and the
    # token-budget tracker is blind.  Default-on; set
    # providers.llm[].stream_usage: false to opt out for a misbehaving
    # provider.
    if provider_cfg.get("stream_usage") is not False:
        model_kwargs.setdefault("stream_usage", True)

    _apply_concurrency_cap(model_kwargs, provider_cfg)

    return ChatOpenRouter(**model_kwargs)


def _build_local_model(
    model_spec: str,
    provider_cfg: dict[str, Any],
) -> BaseChatModel:
    """Build a ChatOpenAI instance for a local/OpenAI-compatible server.

    When the provider config includes ``base_url`` and ``api_key`` (e.g. a
    local vLLM instance), we construct a :class:`ChatOpenAI` directly so
    those fields are wired in.  Without this, ``init_chat_model("openai:…")``
    creates a default ``ChatOpenAI`` that looks for ``OPENAI_API_KEY`` in the
    environment — which doesn't exist for local servers, causing a
    "missing credentials" error.

    Schema-constrained decoding (``response_format``) is bound at the call
    site by ``subagents._bind_response_format`` so the schema is attached
    after the model is built — no schema argument is needed here.

    Args:
        model_spec: Full model spec like ``"openai:model"``.
        provider_cfg: The full provider dict from config (has ``base_url``,
            ``api_key``, ``temperature``, etc.).

    Returns:
        A configured ``ChatOpenAI`` instance pointed at the local server.
    """
    from langchain_openai import ChatOpenAI

    model_name = model_spec.removeprefix("openai:")

    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": provider_cfg["base_url"],
    }

    # api_key is required by ChatOpenAI even for local servers that ignore it
    if api_key := provider_cfg.get("api_key"):
        kwargs["api_key"] = api_key

    # Pass through optional tuning fields if present
    for key in (
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "max_retries",
        "request_timeout",
        "stream_chunk_timeout",
    ):
        if key in provider_cfg:
            kwargs[key] = provider_cfg[key]

    # ── Per-chunk streaming watchdog for local servers ────────────────
    # langchain_openai raises StreamChunkTimeoutError when no parsed chunk
    # arrives within `stream_chunk_timeout` (default 120s). On local 30B-class
    # backends the model emits NO chunk until prefill completes, and prefill
    # scales with prompt size (~6-7s per 1K tokens measured — trace D13): a
    # 20K-token prompt prefills past 120s and the watchdog fires before the
    # first token, marking the call failed. Default the timeout to cover a
    # near-full-context-window prefill so large implement/plan prompts don't
    # false-timeout; providers can still override explicitly.
    if "stream_chunk_timeout" not in kwargs:
        ctx_window = provider_cfg.get("context_window")
        kwargs["stream_chunk_timeout"] = (
            max(300.0, (ctx_window / 1000.0) * 8.0) if ctx_window else 600.0
        )

    # ── Disable model reasoning when `reasoning: false` is configured ──
    # Reasoning ("thinking") models spend their output budget on
    # chain-of-thought before emitting the answer. For mechanical phases
    # like extractive summarization that don't benefit from reasoning, this
    # is pure waste: with a tight completion cap the model is truncated
    # mid-thought and `content` comes back empty (reasoning lands in the
    # separate reasoning_content channel).
    #
    # Two suppression levers are sent because llama.cpp models split on which
    # one they honour: `reasoning_budget: 0` works for hybrid models that read
    # the server-side budget, while template-gated models (e.g. Gemma/Qwen
    # instruct) only stop thinking when their chat template sees
    # `enable_thinking: false`. Empirically the Gemma-4-26B-A4B QAT build
    # *ignores* reasoning_budget entirely and reasons to the token cap (empty
    # `content`); enable_thinking=false is the only switch it respects. Sending
    # both is safe — each model picks up whichever it understands and ignores
    # the unused jinja kwarg. Only injected when reasoning is explicitly False;
    # unset/True leaves the model's default behaviour intact.
    if provider_cfg.get("reasoning") is False:
        kwargs["extra_body"] = {
            "reasoning_budget": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    # Fall back to global SpineConfig.max_completion_tokens when neither
    # max_completion_tokens nor max_tokens is set on the provider. Without
    # this, finite-window local backends (vLLM/SGLang) use the full
    # remaining window as output budget and 400 when prompt+budget exceeds
    # the model window. See SpineConfig.max_completion_tokens.
    if "max_completion_tokens" not in kwargs and "max_tokens" not in kwargs:
        from spine.config import SpineConfig as _SpineConfig
        global_cap = _SpineConfig.load().max_completion_tokens
        if global_cap and global_cap > 0:
            kwargs["max_completion_tokens"] = int(global_cap)

    # ── Clamp the completion reservation against the context window ───
    # The completion cap is a slot the server must reserve in its KV cache
    # *in addition to* the prompt. If it approaches the configured
    # context_window, a normal-sized prompt no longer fits and a finite local
    # backend (llama.cpp/vLLM) OOM-crashes while allocating the generation
    # slot — dropping every in-flight request with "CURL error: Could not
    # connect" and fanning out into a fallback-decompose retry storm (trace
    # 019ed360). Never let the reservation exceed half the window, which
    # guarantees at least that much room for the prompt. Phases that set a sane
    # per-phase cap (implement/synthesize = 8K, decompose = 4K) are untouched;
    # this only catches calls that inherited the large global cap. The
    # DynamicCompletionCapMiddleware still trims further per-turn for the
    # tool-using agents that carry it.
    context_window = provider_cfg.get("context_window")
    cap_field = (
        "max_tokens"
        if "max_tokens" in kwargs
        else "max_completion_tokens"
        if "max_completion_tokens" in kwargs
        else None
    )
    if context_window and cap_field and kwargs.get(cap_field):
        ceiling = int(context_window) // 2
        if 0 < ceiling < int(kwargs[cap_field]):
            logger.warning(
                "Clamping %s %d -> %d to fit context_window=%d "
                "(reserve room for the prompt; avoids backend OOM)",
                cap_field,
                int(kwargs[cap_field]),
                ceiling,
                int(context_window),
            )
            kwargs[cap_field] = ceiling

    # ── Default request_timeout ───────────────────────────────────────
    # If not explicitly configured, default to 300s (5 min) to prevent
    # hung connections from blocking the workflow for 30+ minutes.
    if "request_timeout" not in kwargs:
        kwargs["request_timeout"] = 300

    # ── Enable streaming for stall detection ─────────────────────────
    # Without streaming=True, ChatOpenAI uses the non-streaming API
    # endpoint and emits no on_llm_new_token callbacks during generation.
    # LangGraph's stream_mode=["messages"] hooks into those callbacks to
    # yield token-level chunks — which is what keeps the stall timer
    # alive during long agent runs.  With streaming=False (the default),
    # a slow local model that takes >120s to generate produces zero
    # intermediate chunks, and the stall detector falsely marks the work
    # as stalled even though inference is still running.
    kwargs.setdefault("streaming", True)

    # ── Enable stream_usage for token counting (default-on) ──────────
    # ChatOpenAI sends `stream_options: {"include_usage": true}` when
    # stream_usage=True so the final stream chunk carries token counts —
    # required for LangSmith reporting and the per-work_id budget tracker.
    # Default-on parity with _build_openrouter_model. Some strict local
    # vLLM backends 400 on unexpected request fields; opt out by setting
    # providers.llm[].stream_usage: false for that provider.
    if provider_cfg.get("stream_usage") is not False:
        kwargs.setdefault("stream_usage", True)

    _apply_concurrency_cap(kwargs, provider_cfg)

    return ChatOpenAI(**kwargs)


def _apply_concurrency_cap(
    kwargs: dict[str, Any],
    provider_cfg: dict[str, Any],
) -> None:
    """Inject shared httpx clients when ``max_concurrent_calls`` is set.

    Both ChatOpenAI and ChatOpenRouter accept ``http_client`` /
    ``http_async_client``; using cached, connection-capped clients keyed
    by provider name caps concurrent in-flight requests globally across
    every agent that resolves to the same provider.
    """
    raw = provider_cfg.get("max_concurrent_calls")
    if raw is None:
        return
    try:
        max_concurrent = int(raw)
    except (TypeError, ValueError):
        return
    if max_concurrent <= 0:
        return

    from spine.agents.http_clients import get_async_http_client, get_sync_http_client

    provider_name = provider_cfg.get("name") or "default"
    kwargs.setdefault("http_async_client", get_async_http_client(provider_name, max_concurrent))
    kwargs.setdefault("http_client", get_sync_http_client(provider_name, max_concurrent))


def _extract_model_name(model: Any) -> str:
    """Extract a lowercase model name string from a model spec or instance.

    Handles:
    - String specs: ``"openrouter:qwen/qwen3-235b-a22b:free"`` → ``"qwen/qwen3-235b-a22b"``
    - String specs without org: ``"openai:gpt-4o"`` → ``"gpt-4o"``
    - Pre-built instances with ``.model`` attr (ChatOpenRouter, ChatOpenAI)
    - Pre-built instances with ``.model_name`` attr (ChatAnthropic)

    Returns:
        Lowercase model name with provider prefix and quality suffix stripped.
    """
    raw: str = ""
    if isinstance(model, str):
        raw = model
    elif hasattr(model, "model_name"):
        raw = str(model.model_name)
    elif hasattr(model, "model"):
        raw = str(model.model)

    # Strip provider prefix (e.g. "openrouter:" or "openai:")
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    # Strip trailing quality suffix (:free, :beta, etc.) — these appear
    # after the model name and contain only short alpha-only strings.
    # Model names like "qwen3-235b-a22b" contain hyphens/digits, so a
    # trailing segment with only letters + digits/periods is a quality tag.
    while ":" in raw:
        last_part = raw.rsplit(":", 1)[1]
        # If the last segment contains no "/" and looks like a quality tag
        # (short, no hyphens suggesting model version), strip it.
        if "/" not in last_part and not any(c == "-" for c in last_part):
            raw = raw.rsplit(":", 1)[0]
        else:
            break

    return raw.lower()


def debug_enabled() -> bool:
    """Check if LLM debug logging is enabled via the SPINE_DEBUG_LLM env var."""
    return os.getenv("SPINE_DEBUG_LLM", "").strip().lower() in ("1", "true", "yes")


def _resolve_timeout_from_config(default: int = 300, phase: str | None = None) -> int:
    """Resolve the request_timeout in seconds from provider config.

    Checks the phase-aware provider config for a ``request_timeout`` field.
    Falls back to ``default`` when not configured.  The return value is
    always in **seconds** — callers must convert to milliseconds when
    needed by the underlying client.

    Args:
        default: Default timeout in seconds when not configured.
        phase: Optional phase path for provider config resolution.

    Returns:
        Timeout value in seconds.
    """
    provider_cfg = _active_provider_config(phase=phase)
    if provider_cfg and "request_timeout" in provider_cfg:
        try:
            return int(provider_cfg["request_timeout"])
        except (ValueError, TypeError):
            pass
    return default


def extract_response(result: dict[str, Any]) -> str:
    """Extract the text content from a Deep Agent's last message.

    For thinking/reasoning models (e.g. DeepSeek-v4-flash), the final
    message content may be chain-of-thought reasoning rather than
    structured output.  We detect this pattern and return an empty string
    to avoid polluting artifacts with leaked reasoning.

    Args:
        result: The agent result dict (has ``"messages"`` key).

    Returns:
        The content string of the final message, or empty string if the
        content appears to be leaked reasoning or is absent.
    """
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))
        # ── Try content first ──────────────────────────────────────────
        if content and len(content.strip()) > 0:
            # Detect leaked thinking-model reasoning
            stripped = content.strip()
            if (
                stripped
                and not stripped[0].isupper()
                and stripped[0] not in ("#", "*", "-", "|", "`", "[", '"')
            ):
                # Looks like reasoning, not a structured artifact
                return ""
            return content
        # ── Fall back to reasoning_content for thinking models ─────────
        reasoning = getattr(last, "additional_kwargs", {}).get("reasoning_content", "") or ""
        if reasoning and len(reasoning.strip()) > 10:
            return reasoning.strip()
    return ""