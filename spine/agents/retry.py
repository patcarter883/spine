"""SPINE agent retry — exponential backoff for transient LLM API errors.

When an LLM provider returns a transient error (HTTP 5xx, rate limit 429,
or an OpenRouter ``ResponseValidationError`` from an error body), the
agent invocation should be retried with exponential backoff rather than
failing the entire workflow immediately.

This module provides ``invoke_with_retry()`` and
``ainvoke_with_retry()`` which wrap ``agent.invoke()`` / ``agent.ainvoke()``
with configurable retry logic. It classifies errors as transient (retryable)
or permanent (raise immediately).

Updated to support DA's ``context=`` kwarg for passing SpineContext
at invoke time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Default retry configuration ──

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds


# ── Token budget enforcement ────────────────────────────────────────────────
#
# Cumulative input+output tokens per work_id. Phases call ainvoke_with_retry
# repeatedly; once cumulative usage crosses the per-work-type budget, the next
# invocation raises MaxTokenBudgetExceeded so the phase subgraph wrapper can
# route to needs_review instead of running unbounded.
#
# Budget is only effective when the LLM returns usage_metadata. OpenRouter
# requires stream_usage=True (see _build_openrouter_model); local vLLM needs
# providers.llm[].stream_usage: true.

# Defaults derived from operational telemetry — quick workflows almost never
# exceed 200K when behaving correctly; spec workflows top out around 500K;
# critical_reviewed_task is allowed up to 1M before forcing human review.
_DEFAULT_BUDGETS_BY_WORK_TYPE: dict[str, int] = {
    "quick": 200_000,
    "spec": 500_000,
    "critical_reviewed_task": 1_000_000,
}

_cumulative_tokens: dict[str, int] = {}


class MaxTokenBudgetExceeded(Exception):
    """Raised when a work_id's cumulative LLM token usage exceeds its budget.

    Phase subgraph wrappers catch this and convert it into a needs_review
    outcome so the workflow stops instead of burning unbounded tokens.
    """

    def __init__(self, work_id: str, cumulative: int, budget: int, result: Any = None):
        self.work_id = work_id
        self.cumulative = cumulative
        self.budget = budget
        # The agent result that pushed cumulative over the budget. The call had
        # already SUCCEEDED and produced output before the post-hoc budget check
        # raised, so callers can salvage that output (e.g. a verify judge's
        # one-shot verdict) instead of discarding paid-for work as a "crash".
        self.result = result
        super().__init__(
            f"Token budget exceeded for work_id={work_id}: "
            f"{cumulative:,} / {budget:,} tokens"
        )


def _get_token_budget(work_type: str) -> int:
    """Return the token budget for ``work_type``.

    Honours the ``SPINE_TOKEN_BUDGET`` env var as a global override; falls
    back to ``_DEFAULT_BUDGETS_BY_WORK_TYPE``; defaults to 1M for unknown
    work types so an unfamiliar workflow doesn't trip on a missing entry.
    """
    override = os.environ.get("SPINE_TOKEN_BUDGET", "").strip()
    if override:
        try:
            return int(override)
        except ValueError:
            logger.warning(
                "SPINE_TOKEN_BUDGET=%r is not an integer — ignoring", override
            )
    return _DEFAULT_BUDGETS_BY_WORK_TYPE.get(work_type, 1_000_000)


def _tokens_from_result(result: Any) -> int:
    """Sum input+output tokens from any AIMessage.usage_metadata in ``result``.

    LangGraph agent results carry an ``"messages"`` list whose ``AIMessage``
    entries have a ``usage_metadata`` dict (``input_tokens`` /
    ``output_tokens`` / ``total_tokens``). Some providers attach usage only
    to the final chunk; others attach per-message. We sum across every
    AIMessage so per-message providers are accounted for, then deduplicate
    by message id when present.
    """
    if not isinstance(result, dict):
        return 0
    messages = result.get("messages") or []
    if not isinstance(messages, list):
        return 0
    total = 0
    seen_ids: set[str] = set()
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        msg_id = getattr(msg, "id", None)
        if isinstance(msg_id, str) and msg_id:
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)
        total += int(usage.get("input_tokens") or 0)
        total += int(usage.get("output_tokens") or 0)
    return total


def reset_token_budget(work_id: str) -> None:
    """Clear cumulative token tracking for a work_id.

    Called by the dispatcher at workflow launch so a re-run of the same
    work_id starts with a fresh budget rather than inheriting whatever was
    left over from the prior process's in-memory counter.
    """
    _cumulative_tokens.pop(work_id, None)


def _check_and_update_token_budget(
    *, work_id: str, work_type: str, result: Any
) -> int:
    """Add this call's tokens to the work_id counter and enforce the budget.

    Returns the new cumulative total. Raises ``MaxTokenBudgetExceeded``
    when the cumulative crosses the work-type budget. Silently no-ops
    (returns the prior cumulative) when ``work_id`` is missing or the LLM
    response lacks usage_metadata.
    """
    if not work_id:
        return 0
    tokens = _tokens_from_result(result)
    if tokens <= 0:
        return _cumulative_tokens.get(work_id, 0)
    new_total = _cumulative_tokens.get(work_id, 0) + tokens
    _cumulative_tokens[work_id] = new_total
    budget = _get_token_budget(work_type)
    if new_total > budget:
        raise MaxTokenBudgetExceeded(work_id, new_total, budget, result=result)
    return new_total


# ── Connection-failure circuit breaker ──────────────────────────────────────
#
# Per-call retry caps (max_retries) bound a SINGLE invocation, but the graph
# re-invokes nodes hundreds-to-thousands of times via per-slice Send fan-out and
# slice re-routing. When the LOCAL model server is genuinely down/restarting (it
# crashes/OOMs/disconnects far more than a cloud API), every one of those calls
# fails on connect, making no progress while burning minutes and thousands of
# dead requests (trace 019ece87: ~4000 "Could not connect to server"; 019ed360:
# 702). This process-wide counter trips ONLY on true endpoint-unreachable errors
# (never on 5xx/429/timeouts/mid-stream drops) after N consecutive failures with
# zero intervening successes, raising ServerUnreachable so the run aborts fast.
_DEFAULT_CONN_FAILURE_THRESHOLD = 8
_consecutive_conn_failures = 0


class ServerUnreachable(Exception):
    """Raised when the LLM endpoint is unreachable for too many consecutive calls.

    Phase subgraph wrappers catch this — like :class:`MaxTokenBudgetExceeded` —
    and convert it into a needs_review outcome so a down/restarting local server
    aborts the run fast instead of being hammered by Send fan-out re-dispatch.
    """

    def __init__(self, count: int, threshold: int):
        self.count = count
        self.threshold = threshold
        super().__init__(
            f"LLM endpoint unreachable: {count} consecutive connection "
            f"failures (threshold {threshold}). Is the local model server up?"
        )


def _conn_breaker_threshold() -> int:
    """Consecutive-connection-failure threshold (``SPINE_CONN_FAILURE_THRESHOLD``)."""
    override = os.environ.get("SPINE_CONN_FAILURE_THRESHOLD", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            logger.warning(
                "SPINE_CONN_FAILURE_THRESHOLD=%r is not an integer — ignoring",
                override,
            )
    return _DEFAULT_CONN_FAILURE_THRESHOLD


def reset_conn_breaker() -> None:
    """Reset the consecutive-connection-failure counter (called on any success)."""
    global _consecutive_conn_failures
    _consecutive_conn_failures = 0


def _is_connection_unreachable(exc: Exception) -> bool:
    """True only for "cannot reach the endpoint" errors — never 5xx/429/timeout.

    Deliberately narrow: a down server is a different failure class from an
    overloaded one. We trip the breaker for connection-refused / DNS / curl
    connect failures, but let transient 5xx/429/timeouts keep retrying normally.
    """
    name = type(exc).__name__
    if name in ("APIConnectionError", "ConnectionError", "ConnectError"):
        return True
    s = str(exc).lower()
    if "could not connect to server" in s or "connection refused" in s:
        return True
    if "failed to establish a new connection" in s or "name or service not known" in s:
        return True
    if "curl error" in s and "connect" in s:
        return True
    return False


def _note_conn_failure() -> int:
    """Increment and return the consecutive-connection-failure counter."""
    global _consecutive_conn_failures
    _consecutive_conn_failures += 1
    return _consecutive_conn_failures


def _trip_breaker_if_unreachable(exc: Exception, *, prefix: str, phase_label: str) -> None:
    """Update the breaker for ``exc`` and raise ServerUnreachable past threshold.

    Called from the except branch of every retry loop BEFORE the transient/
    permanent classification, so a down endpoint aborts the whole run rather
    than being masked by per-call retry exhaustion. No-op for non-connection
    errors (the counter only resets on a genuine success).
    """
    if not _is_connection_unreachable(exc):
        return
    count = _note_conn_failure()
    threshold = _conn_breaker_threshold()
    if count >= threshold:
        logger.error(
            "%s%s LLM endpoint unreachable for %d consecutive calls — "
            "tripping circuit breaker",
            prefix,
            phase_label,
            count,
        )
        raise ServerUnreachable(count, threshold)


def _is_transient_error(exc: Exception) -> bool:
    """Classify an exception as transient (retryable) or permanent.

    Transient errors include:
    - OpenRouter ``ResponseValidationError`` (error body instead of response)
    - HTTP 5xx server errors
    - Rate limit 429
    - Connection errors (transient network issues)
    - Timeouts

    Permanent errors include:
    - Authentication errors (401/403)
    - Invalid request errors (400)
    - Model not found (404)
    - Any non-API Python exception (logic bugs, etc.)

    Args:
        exc: The exception to classify.

    Returns:
        True if the error is transient and should be retried.
    """
    exc_type_name = type(exc).__name__
    exc_module = type(exc).__module__

    # ── OpenRouter ResponseValidationError ──
    # The openrouter SDK raises this when the response body is an error
    # object (e.g. {'error': {'message': '...', 'code': 520}}) instead
    # of a valid ChatCompletion. This is always transient — it means the
    # upstream provider returned an error.
    if "ResponseValidationError" in exc_type_name:
        return True

    # ── LangChain / OpenAI error classes ──
    # Check for error type names from langchain, openai, and httpx
    transient_names = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
        "GatewayTimeoutError",
        "TimeoutError",
        "ConnectionError",
    }
    if exc_type_name in transient_names:
        return True

    # ── Check HTTP status codes on exception attributes ──
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        # Some exceptions use .http_status or .code
        status_code = getattr(exc, "http_status", None) or getattr(exc, "code", None)

    if isinstance(status_code, int):
        # 5xx = server errors, 429 = rate limit → retryable
        if 500 <= status_code < 600 or status_code == 429:
            return True
        # 4xx (except 429) = client errors → permanent
        if 400 <= status_code < 500:
            return False

    # ── Check the error message for HTTP status codes ──
    # Match the code as a standalone token, not a substring — otherwise a
    # permanent error whose message embeds a number like "max 12500 tokens"
    # matches "500" and is wrongly retried. Require word boundaries so only a
    # real status code (e.g. "HTTP 500", "status 429") counts.
    exc_str = str(exc).lower()
    for code in (520, 502, 503, 504, 500, 429):
        if re.search(rf"(?<!\d){code}(?!\d)", exc_str):
            return True

    # ── OpenRouter upstream/provider failures ──
    # OpenRouter stream errors raise ValueError. When they indicate an upstream
    # provider error, these are typically transient provider overloads or timeouts.
    if "provider returned error" in exc_str or "openrouter api returned an error" in exc_str:
        return True

    # ── httpx transport errors ────────────────────────────────────────
    # Mid-stream drops (RemoteProtocolError, "peer closed connection") were
    # classified PERMANENT when the likely cause was a healthy-but-hung
    # stream — retrying just re-sent the same prompt. Since the
    # fallback_provider feature, the dominant cause is a CRASHED backend
    # (mini-sglang dies under concurrent load: batch 1, run d8bc459c FAILED
    # outright at specify on this classification while a healthy Lemonade
    # standby sat idle). Transient is now strictly better: in-call retries
    # may waste a couple of attempts against the dead endpoint, but the
    # phase-level retry that follows REBUILDS the model — the health check
    # sees the endpoint down and the standby takes over.
    if "httpx" in exc_module:
        exc_type_lower = exc_type_name.lower()
        if "remoteprotocolerror" in exc_type_lower:
            return True
        if "transport" in exc_type_lower:
            return True

    # Default: not transient — don't retry logic bugs or auth errors
    return False


def _find_response_headers(exc: Exception) -> Any:
    """Locate an HTTP response's headers mapping on ``exc`` or its wrapped cause.

    OpenAI/httpx errors carry the response on ``exc.response``; a wrapper
    exception may instead carry the real error as ``exc.args[0]``. Returns the
    headers mapping (dict / ``httpx.Headers``) or ``None`` when none is found.
    """
    candidates: list[Any] = [exc]
    args = getattr(exc, "args", ()) or ()
    if args:
        candidates.append(args[0])
    for cand in candidates:
        resp = getattr(cand, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
        if headers is not None:
            return headers
    return None


def _header_get(headers: Any, name: str) -> Any:
    """Case-insensitive header lookup over any ``.items()``-able mapping."""
    try:
        items = headers.items()
    except Exception:  # noqa: BLE001 — not a mapping / mock without items
        return None
    target = name.lower()
    for key, value in items:
        try:
            if str(key).lower() == target:
                return value
        except Exception:  # noqa: BLE001 — defensive against odd key types
            continue
    return None


def _parse_rate_limit_sleep(exc: Exception) -> float | None:
    """Return how long to wait from a rate-limit response's headers, or None.

    Honors, in priority order:
    1. ``Retry-After`` — integer seconds (only when strictly positive).
    2. ``X-RateLimit-Reset`` — a future epoch-millisecond timestamp; the wait is
       ``reset - now`` (None when already in the past).

    Malformed or absent values yield ``None`` so the caller falls back to
    exponential backoff. Honoring the server's own hint avoids hammering a
    rate-limited endpoint (and the wasted tokens / 429-storms that causes).
    """
    headers = _find_response_headers(exc)
    if not headers:
        return None

    retry_after = _header_get(headers, "retry-after")
    if retry_after is not None:
        try:
            secs = float(retry_after)
            if secs > 0:
                return secs
        except (TypeError, ValueError):
            pass

    reset = _header_get(headers, "x-ratelimit-reset")
    if reset is not None:
        try:
            delta = float(reset) / 1000.0 - time.time()
            if delta > 0:
                return delta
        except (TypeError, ValueError):
            pass

    return None


def _compute_sleep(
    exc: Exception, attempt: int, base_delay: float, max_delay: float
) -> float:
    """Seconds to sleep before the next retry.

    Prefers the server's rate-limit hint (``_parse_rate_limit_sleep``), capped at
    ``max_delay`` and used as-is (a small ``Retry-After`` is honored, not floored).
    Otherwise falls back to exponential backoff with ±10% jitter, capped at
    ``max_delay`` and floored at 0.5s.
    """
    header_sleep = _parse_rate_limit_sleep(exc)
    if header_sleep is not None:
        return min(header_sleep, max_delay)

    delay = min(base_delay * (2**attempt), max_delay)
    jitter = delay * 0.1
    sleep_time = min(delay + random.uniform(-jitter, jitter), max_delay)
    return max(sleep_time, 0.5)


def invoke_with_retry(
    agent: Any,
    input_: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    phase_name: str = "",
    work_id: str = "",
    work_type: str = "",
    context: Any = None,
) -> dict[str, Any]:
    """Invoke a Deep Agent with exponential backoff retry for transient errors.

    Calls ``agent.invoke(input_, context=context)`` and retries on transient
    API errors (5xx, 429, OpenRouter validation errors) with exponential
    backoff and jitter. Permanent errors (4xx, auth, logic bugs) raise
    immediately.

    .. deprecated::
        Prefer :func:`ainvoke_with_retry` for async phase nodes.
        This sync wrapper is kept for backward compatibility but should
        not be used in new async node functions.

    Args:
        agent: A compiled Deep Agent (result of ``create_deep_agent()``).
        input_: The input dict (typically ``{"messages": [...]}``).
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay between retries in seconds (default 2).
        max_delay: Maximum delay cap in seconds (default 60).
        phase_name: Phase name for logging context.
        work_id: Work ID for logging context.
        context: Optional SpineContext to pass via DA's context= kwarg.
            Propagates to subagents automatically.

    Returns:
        The agent's result dict.

    Raises:
        Exception: The last exception if all retries are exhausted, or
            immediately if the error is permanent (non-transient).
    """
    prefix = f"[{work_id}]" if work_id else ""
    phase_label = f" {phase_name}" if phase_name else ""

    # Build invoke kwargs — only add context if provided
    invoke_kwargs: dict[str, Any] = {}
    if context is not None:
        invoke_kwargs["context"] = context

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = agent.invoke(input_, **invoke_kwargs)
            # A genuine success means the endpoint is reachable — clear the
            # consecutive-connection-failure breaker.
            reset_conn_breaker()
            try:
                _check_and_update_token_budget(
                    work_id=work_id, work_type=work_type, result=result
                )
            except MaxTokenBudgetExceeded as budget_exc:
                logger.warning(
                    f"{prefix}{phase_label} {budget_exc} — aborting phase"
                )
                raise
            return result
        except MaxTokenBudgetExceeded:
            raise
        except ServerUnreachable:
            # Circuit breaker tripped — propagate without retrying so the
            # subgraph wrapper can route to needs_review.
            raise
        except Exception as exc:
            last_exc = exc

            # Trip the run-level breaker BEFORE the transient/permanent branch
            # so a persistently-down endpoint aborts fast instead of being
            # masked by per-call retry exhaustion across thousands of Sends.
            _trip_breaker_if_unreachable(exc, prefix=prefix, phase_label=phase_label)

            if not _is_transient_error(exc):
                logger.error(
                    f"{prefix}{phase_label} permanent error (attempt {attempt + 1}): {exc}"
                )
                raise

            if attempt >= max_retries:
                logger.error(
                    f"{prefix}{phase_label} exhausted {max_retries} retries for transient error: {exc}"
                )
                raise

            # Honor the server's rate-limit hint when present, else exponential
            # backoff with jitter.
            sleep_time = _compute_sleep(exc, attempt, base_delay, max_delay)

            logger.warning(
                f"{prefix}{phase_label} transient error (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {sleep_time:.1f}s: {type(exc).__name__}: {exc}"
            )
            time.sleep(sleep_time)

    # Should never reach here, but satisfy the type checker
    if last_exc:
        raise last_exc
    raise RuntimeError("invoke_with_retry: unexpected state")


async def ainvoke_with_retry(
    agent: Any,
    input_: dict[str, Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    phase_name: str = "",
    work_id: str = "",
    work_type: str = "",
    context: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Async invoke a Deep Agent with exponential backoff retry for transient errors.

    Calls ``agent.ainvoke(input_, context=context)`` and retries on transient
    API errors with exponential backoff and jitter. This is the async
    counterpart to :func:`invoke_with_retry` and should be used in all
    async phase node functions.

    Using ``ainvoke`` instead of ``invoke`` is critical for event loop
    correctness: when the outer LangGraph graph runs via ``graph.astream()``,
    sync node functions are dispatched to a thread pool.  Inside that thread,
    subagents that inherit the parent checkpointer encounter an
    ``asyncio.Lock`` bound to the original event loop, producing
    ``RuntimeError: is bound to a different event loop``.  Async nodes
    stay on the same event loop throughout, avoiding this class of bug
    entirely.

    Args:
        agent: A compiled Deep Agent (result of ``create_deep_agent()``).
        input_: The input dict (typically ``{"messages": [...]}``).
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay between retries in seconds (default 2).
        max_delay: Maximum delay cap in seconds (default 60).
        phase_name: Phase name for logging context.
        work_id: Work ID for logging context.
        context: Optional SpineContext to pass via DA's context= kwarg.
            Propagates to subagents automatically.

    Returns:
        The agent's result dict.

    Raises:
        Exception: The last exception if all retries are exhausted, or
            immediately if the error is permanent (non-transient).
    """
    prefix = f"[{work_id}]" if work_id else ""
    phase_label = f" {phase_name}" if phase_name else ""

    # Build invoke kwargs — only add context if provided
    invoke_kwargs: dict[str, Any] = {}
    if context is not None:
        invoke_kwargs["context"] = context
    if config is not None:
        invoke_kwargs["config"] = config

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await agent.ainvoke(input_, **invoke_kwargs)
            # A genuine success means the endpoint is reachable — clear the
            # consecutive-connection-failure breaker.
            reset_conn_breaker()
            # Snapshot the deduper cache onto the result so the calling node
            # can forward it into LangGraph state via the read_cache reducer.
            # ReadCacheMiddleware mutates context.read_cache in place during
            # the invocation; we take a shallow copy to decouple ownership.
            if context is not None and hasattr(context, "read_cache"):
                cache_snapshot = getattr(context, "read_cache", None) or {}
                if cache_snapshot and isinstance(result, dict):
                    result["read_cache"] = dict(cache_snapshot)
            # Token-budget enforcement: account for this call and raise
            # MaxTokenBudgetExceeded if the cumulative crosses the budget.
            # The subgraph wrapper catches this and routes to needs_review.
            try:
                _check_and_update_token_budget(
                    work_id=work_id, work_type=work_type, result=result
                )
            except MaxTokenBudgetExceeded as budget_exc:
                logger.warning(
                    f"{prefix}{phase_label} {budget_exc} — aborting phase"
                )
                raise
            return result
        except MaxTokenBudgetExceeded:
            # Already logged inside the inner try-block above. Propagate
            # without the "permanent error" log line and without retrying.
            raise
        except ServerUnreachable:
            # Circuit breaker tripped — propagate without retrying so the
            # subgraph wrapper can route to needs_review.
            raise
        except Exception as exc:
            last_exc = exc

            # Trip the run-level breaker BEFORE the transient/permanent branch
            # so a persistently-down endpoint aborts fast instead of being
            # masked by per-call retry exhaustion across thousands of Sends.
            _trip_breaker_if_unreachable(exc, prefix=prefix, phase_label=phase_label)

            if not _is_transient_error(exc):
                logger.error(
                    f"{prefix}{phase_label} permanent error (attempt {attempt + 1}): {exc}"
                )
                raise

            if attempt >= max_retries:
                logger.error(
                    f"{prefix}{phase_label} exhausted {max_retries} retries for transient error: {exc}"
                )
                raise

            # Honor the server's rate-limit hint when present, else exponential
            # backoff with jitter.
            sleep_time = _compute_sleep(exc, attempt, base_delay, max_delay)

            logger.warning(
                f"{prefix}{phase_label} transient error (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {sleep_time:.1f}s: {type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(sleep_time)

    # Should never reach here, but satisfy the type checker
    if last_exc:
        raise last_exc
    raise RuntimeError("ainvoke_with_retry: unexpected state")
