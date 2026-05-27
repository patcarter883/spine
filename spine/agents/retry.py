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
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Default retry configuration ──

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds


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
    exc_str = str(exc).lower()
    for code in (520, 502, 503, 504, 500, 429):
        if str(code) in exc_str:
            return True

    # ── OpenRouter upstream/provider failures ──
    # OpenRouter stream errors raise ValueError. When they indicate an upstream
    # provider error, these are typically transient provider overloads or timeouts.
    if "provider returned error" in exc_str or "openrouter api returned an error" in exc_str:
        return True

    # ── httpx transport errors ────────────────────────────────────────
    # Distinguish between transient errors (connection refused, timeout)
    # and mid-stream drops (server started responding then disconnected).
    #
    # RemoteProtocolError with "peer closed connection" means the server
    # sent a partial response then dropped — retrying re-sends the same
    # prompt for zero or marginal benefit and wastes tokens.  Mark as
    # permanent so we fail fast instead of burning retries on a hung
    # stream.
    if "httpx" in exc_module:
        exc_type_lower = exc_type_name.lower()
        if "remoteprotocolerror" in exc_type_lower:
            return False
        if "transport" in exc_type_lower:
            return True

    # Default: not transient — don't retry logic bugs or auth errors
    return False


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
            return agent.invoke(input_, **invoke_kwargs)
        except Exception as exc:
            last_exc = exc

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

            # Exponential backoff with jitter
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = delay * 0.1  # ±10% jitter
            import random

            sleep_time = delay + random.uniform(-jitter, jitter)
            sleep_time = max(sleep_time, 0.5)  # floor at 0.5s

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

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await agent.ainvoke(input_, **invoke_kwargs)
            # Snapshot the deduper cache onto the result so the calling node
            # can forward it into LangGraph state via the read_cache reducer.
            # ReadCacheMiddleware mutates context.read_cache in place during
            # the invocation; we take a shallow copy to decouple ownership.
            if context is not None and hasattr(context, "read_cache"):
                cache_snapshot = getattr(context, "read_cache", None) or {}
                if cache_snapshot and isinstance(result, dict):
                    result["read_cache"] = dict(cache_snapshot)
            return result
        except Exception as exc:
            last_exc = exc

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

            # Exponential backoff with jitter
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = delay * 0.1  # ±10% jitter
            import random

            sleep_time = delay + random.uniform(-jitter, jitter)
            sleep_time = max(sleep_time, 0.5)  # floor at 0.5s

            logger.warning(
                f"{prefix}{phase_label} transient error (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {sleep_time:.1f}s: {type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(sleep_time)

    # Should never reach here, but satisfy the type checker
    if last_exc:
        raise last_exc
    raise RuntimeError("ainvoke_with_retry: unexpected state")
