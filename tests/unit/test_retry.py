"""Tests for spine.agents.retry — transient error classification and retry logic."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.agents.retry import _is_transient_error, invoke_with_retry


# ── Transient error classification ──


class TestIsTransientError:
    """Test the _is_transient_error classifier."""

    def test_openrouter_response_validation_error_by_name(self):
        """OpenRouter ResponseValidationError is always transient.
        We can't import it directly (it's from the openrouter SDK), so test
        by creating a class with the same name.
        """
        # Simulate the actual exception class name
        cls = type("ResponseValidationError", (Exception,), {})
        exc = cls("Response validation failed: 6 validation errors")
        assert _is_transient_error(exc) is True

    def test_rate_limit_error(self):
        """Rate limit errors (429) are transient."""
        from openai import RateLimitError

        response = MagicMock()
        response.status_code = 429
        exc = RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body=None,
        )
        assert _is_transient_error(exc) is True

    def test_api_connection_error(self):
        """Connection errors are transient."""
        from openai import APIConnectionError

        exc = APIConnectionError(
            message="Connection refused",
            request=MagicMock(url="https://api.openai.com"),
        )
        assert _is_transient_error(exc) is True

    def test_api_timeout_error(self):
        """Timeout errors are transient."""
        from openai import APITimeoutError

        exc = APITimeoutError(request=MagicMock(url="https://api.openai.com"))
        assert _is_transient_error(exc) is True

    def test_internal_server_error(self):
        """500 Internal Server Error is transient."""
        from openai import InternalServerError

        response = MagicMock()
        response.status_code = 500
        exc = InternalServerError(
            message="Internal server error",
            response=response,
            body=None,
        )
        assert _is_transient_error(exc) is True

    def test_remote_protocol_error_is_transient(self):
        """Mid-stream drops retry now: the dominant cause is a crashed
        backend, and the phase retry that follows a transient exhaustion
        rebuilds the model through the fallback_provider health check
        (batch 1, run d8bc459c: the permanent classification FAILED a run
        at specify while a healthy standby sat idle)."""
        import httpx

        from spine.agents.retry import _is_transient_error

        exc = httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        )
        assert _is_transient_error(exc) is True

    def test_authentication_error_not_transient(self):
        """401 Authentication errors are permanent, not transient."""
        from openai import AuthenticationError

        response = MagicMock()
        response.status_code = 401
        exc = AuthenticationError(
            message="Invalid API key",
            response=response,
            body=None,
        )
        assert _is_transient_error(exc) is False

    def test_bad_request_not_transient(self):
        """400 Bad Request errors are permanent."""
        from openai import BadRequestError

        response = MagicMock()
        response.status_code = 400
        exc = BadRequestError(
            message="Invalid request",
            response=response,
            body=None,
        )
        assert _is_transient_error(exc) is False

    def test_python_logic_bug_not_transient(self):
        """Standard Python exceptions (logic bugs) are NOT transient."""
        assert _is_transient_error(ValueError("bad value")) is False
        assert _is_transient_error(TypeError("wrong type")) is False
        assert _is_transient_error(KeyError("missing key")) is False
        assert _is_transient_error(AttributeError("no attr")) is False

    def test_http_520_in_message(self):
        """520 status code in error message is detected as transient."""
        exc = Exception("Provider returned error with code 520")
        assert _is_transient_error(exc) is True

    def test_http_502_in_message(self):
        """502 status code in error message is detected as transient."""
        exc = Exception("Bad gateway 502")
        assert _is_transient_error(exc) is True

    def test_http_503_in_message(self):
        """503 status code in error message is detected as transient."""
        exc = Exception("Service unavailable (503)")
        assert _is_transient_error(exc) is True

    def test_status_code_attribute(self):
        """Exceptions with status_code attribute of 5xx are transient."""
        exc = RuntimeError("server error")
        exc.status_code = 503  # type: ignore[attr-defined]
        assert _is_transient_error(exc) is True

    def test_status_code_4xx_not_transient(self):
        """Exceptions with status_code attribute of 4xx are NOT transient (except 429)."""
        exc = RuntimeError("client error")
        exc.status_code = 404  # type: ignore[attr-defined]
        assert _is_transient_error(exc) is False

    def test_status_code_429_transient(self):
        """429 rate limit is transient even though it's 4xx."""
        exc = RuntimeError("rate limited")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_transient_error(exc) is True


# ── Retry invocation ──


class TestInvokeWithRetry:
    """Test the invoke_with_retry wrapper."""

    def test_succeeds_immediately(self):
        """No retries needed when invocation succeeds."""
        agent = MagicMock()
        expected = {"messages": [MagicMock(content="done")]}
        agent.invoke.return_value = expected

        result = invoke_with_retry(agent, {"messages": []}, max_retries=3)
        assert agent.invoke.call_count == 1
        assert result is expected

    def test_retries_on_transient_then_succeeds(self):
        """Retries on transient errors and succeeds on second attempt."""
        agent = MagicMock()

        # Create a transient error using a class name the classifier recognizes
        cls = type("RateLimitError", (Exception,), {})
        transient_exc = cls("Rate limit exceeded")

        expected = {"messages": [MagicMock(content="recovered")]}
        # First call fails, second succeeds
        agent.invoke.side_effect = [transient_exc, expected]

        result = invoke_with_retry(
            agent,
            {"messages": []},
            max_retries=3,
            base_delay=0.01,  # fast for tests
        )
        assert agent.invoke.call_count == 2
        assert result is expected

    def test_raises_after_exhausting_retries(self):
        """Raises the last transient error after all retries are exhausted."""
        agent = MagicMock()

        cls = type("InternalServerError", (Exception,), {})
        agent.invoke.side_effect = cls("still failing")

        with pytest.raises(Exception, match="still failing"):
            invoke_with_retry(
                agent,
                {"messages": []},
                max_retries=2,
                base_delay=0.01,
            )
        # Initial call + 2 retries = 3 total
        assert agent.invoke.call_count == 3

    def test_raises_immediately_on_permanent_error(self):
        """Permanent errors (4xx, logic bugs) are NOT retried."""
        agent = MagicMock()
        agent.invoke.side_effect = ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            invoke_with_retry(
                agent,
                {"messages": []},
                max_retries=3,
                base_delay=0.01,
            )
        # Should only be called once — no retries
        assert agent.invoke.call_count == 1

    def test_logs_context(self):
        """Phase name and work_id don't crash the function."""
        agent = MagicMock()
        agent.invoke.return_value = {"messages": []}

        result = invoke_with_retry(
            agent,
            {"messages": []},
            phase_name="implement",
            work_id="abc123",
            max_retries=0,
        )
        assert agent.invoke.call_count == 1


# ── Rate-limit header parsing ──


class TestParseRateLimitSleep:
    """Test the _parse_rate_limit_sleep header extractor."""

    @staticmethod
    def _make_exc_with_headers(headers: dict[str, str]) -> Exception:
        """Build a fake OpenAI-style exception carrying response headers."""
        from openai import RateLimitError

        response = MagicMock()
        response.headers = headers
        return RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body=None,
        )

    def test_retry_after_seconds(self):
        """Retry-After header with integer seconds is returned directly."""
        from spine.agents.retry import _parse_rate_limit_sleep

        exc = self._make_exc_with_headers({"Retry-After": "30"})
        assert _parse_rate_limit_sleep(exc) == 30.0

    def test_retry_after_lowercase(self):
        """retry-after (lowercase) is also matched — httpx normalises case."""
        from spine.agents.retry import _parse_rate_limit_sleep

        exc = self._make_exc_with_headers({"retry-after": "15"})
        assert _parse_rate_limit_sleep(exc) == 15.0

    def test_x_ratelimit_reset_epoch_millis(self):
        """X-RateLimit-Reset with a future epoch-ms timestamp."""
        from spine.agents.retry import _parse_rate_limit_sleep
        import time

        future_ms = str(int((time.time() + 45.0) * 1000))
        exc = self._make_exc_with_headers({"X-RateLimit-Reset": future_ms})
        result = _parse_rate_limit_sleep(exc)
        assert result is not None
        # Allow 1-second tolerance for test execution time
        assert 43.0 <= result <= 47.0

    def test_x_ratelimit_reset_lowercase(self):
        """x-ratelimit-reset (lowercase) is also matched."""
        from spine.agents.retry import _parse_rate_limit_sleep
        import time

        future_ms = str(int((time.time() + 10.0) * 1000))
        exc = self._make_exc_with_headers({"x-ratelimit-reset": future_ms})
        result = _parse_rate_limit_sleep(exc)
        assert result is not None
        assert 8.0 <= result <= 12.0

    def test_expired_reset_returns_none(self):
        """If X-RateLimit-Reset is in the past, return None (fall back)."""
        from spine.agents.retry import _parse_rate_limit_sleep
        import time

        past_ms = str(int((time.time() - 100.0) * 1000))
        exc = self._make_exc_with_headers({"X-RateLimit-Reset": past_ms})
        assert _parse_rate_limit_sleep(exc) is None

    def test_zero_retry_after_returns_none(self):
        """Retry-After of 0 is treated as absent (not positive)."""
        from spine.agents.retry import _parse_rate_limit_sleep

        exc = self._make_exc_with_headers({"Retry-After": "0"})
        assert _parse_rate_limit_sleep(exc) is None

    def test_retry_after_takes_priority_over_reset(self):
        """When both headers are present, Retry-After wins."""
        from spine.agents.retry import _parse_rate_limit_sleep
        import time

        future_ms = str(int((time.time() + 99.0) * 1000))
        exc = self._make_exc_with_headers({
            "Retry-After": "5",
            "X-RateLimit-Reset": future_ms,
        })
        assert _parse_rate_limit_sleep(exc) == 5.0

    def test_no_response_attribute(self):
        """Plain Exception without .response returns None."""
        from spine.agents.retry import _parse_rate_limit_sleep

        exc = RuntimeError("generic error")
        assert _parse_rate_limit_sleep(exc) is None

    def test_no_headers_on_response(self):
        """Exception with a response that has no headers returns None."""
        from spine.agents.retry import _parse_rate_limit_sleep

        response = MagicMock(spec=[])  # no headers attr
        exc = RuntimeError("error")
        exc.response = response  # type: ignore[attr-defined]
        assert _parse_rate_limit_sleep(exc) is None

    def test_empty_exception_args_fallback(self):
        """Exception with empty args doesn't crash the fallback path."""
        from spine.agents.retry import _parse_rate_limit_sleep

        exc = RuntimeError("error")
        exc.args = ()  # type: ignore[assignment]
        assert _parse_rate_limit_sleep(exc) is None

    def test_nested_exception_with_headers(self):
        """Wrapper exception whose .args[0] carries the real response."""
        from spine.agents.retry import _parse_rate_limit_sleep
        from openai import RateLimitError

        response = MagicMock()
        response.headers = {"Retry-After": "20"}
        inner = RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body=None,
        )
        wrapper = RuntimeError("wrapped")
        wrapper.args = (inner,)
        assert _parse_rate_limit_sleep(wrapper) == 20.0

    def test_full_openrouter_header_set(self):
        """Simulate the exact OpenRouter 429 headers from production."""
        from spine.agents.retry import _parse_rate_limit_sleep
        import time

        # Use a future timestamp so the delta is positive
        future_ms = str(int((time.time() + 60.0) * 1000))
        exc = self._make_exc_with_headers({
            "X-RateLimit-Limit": "20",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": future_ms,
        })
        result = _parse_rate_limit_sleep(exc)
        assert result is not None
        assert 58.0 <= result <= 62.0

    def test_malformed_header_values(self):
        """Non-numeric header values are silently ignored."""
        from spine.agents.retry import _parse_rate_limit_sleep

        exc = self._make_exc_with_headers({
            "Retry-After": "not-a-number",
            "X-RateLimit-Reset": "also-bad",
        })
        assert _parse_rate_limit_sleep(exc) is None


# ── Sleep computation ──


class TestComputeSleep:
    """Test the _compute_sleep backoff calculator."""

    def test_uses_retry_after_header(self):
        """When Retry-After is present, it overrides exponential backoff."""
        from spine.agents.retry import _compute_sleep

        exc = TestParseRateLimitSleep._make_exc_with_headers(
            {"Retry-After": "7"}
        )
        result = _compute_sleep(exc, attempt=5, base_delay=2.0, max_delay=60.0)
        assert result == 7.0

    def test_caps_at_max_delay(self):
        """Header sleep is capped at max_delay."""
        from spine.agents.retry import _compute_sleep

        exc = TestParseRateLimitSleep._make_exc_with_headers(
            {"Retry-After": "999"}
        )
        result = _compute_sleep(exc, attempt=0, base_delay=2.0, max_delay=60.0)
        assert result == 60.0

    def test_falls_back_to_exponential_backoff(self):
        """Without headers, falls back to exponential backoff."""
        from spine.agents.retry import _compute_sleep

        exc = RuntimeError("no headers")
        # attempt=2, base=2.0 → delay = min(2*4, 60) = 8.0
        # jitter ±10% → 7.2..8.8
        result = _compute_sleep(exc, attempt=2, base_delay=2.0, max_delay=60.0)
        assert 7.0 <= result <= 9.5

    def test_fallback_respects_max_delay_cap(self):
        """Exponential backoff is capped at max_delay."""
        from spine.agents.retry import _compute_sleep

        exc = RuntimeError("no headers")
        # attempt=10, base=2.0 → 2*1024 = 2048, capped to 60
        result = _compute_sleep(exc, attempt=10, base_delay=2.0, max_delay=60.0)
        assert result <= 60.0

    def test_fallback_minimum_half_second(self):
        """Sleep is never below 0.5s even with tiny backoff values."""
        from spine.agents.retry import _compute_sleep

        exc = RuntimeError("no headers")
        result = _compute_sleep(exc, attempt=0, base_delay=0.01, max_delay=60.0)
        assert result >= 0.5

    def test_header_sleep_capped_but_not_floored(self):
        """A small positive header value is used as-is (not raised to 0.5)."""
        from spine.agents.retry import _compute_sleep

        exc = TestParseRateLimitSleep._make_exc_with_headers(
            {"Retry-After": "1"}
        )
        result = _compute_sleep(exc, attempt=0, base_delay=2.0, max_delay=60.0)
        assert result == 1.0


# ── Retry invocation with rate-limit headers ──


class TestInvokeWithRetryRateAware:
    """Test that invoke_with_retry uses header-based sleep for 429s."""

    def test_429_with_retry_after_sleeps_correctly(self):
        """A 429 with Retry-After: 2 should sleep ~2s, not exponential."""
        import time
        from openai import RateLimitError
        from spine.agents.retry import invoke_with_retry

        agent = MagicMock()
        response = MagicMock()
        response.headers = {"Retry-After": "2"}
        rate_exc = RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body=None,
        )
        expected = {"messages": [MagicMock(content="ok")]}
        agent.invoke.side_effect = [rate_exc, expected]

        start = time.time()
        result = invoke_with_retry(
            agent,
            {"messages": []},
            max_retries=3,
            base_delay=2.0,
        )
        elapsed = time.time() - start

        assert result is expected
        assert agent.invoke.call_count == 2
        # Should sleep ~2s (the Retry-After value), not ~4s (exponential)
        assert 1.5 <= elapsed <= 4.0

    def test_429_without_headers_uses_exponential_backoff(self):
        """A 429 without headers falls back to exponential backoff."""
        import time
        from openai import RateLimitError
        from spine.agents.retry import invoke_with_retry

        agent = MagicMock()
        response = MagicMock()
        response.headers = {}  # no rate-limit headers
        rate_exc = RateLimitError(
            message="Rate limit exceeded",
            response=response,
            body=None,
        )
        expected = {"messages": [MagicMock(content="ok")]}
        agent.invoke.side_effect = [rate_exc, expected]

        start = time.time()
        result = invoke_with_retry(
            agent,
            {"messages": []},
            max_retries=3,
            base_delay=0.05,  # fast for tests
        )
        elapsed = time.time() - start

        assert result is expected
        assert agent.invoke.call_count == 2
        # With base_delay=0.05, attempt=0: delay = min(0.05, 60) = 0.05
        # jitter ±10% → 0.045..0.055, floored to 0.5
        assert 0.4 <= elapsed <= 1.0


# ── Helpers module ──


class TestHelpers:
    """Test spine.agents.helpers shared utilities."""

    def test_extract_response_from_message(self):
        from spine.agents.helpers import extract_response

        msg = MagicMock()
        msg.content = "Hello world"
        result = extract_response({"messages": [msg]})
        assert result == "Hello world"

    def test_extract_response_empty(self):
        from spine.agents.helpers import extract_response

        result = extract_response({"messages": []})
        assert result == ""

    def test_extract_response_no_messages_key(self):
        from spine.agents.helpers import extract_response

        result = extract_response({})
        assert result == ""
