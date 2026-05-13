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
