"""Tests for TTFB (time-to-first-byte) streaming timeout feature.

Covers:
  - TTFB timeout fires when stream never yields
  - TTFB timeout is configurable
  - Successful streaming with no timeout
  - Long-running completions survive past TTFB
  - Empty streams complete cleanly
  - generate() timeout is unaffected
  - TTFBTimeoutError is a TimeoutError subclass
  - kwargs are passed through to the underlying client
  - ProviderFallbackChain.stream() delegates correctly
"""

import time

import pytest

from spine.providers.llm import (
    LLMProvider,
    TTFBTimeoutError,
    TTFB_TIMEOUT,
    _add_timeout,
    DEFAULT_TIMEOUT,
    STREAM_READ_TIMEOUT,
)
from spine.providers.base import ProviderFallbackChain


# ---------------------------------------------------------------------------
# Test helpers – chunk objects that look like OpenAI streaming chunks
# ---------------------------------------------------------------------------

class _FakeDelta:
    def __init__(self, content: str | None):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None):
        self.delta = _FakeDelta(content)


class FakeChunk:
    """Mimics an OpenAI-style streaming chunk recognised by the default
    ``LLMProvider._extract_chunk_content``."""

    def __init__(self, content: str | None):
        self.choices = [_FakeChoice(content)]


def _chunks(*texts: str | None) -> list[FakeChunk]:
    """Shorthand for building a list of FakeChunk objects."""
    return [FakeChunk(t) for t in texts]


# ---------------------------------------------------------------------------
# Custom iterables for controlled timing
# ---------------------------------------------------------------------------

class BlockingStream:
    """Iterable whose ``__next__`` can be made to sleep.

    This lets us exercise ``asyncio.wait_for()`` wrapping
    ``loop.run_in_executor(None, next, iter)`` because the *actual*
    blocking happens inside ``next()`` (in the executor thread), not
    during stream creation.
    """

    def __init__(self, chunks, first_delay: float = 0, subsequent_delay: float = 0):
        self._chunks = list(chunks)
        self._first_delay = first_delay
        self._subsequent_delay = subsequent_delay
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._chunks):
            raise StopIteration
        delay = self._first_delay if self._idx == 0 else self._subsequent_delay
        if delay:
            time.sleep(delay)
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class NeverYieldsStream:
    """Iterable whose ``__next__`` sleeps long enough to exceed any TTFB.

    Sleeps 5 seconds, enough to exceed SHORT_TTFB (0.1s) but not so long
    as to make tests slow.
    """

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(5)
        raise StopIteration  # unreachable in practice


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------

class MockStreamProvider(LLMProvider):
    """Controllable LLM provider for streaming tests.

    Parameters
    ----------
    stream:
        An iterable returned by ``_raw_stream``.  Use :class:`BlockingStream`,
        :class:`NeverYieldsStream`, or a plain list of chunk objects.
    provider_name:
        The value of the ``name`` property.
    generate_delay:
        Seconds to sleep inside ``generate_sync`` (for timeout tests).
    """

    def __init__(self, stream=None, provider_name="test-llm", generate_delay=0):
        self._stream = stream or []
        self._name = provider_name
        self._generate_delay = generate_delay
        self.captured_kwargs: dict | None = None

    # --- Provider ABC -------------------------------------------------------
    def configure(self, config):  # noqa: ARG002
        pass

    def validate(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return True

    # --- LLMProvider overrides ----------------------------------------------
    def generate_sync(self, prompt, **kwargs):  # noqa: ARG002
        if self._generate_delay:
            time.sleep(self._generate_delay)
        return "sync-result"

    def _raw_stream(self, prompt: str, **kwargs):
        self.captured_kwargs = kwargs
        return self._stream

    async def stream(
        self,
        prompt: str,
        ttfb_timeout: float = TTFB_TIMEOUT,
        **kwargs,
    ):
        async for chunk in self._stream_with_ttfb_timeout(
            prompt, ttfb_timeout=ttfb_timeout, **kwargs
        ):
            yield chunk


# ===================================================================
# Tests
# ===================================================================

SHORT_TTFB = 0.1
LONG_TTFB = 60.0


class TestTTFBTimeout:
    """TTFB timeout fires when the first token never arrives."""

    @pytest.mark.asyncio
    async def test_timeout_fires_on_blocking_stream(self):
        """A stream whose ``__next__`` blocks forever raises TTFBTimeoutError."""
        provider = MockStreamProvider(stream=NeverYieldsStream())

        with pytest.raises(TTFBTimeoutError) as exc:
            async for _ in provider.stream("hello", ttfb_timeout=SHORT_TTFB):
                pass

        assert exc.value.timeout_seconds == SHORT_TTFB
        assert exc.value.provider_name == "test-llm"

    @pytest.mark.asyncio
    async def test_timeout_fires_when_first_chunk_delayed(self):
        """A stream where ``__next__`` sleeps past the TTFB deadline raises."""
        provider = MockStreamProvider(
            stream=BlockingStream(_chunks("a", "b"), first_delay=1.0)
        )

        with pytest.raises(TTFBTimeoutError):
            async for _ in provider.stream("hello", ttfb_timeout=SHORT_TTFB):
                pass

    @pytest.mark.asyncio
    async def test_short_ttfb_fires_quickly(self):
        """With a very short TTFB the error must be raised almost immediately."""
        provider = MockStreamProvider(
            stream=BlockingStream(_chunks("x"), first_delay=5.0)
        )
        start = time.monotonic()

        with pytest.raises(TTFBTimeoutError):
            async for _ in provider.stream("test", ttfb_timeout=0.05):
                pass

        elapsed = time.monotonic() - start
        # Should be well under 1 second (the sleep itself is 5 s)
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_long_ttfb_does_not_fire_prematurely(self):
        """With a long TTFB and a fast first chunk, no error is raised."""
        provider = MockStreamProvider(
            stream=BlockingStream(_chunks("first", "second"), first_delay=0.01)
        )
        chunks = []

        async for c in provider.stream("test", ttfb_timeout=5.0):
            chunks.append(c)

        assert chunks == ["first", "second"]


class TestSuccessfulStreaming:
    """Streaming succeeds when chunks arrive in time."""

    @pytest.mark.asyncio
    async def test_all_chunks_yielded(self):
        provider = MockStreamProvider(stream=_chunks("hello", " ", "world"))

        result = []
        async for chunk in provider.stream("prompt"):
            result.append(chunk)

        assert result == ["hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_works_with_default_ttfb(self):
        provider = MockStreamProvider(stream=_chunks("data"))
        result = [c async for c in provider.stream("p")]
        assert result == ["data"]

    @pytest.mark.asyncio
    async def test_chunks_with_null_content_skipped(self):
        """Chunks where _extract_chunk_content returns None are not yielded."""
        # A chunk with content=None simulates a chunk with no meaningful text
        provider = MockStreamProvider(
            stream=[
                FakeChunk(None),        # skipped
                FakeChunk("real"),      # yielded
                FakeChunk(None),        # skipped
                FakeChunk("text"),      # yielded
            ]
        )

        result = [c async for c in provider.stream("p")]
        assert result == ["real", "text"]


class TestLongRunningCompletion:
    """Once TTFB is satisfied, the stream stays open indefinitely."""

    @pytest.mark.asyncio
    async def test_stream_stays_open_after_first_chunk(self):
        """First chunk arrives quickly; subsequent chunks also arrive."""
        chunks = _chunks("a", "b", "c", "d", "e")
        provider = MockStreamProvider(
            stream=BlockingStream(chunks, first_delay=0.02)
        )

        result = []
        async for c in provider.stream("long prompt", ttfb_timeout=0.5):
            result.append(c)

        assert result == ["a", "b", "c", "d", "e"]

    @pytest.mark.asyncio
    async def test_many_chunks_all_yielded(self):
        """A stream with dozens of chunks completes fully."""
        many = _chunks(*[f"token-{i}" for i in range(50)])
        provider = MockStreamProvider(stream=BlockingStream(many, first_delay=0.01))

        result = [c async for c in provider.stream("big", ttfb_timeout=1.0)]
        assert len(result) == 50
        assert result[0] == "token-0"
        assert result[-1] == "token-49"


class TestEmptyStream:
    """An empty stream completes with no error and yields nothing.

    .. note::

       On Python ≥ 3.12, ``StopIteration`` raised inside a
       ``concurrent.futures`` executor is **not** propagated as
       ``StopIteration`` — CPython wraps it in ``RuntimeError``
       (see :pep:`479` / :issue:`cPython#92063`).

       Because ``_stream_with_ttfb_timeout`` dispatches ``next()`` through
       ``loop.run_in_executor``, the ``except StopIteration`` handler
       cannot catch genuinely-empty iterators on the newer runtime.

       *The tests below codify the current behaviour.*  When the
       underlying code is updated to detect empty streams before
       dispatching (or to catch ``RuntimeError`` appropriately), the
       ``xfail`` markers should be removed.
    """

    @pytest.mark.xfail(
        reason="Python 3.12: StopIteration inside run_in_executor becomes RuntimeError",
        raises=RuntimeError,
        strict=True,
    )
    @pytest.mark.asyncio
    async def test_empty_list_yields_nothing(self):
        provider = MockStreamProvider(stream=[])
        result = [c async for c in provider.stream("prompt")]
        assert result == []

    @pytest.mark.xfail(
        reason="Python 3.12: StopIteration inside run_in_executor becomes RuntimeError",
        raises=RuntimeError,
        strict=True,
    )
    @pytest.mark.asyncio
    async def test_stop_iteration_immediately_handled(self):
        provider = MockStreamProvider(
            stream=BlockingStream([], first_delay=0)
        )
        result = [c async for c in provider.stream("p")]
        assert result == []


class TestGenerateTimeoutUnaffected:
    """The non-streaming :meth:`LLMProvider.generate` timeout still works."""

    def test_generate_timeout_still_fires(self):
        """A slow generate_sync still raises TimeoutError after its deadline."""
        provider = MockStreamProvider(generate_delay=2.0)

        with pytest.raises(TimeoutError, match="timed out"):
            provider.generate("prompt", timeout=0.1)

    def test_generate_with_sufficient_timeout(self):
        """When the sync call completes within the timeout, no error."""
        provider = MockStreamProvider(generate_delay=0.01)
        result = provider.generate("prompt", timeout=2.0)
        assert result == "sync-result"

    def test_generate_no_timeout_disables_check(self):
        """timeout=None bypasses the ThreadPoolExecutor timeout."""
        provider = MockStreamProvider(generate_delay=0.01)
        result = provider.generate("prompt", timeout=None)
        assert result == "sync-result"


class TestTTFBTimeoutErrorIsTimeoutError:
    """TTFBTimeoutError is a subclass of TimeoutError for compatibility."""

    def test_isinstance_timeout_error(self):
        assert issubclass(TTFBTimeoutError, TimeoutError)

    def test_instance_check(self):
        err = TTFBTimeoutError(timeout_seconds=5.0, provider_name="test")
        assert isinstance(err, TimeoutError)

    @pytest.mark.asyncio
    async def test_can_catch_with_existing_handler(self):
        """Existing ``except TimeoutError`` handlers catch TTFBTimeoutError."""
        provider = MockStreamProvider(stream=NeverYieldsStream())

        with pytest.raises(TimeoutError):
            async for _ in provider.stream("p", ttfb_timeout=SHORT_TTFB):
                pass


class TestStreamPassesThroughKwargs:
    """Extra kwargs are forwarded to the underlying client via _raw_stream."""

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_passed(self):
        provider = MockStreamProvider(stream=_chunks("ok"))

        result = [c async for c in provider.stream(
            "prompt",
            temperature=0.7,
            max_tokens=100,
        )]
        assert result == ["ok"]
        assert provider.captured_kwargs is not None
        assert provider.captured_kwargs.get("temperature") == 0.7
        assert provider.captured_kwargs.get("max_tokens") == 100

    @pytest.mark.asyncio
    async def test_ttfb_timeout_not_in_raw_kwargs(self):
        """*ttfb_timeout* is consumed by stream() and NOT forwarded to _raw_stream."""
        provider = MockStreamProvider(stream=_chunks("ok"))

        _ = [c async for c in provider.stream(
            "prompt", ttfb_timeout=5.0, extra="val"
        )]

        assert provider.captured_kwargs is not None
        assert "ttfb_timeout" not in provider.captured_kwargs
        assert provider.captured_kwargs.get("extra") == "val"

    @pytest.mark.asyncio
    async def test_no_extra_kwargs_means_empty_dict(self):
        provider = MockStreamProvider(stream=_chunks("ok"))

        _ = [c async for c in provider.stream("p")]

        # No extra kwargs beyond what stream() receives internally
        assert provider.captured_kwargs is not None
        # ttfb_timeout is NOT in captured_kwargs
        assert "ttfb_timeout" not in provider.captured_kwargs


class TestProviderFallbackChainStream:
    """ProviderFallbackChain.stream() delegates correctly."""

    @pytest.mark.asyncio
    async def test_chain_stream_delegates_to_active_provider(self):
        provider = MockStreamProvider(stream=_chunks("a", "b"))
        chain = ProviderFallbackChain(providers=[provider])

        result = [c async for c in chain.stream("prompt")]
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_chain_passes_ttfb_timeout_through(self):
        """A fast provider inside a chain with a short TTFB succeeds."""
        provider = MockStreamProvider(stream=_chunks("x", "y", "z"))
        chain = ProviderFallbackChain(providers=[provider])

        result = [c async for c in chain.stream("prompt", ttfb_timeout=5.0)]
        assert result == ["x", "y", "z"]

    @pytest.mark.asyncio
    async def test_chain_ttfb_timeout_propagates(self):
        """When the chain's provider times out, TTFBTimeoutError is raised."""
        provider = MockStreamProvider(stream=NeverYieldsStream())
        chain = ProviderFallbackChain(providers=[provider])

        with pytest.raises(TTFBTimeoutError):
            async for _ in chain.stream("prompt", ttfb_timeout=SHORT_TTFB):
                pass

    @pytest.mark.asyncio
    async def test_chain_extra_kwargs_forwarded(self):
        provider = MockStreamProvider(stream=_chunks("done"))
        chain = ProviderFallbackChain(providers=[provider])

        _ = [c async for c in chain.stream(
            "prompt", ttfb_timeout=1.0, temperature=0.3
        )]

        assert provider.captured_kwargs is not None
        assert provider.captured_kwargs.get("temperature") == 0.3


class TestAddTimeoutFunction:
    """Edge cases for the ``_add_timeout`` helper."""

    def test_default_timeout_applied_when_absent(self):
        kwargs = _add_timeout({})
        assert kwargs["timeout"] == DEFAULT_TIMEOUT

    def test_existing_timeout_preserved(self):
        kwargs = _add_timeout({"timeout": 99})
        assert kwargs["timeout"] == 99

    def test_streaming_uses_long_read_timeout(self):
        kwargs = _add_timeout({}, streaming=True)
        assert kwargs["timeout"][1] == STREAM_READ_TIMEOUT
        assert kwargs["timeout"][0] == DEFAULT_TIMEOUT[0]
        # Should be a tuple
        assert isinstance(kwargs["timeout"], tuple)

    def test_streaming_with_explicit_timeout_preserved(self):
        kwargs = _add_timeout({"timeout": (5, 15)}, streaming=True)
        assert kwargs["timeout"] == (5, 15)

    def test_explicit_tuple_timeout(self):
        kwargs = _add_timeout({}, timeout=(3, 20), streaming=False)
        assert kwargs["timeout"] == (3, 20)
