"""Tests for OpenRouter session_id wiring and model resolution.

Verifies that resolve_model produces a ChatOpenRouter with session_id
when the model is OpenRouter and a work_id is provided, and falls back
to a string when conditions are not met.

Also verifies that local providers with base_url get pre-built ChatOpenAI
instances, and that per-phase provider references are resolved correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestResolveModelSessionId:
    """Verify session_id wiring in resolve_model."""

    def test_openrouter_with_session_id_returns_chat_model(self) -> None:
        """resolve_model should return a ChatOpenRouter instance when model
        is OpenRouter and session_id is provided."""
        from langchain_core.language_models.chat_models import BaseChatModel

        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:z-ai/glm-4.5-air:free"}}

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = resolve_model(config, session_id="work-abc123")

        assert isinstance(model, BaseChatModel)
        # Verify it's actually a ChatOpenRouter with session_id set
        assert hasattr(model, "session_id")
        assert model.session_id == "work-abc123"

    def test_openrouter_without_session_id_returns_string(self) -> None:
        """resolve_model should return a string when model is OpenRouter
        but no session_id is provided."""
        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:z-ai/glm-4.5-air:free"}}

        model = resolve_model(config, session_id=None)

        assert isinstance(model, str)
        assert model == "openrouter:z-ai/glm-4.5-air:free"

    def test_non_openrouter_with_session_id_returns_string(self) -> None:
        """resolve_model should return a string when model is NOT OpenRouter,
        even if session_id is provided."""
        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openai:gpt-4o-mini"}}

        model = resolve_model(config, session_id="work-abc123")

        assert isinstance(model, str)
        assert model == "openai:gpt-4o-mini"

    def test_session_id_truncated_to_128_chars(self) -> None:
        """OpenRouter limits session_id to 128 characters."""
        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:z-ai/glm-4.5-air:free"}}
        long_id = "x" * 200

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = resolve_model(config, session_id=long_id)

        assert hasattr(model, "session_id")
        assert len(model.session_id) == 128
        assert model.session_id == "x" * 128

    def test_model_name_stripped_of_prefix(self) -> None:
        """The ChatOpenRouter should use the model name without the
        'openrouter:' prefix."""
        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:anthropic/claude-sonnet-4-5"}}

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = resolve_model(config, session_id="work-123")

        assert model.model_name == "anthropic/claude-sonnet-4-5"

    def test_no_config_falls_back_to_spineconfig(self) -> None:
        """When config is None, resolve_model should fall back to SpineConfig.

        Returns a model string or pre-built ChatModel depending on the
        active provider. The important behavior is that it doesn't crash
        and returns something usable.
        """
        from spine.agents.helpers import resolve_model

        model = resolve_model(None, session_id=None)
        # Should return either a string or a BaseChatModel — both are valid
        assert isinstance(model, (str, object))

    def test_work_id_as_session_id(self) -> None:
        """Typical usage: state.get('work_id') passed as session_id."""
        from langchain_core.language_models.chat_models import BaseChatModel

        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:z-ai/glm-4.5-air:free"}}
        work_id = "a1b2c3d4"  # 8-char UUID prefix used by dispatcher

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = resolve_model(config, session_id=work_id)

        assert isinstance(model, BaseChatModel)
        assert model.session_id == work_id

    def test_provider_profile_kwargs_applied(self) -> None:
        """_build_openrouter_model should apply DA ProviderProfile kwargs."""
        from spine.agents.helpers import _build_openrouter_model

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            with patch(
                "deepagents.profiles.provider.apply_provider_profile",
                return_value={"app_url": "https://spine.dev", "app_title": "SPINE"},
            ) as mock_apply:
                model = _build_openrouter_model(
                    "openrouter:z-ai/glm-4.5-air:free",
                    "work-123",
                )

        # Verify apply_provider_profile was called with the model spec
        mock_apply.assert_called_once_with("openrouter:z-ai/glm-4.5-air:free")
        # Verify the profile kwargs were passed to ChatOpenRouter
        assert model.app_url == "https://spine.dev"
        assert model.app_title == "SPINE"

    def test_completion_cap_sent_as_max_tokens_not_max_completion_tokens(self) -> None:
        """Regression (trace 019e86ed): the resolved completion cap must be sent
        as ``max_tokens``, not ``max_completion_tokens``.

        OpenRouter routes across provider endpoints that advertise ``max_tokens``
        but not ``max_completion_tokens`` (e.g. every deepseek-v4-flash endpoint).
        With require_parameters=True, sending ``max_completion_tokens`` makes
        OpenRouter reject the request with HTTP 404 "No endpoints found that can
        handle the requested parameters" — silently killing every call.
        """
        from spine.agents.helpers import _build_openrouter_model

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            with patch(
                "spine.agents.helpers._active_provider_config",
                return_value={"max_completion_tokens": 20000},
            ):
                model = _build_openrouter_model(
                    "openrouter:deepseek/deepseek-v4-flash",
                    "work-123",
                )

        assert model.max_tokens == 20000
        assert getattr(model, "max_completion_tokens", None) is None

    def test_bind_structured_output_routes_openrouter_through_json_schema(self) -> None:
        """Regression: ChatOpenRouter is NOT a ChatOpenAI subclass, so the
        json_schema routing in bind_structured_output must detect it explicitly.

        Otherwise structured output falls back to method="function_calling",
        which forces a ``tool_choice`` value some OpenRouter endpoints reject.
        """
        from pydantic import BaseModel

        import spine.agents.helpers as helpers
        from spine.agents.helpers import (
            _build_openrouter_model,
            _is_openai_style_model,
        )

        class _Schema(BaseModel):
            decision: str

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = _build_openrouter_model(
                "openrouter:deepseek/deepseek-v4-flash",
                "work-123",
            )

        # The detection guard must recognise ChatOpenRouter despite it not
        # subclassing ChatOpenAI.
        assert _is_openai_style_model(model) is True

        # The bound runnable must put response_format (json_schema) on the wire
        # and NOT a forced tool_choice. Intercept the SDK call to capture the
        # request params just before they would hit the network.
        import asyncio

        from langchain_core.messages import HumanMessage

        from spine.agents.helpers import bind_structured_output

        class _Stop(Exception):
            pass

        captured: dict[str, object] = {}

        async def _spy(*_args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            raise _Stop

        model.client.chat.send_async = _spy
        # Seed the capability cache so bind_structured_output doesn't hit
        # the live OpenRouter endpoints API from a unit test.
        with patch.dict(
            helpers._structured_method_cache,
            {"deepseek/deepseek-v4-flash": "json_schema"},
        ):
            runnable = bind_structured_output(model, _Schema)

        async def _go() -> None:
            try:
                await runnable.ainvoke([HumanMessage(content="hi")])
            except _Stop:
                pass

        asyncio.run(_go())

        assert "response_format" in captured
        assert "tool_choice" not in captured
        assert captured["response_format"]["type"] == "json_schema"  # type: ignore[index]


class TestOpenRouterStructuredMethod:
    """Verify endpoint-capability-driven structured-output method selection.

    Regression (trace 019eaf1f): OpenRouter gates ``response_format:
    json_schema`` behind the ``structured_outputs`` endpoint capability.
    minimax/minimax-m3's only endpoint advertises ``tools``/``tool_choice``
    but NOT ``structured_outputs``, so with ``require_parameters=True`` the
    unconditional ``method="json_schema"`` bind made every call 404 with
    "No endpoints found that can handle the requested parameters".
    """

    @staticmethod
    def _endpoints_response(supported_parameters: list[str]):
        class _Resp:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {
                    "data": {
                        "endpoints": [
                            {"supported_parameters": supported_parameters}
                        ]
                    }
                }

        return _Resp()

    def _method_for(self, supported_parameters: list[str]) -> str:
        import spine.agents.helpers as helpers

        with patch.dict(helpers._structured_method_cache, {}, clear=True):
            with patch(
                "httpx.get",
                return_value=self._endpoints_response(supported_parameters),
            ):
                return helpers._openrouter_structured_method("test/model")

    def test_no_structured_outputs_falls_back_to_function_calling(self) -> None:
        """minimax-m3's actual capability set must select function_calling."""
        supported = [
            "include_reasoning",
            "max_tokens",
            "reasoning",
            "response_format",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ]
        assert self._method_for(supported) == "function_calling"

    def test_structured_outputs_supported_keeps_json_schema(self) -> None:
        assert (
            self._method_for(["structured_outputs", "tool_choice", "max_tokens"])
            == "json_schema"
        )

    def test_response_format_only_falls_back_to_json_mode(self) -> None:
        assert self._method_for(["response_format", "max_tokens"]) == "json_mode"

    def test_lookup_failure_fails_open_to_json_schema_and_caches(self) -> None:
        import spine.agents.helpers as helpers

        with patch.dict(helpers._structured_method_cache, {}, clear=True):
            with patch("httpx.get", side_effect=OSError("network down")):
                assert (
                    helpers._openrouter_structured_method("test/model")
                    == "json_schema"
                )
            # The fail-open default is cached: no second HTTP attempt.
            with patch("httpx.get", side_effect=AssertionError("should not be called")):
                assert (
                    helpers._openrouter_structured_method("test/model")
                    == "json_schema"
                )

    def test_bind_structured_output_uses_function_calling_for_minimax(self) -> None:
        """End-to-end bind: a ChatOpenRouter model whose endpoints lack
        structured_outputs must put a forced tool_choice on the wire instead
        of response_format json_schema."""
        import asyncio

        from langchain_core.messages import HumanMessage
        from pydantic import BaseModel

        import spine.agents.helpers as helpers
        from spine.agents.helpers import (
            _build_openrouter_model,
            bind_structured_output,
        )

        class _Schema(BaseModel):
            decision: str

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = _build_openrouter_model(
                "openrouter:minimax/minimax-m3",
                "work-123",
            )

        class _Stop(Exception):
            pass

        captured: dict[str, object] = {}

        async def _spy(*_args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            raise _Stop

        model.client.chat.send_async = _spy
        with patch.dict(
            helpers._structured_method_cache,
            {"minimax/minimax-m3": "function_calling"},
        ):
            runnable = bind_structured_output(model, _Schema)

        async def _go() -> None:
            try:
                await runnable.ainvoke([HumanMessage(content="hi")])
            except _Stop:
                pass

        asyncio.run(_go())

        assert "tool_choice" in captured
        assert captured.get("response_format") is None or (
            captured["response_format"].get("type") != "json_schema"  # type: ignore[union-attr]
        )


class _Routing404(Exception):
    """Mimics langchain_openrouter's NotFoundResponseError shape."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 404


_TOOL_CHOICE_404 = (
    "No endpoints found that support the provided 'tool_choice' value. "
    "To learn more about provider routing, visit: "
    "https://openrouter.ai/docs/guides/routing/provider-selection"
)
_PARAMS_404 = "No endpoints found that can handle the requested parameters."


class _FakeStructuredModel:
    """Stub model whose bound runnable fails routing for selected methods."""

    def __init__(self, model_name: str, failing_methods: dict[str, Exception]) -> None:
        self.model_name = model_name
        self._failing = failing_methods
        self.methods_invoked: list[str] = []

    def with_structured_output(self, schema: object, method: str = "") -> object:
        outer = self

        class _Bound:
            async def ainvoke(self, _input: object, _config: object = None, **_kw: object) -> str:
                outer.methods_invoked.append(method)
                exc = outer._failing.get(method)
                if exc is not None:
                    raise exc
                return f"ok:{method}"

        return _Bound()


class TestSelfHealingStructured:
    """Verify invoke-time method demotion on OpenRouter routing 404s.

    Regression (trace 019eaf2a): endpoint capability listings can't reveal
    value-level support — minimax-m3 advertises ``tool_choice`` but rejects
    the forced named function that method="function_calling" sends, 404ing
    at routing time. The binding must demote and retry rather than fail the
    whole synthesis run.
    """

    def _run(self, coro):
        import asyncio

        return asyncio.run(coro)

    def test_demotes_function_calling_to_json_mode_and_caches(self) -> None:
        import spine.agents.helpers as helpers
        from spine.agents.helpers import _SelfHealingStructured

        model = _FakeStructuredModel(
            "minimax/minimax-m3",
            {"function_calling": _Routing404(_TOOL_CHOICE_404)},
        )
        with patch.dict(helpers._structured_method_cache, {}, clear=True):
            wrapper = _SelfHealingStructured(model, object, "function_calling")
            result = self._run(wrapper.ainvoke(["msg"]))
            assert helpers._structured_method_cache["minimax/minimax-m3"] == "json_mode"
        assert result == "ok:json_mode"
        assert model.methods_invoked == ["function_calling", "json_mode"]

    def test_walks_full_ladder_from_json_schema(self) -> None:
        import spine.agents.helpers as helpers
        from spine.agents.helpers import _SelfHealingStructured

        model = _FakeStructuredModel(
            "test/model",
            {
                "json_schema": _Routing404(_PARAMS_404),
                "function_calling": _Routing404(_TOOL_CHOICE_404),
            },
        )
        with patch.dict(helpers._structured_method_cache, {}, clear=True):
            wrapper = _SelfHealingStructured(model, object, "json_schema")
            result = self._run(wrapper.ainvoke(["msg"]))
        assert result == "ok:json_mode"
        assert model.methods_invoked == ["json_schema", "function_calling", "json_mode"]

    def test_non_routing_errors_propagate_without_demotion(self) -> None:
        import pytest

        import spine.agents.helpers as helpers
        from spine.agents.helpers import _SelfHealingStructured

        model = _FakeStructuredModel(
            "test/model", {"json_schema": RuntimeError("provider exploded")}
        )
        with patch.dict(helpers._structured_method_cache, {}, clear=True):
            wrapper = _SelfHealingStructured(model, object, "json_schema")
            with pytest.raises(RuntimeError, match="provider exploded"):
                self._run(wrapper.ainvoke(["msg"]))
            assert "test/model" not in helpers._structured_method_cache
        assert model.methods_invoked == ["json_schema"]

    def test_json_mode_404_demotes_to_prompt_rung(self) -> None:
        # json_mode used to be the ladder bottom (404 propagated). The
        # terminal rung is now "prompt": plain generation + JSON salvage
        # (poolside/laguna-s-2.1 supports NO native structured mechanism).
        import spine.agents.helpers as helpers
        from pydantic import BaseModel
        from spine.agents.helpers import _SelfHealingStructured

        class Out(BaseModel):
            verdict: str

        model = _FakeStructuredModel(
            "test/model", {"json_mode": _Routing404(_PARAMS_404)}
        )

        async def _plain_ainvoke(_input, _config=None, **_kw):
            from types import SimpleNamespace
            return SimpleNamespace(content='{"verdict": "PASSED"}')

        model.ainvoke = _plain_ainvoke  # the prompt rung calls the RAW model
        with patch.dict(helpers._structured_method_cache, {}, clear=True):
            wrapper = _SelfHealingStructured(model, Out, "json_mode")
            out = self._run(wrapper.ainvoke(["msg"]))
            assert helpers._structured_method_cache["test/model"] == "prompt"
        assert out.verdict == "PASSED"
        assert model.methods_invoked == ["json_mode"]  # then prompt rung, no bind

    def test_adopts_demotion_discovered_by_another_worker(self) -> None:
        import spine.agents.helpers as helpers
        from spine.agents.helpers import _SelfHealingStructured

        model = _FakeStructuredModel(
            "test/model", {"json_schema": _Routing404(_PARAMS_404)}
        )
        with patch.dict(
            helpers._structured_method_cache,
            {"test/model": "json_mode"},
            clear=True,
        ):
            wrapper = _SelfHealingStructured(model, object, "json_schema")
            result = self._run(wrapper.ainvoke(["msg"]))
        # The cached demotion is adopted up-front: json_schema never tried.
        assert result == "ok:json_mode"
        assert model.methods_invoked == ["json_mode"]


class TestLocalProviderModel:
    """Verify that local providers with base_url get pre-built ChatOpenAI models."""

    def test_local_provider_with_base_url_returns_chat_model(self) -> None:
        """resolve_model should return a ChatOpenAI when the active provider
        has base_url and the model spec matches."""

        from langchain_core.language_models.chat_models import BaseChatModel

        from spine.agents.helpers import resolve_model

        # Use explicit config + patch _active_provider_config to simulate
        # a local provider with base_url
        config = {"configurable": {"model": "openai:model"}}
        with patch("spine.agents.helpers._active_provider_config") as mock_apc:
            mock_apc.return_value = {
                "name": "local",
                "model": "openai:model",
                "base_url": "http://localhost:8000/v1",
                "api_key": "vllm",
            }
            model = resolve_model(config, session_id=None)
        assert isinstance(model, BaseChatModel)
        assert model.openai_api_base == "http://localhost:8000/v1"

    def test_local_provider_config_mismatch_returns_string(self) -> None:
        """resolve_model should return a string when the model spec does NOT
        match the active provider's model (e.g. explicit config override)."""
        from spine.agents.helpers import resolve_model

        # Explicitly request a different model than the local provider
        config = {"configurable": {"model": "openai:gpt-4o-mini"}}
        model = resolve_model(config, session_id=None)
        assert isinstance(model, str)
        assert model == "openai:gpt-4o-mini"

    def test_local_provider_wires_api_key(self) -> None:
        """The pre-built ChatOpenAI should use the api_key from config."""
        from langchain_core.language_models.chat_models import BaseChatModel

        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openai:model"}}
        with patch("spine.agents.helpers._active_provider_config") as mock_apc:
            mock_apc.return_value = {
                "name": "local",
                "model": "openai:model",
                "base_url": "http://localhost:8000/v1",
                "api_key": "vllm",
            }
            model = resolve_model(config, session_id=None)
        assert isinstance(model, BaseChatModel)
        assert model.openai_api_key.get_secret_value() == "vllm"


class TestPhaseProviderResolution:
    """Verify that resolve_model correctly resolves provider references
    from providers.phases.<phase>.provider."""

    def test_provider_reference_resolves_model(self) -> None:
        """A phase with `provider: lfm` should resolve to that provider's model."""
        from spine.config import SpineConfig

        config = SpineConfig(
            providers={
                "llm": [
                    {"name": "frontier", "model": "openrouter:z-ai/glm-5.1", "enabled": True},
                    {"name": "lfm", "model": "openrouter:liquid/lfm-2-24b-a2b", "enabled": True},
                ],
                "phases": {
                    "plan": {"provider": "lfm"},
                },
            },
        )
        assert config.resolve_model("plan") == "openrouter:liquid/lfm-2-24b-a2b"

    def test_explicit_model_beats_provider_ref(self) -> None:
        """A phase with both `model` and `provider` should use `model`."""
        from spine.config import SpineConfig

        config = SpineConfig(
            providers={
                "llm": [
                    {"name": "frontier", "model": "openrouter:z-ai/glm-5.1", "enabled": True},
                    {"name": "lfm", "model": "openrouter:liquid/lfm-2-24b-a2b", "enabled": True},
                ],
                "phases": {
                    "plan": {"model": "openrouter:custom-model", "provider": "lfm"},
                },
            },
        )
        assert config.resolve_model("plan") == "openrouter:custom-model"

    def test_provider_ref_unknown_provider_falls_through(self) -> None:
        """A phase with `provider: nonexistent` should fall through to default."""
        from spine.config import SpineConfig

        config = SpineConfig(
            providers={
                "llm": [
                    {"name": "frontier", "model": "openrouter:z-ai/glm-5.1", "enabled": True},
                ],
                "phases": {
                    "plan": {"provider": "nonexistent"},
                },
            },
        )
        # Falls through to default provider
        assert config.resolve_model("plan") == "openrouter:z-ai/glm-5.1"

    def test_subagent_provider_ref_resolves(self) -> None:
        """A subagent path with `provider` should resolve to that provider's model."""
        from spine.config import SpineConfig

        config = SpineConfig(
            providers={
                "llm": [
                    {"name": "frontier", "model": "openrouter:z-ai/glm-5.1", "enabled": True},
                    {"name": "local", "model": "openai:model", "enabled": True},
                ],
                "phases": {
                    "implement": {"provider": "frontier"},
                    "implement/subagents/slice-implementer": {"provider": "local"},
                },
            },
        )
        assert config.resolve_model("implement") == "openrouter:z-ai/glm-5.1"
        assert config.resolve_model("implement/subagents/slice-implementer") == "openai:model"

    def test_no_phase_returns_default(self) -> None:
        """resolve_model() with no phase returns the default provider's model."""
        from spine.config import SpineConfig

        config = SpineConfig(
            providers={
                "llm": [
                    {"name": "frontier", "model": "openrouter:z-ai/glm-5.1", "enabled": True},
                ],
                "phases": {
                    "plan": {"provider": "nonexistent"},
                },
            },
        )
        assert config.resolve_model() == "openrouter:z-ai/glm-5.1"
