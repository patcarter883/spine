"""Per-provider ``structured_method`` pin for structured output.

Probe 13 (2026-07-09, agripath clone): the Qwen3.6-35B serve applies the
``response_format`` json_schema grammar from the first token, which
suppresses the model's thinking phase — structured replies come back
truncated mid-string (``finish_reason=stop``) or with chain-of-thought
leaked into the JSON fields. The same serve handles a forced named tool
call correctly, so the provider pins ``structured_method:
function_calling`` and ``bind_structured_output`` must honour it.
"""

from unittest.mock import patch

import pytest
from pydantic import BaseModel


class _Schema(BaseModel):
    decision: str


def _build(provider_cfg: dict):
    from spine.agents.helpers import _build_local_model

    cfg = {"base_url": "http://localhost:1919/v1", "api_key": "vllm"}
    cfg.update(provider_cfg)
    return _build_local_model("openai:test-org/Test-Model-MXFP4", cfg)


@pytest.fixture(autouse=True)
def _clean_method_cache():
    import spine.agents.helpers as helpers

    saved = dict(helpers._structured_method_cache)
    helpers._structured_method_cache.clear()
    yield
    helpers._structured_method_cache.clear()
    helpers._structured_method_cache.update(saved)


class TestStructuredMethodOverride:
    def test_build_seeds_method_cache(self) -> None:
        import spine.agents.helpers as helpers

        _build({"structured_method": "function_calling"})

        assert (
            helpers._structured_method_cache.get("test-org/test-model-mxfp4")
            == "function_calling"
        )

    def test_bind_uses_pinned_method(self) -> None:
        from spine.agents.helpers import bind_structured_output

        model = _build({"structured_method": "function_calling"})

        with patch.object(
            type(model), "with_structured_output", return_value="bound"
        ) as wso:
            result = bind_structured_output(model, _Schema)

        assert result == "bound"
        wso.assert_called_once_with(_Schema, method="function_calling")

    def test_bind_defaults_to_json_schema_without_pin(self) -> None:
        from spine.agents.helpers import bind_structured_output

        model = _build({})

        with patch.object(
            type(model), "with_structured_output", return_value="bound"
        ) as wso:
            bind_structured_output(model, _Schema)

        wso.assert_called_once_with(_Schema, method="json_schema")

    def test_invalid_method_rejected_at_build(self) -> None:
        with pytest.raises(ValueError, match="structured_method"):
            _build({"structured_method": "grammar"})

    def test_pin_scoped_to_its_own_model(self) -> None:
        """A pin for one provider must not leak onto other models."""
        from spine.agents.helpers import _build_local_model, bind_structured_output

        _build({"structured_method": "function_calling"})
        other = _build_local_model(
            "openai:other/Other-Model",
            {"base_url": "http://localhost:8010/v1", "api_key": "vllm"},
        )

        with patch.object(
            type(other), "with_structured_output", return_value="bound"
        ) as wso:
            bind_structured_output(other, _Schema)

        wso.assert_called_once_with(_Schema, method="json_schema")
