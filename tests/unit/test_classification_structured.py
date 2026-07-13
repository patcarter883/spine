"""classify_task binds a schema — no prose-JSON scraping (trace 019f5a37)."""

from __future__ import annotations

import pytest

from spine.agents import classification
from spine.agents.classification import TaskClassificationResult, classify_task


class _StructuredModel:
    def __init__(self, result):
        self._result = result
        self.invoked_with = None

    async def ainvoke(self, messages):
        self.invoked_with = messages
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


@pytest.mark.asyncio
async def test_returns_schema_bound_result(monkeypatch):
    expected = TaskClassificationResult(
        reasoning="routes and controllers", category="Backend/API", confidence=0.9
    )
    bound = _StructuredModel(expected)
    seen = {}

    def fake_bind(model, schema):
        seen["schema"] = schema
        return bound

    monkeypatch.setattr(classification, "resolve_chat_model", lambda config, phase: object())
    monkeypatch.setattr(classification, "bind_structured_output", fake_bind)

    result = await classify_task("Add CRUD routes", config={})

    assert result is expected
    assert seen["schema"] is TaskClassificationResult
    assert bound.invoked_with is not None


@pytest.mark.asyncio
async def test_falls_back_to_generic_on_error(monkeypatch):
    monkeypatch.setattr(classification, "resolve_chat_model", lambda config, phase: object())
    monkeypatch.setattr(
        classification,
        "bind_structured_output",
        lambda model, schema: _StructuredModel(RuntimeError("serve down")),
    )

    result = await classify_task("Add CRUD routes", config={})

    assert result.category == "Generic"
    assert result.confidence == 0.5
    assert "serve down" in result.reasoning


@pytest.mark.asyncio
async def test_non_schema_result_falls_back(monkeypatch):
    """A None/str result (e.g. parser returned nothing) must not propagate."""
    monkeypatch.setattr(classification, "resolve_chat_model", lambda config, phase: object())
    monkeypatch.setattr(
        classification,
        "bind_structured_output",
        lambda model, schema: _StructuredModel(None),
    )

    result = await classify_task("Add CRUD routes", config={})

    assert result.category == "Generic"
