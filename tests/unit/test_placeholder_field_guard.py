"""Placeholder-field guard in ainvoke_structured_with_retry + rerank doc trim.

Work d8bc459c attempt 8: classification returned
``{"reasoning": "...", "category": "Backend/API", "confidence": 0.9}`` —
grammar-valid JSON whose free-text field is scaffold filler. The structured
retry layer now detects whole-field placeholder strings, retries once with
the offending fields named, and accepts-with-warning if the retry repeats it.

Same run surfaced the rerank root cause: (query+doc) pairs over the llama.cpp
physical batch (512) 500 inside an HTTP-200 envelope. Docs are now trimmed to
a token budget before posting.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from spine.agents.helpers import (
    _placeholder_string_fields,
    ainvoke_structured_with_retry,
)
from spine.agents.tools.reranker import _MAX_PAIR_TOKENS, _trim_to_tokens


class _Verdict(BaseModel):
    reasoning: str = Field(default="")
    category: str = Field(default="Generic")
    confidence: float = Field(default=0.5)


class _SequencedStructured:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self._payloads.pop(0)


class TestPlaceholderFieldDetection:
    def test_ellipsis_field_flagged(self):
        v = _Verdict(reasoning="...", category="Backend/API", confidence=0.9)
        assert _placeholder_string_fields(v) == ["reasoning"]

    def test_unicode_and_dashes_flagged(self):
        assert _placeholder_string_fields(_Verdict(reasoning="…")) == ["reasoning"]
        assert _placeholder_string_fields(_Verdict(reasoning="---")) == ["reasoning"]

    def test_empty_string_is_legal(self):
        assert _placeholder_string_fields(_Verdict(reasoning="")) == []

    def test_real_prose_with_ellipsis_passes(self):
        v = _Verdict(reasoning="Handles routes, models, ... and policies")
        assert _placeholder_string_fields(v) == []

    def test_non_pydantic_result_ignored(self):
        assert _placeholder_string_fields("just a string") == []
        assert _placeholder_string_fields(None) == []


def test_placeholder_retry_then_clean(caplog):
    model = _SequencedStructured(
        [
            _Verdict(reasoning="...", category="Backend/API", confidence=0.9),
            _Verdict(reasoning="CRUD routes and entities", category="Backend/API", confidence=0.9),
        ]
    )
    caplog.set_level(logging.WARNING, logger="spine.agents.helpers")
    result = asyncio.run(
        ainvoke_structured_with_retry(model, [], label="test-guard")
    )
    assert result.reasoning == "CRUD routes and entities"
    assert len(model.calls) == 2
    # Retry conversation names the offending field.
    assert any(
        "reasoning" in getattr(m, "content", "") for m in model.calls[1]
    )
    assert [r for r in caplog.records if "placeholder-only" in r.getMessage()]


def test_placeholder_persisting_is_accepted_with_warning(caplog):
    junk = _Verdict(reasoning="...", category="Backend/API", confidence=0.9)
    model = _SequencedStructured([junk, junk])
    caplog.set_level(logging.WARNING, logger="spine.agents.helpers")
    result = asyncio.run(
        ainvoke_structured_with_retry(model, [], label="test-guard")
    )
    assert result is junk  # accepted, not raised — shape is valid
    assert len(model.calls) == 2
    assert [r for r in caplog.records if "accepting response as-is" in r.getMessage()]


def test_clean_response_single_call():
    model = _SequencedStructured(
        [_Verdict(reasoning="real analysis", category="Database", confidence=0.8)]
    )
    result = asyncio.run(
        ainvoke_structured_with_retry(model, [], label="test-guard")
    )
    assert result.category == "Database"
    assert len(model.calls) == 1


class TestRerankDocTrim:
    def test_short_doc_untouched(self):
        assert _trim_to_tokens("short doc", 100) == "short doc"

    def test_long_doc_trimmed_under_budget(self):
        from spine.agents._tokens import count_tokens

        doc = "public function up(): void { Schema::create('farm_rain_gauges'); } " * 40
        trimmed = _trim_to_tokens(doc, 128)
        assert count_tokens(trimmed) <= 128
        assert trimmed  # never trimmed to nothing

    def test_pair_budget_leaves_query_room(self):
        # The invariant the 500 regression depends on: budget + typical query
        # stays well under the 512 physical batch.
        assert _MAX_PAIR_TOKENS <= 448
