"""Unit tests for ``spine.agents.decomposer.run_decomposer``."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.agents.decomposer import DecompositionResult, FeatureSliceSchema, run_decomposer


def _decomposition(*ids: str) -> DecompositionResult:
    return DecompositionResult(
        slices=[
            FeatureSliceSchema(
                id=i,
                title=f"slice {i}",
                description=f"description for {i}",
                target_files=[f"src/{i}.py"],
                acceptance_criteria=[f"{i} works"],
            )
            for i in ids
        ]
    )


def _make_structured_mock(result: DecompositionResult) -> MagicMock:
    """Build a mocked chat model whose ``with_structured_output(...)`` returns
    something whose ``.ainvoke(...)`` resolves to ``result``."""
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=result)
    model = MagicMock()
    model.with_structured_output = MagicMock(return_value=structured)
    return model


@pytest.mark.asyncio
async def test_plan_mode_returns_slice_dicts():
    model = _make_structured_mock(_decomposition("a", "b"))
    with patch("spine.agents.decomposer.resolve_model", return_value=model):
        slices = await run_decomposer(
            mode="PLAN",
            spec_markdown="# Spec\n\nDo the thing.",
        )
    assert [s["id"] for s in slices] == ["a", "b"]
    assert all("acceptance_criteria" in s for s in slices)


@pytest.mark.asyncio
async def test_plan_mode_rejects_empty_spec():
    with pytest.raises(ValueError, match="spec_markdown"):
        await run_decomposer(mode="PLAN", spec_markdown="")


@pytest.mark.asyncio
async def test_fallback_mode_assigns_micro_suffix_ids():
    # Decomposer returns ids that DON'T follow the convention; the wrapper
    # must rewrite them to '<parent>-micro-N'.
    model = _make_structured_mock(_decomposition("alpha", "beta"))
    with patch("spine.agents.decomposer.resolve_model", return_value=model):
        slices = await run_decomposer(
            mode="FALLBACK",
            failed_slice={"id": "parent-slice", "title": "Parent", "target_files": ["x.py"]},
            error_traceback="Traceback (most recent call last): ...",
        )
    assert [s["id"] for s in slices] == ["parent-slice-micro-1", "parent-slice-micro-2"]


@pytest.mark.asyncio
async def test_fallback_mode_preserves_correct_suffix():
    model = _make_structured_mock(
        _decomposition("parent-slice-micro-1", "parent-slice-micro-2")
    )
    with patch("spine.agents.decomposer.resolve_model", return_value=model):
        slices = await run_decomposer(
            mode="FALLBACK",
            failed_slice={"id": "parent-slice"},
            error_traceback="boom",
        )
    assert [s["id"] for s in slices] == ["parent-slice-micro-1", "parent-slice-micro-2"]


@pytest.mark.asyncio
async def test_fallback_mode_requires_failed_slice_with_id():
    with pytest.raises(ValueError, match="failed_slice"):
        await run_decomposer(mode="FALLBACK", failed_slice={}, error_traceback="boom")


@pytest.mark.asyncio
async def test_fallback_mode_requires_traceback():
    with pytest.raises(ValueError, match="error_traceback"):
        await run_decomposer(
            mode="FALLBACK",
            failed_slice={"id": "p"},
            error_traceback="",
        )


@pytest.mark.asyncio
async def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown decomposer mode"):
        await run_decomposer(mode="WAT")  # type: ignore[arg-type]
