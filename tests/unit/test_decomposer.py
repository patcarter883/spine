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
    # run_decomposer clamps the completion cap via cap_completion_tokens, which
    # returns model.model_copy(update=...). A real ChatOpenAI yields a fresh
    # usable model; mirror that by returning the same configured mock so the
    # bound async structured output survives the copy.
    model.model_copy = MagicMock(return_value=model)
    return model


@pytest.mark.asyncio
async def test_plan_mode_returns_slice_dicts():
    model = _make_structured_mock(_decomposition("a", "b"))
    with patch("spine.agents.decomposer.resolve_chat_model", return_value=model):
        slices = await run_decomposer(
            mode="PLAN",
            spec_markdown="# Spec\n\nDo the thing.",
        )
    assert [s["id"] for s in slices] == ["a", "b"]
    assert all("acceptance_criteria" in s for s in slices)


@pytest.mark.asyncio
async def test_decomposer_clamps_completion_budget():
    """The decomposer must clamp its completion reservation rather than inherit
    the large global max_completion_tokens — a 30K reservation against a finite
    local window OOM-crashes the backend (trace 019ed360)."""
    model = _make_structured_mock(_decomposition("a"))
    with (
        patch("spine.agents.decomposer.resolve_chat_model", return_value=model),
        patch(
            "spine.agents.decomposer.cap_completion_tokens", return_value=model
        ) as cap,
    ):
        await run_decomposer(mode="PLAN", spec_markdown="# Spec\n\nDo it.")
    cap.assert_called_once()
    # Called as cap_completion_tokens(model, <decompose_max_completion_tokens>)
    capped_to = cap.call_args.args[1]
    assert isinstance(capped_to, int) and 0 < capped_to <= 8192


@pytest.mark.asyncio
async def test_plan_mode_rejects_empty_spec():
    with pytest.raises(ValueError, match="spec_markdown"):
        await run_decomposer(mode="PLAN", spec_markdown="")


@pytest.mark.asyncio
async def test_fallback_mode_assigns_micro_suffix_ids():
    # Decomposer returns ids that DON'T follow the convention; the wrapper
    # must rewrite them to '<parent>-micro-N'.
    model = _make_structured_mock(_decomposition("alpha", "beta"))
    with patch("spine.agents.decomposer.resolve_chat_model", return_value=model):
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
    with patch("spine.agents.decomposer.resolve_chat_model", return_value=model):
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


# ── PER_FILE mode ────────────────────────────────────────────────────────


def _per_file_result(*files: str) -> DecompositionResult:
    """A decomposition with one slice per file, each targeting that file."""
    return DecompositionResult(
        slices=[
            FeatureSliceSchema(
                id=f"raw-{i}",
                title=f"raw {i}",
                description=f"work on {f}",
                target_files=[f],
                acceptance_criteria=["ignored — overwritten by parent"],
            )
            for i, f in enumerate(files, start=1)
        ]
    )


_PARENT = {
    "id": "add-auth",
    "title": "Add auth",
    "target_files": ["src/models.py", "src/login.py", "tests/test_auth.py"],
    "acceptance_criteria": ["pytest passes", "imports clean"],
}


@pytest.mark.asyncio
async def test_per_file_mode_one_subslice_per_file_ordered():
    model = _make_structured_mock(
        _per_file_result("src/models.py", "src/login.py", "tests/test_auth.py")
    )
    with patch("spine.agents.decomposer.resolve_chat_model", return_value=model):
        slices = await run_decomposer(mode="PER_FILE", source_slice=_PARENT)

    assert [s["id"] for s in slices] == [
        "add-auth::1-models.py",
        "add-auth::2-login.py",
        "add-auth::3-test_auth.py",
    ]
    # Each sub-slice owns exactly one file...
    assert [s["target_files"] for s in slices] == [
        ["src/models.py"],
        ["src/login.py"],
        ["tests/test_auth.py"],
    ]
    # ...and carries the parent's full acceptance criteria verbatim.
    assert all(s["acceptance_criteria"] == _PARENT["acceptance_criteria"] for s in slices)


@pytest.mark.asyncio
async def test_per_file_mode_appends_uncovered_files():
    # Model only covered 2 of the 3 parent files; the missing one is appended
    # at the end so coverage is never lost.
    model = _make_structured_mock(
        _per_file_result("src/models.py", "src/login.py")
    )
    with patch("spine.agents.decomposer.resolve_chat_model", return_value=model):
        slices = await run_decomposer(mode="PER_FILE", source_slice=_PARENT)

    files = [s["target_files"][0] for s in slices]
    assert files == ["src/models.py", "src/login.py", "tests/test_auth.py"]
    assert slices[-1]["id"] == "add-auth::3-test_auth.py"


@pytest.mark.asyncio
async def test_per_file_mode_requires_source_slice():
    with pytest.raises(ValueError, match="source_slice"):
        await run_decomposer(mode="PER_FILE", source_slice={})


@pytest.mark.asyncio
async def test_per_file_mode_requires_two_files():
    with pytest.raises(ValueError, match="≥2 target_files"):
        await run_decomposer(
            mode="PER_FILE",
            source_slice={"id": "solo", "target_files": ["only.py"]},
        )
