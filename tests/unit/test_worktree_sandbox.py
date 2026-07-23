"""Unit tests for mandatory worktree sandboxing of code-producing work."""

from __future__ import annotations

import dataclasses

import pytest

from spine.config import SpineConfig
from spine.git.sandbox import WorktreeSandbox, work_type_writes_code


# ── work_type_writes_code ──


@pytest.mark.parametrize(
    "work_type,expected",
    [
        ("task", True),
        ("critical_task", True),
        ("reviewed_task", False),
        ("critical_reviewed_task", False),
        ("onboarding", False),
        ("definitely-not-a-work-type", False),
    ],
)
def test_work_type_writes_code(work_type, expected):
    """Only work types whose sequence runs IMPLEMENT write code."""
    assert work_type_writes_code(work_type) is expected


# ── Fake orchestrator ──


class _FakeOrchestrator:
    """Records sandbox lifecycle calls instead of touching real git."""

    def __init__(self, base_config=None):
        self.base_config = base_config
        self.master_dir = "/orig/cwd"
        self.calls: list[str] = []

    def prepare_sandbox(self) -> str:
        self.calls.append("prepare")
        return "/tmp/spine-sandbox-test"

    def run_validation_pipeline(self) -> dict:
        self.calls.append("validate")
        return {"success": True}

    def commit_and_merge(self) -> dict:
        self.calls.append("merge")
        return {"success": True, "merged": True}

    def commit_and_preserve(self) -> dict:
        self.calls.append("preserve")
        return self.preserve_result

    preserve_result: dict = {
        "preserved": True,
        "branch": "spine/patch-test",
        "sandbox_dir": "/tmp/spine-sandbox-test",
    }

    def rollback_workspace(self) -> dict:
        self.calls.append("rollback")
        return {"rolled_back": True}


@pytest.fixture
def fake_orch(monkeypatch):
    """Patch SpineGitOrchestrator (imported lazily inside enter())."""
    created: list[_FakeOrchestrator] = []

    def _factory(base_config=None):
        orch = _FakeOrchestrator(base_config=base_config)
        created.append(orch)
        return orch

    import spine.git.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "SpineGitOrchestrator", _factory)
    return created


# ── Inactive (non-code) work types ──


def test_inactive_enter_returns_same_config(fake_orch):
    config = SpineConfig(workspace_root="/repo")
    sandbox = WorktreeSandbox(config, "reviewed_task")

    assert sandbox.active is False
    assert sandbox.enter() is config
    # No orchestrator constructed for non-code work.
    assert fake_orch == []
    # finalize / abort are no-ops and must not raise.
    sandbox.finalize("completed")
    sandbox.abort()


# ── Active (code-producing) work types ──


def test_active_enter_swaps_workspace_root(fake_orch):
    config = SpineConfig(workspace_root="/repo")
    sandbox = WorktreeSandbox(config, "task")

    run_config = sandbox.enter()

    assert sandbox.active is True
    assert run_config is not config
    assert run_config.workspace_root == "/tmp/spine-sandbox-test"
    # The rest of the config is preserved.
    assert dataclasses.replace(run_config, workspace_root="/repo") == config
    # master_dir anchored to the resolved repo root, not the process CWD.
    assert fake_orch[0].master_dir == "/repo"
    assert fake_orch[0].calls == ["prepare"]


def test_active_finalize_merges_on_completed(fake_orch):
    sandbox = WorktreeSandbox(SpineConfig(workspace_root="/repo"), "task")
    sandbox.enter()
    sandbox.finalize("completed")
    assert fake_orch[0].calls == ["prepare", "validate", "merge"]


@pytest.mark.parametrize("status", ["stalled", "failed", "error", ""])
def test_active_finalize_rolls_back_on_non_success(fake_orch, status):
    sandbox = WorktreeSandbox(SpineConfig(workspace_root="/repo"), "critical_task")
    sandbox.enter()
    sandbox.finalize(status)
    assert fake_orch[0].calls == ["prepare", "rollback"]


def test_active_finalize_preserves_on_needs_review(fake_orch):
    """A review park keeps the patch: the sandbox worktree and branch ARE
    the artifact a human reviews (run d8bc459c 2026-07-24: rollback nuked
    a 13-file best state with 4/7 slices VERIFIED)."""
    sandbox = WorktreeSandbox(SpineConfig(workspace_root="/repo"), "task")
    sandbox.enter()
    sandbox.finalize("needs_review")
    assert fake_orch[0].calls == ["prepare", "preserve"]


def test_active_finalize_needs_review_empty_sandbox_rolls_back(fake_orch):
    """Nothing to review (no commits ahead of main) → normal rollback."""
    sandbox = WorktreeSandbox(SpineConfig(workspace_root="/repo"), "task")
    sandbox.enter()
    fake_orch[0].preserve_result = {"preserved": False}
    sandbox.finalize("needs_review")
    assert fake_orch[0].calls == ["prepare", "preserve", "rollback"]


def test_active_abort_rolls_back(fake_orch):
    sandbox = WorktreeSandbox(SpineConfig(workspace_root="/repo"), "task")
    sandbox.enter()
    sandbox.abort()
    assert fake_orch[0].calls == ["prepare", "rollback"]


def test_finalize_is_idempotent_after_completion(fake_orch):
    """A second finalize/abort after the sandbox is consumed is a no-op."""
    sandbox = WorktreeSandbox(SpineConfig(workspace_root="/repo"), "task")
    sandbox.enter()
    sandbox.finalize("completed")
    # Sandbox consumed — further calls do nothing.
    sandbox.finalize("completed")
    sandbox.abort()
    assert fake_orch[0].calls == ["prepare", "validate", "merge"]


def test_finalize_blocks_merge_when_validation_fails(fake_orch, monkeypatch):
    """A red validation gate must block the merge, not land broken code.

    Regression: work 545264cc was verified and completed, but its test
    slice errored at collection — slice verification reads code as
    evidence without executing it, and finalize merged the patch anyway,
    leaving 9 erroring tests on main.
    """
    from spine.git.orchestrator import MergeError

    sandbox = WorktreeSandbox(SpineConfig(workspace_root="/repo"), "task")
    sandbox.enter()
    monkeypatch.setattr(
        fake_orch[0],
        "run_validation_pipeline",
        lambda: {
            "success": False,
            "gate": "unit_tests",
            "output": "9 errors",
            "failure_message": "Unit tests failed",
        },
    )
    with pytest.raises(MergeError):
        sandbox.finalize("completed")
    assert "merge" not in fake_orch[0].calls
    # The orchestrator is still attached, so the dispatcher's abort path
    # can roll the un-merged sandbox back.
    sandbox.abort()
    assert fake_orch[0].calls[-1] == "rollback"


def test_abort_never_raises_even_if_rollback_fails(fake_orch, monkeypatch):
    sandbox = WorktreeSandbox(SpineConfig(workspace_root="/repo"), "task")
    sandbox.enter()

    def _boom():
        raise RuntimeError("git exploded")

    monkeypatch.setattr(fake_orch[0], "rollback_workspace", _boom)
    # Must swallow the error so it never masks the original failure.
    sandbox.abort()
