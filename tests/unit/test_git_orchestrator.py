"""Fast unit tests for the transactional git-sandbox orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from spine.config import SpineConfig
from spine.exceptions import SandboxPreparationError
from spine.git import orchestrator as orch_module
from spine.git.orchestrator import SpineGitOrchestrator


def _make_orchestrator(tmp_path: Path) -> SpineGitOrchestrator:
    """Build an orchestrator with default gate config and a tmp master dir."""
    base_config = SpineConfig()
    orch = SpineGitOrchestrator(
        config_path=str(tmp_path / "does-not-exist.yaml"),
        base_config=base_config,
    )
    # Pin master_dir to a deterministic absolute path for assertions.
    orch.master_dir = str(tmp_path)
    return orch


class _ScriptedShell:
    """Callable that returns scripted (success, stdout, stderr) tuples.

    Each entry may be either a ready tuple or a predicate over the command
    string returning a tuple. Records every command it was asked to run.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, cmd, cwd=None, timeout=60):
        self.calls.append(cmd)
        if not self._responses:
            return (True, "", "")
        resp = self._responses.pop(0)
        return resp


# ── _resolve_validation_command ──


def test_resolve_validation_command_rewrites_venv(tmp_path):
    orch = _make_orchestrator(tmp_path)
    resolved = orch._resolve_validation_command(".venv/bin/ruff check .")
    assert resolved == str(Path(tmp_path) / ".venv/bin/ruff check .")
    assert resolved.startswith(str(tmp_path))


def test_resolve_validation_command_leaves_others_alone(tmp_path):
    orch = _make_orchestrator(tmp_path)
    assert orch._resolve_validation_command("echo hi") == "echo hi"
    assert orch._resolve_validation_command("pytest tests/") == "pytest tests/"


# ── run_validation_pipeline ──


def test_validation_pipeline_all_pass(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    orch.gate_config["validation_pipeline"] = {
        "lint": {"command": "echo lint"},
        "test": {"command": "echo test"},
    }
    shell = _ScriptedShell([(True, "", ""), (True, "", "")])
    monkeypatch.setattr(orch, "_execute_shell", shell)

    result = orch.run_validation_pipeline()

    assert result == {"success": True}
    assert len(shell.calls) == 2


def test_validation_pipeline_first_gate_fails_and_stops(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    orch.gate_config["validation_pipeline"] = {
        "lint": {"command": "echo lint", "failure_message": "Lint failed."},
        "typecheck": {"command": "echo typecheck"},
        "test": {"command": "echo test"},
    }
    shell = _ScriptedShell([(False, "out", "err")])
    monkeypatch.setattr(orch, "_execute_shell", shell)

    result = orch.run_validation_pipeline()

    assert result["success"] is False
    assert result["gate"] == "lint"
    assert result["failure_message"] == "Lint failed."
    assert "out" in result["output"] and "err" in result["output"]
    # Only the first gate ran; later gates were not executed.
    assert len(shell.calls) == 1


# ── prepare_sandbox ──


def test_prepare_sandbox_raises_on_dirty_tree(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    # git status --porcelain returns nonempty -> dirty.
    shell = _ScriptedShell([(True, " M some_file.py\n", "")])
    monkeypatch.setattr(orch, "_execute_shell", shell)

    with pytest.raises(SandboxPreparationError, match="not clean"):
        orch.prepare_sandbox()


def test_prepare_sandbox_raises_on_status_failure(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    shell = _ScriptedShell([(False, "", "fatal: not a git repo")])
    monkeypatch.setattr(orch, "_execute_shell", shell)

    with pytest.raises(SandboxPreparationError, match="inspect working tree"):
        orch.prepare_sandbox()


def test_prepare_sandbox_raises_on_worktree_add_failure(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    orch.strategy = "worktree"
    # clean status, then worktree add fails.
    shell = _ScriptedShell(
        [
            (True, "", ""),
            (False, "", "fatal: worktree add failed"),
        ]
    )
    monkeypatch.setattr(orch, "_execute_shell", shell)

    with pytest.raises(SandboxPreparationError, match="create worktree"):
        orch.prepare_sandbox()


def test_prepare_sandbox_success_sets_state(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    orch.strategy = "worktree"
    shell = _ScriptedShell([(True, "", ""), (True, "", "")])
    monkeypatch.setattr(orch, "_execute_shell", shell)

    sandbox = orch.prepare_sandbox()

    assert orch.sandbox_dir == sandbox
    assert orch.patch_branch is not None
    assert orch.patch_branch.startswith(orch.branch_prefix)
    assert sandbox.startswith(orch.sandbox_dir_base)


# ── rollback_workspace ──


def test_rollback_never_raises_and_returns_flag(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    orch.strategy = "worktree"
    orch.patch_branch = "spine/patch-deadbeef"
    # Point sandbox at a non-existent path so rmtree branch is exercised safely.
    orch.sandbox_dir = str(tmp_path / "nonexistent-sandbox")

    # Every shell call "fails".
    def failing_shell(cmd, cwd=None, timeout=60):
        return (False, "", "boom")

    monkeypatch.setattr(orch, "_execute_shell", failing_shell)

    result = orch.rollback_workspace()

    assert result == {"rolled_back": True}


# ── execute_transactional_run (validation failure path) ──


def test_execute_transactional_run_validation_failure_rolls_back(
    tmp_path, monkeypatch
):
    orch = _make_orchestrator(tmp_path)
    orch.strategy = "worktree"

    # Avoid touching real git: stub prepare_sandbox to set state directly.
    def fake_prepare():
        orch.patch_branch = "spine/patch-aaaa1111"
        orch.sandbox_dir = str(tmp_path / "sandbox")
        return orch.sandbox_dir

    monkeypatch.setattr(orch, "prepare_sandbox", fake_prepare)

    # Stub submit_work (the symbol the orchestrator imported) — it is awaited
    # via asyncio.run, so provide an async stub.
    async def fake_submit_work(description, work_type, config):
        return {
            "work_id": "abc",
            "status": "completed",
            "work_type": "task",
        }

    monkeypatch.setattr(orch_module, "submit_work", fake_submit_work)

    monkeypatch.setattr(orch, "_check_phase_prerequisites", lambda *a, **k: True)

    # Force validation to fail.
    monkeypatch.setattr(
        orch,
        "run_validation_pipeline",
        lambda: {
            "success": False,
            "gate": "test",
            "command": "pytest",
            "output": "1 failed",
            "failure_message": "Tests failed.",
        },
    )

    rollback_calls = []
    real_rollback = orch.rollback_workspace

    def spy_rollback():
        rollback_calls.append(True)
        # Don't run real git; just return the contract shape.
        return {"rolled_back": True}

    monkeypatch.setattr(orch, "rollback_workspace", spy_rollback)
    # commit_and_merge must never be reached on a failure path.
    monkeypatch.setattr(
        orch,
        "commit_and_merge",
        lambda: pytest.fail("commit_and_merge should not run on validation failure"),
    )

    result = orch.execute_transactional_run("do a thing", work_type="task")

    assert result["status"] == "rolled_back"
    assert result["stage"] == "validation"
    assert result["gate"] == "test"
    assert result["work_id"] == "abc"
    assert rollback_calls == [True]
    assert real_rollback is not None  # sanity: attribute existed before patching


def test_execute_transactional_run_workflow_error_rolls_back(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)

    def fake_prepare():
        orch.patch_branch = "spine/patch-bbbb2222"
        orch.sandbox_dir = str(tmp_path / "sandbox")
        return orch.sandbox_dir

    monkeypatch.setattr(orch, "prepare_sandbox", fake_prepare)

    async def fake_submit_work(description, work_type, config):
        return {"work_id": "xyz", "error": "boom", "status": "failed"}

    monkeypatch.setattr(orch_module, "submit_work", fake_submit_work)

    rollback_calls = []
    monkeypatch.setattr(
        orch,
        "rollback_workspace",
        lambda: rollback_calls.append(True) or {"rolled_back": True},
    )

    result = orch.execute_transactional_run("do a thing")

    assert result["status"] == "failed"
    assert result["stage"] == "workflow"
    assert result["error"] == "boom"
    assert rollback_calls == [True]


# ── status ──


def test_status_reports_inactive_then_active(tmp_path):
    orch = _make_orchestrator(tmp_path)
    inactive = orch.status()
    assert inactive["active"] is False
    assert inactive["branch"] is None
    assert inactive["strategy"] == orch.strategy

    orch.patch_branch = "spine/patch-cccc3333"
    orch.sandbox_dir = "/tmp/sandbox-x"
    active = orch.status()
    assert active["active"] is True
    assert active["branch"] == "spine/patch-cccc3333"
    assert active["sandbox_dir"] == "/tmp/sandbox-x"
