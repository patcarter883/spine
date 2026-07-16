"""Transactional git-sandbox orchestrator for SPINE workflow runs."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import yaml

from spine.config import SpineConfig
from spine.exceptions import (
    MergeError,
    SandboxPreparationError,
)
from spine.persistence.checkpoint import CheckpointStore
from spine.work.dispatcher import submit_work

logger = logging.getLogger(__name__)


# Defaults mirror the shipped spine-gate.yaml so the orchestrator behaves
# sanely even when no config file is present.
_DEFAULT_GATE_CONFIG: dict = {
    "git": {
        "main_branch": "main",
        "branch_prefix": "spine/patch-",
        "strategy": "worktree",
        "sandbox_dir": "/tmp/spine-sandbox",
    },
    "validation_pipeline": {
        "lint": {
            "command": ".venv/bin/ruff check .",
            "timeout_seconds": 120,
            "failure_message": "Lint failed.",
        },
        "typecheck": {
            "command": ".venv/bin/mypy spine/",
            "timeout_seconds": 300,
            "failure_message": "Type check failed.",
        },
        "test": {
            "command": ".venv/bin/pytest tests/ -x -q",
            "timeout_seconds": 600,
            "failure_message": "Tests failed.",
        },
    },
    "artifact_path": ".spine/artifacts",
    "auto_merge_on_success": True,
    "require_successful_phases": ["implement_completed", "verify_completed"],
}


def load_gate_config(path: str = "spine-gate.yaml") -> dict:
    """Load the gate configuration, falling back to built-in defaults.

    Args:
        path: Path to the ``spine-gate.yaml`` file.

    Returns:
        The parsed gate configuration dict, or the default configuration
        when the file is missing or empty.
    """
    config_file = Path(path)
    if not config_file.is_file():
        logger.debug("Gate config %s not found; using defaults", path)
        return dict(_DEFAULT_GATE_CONFIG)
    with config_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not loaded:
        logger.debug("Gate config %s empty; using defaults", path)
        return dict(_DEFAULT_GATE_CONFIG)
    return loaded


class SpineGitOrchestrator:
    """Wrap a workflow run in an atomically-merged or rolled-back git sandbox.

    The orchestrator prepares an isolated sandbox (a git worktree or a
    throwaway branch), runs the workflow against it, validates the produced
    changes through an ordered gate pipeline, and either fast-forward merges
    the verified patch to the main branch or performs a hard rollback.
    """

    def __init__(
        self,
        config_path: str = "spine-gate.yaml",
        base_config: SpineConfig | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config_path: Path to the gate YAML configuration.
            base_config: SPINE configuration to derive the sandbox config
                from. Defaults to ``SpineConfig.load()``.
        """
        self.gate_config: dict = load_gate_config(config_path)
        git_cfg: dict = self.gate_config.get("git", {})
        self.main_branch: str = git_cfg.get("main_branch", "main")
        self.branch_prefix: str = git_cfg.get("branch_prefix", "spine/patch-")
        self.strategy: str = git_cfg.get("strategy", "worktree")
        self.sandbox_dir_base: str = git_cfg.get("sandbox_dir", "/tmp/spine-sandbox")

        self.base_config: SpineConfig = base_config or SpineConfig.load()
        self.master_dir: str = os.getcwd()

        self.patch_branch: str | None = None
        self.sandbox_dir: str | None = None
        # The branch the OPERATOR had checked out in the master dir. Landing
        # and rollback must return here — leaving `main` checked out silently
        # swaps the live (branch-committed) config/state under any process
        # reading files from the master dir (observed live 2026-07-16: a
        # stalled run reverted .spine/config.yaml to main's paid-provider
        # version). None when detached/unresolvable — falls back to main.
        self.original_branch: str | None = self._current_master_branch()

    def _current_master_branch(self) -> str | None:
        """The master dir's checked-out branch, or ``None`` if detached."""
        ok, out, _err = self._execute_shell(
            "git rev-parse --abbrev-ref HEAD", cwd=self.master_dir
        )
        name = (out or "").strip()
        return name if ok and name and name != "HEAD" else None

    def _restore_master_branch(self) -> None:
        """Best-effort: put the master dir back on the operator's branch."""
        restore = self.original_branch or self.main_branch
        if restore == self.patch_branch:
            restore = self.main_branch
        self._execute_shell(f"git checkout {restore}", cwd=self.master_dir)

    # ── low-level helpers ──

    def _execute_shell(
        self,
        cmd: str,
        cwd: str | None = None,
        timeout: int = 60,
    ) -> tuple[bool, str, str]:
        """Run a shell command and capture its result.

        Args:
            cmd: Shell command line to execute.
            cwd: Working directory for the command.
            timeout: Maximum seconds before the command is killed.

        Returns:
            A ``(success, stdout, stderr)`` tuple. ``success`` is True when
            the command exits zero. On timeout returns ``(False, "", "timeout")``.
        """
        logger.debug("Executing shell command (cwd=%s): %s", cwd, cmd)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Command timed out after %ss: %s", timeout, cmd)
            return (False, "", "timeout")
        return (proc.returncode == 0, proc.stdout, proc.stderr)

    def _resolve_validation_command(self, command: str) -> str:
        """Resolve a validation command relative to the master tree.

        Worktrees do not copy the virtualenv, so commands referencing
        ``.venv/`` are rewritten to point at the master repo's venv.

        Args:
            command: The configured gate command.

        Returns:
            The command with any leading ``.venv/`` made absolute.
        """
        if command.startswith(".venv/"):
            return str(Path(self.master_dir) / command)
        return command

    # ── lifecycle ──

    def ensure_tree_clean(self) -> None:
        """Raise if the master working tree has uncommitted changes.

        This is the precondition :meth:`prepare_sandbox` enforces, factored
        out so callers can check it *without* the side effect of creating a
        worktree. A dirty tree leaves no state behind, so a caller can run
        this before mutating any status and remain retryable once the tree
        is cleaned.

        Raises:
            SandboxPreparationError: If git status reports a dirty tree or
                the status command itself fails.
        """
        ok, stdout, stderr = self._execute_shell(
            "git status --porcelain", cwd=self.master_dir
        )
        if not ok:
            raise SandboxPreparationError(
                f"Failed to inspect working tree: {stderr}"
            )
        if stdout.strip():
            raise SandboxPreparationError(
                "Working tree is not clean; commit or stash changes before "
                "preparing a sandbox."
            )

    def prepare_sandbox(self) -> str:
        """Create the isolated sandbox worktree or branch.

        Returns:
            The absolute sandbox directory path.

        Raises:
            SandboxPreparationError: If the working tree is dirty or the
                git command to create the sandbox fails.
        """
        self.ensure_tree_clean()

        token = uuid.uuid4().hex[:8]
        branch = f"{self.branch_prefix}{token}"

        if self.strategy == "worktree":
            sandbox_dir = f"{self.sandbox_dir_base}-{token}"
            ok, _stdout, stderr = self._execute_shell(
                f"git worktree add -b {branch} {sandbox_dir} {self.main_branch}",
                cwd=self.master_dir,
            )
            if not ok:
                raise SandboxPreparationError(
                    f"Failed to create worktree: {stderr}"
                )
        else:
            sandbox_dir = self.master_dir
            ok, _stdout, stderr = self._execute_shell(
                f"git checkout -b {branch}", cwd=self.master_dir
            )
            if not ok:
                raise SandboxPreparationError(
                    f"Failed to create branch: {stderr}"
                )

        self.patch_branch = branch
        self.sandbox_dir = sandbox_dir

        # ── Sandbox setup hooks (spine-gate.yaml: sandbox_setup) ──
        # A fresh worktree contains only TRACKED files; projects whose build
        # or test tooling needs untracked state get it seeded here. The
        # motivating case is a Laravel Sail project (agripath clone): pest
        # inside the Sail container needs the project's .env and the
        # gitignored storage/oauth-*.key files copied into every sandbox —
        # without them the whole suite fails before testing anything. Hooks
        # run IN the sandbox, in order; a failing hook aborts the run before
        # any tokens are spent, exactly like a dirty tree.
        for hook in self.gate_config.get("sandbox_setup") or []:
            command = (hook or {}).get("command", "")
            if not command:
                continue
            timeout = int((hook or {}).get("timeout_seconds", 120))
            logger.info("Sandbox setup hook: %s", command)
            ok, stdout, stderr = self._execute_shell(
                command, cwd=sandbox_dir, timeout=timeout
            )
            if not ok:
                self.rollback_workspace()
                raise SandboxPreparationError(
                    f"Sandbox setup hook failed: {command!r}: "
                    f"{(stdout + stderr).strip()[:500]}"
                )

        logger.info(
            "Prepared sandbox (strategy=%s, branch=%s, dir=%s)",
            self.strategy,
            branch,
            sandbox_dir,
        )
        return sandbox_dir

    def run_validation_pipeline(self) -> dict:
        """Run all validation gates in order against the sandbox.

        Returns:
            ``{"success": True}`` if every gate passes, otherwise
            ``{"success": False, "gate", "command", "output",
            "failure_message"}`` describing the first failing gate.
        """
        pipeline: dict = self.gate_config.get("validation_pipeline", {})
        for name, gate in pipeline.items():
            command = gate.get("command", "")
            resolved = self._resolve_validation_command(command)
            timeout = int(gate.get("timeout_seconds", 60))
            logger.info("Running validation gate '%s': %s", name, resolved)
            ok, stdout, stderr = self._execute_shell(
                resolved, cwd=self.sandbox_dir, timeout=timeout
            )
            if not ok:
                logger.warning("Validation gate '%s' failed", name)
                return {
                    "success": False,
                    "gate": name,
                    "command": resolved,
                    "output": stdout + stderr,
                    "failure_message": gate.get("failure_message", ""),
                }
        return {"success": True}

    def _check_phase_prerequisites(
        self, work_id: str, required_phases: list[str]
    ) -> bool:
        """Verify the required workflow phases completed for a work item.

        Args:
            work_id: The work item identifier.
            required_phases: Phase names (``"implement"`` or
                ``"implement_completed"`` form) that must be completed.

        Returns:
            True if all required phases are marked completed in the
            checkpoint state, otherwise False.
        """
        if not required_phases:
            return True

        state = (
            asyncio.run(
                CheckpointStore(
                    db_path=self.base_config.checkpoint_path
                ).get_state(work_id)
            )
            or {}
        )
        for phase in required_phases:
            base = phase[: -len("_completed")] if phase.endswith("_completed") else phase
            flag = f"{base}_completed"
            if not state.get(flag):
                logger.warning(
                    "Phase prerequisite not met for work %s: %s", work_id, flag
                )
                return False
        return True

    def commit_and_merge(self) -> dict:
        """Commit sandbox changes and fast-forward merge them to main.

        Returns:
            ``{"success": True, "branch": <branch>, "merged": True}``.

        Raises:
            MergeError: If the fast-forward merge fails (e.g. a conflict).
        """
        branch = self.patch_branch
        self._execute_shell("git add .", cwd=self.sandbox_dir)
        commit_ok, commit_out, commit_err = self._execute_shell(
            f'git commit -m "spine(auto): verified patch {branch}"',
            cwd=self.sandbox_dir,
        )
        if not commit_ok and "nothing to commit" not in (commit_out + commit_err):
            logger.warning("Commit reported non-zero: %s", commit_out + commit_err)

        # Bring the patch branch up to date with any commits that landed on
        # main while the run was in flight (another work item, a human push).
        # Without this the --ff-only merge below fails for every run that
        # isn't the sole writer to the repo, discarding a verified patch. A
        # rebase conflict means the patch genuinely overlaps the new main
        # state — surface it as a MergeError rather than guessing.
        rebase_ok, _rebase_out, rebase_err = self._execute_shell(
            f"git rebase {self.main_branch}", cwd=self.sandbox_dir
        )
        if not rebase_ok:
            self._execute_shell("git rebase --abort", cwd=self.sandbox_dir)
            raise MergeError(
                f"Rebase of '{branch}' onto {self.main_branch} failed "
                f"(main advanced during the run and the patch conflicts): "
                f"{rebase_err}"
            )

        self._execute_shell(
            f"git checkout {self.main_branch}", cwd=self.master_dir
        )
        merge_ok, _merge_out, merge_err = self._execute_shell(
            f"git merge --ff-only {branch}", cwd=self.master_dir
        )
        if not merge_ok:
            raise MergeError(
                f"Fast-forward merge of '{branch}' failed: {merge_err}"
            )

        self._execute_shell(f"git branch -d {branch}", cwd=self.master_dir)
        if self.strategy == "worktree":
            self._execute_shell(
                f"git worktree remove {self.sandbox_dir}", cwd=self.master_dir
            )
        # The merge advanced main; now hand the master dir back to the
        # operator's branch (no-op when they were on main).
        self._restore_master_branch()

        logger.info("Merged verified patch branch %s into %s", branch, self.main_branch)
        return {"success": True, "branch": branch, "merged": True}

    def rollback_workspace(self) -> dict:
        """Nuke the sandbox and restore the master tree, best-effort.

        Every step is best-effort and never raises so it is safe to call
        from error/finally paths.

        Returns:
            ``{"rolled_back": True}``.
        """
        branch = self.patch_branch
        self._restore_master_branch()
        if self.strategy == "worktree" and self.sandbox_dir:
            self._execute_shell(
                f"git worktree remove --force {self.sandbox_dir}",
                cwd=self.master_dir,
            )
        if branch:
            self._execute_shell(f"git branch -D {branch}", cwd=self.master_dir)
        self._execute_shell("git worktree prune", cwd=self.master_dir)

        if (
            self.sandbox_dir
            and self.sandbox_dir != self.master_dir
            and Path(self.sandbox_dir).exists()
        ):
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

        # Only scrub the master tree when the run actually used it as the
        # sandbox (non-worktree strategy). Under worktree isolation the
        # master tree was never touched by the run — a hard reset + clean
        # here destroys the OPERATOR's uncommitted/untracked state.
        if self.strategy != "worktree":
            self._execute_shell("git reset --hard HEAD", cwd=self.master_dir)
            self._execute_shell("git clean -fd", cwd=self.master_dir)

        logger.info("Rolled back sandbox (branch=%s)", branch)
        return {"rolled_back": True}

    def execute_transactional_run(
        self, description: str, work_type: str = "task"
    ) -> dict:
        """Run a full transactional workflow lifecycle.

        Prepares a sandbox, runs the workflow, checks phase prerequisites,
        validates the result, then merges or rolls back atomically.

        .. deprecated::
            Worktree sandboxing is now mandatory inside the dispatcher
            itself (see :class:`spine.git.sandbox.WorktreeSandbox`), so
            every ``submit_work`` already isolates code-producing runs.
            Do NOT call this against ``submit_work`` — it would prepare a
            *second* nested sandbox. Retained only for the standalone
            sandbox primitives and their tests. The
            ``prepare_sandbox`` / ``commit_and_merge`` / ``rollback_workspace``
            methods remain the live mechanism the mandatory path builds on.

        Args:
            description: The work description to submit.
            work_type: The work type to dispatch.

        Returns:
            A result dict whose ``status`` is one of ``"merged"``,
            ``"validated_pending_merge"``, ``"rolled_back"``, ``"failed"``,
            or ``"error"``.
        """
        original_dir = os.getcwd()
        try:
            try:
                sandbox_dir = self.prepare_sandbox()
                sandbox_config = dataclasses.replace(
                    self.base_config, workspace_root=sandbox_dir
                )
                work_result = asyncio.run(
                    submit_work(description, work_type, sandbox_config)
                )

                if work_result.get("error"):
                    self.rollback_workspace()
                    return {
                        "status": "failed",
                        "stage": "workflow",
                        "error": work_result.get("error"),
                        **work_result,
                    }

                work_id = work_result["work_id"]
                required = self.gate_config.get("require_successful_phases", [])
                if required and not self._check_phase_prerequisites(work_id, required):
                    self.rollback_workspace()
                    return {
                        "status": "rolled_back",
                        "stage": "prerequisites",
                        "work_id": work_id,
                    }

                validation = self.run_validation_pipeline()
                if not validation["success"]:
                    self.rollback_workspace()
                    return {
                        "status": "rolled_back",
                        "stage": "validation",
                        "gate": validation["gate"],
                        "output": validation["output"],
                        "work_id": work_id,
                    }

                if self.gate_config.get("auto_merge_on_success", True):
                    merge = self.commit_and_merge()
                    return {
                        "status": "merged",
                        "work_id": work_id,
                        "branch": merge["branch"],
                    }

                return {
                    "status": "validated_pending_merge",
                    "work_id": work_id,
                    "branch": self.patch_branch,
                }
            except Exception as exc:  # noqa: BLE001 — defensive transactional guard
                logger.exception("Transactional run failed; rolling back")
                self.rollback_workspace()
                return {"status": "error", "error": str(exc)}
        finally:
            os.chdir(original_dir)

    def status(self) -> dict:
        """Return the current orchestrator state.

        Returns:
            A dict describing whether a sandbox is active and its details.
        """
        return {
            "active": bool(self.patch_branch),
            "branch": self.patch_branch,
            "sandbox_dir": self.sandbox_dir,
            "strategy": self.strategy,
        }
