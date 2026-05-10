"""Git worktree manager for parallel feature development.

Creates isolated worktrees per task, manages branch lifecycle, and
integrates with SPINE's swarm supervisor and hive orchestration.
"""

import os
import re
import json
import subprocess
import shutil
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


@dataclass
class WorktreeInfo:
    """Metadata about a managed worktree."""

    task_id: str
    path: str
    branch: str
    base_branch: str = "main"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorktreeInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class WorktreeCreationError(Exception):
    """Raised when worktree creation fails."""


class WorktreeCleanupError(Exception):
    """Raised when worktree cleanup fails."""


class WorktreeManager:
    """Manages git worktrees for parallel task execution.

    Integrates with spine.hive for cell tracking and spine.swarm.supervisor
    for orchestration. Uses Python subprocess for all git operations.

    Worktrees are created under ``<repo>/.spine/worktrees/<task_id>/``.
    Each worktree has its own branch named ``spine/<task_id>-<uuid_short>``.
    """

    # Characters to remove from task IDs for branch names
    _SANITIZE_RE = re.compile(r"[^a-zA-Z0-9._\-]")

    def __init__(
        self,
        repo_path: str,
        worktree_base: Optional[str] = None,
        hive: Any = None,
    ):
        """Initialize the worktree manager.

        Args:
            repo_path: Path to the git repository root.
            worktree_base: Base directory for worktrees. Defaults to
                           ``<repo_path>/.spine/worktrees``.
            hive: Optional Hive instance for cell tracking integration.
        """
        self.repo_path = os.path.abspath(repo_path)
        self.worktree_base = worktree_base or os.path.join(
            self.repo_path, ".spine", "worktrees"
        )
        self._hive = hive
        self._manifest_path = os.path.join(self.worktree_base, "manifest.json")
        os.makedirs(self.worktree_base, exist_ok=True)
        self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest (local tracking file for metadata not in git)
    # ------------------------------------------------------------------

    def _load_manifest(self) -> None:
        """Load the local manifest of managed worktrees."""
        if os.path.exists(self._manifest_path):
            with open(self._manifest_path, "r") as f:
                self._manifest: dict[str, dict[str, Any]] = json.load(f)
        else:
            self._manifest = {}

    def _save_manifest(self) -> None:
        """Persist the local manifest to disk."""
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def _run_git(
        self,
        args: list[str],
        cwd: Optional[str] = None,
        check: bool = True,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        """Run a git command.

        Args:
            args: Command arguments (without 'git' prefix).
            cwd: Working directory for the command.
            check: If True, raise CalledProcessError on failure.
            timeout: Timeout in seconds.

        Returns:
            CompletedProcess instance.
        """
        cmd = ["git", "-C", cwd or self.repo_path] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def create_worktree(
        self,
        task_id: str,
        base_branch: str = "main",
    ) -> WorktreeInfo:
        """Create an isolated worktree for a task.

        Creates a new branch ``spine/<sanitized_task_id>-<uuid_short>``
        from ``base_branch``, then creates a worktree at
        ``<repo>/.spine/worktrees/<task_id>/``.

        Args:
            task_id: Unique identifier for the task (cell_id, bead_id, etc.).
            base_branch: Branch to base the worktree on.

        Returns:
            WorktreeInfo with details about the created worktree.

        Raises:
            WorktreeCreationError: If creation fails for any reason.
        """
        sanitized = self._sanitize(task_id)
        short_id = uuid.uuid4().hex[:8]
        branch = f"spine/{sanitized}-{short_id}"
        worktree_path = os.path.join(self.worktree_base, task_id)

        try:
            # Create branch from base
            self._run_git(["checkout", base_branch])
            self._run_git(["branch", branch, base_branch])

            # Create the worktree
            self._run_git(["worktree", "add", worktree_path, branch])

            info = WorktreeInfo(
                task_id=task_id,
                path=worktree_path,
                branch=branch,
                base_branch=base_branch,
            )

            # Track in manifest and hive
            self._manifest[task_id] = info.to_dict()
            self._save_manifest()

            if self._hive and hasattr(self._hive, "update_cell"):
                self._hive.update_cell(task_id, file_reservation={"worktree": info.to_dict()})

            return info

        except Exception as e:
            # Try to clean up partial state
            self._try_cleanup_partial(branch, worktree_path)
            raise WorktreeCreationError(
                f"Failed to create worktree for '{task_id}': {e}"
            ) from e

    def list_worktrees(self) -> list[WorktreeInfo]:
        """List all SPINE-managed worktrees.

        Queries git worktree list and cross-references with the manifest
        to return structured WorktreeInfo objects.

        Returns:
            List of WorktreeInfo objects.
        """
        try:
            result = self._run_git(["worktree", "list", "--porcelain"], check=True)
        except Exception:
            return []

        worktrees: list[WorktreeInfo] = []
        current_wt: dict[str, str] = {}

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                if current_wt:
                    path = current_wt.get("worktree", "")
                    # Only include SPINE-managed worktrees
                    if self.worktree_base in path:
                        task_id = os.path.basename(path)
                        info = self._build_info_from_manifest(task_id, path, current_wt)
                        if info:
                            worktrees.append(info)
                    current_wt = {}
            elif line.startswith("worktree "):
                current_wt["worktree"] = line[len("worktree "):]
            elif line.startswith("branch "):
                ref = line[len("branch "):]
                if ref.startswith("refs/heads/"):
                    current_wt["branch"] = ref[len("refs/heads/"):]
                else:
                    current_wt["branch"] = ref
            elif line.startswith("HEAD "):
                current_wt["head"] = line[len("HEAD "):]

        # Handle last entry
        if current_wt:
            path = current_wt.get("worktree", "")
            if self.worktree_base in path:
                task_id = os.path.basename(path)
                info = self._build_info_from_manifest(task_id, path, current_wt)
                if info:
                    worktrees.append(info)

        return worktrees

    def _build_info_from_manifest(
        self, task_id: str, path: str, git_data: dict[str, str]
    ) -> Optional[WorktreeInfo]:
        """Build WorktreeInfo from manifest and git data."""
        manifest_entry = self._manifest.get(task_id)
        branch = git_data.get("branch", "")

        if manifest_entry:
            return WorktreeInfo(
                task_id=manifest_entry.get("task_id", task_id),
                path=manifest_entry.get("path", path),
                branch=manifest_entry.get("branch", branch),
                base_branch=manifest_entry.get("base_branch", "main"),
            )
        elif branch:
            return WorktreeInfo(
                task_id=task_id,
                path=path,
                branch=branch,
                base_branch="main",
            )
        return None

    def get_worktree_path(self, task_id: str) -> Optional[str]:
        """Get the filesystem path for a task's worktree.

        Args:
            task_id: The task identifier.

        Returns:
            Absolute path to the worktree, or None if not found.
        """
        # Check manifest first
        entry = self._manifest.get(task_id)
        if entry and os.path.isdir(entry.get("path", "")):
            return entry["path"]

        # Fall back to convention
        default_path = os.path.join(self.worktree_base, task_id)
        if os.path.isdir(default_path):
            return default_path

        return None

    def cleanup_worktree(self, task_id: str) -> bool:
        """Remove a worktree and its associated branch.

        Args:
            task_id: The task identifier.

        Returns:
            True if cleanup succeeded, False otherwise.

        Raises:
            WorktreeCleanupError: If cleanup fails after attempts.
        """
        manifest_entry = self._manifest.get(task_id, {})
        worktree_path = manifest_entry.get("path") or os.path.join(
            self.worktree_base, task_id
        )
        branch = manifest_entry.get("branch", "")

        try:
            if os.path.isdir(worktree_path):
                # Remove the worktree
                self._run_git(
                    ["worktree", "remove", "--force", worktree_path],
                    check=True,
                    timeout=15,
                )

            if branch:
                # Delete the branch
                try:
                    self._run_git(
                        ["branch", "-D", branch],
                        check=False,
                        timeout=10,
                    )
                except Exception:
                    pass  # Branch may already be gone

            # Remove from manifest
            self._manifest.pop(task_id, None)
            self._save_manifest()

            # Update hive if available
            if self._hive and hasattr(self._hive, "update_cell"):
                self._hive.update_cell(task_id, file_reservation=None)

            return True

        except Exception as e:
            raise WorktreeCleanupError(
                f"Failed to clean up worktree for '{task_id}': {e}"
            ) from e

    def cleanup_all(self) -> int:
        """Remove all SPINE-managed worktrees.

        Returns:
            Number of worktrees cleaned up.
        """
        worktrees = self.list_worktrees()
        cleaned = 0
        for wt in worktrees:
            try:
                self.cleanup_worktree(wt.task_id)
                cleaned += 1
            except WorktreeCleanupError:
                pass
        return cleaned

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sanitize(self, task_id: str) -> str:
        """Sanitize a task ID for use in branch names.

        Replaces non-alphanumeric characters with hyphens and collapses
        multiple hyphens.
        """
        sanitized = self._SANITIZE_RE.sub("-", task_id)
        sanitized = re.sub(r"-+", "-", sanitized)
        sanitized = sanitized.strip("-")
        return sanitized or "unnamed"

    def _try_cleanup_partial(self, branch: str, worktree_path: str) -> None:
        """Attempt to clean up partial worktree state after a failed creation."""
        try:
            if os.path.isdir(worktree_path):
                self._run_git(
                    ["worktree", "remove", "--force", worktree_path],
                    check=False,
                )
            if branch:
                self._run_git(["branch", "-D", branch], check=False)
        except Exception:
            pass  # Best-effort cleanup
