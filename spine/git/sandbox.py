"""Mandatory git-worktree isolation for code-producing workflow runs.

Any work type whose phase sequence includes IMPLEMENT edits the
repository. Those runs MUST execute against a throwaway git worktree
rather than the live working tree, and only land on the main branch when
the workflow completes successfully. A run that ends in any non-success
state (``needs_review``, ``stalled``, ``failed``, …) is rolled back so
the main tree is never left holding half-finished or rejected changes.

This is no longer optional: the dispatcher wraps every code-producing run
(submit / restart / resume / approved-plan continuation) in a
:class:`WorktreeSandbox`. Planning-only and onboarding work types — which
never run IMPLEMENT — pass through untouched.

The sandbox mechanics (worktree creation, fast-forward merge, nuclear
rollback) are provided by :class:`spine.git.orchestrator.SpineGitOrchestrator`;
this module is the thin, mandatory policy layer that decides *when* a
sandbox is required and *whether* to merge or roll back.
"""

from __future__ import annotations

import dataclasses
import logging

from spine.config import SpineConfig

logger = logging.getLogger(__name__)

# Workflow terminal statuses that represent a verified, mergeable result.
# Everything else (needs_review, stalled, failed, error, needs_gap_fix, …)
# leaves the sandbox un-merged and triggers a rollback. A subsequent
# resume/restart regenerates the work in a fresh sandbox.
_MERGE_STATUSES: frozenset[str] = frozenset({"completed"})


def work_type_writes_code(work_type: str) -> bool:
    """Return True when *work_type*'s phase sequence includes IMPLEMENT.

    These are the work types whose agents edit the repository, so every
    run of them must be isolated in a throwaway git worktree. Planning
    work types (``reviewed_task`` / ``critical_reviewed_task``) terminate
    after ``critic_plan`` and never write code; ``onboarding`` is not a
    graph work type at all. Both return False.

    Args:
        work_type: The work type identifier.

    Returns:
        True if the work type runs an IMPLEMENT phase, False otherwise.
    """
    from spine.models.enums import PhaseName
    from spine.workflow.compose import WORKFLOW_SEQUENCES

    sequence = WORKFLOW_SEQUENCES.get(work_type, [])
    return any(name == PhaseName.IMPLEMENT.value for name, _ in sequence)


class WorktreeSandbox:
    """Mandatory worktree isolation around a single code-producing run.

    Lifecycle::

        sandbox = WorktreeSandbox(config, work_type)
        run_config = sandbox.enter()          # swap workspace_root → worktree
        try:
            result = await run_the_graph(run_config)
        except BaseException:
            sandbox.abort()                   # nuke the worktree, re-raise
            raise
        sandbox.finalize(result["status"])    # ff-merge on success, else roll back

    For non-code work types every method is a no-op and :meth:`enter`
    returns the original config unchanged, so callers can wrap *all* runs
    uniformly without branching on the work type themselves.
    """

    def __init__(self, config: SpineConfig, work_type: str) -> None:
        """Initialise the sandbox.

        Args:
            config: The base SPINE configuration for the run.
            work_type: The work type being executed.
        """
        self.config = config
        self.work_type = work_type
        self.active = work_type_writes_code(work_type)
        self._orch: object | None = None

    def enter(self) -> SpineConfig:
        """Prepare the worktree and return the config to run the graph with.

        For a code-producing work type this creates an isolated worktree
        off the main branch and returns a copy of the config whose
        ``workspace_root`` points at that worktree — so every agent writes
        into the sandbox, never the live tree. For other work types it
        returns the original config unchanged.

        Returns:
            The :class:`SpineConfig` to use for the run.

        Raises:
            SandboxPreparationError: If the working tree is dirty or the
                worktree cannot be created. Callers run this inside their
                own failure guard so the work entry is finalised cleanly.
        """
        if not self.active:
            return self.config

        from spine.git.orchestrator import SpineGitOrchestrator

        orch = SpineGitOrchestrator(base_config=self.config)
        # Anchor git operations to the resolved repo root rather than the
        # process CWD, which Streamlit / the worker may have moved.
        if self.config.workspace_root:
            orch.master_dir = self.config.workspace_root
        sandbox_dir = orch.prepare_sandbox()
        self._orch = orch
        logger.info(
            "Code-producing work (type=%s) isolated in worktree sandbox %s",
            self.work_type,
            sandbox_dir,
        )
        return dataclasses.replace(self.config, workspace_root=sandbox_dir)

    def finalize(self, status: str) -> None:
        """Commit-and-merge on success, otherwise roll back.

        No-op when no sandbox was created (non-code work type, or
        :meth:`enter` was never called).

        Args:
            status: The run's terminal workflow status.
        """
        if self._orch is None:
            return
        if status in _MERGE_STATUSES:
            logger.info(
                "Run completed (status=%s) — merging sandbox patch to main", status
            )
            self._orch.commit_and_merge()  # type: ignore[attr-defined]
        else:
            logger.info(
                "Run ended status=%s — rolling back sandbox (no merge to main)",
                status,
            )
            self._orch.rollback_workspace()  # type: ignore[attr-defined]
        self._orch = None

    def abort(self) -> None:
        """Roll back the sandbox after an unhandled error. Never raises."""
        if self._orch is None:
            return
        try:
            self._orch.rollback_workspace()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — abort must not mask the original error
            logger.exception("Sandbox rollback failed during abort")
        finally:
            self._orch = None
