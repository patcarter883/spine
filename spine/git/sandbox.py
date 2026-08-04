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
# Everything else (stalled, failed, error, needs_gap_fix, …) leaves the
# sandbox un-merged and triggers a rollback. A subsequent resume/restart
# regenerates the work in a fresh sandbox.
_MERGE_STATUSES: frozenset[str] = frozenset({"completed"})

# Review parks: the patch IS the artifact a human reviews, so the sandbox
# worktree and patch branch are committed and kept (run d8bc459c 2026-07-24:
# a needs_review exit rolled back a 13-file best state with 4/7 slices
# VERIFIED — the only reviewable copy of the work). Prior parked sandboxes
# only ever survived because killed runs never reached finalize.
# "stalled" preserves too: a stall is an infrastructure event (endpoint hung
# mid-call), not a verdict on the code — the same day's stalled exit rolled
# back a patch verify had just scored 4/8 slices VERIFIED. The verify
# ratchet keeps the on-disk state at the best scoring cycle, so what is
# preserved is the run's best work, not mid-edit debris. Empty sandboxes
# still roll back via the preserved=False branch.
_PRESERVE_STATUSES: frozenset[str] = frozenset({"needs_review", "stalled"})


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

    def preflight(self) -> None:
        """Verify a sandbox *can* be prepared, without creating one.

        For a code-producing work type this checks the same precondition
        :meth:`enter` enforces — a clean working tree — but creates no
        worktree and mutates no state. Run it *before* any status
        transition so a dirty tree fails fast and leaves the work entry
        untouched (and therefore retryable) instead of progressing it to
        running/failed and stranding the plan. No-op for non-code work
        types.

        Raises:
            SandboxPreparationError: If the working tree is dirty.
        """
        if not self.active:
            return

        from spine.git.orchestrator import SpineGitOrchestrator

        orch = SpineGitOrchestrator(base_config=self.config)
        if self.config.workspace_root:
            orch.master_dir = self.config.workspace_root
        orch.ensure_tree_clean()

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
            # Gate the landing on the validation pipeline (spine-gate.yaml).
            # Slice verification is evidence-based — it reads code, it does
            # not execute it — so a "completed" run can still carry tests
            # that error at collection (work 545264cc landed a test class
            # whose setup called a nonexistent method; 9 tests errored on
            # main). Never merge a patch the pipeline can't green-light.
            # An EMPTY pipeline can't green-light anything either: with no
            # gate the internal LLM verify is grading its own homework
            # (Wallace parity report 2026-07-24: two "verified" merges,
            # both broken — one gutted a shipped 522-line component).
            # Explicit `allow_merge_without_gate: true` opts back in.
            # Raising (instead of silently rolling back) routes through the
            # dispatcher's error path and records the reason on the work
            # entry — but the patch is COMMITTED AND PRESERVED first, so
            # the blocked merge stays reviewable instead of being rolled
            # back with the sandbox.
            gate_cfg = getattr(self._orch, "gate_config", None) or {}
            if not (gate_cfg.get("validation_pipeline") or {}) and not gate_cfg.get(
                "allow_merge_without_gate"
            ):
                from spine.git.orchestrator import MergeError

                preserved = self._orch.commit_and_preserve()  # type: ignore[attr-defined]
                self._orch = None
                raise MergeError(
                    "Refusing to auto-merge: validation_pipeline is empty and "
                    "allow_merge_without_gate is not set — internal verify "
                    "alone cannot green-light a merge. "
                    + (
                        f"Patch preserved for review: branch={preserved.get('branch')} "
                        f"worktree={preserved.get('sandbox_dir')}"
                        if preserved.get("preserved")
                        else "Sandbox had no changes."
                    )
                )
            validation = self._orch.run_validation_pipeline()  # type: ignore[attr-defined]
            if not validation.get("success", False):
                from spine.git.orchestrator import MergeError

                gate = validation.get("gate", "?")
                output = (validation.get("output") or "").strip()[-2000:]
                logger.error(
                    "Validation gate '%s' failed — refusing to merge sandbox patch:\n%s",
                    gate,
                    output,
                )
                preserved = self._orch.commit_and_preserve()  # type: ignore[attr-defined]
                self._orch = None
                pointer = (
                    f" Patch preserved for review: branch={preserved.get('branch')} "
                    f"worktree={preserved.get('sandbox_dir')}."
                    if preserved.get("preserved")
                    else ""
                )
                raise MergeError(
                    f"Validation gate '{gate}' blocked the merge "
                    f"({validation.get('failure_message') or 'gate failed'})."
                    f"{pointer} Output tail:\n{output}"
                )
            logger.info(
                "Run completed (status=%s) — merging sandbox patch to main", status
            )
            self._orch.commit_and_merge()  # type: ignore[attr-defined]
        elif status in _PRESERVE_STATUSES:
            preserved = self._orch.commit_and_preserve()  # type: ignore[attr-defined]
            if preserved.get("preserved"):
                # WARNING so the pointer survives default log filtering — this
                # line is how a reviewer finds the parked patch.
                logger.warning(
                    "Run parked status=%s — patch preserved for review: "
                    "branch=%s worktree=%s",
                    status,
                    preserved.get("branch"),
                    preserved.get("sandbox_dir"),
                )
            else:
                logger.info(
                    "Run parked status=%s with an empty sandbox — rolling back",
                    status,
                )
                self._orch.rollback_workspace()  # type: ignore[attr-defined]
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
