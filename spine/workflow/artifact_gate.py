"""SPINE artifact gate — structural pre-check before phase transitions.

An artifact gate ensures a phase doesn't run if its prerequisite phase
produced no artifacts. Currently only the tasks→implement transition is
gated: implement requires meaningful task artifacts before it can generate code.

The implement→verify transition is NOT gated. Verify always runs after
implement — if implement produced nothing, verify can detect and report that.
There is no reason for a human review gate between implement and verify.

The gate is wired as a **node** in the LangGraph StateGraph, not a conditional
edge function. This is critical: when the gate fails, it must write
``status = "needs_review"`` and a feedback entry to state so the dispatcher
can detect the human-review condition. A conditional edge function cannot
return state updates in LangGraph, so a pure-routing gate would silently
end the workflow with ``status = "running"`` → ``"completed"``.

When the gate passes, it returns ``status = "running"`` unchanged and routes
to the next phase node. When it fails, it sets ``status = "needs_review"``,
adds a feedback entry, and routes to END.

Extended checks for tasks→implement:
- ``codebase-map.md`` must exist on disk (not just tasks.md).
- At least one file path referenced in any slice-*.md must exist in the
  workspace. If every path in every slice is missing, the tasks agent most
  likely hallucinated a fictional project skeleton and the output is useless
  for implement. Route to needs_review immediately rather than letting
  implement spend 30+ turns discovering the problem itself.

Extended checks for plan→implement:
- ``plan.json`` must exist on disk and be valid JSON.
- ``feature_slices`` array must be non-empty.
- Each feature slice must contain the required fields: id, title,
  target_files, execution_requirements, dependencies, acceptance_criteria.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)

# ── Minimum artifact length (characters) to count as meaningful ────────
MIN_ARTIFACT_CHARS = 50


def _has_meaningful_artifacts(state: WorkflowState, required_phase: str) -> bool:
    """Return True if the required phase produced non-trivial artifacts."""
    artifacts = state.get("artifacts", {})
    phase_arts = artifacts.get(required_phase, {})

    if not isinstance(phase_arts, dict):
        return False

    for _name, content in phase_arts.items():
        if content is not None and len(str(content).strip()) >= MIN_ARTIFACT_CHARS:
            return True

    return False


def _check_tasks_quality(
    workspace_root: str,
    work_id: str,
) -> tuple[bool, str]:
    """Validate tasks-phase artifact quality beyond mere presence.

    Performs two additional checks specific to the tasks→implement gate:

    1. **codebase-map.md existence** — Without it the implement orchestrator
       has no pre-built context and falls back to a 20+ turn exploration spiral.

    2. **Slice path grounding** — Parses every ``slice-*.md`` file and extracts
       paths listed under "Files to Modify/Create" or "Files to Modify" headings.
       If at least one extracted path exists in the workspace the check passes.
       If *no* paths exist the tasks agent almost certainly hallucinated a
       generic project skeleton (e.g. ``src/main.py``, ``api/routes.py``) rather
       than grounding in the real codebase.

    Returns:
        ``(passed, reason)`` — ``passed=True`` means gate should proceed;
        ``reason`` is a human-readable explanation on failure (empty on pass).
    """
    root = Path(workspace_root)
    tasks_dir = root / ".spine" / "artifacts" / work_id / "tasks"

    if not tasks_dir.is_dir():
        # Tasks dir doesn't exist — let the main gate handle this
        return True, ""

    # ── Check 1: codebase-map.md must exist ─────────────────────────────
    codebase_map = tasks_dir / "codebase-map.md"
    if not codebase_map.exists():
        return (
            False,
            "tasks phase did not produce codebase-map.md. "
            "The IMPLEMENT orchestrator needs this file to dispatch subagents "
            "without entering a workspace exploration spiral. "
            "Re-run TASKS to produce a proper codebase map.",
        )

    # ── Check 2: slice file paths must be grounded in the workspace ──────
    slice_files = sorted(tasks_dir.glob("slice-*.md"))
    if not slice_files:
        # No slices — let main gate catch "no meaningful artifacts"
        return True, ""

    all_candidate_paths: list[str] = []
    # Patterns that introduce file paths in slice files:
    #   - "- `path/to/file.py`"  (backtick)
    #   - "- path/to/file.py"    (bare)
    #   - "Files to Modify:\n- path"
    _path_re = re.compile(
        r"[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)[`'\"]?",
    )

    for sf in slice_files:
        try:
            text = sf.read_text(encoding="utf-8")
        except OSError:
            continue
        # Collect paths from lines that look like file listings
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("-") and ":" not in line:
                continue
            for match in _path_re.finditer(line):
                candidate = match.group(1)
                # Skip very short tokens and obvious non-paths
                if len(candidate) < 4 or candidate.startswith("."):
                    continue
                # Must contain a directory separator or look like a real file
                if "/" in candidate or candidate.count(".") == 1:
                    all_candidate_paths.append(candidate)

    if not all_candidate_paths:
        # Could not extract any paths — pass (no false-positive gate)
        return True, ""

    # Check whether at least one candidate path exists in workspace
    found_any = False
    for candidate in all_candidate_paths:
        # Try both the path as-is and without leading slash
        for try_path in (candidate, candidate.lstrip("/")):
            if (root / try_path).exists():
                found_any = True
                break
        if found_any:
            break

    if not found_any:
        sampled = all_candidate_paths[:5]
        return (
            False,
            f"tasks phase slice files reference paths that do not exist in the "
            f"workspace (sampled: {sampled}). "
            "This indicates the tasks agent produced a hallucinated project "
            "skeleton rather than grounding slices in the actual codebase. "
            "Re-run TASKS — the agent must verify that file paths exist before "
            "writing slice artifacts.",
        )

    return True, ""


# ── Required fields for each feature slice in plan.json ─────────────────
_REQUIRED_SLICE_FIELDS: set[str] = {
    "id",
    "title",
    "target_files",
    "execution_requirements",
    "dependencies",
    "acceptance_criteria",
}


def _check_plan_quality(
    workspace_root: str,
    work_id: str,
) -> tuple[bool, str]:
    """Validate plan-phase artifact quality beyond mere presence.

    Performs additional checks specific to the plan→implement gate:

    1. **plan.json existence** — Without it the implement orchestrator has
       no structured decomposition to drive slice-based implementation.

    2. **Non-empty feature_slices** — The plan must contain at least one
       feature slice for implement to execute.

    3. **Slice field completeness** — Each feature slice must declare all
       required fields (id, title, target_files, execution_requirements,
       dependencies, acceptance_criteria) so implement can dispatch subagents
       with sufficient context.

    Returns:
        ``(passed, reason)`` — ``passed=True`` means gate should proceed;
        ``reason`` is a human-readable explanation on failure (empty on pass).
    """
    root = Path(workspace_root)
    plan_dir = root / ".spine" / "artifacts" / work_id / "plan"

    if not plan_dir.is_dir():
        # Plan dir doesn't exist — let the main gate handle this
        return True, ""

    # ── Check 1: plan.json must exist ─────────────────────────────────────
    plan_json_path = plan_dir / "plan.json"
    if not plan_json_path.exists():
        return (
            False,
            "plan phase did not produce plan.json. "
            "The IMPLEMENT orchestrator needs a structured plan with "
            "feature_slices to dispatch subagents. "
            "Re-run PLAN to produce a proper plan.json.",
        )

    # ── Check 2: plan.json must be valid JSON ─────────────────────────────
    try:
        raw = plan_json_path.read_text(encoding="utf-8")
        plan_data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            False,
            f"plan.json is not valid JSON: {exc}. Re-run PLAN to produce a well-formed plan.json.",
        )
    except OSError as exc:
        return (
            False,
            f"Could not read plan.json: {exc}. Re-run PLAN to produce a readable plan.json.",
        )

    if not isinstance(plan_data, dict):
        return (
            False,
            f"plan.json top-level value must be a JSON object, "
            f"got {type(plan_data).__name__}. "
            "Re-run PLAN to produce a properly structured plan.json.",
        )

    # ── Check 3: feature_slices must be non-empty ─────────────────────────
    feature_slices = plan_data.get("feature_slices")
    if feature_slices is None:
        return (
            False,
            "plan.json is missing the 'feature_slices' key. "
            "The IMPLEMENT orchestrator requires a feature_slices array "
            "to decompose work into executable slices. "
            "Re-run PLAN to produce a plan with feature_slices.",
        )

    if not isinstance(feature_slices, list):
        return (
            False,
            f"plan.json 'feature_slices' must be an array, "
            f"got {type(feature_slices).__name__}. "
            "Re-run PLAN to produce a plan with a proper feature_slices array.",
        )

    if len(feature_slices) == 0:
        return (
            False,
            "plan.json 'feature_slices' is empty. "
            "The IMPLEMENT orchestrator requires at least one feature slice "
            "to execute. Re-run PLAN to produce a plan with feature slices.",
        )

    # ── Check 4: each slice must have required fields ─────────────────────
    for idx, slice_data in enumerate(feature_slices):
        if not isinstance(slice_data, dict):
            return (
                False,
                f"plan.json feature_slices[{idx}] must be a JSON object, "
                f"got {type(slice_data).__name__}. "
                "Re-run PLAN to produce properly structured feature slices.",
            )

        missing = _REQUIRED_SLICE_FIELDS - set(slice_data.keys())
        if missing:
            slice_id = slice_data.get("id", f"index {idx}")
            return (
                False,
                f"plan.json feature slice '{slice_id}' (index {idx}) is "
                f"missing required fields: {sorted(missing)}. "
                "Each feature slice must declare: id, title, target_files, "
                "execution_requirements, dependencies, acceptance_criteria. "
                "Re-run PLAN to produce complete feature slices.",
            )

    return True, ""


def make_artifact_gate_node(required_phase: str, next_node: str) -> Any:
    """Create an artifact gate node function for the workflow graph.

    The returned function has the LangGraph node signature
    ``(state, config) -> partial_state_update``. It checks whether
    ``required_phase`` produced meaningful artifacts:

    - **Pass**: returns ``{"status": "running"}`` (unchanged) so the
      conditional edge routes to ``next_node``.
    - **Fail**: returns ``{"status": "needs_review", "feedback": [...]}``
      so the conditional edge routes to END and the dispatcher detects
      the human-review condition.

    Args:
        required_phase: The phase that must have artifacts (e.g. ``"implement"``).
        next_node: The target node if the gate passes (used for the edge map).

    Returns:
        A node function suitable for ``graph.add_node()``.
    """

    def gate_node(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
        work_id = state.get("work_id", "unknown")
        workspace_root = state.get("workspace_root", ".")

        # Check state-level artifacts first (fast, no I/O)
        has_state_artifacts = _has_meaningful_artifacts(state, required_phase)

        # Also validate on-disk presence — state may have a summary but the
        # agent may not have actually written the files, or they may have
        # been written to the wrong path.  Disk is the ground truth.
        has_disk_artifacts = False
        try:
            from spine.agents.artifacts import validate_artifact_dir

            has_disk_artifacts = validate_artifact_dir(workspace_root, work_id, required_phase)
        except Exception:
            # Don't let a disk check failure crash the gate — fall back
            # to the state check result.
            has_disk_artifacts = True

        # Pass if state has artifacts.  Log a warning if disk check fails
        # but don't block — disk may be empty because the dispatcher hasn't
        # persisted yet, or because the workspace_root is different.
        if has_state_artifacts:
            if not has_disk_artifacts:
                logger.warning(
                    "[%s] Artifact gate: %s has artifacts in state but not on "
                    "disk at .spine/artifacts/%s/%s/. Proceeding anyway.",
                    work_id,
                    required_phase,
                    work_id,
                    required_phase,
                )

            # ── Extended quality checks for tasks→implement ──────────────
            # Run these only when the basic presence check passes — we don't
            # want to add I/O overhead to gates that already failed.
            if PhaseName(required_phase) == PhaseName.TASKS:
                try:
                    quality_ok, quality_reason = _check_tasks_quality(workspace_root, work_id)
                except Exception as exc:
                    # Quality check errors must never crash the gate
                    logger.warning(
                        "[%s] Tasks quality check raised unexpectedly: %s (proceeding anyway)",
                        work_id,
                        exc,
                    )
                    quality_ok = True
                    quality_reason = ""

                if not quality_ok:
                    reason = f"Artifact gate (quality): {quality_reason}"
                    logger.warning("[%s] %s Flagging for human review.", work_id, reason)
                    return {
                        "current_phase": required_phase,
                        "status": "needs_review",
                        "feedback": [
                            {
                                "status": "needs_review",
                                "tier": "structural",
                                "reason": reason,
                                "suggestions": [],
                            }
                        ],
                        "prompt_request": None,
                    }

            # ── Extended quality checks for plan→implement ───────────────
            # Validate plan.json has structured feature_slices that implement
            # can use to dispatch subagents.
            if PhaseName(required_phase) == PhaseName.PLAN:
                try:
                    quality_ok, quality_reason = _check_plan_quality(workspace_root, work_id)
                except Exception as exc:
                    # Quality check errors must never crash the gate
                    logger.warning(
                        "[%s] Plan quality check raised unexpectedly: %s (proceeding anyway)",
                        work_id,
                        exc,
                    )
                    quality_ok = True
                    quality_reason = ""

                if not quality_ok:
                    reason = f"Artifact gate (quality): {quality_reason}"
                    logger.warning("[%s] %s Flagging for human review.", work_id, reason)
                    return {
                        "current_phase": required_phase,
                        "status": "needs_review",
                        "feedback": [
                            {
                                "status": "needs_review",
                                "tier": "structural",
                                "reason": reason,
                                "suggestions": [],
                            }
                        ],
                        "prompt_request": None,
                    }

            logger.debug(
                "[%s] Artifact gate passed for %s → %s",
                work_id,
                required_phase,
                next_node,
            )
            return {
                "current_phase": required_phase,
                "status": "running",
                "prompt_request": None,
            }

        # Provide a specific reason for the failure
        reason = (
            f"Artifact gate: {required_phase} produced no "
            f"meaningful artifacts (≥{MIN_ARTIFACT_CHARS} chars), "
            f"cannot proceed to {next_node}."
        )

        logger.warning(
            "[%s] %s Flagging for human review.",
            work_id,
            reason,
        )
        return {
            "current_phase": required_phase,
            "status": "needs_review",
            "feedback": [
                {
                    "status": "needs_review",
                    "tier": "structural",
                    "reason": reason,
                    "suggestions": [],
                }
            ],
            "prompt_request": None,
        }

    # Give the function a readable name for LangGraph Studio / debug
    gate_node.__name__ = f"gate_{required_phase}_to_{next_node}"
    return gate_node


def artifact_gate_router(state: WorkflowState) -> str:
    """Route based on the status set by the gate node.

    Intended as a conditional edge function after a gate node.
    Reads ``state["status"]``: ``"running"`` → proceed, anything else → END.
    """
    if state.get("status") == "running":
        return "proceed"
    return "needs_review"


# ── Prerequisite Gate Node Functions ─────────────────────────────────────
# These create LangGraph nodes that check phase completion invariants.
# When the check fails, they return status="needs_review" so the workflow
# routes to human_review instead of proceeding with empty artifacts.


def make_prerequisite_gate_node(
    check_fn: Callable[[WorkflowState], tuple[bool, str]],
    target_phase: str,
) -> Any:
    """Create a prerequisite gate node that checks a phase completion flag.

    The returned function has the LangGraph node signature
    ``(state, config) -> partial_state_update``.

    Args:
        check_fn: A function that takes WorkflowState and returns (passed, reason).
        target_phase: The phase that requires the prerequisite (for error messages).

    Returns:
        A node function suitable for ``graph.add_node()``.
    """
    phase_completion_flags = {
        "plan": "spec_completed",
        "implement": "plan_completed",
        "verify": "implement_completed",
        "gap_plan": "verification_attempted",
    }

    def gate_node(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
        work_id = state.get("work_id", "unknown")

        passed, reason = check_fn(state)

        if passed:
            logger.debug(
                "[%s] Prerequisite gate passed for %s → proceeding",
                work_id,
                target_phase,
            )
            return {
                "current_phase": target_phase,
                "status": "running",
                "prompt_request": None,
            }

        # Gate failed — flag for human review
        logger.warning(
            "[%s] Prerequisite gate FAILED for %s: %s",
            work_id,
            target_phase,
            reason,
        )
        return {
            "current_phase": target_phase,
            "status": "needs_review",
            "feedback": [
                {
                    "status": "needs_review",
                    "tier": "structural",
                    "reason": f"Prerequisite gate for {target_phase}: {reason}",
                    "suggestions": [],
                }
            ],
            "prompt_request": None,
        }

    # Give the function a readable name
    gate_node.__name__ = f"gate_prereq_{target_phase}"
    return gate_node


# ── Phase Prerequisite Check Functions ──────────────────────────────────
# These functions implement "fail-closed" gates that verify a preceding phase
# completed successfully before allowing the next phase to run. They check
# the state invariants (completion flags) that are set when phases complete.


def _check_spec_prerequisite(state: WorkflowState) -> tuple[bool, str]:
    """Check that SPECIFY phase completed before PLAN runs.

    PLAN requires a valid specification artifact. This check ensures
    the spec_completed flag is True, preventing PLAN from running on
    empty or failed SPECIFY output.

    Returns:
        ``(passed, reason)`` — ``passed=True`` means PLAN should proceed;
        ``reason`` is a human-readable explanation on failure.
    """
    spec_completed = state.get("spec_completed", False)

    if spec_completed:
        logger.debug("plan prerequisite: spec_completed=True, proceeding")
        return True, ""

    return (
        False,
        "PLAN phase requires SPECIFY to have completed successfully. "
        "The specification artifact is missing or the SPECIFY phase did not finish. "
        "Re-run SPECIFY or resolve the prior failure before proceeding.",
    )


def _check_plan_prerequisite(state: WorkflowState) -> tuple[bool, str]:
    """Check that PLAN phase completed before IMPLEMENT runs.

    IMPLEMENT requires a valid plan with feature_slices AND structured
    ``execution_waves``. This check ensures both the ``plan_completed`` flag
    is True AND ``execution_waves`` is non-empty, preventing IMPLEMENT from
    running on empty or failed PLAN output.

    This is a **fail-closed** gate: if either invariant is missing, the
    workflow routes to ``needs_review`` rather than attempting implementation
    with incomplete plan data.

    Returns:
        ``(passed, reason)`` — ``passed=True`` means IMPLEMENT should proceed;
        ``reason`` is a human-readable explanation on failure.
    """
    plan_completed = state.get("plan_completed", False)
    execution_waves = state.get("execution_waves", [])

    if not plan_completed:
        return (
            False,
            "IMPLEMENT phase requires PLAN to have completed successfully. "
            "The plan artifact is missing or the PLAN phase did not finish. "
            "Re-run PLAN or resolve the prior failure before proceeding.",
        )

    if not execution_waves or len(execution_waves) == 0:
        return (
            False,
            "IMPLEMENT phase requires structured execution_waves from the PLAN phase. "
            "The PLAN completed but did not produce execution_waves — this is a "
            "scheduler failure. Re-run PLAN to regenerate execution_waves from "
            "the plan.json feature_slices.",
        )

    logger.debug("implement prerequisite: plan_completed=True, execution_waves present, proceeding")
    return True, ""


def _check_implement_prerequisite(state: WorkflowState) -> tuple[bool, str]:
    """Check that IMPLEMENT phase completed before VERIFY runs.

    VERIFY requires implementation artifacts to validate. This check ensures
    the implement_completed flag is True, preventing VERIFY from running on
    empty or failed IMPLEMENT output.

    Returns:
        ``(passed, reason)`` — ``passed=True`` means VERIFY should proceed;
        ``reason`` is a human-readable explanation on failure.
    """
    implement_completed = state.get("implement_completed", False)

    if implement_completed:
        logger.debug("verify prerequisite: implement_completed=True, proceeding")
        return True, ""

    return (
        False,
        "VERIFY phase requires IMPLEMENT to have completed successfully. "
        "The implementation artifact is missing or the IMPLEMENT phase did not finish. "
        "Re-run IMPLEMENT or resolve the prior failure before proceeding.",
    )


def _check_verify_prerequisite(state: WorkflowState) -> tuple[bool, str]:
    """Check that VERIFY attempted (not necessarily passed) before GAP_PLAN runs.

    GAP_PLAN requires the verification report (whether passed or failed) to
    determine what gaps need remediation. This check ensures the
    verification_attempted flag is True.

    Returns:
        ``(passed, reason)`` — ``passed=True`` means GAP_PLAN should proceed;
        ``reason`` is a human-readable explanation on failure.
    """
    verification_attempted = state.get("verification_attempted", False)

    if verification_attempted:
        logger.debug("gap_plan prerequisite: verification_attempted=True, proceeding")
        return True, ""

    return (
        False,
        "GAP_PLAN phase requires VERIFY to have run at least once. "
        "The verification artifact is missing or VERIFY did not execute. "
        "Run VERIFY before GAP_PLAN can produce a remediation plan.",
    )



