"""SPINE TASKS phase — break the plan into executable feature slices.

This is where decomposition occurs. The tasks Deep Agent reads the plan
(on disk, not inlined) and breaks it into smaller, independent feature
slices that can be implemented in parallel or sequentially.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.tasks_agent import build_tasks_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import invoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    _artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)

# ── Pattern for discovering slice files written by the agent ────────────
# After the tasks agent runs, it may have written individual slice files
# (e.g. slice-auth-middleware.md) alongside the main tasks.md.  We scan
# for these so the artifacts dict in state stays consistent with disk,
# preventing later phases from losing access to them.

_SLICE_PATTERN = "slice-*.md"


def _collect_slice_files(
    workspace_root: str,
    work_id: str,
) -> dict[str, str]:
    """Read any slice-*.md files the agent wrote to the tasks artifact dir.

    Args:
        workspace_root: Absolute path to the project workspace.
        work_id: Work item identifier for path scoping.

    Returns:
        Dict of ``{filename: content}`` for each discovered slice file.
    """
    tasks_dir = Path(workspace_root) / _artifact_path(work_id, PhaseName.TASKS.value)
    if not tasks_dir.is_dir():
        return {}
    slices: dict[str, str] = {}
    for path in sorted(tasks_dir.glob(_SLICE_PATTERN)):
        try:
            slices[path.name] = path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Could not read slice file: %s", path)
    return slices


def call_tasks(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Execute the TASKS phase.

    Delegates to the tasks Deep Agent, which decomposes the plan into
    feature slices with dependencies. If reworking, includes prior feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with task artifacts.
    """
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    retry_count = state.get("retry_count", {}).get(PhaseName.TASKS.value, 0)
    feedback = state.get("feedback", [])
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] TASKS phase starting (retry={retry_count})")

    try:
        agent = build_tasks_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build prompt — plan and spec are on disk (work_id-scoped paths)
        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        prompt = (
            f"Break the plan into smaller, executable feature slices "
            f"with clear dependencies.\n\n"
            f"## Work Description\n{description}\n\n"
            "Prior artifacts are available on disk:\n"
            f"- Specification: `{spec_path}/specification.md`\n"
            f"- Plan: `{plan_path}/plan.md`\n\n"
            "Read them with `read_file` before decomposing.\n\n"
            "Write each slice as a separate file named `slice-<name>.md` "
            "in the tasks artifact directory, then produce a summary "
            "`tasks.md` that references them.\n\n"
        )
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(state, PhaseName.TASKS)

        result = invoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.TASKS.value,
            work_id=work_id,
            context=ctx,
        )

        tasks_content = extract_response(result)

        # Collect any slice files the agent wrote to disk
        slice_files = _collect_slice_files(workspace_root, work_id)

        # Build the full artifacts dict for this phase
        phase_artifacts: dict[str, str] = {"tasks.md": tasks_content}
        # Merge in slice files (existing state files preserved by reducer)
        phase_artifacts.update(slice_files)

        # Materialize this phase's artifacts to disk immediately
        materialize_phase_artifacts(PhaseName.TASKS.value, phase_artifacts, workspace_root, work_id=work_id)

        return {
            "artifacts": {PhaseName.TASKS.value: phase_artifacts},
            "current_phase": PhaseName.TASKS.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] TASKS phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.TASKS.value: {}},
            "current_phase": PhaseName.TASKS.value,
            "status": "running",
            "prompt_request": {
                "message": f"TASKS phase failed: {e}",
                "phase": PhaseName.TASKS.value,
            },
        }


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.TASKS.value,
    call_fn=call_tasks,
    build_agent_fn=build_tasks_agent,
    description="Decompose the plan into executable feature slices",
)
