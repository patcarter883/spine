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

# Maximum characters of artifact content to store in WorkflowState.
# Full content lives on disk via materialize_phase_artifacts(). Keeping
# state compact prevents ~260K tokens of artifact bloat across turns.
_MAX_ARTIFACT_STATE_CHARS = 500


def _collect_slice_files(
    workspace_root: str,
    work_id: str,
) -> dict[str, str]:
    """Read slice files and return truncated previews for state.

    Full content is already on disk — state only needs enough to show
    in artifact prompts and satisfy the artifact gate threshold (≥50 chars).

    Args:
        workspace_root: Absolute path to the project workspace.
        work_id: Work item identifier for path scoping.

    Returns:
        Dict of ``{filename: truncated_preview}`` for each discovered slice file.
    """
    tasks_dir = Path(workspace_root) / _artifact_path(work_id, PhaseName.TASKS.value)
    if not tasks_dir.is_dir():
        return {}
    slices: dict[str, str] = {}
    for path in sorted(tasks_dir.glob(_SLICE_PATTERN)):
        try:
            content = path.read_text(encoding="utf-8")
            slices[path.name] = content[:_MAX_ARTIFACT_STATE_CHARS]
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
    work_type = state.get("work_type", "")
    retry_count = state.get("retry_count", {}).get(PhaseName.TASKS.value, 0)
    feedback = state.get("feedback", [])
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] TASKS phase starting (retry={retry_count})")

    try:
        agent = build_tasks_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build prompt — plan and spec are on disk (work_id-scoped paths).
        # Skip spec/plan references for quick workflows that lack them.
        has_spec = "spec" in work_type  # only spec/critical_spec produce specify+plan
        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)

        prompt_lines = [
            "Break the plan into smaller, executable feature slices "
            "with clear dependencies.",
            "",
            "## Work Description",
            description,
            "",
        ]
        if has_spec:
            prompt_lines.extend([
                "Prior artifacts are available on disk:",
                f"- Specification: `{spec_path}/specification.md`",
                f"- Plan: `{plan_path}/plan.md`",
                "",
                "Read them with `read_file` before decomposing.",
                "",
            ])
        else:
            prompt_lines.extend([
                "This is a quick workflow — no specification or plan artifacts "
                "exist. Work from the task description directly.",
                "",
                "Use researcher subagents via the interpreter (`eval`) to "
                "explore the codebase in parallel — your system prompt has "
                "the detailed strategy. Focus 2-3 researchers on the modules "
                "most relevant to this task, then synthesize into slices.",
                "",
                "Spend at most 2-3 turns on exploration before writing. "
                "Better to produce slices with partial knowledge than none at all.",
                "",
            ])
        prompt_lines.extend([
            "Write each slice as a separate file named `slice-<name>.md` "
            "in the tasks artifact directory, then produce a summary "
            "`tasks.md` that references them.",
            "",
        ])
        prompt = "\n".join(prompt_lines)
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

        # ── Detect empty response (agent explored but never wrote) ─────
        # The agent may read files and reason about the codebase but never
        # produce output — especially with thinking models that exhaust
        # their reasoning budget mid-analysis.  Retry once with a direct,
        # output-focused prompt that skips exploration altogether.
        if not tasks_content.strip() and not slice_files:
            logger.warning(
                "[%s] TASKS agent produced no output and no slice files — "
                "retrying with direct output prompt",
                work_id,
            )
            retry_prompt = (
                f"## Work Description\n{description}\n\n"
                "**Write the task decomposition NOW.** "
                "Do NOT read any more files — you have already gathered "
                "enough context. Produce:\n\n"
                "1. A `tasks.md` summary listing 2-5 feature slices with "
                "dependencies, files to modify, and acceptance criteria.\n"
                "2. Individual `slice-<name>.md` files for each slice.\n\n"
                "Output format in tasks.md:\n"
                "```markdown\n"
                "## Slice 1: <name>\n"
                "- **Files:** ...\n"
                "- **Depends on:** none\n"
                "- **Complexity:** small\n"
                "- **Acceptance:** ...\n```\n"
            )
            retry_result = invoke_with_retry(
                agent,
                {"messages": [{"role": "user", "content": retry_prompt}]},
                phase_name=PhaseName.TASKS.value,
                work_id=work_id,
                context=ctx,
            )
            tasks_content = extract_response(retry_result)
            slice_files = _collect_slice_files(workspace_root, work_id)

            if not tasks_content.strip() and not slice_files:
                logger.error(
                    "[%s] TASKS agent produced no output on retry either — "
                    "flagging for human review",
                    work_id,
                )
                return {
                    "artifacts": {PhaseName.TASKS.value: {}},
                    "current_phase": PhaseName.TASKS.value,
                    "status": "needs_review",
                    "feedback": [{
                        "status": "needs_review",
                        "tier": "structural",
                        "reason": (
                            "Tasks agent produced no output after retry. "
                            "The model may need human guidance to decompose this work."
                        ),
                        "suggestions": [
                            "Provide more specific instructions in the work description",
                            "Use a spec/critical_spec workflow type instead of quick",
                        ],
                    }],
                    "prompt_request": None,
                }

        # Materialize main artifact to disk (slice files already on disk from agent writes)
        materialize_phase_artifacts(
            PhaseName.TASKS.value,
            {"tasks.md": tasks_content},
            workspace_root,
            work_id=work_id,
        )

        # Build state artifacts with truncated previews — full content is on disk
        phase_artifacts: dict[str, str] = {
            "tasks.md": tasks_content[:_MAX_ARTIFACT_STATE_CHARS]
        }
        # Merge in slice files (existing state files preserved by reducer)
        phase_artifacts.update(slice_files)

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
