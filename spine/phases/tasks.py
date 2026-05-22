"""SPINE TASKS phase — break the plan into executable feature slices.

.. deprecated::
    The TASKS phase is deprecated. Use the PLAN phase for decomposition
    instead. This module is retained for backward compatibility and will
    be removed in a future release.

This is where decomposition occurs. The tasks Deep Agent reads the plan
(on disk, not inlined) and breaks it into smaller, independent feature
slices that can be implemented in parallel or sequentially.

Phase node functions are async to avoid event-loop binding errors when
subagents inherit the parent checkpointer.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.tasks_agent import build_tasks_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    scan_artifact_dir,
    materialize_artifacts,
    materialize_phase_artifacts,
    _artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)

# Maximum characters of artifact content to store in WorkflowState.
# Full content lives on disk via scan_artifact_dir(). Keeping
# state compact prevents ~260K tokens of artifact bloat across turns.
_MAX_ARTIFACT_STATE_CHARS = 500


async def call_tasks(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the TASKS phase.

    .. deprecated::
        The TASKS phase is deprecated. Use the PLAN phase for decomposition
        instead. This function is retained for backward compatibility.

    Delegates to the tasks Deep Agent, which decomposes the plan into
    feature slices with dependencies. If reworking, includes prior feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with task artifacts.
    """
    warnings.warn(
        "The TASKS phase (call_tasks) is deprecated. Use the PLAN phase for decomposition instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # TASKS is the first phase in quick workflows (no prior spec/plan exist),
    # so the original description is used directly for quick/critical_quick.
    # For spec/critical_spec workflows, the description is NOT used — TASKS
    # works from the specification and plan artifacts on disk instead.
    description = state.get("description", "")  # noqa: F841 — used in quick-workflow branch below
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

        # Build prompt — plan and spec are loaded via read_prior_artifacts tool.
        # Quick workflows get a researcher dispatch imperative;
        # spec workflows get a simpler "load artifacts then decompose" message.
        has_spec = "spec" in work_type  # only spec/critical_spec produce specify+plan

        # ── Compute the exact artifact output path ──
        # Used in the completion reminder in the user message.
        tasks_artifact_dir = _artifact_path(work_id, PhaseName.TASKS.value)
        tasks_dir = f".spine/artifacts/{work_id}/tasks"
        context_seed = f"globalThis.context = {{work_id: '{work_id}', phase: 'tasks', tasks_dir: '{tasks_dir}'}};\n\n"

        prompt_lines = []
        if retry_count > 0:
            prompt_lines.append(
                "⚠ **REWORK PASS**: Your primary objective is to revise the prior tasks decomposition. Address all points from the critic feedback.\n\n"
            )

        if has_spec:
            prompt_lines.extend(
                [
                    "Call `read_prior_artifacts` first (no arguments) to load "
                    "the specification and plan. Then call `search_codebase` "
                    "to find modification targets. Then call `write_tasks_artifacts`.",
                    "",
                ]
            )
        else:
            # Quick/critical_quick: no prior artifacts — work from the
            # original description. First action must be parallel researcher
            # dispatch via eval — the user message makes this explicit.
            desc_preview = description[:120].replace("'", "\\'")
            researcher_line1 = f"    description: 'Research code relevant to: {desc_preview}\\nInvestigate: <area 1>'}}),"
            researcher_line2 = f"    description: 'Research code relevant to: {desc_preview}\\nInvestigate: <area 2>'}}),"
            prompt_lines.extend(
                [
                    "## Work Description",
                    description,
                    "",
                    "## Your first action MUST be an eval call",
                    "Dispatch 2-3 `researcher` subagents in parallel via "
                    "`Promise.allSettled` inside a single `eval` call. "
                    "Do NOT call `search_codebase` or explore yourself first. "
                    "Each researcher investigates ONE area of the codebase "
                    "relevant to the work description above.",
                    "",
                    "```js",
                    "// FIRST TURN — dispatch researchers in parallel:",
                    "const results = await Promise.allSettled([",
                    "  tools.task({subagent_type: 'researcher',",
                    researcher_line1,
                    "  tools.task({subagent_type: 'researcher',",
                    researcher_line2,
                    "]);",
                    "globalThis.research = results.map(r => r.value || r.reason);",
                    "```",
                    "",
                    "After researchers complete, call `write_tasks_artifacts` "
                    "with all slices, tasks summary, dependency waves, and codebase map. "
                    "Total turns: ~2-3.",
                    "",
                ]
            )

        prompt_lines.extend(
            [
                f"## Completion",
                f"When research is done, call `write_tasks_artifacts` once with all slices, "
                f"the overview, dependency waves, and the full codebase map. "
                f"Artifacts are written to `{tasks_artifact_dir}/` automatically — "
                f"you do not need to specify paths. After it returns, STOP.",
                "",
            ]
        )
        prompt = context_seed + "\n".join(prompt_lines)
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(state, PhaseName.TASKS)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.TASKS.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        # ── Collect artifacts from disk (agent writes via write_file) ───────
        # The authoritative artifacts are the files the agent wrote to disk,
        # NOT the extracted LLM response.  For thinking models (e.g.
        # DeepSeek-v4-flash), the last message content is chain-of-thought
        # reasoning that should NOT become an artifact.
        disk_artifacts = scan_artifact_dir(
            workspace_root,
            work_id,
            PhaseName.TASKS.value,
            max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
        )

        # ── Fallback: if agent wrote nothing, try extract_response ────────
        # This handles agents that produce output in conversation but never
        # called write_file (e.g. non-thinking models on simple tasks).
        if not disk_artifacts:
            tasks_content = extract_response(result)
            if tasks_content.strip():
                # Materialize to disk so later phases can read it
                materialize_phase_artifacts(
                    PhaseName.TASKS.value,
                    {"tasks.md": tasks_content},
                    workspace_root,
                    work_id=work_id,
                )
                disk_artifacts = {"tasks.md": tasks_content[:_MAX_ARTIFACT_STATE_CHARS]}

        # ── Detect empty response (agent explored but never wrote) ─────
        if not disk_artifacts:
            logger.warning(
                "[%s] TASKS agent produced no output and no files — "
                "retrying with direct output prompt",
                work_id,
            )
            retry_prompt = (
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
            retry_result = await ainvoke_with_retry(
                agent,
                {"messages": [{"role": "user", "content": retry_prompt}]},
                phase_name=PhaseName.TASKS.value,
                work_id=work_id,
                work_type=work_type,
                context=ctx,
            )
            # Scan disk again after retry
            disk_artifacts = scan_artifact_dir(
                workspace_root,
                work_id,
                PhaseName.TASKS.value,
                max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
            )

            # If still nothing, try extract_response from retry
            if not disk_artifacts:
                tasks_content = extract_response(retry_result)
                if tasks_content.strip():
                    materialize_phase_artifacts(
                        PhaseName.TASKS.value,
                        {"tasks.md": tasks_content},
                        workspace_root,
                        work_id=work_id,
                    )
                    disk_artifacts = {"tasks.md": tasks_content[:_MAX_ARTIFACT_STATE_CHARS]}

            if not disk_artifacts:
                logger.error(
                    "[%s] TASKS agent produced no output on retry either — "
                    "flagging for human review",
                    work_id,
                )
                return {
                    "artifacts": {PhaseName.TASKS.value: {}},
                    "current_phase": PhaseName.TASKS.value,
                    "status": "needs_review",
                    "feedback": [
                        {
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
                        }
                    ],
                    "prompt_request": None,
                }

        return {
            "artifacts": {PhaseName.TASKS.value: disk_artifacts},
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
