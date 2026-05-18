"""SPINE TASKS phase — break the plan into executable feature slices.

This is where decomposition occurs. The tasks Deep Agent reads the plan
(on disk, not inlined) and breaks it into smaller, independent feature
slices that can be implemented in parallel or sequentially.

Phase node functions are async to avoid event-loop binding errors when
subagents inherit the parent checkpointer.
"""

from __future__ import annotations

import logging
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

    Delegates to the tasks Deep Agent, which decomposes the plan into
    feature slices with dependencies. If reworking, includes prior feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with task artifacts.
    """
    description = state.get("description", "")  # noqa: F841
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

        # ── Compute the exact artifact output path ──
        # The agent needs the full work_id-scoped path to write slice files.
        tasks_artifact_dir = _artifact_path(work_id, PhaseName.TASKS.value)

        prompt_lines = []
        if has_spec:
            # Spec/critical_spec: work from spec + plan artifacts, not the
            # original description (already captured and expanded in them).
            prompt_lines.extend(
                [
                    "Break the plan into smaller, executable feature slices "
                    "with clear dependencies.",
                    "",
                    "Prior artifacts are available on disk:",
                    f"- Specification: `{spec_path}/specification.md`",
                    f"- Plan: `{plan_path}/plan.md`",
                    "",
                    "Read them with `read_file` before decomposing.",
                    "",
                ]
            )
        else:
            # Quick/critical_quick: no prior artifacts — work from the
            # original description and explore the codebase.
            prompt_lines.extend(
                [
                    "Break the work description into smaller, executable "
                    "feature slices with clear dependencies.",
                    "",
                    "## Work Description",
                    description,
                    "",
                    "Use researcher subagents via the interpreter (`eval`) to "
                    "explore the codebase in parallel — your system prompt has "
                    "the detailed strategy. Focus 2-3 researchers on the modules "
                    "most relevant to this task, then synthesize into slices.",
                    "",
                    "Spend at most 2-3 turns on exploration before writing. "
                    "Better to produce slices with partial knowledge than none at all.",
                    "",
                ]
            )

        prompt_lines.extend(
            [
                f"## Artifact Output Directory\n"
                f"Write ALL artifact files (slice files AND tasks.md) to: `{tasks_artifact_dir}/`\n"
                f"Use this relative path with `write_file` — do NOT construct absolute paths.\n",
            ]
        )
        prompt_lines.extend(
            [
                f"## Instructions\n"
                f"1. Explore the codebase (use researcher subagents via the interpreter\n"
                f"   for parallel exploration, or `read_file`/`grep` for quick checks).\n"
                f"2. After exploring, write individual `slice-<name>.md` files using\n"
                f"   `write_file` to `{tasks_artifact_dir}/slice-<name>.md`.\n"
                f"3. Write a summary `tasks.md` to `{tasks_artifact_dir}/tasks.md`\n"
                f"   that references each slice.\n"
                f"4. **You MUST call `write_file`** — do not just describe the slices\n"
                f"   in conversation. Write them to disk.\n"
                f"5. Write a `codebase-map.md` to `{tasks_artifact_dir}/codebase-map.md` that captures your exploration findings:\n"
                f"   - File paths with descriptions (what each file does)\n"
                f"   - Key classes and functions (names, signatures)\n"
                f"   - Import chains between relevant modules\n"
                f"   - Conventions discovered (naming, patterns, error handling)\n"
                f"   This map will be read by the implement and verify phases — it saves them from re-exploring the codebase.\n",
            ]
        )
        prompt = "\n".join(prompt_lines)
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
