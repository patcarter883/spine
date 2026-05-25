"""TASKS phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the tasks Deep Agent.
2. ``save_artifacts`` — scans disk for artifacts, handles retry logic.

State schema: ``TasksSubgraphState`` — isolated from parent ``WorkflowState``.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import TasksSubgraphState
from spine.agents.tasks_agent import build_tasks_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    artifact_path,
)

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


async def _run_tasks_agent(
    state: TasksSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the tasks Deep Agent within the subgraph.

    For quick/critical_quick workflows, TASKS is the first phase so the
    original description is included directly.  For spec/critical_spec
    workflows, TASKS works from the specification and plan artifacts on
    disk — the raw description is NOT included.
    """
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")
    retry_count = state.get("retry_count", 0)
    feedback = state.get("feedback", [])

    logger.info(f"[{work_id}] TASKS subgraph: run_agent starting")

    try:
        agent = build_tasks_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        has_spec = "spec" in work_type
        spec_path = artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = artifact_path(work_id, PhaseName.PLAN.value)
        tasks_artifact_dir = artifact_path(work_id, PhaseName.TASKS.value)

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
                    "Call `read_prior_artifacts` first (no arguments) to load the specification and plan. Then call `search_codebase` or use MCP tools to research change sites.",
                    "",
                ]
            )
        else:
            # Quick/critical_quick: no prior artifacts — work from the
            # original description. First action must be researcher dispatch
            # via the task tool — the user message makes this explicit.
            desc_preview = description[:120].replace("'", "\\'")
            researcher_line1 = f"    description: 'Research code relevant to: {desc_preview}\\nInvestigate: <area 1>'}},"
            researcher_line2 = f"    description: 'Research code relevant to: {desc_preview}\\nInvestigate: <area 2>'}},"
            prompt_lines.extend(
                [
                    "Break the work description into smaller, executable "
                    "feature slices with clear dependencies.",
                    "",
                    "## Work Description",
                    description,
                    "",
                    "## Your first action MUST be to dispatch researcher subagents",
                    "Call `task` 2-3 times (once per area) to dispatch researcher subagents "
                    "in parallel. Do NOT call codebase queries search/explore sequentially "
                    "turn-by-turn yourself first. Each researcher investigates ONE area of "
                    "the codebase relevant to the work description above.",
                    "",
                    "Example (adapt to the actual work description):",
                    f"  tools.task({{subagent_type: 'researcher', {researcher_line1}}})",
                    f"  tools.task({{subagent_type: 'researcher', {researcher_line2}}})",
                    "",
                    "After researchers complete, call `write_tasks_artifacts` "
                    "with all slices, tasks summary, dependency waves, and codebase map. "
                    "Total turns: ~2-3.",
                    "",
                ]
            )

        prompt_lines.extend(
            [
                f"## Artifact Output Directory\n"
                f"Write ALL artifact files (slice files AND tasks.md) to: `{tasks_artifact_dir}/`\n"
                f"This is relative to your workspace root (`{workspace_root}`).\n"
                f"Full path: `{workspace_root}/{tasks_artifact_dir}/`\n",
            ]
        )
        prompt_lines.extend(
            [
                "## Instructions\\n"
                "1. Explore the codebase (dispatch researcher subagents via the `task` tool).\\n"
                "2. After exploring, synthesize everything and write all slices, overview, and codebase-map "
                "using `write_tasks_artifacts` once.\n"
                "3. **You MUST call `write_tasks_artifacts` exactly ONCE** — do not call `write_file`.\n"
                "4. Total phase length is ~2-3 turns. Stop immediately after `write_tasks_artifacts` returns.\n",
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

        ctx = build_context(dict(state), PhaseName.TASKS)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.TASKS.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
        }

    except Exception as e:
        logger.error(f"[{work_id}] TASKS subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "phase_status": "error",
        }


async def _save_tasks_artifacts(
    state: TasksSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the tasks agent to disk and state."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")
    work_type = state.get("work_type", "")

    if existing_phase_status in ("error", "needs_review"):
        return {
            "artifacts_output": {},
            "phase_status": existing_phase_status,
        }

    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        PhaseName.TASKS.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    # ── Fallback: if agent wrote nothing, try extract_response ─────────
    if not disk_artifacts:
        tasks_content = agent_response
        if tasks_content.strip():
            materialize_phase_artifacts(
                PhaseName.TASKS.value,
                {"tasks.md": tasks_content},
                workspace_root,
                work_id=work_id,
            )
            disk_artifacts = {"tasks.md": tasks_content[:_MAX_ARTIFACT_STATE_CHARS]}

    # ── Retry with fresh agent if still nothing ───────────────────────────
    if not disk_artifacts:
        logger.warning(
            "[%s] TASKS agent produced no output — retrying with fresh agent",
            work_id,
        )
        retry_agent = build_tasks_agent(dict(state), config)
        ctx = build_context(dict(state), PhaseName.TASKS)
        retry_prompt = (
            "**Write the task decomposition NOW.** "
            "Do NOT read any more files. Produce:\n\n"
            "1. A `tasks.md` summary listing 2-5 feature slices.\n"
            "2. Individual `slice-<name>.md` files for each slice.\n\n"
            "Output format in tasks.md:\n"
            "```markdown\n"
            "## Slice 1: <name>\n"
            "- **Files:** ...\n"
            "- **Depends on:** none\n"
            "- **Complexity:** small\n"
            "- **Acceptance:** ...\n"
            "```\n"
        )
        retry_result = await ainvoke_with_retry(
            retry_agent,
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
                "[%s] TASKS subgraph: no artifacts produced even with retry",
                work_id,
            )
            return {
                "artifacts_output": {},
                "phase_status": "needs_review",
            }

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success",
    }


def build_tasks_subgraph() -> Any:
    """Build the TASKS phase subgraph."""
    builder = StateGraph(TasksSubgraphState)
    builder.add_node("run_agent", _run_tasks_agent)
    builder.add_node("save_artifacts", _save_tasks_artifacts)
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)
    return builder
