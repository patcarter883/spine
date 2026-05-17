"""IMPLEMENT phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the implement Deep Agent.
2. ``save_artifacts`` — scans disk for artifacts, determines phase status.

State schema: ``ImplementSubgraphState`` — isolated from parent ``WorkflowState``.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import ImplementSubgraphState
from spine.agents.implement_agent import build_implement_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry, MaxTokenBudgetExceeded
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    _artifact_path,
)

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


async def _run_implement_agent(
    state: ImplementSubgraphState,
    config: Any = None,
) -> dict[str, Any]:
    """Run the implement Deep Agent within the subgraph."""
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")
    retry_count = state.get("retry_count", 0)
    feedback = state.get("feedback", [])

    logger.info(f"[{work_id}] IMPLEMENT subgraph: run_agent starting")

    try:
        agent = build_implement_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        has_spec = "spec" in work_type
        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = _artifact_path(work_id, PhaseName.TASKS.value)

        prompt_lines = [
            "Implement the feature slices described below. Write clean, "
            "production-quality code for each slice.",
            "",
            "## Work Description",
            description,
            "",
        ]
        if has_spec:
            prompt_lines.extend([
                "Prior artifacts are available on disk — read them as needed:",
                f"- Specification: `{spec_path}/specification.md`",
                f"- Plan: `{plan_path}/plan.md`",
                f"- Feature Slices: `{tasks_path}/tasks.md`",
                f"- Codebase map: `{tasks_path}/codebase-map.md`",
                "",
            ])
        else:
            prompt_lines.extend([
                "Prior artifacts are available on disk — read them as needed:",
                f"- Feature Slices: `{tasks_path}/tasks.md`",
                f"- Codebase map: `{tasks_path}/codebase-map.md`",
                "",
            ])
        prompt_lines.extend([
            "Use `read_file` and `grep` to inspect them. Do NOT load "
            "everything into context at once.",
            "",
            "Read the codebase map FIRST — it contains file paths, key functions, and conventions "
            "discovered during the tasks phase. Use it instead of re-exploring the codebase.",
            "",
        ])
        prompt = "\n".join(prompt_lines)
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(dict(state), PhaseName.IMPLEMENT)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.IMPLEMENT.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
        }

    except MaxTokenBudgetExceeded as e:
        logger.error(f"[{work_id}] IMPLEMENT subgraph token budget exceeded: {e}")
        return {
            "messages": [],
            "agent_response": f"Token budget exceeded: {e}",
            "phase_status": "needs_review",
        }
    except Exception as e:
        logger.error(f"[{work_id}] IMPLEMENT subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "phase_status": "error",
        }


async def _save_implement_artifacts(
    state: ImplementSubgraphState,
    config: Any = None,
) -> dict[str, Any]:
    """Save artifacts from the implement agent to disk and state."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {
            "artifacts_output": {},
            "phase_status": existing_phase_status,
        }

    disk_artifacts = scan_artifact_dir(
        workspace_root, work_id, PhaseName.IMPLEMENT.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        materialize_phase_artifacts(
            PhaseName.IMPLEMENT.value,
            {"implementation.md": agent_response},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"implementation.md": agent_response[:_MAX_ARTIFACT_STATE_CHARS]}

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }


def build_implement_subgraph() -> Any:
    """Build the IMPLEMENT phase subgraph."""
    builder = StateGraph(ImplementSubgraphState)
    builder.add_node("run_agent", _run_implement_agent)
    builder.add_node("save_artifacts", _save_implement_artifacts)
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)
    return builder.compile()
