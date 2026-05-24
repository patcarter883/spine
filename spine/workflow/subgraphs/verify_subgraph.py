"""VERIFY phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the verify Deep Agent.
2. ``save_artifacts`` — scans disk for artifacts, determines phase status.

State schema: ``VerifySubgraphState`` — isolated from parent ``WorkflowState``.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import VerifySubgraphState
from spine.agents.verify_agent import build_verify_agent
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

# Maximum characters of artifact content to store in state.
_MAX_ARTIFACT_STATE_CHARS = 500


async def _run_verify_agent(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the verify Deep Agent within the subgraph.

    Builds the agent from the subgraph state, materializes prior artifacts,
    constructs the prompt, and invokes with retry.  The original work
    description is NOT included — VERIFY works from prior artifacts on
    disk, not the raw description.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] VERIFY subgraph: run_agent starting")

    try:
        # Build the agent — build_verify_agent expects a dict-like state
        agent = build_verify_agent(dict(state), config)  # type: ignore[dict-constructor]

        # Materialize prior artifacts to disk so agent can read them
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)  # type: ignore[dict-constructor]

        # Build prompt referencing disk paths
        has_spec = "spec" in work_type
        spec_path = artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = artifact_path(work_id, PhaseName.TASKS.value)
        impl_path = artifact_path(work_id, PhaseName.IMPLEMENT.value)

        prompt_lines = [
            "Verify that the implementation meets the requirements. "
            "Check that all feature slices are implemented correctly, "
            "the plan was followed, and the original task is complete.",
            "",
            "Prior artifacts are available on disk — read them as needed:",
        ]
        if has_spec:
            prompt_lines.extend(
                [
                    f"- Specification: `{spec_path}/specification.md`",
                    f"- Plan: `{plan_path}/plan.md`",
                ]
            )
        verify_path = artifact_path(work_id, PhaseName.VERIFY.value)
        prompt_lines.extend(
            [
                f"- Feature Slices: `{tasks_path}/tasks.md`",
                f"- Codebase map: `{tasks_path}/codebase-map.md`",
                f"- Implementation: `{impl_path}/implementation.md`",
                "",
                "Use `read_file` and `grep` to inspect them. Do NOT load "
                "everything into context at once.",
                "",
                "Read the codebase map FIRST — it contains file paths, key "
                "functions, and conventions discovered during the tasks phase.",
                "",
                "Also inspect the actual code files on disk using `ls` and "
                "`read_file` — the implementation summary may not reflect "
                "the actual state of the code.",
                "",
                "## Where to Write Your Output",
                f"Write your verification report to `{verify_path}/verification.md` "
                "using `write_file`.  The report MUST begin with `VERIFIED` or `PASSED` "
                "on the first line if the implementation meets requirements, or "
                "`FAILED` followed by the issues found.  This file is REQUIRED — "
                "without it, the workflow treats the phase as failed.",
                "",
                "**RLM parallel verify pattern:** Use `eval` to read the "
                "tasks artifact, extract the slice list, then dispatch a "
                "`slice-verifier` subagent per slice via "
                "`Promise.allSettled(tools.task(...))`. Synthesize the "
                "verification report from subagent results in code — do NOT "
                "re-read each slice file manually into conversation.",
            ]
        )
        prompt = "\n".join(prompt_lines)

        ctx = build_context(dict(state), PhaseName.VERIFY)  # type: ignore[dict-constructor]

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.VERIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        verify_content = extract_response(result)
        return {
            "messages": result.get("messages", []),
            "agent_response": verify_content,
        }

    except Exception as e:
        logger.error(f"[{work_id}] VERIFY subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "phase_status": "error",
        }


async def _save_verify_artifacts(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the verify agent to disk and state.

    Scans the verify artifact directory for files written by the agent.
    If none found, falls back to the agent response text.
    """
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    # If the agent node already set an error/needs_review status, preserve it
    if existing_phase_status in ("error", "needs_review"):
        return {
            "artifacts_output": {},
            "phase_status": existing_phase_status,
        }

    # Scan what the agent wrote to disk
    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        PhaseName.VERIFY.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    # Fallback: if agent wrote nothing, use the extracted response
    if not disk_artifacts:
        verify_content = agent_response
        if not verify_content or len(verify_content.strip()) < 20:
            verify_content = (
                "Verification could not produce a meaningful report. "
                "The agent returned insufficient output. Manual review required."
            )
        materialize_phase_artifacts(
            PhaseName.VERIFY.value,
            {"verification.md": verify_content},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"verification.md": verify_content[:_MAX_ARTIFACT_STATE_CHARS]}

    # Determine status from verification content
    verify_text = next(iter(disk_artifacts.values()), "") if disk_artifacts else ""
    is_verified = "VERIFIED" in verify_text.upper() or "PASSED" in verify_text.upper()

    # Populate verification_findings from agent structured response
    agent_result = state.get("messages", [])
    verification_findings: list[dict] = []
    if isinstance(agent_result, dict) and "structured_response" in agent_result:
        sr = agent_result.get("structured_response", {})
        if isinstance(sr, dict):
            verification_findings = [sr]

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if is_verified else "needs_review",
        "verification_findings": verification_findings,
    }


def build_verify_subgraph() -> Any:
    """Build the VERIFY phase subgraph.

    Returns a compiled StateGraph with two nodes:
    1. run_agent — builds and invokes the verify Deep Agent.
    2. save_artifacts — scans disk for artifacts written by the agent.
    """
    builder = StateGraph(VerifySubgraphState)

    builder.add_node("run_agent", _run_verify_agent)
    builder.add_node("save_artifacts", _save_verify_artifacts)

    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder
