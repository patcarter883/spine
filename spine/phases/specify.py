"""SPINE SPECIFY phase — generate a detailed spec from a prompt.

This phase takes a work description and produces a specification document.
It delegates to the specify Deep Agent which uses subagents for research
and documentation.

Context engineering: prior artifacts are on disk (not inlined). SpineContext
is passed at invoke time for typed per-run context.

Phase node functions are async to avoid event-loop binding errors when
subagents inherit the parent checkpointer — sync nodes run in a thread
pool, which breaks ``asyncio.Lock`` objects bound to the original loop.
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.specify_agent import build_specify_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.classification import classify_task
from spine.agents.tools.recall_tool import RecallTool
from spine.agents.artifacts import (
    artifact_path,
    materialize_artifacts,
    materialize_phase_artifacts,
)
from spine.workflow.registry import get_registry
from spine.config import SpineConfig

logger = logging.getLogger(__name__)


async def _early_commitment(
    description: str,
    workspace_root: str,
    config: RunnableConfig | None,
) -> tuple[str, list[dict], str]:
    """Perform early commitment: classify task and retrieve relevant code.

    Args:
        description: The work description.
        workspace_root: The workspace root path.
        config: LangGraph runtime config.

    Returns:
        Tuple of (task_category, retrieved_context, classification_reasoning).
    """
    # Step 1: Classify the task
    classification = await classify_task(description, config)
    task_category = classification.category
    reasoning = classification.reasoning

    logger.info("Task classification: %s (confidence: %.2f)", task_category, classification.confidence)

    # Step 2: Retrieve relevant code using RecallTool
    config_obj = SpineConfig.load()
    recall_tool = RecallTool(
        db_path=config_obj.checkpoint_path,
        embedding_provider=config_obj.embedding_model,
    )

    recall_result = await recall_tool._arun(
        query=description,
        k=config_obj.recall_k,
        task_category=task_category,
        max_tokens=50000,
    )

    import json
    result_data = json.loads(recall_result)
    retrieved_context = result_data.get("results", [])

    logger.info("Retrieved %d chunks for SPECIFY context", len(retrieved_context))

    return task_category, retrieved_context, reasoning


async def call_specify(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the SPECIFY phase.

    Delegates to the specify Deep Agent, which generates a specification
    document from the work description. If the phase is being reworked
    (retry > 0), includes prior critic feedback in the prompt.

    Performs early commitment before agent invocation:
    1. Classify the task type for targeted vector search
    2. Retrieve relevant code chunks via RecallTool

    Args:
        state: The current workflow state.
        config: LangGraph runtime config (contains thread_id, providers).

    Returns:
        Partial state update with artifacts and status.
    """
    # SPECIFY is the first phase in spec/critical_spec workflows — it is the
    # ONLY phase that receives the raw work description.  The specification
    # artifact it produces captures and expands on the description, so
    # downstream phases (PLAN, TASKS, IMPLEMENT, VERIFY) work from artifacts
    # on disk instead.  The only additional input to any phase beyond prior
    # artifacts should be review feedback (from critic gates, verify agent,
    # or human review).
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    retry_count = state.get("retry_count", {}).get(PhaseName.SPECIFY.value, 0)
    feedback = state.get("feedback", [])
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] SPECIFY phase starting (retry={retry_count})")

    try:
        # ── EARLY COMMITMENT: Classify and Recall ──
        # Only run on first pass (retry == 0) to avoid re-retrieving
        task_category = None
        retrieved_context = []
        classification_reasoning = ""
        if retry_count == 0:
            task_category, retrieved_context, classification_reasoning = await _early_commitment(
                description, workspace_root, config
            )

        agent = build_specify_agent(state, config, extra_tools=[
            RecallTool(
                db_path=SpineConfig.load().checkpoint_path,
            )
        ])

        # Materialize prior artifacts to disk so the agent can read them
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build the prompt — prior artifacts are on disk, not inlined

        # Include retrieved context for first pass
        recall_section = ""
        if retrieved_context:
            recall_section = "\n## Retrieved Codebase Context\n\n"
            for i, chunk in enumerate(retrieved_context[:5], 1):
                recall_section += f"### Chunk {i}: {chunk.get('symbol_name', 'unknown')} ({chunk.get('file_path', 'unknown')})\n\n"
                recall_section += f"```\n{chunk.get('raw_code', '')[:1000]}\n```\n\n"

        rework_prefix = ""
        if retry_count > 0:
            rework_prefix = "⚠ **REWORK PASS**: Your primary objective is to revise the prior specification. Address all points from the critic feedback.\n\n"

        prompt = (
            rework_prefix
            + f"## Task Classification\nCategory: {task_category}\n{classification_reasoning}\n\n"
            + f"Create a detailed specification for the following work:\n\n{description}"
            + recall_section
        )
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"\n\nPrevious review feedback (please address):\n{feedback_text}"

        # Build runtime context for the agent
        ctx = build_context(state, PhaseName.SPECIFY)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.SPECIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        # Extract the specification from the agent's response
        spec_content = extract_response(result)

        # Materialize this phase's artifacts to disk immediately
        phase_artifacts = {"specification.md": spec_content}
        materialize_phase_artifacts(
            PhaseName.SPECIFY.value, phase_artifacts, workspace_root, work_id=work_id
        )

        return {
            "artifacts": {PhaseName.SPECIFY.value: phase_artifacts},
            "current_phase": PhaseName.SPECIFY.value,
            "status": "running",
            "prompt_request": None,
            "task_category": task_category,
            "retrieved_context": retrieved_context,
        }

    except Exception as e:
        logger.error(f"[{work_id}] SPECIFY phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.SPECIFY.value: {}},
            "current_phase": PhaseName.SPECIFY.value,
            "status": "running",
            "prompt_request": {
                "message": f"SPECIFY phase failed: {e}",
                "phase": PhaseName.SPECIFY.value,
            },
            "task_category": None,
            "retrieved_context": [],
        }


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.SPECIFY.value,
    call_fn=call_specify,
    build_agent_fn=build_specify_agent,
    description="Generate a detailed specification from a work description",
)
