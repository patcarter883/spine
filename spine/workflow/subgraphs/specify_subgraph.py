"""SPECIFY phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the specify Deep Agent with early
   commitment (task classification + vector recall).
2. ``save_artifacts`` — scans disk for artifacts.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import SpecifySubgraphState
from spine.agents.specify_agent import build_specify_agent
from spine.agents.helpers import extract_response, format_classification_block
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.classification import classify_task
from spine.agents.tools.recall_tool import RecallTool
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    artifact_path,
)
from spine.config import SpineConfig
from spine.exceptions import CriticalContractFailure

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


async def _early_commitment(
    description: str,
    workspace_root: str,
    config: RunnableConfig | None,
) -> tuple[str, list[dict], str]:
    """Classify task and retrieve relevant code chunks from vector store."""
    classification = await classify_task(description, config)
    task_category = classification.category
    reasoning = classification.reasoning
    logger.info(
        "Task classification: %s (confidence: %.2f)", task_category, classification.confidence
    )

    config_obj = SpineConfig.load()
    recall = RecallTool(
        db_path=config_obj.checkpoint_path,
        embedding_provider=config_obj.embedding_model,
    )

    import json as _json

    recall_result = await recall._arun(
        query=description,
        k=config_obj.recall_k,
        task_category=task_category,
        max_tokens=50000,
    )
    result_data = _json.loads(recall_result)
    retrieved = result_data.get("results", [])
    logger.info("Retrieved %d chunks for SPECIFY context", len(retrieved))
    return task_category, retrieved, reasoning


async def _run_specify_agent(
    state: SpecifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the specify Deep Agent within the subgraph."""
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] SPECIFY subgraph: run_agent starting")

    try:
        # ── Early commitment: classify + recall ──
        # Reuse task_category from exploration subgraph if available,
        # otherwise classify fresh.
        task_category: str | None = state.get("task_category")
        classification_reasoning = ""
        retrieved_context: list[dict] = []
        try:
            if task_category:
                logger.info("[%s] Using pre-classified task category: %s", work_id, task_category)
            else:
                _cat, _ctx, _reasoning = await _early_commitment(description, workspace_root, config)
                task_category = _cat
                retrieved_context = _ctx
                classification_reasoning = _reasoning
            if not retrieved_context:
                # Run recall even when category was pre-existing
                config_obj = SpineConfig.load()
                recall = RecallTool(db_path=config_obj.checkpoint_path)
                result_text = await recall._arun(
                    query=description,
                    k=config_obj.recall_k,
                    task_category=task_category,
                    max_tokens=50000,
                )
                import json as _json
                result_data = _json.loads(result_text)
                retrieved_context = result_data.get("results", [])
                logger.info(
                    "[%s] SPECIFY recall: %d chunks (category=%s)",
                    work_id, len(retrieved_context), task_category,
                )
        except Exception as exc:
            logger.warning("[%s] Early commitment skipped: %s", work_id, exc)

        # Build recall section for prompt
        recall_section = ""
        if retrieved_context:
            recall_section = "\n## Retrieved Codebase Context\n\n"
            for i, chunk in enumerate(retrieved_context[:5], 1):
                recall_section += (
                    f"### Chunk {i}: {chunk.get('symbol_name', 'unknown')} "
                    f"({chunk.get('file_path', 'unknown')})\n\n"
                    f"```\n{chunk.get('raw_code', '')[:1000]}\n```\n\n"
                )

        agent = build_specify_agent(
            dict(state),
            config,
            extra_tools=[
                RecallTool(db_path=SpineConfig.load().checkpoint_path),
            ],
        )
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        spec_dir = artifact_path(work_id, PhaseName.SPECIFY.value)
        context_seed = (
            f"globalThis.context = {{work_id: '{work_id}', phase: 'specify', "
            f"spec_dir: '{spec_dir}'}};\n\n"
        )

        prompt = (
            context_seed
            + format_classification_block(task_category, classification_reasoning)
            + f"Create a detailed specification for the following work:\n\n{description}"
            + recall_section
        )

        ctx = build_context(dict(state), PhaseName.SPECIFY)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.SPECIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
            "task_category": task_category,
            "retrieved_context": retrieved_context,
        }

    except Exception as e:
        logger.error(f"[{work_id}] SPECIFY subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "phase_status": "error",
        }


async def _save_specify_artifacts(
    state: SpecifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the specify agent."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {"artifacts_output": {}, "phase_status": existing_phase_status}

    # Fail-closed: validate specification.json exists and is well-formed
    spec_json_path = Path(workspace_root) / ".spine" / "artifacts" / work_id / "specify" / "specification.json"
    if spec_json_path.exists():
        try:
            raw = spec_json_path.read_text(encoding="utf-8")
            spec_data = json.loads(raw)
            # Validate required keys exist
            if not isinstance(spec_data, dict):
                raise CriticalContractFailure(
                    phase="specify",
                    reason="specification.json exists but is not a JSON object",
                )
            for key in ("title", "summary", "requirements"):
                if key not in spec_data:
                    raise CriticalContractFailure(
                        phase="specify",
                        reason=f"specification.json is missing required key '{key}' — "
                               f"the specify agent produced malformed structured output. "
                               f"Keys found: {list(spec_data.keys())}",
                    )
        except (json.JSONDecodeError, OSError) as exc:
            raise CriticalContractFailure(
                phase="specify",
                reason=f"specification.json exists but is malformed or unreadable: {exc}",
            )
        except CriticalContractFailure:
            raise
        except Exception as exc:
            raise CriticalContractFailure(
                phase="specify",
                reason=f"specification.json validation error: {exc}",
            )
    else:
        raise CriticalContractFailure(
            phase="specify",
            reason="specification.json does not exist — "
                   "the specify agent did not produce structured output via write_specification. "
                   "This indicates a model invocation failure in the specify node.",
        )

    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        PhaseName.SPECIFY.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        materialize_phase_artifacts(
            PhaseName.SPECIFY.value,
            {"specification.md": agent_response},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"specification.md": agent_response[:_MAX_ARTIFACT_STATE_CHARS]}

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }


def build_specify_subgraph() -> Any:
    """Build the SPECIFY phase subgraph.

    Returns:
        Uncompiled StateGraph builder. Call .compile() to get a runnable graph,
        or .compile(checkpointer=...) for per-phase checkpoint isolation.
    """
    builder = StateGraph(SpecifySubgraphState)
    builder.add_node("run_agent", _run_specify_agent)
    builder.add_node("save_artifacts", _save_specify_artifacts)
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)
    return builder
