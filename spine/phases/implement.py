"""SPINE IMPLEMENT phase — generate code to implement feature slices.

The implement Deep Agent reads the plan artifacts (plan.json with
execution_waves) and dispatches slice-implementer subagents per wave.
Slices within a wave run in parallel (independent); waves run
sequentially (later waves depend on earlier ones).

Prior artifacts are NOT inlined — the agent reads them on demand from
the filesystem.

Context engineering: read cache prevents re-reading files across subagent turns.

Phase node functions are async to avoid event-loop binding errors when
subagents inherit the parent checkpointer.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.implement_agent import build_implement_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    list_slice_files,
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    _artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


# Maximum characters of artifact content to store in WorkflowState.
_MAX_ARTIFACT_STATE_CHARS = 500


# ── Dispatch note builders ────────────────────────────────────────────


def _build_wave_dispatch_note(
    execution_waves: list[list[dict]],
) -> str:
    """Build Step 2 instructions for wave-based dispatch.

    Each execution wave contains independent slices that can run in
    parallel. Waves themselves must run sequentially because later
    waves depend on code produced by earlier ones.

    Args:
        execution_waves: Pre-sorted waves of slice dicts from the
            scheduler.

    Returns:
        Markdown-formatted dispatch instructions for the orchestrator.
    """
    wave_count = len(execution_waves)
    total_slices = sum(len(wave) for wave in execution_waves)

    wave_summaries: list[str] = []
    for i, wave in enumerate(execution_waves):
        slice_ids = [s.get("id", f"slice-{j}") for j, s in enumerate(wave)]
        wave_summaries.append(
            f"  Wave {i + 1}: {', '.join(slice_ids)} "
            f"({len(wave)} slice(s), parallel)"
        )

    return (
        f"Slices are organized into **{wave_count} execution wave(s)** "
        f"({total_slices} total slice(s)). "
        f"Slices within a wave are independent and MUST run in parallel. "
        f"Waves MUST run sequentially — do not start wave N+1 until "
        f"all slices in wave N are complete.\n\n"
        "**Wave structure:**\n"
        + "\n".join(wave_summaries)
        + "\n\n"
        "**Dispatch pattern (wave-sequential):**\n"
        "```js\n"
        "// globalThis.planData is loaded from plan.json via read_slice_files\n"
        "const data = globalThis.planData;\n"
        "const waves = data.execution_waves;\n"
        "const map = data.codebase_map || '';\n"
        "globalThis.sliceResults = [];\n\n"
        "for (let w = 0; w < waves.length; w++) {\n"
        "  const wave = waves[w];\n"
        "  const dispatches = wave.map(slice =>\n"
        "    tools.task({\n"
        '      subagent_type: "slice-implementer",\n'
        "      description: `Implement slice: ${slice.id}\\n\\n"
        "## Slice Definition\\n${JSON.stringify(slice, null, 2)}\\n\\n"
        "## Codebase Map\\n${map}`,\n"
        "    })\n"
        "  );\n"
        "  const results = await Promise.allSettled(dispatches);\n"
        "  globalThis.sliceResults.push(...results);\n"
        "  console.log(`Wave ${w+1} complete: ${results.map(r => r.status)}`);\n"
        "}\n"
        "```"
    )


def _build_legacy_dispatch_note(slice_count: int) -> str:
    """Build Step 2 instructions for legacy (flat parallel) dispatch.

    Used when ``execution_waves`` is absent from state (workflows that
    still use the TASKS phase). All slices are dispatched in a single
    parallel batch.

    Args:
        slice_count: Number of slice files discovered on disk.

    Returns:
        Markdown-formatted dispatch instructions for the orchestrator.
    """
    if slice_count == 0:
        return (
            "⚠ No slices found. Write the implementation report with "
            "an empty result set."
        )

    return (
        f"Dispatch all {slice_count} slices in parallel using a single "
        "`eval` call with `Promise.allSettled`.\n\n"
        "**Dispatch pattern:**\n"
        "```js\n"
        "const data = globalThis.slices;\n"
        "const map = data.codebase_map || '';\n"
        "const dispatches = Object.entries(data.slices).map(([name, content]) =>\n"
        "  tools.task({\n"
        '    subagent_type: "slice-implementer",\n'
        "    description: `Implement slice: ${name}\\n\\n"
        "## Slice Definition\\n${content}\\n\\n"
        "## Codebase Map\\n${map}`,\n"
        "  })\n"
        ");\n"
        "const results = await Promise.allSettled(dispatches);\n"
        "globalThis.sliceResults = results;\n"
        "console.log(JSON.stringify(results.map(r => r.status)));\n"
        "```"
    )


async def call_implement(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the IMPLEMENT phase.

    Delegates to the implement Deep Agent, which writes code for each
    feature slice. If reworking, includes prior feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with implementation artifacts.
    """
    # NOTE: The original work description is NOT needed here — IMPLEMENT works
    # from the feature slices and codebase map produced by TASKS (on disk),
    # not from the raw description.  The only additional input beyond prior
    # artifacts should be review feedback (critic gates, verify agent, human).
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    retry_count = state.get("retry_count", {}).get(PhaseName.IMPLEMENT.value, 0)
    feedback = state.get("feedback", [])
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] IMPLEMENT phase starting (retry={retry_count})")

    try:
        agent = build_implement_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build prompt — all prior artifacts are on disk, NOT inlined.
        # Skip spec/plan references for quick workflows that lack them.
        impl_dir = f".spine/artifacts/{work_id}/implement"
        tasks_dir = f".spine/artifacts/{work_id}/tasks"

        # ── Resolve execution waves from state (preferred) or legacy ──
        execution_waves = state.get("execution_waves")
        has_waves = bool(execution_waves)
        plan_dir = _artifact_path(work_id, PhaseName.PLAN.value)
        plan_json_path = f"{plan_dir}/plan.json"

        if has_waves:
            total_slices = sum(len(wave) for wave in execution_waves)
            wave_count = len(execution_waves)
            logger.info(
                f"[{work_id}] Dispatching {total_slices} slice(s) across "
                f"{wave_count} wave(s) from execution_waves"
            )
            dispatch_note = _build_wave_dispatch_note(execution_waves)
            waves_json = json.dumps(execution_waves, ensure_ascii=False)
            context_seed = (
                f"globalThis.context = {{work_id: '{work_id}', phase: 'implement', "
                f"tasks_dir: '{tasks_dir}', impl_dir: '{impl_dir}', "
                f"plan_json: '{plan_json_path}', mode: 'waves'}};\n"
                f"globalThis.execution_waves = {waves_json};\n\n"
            )
        else:
            # Legacy fallback: use list_slice_files from tasks phase
            logger.info(
                f"[{work_id}] No execution_waves in state — "
                f"falling back to list_slice_files"
            )
            slice_files = list_slice_files(workspace_root, work_id)
            slice_count = len(slice_files)
            dispatch_note = _build_legacy_dispatch_note(slice_count)
            context_seed = (
                f"globalThis.context = {{work_id: '{work_id}', phase: 'implement', "
                f"tasks_dir: '{tasks_dir}', impl_dir: '{impl_dir}', "
                f"mode: 'legacy'}};\n\n"
            )

        rework_prefix = ""
        if retry_count > 0:
            rework_prefix = "⚠ **REWORK PASS**: Your primary objective is to revise the prior implementation. Address all points from the critic feedback.\n\n"

        has_spec = "spec" in work_type
        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = _artifact_path(work_id, PhaseName.TASKS.value)

        prompt_lines = [
            "Implement the feature slices described below. Write clean, "
            "production-quality code for each slice.",
            "",
            "## Task Input",
        ]
        if has_waves:
            prompt_lines.extend(
                [
                    "Work from the plan artifacts produced by the PLAN phase.",
                    f"- Plan: `{plan_path}/plan.md`",
                    f"- Structured plan (JSON): `{plan_json_path}`",
                    f"- Codebase map: included in `{plan_json_path}` under `codebase_map`",
                    "",
                    "The `plan.json` file contains structured `feature_slices` with "
                    "id, title, target_files, execution_requirements, dependencies, "
                    "acceptance_criteria, and complexity for each slice.",
                ]
            )
        else:
            prompt_lines.extend(
                [
                    "Work from the feature slice files and codebase map produced by the TASKS phase.",
                ]
            )
        if has_spec:
            prompt_lines.extend(
                [
                    f"- Specification: `{spec_path}/specification.md`",
                ]
            )
        if not has_waves:
            prompt_lines.extend(
                [
                    f"- Plan: `{plan_path}/plan.md`",
                    f"- Feature Slices: `{tasks_path}/tasks.md` and each `slice-*.md`",
                    f"- Codebase map: `{tasks_path}/codebase-map.md`",
                ]
            )
        prompt_lines.extend(
            [
                "",
                "Read the codebase map FIRST — it contains file paths, key functions, and conventions "
                "discovered during planning. Use it instead of re-exploring the codebase.",
                "",
                "## Step 2 Guidelines",
                dispatch_note,
                "",
            ]
        )
        prompt = context_seed + rework_prefix + "\n".join(prompt_lines)
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(state, PhaseName.IMPLEMENT)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.IMPLEMENT.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        impl_content = extract_response(result)

        # ── Collect artifacts from disk (agent writes via write_file) ─────
        # The authoritative artifacts are the files the agent wrote to disk,
        # NOT the extracted LLM response.  For thinking models the response
        # is chain-of-thought reasoning.
        disk_artifacts = scan_artifact_dir(
            workspace_root,
            work_id,
            PhaseName.IMPLEMENT.value,
            max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
        )

        # Fallback: if agent wrote nothing, materialize from response
        if not disk_artifacts and impl_content.strip():
            materialize_phase_artifacts(
                PhaseName.IMPLEMENT.value,
                {"implementation.md": impl_content},
                workspace_root,
                work_id=work_id,
            )
            disk_artifacts = {"implementation.md": impl_content[:_MAX_ARTIFACT_STATE_CHARS]}

        return {
            "artifacts": {PhaseName.IMPLEMENT.value: disk_artifacts},
            "current_phase": PhaseName.IMPLEMENT.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] IMPLEMENT phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.IMPLEMENT.value: {}},
            "current_phase": PhaseName.IMPLEMENT.value,
            "status": "running",
            "prompt_request": {
                "message": f"IMPLEMENT phase failed: {e}",
                "phase": PhaseName.IMPLEMENT.value,
            },
        }


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.IMPLEMENT.value,
    call_fn=call_implement,
    build_agent_fn=build_implement_agent,
    description="Generate code to implement feature slices",
)
