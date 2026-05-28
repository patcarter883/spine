"""VERIFY phase as a LangGraph subgraph with Send API dispatch.

The subgraph uses the same manager/router/call/aggregate pattern as the
exploration and implement subgraphs: it dispatches one ``slice-verifier``
per slice in parallel via the LangGraph ``Send`` API, then aggregates
the verdicts.

Nodes:
- ``verify_router``: conditional edge — reads ``execution_waves`` from
  state, returns ``[Send("run_slice_verifier", ...)]`` or
  ``"synthesize_verification"``
- ``run_slice_verifier``: builds a ``slice-verifier`` subagent per slice
  and invokes it. Runs in parallel via Send API.
- ``aggregate_verification``: deterministic fan-in point after all
  parallel slice-verifier nodes complete.
- ``synthesize_verification``: writes ``verification.md`` and
  ``verification.json`` from accumulated verdicts, determines
  ``overall_status``.
- ``save_artifacts``: scans disk, materializes to state, determines
  phase status from ``verification.json``.

Edges::

    START → verify_router
    verify_router → Send("run_slice_verifier", {slice}) × N  OR  → synthesize_verification
    run_slice_verifier → aggregate_verification
    aggregate_verification → synthesize_verification → save_artifacts → END
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from spine.agents.artifacts import (
    artifact_path,
    materialize_phase_artifacts,
    scan_artifact_dir,
)
from spine.agents.plan_do import (
    directive_from_state,
    format_directive_for_prompt,
    run_plan_node,
)
from spine.agents.retry import ainvoke_with_retry
from spine.exceptions import CriticalContractFailure
from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import VerifySubgraphState

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


# ── Router: START → run_slice_verifier (Send) or synthesize ─────────────


def _verify_router(
    state: VerifySubgraphState,
) -> list[Send] | Literal["synthesize_verification"]:
    """Fan-out to slice-verifier nodes via Send API.

    Reads ``execution_waves`` from state, flattens all waves into a single
    dispatch list (dependencies resolved by the scheduler during PLAN).

    Raises ``CriticalContractFailure`` if ``execution_waves`` is missing
    or empty — this is a structural invariant violation.
    """
    execution_waves = state.get("execution_waves")

    if not execution_waves:
        raise CriticalContractFailure(
            phase="verify",
            reason="execution_waves is missing or empty in state — "
                   "the PLAN phase did not produce structured data transfer. "
                   "The prerequisite gate should have caught this before "
                   "VERIFY ran; check artifact_gate.py.",
        )

    all_slices: list[dict] = []
    for wave in execution_waves:
        if isinstance(wave, list):
            for sl in wave:
                if isinstance(sl, dict) and sl.get("id"):
                    all_slices.append(sl)

    if not all_slices:
        raise CriticalContractFailure(
            phase="verify",
            reason="execution_waves is present but contains zero valid "
                   "slice dicts with 'id' fields. The PLAN phase produced "
                   "malformed structured data.",
        )

    logger.info(
        "VERIFY router: dispatching %d slice-verifier(s): %s",
        len(all_slices),
        [s.get("id", "?") for s in all_slices],
    )

    base_state = {
        "phase": state.get("phase", "verify"),
        "work_id": state.get("work_id", "unknown"),
        "work_type": state.get("work_type", ""),
        "workspace_root": state.get("workspace_root", "."),
    }
    return [
        # Two-node verifier branch: plan_slice_verifier (no tools) →
        # run_slice_verifier (tools). Each parallel branch carries its
        # own active_slice_directive through the chain.
        Send("plan_slice_verifier", {**base_state, "slice": s})
        for s in all_slices
    ]


# ── Node: plan_slice_verifier (no tools) ────────────────────────────────


async def _plan_slice_verifier_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> Command:
    """No-tool plan step for one slice's verification.

    Produces a per-branch SubagentDirective and dispatches a Send to
    run_slice_verifier carrying both the slice and the directive on the
    per-branch payload. Returning ``Command(goto=Send(...))`` — rather
    than writing the directive to a shared channel — is required
    because parallel Send branches share the subgraph's channel space,
    so N concurrent writes to ``active_slice_directive`` would crash
    apply_writes with ``InvalidUpdateError``.
    """
    work_id = state.get("work_id", "unknown")
    slice_data: dict = state.get("slice", {}) or {}
    slice_id = slice_data.get("id", "unknown")
    title = slice_data.get("title", "")
    target_files = slice_data.get("target_files") or []
    criteria = slice_data.get("acceptance_criteria") or []

    crit_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "(none provided)"
    file_lines = "\n".join(f"- {p}" for p in target_files) if target_files else "(none provided)"
    task = (
        f"Plan a verification pass for slice {slice_id!r} (title: {title!r}). "
        "The do node will read the files, run any lint/tests it needs, and emit a "
        "VerificationResult (verdict, checklist, gaps, recommendations).\n\n"
        f"## Acceptance criteria\n{crit_lines}\n\n"
        f"## Target files\n{file_lines}"
    )
    directive = await run_plan_node(
        state=dict(state),
        config=config,
        phase_path=f"{PhaseName.VERIFY.value}/subagents/slice-verifier",
        task_description=task,
        role_hint=f"slice-verifier for slice {slice_id!r}",
    )
    logger.info(
        "[%s] plan_slice_verifier: slice=%r approach=%r",
        work_id, slice_id, directive.approach[:80],
    )
    send_payload: dict[str, Any] = {
        "phase": state.get("phase", "verify"),
        "work_id": state.get("work_id", "unknown"),
        "work_type": state.get("work_type", ""),
        "workspace_root": state.get("workspace_root", "."),
        "slice": slice_data,
        "active_slice_directive": directive.model_dump(),
    }
    return Command(goto=Send("run_slice_verifier", send_payload))


# ── Node: run_slice_verifier ────────────────────────────────────────────


async def _run_slice_verifier_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run a slice-verifier subagent for one feature slice.

    The slice is injected into state by the Send API via
    ``Send("run_slice_verifier", {"slice": {...}})``.  The subagent
    receives the slice definition, acceptance criteria, and access
    to the filesystem to inspect the implemented code.
    """
    from spine.agents.factory import build_phase_agent
    from spine.agents.subagents import build_subagent_spec

    work_id = state.get("work_id", "unknown")
    slice_data: dict = state.get("slice", {})
    slice_id = slice_data.get("id", "unknown")

    logger.info(
        "[%s] Slice-verifier node: slice=%r (title=%r)",
        work_id,
        slice_id,
        slice_data.get("title", ""),
    )

    try:
        subagent_spec = build_subagent_spec(
            name="slice-verifier",
            phase=PhaseName.VERIFY,
            state=state,
            config=config,
        )

        extra_tools = list(subagent_spec.get("tools", []))
        agent = build_phase_agent(
            state=state,
            config=config,
            phase=PhaseName.VERIFY,
            system_prompt=subagent_spec["system_prompt"],
            is_subagent=True,
            extra_tools=extra_tools,
            response_format=subagent_spec.get("response_format"),
            skip_filesystem_middleware=True,
        )

        slice_json = json.dumps(slice_data, indent=2, ensure_ascii=False)
        directive_block = format_directive_for_prompt(
            directive_from_state(dict(state), "active_slice_directive")
        )
        prompt = (
            (directive_block + "\n" if directive_block else "")
            + f"## Verify Slice: {slice_id}\n\n"
            f"The full slice definition is in the JSON below. "
            f"Verify the implementation against these acceptance criteria. "
            f"Inspect the actual code files on disk — do not trust that "
            f"the implementation summary matches reality.\n\n"
            f"```json\n{slice_json}\n```"
        )

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name="verify-slice",
            work_id=work_id,
        )

        verification_result = _extract_verification_result(result, slice_id)

    except Exception as e:
        logger.error(
            "[%s] Slice-verifier failed for %r: %s",
            work_id,
            slice_id,
            e,
            exc_info=True,
        )
        verification_result = {
            "slice_name": slice_id,
            "verdict": "NOT_VERIFIED",
            "checklist": [
                {
                    "criterion": "Subagent execution",
                    "passed": False,
                    "detail": f"Verifier subagent crashed: {e}",
                }
            ],
            "gaps": [f"Verification could not complete: {e}"],
            "recommendations": ["Re-run verification for this slice"],
        }

    return {"verification_results": [verification_result]}


def _extract_verification_result(result: dict, slice_id: str) -> dict:
    """Extract a VerificationResult dict from an agent result.

    If the agent returned structured output via ``response_format``,
    it'll be in the ``structured_response`` key.  Falls back to the
    last assistant message content.

    The ``slice_name`` field is overridden with the actual slice_id
    from the router to guarantee consistency.
    """
    structured = result.get("structured_response")
    if structured:
        if isinstance(structured, dict):
            structured["slice_name"] = slice_id
            return structured
        if hasattr(structured, "model_dump"):
            d = structured.model_dump()
            d["slice_name"] = slice_id
            return d

    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    parsed["slice_name"] = slice_id
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {
                "slice_name": slice_id,
                "verdict": "NOT_VERIFIED",
                "checklist": [
                    {
                        "criterion": "Agent output",
                        "passed": False,
                        "detail": "Subagent produced unstructured output — verify manually",
                    }
                ],
                "gaps": ["Unstructured output from subagent"],
                "recommendations": [],
            }

    return {
        "slice_name": slice_id,
        "verdict": "NOT_VERIFIED",
        "checklist": [
            {
                "criterion": "Agent output",
                "passed": False,
                "detail": "(no output from subagent)",
            }
        ],
        "gaps": ["Subagent produced no output"],
        "recommendations": [],
    }


# ── Node: aggregate_verification ───────────────────────────────────────


async def _aggregate_verification_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Fan-in point after all parallel slice-verifier nodes complete.

    Results are already accumulated via ``operator.add`` on the
    ``verification_results`` field — no manual merging needed.
    """
    results = state.get("verification_results", [])
    verdicts = [r.get("verdict", "?") for r in results]
    logger.info(
        "VERIFY aggregate: %d verification result(s) — verdicts: %s",
        len(results),
        verdicts,
    )
    return {}


# ── Node: synthesize_verification ───────────────────────────────────────


async def _synthesize_verification_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Write verification.md and verification.json from accumulated verdicts.

    Uses the existing ``WriteVerificationReportTool`` logic (via a
    utility function) to produce both the human-readable markdown and
    the structured JSON artifact.
    """
    from spine.agents.verify_tools import write_verification_files

    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", ".")
    verification_results = state.get("verification_results", [])

    if not verification_results:
        logger.warning("[%s] VERIFY synthesize: zero slice verification results", work_id)
        return {
            "agent_response": "",
            "artifacts_output": {},
            "phase_status": "needs_review",
            "verification_attempted": False,
            "verification_passed": False,
        }

    verify_dir = artifact_path(work_id, PhaseName.VERIFY.value)
    summary = _build_verification_summary(verification_results)

    try:
        write_verification_files(verification_results, summary, workspace_root, verify_dir)
    except Exception as e:
        logger.error(
            "[%s] VERIFY synthesize: failed to write artifacts: %s",
            work_id,
            e,
        )
        return {
            "agent_response": summary,
            "artifacts_output": {},
            "phase_status": "error",
            "verification_attempted": True,
            "verification_passed": False,
        }

    all_verified = all(
        r.get("verdict") == "VERIFIED" for r in verification_results
    )

    logger.info(
        "[%s] VERIFY synthesize: wrote %d slice verdicts to %s/ (all_verified=%s)",
        work_id,
        len(verification_results),
        verify_dir,
        all_verified,
    )

    return {
        "agent_response": summary,
        "artifacts_output": {"verification.md": summary[:_MAX_ARTIFACT_STATE_CHARS]},
        "phase_status": "success" if all_verified else "needs_review",
        "verification_attempted": True,
        "verification_passed": all_verified,
    }


def _build_verification_summary(verification_results: list[dict]) -> str:
    """Build a human-readable verification summary from slice verdicts."""
    total = len(verification_results)
    verdicts: dict[str, int] = {}
    for r in verification_results:
        v = r.get("verdict", "UNKNOWN")
        verdicts[v] = verdicts.get(v, 0) + 1

    verified = verdicts.get("VERIFIED", 0)
    not_verified = total - verified

    parts = [
        f"Verification complete for {total} feature slice(s).",
        f"- VERIFIED: {verified}",
    ]
    if not_verified:
        parts.append(f"- NOT_VERIFIED: {not_verified}")

    if not_verified == 0:
        parts.append("All slices passed verification.")
    else:
        parts.append(f"{not_verified} slice(s) did not pass — see verification.md for details.")

    return "\n".join(parts)


# ── Node: save_artifacts ────────────────────────────────────────────────


async def _save_verify_artifacts(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the verify phase to disk and state.

    Reads ``verification.json`` for authoritative phase status,
    falling back to string-matching on ``verification.md`` content if
    JSON is not available.
    """
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
        workspace_root,
        work_id,
        PhaseName.VERIFY.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

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

    # Determine status from verification.json (authoritative source).
    verify_dir = Path(workspace_root) / artifact_path(work_id, PhaseName.VERIFY.value)
    verify_json_path = verify_dir / "verification.json"
    is_verified = False
    if verify_json_path.exists():
        try:
            vdata = json.loads(verify_json_path.read_text())
            is_verified = vdata.get("overall_status") == "VERIFIED"
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "[%s] verification.json exists but could not be parsed; "
                "defaulting to unverified",
                work_id,
            )

    verification_findings: list[dict] = []
    agent_result = state.get("messages", [])
    if isinstance(agent_result, dict) and "structured_response" in agent_result:
        sr = agent_result.get("structured_response", {})
        if isinstance(sr, dict):
            verification_findings = [sr]

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if is_verified else "needs_review",
        "verification_findings": verification_findings,
    }


# ── Builder ──────────────────────────────────────────────────────────────


def build_verify_subgraph() -> Any:
    """Build the VERIFY phase subgraph with Send API dispatch.

    Returns a compiled StateGraph with five nodes:
    1. verify_router — conditional edge dispatching Send objects
    2. run_slice_verifier — per-slice subagent invocation (parallel)
    3. aggregate_verification — fan-in checkpoint
    4. synthesize_verification — writes verification artifacts
    5. save_artifacts — scans disk, materializes to state
    """
    builder = StateGraph(VerifySubgraphState)

    builder.add_node("plan_slice_verifier", _plan_slice_verifier_node)
    builder.add_node("run_slice_verifier", _run_slice_verifier_node)
    builder.add_node("aggregate_verification", _aggregate_verification_node)
    builder.add_node("synthesize_verification", _synthesize_verification_node)
    builder.add_node("save_artifacts", _save_verify_artifacts)

    builder.add_conditional_edges(
        START,
        _verify_router,
        {
            # Send targets dispatch to plan_slice_verifier; each parallel
            # branch then chains plan → do before fan-in.
            "plan_slice_verifier": "plan_slice_verifier",
            "synthesize_verification": "synthesize_verification",
        },
    )

    # plan_slice_verifier dispatches to run_slice_verifier dynamically
    # via Command(goto=Send) (see the node) so each parallel branch
    # carries its own directive without colliding on a shared LastValue
    # channel. run_slice_verifier → aggregate_verification is a plain
    # fan-in edge that runs once on the merged verification_results.
    builder.add_edge("run_slice_verifier", "aggregate_verification")
    builder.add_edge("aggregate_verification", "synthesize_verification")
    builder.add_edge("synthesize_verification", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder