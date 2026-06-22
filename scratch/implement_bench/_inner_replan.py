#!/usr/bin/env python3
"""In-clone executor for the bench replan subcommand.

Runs INSIDE a disposable git clone (cwd == clone root). Reads the frozen
specification from disk, invokes the **production** PLAN phase standalone, and
writes a JSON result to --out.

PRODUCTION PARITY (this is the whole point of the bench): PLAN ships as the
exploration → synthesis subgraph — parallel researcher subagents map the
codebase upstream, then a single-tool synthesizer turns the spec + findings
into plan.json. We drive exactly that graph here, seeded the same way
``spine.workflow.compose._plan_state_mapper`` seeds it in a real run.

We do NOT use the old linear ``plan_subgraph`` (a 3-tool agent carrying
``search_codebase``). It was REMOVED from spine: a finite-window local model
spiralled on it (~62 ``search_codebase`` calls, never converging) re-researching
what exploration already owns. Benching it measured a path production never
runs — which is why earlier replan benches looked peculiarly bad. If you find
yourself importing ``build_plan_subgraph`` here, stop: it no longer exists.

The exploration subgraph writes plan.json + plan.md into
  .spine/artifacts/{work_id}/plan/
which bench.py replan then copies into the frozen baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from pathlib import Path


def _load_prior_research(work_id: str, phase: str) -> tuple[list[str], list[dict]]:
    """Read ``(topics, findings)`` from a phase's research_log.json, or ([],[]).

    Mirrors ``spine.workflow.compose._load_prior_research`` so the standalone
    replan seeds PLAN with SPECIFY's research the same way production does.
    """
    log_path = Path(".spine/artifacts") / work_id / phase / "research_log.json"
    if not log_path.exists():
        return [], []
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    topics = [t for t in (data.get("topics") or []) if isinstance(t, str)]
    findings = [f for f in (data.get("findings") or []) if isinstance(f, dict)]
    return topics, findings


async def _run(work_id: str, description: str) -> dict:
    from spine.config import SpineConfig
    from spine.agents.artifacts import artifact_path
    from spine.models.enums import PhaseName
    from spine.workflow.subgraphs.exploration_subgraph import (
        build_exploration_subgraph,
    )

    SpineConfig.load(".spine/config.yaml")  # side effect: load config + .env

    # The frozen baseline already has the specification on disk. The exploration
    # synthesizer reads it from disk via ``spec_path`` (the specify DIRECTORY),
    # so we only need to point at the dir — no need to pre-populate
    # state["artifacts"] (the old plan agent's read_prior_artifacts is gone).
    spec_dir = Path(artifact_path(work_id, PhaseName.SPECIFY.value))
    has_spec = (spec_dir / "specification.md").exists()

    # Seed prior research exactly like _plan_state_mapper:
    #   - any prior PLAN research (rework re-runs) seeds topics + findings;
    #   - SPECIFY's findings seed prior_phase_findings so PLAN researchers and
    #     the manager start from the architectural map instead of re-mapping it.
    prior_topics, prior_findings = _load_prior_research(work_id, PhaseName.PLAN.value)
    _, specify_findings = _load_prior_research(work_id, PhaseName.SPECIFY.value)

    # Build the production PLAN graph (exploration → synthesis) and compile it
    # without a checkpointer — a one-shot replan needs no resumability, and this
    # avoids the vec0 module error from spine.db when the checkpointer opens it.
    graph = build_exploration_subgraph(phase=PhaseName.PLAN.value).compile()

    initial_state: dict = {
        "work_id": work_id,
        "work_type": "task",
        "description": description,
        "workspace_root": str(Path.cwd()),
        "phase": PhaseName.PLAN.value,
        "spec_path": str(spec_dir) if has_spec else "",
        "has_spec": has_spec,
        "retry_count": 0,
        "scratchpad": "",
        # Minimal BaseSubgraphState fields
        "messages": [],
        "read_cache": {},
        "config_path": ".spine/config.yaml",
    }
    if prior_topics or prior_findings:
        initial_state["topics"] = prior_topics
        initial_state["findings"] = prior_findings
    if specify_findings:
        initial_state["prior_phase_findings"] = specify_findings

    # spine.config disables tracing process-wide at import; real work runs
    # re-enable it via work_run_tracing (a contextvar-scoped tracer). The
    # standalone plan invocation must do the same or it emits NO trace — which
    # is why earlier replans never appeared in LangSmith.
    from spine.observability import work_run_tracing

    with work_run_tracing(work_id, PhaseName.PLAN.value):
        result = await graph.ainvoke(initial_state)

    artifacts_dir = Path(artifact_path(work_id, PhaseName.PLAN.value))
    plan_files = (
        sorted(p.name for p in artifacts_dir.iterdir()) if artifacts_dir.exists() else []
    )
    return {
        "phase_status": result.get("phase_status"),
        "execution_waves": result.get("execution_waves"),
        "findings_count": len(result.get("findings") or []),
        "plan_files": plan_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_id")
    ap.add_argument("--description", required=True)
    ap.add_argument("--out", required=True, help="result json path")
    args = ap.parse_args()

    out: dict = {
        "work_id": args.work_id,
        "status": None,
        "error": None,
        "elapsed_s": None,
    }
    t0 = time.time()
    try:
        res = asyncio.run(_run(args.work_id, args.description))
        out.update(res)
        out["status"] = res.get("phase_status", "unknown")
    except Exception as exc:
        out["status"] = "exception"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    finally:
        out["elapsed_s"] = round(time.time() - t0, 1)
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
