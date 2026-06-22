#!/usr/bin/env python3
"""In-clone executor for the bench replan subcommand.

Runs INSIDE a disposable git clone (cwd == clone root). Reads the frozen
specification from disk, invokes the PLAN subgraph standalone (bypassing
the work-item state machine entirely), and writes a JSON result to --out.

The plan subgraph writes plan.json + plan.md into
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


async def _run(work_id: str, description: str) -> dict:
    from spine.config import SpineConfig
    from spine.workflow.subgraphs import build_plan_subgraph

    config = SpineConfig.load(".spine/config.yaml")

    spec_path = Path(".spine/artifacts") / work_id / "specify" / "specification.md"
    has_spec = spec_path.exists()

    # Build the plan subgraph and compile it without a checkpointer — we don't
    # need resumability for a one-shot replan; this also avoids the vec0 module
    # error from spine.db when the checkpointer tries to open it.
    graph = build_plan_subgraph().compile()

    from langgraph.types import Command

    initial_state = {
        "work_id": work_id,
        "description": description,
        "workspace_root": str(Path.cwd()),
        "spec_path": str(spec_path) if has_spec else "",
        "has_spec": has_spec,
        # Minimal fields from BaseSubgraphState
        "messages": [],
        "phase": "plan",
        "config_path": ".spine/config.yaml",
    }

    # Run the plan subgraph — it writes artifacts to disk as a side effect
    result = await graph.ainvoke(initial_state)

    artifacts_dir = Path(".spine/artifacts") / work_id / "plan"
    plan_files = sorted(p.name for p in artifacts_dir.iterdir()) if artifacts_dir.exists() else []
    return {
        "phase_status": result.get("phase_status"),
        "execution_waves": result.get("execution_waves"),
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
