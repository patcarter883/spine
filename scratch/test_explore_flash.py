"""Minimal harness: invoke the researcher subagent on a single topic
using deepseek-v4-flash and report tool-call activity + findings.

Usage:
    .venv/bin/python scratch/test_explore_flash.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Make sure imports resolve when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Override the SPECIFY phase provider BEFORE any spine imports ────────
# We monkey-patch SpineConfig._model_spec_from_config via env? Easier:
# monkey-patch SpineConfig.resolve_model after import.

import spine.config as _spine_config

_orig_resolve_model = _spine_config.SpineConfig.resolve_model
_orig_resolve_provider = _spine_config.SpineConfig.resolve_provider_config


def _patched_resolve_model(self, phase: str | None = None) -> str:
    return "openrouter:deepseek/deepseek-v4-flash"


def _patched_resolve_provider(self, phase: str | None = None):
    # Match the deepseek-v4-flash provider entry so guided_decoding etc are picked up
    return {
        "name": "deepseek-v4-flash",
        "type": "deepagents-model",
        "model": "openrouter:deepseek/deepseek-v4-flash",
        "enabled": True,
        "guided_decoding": True,
    }


_spine_config.SpineConfig.resolve_model = _patched_resolve_model
_spine_config.SpineConfig.resolve_provider_config = _patched_resolve_provider

# ── Now run a single explore node ──────────────────────────────────────

from spine.agents.exploration_agents import run_explore_node, _extract_findings
from spine.agents.factory import build_phase_agent
from spine.agents.retry import ainvoke_with_retry
from spine.agents.subagents import build_subagent_spec
from spine.models.enums import PhaseName
from spine.agents.garbage_collector import commit_findings_and_clear_search


TOPIC = (
    "Spine framework architecture and core concepts relevant to project "
    "setup and analysis"
)


async def main() -> None:
    state = {
        "work_id": "harness-flash-test",
        "phase": "specify",
        "workspace_root": str(Path(__file__).resolve().parents[1]),
        "description": (
            "Initial greenfields project setup and brownfields project "
            "analysis and mapping features"
        ),
        "topics": [TOPIC],
        "scratchpad": "",
    }
    from spine.agents.exploration_agents import run_explore_node
    result = await run_explore_node(state, config=None, topic=TOPIC)
    print("\n=== EXPLORE NODE RESULT ===")
    findings = result.get("findings", [])
    print(f"findings count: {len(findings)}")
    for i, f in enumerate(findings):
        print(f"\n--- finding[{i}] ---")
        if isinstance(f, dict):
            for k, v in f.items():
                if isinstance(v, (dict, list)):
                    v_str = json.dumps(v, indent=2)[:2000]
                else:
                    v_str = str(v)[:2000]
                print(f"{k}: {v_str}")
        else:
            print(repr(f)[:2000])

    # Also dump full result for visibility
    print("\n=== RAW STATE KEYS ===")
    print(list(result.keys()))


if __name__ == "__main__":
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    asyncio.run(main())
