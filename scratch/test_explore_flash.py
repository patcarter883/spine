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
    # Replicate run_explore_node so we can inspect the full message trace.
    spec = build_subagent_spec(name="researcher", phase=PhaseName.SPECIFY, state=state, config=None)
    extra_tools = list(spec.get("tools", []))
    agent = build_phase_agent(
        state=state,
        config=None,
        phase=PhaseName.SPECIFY,
        system_prompt=spec["system_prompt"],
        is_subagent=True,
        extra_tools=extra_tools,
        response_format=spec.get("response_format"),
        skip_filesystem_middleware=True,
    )
    prompt = (
        f"## Research Topic\n{TOPIC}\n\n"
        "Investigate this specific area of the codebase. "
        "Use MCP tools first for structural navigation, then fall back to read_file/glob/grep if needed. "
        "Return your findings in the ResearchFindings format."
    )
    agent_result = await ainvoke_with_retry(
        agent, {"messages": [{"role": "user", "content": prompt}]},
        phase_name="explore", work_id="harness-flash-test",
    )

    msgs = agent_result.get("messages", [])
    print(f"\n=== MESSAGE COUNT: {len(msgs)} ===")
    tool_calls_total = 0
    for i, m in enumerate(msgs):
        mtype = type(m).__name__
        tcs = getattr(m, "tool_calls", None) or []
        tool_calls_total += len(tcs)
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = str(content)
        head = (content or "")[:200].replace("\n", " ")
        tc_summary = ""
        if tcs:
            tc_summary = " tool_calls=" + ",".join(
                f"{t.get('name')}({list((t.get('args') or {}).keys())})" for t in tcs
            )
        print(f"[{i}] {mtype}{tc_summary} :: {head}")
    print(f"\nTOTAL TOOL CALLS: {tool_calls_total}")

    print("\n=== STRUCTURED_RESPONSE PRESENT? ===")
    print(repr(agent_result.get("structured_response"))[:500])
    print("\n=== FINAL MSG CONTENT (raw) ===")
    last = msgs[-1] if msgs else None
    if last is not None:
        c = getattr(last, "content", "")
        print(f"type={type(c).__name__} len={len(str(c))}")
        print(str(c)[:1200])

    result = {"findings": _extract_findings(agent_result)}
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
