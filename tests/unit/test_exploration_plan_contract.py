"""Exploration-subgraph PLAN synthesize contract behaviour (trace 019eb412).

The corrective-retry + CriticalContractFailure protections from trace
019eb00d lived only in plan_subgraph._run_plan_agent — but the workflow
routes PLAN through the exploration subgraph (_USE_EXPLORATION_SUBGRAPH),
whose _synthesize_plan swallowed synthesis failures into a soft
phase_status="error". Trace 019eb412: the synthesizer hallucinated
read_file ×3, then burned its whole 8K completion budget in the reasoning
channel (LengthFinishReasonError), and the workflow still reached
human_review reporting plan "success" with artifact_count=0.

Pinned here:
1. _synthesize_plan issues ONE corrective retry (same conversation +
   explicit nudge) before failing the contract.
2. A raising first invocation (LengthFinishReasonError) still gets the
   corrective retry rather than an immediate soft error.
3. CriticalContractFailure propagates when plan.json is still missing.
4. _save_exploration_artifacts fails closed for PLAN without plan.json.
5. build_plan_synthesizer carries ForceToolUntilCalledMiddleware so the
   model cannot hallucinate off-surface tools or stall in reasoning.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from spine.exceptions import CriticalContractFailure
from spine.workflow.subgraphs import exploration_subgraph as es


def _state(tmp_path) -> dict[str, Any]:
    return {
        "phase": "plan",
        "work_id": "wk-explan",
        "work_type": "task",
        "description": "do the thing",
        "workspace_root": str(tmp_path),
        "findings": [{"topic": "t", "summary": "s"}],
        "feedback": [],
        "retry_count": 0,
        "artifacts": {},
        "scratchpad": "",
    }


def _plan_json_path(tmp_path):
    return tmp_path / ".spine" / "artifacts" / "wk-explan" / "plan" / "plan.json"


def _write_valid_plan(tmp_path) -> None:
    path = _plan_json_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "architecture_overview": "x",
                "feature_slices": [{"id": "s1", "title": "slice one"}],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def _stub_synth(monkeypatch):
    import spine.agents.plan_agent as plan_agent

    monkeypatch.setattr(plan_agent, "build_plan_synthesizer", lambda state, config: object())
    monkeypatch.setattr(es, "materialize_artifacts", lambda *a, **kw: None)
    monkeypatch.setattr(es, "build_context", lambda *a, **kw: None)


@pytest.mark.asyncio
async def test_corrective_retry_recovers_missing_plan(tmp_path, _stub_synth, monkeypatch):
    """First invoke produces nothing; the node retries once with a nudge in
    the same conversation, and the second invoke's plan.json is accepted."""
    calls: list[dict[str, Any]] = []

    async def _fake_invoke(agent, payload, **kw):
        calls.append(payload)
        if len(calls) == 2:
            _write_valid_plan(tmp_path)
        return {"messages": payload["messages"] + [{"role": "assistant", "content": ""}]}

    monkeypatch.setattr(es, "ainvoke_with_retry", _fake_invoke)

    result = await es._synthesize_plan(_state(tmp_path), None)

    assert len(calls) == 2
    nudge = calls[1]["messages"][-1]["content"]
    assert "write_structured_plan" in nudge
    assert result.get("plan_json")
    assert result.get("phase_status") != "error"


@pytest.mark.asyncio
async def test_raising_first_invoke_still_gets_corrective_retry(
    tmp_path, _stub_synth, monkeypatch
):
    """A LengthFinishReasonError-style raise from the first invocation must
    not soft-fail the node — the corrective retry runs from the original
    prompt (trace 019eb412)."""
    calls: list[dict[str, Any]] = []

    async def _fake_invoke(agent, payload, **kw):
        calls.append(payload)
        if len(calls) == 1:
            raise RuntimeError("LengthFinishReasonError: length limit reached")
        _write_valid_plan(tmp_path)
        return {"messages": payload["messages"] + [{"role": "assistant", "content": ""}]}

    monkeypatch.setattr(es, "ainvoke_with_retry", _fake_invoke)

    result = await es._synthesize_plan(_state(tmp_path), None)

    assert len(calls) == 2
    # Retry conversation restarts from the original prompt + nudge.
    assert len(calls[1]["messages"]) == 2
    assert "write_structured_plan" in calls[1]["messages"][-1]["content"]
    assert result.get("plan_json")


@pytest.mark.asyncio
async def test_contract_failure_propagates_when_plan_still_missing(
    tmp_path, _stub_synth, monkeypatch
):
    """plan.json missing after the corrective retry → CriticalContractFailure
    must escape the blanket except (not a soft phase_status='error')."""

    async def _fake_invoke(agent, payload, **kw):
        return {"messages": payload["messages"] + [{"role": "assistant", "content": ""}]}

    monkeypatch.setattr(es, "ainvoke_with_retry", _fake_invoke)

    with pytest.raises(CriticalContractFailure):
        await es._synthesize_plan(_state(tmp_path), None)


@pytest.mark.asyncio
async def test_save_artifacts_fails_closed_for_plan_without_plan_json(tmp_path):
    """Backstop: save_artifacts must not report a PLAN phase without
    plan.json (trace 019eb412: human_review saw success, artifact_count=0)."""
    state = {
        "phase": "plan",
        "work_id": "wk-explan",
        "workspace_root": str(tmp_path),
        "agent_response": "",
        "phase_status": "",
    }
    with pytest.raises(CriticalContractFailure):
        await es._save_exploration_artifacts(state, None)


@pytest.mark.asyncio
async def test_save_artifacts_passes_for_plan_with_plan_json(tmp_path):
    _write_valid_plan(tmp_path)
    state = {
        "phase": "plan",
        "work_id": "wk-explan",
        "workspace_root": str(tmp_path),
        "agent_response": "",
        "phase_status": "",
        "plan_json": _plan_json_path(tmp_path).read_text(encoding="utf-8"),
        "execution_waves": [],
    }
    result = await es._save_exploration_artifacts(state, None)
    assert result["phase_status"] == "success"


def test_plan_synthesizer_forces_write_structured_plan(tmp_path, monkeypatch):
    """build_plan_synthesizer must carry ForceToolUntilCalledMiddleware —
    without it the model hallucinated read_file and stalled in reasoning."""
    import spine.agents.plan_agent as plan_agent
    from spine.agents.tool_forcing import ForceToolUntilCalledMiddleware

    captured: dict[str, Any] = {}

    def _fake_build_phase_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(plan_agent, "build_phase_agent", _fake_build_phase_agent)

    plan_agent.build_plan_synthesizer(
        {
            "work_id": "wk-explan",
            "workspace_root": str(tmp_path),
            "work_type": "task",
            "description": "d",
            "feedback": [],
            "artifacts": {},
        },
        None,
    )

    forcing = [
        m
        for m in (captured.get("extra_middleware") or [])
        if isinstance(m, ForceToolUntilCalledMiddleware)
    ]
    assert forcing, "plan synthesizer is missing ForceToolUntilCalledMiddleware"
    # Single-tool surface (trace 019eb52c): read_prior_artifacts always
    # returned empty on this path while the prompt promised it loaded the
    # spec — the forced-tool loop re-called it 23×. With only the write
    # tool bound, tool_choice="any" IS a pin on every provider.
    assert forcing[0].gate_tool is None
    tool_names = [getattr(t, "name", "?") for t in (captured.get("extra_tools") or [])]
    assert tool_names == ["write_structured_plan"], (
        f"plan synthesizer surface must be exactly the write tool, got {tool_names}"
    )


@pytest.mark.asyncio
async def test_synthesize_plan_inlines_spec_into_prompt(
    tmp_path, _stub_synth, monkeypatch
):
    """The spec is prompt content now, not a tool fetch (trace 019eb52c)."""
    spec_dir = tmp_path / ".spine" / "artifacts" / "wk-explan" / "specify"
    spec_dir.mkdir(parents=True)
    (spec_dir / "specification.md").write_text(
        "# Spec\nREQUIREMENT-MARKER-42: the flag must default to false.\n",
        encoding="utf-8",
    )
    captured: list[dict[str, Any]] = []

    async def _fake_invoke(agent, payload, **kw):
        captured.append(payload)
        _write_valid_plan(tmp_path)
        return {"messages": payload["messages"] + [{"role": "assistant", "content": ""}]}

    monkeypatch.setattr(es, "ainvoke_with_retry", _fake_invoke)

    await es._synthesize_plan(_state(tmp_path), None)

    prompt = captured[0]["messages"][0]["content"]
    assert "<specification>" in prompt
    assert "REQUIREMENT-MARKER-42" in prompt
    assert "read_prior_artifacts" not in prompt


def test_forcing_releases_on_real_write_structured_plan_output(tmp_path):
    """The middleware must release after a real successful write (trace 019eb43f).

    The old release check matched a success substring ("written to") against
    the tool's return string; the plan tool's message lacked it, so forcing
    never released and the synthesizer was compelled to call
    write_structured_plan every turn (24×) until the context window
    overflowed. The check now matches failure prefixes instead, but keep
    running the REAL tool and feeding its REAL output through the middleware
    so message drift can't silently re-open the loop.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    from spine.agents.plan_tools import StructuredWritePlanTool
    from spine.agents.tool_forcing import ForceToolUntilCalledMiddleware

    tool = StructuredWritePlanTool(
        workspace_root=str(tmp_path),
        plan_dir=".spine/artifacts/wk-explan/plan",
    )
    output = tool._run(
        architecture_overview="x",
        feature_slices=[
            {
                "id": "s1",
                "title": "slice one",
                "execution_requirements": "do it",
                "acceptance_criteria": ["works"],
            }
        ],
        testing_strategy="pytest",
    )
    assert "VALIDATION_ERROR" not in output and "ERROR" not in output

    mw = ForceToolUntilCalledMiddleware(final_tool="write_structured_plan")
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "write_structured_plan", "args": {}, "id": "tc1"}],
        ),
        ToolMessage(content=output, tool_call_id="tc1"),
    ]
    assert mw._final_tool_succeeded(messages), (
        "middleware did not release on the plan tool's success message — "
        f"output looked like a failure to it: {output!r}"
    )

    # A rejected write must KEEP forcing so the model self-corrects in-loop.
    for failure in (
        "VALIDATION_ERROR: plan rejected before writing.\nduplicate slice id",
        "ERROR: Could not write plan.json: disk full",
    ):
        failed = [
            AIMessage(
                content="",
                tool_calls=[{"name": "write_structured_plan", "args": {}, "id": "tc2"}],
            ),
            ToolMessage(content=failure, tool_call_id="tc2"),
        ]
        assert not mw._final_tool_succeeded(failed), (
            f"middleware released on a failed write: {failure!r}"
        )
