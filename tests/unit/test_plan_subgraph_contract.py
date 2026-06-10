"""Plan-agent node contract behaviour (trace 019eb00d).

A local thinking model burned its whole generation in the reasoning channel
and stopped with empty content and no ``write_structured_plan`` call, so
plan.json was never written. Two failure modes are pinned here:

1. The node issues ONE corrective retry (same conversation + explicit nudge)
   before failing the contract.
2. ``CriticalContractFailure`` must propagate out of ``_run_plan_agent`` —
   the blanket ``except Exception`` used to swallow it into a soft
   ``phase_status="error"``, so the missing plan_json only surfaced at the
   critic gate and triggered a full phase rework (research included)
   instead of subgraph_wrapper's clean-thread retry.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from spine.exceptions import CriticalContractFailure
from spine.workflow.subgraphs import plan_subgraph as ps


def _state(tmp_path) -> dict[str, Any]:
    return {
        "work_id": "wk-plan",
        "work_type": "task",
        "description": "do the thing",
        "workspace_root": str(tmp_path),
        "has_spec": False,
        "spec_path": "",
        "feedback": [],
        "retry_count": 0,
    }


def _plan_json_path(tmp_path):
    return tmp_path / ".spine" / "artifacts" / "wk-plan" / "plan" / "plan.json"


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
def _stub_agent_build(monkeypatch):
    monkeypatch.setattr(ps, "build_plan_agent", lambda state, config: object())
    monkeypatch.setattr(ps, "materialize_artifacts", lambda *a, **kw: None)
    monkeypatch.setattr(ps, "build_context", lambda *a, **kw: None)


@pytest.mark.asyncio
async def test_corrective_retry_recovers_missing_plan(tmp_path, _stub_agent_build, monkeypatch):
    """First invoke produces nothing; the node retries once with a nudge in
    the same conversation, and the second invoke's plan.json is accepted."""
    calls: list[dict[str, Any]] = []

    async def _fake_invoke(agent, payload, **kw):
        calls.append(payload)
        if len(calls) == 2:
            _write_valid_plan(tmp_path)
        return {"messages": payload["messages"] + [{"role": "assistant", "content": ""}]}

    monkeypatch.setattr(ps, "ainvoke_with_retry", _fake_invoke)

    result = await ps._run_plan_agent(_state(tmp_path), None)

    assert len(calls) == 2
    # The retry continues the same conversation and appends the nudge.
    nudge = calls[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "write_structured_plan" in nudge["content"]
    assert len(calls[1]["messages"]) > len(calls[0]["messages"])
    # Recovered plan is propagated to state.
    assert result["plan_json"]
    assert result["execution_waves"]
    assert result.get("phase_status") != "error"


@pytest.mark.asyncio
async def test_no_retry_when_plan_written_first_try(tmp_path, _stub_agent_build, monkeypatch):
    calls: list[dict[str, Any]] = []

    async def _fake_invoke(agent, payload, **kw):
        calls.append(payload)
        _write_valid_plan(tmp_path)
        return {"messages": [{"role": "assistant", "content": "done"}]}

    monkeypatch.setattr(ps, "ainvoke_with_retry", _fake_invoke)

    result = await ps._run_plan_agent(_state(tmp_path), None)

    assert len(calls) == 1
    assert result["plan_json"]


@pytest.mark.asyncio
async def test_contract_failure_propagates_after_failed_retry(
    tmp_path, _stub_agent_build, monkeypatch
):
    """Neither attempt writes plan.json — CriticalContractFailure must escape
    _run_plan_agent (NOT be swallowed into phase_status='error') so
    subgraph_wrapper's structural retry can re-run the phase."""

    async def _fake_invoke(agent, payload, **kw):
        return {"messages": [{"role": "assistant", "content": ""}]}

    monkeypatch.setattr(ps, "ainvoke_with_retry", _fake_invoke)

    with pytest.raises(CriticalContractFailure, match="plan.json does not exist"):
        await ps._run_plan_agent(_state(tmp_path), None)


@pytest.mark.asyncio
async def test_malformed_plan_json_propagates(tmp_path, _stub_agent_build, monkeypatch):
    async def _fake_invoke(agent, payload, **kw):
        path = _plan_json_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        return {"messages": [{"role": "assistant", "content": ""}]}

    monkeypatch.setattr(ps, "ainvoke_with_retry", _fake_invoke)

    with pytest.raises(CriticalContractFailure, match="malformed or unreadable"):
        await ps._run_plan_agent(_state(tmp_path), None)


@pytest.mark.asyncio
async def test_unexpected_errors_still_soft_fail(tmp_path, _stub_agent_build, monkeypatch):
    """Non-contract exceptions keep the existing soft-degradation contract."""

    async def _fake_invoke(agent, payload, **kw):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(ps, "ainvoke_with_retry", _fake_invoke)

    result = await ps._run_plan_agent(_state(tmp_path), None)
    assert result["phase_status"] == "error"
    assert "connection reset" in result["agent_response"]
