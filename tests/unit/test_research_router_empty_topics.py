"""Router behaviour when the research manager returns degenerate output.

Pairs with ``test_research_manager_empty_topics`` — the manager is supposed
to coerce ``explore + topics=[]`` to ``done`` before the router sees it, but
if any caller bypasses that coercion the router must also degrade gracefully
(trace 019e72bc: the unsoftened router raised CriticalContractFailure and
killed the whole subgraph).

Missing or unrecognised decisions are still hard failures — they indicate
structured-output collapse and should fail loud.
"""

from __future__ import annotations

import pytest

from spine.exceptions import CriticalContractFailure
from spine.workflow.subgraphs.exploration_subgraph import _research_router


def _state(**overrides):
    base = {
        "phase": "specify",
        "work_id": "wk-router",
        "work_type": "task",
        "description": "test",
        "workspace_root": "/tmp",
        "retry_count": 0,
        "feedback": [],
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "research_round": 0,
        "max_rounds": 3,
        "manager_decision": "explore",
        "topics": [],
        "findings": [],
        "agent_response": "",
    }
    base.update(overrides)
    return base


def test_router_softens_empty_topics_explore_to_synthesize(caplog):
    """When the manager somehow returns explore + topics=[], the router
    must NOT crash the subgraph — degrade to synthesize with a warning."""

    import logging

    caplog.set_level(
        logging.WARNING,
        logger="spine.workflow.subgraphs.exploration_subgraph",
    )
    result = _research_router(_state(manager_decision="explore", topics=[]))

    assert result == "synthesize"
    assert any(
        "empty topics" in r.getMessage().lower() for r in caplog.records
    ), f"Expected an 'empty topics' warning; got: {[r.getMessage() for r in caplog.records]}"


def test_router_still_raises_on_missing_decision():
    """Missing manager_decision is a real structured-output collapse —
    keep the loud failure."""

    state = _state()
    state.pop("manager_decision")
    with pytest.raises(CriticalContractFailure) as exc:
        _research_router(state)
    assert "manager_decision is missing" in str(exc.value)


def test_router_still_raises_on_invalid_decision():
    """Decisions other than explore/done are also structured-output
    collapse — fail loud."""

    with pytest.raises(CriticalContractFailure) as exc:
        _research_router(_state(manager_decision="hmm-not-sure", topics=[]))
    assert "unexpected value" in str(exc.value)


def test_router_done_still_routes_to_synthesize():
    """Sanity check: the soften path doesn't shadow the normal 'done' path."""

    result = _research_router(_state(manager_decision="done", topics=[]))
    assert result == "synthesize"


def test_router_sends_carry_work_id():
    """Send payloads are the target node's ENTIRE state — omitting work_id
    ran every explore_do scout with work_id=None, which bypassed the
    cross-scout symbol_cache dedupe (trace 019eb00d: get_source(SpineConfig)
    fetched 3× per research round)."""
    from langgraph.types import Send

    result = _research_router(
        _state(
            manager_decision="explore",
            topics=["topic a", "topic b"],
            workspace_root="/tmp/ws",
        )
    )

    assert isinstance(result, list) and result
    for send in result:
        assert isinstance(send, Send)
        assert send.arg["work_id"] == "wk-router"
        assert send.arg["work_type"] == "task"
        assert send.arg["workspace_root"] == "/tmp/ws"
        assert send.arg["topic"]
        assert send.arg["phase"] == "specify"


def test_router_sends_carry_covered_ground_digest():
    """Round-2+ dispatches must ship a digest of prior findings so new
    researchers don't re-establish ground siblings already mapped
    (trace 019eb4c7: rounds 2-3 re-fetched the same hot symbols and
    re-derived round 1's architecture facts)."""
    from langgraph.types import Send

    result = _research_router(
        _state(
            manager_decision="explore",
            topics=["a fresh round-two topic"],
            findings=[
                {
                    "topic": "how is provider config persisted",
                    "summary": "UIApi mutates .spine/config.yaml via yaml.dump",
                    "file_map": {"spine/ui_api/api.py": "UIApi"},
                },
            ],
        )
    )

    assert isinstance(result, list) and result
    digest = result[0].arg["covered_ground"]
    assert "do NOT" in digest
    assert "spine/ui_api/api.py" in digest
    assert "UIApi mutates" in digest


def test_router_round_one_sends_have_empty_covered_ground():
    """No findings yet → no digest → round-1 branches pay zero extra tokens."""
    result = _research_router(
        _state(manager_decision="explore", topics=["topic a"], findings=[])
    )
    assert isinstance(result, list) and result
    assert result[0].arg["covered_ground"] == ""


class TestRenderCoveredGround:
    def test_error_sentinels_are_skipped(self):
        from spine.agents.exploration_agents import render_covered_ground

        digest = render_covered_ground(
            [{"topic": "t1", "summary": "failed", "error": True}]
        )
        assert digest == ""

    def test_budget_cap_elides_overflow(self):
        from spine.agents.exploration_agents import (
            _COVERED_GROUND_BUDGET_CHARS,
            render_covered_ground,
        )

        findings = [
            {
                "topic": f"topic number {i}",
                "summary": "s" * 200,
                "file_map": {f"spine/mod_{i}.py": "x"},
            }
            for i in range(30)
        ]
        digest = render_covered_ground(findings)
        # Preamble + elision note sit outside the per-entry budget; the
        # entries themselves must respect it.
        assert len(digest) < _COVERED_GROUND_BUDGET_CHARS + 400
        assert "elided for size" in digest
