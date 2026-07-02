"""Channel-declaration regression tests for subgraph state TypedDicts.

LangGraph silently drops node-return keys that aren't declared as channels
on the StateGraph's state class. We hit this exact bug with
``specification_json`` being declared on ``SpecifySubgraphState`` (a class
not actually used by the exploration subgraph) but missing from
``ExplorationSubgraphState`` (the one that IS used at
exploration_subgraph.py:1108). Result: ``_synthesize_specify`` returned a
populated ``specification_json``, the channel didn't exist, the value was
dropped, and the critic's contract gate blew up.
"""

from __future__ import annotations

from spine.workflow.subgraph_state import (
    CriticSubgraphState,
    ExplorationSubgraphState,
    GapPlanSubgraphState,
    ImplementSubgraphState,
    PlanSubgraphState,
    SpecifySubgraphState,
    TasksSubgraphState,
    VerifySubgraphState,
)


def test_exploration_state_declares_specification_json():
    """ExplorationSubgraphState must declare specification_json so
    _synthesize_specify's return value survives the subgraph."""
    assert "specification_json" in ExplorationSubgraphState.__annotations__


def test_exploration_state_declares_plan_json():
    """Same property, but for the PLAN side (already worked — guard so
    nobody removes it during a future cleanup)."""
    assert "plan_json" in ExplorationSubgraphState.__annotations__


def test_exploration_state_declares_evidence_channel():
    """explore_do writes ``exploration_evidence``, summarise reads it.

    Without the declaration LangGraph would silently drop the channel
    between the two nodes and summarise would have no input to convert.
    """
    assert "exploration_evidence" in ExplorationSubgraphState.__annotations__


def test_specify_state_declares_directive_channel():
    assert "specify_directive" in SpecifySubgraphState.__annotations__


def test_plan_state_declares_directive_channel():
    assert "plan_directive" in PlanSubgraphState.__annotations__


def test_tasks_state_declares_directive_channel():
    assert "tasks_directive" in TasksSubgraphState.__annotations__


def test_gap_plan_state_declares_directive_channel():
    assert "gap_plan_directive" in GapPlanSubgraphState.__annotations__


def test_critic_state_has_no_directive_channel():
    """The critic's plan->do directive was removed: a no-tool directive twice
    became the critic's de-facto review rubric with invented requirements
    (traces 019f1204 Tkinter, 019f2131 'config_page' node)."""
    assert "critic_directive" not in CriticSubgraphState.__annotations__


def test_implement_state_declares_active_slice_directive():
    """plan_slice_implementer writes per-branch active_slice_directive."""
    assert "active_slice_directive" in ImplementSubgraphState.__annotations__


def test_verify_state_declares_active_slice_directive():
    """plan_slice_verifier writes per-branch active_slice_directive."""
    assert "active_slice_directive" in VerifySubgraphState.__annotations__
