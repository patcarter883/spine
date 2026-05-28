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

from spine.workflow.subgraph_state import ExplorationSubgraphState


def test_exploration_state_declares_specification_json():
    """ExplorationSubgraphState must declare specification_json so
    _synthesize_specify's return value survives the subgraph."""
    assert "specification_json" in ExplorationSubgraphState.__annotations__


def test_exploration_state_declares_plan_json():
    """Same property, but for the PLAN side (already worked — guard so
    nobody removes it during a future cleanup)."""
    assert "plan_json" in ExplorationSubgraphState.__annotations__
