"""Integration test for specification_json propagation through the
exploration subgraph state.

A minimal ``StateGraph(ExplorationSubgraphState)`` with a single node that
returns ``{"specification_json": "..."}`` must surface the value in the
compiled graph's final state. Before the channel-declaration fix, this
test would fail because LangGraph silently dropped the undeclared
``specification_json`` key.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END, START, StateGraph

from spine.workflow.subgraph_state import ExplorationSubgraphState


@pytest.mark.asyncio
async def test_specification_json_survives_subgraph_state():
    PAYLOAD = '{"summary": "hello"}'

    def _node(state):
        return {"specification_json": PAYLOAD, "phase_status": "success"}

    builder = StateGraph(ExplorationSubgraphState)
    builder.add_node("synthesize", _node)
    builder.add_edge(START, "synthesize")
    builder.add_edge("synthesize", END)
    graph = builder.compile()

    result = await graph.ainvoke({})
    assert result.get("specification_json") == PAYLOAD
    assert result.get("phase_status") == "success"
