"""Test script to determine LangGraph streaming format with subgraphs=True."""
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    value: str


def node_a(state):
    return {"value": state["value"] + "_a"}


def node_b(state):
    return {"value": state["value"] + "_b"}


graph = (
    StateGraph(State)
    .add_node("node_a", node_a)
    .add_node("node_b", node_b)
    .add_edge(START, "node_a")
    .add_edge("node_a", "node_b")
    .add_edge("node_b", END)
    .compile()
)


async def test_v1():
    """Test v1 format (default) with subgraphs=True + multiple stream modes."""
    print("=" * 60)
    print("V1 FORMAT (default) — stream_mode=['updates', 'messages'], subgraphs=True")
    print("=" * 60)
    async for chunk in graph.astream(
        {"value": "start"},
        stream_mode=["updates", "messages"],
        subgraphs=True,
    ):
        print(f"CHUNK TYPE: {type(chunk).__name__}")
        if isinstance(chunk, tuple):
            print(f"  LENGTH: {len(chunk)}")
            for i, part in enumerate(chunk):
                part_repr = repr(part)[:300]
                print(f"  PART[{i}]: type={type(part).__name__}, repr={part_repr}")
        else:
            print(f"  VALUE: {repr(chunk)[:300]}")
        print()


async def test_v1_no_subgraphs():
    """Test v1 format without subgraphs."""
    print("=" * 60)
    print("V1 FORMAT — stream_mode=['updates'], NO subgraphs")
    print("=" * 60)
    async for chunk in graph.astream(
        {"value": "start"},
        stream_mode=["updates"],
    ):
        print(f"CHUNK TYPE: {type(chunk).__name__}")
        if isinstance(chunk, tuple):
            print(f"  LENGTH: {len(chunk)}")
            for i, part in enumerate(chunk):
                part_repr = repr(part)[:300]
                print(f"  PART[{i}]: type={type(part).__name__}, repr={part_repr}")
        else:
            print(f"  VALUE: {repr(chunk)[:300]}")
        print()


async def test_v2():
    """Test v2 format with subgraphs=True."""
    print("=" * 60)
    print("V2 FORMAT — stream_mode=['updates', 'messages'], subgraphs=True, version='v2'")
    print("=" * 60)
    async for chunk in graph.astream(
        {"value": "start"},
        stream_mode=["updates", "messages"],
        subgraphs=True,
        version="v2",
    ):
        print(f"CHUNK TYPE: {type(chunk).__name__}")
        chunk_repr = repr(chunk)[:500]
        print(f"  VALUE: {chunk_repr}")
        print()


if __name__ == "__main__":
    asyncio.run(test_v1_no_subgraphs())
    asyncio.run(test_v1())
    asyncio.run(test_v2())
