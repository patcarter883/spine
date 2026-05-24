"""Unit tests for agentic garbage collection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from spine.agents.garbage_collector import (
    EVICTION_ANCHOR,
    calculate_safe_eviction,
)


class TestCalculateSafeEviction:
    """Tests for the boundary-preserving eviction algorithm."""

    def test_no_anchor_returns_empty(self):
        """No EVICTION_ANCHOR in messages -> returns []."""
        messages = [
            HumanMessage(content="hello", id="h1"),
            AIMessage(content="hi", id="a1"),
            ToolMessage(content="some result", tool_call_id="t1", id="tm1"),
        ]
        result = calculate_safe_eviction(messages)
        assert result == []

    def test_preserves_boundary_and_siblings(self):
        """Boundary AIMessage and all sibling ToolMessages survive eviction."""
        boundary = AIMessage(
            content="",
            id="bnd",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "commit_findings_and_clear_search",
                    "args": {"note": "found X", "relevant_code": "file.py"},
                },
                {
                    "id": "call_2",
                    "name": "list_files",
                    "args": {"path": "src/"},
                },
            ],
        )
        messages = [
            HumanMessage(content="explore", id="h1"),
            AIMessage(content="old response", id="old_ai"),
            ToolMessage(content="old result", tool_call_id="old_1", id="old_tm"),
            boundary,
            ToolMessage(content=EVICTION_ANCHOR, tool_call_id="call_1", id="anchor_tm"),
            ToolMessage(content="files listed", tool_call_id="call_2", id="sibling_tm"),
        ]

        result = calculate_safe_eviction(messages)
        removed_ids = {rm.id for rm in result}

        # Older messages before the boundary should be removed
        assert "old_ai" in removed_ids
        assert "old_tm" in removed_ids

        # Boundary and siblings must survive
        assert "bnd" not in removed_ids
        assert "anchor_tm" not in removed_ids
        assert "sibling_tm" not in removed_ids

        # HumanMessages are never removed
        assert "h1" not in removed_ids

    def test_removes_prior_messages(self):
        """Messages before the boundary are removed."""
        boundary = AIMessage(
            content="",
            id="bnd",
            tool_calls=[
                {
                    "id": "call_gc",
                    "name": "commit_findings_and_clear_search",
                    "args": {"note": "done", "relevant_code": "x.py"},
                },
            ],
        )
        messages = [
            HumanMessage(content="start", id="h1"),
            AIMessage(content="searching", id="prior_ai"),
            ToolMessage(content="search result", tool_call_id="srch_1", id="prior_tm"),
            boundary,
            ToolMessage(content=EVICTION_ANCHOR, tool_call_id="call_gc", id="anchor_tm"),
        ]

        result = calculate_safe_eviction(messages)
        removed_ids = {rm.id for rm in result}

        # Prior AIMessage and ToolMessage should be removed
        assert "prior_ai" in removed_ids
        assert "prior_tm" in removed_ids

        # HumanMessage should never be removed
        assert "h1" not in removed_ids

        # Boundary and anchor survive
        assert "bnd" not in removed_ids
        assert "anchor_tm" not in removed_ids

    def test_parallel_tools_all_preserved(self):
        """Parallel tool calls in the same turn are all preserved."""
        boundary = AIMessage(
            content="",
            id="bnd",
            tool_calls=[
                {
                    "id": "gc_1",
                    "name": "commit_findings_and_clear_search",
                    "args": {"note": "X", "relevant_code": "a.py"},
                },
                {"id": "read_1", "name": "read_file", "args": {"path": "b.py"}},
                {"id": "grep_1", "name": "grep", "args": {"pattern": "TODO"}},
            ],
        )
        messages = [
            HumanMessage(content="go", id="h1"),
            AIMessage(content="old", id="old_ai"),
            ToolMessage(content="old_tool", tool_call_id="old_t", id="old_tm"),
            boundary,
            ToolMessage(content=EVICTION_ANCHOR, tool_call_id="gc_1", id="gc_tm"),
            ToolMessage(content="file content", tool_call_id="read_1", id="read_tm"),
            ToolMessage(content="grep result", tool_call_id="grep_1", id="grep_tm"),
        ]

        result = calculate_safe_eviction(messages)
        removed_ids = {rm.id for rm in result}

        # Old tools before the boundary are removed
        assert "old_ai" in removed_ids
        assert "old_tm" in removed_ids

        # Parallel sibling tools are NOT removed
        assert "read_tm" not in removed_ids
        assert "grep_tm" not in removed_ids

        # The GC anchor and boundary survive
        assert "gc_tm" not in removed_ids
        assert "bnd" not in removed_ids

        # HumanMessage survives
        assert "h1" not in removed_ids

    def test_empty_messages(self):
        """Empty message list -> []."""
        result = calculate_safe_eviction([])
        assert result == []
