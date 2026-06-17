"""Tests for ToolOutputTrimmer — smart eviction metadata & AI arg trimming."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from spine.agents.context_editing import ToolOutputTrimmer


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def trimmer() -> ToolOutputTrimmer:
    return ToolOutputTrimmer(max_full_tool_results=2)


# ── 1. _extract_metadata: read_file ──────────────────────────────────────


class TestReadFileMetadata:
    def test_basic_path_and_line_count(self, trimmer: ToolOutputTrimmer):
        content = "line1\ndef foo():\n    pass\nclass Bar:\n    pass\n"
        meta = trimmer._extract_metadata(content, "read_file", {"file_path": "/src/app.py"})
        assert "[read: /src/app.py (5 lines)" in meta
        assert "def foo" in meta
        assert "class Bar" in meta

    def test_no_symbols(self, trimmer: ToolOutputTrimmer):
        content = "# just a comment\n# another comment\n"
        meta = trimmer._extract_metadata(content, "read_file", {"file_path": "/README.md"})
        assert "[read: /README.md (2 lines)" in meta
        assert "no symbols" in meta

    def test_single_line_file(self, trimmer: ToolOutputTrimmer):
        content = "print('hello')"
        meta = trimmer._extract_metadata(content, "read_file", {"file_path": "/x.py"})
        assert "[read: /x.py (1 lines)" in meta


# ── 2. _extract_metadata: execute ────────────────────────────────────────


class TestExecuteMetadata:
    def test_exit_code_found(self, trimmer: ToolOutputTrimmer):
        content = "some output\nexit code: 1"
        meta = trimmer._extract_metadata(content, "execute", {"command": "pytest"})
        assert "[exec: pytest — exit code 1]" in meta

    def test_last_line_summary(self, trimmer: ToolOutputTrimmer):
        content = "Building...\nDone in 3.2s"
        meta = trimmer._extract_metadata(content, "execute", {"command": "make all"})
        assert "[exec: make all — Done in 3.2s]" in meta

    def test_empty_output(self, trimmer: ToolOutputTrimmer):
        content = ""
        meta = trimmer._extract_metadata(content, "execute", {"command": "true"})
        assert "[exec: true]" in meta


# ── 3. _extract_metadata: grep ───────────────────────────────────────────


class TestGrepMetadata:
    def test_pattern_and_counts(self, trimmer: ToolOutputTrimmer):
        content = "src/a.py:match1\nsrc/a.py:match2\nsrc/b.py:match3\n"
        meta = trimmer._extract_metadata(content, "grep", {"pattern": "TODO", "path": "src"})
        assert "[grep: 'TODO' in src — 2 files, 3 lines]" in meta

    def test_no_matches(self, trimmer: ToolOutputTrimmer):
        content = ""
        meta = trimmer._extract_metadata(content, "grep", {"pattern": "FIXME", "path": "."})
        assert "[grep: 'FIXME' in . — 0 files, 0 lines]" in meta


# ── 4. _extract_metadata: write_file / edit_file / glob / ls ─────────────


class TestWriteFileMetadata:
    def test_write_file(self, trimmer: ToolOutputTrimmer):
        meta = trimmer._extract_metadata("ok", "write_file", {"file_path": "/out.py"})
        assert "[written: /out.py]" in meta

    def test_edit_file_short_new(self, trimmer: ToolOutputTrimmer):
        meta = trimmer._extract_metadata(
            "ok", "edit_file", {"file_path": "/mod.py", "new_string": "x = 1"}
        )
        assert "[edited: /mod.py → x = 1]" in meta

    def test_edit_file_long_new_truncated(self, trimmer: ToolOutputTrimmer):
        long_new = "a" * 100
        meta = trimmer._extract_metadata(
            "ok", "edit_file", {"file_path": "/mod.py", "new_string": long_new}
        )
        assert "[edited: /mod.py → " in meta
        assert "..." in meta

    def test_glob(self, trimmer: ToolOutputTrimmer):
        content = "a.py\nb.py\nc.py\n"
        meta = trimmer._extract_metadata(content, "glob", {"pattern": "**/*.py"})
        assert "[glob: '**/*.py' — 3 files]" in meta

    def test_ls(self, trimmer: ToolOutputTrimmer):
        content = "file1.py\nfile2.py\ndir1\n"
        meta = trimmer._extract_metadata(content, "ls", {"path": "/src"})
        assert "[ls: /src — 3 entries]" in meta


# ── 5. _extract_metadata: default fallback ───────────────────────────────


class TestDefaultMetadata:
    def test_unknown_tool(self, trimmer: ToolOutputTrimmer):
        content = (
            "some long output from an unknown tool that exceeds 80 chars by being quite verbose"
        )
        meta = trimmer._extract_metadata(content, "custom_tool", {})
        assert "[evicted(custom_tool):" in meta
        assert "some long output from an unknown tool" in meta


# ── 5b. _extract_metadata: codebase_query (structural lookups) ───────────


class TestCodebaseQueryMetadata:
    def test_find_symbol_records_location(self, trimmer: ToolOutputTrimmer):
        content = '{"file": "spine/ui_api/api.py", "line": 34, "end_line": 120}'
        meta = trimmer._extract_metadata(
            content, "codebase_query", {"action": "find_symbol", "name": "UIApi"}
        )
        assert "codebase_query find_symbol 'UIApi'" in meta
        assert "spine/ui_api/api.py:34" in meta

    def test_search_lists_hits(self, trimmer: ToolOutputTrimmer):
        content = (
            '[{"file": "a.py", "line": 10}, {"file": "b.py", "line": 20}, '
            '{"file": "c.py", "line": 30}]'
        )
        meta = trimmer._extract_metadata(
            content, "codebase_query", {"action": "search", "pattern": "foo|bar"}
        )
        assert "codebase_query search 'foo|bar'" in meta
        assert "3 hit(s)" in meta
        assert "a.py:10" in meta

    def test_empty_search_result(self, trimmer: ToolOutputTrimmer):
        meta = trimmer._extract_metadata(
            content="[]", tool_name="codebase_query",
            tool_args={"action": "search", "pattern": "nope"},
        )
        assert "no results" in meta

    def test_mcp_codebase_index_tool_handled(self, trimmer: ToolOutputTrimmer):
        content = '{"file": "x.py", "line": 5}'
        meta = trimmer._extract_metadata(
            content, "mcp_codebase-index_find_symbol", {"name": "Foo"}
        )
        assert "codebase_query" in meta
        assert "x.py:5" in meta


# ── 5c. Structured (multimodal) content is not dropped on eviction ───────


class TestStructuredContentEviction:
    """Regression: codebase_query/MCP tools return list-of-blocks content;
    eviction used to coerce it to "" → empty ``[evicted(codebase_query): ]``
    stubs that erased the agent's lookup memory (trace 019ed3b8)."""

    def test_list_content_evicted_with_location(self, trimmer: ToolOutputTrimmer):
        structured = [
            {"type": "text", "text": '{"file": "spine/config.py", "line": 42}'}
        ]
        msg = ToolMessage(
            content=structured,
            tool_call_id="tc_q",
            name="codebase_query",
        )
        from spine.agents.context_editing import _stringify_content, extract_metadata

        coerced = _stringify_content(msg.content)
        assert "spine/config.py" in coerced  # not dropped to ""
        meta = extract_metadata(
            coerced, "codebase_query", {"action": "find_symbol", "name": "C"}
        )
        assert "spine/config.py:42" in meta
        assert meta != "[evicted(codebase_query): ]"

    def test_stringify_plain_str_passthrough(self):
        from spine.agents.context_editing import _stringify_content

        assert _stringify_content("hello") == "hello"
        assert _stringify_content(None) == ""
        assert _stringify_content([{"type": "text", "text": "a"}, "b"]) == "a\nb"


# ── 6. _trim_ai_args: write_file content trimming ────────────────────────


class TestAIArgTrimming:
    def test_write_file_content_trimmed(self, trimmer: ToolOutputTrimmer):
        long_content = "x" * 500
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_write",
                        "name": "write_file",
                        "args": {"file_path": "/out.py", "content": long_content},
                    }
                ],
            ),
        ]
        result = trimmer._trim_ai_args(msgs, {"tc_write"})
        assert isinstance(result[0], AIMessage)
        tc = result[0].tool_calls[0]
        assert tc["args"]["content"] == "[500 chars written to /out.py]"

    def test_edit_file_old_and_new_trimmed(self, trimmer: ToolOutputTrimmer):
        old = "a" * 200
        new = "b" * 200
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_edit",
                        "name": "edit_file",
                        "args": {"file_path": "/mod.py", "old_string": old, "new_string": new},
                    }
                ],
            ),
        ]
        result = trimmer._trim_ai_args(msgs, {"tc_edit"})
        tc = result[0].tool_calls[0]
        assert tc["args"]["old_string"] == "[200 chars from /mod.py]"
        assert tc["args"]["new_string"] == "[200 chars → /mod.py]"

    def test_edit_file_short_args_untouched(self, trimmer: ToolOutputTrimmer):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_edit",
                        "name": "edit_file",
                        "args": {
                            "file_path": "/mod.py",
                            "old_string": "short",
                            "new_string": "also short",
                        },
                    }
                ],
            ),
        ]
        result = trimmer._trim_ai_args(msgs, {"tc_edit"})
        tc = result[0].tool_calls[0]
        # Short args should remain unchanged
        assert tc["args"]["old_string"] == "short"
        assert tc["args"]["new_string"] == "also short"


# ── 7. _trim_ai_args does NOT touch read_file args ───────────────────────


class TestReadFileArgsPreserved:
    def test_read_file_args_untouched(self, trimmer: ToolOutputTrimmer):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_read",
                        "name": "read_file",
                        "args": {"file_path": "/src/app.py"},
                    }
                ],
            ),
        ]
        result = trimmer._trim_ai_args(msgs, {"tc_read"})
        tc = result[0].tool_calls[0]
        assert tc["args"] == {"file_path": "/src/app.py"}


# ── 8. _trim_ai_args with no evicted IDs ─────────────────────────────────


class TestNoEvictionNoTrim:
    def test_no_eviction_no_trim(self, trimmer: ToolOutputTrimmer):
        long_content = "x" * 500
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_write",
                        "name": "write_file",
                        "args": {"file_path": "/out.py", "content": long_content},
                    }
                ],
            ),
        ]
        # Not evicting tc_write — args should stay
        result = trimmer._trim_ai_args(msgs, set())
        tc = result[0].tool_calls[0]
        assert tc["args"]["content"] == long_content


# ── 9. _build_tool_call_map ──────────────────────────────────────────────


class TestBuildToolCallMap:
    def test_maps_tool_call_ids(self, trimmer: ToolOutputTrimmer):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc1", "name": "read_file", "args": {"file_path": "/a.py"}},
                    {
                        "id": "tc2",
                        "name": "write_file",
                        "args": {"file_path": "/b.py", "content": "hi"},
                    },
                ],
            ),
            ToolMessage(content="file", tool_call_id="tc1", name="read_file"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc3", "name": "execute", "args": {"command": "ls"}},
                ],
            ),
        ]
        call_map = trimmer._build_tool_call_map(msgs)
        assert call_map["tc1"] == ("read_file", {"file_path": "/a.py"})
        assert call_map["tc2"] == ("write_file", {"file_path": "/b.py", "content": "hi"})
        assert call_map["tc3"] == ("execute", {"command": "ls"})

    def test_skips_non_ai_messages(self, trimmer: ToolOutputTrimmer):
        msgs = [
            HumanMessage(content="hello"),
            ToolMessage(content="result", tool_call_id="tc1", name="read_file"),
        ]
        call_map = trimmer._build_tool_call_map(msgs)
        assert call_map == {}

    def test_skips_calls_without_id(self, trimmer: ToolOutputTrimmer):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "", "name": "read_file", "args": {"file_path": "/a.py"}},
                ],
            ),
        ]
        call_map = trimmer._build_tool_call_map(msgs)
        assert call_map == {}


# ── 10. Integration: awrap_model_call ────────────────────────────────────


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_eviction_flow(self):
        """Integration test: trimmer evicts old results, trims AI args."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=2)

        file_content = "def foo():\n    pass\nclass Bar:\n    pass\n"
        write_content = "x" * 500

        messages = [
            HumanMessage(content="Do stuff"),
            # First tool call: read_file
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc1", "name": "read_file", "args": {"file_path": "/src/app.py"}},
                ],
            ),
            ToolMessage(content=file_content, tool_call_id="tc1", name="read_file"),
            # Second tool call: write_file
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc2",
                        "name": "write_file",
                        "args": {"file_path": "/out.py", "content": write_content},
                    },
                ],
            ),
            ToolMessage(content="ok", tool_call_id="tc2", name="write_file"),
            # Third tool call: execute (should be kept)
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tc3", "name": "execute", "args": {"command": "pytest"}},
                ],
            ),
            ToolMessage(content="exit code: 0", tool_call_id="tc3", name="execute"),
            # Fourth tool call: grep (should be kept — last 2)
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc4",
                        "name": "grep",
                        "args": {"pattern": "TODO", "path": "src"},
                    },
                ],
            ),
            ToolMessage(
                content="src/a.py:TODO here\nsrc/b.py:TODO there\n",
                tool_call_id="tc4",
                name="grep",
            ),
        ]

        # We need a mock handler that captures the overridden request
        captured: dict = {}

        async def mock_handler(request):
            captured["messages"] = request.messages
            return "ok"

        # Build a minimal request-like object
        class FakeRequest:
            def __init__(self, msgs):
                self.messages = msgs

            def override(self, **kwargs):
                """Return a new FakeRequest with overridden messages."""
                new = FakeRequest(kwargs.get("messages", self.messages))
                new._override = True
                return new

        req = FakeRequest(messages)
        await trimmer.awrap_model_call(req, mock_handler)

        result = captured["messages"]

        # tc1 (read_file) should be evicted → structured metadata
        tc1_msg = result[2]
        assert isinstance(tc1_msg, ToolMessage)
        assert "[read: /src/app.py (4 lines)" in tc1_msg.content
        assert "def foo" in tc1_msg.content
        assert "class Bar" in tc1_msg.content

        # tc2 (write_file) should be evicted → structured metadata
        tc2_msg = result[4]
        assert isinstance(tc2_msg, ToolMessage)
        assert "[written: /out.py]" in tc2_msg.content

        # tc3 (execute) should be kept in full
        tc3_msg = result[6]
        assert isinstance(tc3_msg, ToolMessage)
        assert tc3_msg.content == "exit code: 0"

        # tc4 (grep) should be kept in full
        tc4_msg = result[8]
        assert isinstance(tc4_msg, ToolMessage)
        assert "src/a.py:TODO here" in tc4_msg.content

        # AI message for tc2 should have write_file args trimmed
        ai_msg_tc2 = result[3]
        assert isinstance(ai_msg_tc2, AIMessage)
        tc2_call = ai_msg_tc2.tool_calls[0]
        assert tc2_call["args"]["content"] == "[500 chars written to /out.py]"

        # AI message for tc1 should be unchanged (read_file args not trimmed)
        ai_msg_tc1 = result[1]
        assert isinstance(ai_msg_tc1, AIMessage)
        tc1_call = ai_msg_tc1.tool_calls[0]
        assert tc1_call["args"] == {"file_path": "/src/app.py"}

    @pytest.mark.asyncio
    async def test_within_budget_passes_through(self):
        """If tool result count <= max_full_tool_results, no trimming."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=5)

        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"id": "tc1", "name": "read_file", "args": {"file_path": "/a.py"}}],
            ),
            ToolMessage(content="file content", tool_call_id="tc1", name="read_file"),
        ]

        captured: dict = {}

        async def mock_handler(request):
            captured["messages"] = request.messages
            return "ok"

        class FakeRequest:
            def __init__(self, msgs):
                self.messages = msgs

            def override(self, **kwargs):
                new = FakeRequest(kwargs.get("messages", self.messages))
                new._override = True
                return new

        req = FakeRequest(messages)
        await trimmer.awrap_model_call(req, mock_handler)

        # No trimming should have happened
        assert captured["messages"][2].content == "file content"
