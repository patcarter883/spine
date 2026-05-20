"""Integration tests for SPINE three-layer context efficiency defense.

These tests verify that the assembled middleware stack works together
correctly — not the individual methods (covered by unit tests in
test_context_editing.py and test_prompt_efficiency.py), but the
end-to-end integration of:

1. ToolOutputTrimmer + AI arg trimming — eviction triggers arg compaction
2. ToolSchemaValidator edit_file empty old_string — rebound error
3. Summarization trigger is 60K — factory builds with correct threshold
4. PTC allowlist excludes read_file — interpreter doesn't expose readFile
5. codebase-map prompt enrichment — tasks prompt has Modification Targets
6. Researcher minimum output requirements — prompt includes guardrails
"""

from __future__ import annotations

import inspect

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from spine.agents.context_editing import ToolOutputTrimmer
from spine.agents.factory import _add_summarization_middleware
from spine.agents.interpreter import _PTC_ALLOWLISTS
from spine.agents.subagents import SUBAGENT_PROMPTS
from spine.agents.tool_schema_validator import ToolSchemaValidator
from spine.models.enums import PhaseName


# ── Helpers ──────────────────────────────────────────────────────────────


class _FakeRequest:
    """Minimal request stand-in with override support for middleware tests."""

    def __init__(self, messages: list) -> None:
        self.messages = messages

    def override(self, **kwargs: object) -> _FakeRequest:
        new = _FakeRequest(kwargs.get("messages", self.messages))
        return new


# ── 1. ToolOutputTrimmer + AI arg trimming ──────────────────────────────


class TestTrimmerAndAIArgTrimming:
    """Verify that when tool results are evicted, AI message args are also trimmed."""

    @pytest.mark.asyncio
    async def test_eviction_trims_corresponding_write_file_args(self) -> None:
        """When a write_file ToolMessage is evicted, the AI message's
        write_file content arg should be compacted too."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=1)

        long_content = "x" * 500
        messages = [
            HumanMessage(content="go"),
            # First tool call (will be evicted — beyond the budget of 1)
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_write_1",
                        "name": "write_file",
                        "args": {"file_path": "src/old.py", "content": long_content},
                    },
                ],
            ),
            ToolMessage(content="ok", tool_call_id="tc_write_1", name="write_file"),
            # Second tool call (kept — within budget)
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_write_2",
                        "name": "write_file",
                        "args": {"file_path": "src/new.py", "content": "short"},
                    },
                ],
            ),
            ToolMessage(content="ok", tool_call_id="tc_write_2", name="write_file"),
        ]

        captured: dict = {}

        async def handler(request: _FakeRequest) -> str:
            captured["messages"] = request.messages
            return "ok"

        await trimmer.awrap_model_call(_FakeRequest(messages), handler)

        result = captured["messages"]

        # ToolMessage for tc_write_1 should be evicted to metadata
        evicted_tm = result[2]
        assert isinstance(evicted_tm, ToolMessage)
        assert "[written: src/old.py]" in evicted_tm.content

        # AI message for tc_write_1 should have args trimmed
        ai_msg = result[1]
        assert isinstance(ai_msg, AIMessage)
        trimmed_args = ai_msg.tool_calls[0]["args"]
        assert trimmed_args["content"] == "[500 chars written to src/old.py]"

        # tc_write_2 should remain untouched (within budget)
        kept_tm = result[4]
        assert isinstance(kept_tm, ToolMessage)
        assert kept_tm.content == "ok"

        kept_ai = result[3]
        assert isinstance(kept_ai, AIMessage)
        assert kept_ai.tool_calls[0]["args"]["content"] == "short"

    @pytest.mark.asyncio
    async def test_eviction_trims_edit_file_old_and_new(self) -> None:
        """When an edit_file ToolMessage is evicted, both old_string and
        new_string args in the AI message should be compacted if long."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=1)

        old_str = "a" * 200
        new_str = "b" * 200
        messages = [
            HumanMessage(content="go"),
            # Will be evicted
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_edit_1",
                        "name": "edit_file",
                        "args": {
                            "file_path": "src/core.py",
                            "old_string": old_str,
                            "new_string": new_str,
                        },
                    },
                ],
            ),
            ToolMessage(content="ok", tool_call_id="tc_edit_1", name="edit_file"),
            # Kept
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_read_1",
                        "name": "read_file",
                        "args": {"file_path": "src/core.py"},
                    },
                ],
            ),
            ToolMessage(content="file", tool_call_id="tc_read_1", name="read_file"),
        ]

        captured: dict = {}

        async def handler(request: _FakeRequest) -> str:
            captured["messages"] = request.messages
            return "ok"

        await trimmer.awrap_model_call(_FakeRequest(messages), handler)

        result = captured["messages"]
        ai_msg = result[1]
        assert isinstance(ai_msg, AIMessage)
        trimmed_args = ai_msg.tool_calls[0]["args"]
        assert "200 chars from src/core.py" in trimmed_args["old_string"]
        assert "200 chars → src/core.py" in trimmed_args["new_string"]

    @pytest.mark.asyncio
    async def test_read_file_args_not_trimmed_even_when_evicted(self) -> None:
        """read_file args are small — they should never be trimmed even if
        the ToolMessage is evicted."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=0)

        messages = [
            HumanMessage(content="go"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_read",
                        "name": "read_file",
                        "args": {"file_path": "src/app.py"},
                    },
                ],
            ),
            ToolMessage(content="def foo(): pass", tool_call_id="tc_read", name="read_file"),
        ]

        captured: dict = {}

        async def handler(request: _FakeRequest) -> str:
            captured["messages"] = request.messages
            return "ok"

        await trimmer.awrap_model_call(_FakeRequest(messages), handler)

        result = captured["messages"]
        ai_msg = result[1]
        assert isinstance(ai_msg, AIMessage)
        # read_file args should be unchanged
        assert ai_msg.tool_calls[0]["args"] == {"file_path": "src/app.py"}

        # But the ToolMessage itself should be evicted to metadata
        tm = result[2]
        assert isinstance(tm, ToolMessage)
        assert "[read: src/app.py" in tm.content


# ── 2. ToolSchemaValidator edit_file empty old_string ───────────────────


class TestEditFileEmptyOldString:
    """Verify that edit_file with old_string="" gets caught as an error
    and returned as a ToolMessage(status="error")."""

    @pytest.mark.asyncio
    async def test_empty_old_string_returns_error_tool_message(self) -> None:
        """edit_file with empty old_string should produce a rebound error
        ToolMessage, not execute the tool."""
        validator = ToolSchemaValidator(max_rebound=3)

        tc: dict = {
            "name": "edit_file",
            "id": "tc_edit_empty",
            "args": {"file_path": "src/app.py", "old_string": "", "new_string": "x = 1"},
        }

        class FakeTool:
            name = "edit_file"

            def get_input_schema(self) -> None:
                return None

        class FakeRequest:
            def __init__(self) -> None:
                self.tool = FakeTool()
                self.tool_call = tc  # noqa: RUF012 – test code

        async def handler(request: FakeRequest) -> str:
            # Should NOT be reached — the validator should intercept
            return "should_not_reach"

        result = await validator.awrap_tool_call(FakeRequest(), handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.name == "edit_file"
        assert result.tool_call_id == "tc_edit_empty"
        assert "old_string cannot be empty" in result.content

    @pytest.mark.asyncio
    async def test_missing_old_string_returns_error(self) -> None:
        """edit_file with old_string key absent (defaults to empty) should
        also be caught."""
        validator = ToolSchemaValidator(max_rebound=3)

        tc: dict = {
            "name": "edit_file",
            "id": "tc_edit_missing",
            "args": {"file_path": "src/app.py", "new_string": "x = 1"},
        }

        class FakeTool:
            name = "edit_file"

            def get_input_schema(self) -> None:
                return None

        class FakeRequest:
            def __init__(self) -> None:
                self.tool = FakeTool()
                self.tool_call = tc  # noqa: RUF012

        async def handler(request: FakeRequest) -> str:
            return "should_not_reach"

        result = await validator.awrap_tool_call(FakeRequest(), handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "old_string cannot be empty" in result.content

    @pytest.mark.asyncio
    async def test_nonempty_old_string_passes_through(self) -> None:
        """edit_file with a non-empty old_string should NOT be intercepted."""
        validator = ToolSchemaValidator(max_rebound=3, catch_runtime_errors=False)

        tc: dict = {
            "name": "edit_file",
            "id": "tc_edit_valid",
            "args": {
                "file_path": "src/app.py",
                "old_string": "old_val",
                "new_string": "new_val",
            },
        }

        class FakeTool:
            name = "edit_file"

            def get_input_schema(self) -> None:
                return None

        class FakeRequest:
            def __init__(self) -> None:
                self.tool = FakeTool()
                self.tool_call = tc  # noqa: RUF012

        async def handler(request: FakeRequest) -> str:
            return "executed"

        result = await validator.awrap_tool_call(FakeRequest(), handler)
        assert result == "executed"


# ── 3. Summarization trigger is 60K ─────────────────────────────────────


class TestSummarizationTrigger:
    """Verify the factory builds the summarization middleware with 60K trigger."""

    def test_trigger_value_in_source(self) -> None:
        """The _add_summarization_middleware source must contain 60000."""
        source = inspect.getsource(_add_summarization_middleware)
        assert "60000" in source, (
            'Summarization trigger should be 60K tokens — expected trigger=("tokens", 60000)'
        )

    def test_trigger_not_80k(self) -> None:
        """The old 80K trigger should no longer be present."""
        source = inspect.getsource(_add_summarization_middleware)
        assert "80000" not in source, "Summarization trigger should be 60K, not 80K"

    def test_keep_window_is_20_messages(self) -> None:
        """The keep window should be 20 messages."""
        source = inspect.getsource(_add_summarization_middleware)
        assert 'keep=("messages", 20)' in source or 'keep=("messages", 20)' in source, (
            "Summarization keep window should be 20 messages"
        )

    def test_custom_summary_prompt_used(self) -> None:
        """The factory must pass _SPINE_SUMMARY_PROMPT to the middleware."""
        source = inspect.getsource(_add_summarization_middleware)
        assert "_SPINE_SUMMARY_PROMPT" in source or "summary_prompt" in source, (
            "Summarization middleware must use the custom SPINE summary prompt"
        )


# ── 4. PTC allowlist excludes read_file ─────────────────────────────────


class TestPTCAllowlistExcludesReadFile:
    """Verify that read_file is NOT in any phase's PTC allowlist."""

    @pytest.mark.parametrize(
        "phase_value",
        [
            PhaseName.SPECIFY.value,
            PhaseName.TASKS.value,
            PhaseName.IMPLEMENT.value,
            PhaseName.VERIFY.value,
        ],
    )
    def test_read_file_absent_from_allowlist(self, phase_value: str) -> None:
        """read_file should not appear in any PTC allowlist."""
        allowlist = _PTC_ALLOWLISTS.get(phase_value, [])
        assert "read_file" not in allowlist, (
            f"Phase {phase_value!r} PTC allowlist should not include read_file "
            f"(causes null returns under virtual_mode=True). "
            f"Got: {allowlist}"
        )

    def test_all_phases_have_allowlists(self) -> None:
        """SPECIFY, TASKS, IMPLEMENT, VERIFY must all have PTC entries."""
        for phase in [PhaseName.SPECIFY, PhaseName.TASKS, PhaseName.IMPLEMENT, PhaseName.VERIFY]:
            assert phase.value in _PTC_ALLOWLISTS, (
                f"Phase {phase.value!r} missing from _PTC_ALLOWLISTS"
            )

    def test_task_present_in_all_allowlists(self) -> None:
        """Every phase with a PTC allowlist must include 'task' for
        subagent orchestration."""
        for phase_value, allowlist in _PTC_ALLOWLISTS.items():
            assert "task" in allowlist, (
                f"Phase {phase_value!r} PTC allowlist must include 'task'. Got: {allowlist}"
            )

    def test_no_phase_has_read_file(self) -> None:
        """No phase allowlist should contain read_file (defensive scan)."""
        for phase_value, allowlist in _PTC_ALLOWLISTS.items():
            assert "read_file" not in allowlist, (
                f"Phase {phase_value!r} includes read_file in PTC allowlist"
            )


# ── 5. codebase-map prompt enrichment ───────────────────────────────────


class TestCodebaseMapPromptEnrichment:
    """Verify tasks agent prompt includes Modification Targets section and
    snippet examples for the codebase-map artifact."""

    def test_tasks_prompt_has_modification_targets(self) -> None:
        """The tasks agent prompt must include a 'Modification Targets'
        section that tells the agent to include code snippets around
        change sites in codebase-map.md."""
        import spine.agents.tasks_agent as tasks_mod

        source = inspect.getsource(tasks_mod)
        assert "Modification Targets" in source, (
            "Tasks agent prompt must include a 'Modification Targets' section "
            "in the codebase-map instructions"
        )

    def test_tasks_prompt_has_snippet_example(self) -> None:
        """The tasks agent prompt must include an example code snippet
        showing how to annotate modification targets with line ranges."""
        import spine.agents.tasks_agent as tasks_mod

        source = inspect.getsource(tasks_mod)
        # The example should show a code snippet with line range annotation
        assert "L" in source and "snippet" in source.lower() or "line range" in source.lower(), (
            "Tasks agent prompt must include snippet examples with line ranges "
            "for modification targets"
        )

    def test_tasks_prompt_references_codebase_map(self) -> None:
        """The tasks agent prompt must reference codebase-map.md."""
        import spine.agents.tasks_agent as tasks_mod

        source = inspect.getsource(tasks_mod)
        assert "codebase-map" in source, "Tasks agent prompt must reference codebase-map.md"


# ── 6. Researcher minimum output requirements ───────────────────────────


class TestResearcherMinOutput:
    """Verify the researcher subagent prompt includes minimum output
    requirements to prevent empty-result re-dispatches."""

    def test_researcher_prompt_has_minimum_output_section(self) -> None:
        """The researcher prompt must contain 'Hard limits' section that enforces
        minimum output requirements."""
        prompt = SUBAGENT_PROMPTS["researcher"]
        assert "Hard limits" in prompt, (
            "Researcher prompt must include a Hard limits section"
        )

    def test_researcher_prompt_requires_at_least_2_files(self) -> None:
        """The researcher must use at least 2 MCP tools (or read 2 files) before reporting."""
        prompt = SUBAGENT_PROMPTS["researcher"]
        assert (
            "at least 2 MCP tools" in prompt
            or "at least 2 files" in prompt
        ), "Researcher prompt must require tool use before reporting"

    def test_researcher_prompt_requires_file_map_entry(self) -> None:
        """The researcher must produce a file_map with at least 1 entry."""
        prompt = SUBAGENT_PROMPTS["researcher"]
        assert "file_map" in prompt and "at least 1 entry" in prompt, (
            "Researcher prompt must require file_map with at least 1 entry"
        )

    def test_researcher_prompt_warns_about_empty_results(self) -> None:
        """The researcher prompt must warn that empty results cause
        re-dispatch, wasting time and tokens."""
        prompt = SUBAGENT_PROMPTS["researcher"]
        assert "re-dispatched" in prompt.lower() or "empty results" in prompt.lower(), (
            "Researcher prompt must warn about the consequences of empty results"
        )

    def test_researcher_prompt_requires_2_sentence_summary(self) -> None:
        """The researcher summary must be at least 2 sentences."""
        prompt = SUBAGENT_PROMPTS["researcher"]
        assert "at least 2 sentences" in prompt, (
            "Researcher prompt must require a summary of at least 2 sentences"
        )
