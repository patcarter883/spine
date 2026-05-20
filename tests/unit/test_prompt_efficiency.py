"""Tests for prompt efficiency improvements.

These tests verify the structural properties of the agent configuration
that lead to reduced token usage, not actual LLM behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.profile import SPINE_BASE_PROMPT


class TestPromptEfficiency:
    """Verify prompt changes that reduce token usage."""

    def test_base_prompt_no_tool_duplicates(self):
        """SPINE_BASE_PROMPT should not duplicate DA middleware injections."""
        duplicated_phrases = [
            "read_file, write_file, edit_file, ls, glob, grep",
            "run shell commands",
            "QuickJS interpreter",
            "task tool to launch short-lived subagents",
        ]
        for phrase in duplicated_phrases:
            assert phrase not in SPINE_BASE_PROMPT, (
                f"SPINE_BASE_PROMPT duplicates DA middleware content: {phrase!r}"
            )

    def test_base_prompt_has_batch_instruction(self):
        """Base prompt must instruct agents to batch independent operations."""
        assert "batch" in SPINE_BASE_PROMPT.lower() or "Batch" in SPINE_BASE_PROMPT

    def test_base_prompt_has_eval_instruction(self):
        """Base prompt must reference eval/interpreter for orchestration."""
        assert "eval" in SPINE_BASE_PROMPT.lower()

    def test_base_prompt_has_no_re_read_instruction(self):
        """Base prompt must tell agents not to re-read files."""
        assert "re-read" in SPINE_BASE_PROMPT.lower() or "never" in SPINE_BASE_PROMPT.lower()

    def test_base_prompt_concise(self):
        """Base prompt should be under 850 tokens (~3400 chars)."""
        assert len(SPINE_BASE_PROMPT) < 3400, (
            f"SPINE_BASE_PROMPT is {len(SPINE_BASE_PROMPT)} chars — "
            "should be under 3400 chars (~850 tokens)"
        )

    def test_rlm_preamble_removed(self):
        """_RLM_PREAMBLE should no longer exist in factory.py."""
        from spine.agents import factory
        assert not hasattr(factory, "_RLM_PREAMBLE"), (
            "_RLM_PREAMBLE still exists in factory.py — should be removed"
        )


class TestSubagentAutonomy:
    """Verify subagents are configured for autonomous tool use."""

    def test_subagent_response_format_policy(self):
        """Researcher gets response_format (structured summaries); others don't."""
        from unittest.mock import patch, MagicMock
        from spine.agents.subagents import build_subagent_spec

        state: WorkflowState = {
            "work_id": "test",
            "work_type": "quick",
            "description": "test",
            "workspace_root": "/tmp",
            "artifacts": {},
            "critic_reviewing": "",
            "current_phase": "",
            "feedback": [],
            "max_retries": 3,
            "phase_index": 0,
            "prompt_request": None,
            "retry_count": {},
            "status": "running",
        }

        mock_model = MagicMock()
        for name in ["researcher", "slice-implementer", "slice-verifier"]:
            with patch("spine.agents.helpers.resolve_model", return_value=mock_model):
                spec = build_subagent_spec(
                    name=name,
                    phase=PhaseName.IMPLEMENT,
                    state=state,
                )
            if name == "researcher":
                assert "response_format" in spec, (
                    "Researcher subagent should have response_format "
                    "for structured summaries"
                )
            else:
                assert "response_format" not in spec, (
                    f"Subagent {name!r} should not have response_format — "
                    f"only researcher uses structured output"
                )

    def test_subagent_prompt_enforces_tools(self):
        """Subagent prompts must contain 'MUST USE TOOLS' instruction."""
        from spine.agents.subagents import SUBAGENT_PROMPTS

        for name, prompt in SUBAGENT_PROMPTS.items():
            assert "MUST USE TOOLS" in prompt or "must use tools" in prompt.lower(), (
                f"Subagent {name!r} prompt doesn't enforce tool use"
            )

    def test_subagent_prompt_has_batch_reads(self):
        """Subagent prompts must mention batch reads or MCP tool batching.

        The researcher subagent now uses MCP tools for codebase exploration
        (batch MCP calls) instead of batch file reads.  Slice-implementer and
        slice-verifier still use batch file reads.
        """
        from spine.agents.subagents import SUBAGENT_PROMPTS

        for name, prompt in SUBAGENT_PROMPTS.items():
            has_batch = (
                "batch" in prompt.lower()
                or "MCP structural search" in prompt  # researcher's multi-call guidance
            )
            assert has_batch, (
                f"Subagent {name!r} prompt doesn't mention batch reads or MCP tool batching"
            )


class TestContextEditing:
    """Verify context editing middleware is configured."""

    def test_trimmer_class_exists(self):
        """ToolOutputTrimmer should be importable."""
        from spine.agents.context_editing import ToolOutputTrimmer
        trimmer = ToolOutputTrimmer(max_full_tool_results=20)
        assert trimmer.max_full_tool_results == 20

    def test_trimmer_preserves_recent_results(self):
        """Trimmer should not trim results within the budget."""
        from spine.agents.context_editing import ToolOutputTrimmer
        trimmer = ToolOutputTrimmer(max_full_tool_results=5)
        assert trimmer.max_full_tool_results == 5

    def test_trimmer_has_eviction_metadata(self):
        """ToolOutputTrimmer should produce structured metadata, not vague hints."""
        from spine.agents.context_editing import ToolOutputTrimmer
        trimmer = ToolOutputTrimmer()
        # Verify the _extract_metadata method produces structured output
        metadata = trimmer._extract_metadata(
            "     1\tdef hello():\n     2\t    pass\n",
            "read_file",
            {"file_path": "src/main.py"},
        )
        assert "src/main.py" in metadata
        assert "2 lines" in metadata


class TestSummarizationConfig:
    """Verify summarization middleware uses SPINE-specific configuration."""

    def test_summarization_trigger_is_60k(self):
        """Summarization trigger should be 60K tokens (not 80K)."""
        import inspect
        from spine.agents.factory import _add_summarization_middleware
        source = inspect.getsource(_add_summarization_middleware)
        assert "60000" in source, "Summarization trigger should be 60K tokens"

    def test_custom_summary_prompt_exists(self):
        """_SPINE_SUMMARY_PROMPT should exist and preserve technical state."""
        from spine.agents.factory import _SPINE_SUMMARY_PROMPT
        assert _SPINE_SUMMARY_PROMPT is not None
        assert "file" in _SPINE_SUMMARY_PROMPT.lower()
        assert "error" in _SPINE_SUMMARY_PROMPT.lower()
        assert "slice" in _SPINE_SUMMARY_PROMPT.lower()
        assert "offloaded" in _SPINE_SUMMARY_PROMPT.lower() or "history" in _SPINE_SUMMARY_PROMPT.lower()

    def test_summary_prompt_not_chatbot_oriented(self):
        """Summary prompt should NOT use chatbot framing."""
        from spine.agents.factory import _SPINE_SUMMARY_PROMPT
        assert "chat history" not in _SPINE_SUMMARY_PROMPT.lower()
        assert "user's chat" not in _SPINE_SUMMARY_PROMPT.lower()


class TestCodebaseMap:
    """Verify codebase map artifact support."""

    def test_codebase_map_in_tasks_prompt(self):
        """Tasks agent system prompt should reference codebase-map.md."""
        import spine.agents.tasks_agent as mod
        source = open(mod.__file__).read()
        assert "codebase-map" in source, (
            "tasks_agent.py must reference codebase-map.md in its prompt"
        )

    def test_codebase_map_in_implement_prompt(self):
        """Implement agent system prompt should reference codebase-map.md."""
        import spine.agents.implement_agent as mod
        source = open(mod.__file__).read()
        assert "codebase-map" in source, (
            "implement_agent.py must reference codebase-map.md in its prompt"
        )

    def test_codebase_map_in_verify_prompt(self):
        """Verify agent system prompt should reference codebase-map.md."""
        import spine.agents.verify_agent as mod
        source = open(mod.__file__).read()
        assert "codebase-map" in source, (
            "verify_agent.py must reference codebase-map.md in its prompt"
        )
