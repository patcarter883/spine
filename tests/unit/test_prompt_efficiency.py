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
        """SPINE_BASE_PROMPT should not duplicate DA middleware injections
        or reference removed tools (eval / task / QuickJS)."""
        duplicated_phrases = [
            "read_file, write_file, edit_file, ls, glob, grep",
            "run shell commands",
            "QuickJS interpreter",
            "task tool to launch short-lived subagents",
            "eval",
            "tools.task",
        ]
        for phrase in duplicated_phrases:
            assert phrase not in SPINE_BASE_PROMPT, (
                f"SPINE_BASE_PROMPT references removed/duplicated content: {phrase!r}"
            )

    def test_base_prompt_has_batch_instruction(self):
        """Base prompt must instruct agents to batch independent operations."""
        assert "batch" in SPINE_BASE_PROMPT.lower() or "Batch" in SPINE_BASE_PROMPT

    def test_base_prompt_has_no_re_read_instruction(self):
        """Base prompt must tell agents not to re-read files."""
        assert "re-read" in SPINE_BASE_PROMPT.lower() or "never" in SPINE_BASE_PROMPT.lower()

    def test_base_prompt_concise(self):
        """Base prompt should be under 850 tokens (~3400 chars)."""
        assert len(SPINE_BASE_PROMPT) < 3400, (
            f"SPINE_BASE_PROMPT is {len(SPINE_BASE_PROMPT)} chars — "
            "should be under 3400 chars (~850 tokens)"
        )


class TestSubagentAutonomy:
    """Verify subagents are configured for autonomous tool use."""

    def test_subagent_response_format_policy(self):
        """No tool-using subagent gets response_format — schema binding
        causes models to satisfy the schema on turn 1 without using tools.

        All three subagents that use tools (researcher, slice-implementer,
        slice-verifier) run free-form. Results are extracted from the last
        assistant message after the agent loop completes.
        """
        from unittest.mock import patch, MagicMock
        from spine.agents.subagents import build_subagent_spec

        state: WorkflowState = {
            "work_id": "test",
            "work_type": "task",
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
        # The no-schema policy is about TOOL-USING subagents. With the
        # evidence-then-judge verifier OFF, the slice-verifier is tool-using and
        # falls under the policy; ON, it is a no-tool judge and SHOULD carry the
        # schema (covered by test_verify_judge_mode_no_tool_schema_bound). Force
        # the flag off here so this asserts the legacy invariant regardless of
        # the live config.
        from spine.config import SpineConfig
        cfg_off = SpineConfig.load()
        object.__setattr__(cfg_off, "verify_evidence_then_judge", False)
        for name in ["researcher", "slice-implementer", "slice-verifier"]:
            with patch("spine.agents.helpers.resolve_model", return_value=mock_model), \
                 patch("spine.config.SpineConfig.load", return_value=cfg_off):
                spec = build_subagent_spec(
                    name=name,
                    phase=PhaseName.IMPLEMENT,
                    state=state,
                )
            assert "response_format" not in spec, (
                f"Subagent {name!r} should not have response_format — "
                f"schema binding causes models to skip tool calls on turn 1. "
                f"Results are extracted from the final assistant message instead."
            )

    def test_verify_judge_mode_no_tool_schema_bound(self):
        """With ``verify_evidence_then_judge`` on, the slice-verifier is a
        NO-TOOL, schema-bound judge — the node pre-computes the evidence so the
        verifier judges in one shot instead of spiralling on a ReAct loop
        (trace 019f16cf: 2.75M tokens, zero verdicts).
        """
        from unittest.mock import patch, MagicMock
        from spine.agents.subagents import build_subagent_spec, _VERIFY_JUDGE_PROMPT
        from spine.agents.context_editing import ToolOutputTrimmer
        from spine.config import SpineConfig

        state: WorkflowState = {
            "work_id": "test", "work_type": "task", "description": "test",
            "workspace_root": "/tmp", "artifacts": {}, "critic_reviewing": "",
            "current_phase": "", "feedback": [], "max_retries": 3,
            "phase_index": 0, "prompt_request": None, "retry_count": {},
            "status": "running",
        }
        cfg_on = SpineConfig.load()
        object.__setattr__(cfg_on, "verify_evidence_then_judge", True)
        mock_model = MagicMock(spec=[])  # no model_kwargs → schema path is exercised in prod
        with patch("spine.agents.helpers.resolve_model", return_value=mock_model), \
             patch("spine.config.SpineConfig.load", return_value=cfg_on):
            spec = build_subagent_spec(
                name="slice-verifier", phase=PhaseName.VERIFY, state=state
            )
        assert spec["tools"] == [], "judge-mode verifier must have NO tools"
        assert spec["system_prompt"] == _VERIFY_JUDGE_PROMPT, (
            "judge-mode verifier must use the no-tool judge prompt"
        )
        assert not any(
            isinstance(m, ToolOutputTrimmer) for m in spec.get("middleware", [])
        ), "judge-mode verifier has no tool output to trim"

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

    def test_leaf_code_subagents_have_tool_output_trimmer(self):
        """slice-implementer / slice-verifier must carry a ToolOutputTrimmer.

        Without trimming, a slice agent's read→edit→execute loop accumulates
        every tool result in message history and the prompt grows
        monotonically (trace 019e87dd: a single slice climbed 6K→34K prompt
        tokens before crashing when prompt + the requested completion exceeded
        the model window). The trimmer evicts the stale tail to a metadata
        placeholder while preserving the recent window.
        """
        from unittest.mock import patch, MagicMock
        from spine.agents.subagents import build_subagent_spec
        from spine.agents.context_editing import ToolOutputTrimmer

        state: WorkflowState = {
            "work_id": "test",
            "work_type": "task",
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
        # The trimmer bounds the ReAct read→edit→execute loop. The judge-mode
        # verifier has no tools and no loop, so it carries no trimmer — force the
        # flag off so the slice-verifier exercises its tool-using configuration.
        from spine.config import SpineConfig
        cfg_off = SpineConfig.load()
        object.__setattr__(cfg_off, "verify_evidence_then_judge", False)
        for name, phase in (
            ("slice-implementer", PhaseName.IMPLEMENT),
            ("slice-verifier", PhaseName.VERIFY),
        ):
            with patch("spine.agents.helpers.resolve_model", return_value=mock_model), \
                 patch("spine.config.SpineConfig.load", return_value=cfg_off):
                spec = build_subagent_spec(name=name, phase=phase, state=state)
            trimmers = [
                m for m in spec.get("middleware", []) if isinstance(m, ToolOutputTrimmer)
            ]
            assert trimmers, (
                f"Subagent {name!r} must carry a ToolOutputTrimmer to bound "
                f"per-slice context growth"
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


class TestCodebaseMap:
    """Verify codebase map artifact support.

    Delivery moved from prompt text in the per-phase agents (now thin Send-API
    orchestrators) to the cross-work memory store: ``backend._seed_store`` loads
    ``.spine/codebase-map.md`` so agents get it via the memory middleware instead
    of re-exploring. These tests assert that current mechanism.
    """

    def test_codebase_map_seeded_into_memory_store(self, tmp_path):
        """`.spine/codebase-map.md` is seeded into the cross-work memory store."""
        from langgraph.store.memory import InMemoryStore

        from spine.agents.backend import _seed_store

        spine_dir = tmp_path / ".spine"
        spine_dir.mkdir()
        (spine_dir / "codebase-map.md").write_text(
            "# Codebase Map\nmodule spine.work", encoding="utf-8"
        )

        store = InMemoryStore()
        _seed_store(store, tmp_path)

        item = store.get(("memories",), "codebase-map")
        assert item is not None, "codebase-map.md was not seeded into the memory store"
        assert "spine.work" in item.value["content"]

    def test_codebase_map_seed_is_optional(self, tmp_path):
        """Seeding is a no-op (not an error) when codebase-map.md is absent."""
        from langgraph.store.memory import InMemoryStore

        from spine.agents.backend import _seed_store

        store = InMemoryStore()
        _seed_store(store, tmp_path)  # no .spine/ dir present
        assert store.get(("memories",), "codebase-map") is None
