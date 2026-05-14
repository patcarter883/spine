"""Tests for per-phase model resolution and subagent factory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from spine.config import SpineConfig


class TestPhaseAwareModelResolution:
    """Test SpineConfig.resolve_model(phase=) with providers.phases."""

    def test_default_no_phase(self) -> None:
        cfg = SpineConfig(providers={
            "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
        })
        assert cfg.resolve_model() == "openrouter:default-model"

    def test_phase_override(self) -> None:
        cfg = SpineConfig(providers={
            "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
            "phases": {
                "implement": {"model": "openrouter:impl-model"},
            },
        })
        assert cfg.resolve_model(phase="implement") == "openrouter:impl-model"

    def test_subagent_override_beats_phase(self) -> None:
        """Most specific key wins over parent phase key."""
        cfg = SpineConfig(providers={
            "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
            "phases": {
                "implement": {"model": "openrouter:impl-model"},
                "implement/subagents/slice-implementer": {"model": "openrouter:mini-model"},
            },
        })
        assert cfg.resolve_model(phase="implement/subagents/slice-implementer") == "openrouter:mini-model"

    def test_unknown_subagent_falls_back_to_phase(self) -> None:
        """Subagent path without own key inherits parent phase model."""
        cfg = SpineConfig(providers={
            "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
            "phases": {
                "implement": {"model": "openrouter:impl-model"},
            },
        })
        assert cfg.resolve_model(phase="implement/subagents/unknown") == "openrouter:impl-model"

    def test_unknown_phase_falls_back_to_default(self) -> None:
        """Phase without override uses default provider."""
        cfg = SpineConfig(providers={
            "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
            "phases": {
                "implement": {"model": "openrouter:impl-model"},
            },
        })
        assert cfg.resolve_model(phase="plan") == "openrouter:default-model"


class TestSubagentFactory:
    """Test spine.agents.subagents module."""

    def test_phase_subagents_mapping(self) -> None:
        from spine.agents.subagents import PHASE_SUBAGENTS

        assert "researcher" in PHASE_SUBAGENTS.get("specify", [])
        assert "slice-implementer" in PHASE_SUBAGENTS.get("implement", [])
        assert "slice-verifier" in PHASE_SUBAGENTS.get("verify", [])
        # Phases without subagents
        assert PHASE_SUBAGENTS.get("plan") is None
        assert PHASE_SUBAGENTS.get("critic") is None

    def test_response_models_defined(self) -> None:
        from spine.agents.subagents import SUBAGENT_RESPONSE_MODELS

        assert "researcher" in SUBAGENT_RESPONSE_MODELS
        assert "slice-implementer" in SUBAGENT_RESPONSE_MODELS
        assert "slice-verifier" in SUBAGENT_RESPONSE_MODELS

    def test_tool_restrictions(self) -> None:
        from spine.agents.subagents import SUBAGENT_TOOLS

        # Researcher is read-only
        researcher_tools = SUBAGENT_TOOLS["researcher"]
        assert "write_file" not in researcher_tools
        assert "edit_file" not in researcher_tools
        assert "execute" not in researcher_tools
        assert "read_file" in researcher_tools

        # Implementer has full tools
        impl_tools = SUBAGENT_TOOLS["slice-implementer"]
        assert "write_file" in impl_tools
        assert "execute" in impl_tools

        # Verifier can execute (tests/lint) but not write
        verifier_tools = SUBAGENT_TOOLS["slice-verifier"]
        assert "execute" in verifier_tools
        assert "write_file" not in verifier_tools

    def test_build_subagent_spec_rejects_unknown(self) -> None:
        from spine.agents.subagents import build_subagent_spec
        from spine.models.enums import PhaseName
        from spine.models.state import WorkflowState

        with pytest.raises(ValueError, match="Unknown subagent"):
            build_subagent_spec(
                name="nonexistent",
                phase=PhaseName.IMPLEMENT,
                state=WorkflowState(work_id="test"),
                config=None,
            )
