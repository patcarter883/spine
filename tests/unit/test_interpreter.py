"""Tests for SPINE interpreter factory and agent integration.

Verifies that:
1. build_interpreter_middleware() creates middleware with correct PTC allowlists
2. interpreter_enabled() respects env var, config, and library availability
3. Agent builders conditionally include interpreter middleware
4. Phase-specific PTC allowlists are correct
5. System prompts include RLM guidance when interpreter is active
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestBuildInterpreterMiddleware:
    """Tests for the interpreter middleware factory."""

    def test_specify_phase_gets_task_ptc(self) -> None:
        """SPECIFY phase should get 'task' in PTC allowlist."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("specify")
        assert mw._ptc is not None
        ptc_names = [t if isinstance(t, str) else t.name for t in mw._ptc]
        assert "task" in ptc_names

    def test_implement_phase_gets_task_ptc(self) -> None:
        """IMPLEMENT phase should get 'task' in PTC allowlist."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("implement")
        assert mw._ptc is not None

    def test_tasks_phase_gets_task_ptc(self) -> None:
        """TASKS phase should get 'task' in PTC allowlist."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("tasks")
        assert mw._ptc is not None

    def test_verify_phase_gets_task_ptc(self) -> None:
        """VERIFY phase should get 'task' in PTC allowlist."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("verify")
        assert mw._ptc is not None

    def test_critic_phase_gets_no_ptc(self) -> None:
        """CRITIC phase should have no PTC allowlist."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("critic")
        assert mw._ptc is None

    def test_plan_phase_gets_no_ptc(self) -> None:
        """PLAN phase should have no PTC allowlist."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("plan")
        assert mw._ptc is None

    def test_custom_memory_limit(self) -> None:
        """Custom memory_limit should be forwarded to middleware."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("specify", memory_limit=128 * 1024 * 1024)
        assert mw._memory_limit == 128 * 1024 * 1024

    def test_custom_timeout(self) -> None:
        """Custom timeout should be forwarded to middleware."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("specify", timeout=30.0)
        assert mw._timeout == 30.0

    def test_import_error_when_quickjs_missing(self) -> None:
        """Should raise ImportError if langchain-quickjs is not available."""
        from spine.agents.interpreter import build_interpreter_middleware

        with patch.dict("sys.modules", {"langchain_quickjs": None}):
            with pytest.raises(ImportError, match="langchain-quickjs"):
                build_interpreter_middleware("specify")

    def test_snapshot_between_turns_default(self) -> None:
        """Snapshot between turns should default to True."""
        from spine.agents.interpreter import build_interpreter_middleware

        mw = build_interpreter_middleware("specify")
        assert mw._snapshot_between_turns is True


class TestInterpreterEnabled:
    """Tests for the interpreter_enabled() feature flag."""

    def test_disabled_by_default(self) -> None:
        """Interpreter should be disabled when no env var or config is set."""
        from spine.agents.interpreter import interpreter_enabled

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SPINE_INTERPRETER", None)
            with patch("spine.config.SpineConfig") as mock_cls:
                mock_cls.load.return_value.interpreter_enabled = False
                assert interpreter_enabled() is False

    def test_enabled_via_env_var(self) -> None:
        """SPINE_INTERPRETER=1 should enable the interpreter."""
        from spine.agents.interpreter import interpreter_enabled

        with patch.dict(os.environ, {"SPINE_INTERPRETER": "1"}):
            assert interpreter_enabled() is True

    def test_enabled_via_env_true(self) -> None:
        """SPINE_INTERPRETER=true should enable the interpreter."""
        from spine.agents.interpreter import interpreter_enabled

        with patch.dict(os.environ, {"SPINE_INTERPRETER": "true"}):
            assert interpreter_enabled() is True

    def test_env_var_overrides_config(self) -> None:
        """SPINE_INTERPRETER=0 should override config file setting."""
        from spine.agents.interpreter import interpreter_enabled

        with patch.dict(os.environ, {"SPINE_INTERPRETER": "0"}):
            assert interpreter_enabled() is False

    def test_enabled_via_config_file(self) -> None:
        """Config file interpreter_enabled=True should enable interpreter."""
        from spine.agents.interpreter import interpreter_enabled

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SPINE_INTERPRETER", None)
            mock_cfg = MagicMock()
            mock_cfg.interpreter_enabled = True
            with patch("spine.config.SpineConfig") as mock_cls:
                mock_cls.load.return_value = mock_cfg
                assert interpreter_enabled() is True

    def test_disabled_when_quickjs_missing(self) -> None:
        """Should return False if langchain_quickjs is not importable."""
        from spine.agents.interpreter import interpreter_enabled

        with patch.dict(os.environ, {"SPINE_INTERPRETER": "1"}):
            with patch.dict("sys.modules", {"langchain_quickjs": None}):
                assert interpreter_enabled() is False

    def test_env_yes_enables(self) -> None:
        """SPINE_INTERPRETER=yes should enable the interpreter."""
        from spine.agents.interpreter import interpreter_enabled

        with patch.dict(os.environ, {"SPINE_INTERPRETER": "yes"}):
            assert interpreter_enabled() is True

    def test_env_random_value_disables(self) -> None:
        """SPINE_INTERPRETER=maybe should disable the interpreter."""
        from spine.agents.interpreter import interpreter_enabled

        with patch.dict(os.environ, {"SPINE_INTERPRETER": "maybe"}):
            assert interpreter_enabled() is False


def _make_state(**overrides: object) -> dict:
    """Create a minimal WorkflowState dict for agent builder tests."""
    base: dict = {
        "work_id": "test-1",
        "work_type": "spec",
        "description": "Build a REST API",
        "current_phase": "specify",
        "phase_index": 0,
        "retry_count": {},
        "max_retries": 3,
        "artifacts": {},
        "feedback": [],
        "status": "running",
        "prompt_request": None,
        "critic_reviewing": "",
        "workspace_root": "/tmp/test",
    }
    base.update(overrides)
    return base


class TestAgentBuilderIntegration:
    """Tests that agent builders conditionally include interpreter middleware."""

    @patch("spine.agents.interpreter.interpreter_enabled", return_value=False)
    @patch("spine.agents.specify_agent.resolve_model", return_value="openai:gpt-4o-mini")
    def test_specify_agent_no_interpreter_when_disabled(
        self, mock_model: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should not include interpreter middleware when disabled."""
        from deepagents import create_deep_agent

        from spine.agents.specify_agent import build_specify_agent

        with patch("deepagents.create_deep_agent", wraps=create_deep_agent) as mock_da:
            mock_da.return_value = MagicMock()
            build_specify_agent(_make_state())
            call_kwargs = mock_da.call_args[1]
            assert call_kwargs["middleware"] == []

    @patch("spine.agents.interpreter.interpreter_enabled", return_value=True)
    @patch("spine.agents.specify_agent.resolve_model", return_value="openai:gpt-4o-mini")
    def test_specify_agent_includes_interpreter_when_enabled(
        self, mock_model: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should include interpreter middleware when enabled."""
        from deepagents import create_deep_agent

        from spine.agents.specify_agent import build_specify_agent

        with patch("deepagents.create_deep_agent", wraps=create_deep_agent) as mock_da:
            mock_da.return_value = MagicMock()
            build_specify_agent(_make_state())
            call_kwargs = mock_da.call_args[1]
            assert len(call_kwargs["middleware"]) == 1

    @patch("spine.agents.interpreter.interpreter_enabled", return_value=True)
    @patch("spine.agents.specify_agent.resolve_model", return_value="openai:gpt-4o-mini")
    def test_specify_agent_rlm_prompt_when_enabled(
        self, mock_model: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should include RLM interpreter guidance in prompt when enabled."""
        from deepagents import create_deep_agent

        from spine.agents.specify_agent import build_specify_agent

        with patch("deepagents.create_deep_agent", wraps=create_deep_agent) as mock_da:
            mock_da.return_value = MagicMock()
            build_specify_agent(_make_state())
            call_kwargs = mock_da.call_args[1]
            prompt = call_kwargs["system_prompt"]
            assert "Interpreter Workspace" in prompt
            assert "RLM Pattern" in prompt
            assert "tools.task" in prompt

    @patch("spine.agents.interpreter.interpreter_enabled", return_value=False)
    @patch("spine.agents.specify_agent.resolve_model", return_value="openai:gpt-4o-mini")
    def test_specify_agent_no_rlm_prompt_when_disabled(
        self, mock_model: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should NOT include RLM guidance in prompt when disabled."""
        from deepagents import create_deep_agent

        from spine.agents.specify_agent import build_specify_agent

        with patch("deepagents.create_deep_agent", wraps=create_deep_agent) as mock_da:
            mock_da.return_value = MagicMock()
            build_specify_agent(_make_state())
            call_kwargs = mock_da.call_args[1]
            prompt = call_kwargs["system_prompt"]
            assert "Interpreter Workspace" not in prompt

    @patch("spine.agents.interpreter.interpreter_enabled", return_value=True)
    @patch("spine.agents.implement_agent.resolve_model", return_value="openai:gpt-4o-mini")
    def test_implement_agent_includes_parallel_execution_guidance(
        self, mock_model: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """implement_agent should include parallel execution guidance in prompt."""
        from deepagents import create_deep_agent

        from spine.agents.implement_agent import build_implement_agent

        with patch("deepagents.create_deep_agent", wraps=create_deep_agent) as mock_da:
            mock_da.return_value = MagicMock()
            build_implement_agent(_make_state(current_phase="implement"))
            call_kwargs = mock_da.call_args[1]
            prompt = call_kwargs["system_prompt"]
            assert "Promise.all" in prompt
            assert "Error handling" in prompt
            assert "Progress tracking" in prompt

    @patch("spine.agents.interpreter.interpreter_enabled", return_value=True)
    @patch("spine.agents.tasks_agent.resolve_model", return_value="openai:gpt-4o-mini")
    def test_tasks_agent_includes_decomposition_guidance(
        self, mock_model: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """tasks_agent should include decomposition guidance in prompt."""
        from deepagents import create_deep_agent

        from spine.agents.tasks_agent import build_tasks_agent

        with patch("deepagents.create_deep_agent", wraps=create_deep_agent) as mock_da:
            mock_da.return_value = MagicMock()
            build_tasks_agent(_make_state(current_phase="tasks"))
            call_kwargs = mock_da.call_args[1]
            prompt = call_kwargs["system_prompt"]
            assert "Dependency sorting" in prompt
            assert "Parallel research" in prompt

    @patch("spine.agents.interpreter.interpreter_enabled", return_value=True)
    @patch("spine.agents.verify_agent.resolve_model", return_value="openai:gpt-4o-mini")
    def test_verify_agent_includes_verification_guidance(
        self, mock_model: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """verify_agent should include multi-slice verification guidance in prompt."""
        from deepagents import create_deep_agent

        from spine.agents.verify_agent import build_verify_agent

        with patch("deepagents.create_deep_agent", wraps=create_deep_agent) as mock_da:
            mock_da.return_value = MagicMock()
            build_verify_agent(_make_state(current_phase="verify"))
            call_kwargs = mock_da.call_args[1]
            prompt = call_kwargs["system_prompt"]
            assert "Multi-slice verification" in prompt
            assert "Result aggregation" in prompt


class TestPTCAllowlists:
    """Tests that PTC allowlists are correct per phase."""

    def test_all_orchestration_phases_have_task(self) -> None:
        """SPECIFY, TASKS, IMPLEMENT, VERIFY should all have 'task' in PTC."""
        from spine.agents.interpreter import _PTC_ALLOWLISTS

        orchestration_phases = ["specify", "tasks", "implement", "verify"]
        for phase in orchestration_phases:
            assert phase in _PTC_ALLOWLISTS, f"Phase {phase!r} missing from PTC allowlists"
            assert "task" in _PTC_ALLOWLISTS[phase], (
                f"Phase {phase!r} should have 'task' in PTC allowlist"
            )

    def test_review_phases_have_no_ptc(self) -> None:
        """CRITIC and PLAN should have no PTC allowlist."""
        from spine.agents.interpreter import _PTC_ALLOWLISTS

        assert "critic" not in _PTC_ALLOWLISTS
        assert "plan" not in _PTC_ALLOWLISTS
