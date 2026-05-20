"""Tests for SPINE interpreter factory and agent integration.

Verifies that:
1. build_interpreter_middleware() creates middleware with correct PTC allowlists
2. interpreter_enabled() respects env var, config, and library availability
3. Agent builders use the shared factory with context engineering
4. Phase-specific PTC allowlists are correct
5. Skills are loaded for the right phases
6. RLM guidance is in skills, not inline in system prompt
7. FilesystemMiddleware uses SPINE custom prompt (relative paths)
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


def _mock_create_agent() -> MagicMock:
    """Create a mock create_agent return value.

    The mock agent has ``.with_config()`` that returns itself,
    matching the chained call pattern in the factory.
    """
    mock_agent = MagicMock()
    mock_agent.with_config.return_value = mock_agent
    return mock_agent


# ── Shared patch decorators for agent builder tests ──────────────────────
#
# Agent construction hits real LLM constructors (ChatOpenRouter,
# ChatOpenAI) during model resolution.  Patching at the constructor level
# prevents API key validation failures while still exercising the
# middleware assembly logic.
#
# Patch order (bottom-to-top in decorator stack, innermost first):
# 1. ChatOpenRouter.__init__ — prevents OpenRouter API key check
# 2. create_agent — captures middleware and system_prompt args
# 3. interpreter_enabled — controls CodeInterpreterMiddleware presence


class TestAgentBuilderIntegration:
    """Tests that agent builders use the shared factory correctly."""

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_specify_agent_no_interpreter_when_disabled(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should not include interpreter middleware when disabled."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.specify_agent import build_specify_agent

        build_specify_agent(_make_state())
        call_kwargs = mock_ca.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        assert not any(
            "Interpreter" in type(m).__name__
            for m in middleware
        )

    @patch("spine.agents.factory.interpreter_enabled", return_value=True)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_specify_agent_includes_interpreter_when_enabled(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should include interpreter middleware when enabled."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.specify_agent import build_specify_agent

        build_specify_agent(_make_state())
        call_kwargs = mock_ca.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        assert any(
            "Interpreter" in type(m).__name__
            for m in middleware
        )

    @patch("spine.agents.factory.interpreter_enabled", return_value=True)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_specify_agent_rlm_skill_loaded_when_enabled(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should load the rlm-pattern skill when interpreter is enabled."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.specify_agent import build_specify_agent

        build_specify_agent(_make_state())
        call_kwargs = mock_ca.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        skills_mw = next(
            (m for m in middleware if "SkillsMiddleware" in type(m).__name__),
            None,
        )
        assert skills_mw is not None, "SkillsMiddleware should be in the stack when skills are present"
        sources = getattr(skills_mw, "_sources", []) or getattr(skills_mw, "sources", [])
        skill_names = [s.split("/")[-1] for s in sources]
        assert "rlm-pattern" in skill_names

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_specify_agent_no_rlm_skill_when_disabled(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should NOT load rlm-pattern skill when interpreter is disabled."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.specify_agent import build_specify_agent

        build_specify_agent(_make_state())
        call_kwargs = mock_ca.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        skills_mws = [m for m in middleware if "SkillsMiddleware" in type(m).__name__]
        for smw in skills_mws:
            sources = getattr(smw, "_sources", []) or getattr(smw, "sources", [])
            skill_names = [s.split("/")[-1] for s in sources]
            assert "rlm-pattern" not in skill_names

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_specify_agent_no_rlm_in_prompt(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should NOT have RLM guidance inline in system_prompt."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.specify_agent import build_specify_agent

        build_specify_agent(_make_state())
        call_kwargs = mock_ca.call_args[1]
        prompt = call_kwargs["system_prompt"]
        assert "Interpreter Workspace" not in prompt

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_specify_agent_loads_spec_writing_skill(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """specify_agent should load the spec-writing skill."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.specify_agent import build_specify_agent

        build_specify_agent(_make_state())
        call_kwargs = mock_ca.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        skills_mw = next(
            (m for m in middleware if "SkillsMiddleware" in type(m).__name__),
            None,
        )
        if skills_mw is not None:
            sources = getattr(skills_mw, "_sources", []) or getattr(skills_mw, "sources", [])
            skill_names = [s.split("/")[-1] for s in sources]
            assert "spec-writing" in skill_names

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_tasks_agent_loads_decomposition_skill(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """tasks_agent should load the feature-slice-decomposition skill."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.tasks_agent import build_tasks_agent

        build_tasks_agent(_make_state(current_phase="tasks"))
        call_kwargs = mock_ca.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        skills_mw = next(
            (m for m in middleware if "SkillsMiddleware" in type(m).__name__),
            None,
        )
        if skills_mw is not None:
            sources = getattr(skills_mw, "_sources", []) or getattr(skills_mw, "sources", [])
            skill_names = [s.split("/")[-1] for s in sources]
            assert "feature-slice-decomposition" in skill_names

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_verify_agent_loads_code_review_skill(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """verify_agent should load the code-review skill."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.verify_agent import build_verify_agent

        build_verify_agent(_make_state(current_phase="verify"))
        call_kwargs = mock_ca.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        skills_mw = next(
            (m for m in middleware if "SkillsMiddleware" in type(m).__name__),
            None,
        )
        if skills_mw is not None:
            sources = getattr(skills_mw, "_sources", []) or getattr(skills_mw, "sources", [])
            skill_names = [s.split("/")[-1] for s in sources]
            assert "code-review" in skill_names

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_implement_agent_requests_summarization(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """implement_agent should request summarization middleware."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.implement_agent import build_implement_agent

        build_implement_agent(_make_state(current_phase="implement"))
        assert mock_ca.called

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_agent_context_schema_is_spine_context(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """All agents should use SpineContext as context_schema."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.specify_agent import build_specify_agent
        from spine.agents.context import SpineContext

        build_specify_agent(_make_state())
        call_kwargs = mock_ca.call_args[1]
        assert call_kwargs["context_schema"] is SpineContext

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_artifacts_referenced_by_path_not_inlined(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """System prompt should reference artifacts by path, not inline content."""
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.implement_agent import build_implement_agent

        state = _make_state(
            current_phase="implement",
            artifacts={
                "specify": {"specification.md": "# A very long spec\n" * 100},
                "plan": {"plan.md": "# A very long plan\n" * 100},
            },
        )

        build_implement_agent(state)
        call_kwargs = mock_ca.call_args[1]
        prompt = call_kwargs["system_prompt"]
        assert "A very long spec" not in prompt
        assert "A very long plan" not in prompt
        assert ".spine/artifacts/test-1/specify/" in prompt
        assert ".spine/artifacts/test-1/plan/" in prompt

    @patch("spine.agents.factory.interpreter_enabled", return_value=False)
    @patch("spine.agents.factory.create_agent")
    @patch("langchain_openrouter.chat_models.ChatOpenRouter.__init__", return_value=None)
    def test_filesystem_middleware_uses_spine_prompt(
        self, mock_or: MagicMock, mock_ca: MagicMock, mock_enabled: MagicMock
    ) -> None:
        """Specify agent should use custom tools (no FilesystemMiddleware).

        Since specify now uses skip_filesystem_middleware=True and injects
        purpose-built tools (read_work_context, write_specification) via
        extra_tools, FilesystemMiddleware should NOT be in the stack.
        The custom tools replace all filesystem access.
        """
        mock_ca.return_value = _mock_create_agent()
        from spine.agents.specify_agent import build_specify_agent

        build_specify_agent(_make_state())
        call_kwargs = mock_ca.call_args[1]

        # FilesystemMiddleware is intentionally absent
        middleware = call_kwargs.get("middleware", [])
        fs_mw = next(
            (m for m in middleware if "FilesystemMiddleware" in type(m).__name__),
            None,
        )
        assert fs_mw is None, (
            "FilesystemMiddleware should NOT be in the specify stack — "
            "custom tools (read_work_context, write_specification) replace it"
        )

        # Custom tools should be injected via create_agent tools= argument
        tools = call_kwargs.get("tools", [])
        tool_names = {getattr(t, "name", "") for t in tools}
        assert "read_work_context" in tool_names, "read_work_context tool must be present"
        assert "write_specification" in tool_names, "write_specification tool must be present"


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
