"""Tests for agent provider system.

Covers:
- AgentResult dataclass
- OpenCodeAgentProvider command building
- CodexAgentProvider command building
- ClaudeCodeAgentProvider CLI command building
- AgentFallbackChain with mock providers
- create_agent_provider factory
- create_agent_chain_from_config
- SwarmAgent integration with agent_provider
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from spine.providers.agents import (
    AgentResult,
    AgentProvider,
    OpenCodeAgentProvider,
    CodexAgentProvider,
    ClaudeCodeAgentProvider,
    AgentFallbackChain,
    create_agent_provider,
    create_agent_chain_from_config,
    _git_changed_files,
)
from spine.providers.base import ProviderType


# ── AgentResult ───────────────────────────────────────────────────────

class TestAgentResult:
    def test_success_when_zero_exit_and_no_error(self):
        r = AgentResult(output="done", exit_code=0)
        assert r.success is True

    def test_failure_on_nonzero_exit(self):
        r = AgentResult(output="", exit_code=1, error="bad")
        assert r.success is False

    def test_failure_on_error_set(self):
        r = AgentResult(output="", exit_code=0, error="something wrong")
        assert r.success is False

    def test_to_dict(self):
        r = AgentResult(output="hello", exit_code=0, files_changed=["a.py"])
        d = r.to_dict()
        assert d["output"] == "hello"
        assert d["exit_code"] == 0
        assert d["files_changed"] == ["a.py"]
        assert d["success"] is True

    def test_default_empty_fields(self):
        r = AgentResult()
        assert r.output == ""
        assert r.files_changed == []
        assert r.metadata == {}
        assert r.error is None


# ── OpenCodeAgentProvider ─────────────────────────────────────────────

class TestOpenCodeAgentProvider:
    def test_name(self):
        p = OpenCodeAgentProvider()
        assert p.name == "opencode"

    def test_provider_type(self):
        p = OpenCodeAgentProvider()
        assert p.provider_type == ProviderType.AGENT

    def test_default_mode_is_run(self):
        p = OpenCodeAgentProvider()
        assert p.mode == "run"

    def test_configure_sets_mode(self):
        p = OpenCodeAgentProvider()
        p.configure({"mode": "serve", "model": "openrouter/google/gemini-2.5-flash"})
        assert p.mode == "serve"

    def test_build_run_command_minimal(self):
        p = OpenCodeAgentProvider()
        cmd = p._build_run_command("Fix the bug")
        assert cmd[0] == "opencode"
        assert cmd[1] == "run"
        assert "--format" in cmd
        assert "json" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert cmd[-1] == "Fix the bug"

    def test_build_run_command_with_model(self):
        p = OpenCodeAgentProvider()
        p.configure({"model": "openrouter/google/gemini-2.5-flash"})
        cmd = p._build_run_command("Add login page")
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "openrouter/google/gemini-2.5-flash"

    def test_build_run_command_with_agent_type(self):
        p = OpenCodeAgentProvider()
        cmd = p._build_run_command("Analyze code", agent="plan")
        assert "--agent" in cmd
        idx = cmd.index("--agent")
        assert cmd[idx + 1] == "plan"

    def test_build_run_command_no_auto_approve(self):
        p = OpenCodeAgentProvider()
        p.configure({"auto_approve": False})
        cmd = p._build_run_command("Do stuff")
        assert "--dangerously-skip-permissions" not in cmd

    def test_build_run_command_with_files(self):
        p = OpenCodeAgentProvider()
        cmd = p._build_run_command("Fix bug", files=["main.py", "test.py"])
        assert "--file" in cmd
        assert "main.py" in cmd
        assert "test.py" in cmd

    def test_build_run_command_with_workdir(self):
        p = OpenCodeAgentProvider()
        cmd = p._build_run_command("Fix bug", workdir="/tmp/project")
        assert "--dir" in cmd
        assert "/tmp/project" in cmd

    def test_is_available_with_opencode(self):
        p = OpenCodeAgentProvider()
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            assert p.is_available() is True

    def test_is_available_without_opencode(self):
        p = OpenCodeAgentProvider()
        with patch("shutil.which", return_value=None):
            assert p.is_available() is False

    def test_enabled_respects_config(self):
        p = OpenCodeAgentProvider()
        p.configure({"enabled": False})
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            assert p.enabled is False

    def test_execute_run_missing_binary(self):
        p = OpenCodeAgentProvider()
        with patch("subprocess.run", side_effect=FileNotFoundError("opencode not found")):
            result = p._execute_run("Fix bug", workdir="/tmp")
            assert result.success is False
            assert "not found" in result.error

    def test_execute_run_timeout(self):
        p = OpenCodeAgentProvider()
        with patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("opencode", 300)):
            result = p._execute_run("Fix bug", timeout=300)
            assert result.success is False
            assert "Timeout" in result.error

    def test_execute_delegates_to_run_by_default(self):
        p = OpenCodeAgentProvider()
        with patch.object(p, "_execute_run", return_value=AgentResult(output="ok")) as mock:
            p.execute("Fix bug", workdir="/tmp")
            mock.assert_called_once()

    def test_execute_delegates_to_serve(self):
        p = OpenCodeAgentProvider()
        p.configure({"mode": "serve"})
        with patch.object(p, "_execute_serve", return_value=AgentResult(output="ok")) as mock:
            p.execute("Fix bug", workdir="/tmp")
            mock.assert_called_once()

    def test_execute_delegates_to_acp(self):
        p = OpenCodeAgentProvider()
        with patch.object(p, "_execute_acp", return_value=AgentResult(output="ok")) as mock:
            p.execute("Fix bug", workdir="/tmp", mode="acp")
            mock.assert_called_once()

    def test_validate_delegates_to_is_available(self):
        p = OpenCodeAgentProvider()
        with patch.object(p, "is_available", return_value=True):
            assert p.validate() is True


# ── CodexAgentProvider ────────────────────────────────────────────────

class TestCodexAgentProvider:
    def test_name(self):
        p = CodexAgentProvider()
        assert p.name == "codex"

    def test_build_command_defaults(self):
        p = CodexAgentProvider()
        cmd = p._build_command("Add login")
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--sandbox" in cmd
        assert "workspace-write" in cmd
        assert "--approval-mode" in cmd
        assert "full-auto" in cmd
        assert cmd[-1] == "Add login"

    def test_build_command_custom_model(self):
        p = CodexAgentProvider()
        p.configure({"model": "o3", "sandbox": "read-only"})
        cmd = p._build_command("Analyze code")
        assert "--model" in cmd
        assert "o3" in cmd
        assert "--sandbox" in cmd
        assert "read-only" in cmd

    def test_is_available(self):
        p = CodexAgentProvider()
        with patch("shutil.which", return_value="/usr/local/bin/codex"):
            assert p.is_available() is True
        with patch("shutil.which", return_value=None):
            assert p.is_available() is False


# ── ClaudeCodeAgentProvider ──────────────────────────────────────────

class TestClaudeCodeAgentProvider:
    def test_name(self):
        p = ClaudeCodeAgentProvider()
        assert p.name == "claude-code"

    def test_build_cli_command(self):
        p = ClaudeCodeAgentProvider()
        cmd = p._build_cli_command("Add a test")
        assert cmd[0] == "claude"
        assert "--print" in cmd
        assert cmd[-1] == "Add a test"

    def test_build_cli_command_with_model(self):
        p = ClaudeCodeAgentProvider()
        p.configure({"model": "claude-sonnet-4-20250514", "max_turns": 20})
        cmd = p._build_cli_command("Fix bug")
        assert "--model" in cmd
        assert "--max-turns" in cmd
        assert "20" in cmd

    def test_build_cli_command_with_allowed_tools(self):
        p = ClaudeCodeAgentProvider()
        cmd = p._build_cli_command("Code", allowed_tools=["read", "write", "bash"])
        assert "--allowedTools" in cmd
        assert "read" in cmd
        assert "write" in cmd

    def test_is_available_with_cli(self):
        p = ClaudeCodeAgentProvider()
        with patch.object(p, "_sdk_available", return_value=False):
            with patch("shutil.which", return_value="/usr/bin/claude"):
                assert p.is_available() is True

    def test_is_available_with_sdk(self):
        p = ClaudeCodeAgentProvider()
        with patch.object(p, "_sdk_available", return_value=True):
            assert p.is_available() is True

    def test_is_available_with_nothing(self):
        p = ClaudeCodeAgentProvider()
        with patch.object(p, "_sdk_available", return_value=False):
            with patch("shutil.which", return_value=None):
                assert p.is_available() is False

    def test_execute_falls_back_to_cli_when_no_sdk(self):
        p = ClaudeCodeAgentProvider()
        with patch.object(p, "_sdk_available", return_value=False):
            with patch.object(p, "_execute_cli", return_value=AgentResult(output="ok")) as mock:
                with patch("shutil.which", return_value="/usr/bin/claude"):
                    result = p.execute("Fix bug")
                    mock.assert_called_once()

    def test_execute_returns_error_when_nothing_available(self):
        p = ClaudeCodeAgentProvider()
        with patch.object(p, "_sdk_available", return_value=False):
            with patch("shutil.which", return_value=None):
                result = p.execute("Fix bug")
                assert result.success is False
                assert "not found" in result.error.lower() or "neither" in result.error.lower()


# ── AgentFallbackChain ────────────────────────────────────────────────

class TestAgentFallbackChain:
    def _make_provider(self, name: str, available: bool, output: str = "ok"):
        """Create a mock AgentProvider."""
        provider = MagicMock(spec=AgentProvider)
        provider.name = name
        provider.enabled = available
        provider.execute.return_value = AgentResult(output=output, exit_code=0)
        return provider

    def test_add_and_active_provider(self):
        chain = AgentFallbackChain()
        p1 = self._make_provider("opencode", True)
        chain.add(p1, priority=0)
        assert chain.active_provider is p1

    def test_active_provider_skips_disabled(self):
        chain = AgentFallbackChain()
        p1 = self._make_provider("opencode", False)
        p2 = self._make_provider("codex", True)
        chain.add(p1, priority=0)
        chain.add(p2, priority=1)
        assert chain.active_provider is p2

    def test_active_provider_none_when_all_disabled(self):
        chain = AgentFallbackChain()
        p1 = self._make_provider("opencode", False)
        chain.add(p1, priority=0)
        assert chain.active_provider is None

    def test_execute_uses_first_available(self):
        chain = AgentFallbackChain()
        p1 = self._make_provider("opencode", True, output="from-opencode")
        chain.add(p1, priority=0)
        result = chain.execute("Fix bug")
        assert result.output == "from-opencode"

    def test_execute_falls_back_on_failure(self):
        chain = AgentFallbackChain()
        p1 = self._make_provider("opencode", True)
        p1.execute.return_value = AgentResult(output="", exit_code=1, error="crashed")
        p2 = self._make_provider("codex", True, output="from-codex")
        chain.add(p1, priority=0)
        chain.add(p2, priority=1)
        result = chain.execute("Fix bug")
        assert result.output == "from-codex"

    def test_execute_returns_error_when_all_fail(self):
        chain = AgentFallbackChain()
        p1 = self._make_provider("opencode", True)
        p1.execute.return_value = AgentResult(output="", exit_code=1, error="bad")
        p2 = self._make_provider("codex", True)
        p2.execute.return_value = AgentResult(output="", exit_code=1, error="worse")
        chain.add(p1, priority=0)
        chain.add(p2, priority=1)
        result = chain.execute("Fix bug")
        assert result.success is False

    def test_execute_no_providers_at_all(self):
        chain = AgentFallbackChain()
        result = chain.execute("Fix bug")
        assert result.success is False
        assert "No agent providers available" in result.error

    def test_execute_skips_disabled_providers(self):
        chain = AgentFallbackChain()
        p1 = self._make_provider("opencode", False)
        p2 = self._make_provider("codex", True, output="from-codex")
        chain.add(p1, priority=0)
        chain.add(p2, priority=1)
        result = chain.execute("Fix bug")
        assert result.output == "from-codex"
        p1.execute.assert_not_called()


# ── Factory functions ─────────────────────────────────────────────────

class TestCreateAgentProvider:
    def test_create_opencode(self):
        p = create_agent_provider("opencode")
        assert isinstance(p, OpenCodeAgentProvider)

    def test_create_codex(self):
        p = create_agent_provider("codex")
        assert isinstance(p, CodexAgentProvider)

    def test_create_claude_code(self):
        p = create_agent_provider("claude-code")
        assert isinstance(p, ClaudeCodeAgentProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown agent provider"):
            create_agent_provider("nonexistent")

    def test_create_with_config(self):
        p = create_agent_provider("opencode", {"mode": "serve", "model": "test"})
        assert p.mode == "serve"


class TestCreateAgentChainFromConfig:
    def test_builds_chain(self):
        configs = [
            {"name": "opencode", "type": "opencode", "priority": 0, "config": {"model": "test"}},
            {"name": "codex", "type": "codex", "priority": 1, "config": {}},
        ]
        chain = create_agent_chain_from_config(configs)
        assert chain.active_provider is not None
        assert chain.active_provider.name == "opencode"

    def test_empty_configs(self):
        chain = create_agent_chain_from_config([])
        assert chain.active_provider is None


# ── SwarmAgent integration ────────────────────────────────────────────

class TestSwarmAgentIntegration:
    def test_coder_uses_agent_provider(self):
        from spine.swarm.agents import SwarmAgent

        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_provider.execute.return_value = AgentResult(
            output="code written", exit_code=0, files_changed=["main.py"]
        )

        agent = SwarmAgent("coder", ["write"], agent_provider=mock_provider)
        state = {"requirement": "Add login page", "phase": "EXECUTION", "variables": {}}

        result = agent.execute(state, "write")
        mock_provider.execute.assert_called_once()
        assert result["result"] == "code written"
        assert result["files_changed"] == ["main.py"]
        assert result["success"] is True

    def test_planner_ignores_agent_provider(self):
        """Decision-making roles should NOT use agent_provider."""
        from spine.swarm.agents import SwarmAgent

        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "plan created"

        agent = SwarmAgent("planner", ["draft"], llm_provider=mock_llm, agent_provider=mock_provider)
        state = {"requirement": "Plan the project", "phase": "PLANNING", "variables": {}}

        result = agent.execute(state, "draft")
        # Should use LLM, not agent provider
        mock_provider.execute.assert_not_called()
        mock_llm.generate.assert_called_once()
        assert result["result"] == "plan created"

    def test_set_agent_provider(self):
        from spine.swarm.agents import SwarmAgent

        agent = SwarmAgent("coder", ["write"])
        assert agent._agent_provider is None

        mock_provider = MagicMock(spec=AgentProvider)
        agent.set_agent_provider(mock_provider)
        assert agent._agent_provider is mock_provider

    def test_implementation_roles_constant(self):
        from spine.swarm.agents import SwarmAgent

        assert "coder" in SwarmAgent.IMPLEMENTATION_ROLES
        assert "test_engineer" in SwarmAgent.IMPLEMENTATION_ROLES
        assert "reviewer" in SwarmAgent.IMPLEMENTATION_ROLES
        assert "planner" not in SwarmAgent.IMPLEMENTATION_ROLES
        assert "critic" not in SwarmAgent.IMPLEMENTATION_ROLES


# ── git changed files helper ─────────────────────────────────────────

class TestGitChangedFiles:
    def test_returns_list_on_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="a.py\nb.py\n"),
                MagicMock(stdout="c.py\n"),
            ]
            result = _git_changed_files("/tmp/project")
            assert "a.py" in result
            assert "b.py" in result
            assert "c.py" in result

    def test_returns_empty_on_error(self):
        with patch("subprocess.run", side_effect=Exception("no git")):
            result = _git_changed_files("/tmp/project")
            assert result == []
