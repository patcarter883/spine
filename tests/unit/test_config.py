"""Unit tests for SPINE configuration management."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from spine.config import SpineConfig


class TestSpineConfig:
    """Test cases for SpineConfig class."""

    def test_default_config_values(self) -> None:
        """Test that default configuration values are set correctly."""
        config = SpineConfig()

        assert config.checkpoint_path == ".spine/spine.db"
        assert config.artifact_path == ".spine/artifacts"
        assert config.max_critic_retries == 2
        assert config.max_adversarial_retries == 2
        assert config.work_type == "task"
        assert config.queue_backend == "sqlite"
        assert config.queue_path == ".spine/queue.db"
        assert config.workspace_root == ""

    def test_load_config_from_file(self, temp_dir: Path) -> None:
        """Test loading configuration from a YAML file."""
        config_data = {
            "spine": {
                "checkpoint_path": ".spine/custom.db",
                "artifact_path": ".spine/custom_artifacts",
                "max_critic_retries": 5,
                "work_type": "task",
            },
            "providers": {"llm": [{"enabled": True, "model": "openai:gpt-4o-mini"}]},
        }

        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = SpineConfig.load(str(config_file))

        assert config.checkpoint_path == ".spine/custom.db"
        assert config.artifact_path == ".spine/custom_artifacts"
        assert config.max_critic_retries == 5
        assert config.work_type == "task"
        assert len(config.providers["llm"]) == 1
        assert config.providers["llm"][0]["model"] == "openai:gpt-4o-mini"

    def test_load_config_with_env_override(self, temp_dir: Path) -> None:
        """Test that environment variables override config file values."""
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"spine": {"checkpoint_path": ".spine/file.db"}}, f)

        # Set environment variable
        os.environ["SPINE_CHECKPOINT_PATH"] = ".spine/env.db"
        os.environ["SPINE_MAX_ADVERSARIAL_RETRIES"] = "4"

        try:
            config = SpineConfig.load(str(config_file))
            assert config.checkpoint_path == ".spine/env.db"
            assert config.max_adversarial_retries == 4
        finally:
            # Clean up environment variable
            del os.environ["SPINE_CHECKPOINT_PATH"]
            del os.environ["SPINE_MAX_ADVERSARIAL_RETRIES"]

    def test_resolve_model_from_config(self) -> None:
        """Test resolving model from provider configuration."""
        config = SpineConfig()
        config.providers = {
            "llm": [
                {"enabled": True, "model": "openai:gpt-4o-mini"},
                {"enabled": False, "model": "openai:gpt-4o"},
            ]
        }

        model = config.resolve_model()
        assert model == "openai:gpt-4o-mini"

    def test_resolve_model_intermediate_phase_prefix(self) -> None:
        """An intermediate path key (implement/decomposer) overrides all of its
        modes, while an exact key still wins and the bare phase is the floor."""
        config = SpineConfig()
        config.providers = {
            "llm": [
                {"name": "local", "enabled": True, "model": "openai:local"},
                {"name": "strong", "enabled": False, "model": "openai:strong"},
                {"name": "exact", "enabled": False, "model": "openai:exact"},
            ],
            "phases": {
                "implement": {"provider": "local"},
                "implement/decomposer": {"provider": "strong"},
                "implement/decomposer/fallback": {"provider": "exact"},
            },
        }
        # Exact key wins for the mode it names.
        assert config.resolve_model("implement/decomposer/fallback") == "openai:exact"
        # Intermediate key covers the other modes.
        assert config.resolve_model("implement/decomposer/plan") == "openai:strong"
        assert config.resolve_model("implement/decomposer/per_file") == "openai:strong"
        # Unrelated implement sub-paths fall through to the bare phase default.
        assert (
            config.resolve_model("implement/subagents/slice-implementer")
            == "openai:local"
        )

    def test_resolve_model_from_env(self) -> None:
        """Test resolving model from environment variable."""
        config = SpineConfig()
        config.providers = {"llm": []}

        os.environ["SPINE_MODEL"] = "openrouter:z-ai/glm-4.5-air:free"

        try:
            model = config.resolve_model()
            assert model == "openrouter:z-ai/glm-4.5-air:free"
        finally:
            del os.environ["SPINE_MODEL"]

    def test_resolve_model_raises_error(self) -> None:
        """Test that resolve_model raises ValueError when no model is configured."""
        config = SpineConfig()
        config.providers = {"llm": []}

        with pytest.raises(ValueError, match="No LLM model configured"):
            config.resolve_model()

    def test_ensure_creates_directories(self, temp_dir: Path) -> None:
        """Test that ensure_dirs creates necessary directories."""
        config = SpineConfig()
        config.checkpoint_path = str(temp_dir / "checkpoints" / "test.db")
        config.artifact_path = str(temp_dir / "artifacts" / "subdir")
        config.queue_path = str(temp_dir / "queue" / "queue.db")

        config.ensure_dirs()

        assert Path(config.checkpoint_path).parent.exists()
        assert Path(config.artifact_path).parent.exists()
        assert Path(config.queue_path).parent.exists()

    def test_config_nonexistent_file(self, temp_dir: Path) -> None:
        """Test loading config from nonexistent file uses defaults."""
        nonexistent_file = temp_dir / "nonexistent.yaml"
        config = SpineConfig.load(str(nonexistent_file))

        # Should still have default values
        assert config.checkpoint_path == ".spine/spine.db"
        assert config.artifact_path == ".spine/artifacts"

    def test_config_invalid_yaml(self, temp_dir: Path) -> None:
        """Test loading config with invalid YAML content."""
        invalid_file = temp_dir / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("spine:\n    checkpoint_path: [invalid")

        config = SpineConfig.load(str(invalid_file))

        # Should fall back to defaults
        assert config.checkpoint_path == ".spine/spine.db"

    def test_providers_config_merging(self, temp_dir: Path) -> None:
        """Test that providers configuration is properly merged."""
        config_data = {
            "providers": {
                "llm": [{"enabled": True, "model": "openai:gpt-4o-mini"}],
                "embedding": [{"enabled": True, "model": "openai:text-embedding-3-small"}],
            }
        }

        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = SpineConfig.load(str(config_file))

        assert "llm" in config.providers
        assert "embedding" in config.providers
        assert len(config.providers["llm"]) == 1
        assert len(config.providers["embedding"]) == 1

    def test_work_type_validation(self) -> None:
        """Test that work_type accepts valid values."""
        valid_work_types = ["task", "critical_task"]

        for work_type in valid_work_types:
            config = SpineConfig()
            config.work_type = work_type
            assert config.work_type == work_type

    def test_config_with_empty_yaml(self, temp_dir: Path) -> None:
        """Test loading config from empty YAML file."""
        empty_file = temp_dir / "empty.yaml"
        with open(empty_file, "w") as f:
            f.write("")

        config = SpineConfig.load(str(empty_file))

        # Should use all defaults
        assert config.checkpoint_path == ".spine/spine.db"
        assert config.artifact_path == ".spine/artifacts"
        assert config.max_critic_retries == 2

    # ── Window-aware synthesis budgeting (trace 019eb3dd) ─────────────

    def test_synthesis_budget_fields_defaults(self) -> None:
        """New synthesis budget knobs have sane defaults."""
        config = SpineConfig()
        assert config.synthesize_max_completion_tokens == 8000
        assert config.synthesize_overhead_tokens == 4000
        assert config.evidence_compression_enabled is True

    def test_synthesis_budget_fields_from_yaml(self, temp_dir: Path) -> None:
        """Synthesis budget knobs parse from the spine: section."""
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(
                {
                    "spine": {
                        "synthesize_max_completion_tokens": 6000,
                        "synthesize_overhead_tokens": 2000,
                        "evidence_compression_enabled": False,
                    }
                },
                f,
            )
        config = SpineConfig.load(str(config_file))
        assert config.synthesize_max_completion_tokens == 6000
        assert config.synthesize_overhead_tokens == 2000
        assert config.evidence_compression_enabled is False

    def test_context_window_resolves_per_provider_and_phase(
        self, temp_dir: Path
    ) -> None:
        """context_window flows through provider resolution + phase override."""
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(
                {
                    "providers": {
                        "llm": [
                            {
                                "name": "local",
                                "model": "openai:m",
                                "enabled": True,
                                "context_window": 60000,
                            }
                        ],
                        "phases": {
                            "specify": {"provider": "local"},
                            "plan": {
                                "provider": "local",
                                "context_window": 32000,  # phase override
                            },
                        },
                    }
                },
                f,
            )
        config = SpineConfig.load(str(config_file))
        assert config.resolve_provider_config(phase="specify")["context_window"] == 60000
        assert config.resolve_provider_config(phase="plan")["context_window"] == 32000

    def test_context_window_absent_by_default(self) -> None:
        """Providers without context_window stay legacy (no key)."""
        config = SpineConfig(
            providers={"llm": [{"name": "c", "model": "openrouter:x", "enabled": True}]}
        )
        assert config.resolve_provider_config(phase="specify").get("context_window") is None

    # ── MCP config parsing ────────────────────────────────────────────

    def test_mcp_servers_default_empty(self) -> None:
        """Config without mcp_servers should have empty dict."""
        config = SpineConfig()
        assert config.mcp_servers == {}

    def test_mcp_servers_from_config_file(self, temp_dir: Path) -> None:
        """mcp_servers should be parsed correctly from YAML."""
        config_data = {
            "mcp_servers": {
                "codebase-index": {
                    "transport": "stdio",
                    "command": "mcp-codebase-index",
                    "args": [],
                    "env": {"PROJECT_ROOT": "/test"},
                }
            }
        }
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = SpineConfig.load(str(config_file))
        assert "codebase-index" in config.mcp_servers
        server = config.mcp_servers["codebase-index"]
        assert server["transport"] == "stdio"
        assert server["command"] == "mcp-codebase-index"
        assert server["env"]["PROJECT_ROOT"] == "/test"

    def test_mcp_servers_defaults_for_missing_keys(self, temp_dir: Path) -> None:
        """Missing optional MCP server keys should get defaults."""
        config_data = {
            "mcp_servers": {
                "minimal": {
                    "command": "my-server",
                }
            }
        }
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = SpineConfig.load(str(config_file))
        server = config.mcp_servers["minimal"]
        assert server["transport"] == "stdio"
        assert server["command"] == "my-server"
        assert server["args"] == []
        assert server["env"] == {}

    def test_mcp_servers_env_override(self, temp_dir: Path) -> None:
        """SPINE_MCP_SERVERS env var should merge with config."""
        config_data = {
            "mcp_servers": {
                "server-a": {"command": "cmd-a"},
            }
        }
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        os.environ["SPINE_MCP_SERVERS"] = '{"server-b": {"command": "cmd-b"}}'
        try:
            config = SpineConfig.load(str(config_file))
            assert "server-a" in config.mcp_servers
            assert "server-b" in config.mcp_servers
            assert config.mcp_servers["server-b"]["command"] == "cmd-b"
        finally:
            del os.environ["SPINE_MCP_SERVERS"]

    def test_mcp_servers_env_override_ignores_invalid_json(self, temp_dir: Path) -> None:
        """Invalid JSON in SPINE_MCP_SERVERS should be ignored gracefully."""
        os.environ["SPINE_MCP_SERVERS"] = "not valid json"
        try:
            config = SpineConfig.load()
            assert isinstance(config.mcp_servers, dict)
        finally:
            del os.environ["SPINE_MCP_SERVERS"]

    def test_mcp_servers_skips_non_dict_entries(self, temp_dir: Path) -> None:
        """Non-dict server configs should be silently skipped."""
        config_data = {
            "mcp_servers": {
                "good": {"command": "cmd"},
                "bad": "just a string, not a dict",
            }
        }
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = SpineConfig.load(str(config_file))
        assert "good" in config.mcp_servers
        assert "bad" not in config.mcp_servers
