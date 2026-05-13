"""Unit tests for SPINE configuration management."""

from __future__ import annotations

import os
import tempfile
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
        assert config.max_critic_retries == 3
        assert config.work_type == "spec"
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
                "work_type": "quick"
            },
            "providers": {
                "llm": [
                    {
                        "enabled": True,
                        "model": "openai:gpt-4o-mini"
                    }
                ]
            }
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config = SpineConfig.load(str(config_file))
        
        assert config.checkpoint_path == ".spine/custom.db"
        assert config.artifact_path == ".spine/custom_artifacts"
        assert config.max_critic_retries == 5
        assert config.work_type == "quick"
        assert len(config.providers["llm"]) == 1
        assert config.providers["llm"][0]["model"] == "openai:gpt-4o-mini"

    def test_load_config_with_env_override(self, temp_dir: Path) -> None:
        """Test that environment variables override config file values."""
        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"spine": {"checkpoint_path": ".spine/file.db"}}, f)
        
        # Set environment variable
        os.environ["SPINE_CHECKPOINT_PATH"] = ".spine/env.db"
        
        try:
            config = SpineConfig.load(str(config_file))
            assert config.checkpoint_path == ".spine/env.db"
        finally:
            # Clean up environment variable
            del os.environ["SPINE_CHECKPOINT_PATH"]

    def test_resolve_model_from_config(self) -> None:
        """Test resolving model from provider configuration."""
        config = SpineConfig()
        config.providers = {
            "llm": [
                {"enabled": True, "model": "openai:gpt-4o-mini"},
                {"enabled": False, "model": "openai:gpt-4o"}
            ]
        }
        
        model = config.resolve_model()
        assert model == "openai:gpt-4o-mini"

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
        config.artifact_path = str(temp_dir / "artifacts")
        config.queue_path = str(temp_dir / "queue" / "queue.db")
        
        config.ensure_dirs()
        
        assert Path(config.checkpoint_path).parent.exists()
        assert Path(config.artifact_path).exists()
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
                "llm": [
                    {"enabled": True, "model": "openai:gpt-4o-mini"}
                ],
                "embedding": [
                    {"enabled": True, "model": "openai:text-embedding-3-small"}
                ]
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
        valid_work_types = ["quick", "critical_quick", "spec", "critical_spec"]
        
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
        assert config.max_critic_retries == 3