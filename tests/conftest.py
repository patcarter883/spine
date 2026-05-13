"""Test configuration and shared fixtures for SPINE tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
from pytest_asyncio import fixture as asyncio_fixture

from spine.config import SpineConfig
from spine.models.types import Task, Artifact, ReviewFeedback, PromptRequest


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config(temp_dir: Path) -> SpineConfig:
    """Create a test configuration with temporary paths."""
    config = SpineConfig()
    config.checkpoint_path = str(temp_dir / "test_spine.db")
    config.artifact_path = str(temp_dir / "artifacts")
    config.queue_path = str(temp_dir / "queue.db")
    config.workspace_root = str(temp_dir)
    config.ensure_dirs()
    return config


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        id="test-task-1",
        description="Test task description",
        status="pending",
        artifact_paths=["/path/to/artifact1.txt"],
    )


@pytest.fixture
def sample_artifact() -> Artifact:
    """Create a sample artifact for testing."""
    return Artifact(
        path="/path/to/artifact.txt",
        content="Sample artifact content",
        phase="test_phase",
    )


@pytest.fixture
def sample_review_feedback() -> ReviewFeedback:
    """Create sample review feedback for testing."""
    return ReviewFeedback(
        status="passed",
        tier="structural",
        reason="Test review passed",
        suggestions=["Consider improving X", "Look at Y"],
    )


@pytest.fixture
def sample_prompt_request() -> PromptRequest:
    """Create a sample prompt request for testing."""
    return PromptRequest(
        message="Please review this code",
        phase="verify",
        context={"file_path": "/test/code.py", "line_number": 10},
    )


@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Mock OpenAI API response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Mocked response from OpenAI"
                }
            }
        ]
    }


@pytest.fixture
def sample_work_description() -> str:
    """Sample work description for testing."""
    return "Create a simple Python web application using Flask"


@pytest.fixture
def sample_work_config() -> Dict[str, Any]:
    """Sample work configuration for testing."""
    return {
        "spine": {
            "checkpoint_path": ".spine/test.db",
            "artifact_path": ".spine/artifacts",
            "max_critic_retries": 2,
            "work_type": "spec"
        },
        "providers": {
            "llm": [
                {
                    "enabled": True,
                    "model": "openai:gpt-4o-mini",
                    "api_key": "test-api-key"
                }
            ]
        }
    }


# Async test utilities
@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@asyncio_fixture
async def async_test_config(temp_dir: Path) -> SpineConfig:
    """Async version of test_config fixture."""
    config = SpineConfig()
    config.checkpoint_path = str(temp_dir / "async_test_spine.db")
    config.artifact_path = str(temp_dir / "async_artifacts")
    config.queue_path = str(temp_dir / "async_queue.db")
    config.workspace_root = str(temp_dir)
    config.ensure_dirs()
    return config