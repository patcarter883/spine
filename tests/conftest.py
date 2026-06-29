"""Test configuration and shared fixtures for SPINE tests."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
from pytest_asyncio import fixture as asyncio_fixture

from spine.config import SpineConfig
from spine.models.types import Task, Artifact, ReviewFeedback, PromptRequest

# Absolute path to THIS repository — fixed regardless of any test's cwd changes.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_head() -> str:
    """Return the real repo's symbolic HEAD ref (e.g. ``refs/heads/main``)."""
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "symbolic-ref", "-q", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — guard must never crash the suite
        return ""


@pytest.fixture(autouse=True)
def _guard_repo_git_head() -> Generator[None, None, None]:
    """Fail loudly if a test mutates the real repo's git HEAD, and restore it.

    ``SpineGitOrchestrator`` defaults ``master_dir`` to ``os.getcwd()`` and its
    merge/rollback paths run ``git checkout <main>`` (and, on rollback,
    ``git reset --hard`` / ``git clean -fd``). A test that drives those paths
    without overriding ``master_dir`` to a tmp repo mutates THIS checkout —
    flipping HEAD to main and potentially wiping the working tree. Running the
    full suite used to leave HEAD on ``main`` for exactly this reason.

    This autouse guard snapshots HEAD before each test and, if it changed,
    re-points it (via ``symbolic-ref`` — no working-tree checkout) and fails the
    offending test by name so the leak is caught at its source instead of
    silently poisoning the rest of the run.
    """
    before = _repo_head()
    yield
    after = _repo_head()
    if before and after and before != after:
        # Re-point HEAD without touching the working tree, so subsequent tests
        # (and the developer's checkout) aren't left on the wrong branch.
        subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "symbolic-ref", "HEAD", before],
            capture_output=True, text=True,
        )
        pytest.fail(
            f"Test mutated the real repo's git HEAD ({before} -> {after}). It "
            "likely constructed SpineGitOrchestrator without overriding "
            "master_dir (which defaults to os.getcwd()). Point it at a tmp repo.",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _no_langsmith_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the test suite out of LangSmith.

    Importing ``spine.config`` loads the repo's ``.env``, so the real
    ``LANGSMITH_API_KEY`` sits in ``os.environ`` during tests. The ambient
    tracing flag is already forced off, but ``work_run_tracing()`` opts the
    work-run code path back in whenever a key is present — so dispatcher and
    onboarding-engine tests emit real traces. Deleting the key makes
    ``work_run_tracing`` a no-op; ``SPINE_TRACE_ALL`` is cleared so the
    trace-everything escape hatch cannot re-enable tracing under pytest
    either.
    """
    for var in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "SPINE_TRACE_ALL"):
        monkeypatch.delenv(var, raising=False)


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
        "choices": [{"message": {"role": "assistant", "content": "Mocked response from OpenAI"}}]
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
            "work_type": "task",
        },
        "providers": {
            "llm": [{"enabled": True, "model": "openai:gpt-4o-mini", "api_key": "test-api-key"}]
        },
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
