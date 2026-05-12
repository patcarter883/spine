"""Tests for the model I/O debug logger and critic gate fixes."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from spine.debug.model_io import (
    ModelIOLogger,
    is_debug_enabled,
    set_debug_phase,
    get_debug_phase,
)
from spine.middleware.critic_gate import CriticGateMiddleware


# ── ModelIOLogger tests ──────────────────────────────────────────────────


class TestDebugEnabled:
    """Tests for is_debug_enabled()."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert is_debug_enabled() is False

    def test_enabled_with_1(self):
        with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "1"}):
            assert is_debug_enabled() is True

    def test_enabled_with_true(self):
        with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "true"}):
            assert is_debug_enabled() is True

    def test_enabled_with_yes(self):
        with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "yes"}):
            assert is_debug_enabled() is True

    def test_not_enabled_with_other(self):
        with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "0"}):
            assert is_debug_enabled() is False


class TestDebugPhaseContext:
    """Tests for set_debug_phase / get_debug_phase."""

    def test_default_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            # Default is empty string
            assert get_debug_phase() in ("", "PLANNING", "EXECUTION", "VERIFICATION")

    def test_set_and_get(self):
        set_debug_phase("EXECUTION")
        assert get_debug_phase() == "EXECUTION"

    def test_set_overrides(self):
        set_debug_phase("PLANNING")
        set_debug_phase("VERIFICATION")
        assert get_debug_phase() == "VERIFICATION"


class TestModelIOLogger:
    """Tests for the ModelIOLogger wrapper."""

    def test_wrap_returns_original_when_disabled(self):
        """When debug is disabled, wrap() returns the original model."""
        mock_model = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            result = ModelIOLogger.wrap(mock_model)
            assert result is mock_model

    def test_wrap_wraps_when_enabled(self):
        """When debug is enabled, wrap() returns a ModelIOLogger."""
        mock_model = MagicMock()
        mock_model._llm_type = "test"
        with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "1"}):
            result = ModelIOLogger.wrap(mock_model)
            assert isinstance(result, ModelIOLogger)

    def test_wrap_idempotent(self):
        """Wrapping an already-wrapped model returns it unchanged."""
        mock_model = MagicMock()
        mock_model._llm_type = "test"
        with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "1"}):
            wrapped = ModelIOLogger.wrap(mock_model)
            double_wrapped = ModelIOLogger.wrap(wrapped)
            assert wrapped is double_wrapped

    def test_invoke_logs_files(self):
        """invoke() writes _in.json and _out.json files."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "test response"
        mock_model.invoke.return_value = mock_response
        mock_model._llm_type = "test"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "1"}):
                wrapper = ModelIOLogger(mock_model, debug_dir=tmpdir)
                set_debug_phase("TEST_PHASE")

                result = wrapper.invoke("hello world")

                assert result is mock_response
                files = os.listdir(tmpdir)
                in_files = [f for f in files if f.endswith("_in.json")]
                out_files = [f for f in files if f.endswith("_out.json")]
                assert len(in_files) == 1
                assert len(out_files) == 1

                # Check input file content
                with open(os.path.join(tmpdir, in_files[0])) as f:
                    data = json.load(f)
                    assert data["_meta"]["phase"] == "TEST_PHASE"
                    assert data["_meta"]["direction"] == "in"
                    assert "hello world" in str(data["data"])

                # Check output file content
                with open(os.path.join(tmpdir, out_files[0])) as f:
                    data = json.load(f)
                    assert data["_meta"]["direction"] == "out"
                    assert "test response" in str(data["data"])

    def test_invoke_with_message_list(self):
        """invoke() handles message list input correctly."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "response"
        mock_model.invoke.return_value = mock_response
        mock_model._llm_type = "test"

        from langchain_core.messages import HumanMessage, AIMessage

        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
            HumanMessage(content="how are you?"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "1"}):
                wrapper = ModelIOLogger(mock_model, debug_dir=tmpdir)
                result = wrapper.invoke(messages)

                assert result is mock_response
                in_files = [f for f in os.listdir(tmpdir) if f.endswith("_in.json")]
                with open(os.path.join(tmpdir, in_files[0])) as f:
                    data = json.load(f)
                    # Should have serialized all 3 messages
                    assert len(data["data"]) == 3

    def test_llm_type(self):
        """_llm_type includes the wrapped model type."""
        mock_model = MagicMock()
        mock_model._llm_type = "openai"
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = ModelIOLogger(mock_model, debug_dir=tmpdir)
            assert "openai" in wrapper._llm_type

    def test_write_failure_does_not_crash(self):
        """A failure to write debug logs should not crash the invoke."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "response"
        mock_model.invoke.return_value = mock_response
        mock_model._llm_type = "test"

        # Use a read-only directory to force write failure
        with tempfile.TemporaryDirectory() as tmpdir:
            read_only_dir = os.path.join(tmpdir, "readonly")
            os.makedirs(read_only_dir)
            os.chmod(read_only_dir, 0o444)

            with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "1"}):
                wrapper = ModelIOLogger(mock_model, debug_dir=read_only_dir)
                # Should not raise even though write fails
                result = wrapper.invoke("test")
                assert result is mock_response

    def test_serializes_response_with_tool_calls(self):
        """Output serialization includes tool_calls and usage_metadata."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "let me look that up"
        mock_response.tool_calls = [
            {"name": "read_file", "args": {"path": "/tmp/test.py"}}
        ]
        mock_response.usage_metadata = {"total_tokens": 150}
        mock_response.id = "chatcmpl-123"
        mock_model.invoke.return_value = mock_response
        mock_model._llm_type = "test"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SPINE_DEBUG_MODEL_IO": "1"}):
                wrapper = ModelIOLogger(mock_model, debug_dir=tmpdir)
                wrapper.invoke("use the tool")

                out_files = [f for f in os.listdir(tmpdir) if f.endswith("_out.json")]
                with open(os.path.join(tmpdir, out_files[0])) as f:
                    data = json.load(f)
                    assert data["data"]["tool_calls"] is not None
                    assert data["data"]["usage_metadata"] is not None


# ── CriticGateMiddleware tests ────────────────────────────────────────────


class TestCriticGateMiddlewareMaxRevisions:
    """Tests for the critic gate max_revisions limit."""

    def _make_state(self, content: str, spine_phase: str = "PLANNING") -> dict:
        """Create a minimal DA agent state dict."""
        return {
            "messages": [MagicMock(content=content)],
            "spine_phase": spine_phase,
        }

    def test_no_plan_complete_returns_none(self):
        """If the agent hasn't produced PLAN_COMPLETE, return None."""
        mw = CriticGateMiddleware()
        result = mw.after_model(
            self._make_state("I'm still working on the plan..."),
            runtime=MagicMock(),
        )
        assert result is None

    def test_non_planning_phase_returns_none(self):
        """During non-PLANNING phases, return None."""
        mw = CriticGateMiddleware()
        result = mw.after_model(
            self._make_state("PLAN_COMPLETE", spine_phase="EXECUTION"),
            runtime=MagicMock(),
        )
        assert result is None

    def test_max_revisions_flags_human_review(self):
        """After max_revisions rejections, the plan is flagged NEEDS_HUMAN_REVIEW."""
        mw = CriticGateMiddleware(max_revisions=2)
        runtime = MagicMock()

        # First PLAN_COMPLETE — mock the critic to always reject
        with patch.object(mw, "_run_critic", return_value="NEEDS_REVISION"):
            state1 = self._make_state("PLAN_COMPLETE\nPlan v1")
            result1 = mw.after_model(state1, runtime)
            # First rejection: injects feedback (revision 1/2)
            assert "messages" in result1
            assert result1["critic_gate_result"] == "NEEDS_REVISION"

        # Second PLAN_COMPLETE — critic rejects again (revision 2/2)
        with patch.object(mw, "_run_critic", return_value="NEEDS_REVISION"):
            state2 = self._make_state("PLAN_COMPLETE\nPlan v2")
            result2 = mw.after_model(state2, runtime)
            # Second rejection: still injects feedback (revision 2/2)
            assert result2["critic_gate_result"] == "NEEDS_REVISION"

        # Third PLAN_COMPLETE — max revisions exceeded, flag for human review
        state3 = self._make_state("PLAN_COMPLETE\nPlan v3")
        result3 = mw.after_model(state3, runtime)
        # Should flag for human review, NOT auto-approve
        assert result3["critic_gate_result"] == "NEEDS_HUMAN_REVIEW"

    def test_default_max_revisions_is_3(self):
        """Default max_revisions should be 3."""
        mw = CriticGateMiddleware()
        assert mw._max_revisions == 3

    def test_approved_plan_returns_immediately(self):
        """If the critic approves, return APPROVED immediately."""
        mw = CriticGateMiddleware()
        with patch.object(mw, "_run_critic", return_value="APPROVED"):
            state = self._make_state("PLAN_COMPLETE\nGreat plan")
            result = mw.after_model(state, runtime=MagicMock())
            assert result["critic_gate_result"] == "APPROVED"
            assert "messages" not in result  # No revision feedback

    def test_max_revisions_zero_flags_immediately(self):
        """Setting max_revisions=0 flags for human review on first PLAN_COMPLETE."""
        mw = CriticGateMiddleware(max_revisions=0)
        state = self._make_state("PLAN_COMPLETE")
        result = mw.after_model(state, runtime=MagicMock())
        # Immediately flag for human review since 0 >= 0
        assert result["critic_gate_result"] == "NEEDS_HUMAN_REVIEW"

    def test_revision_count_in_feedback(self):
        """Revision feedback includes current/max count."""
        mw = CriticGateMiddleware(max_revisions=5)
        with patch.object(mw, "_run_critic", return_value="NEEDS_REVISION"):
            state = self._make_state("PLAN_COMPLETE")
            result = mw.after_model(state, runtime=MagicMock())
            # The feedback message should mention revision count
            feedback = result["messages"][-1]
            assert "1/5" in feedback["content"]


# ── Planning retry limit integration test ─────────────────────────────────


class TestPlanningRetryLimit:
    """Tests for the planning_retry_count mechanism in planning_phase."""

    def test_retry_count_incremented(self):
        """planning_retry_count only increments on critic gate re-entry (previous_phase=PLANNING)."""
        from spine.core.state_machine import planning_phase
        from spine.models.types import SpineState

        # From INIT (first entry) — counter should NOT increment
        state = SpineState(
            phase="PLANNING",
            previous_phase="INIT",
            requirement="test",
            plan=None,
            tasks={},
            completed_tasks=[],
            failed_tasks=[],
            swarm_state={},
            hive_cells={},
            swarm_events=[],
            variables={"debug_prompts": False},
            errors=[],
            providers={},
            agent_provider=None,
            critic_gate_result=None,
            error_state=None,
            error_history=[],
            planning_retry_count=0,
        )

        with patch("spine.core.state_machine._get_providers", return_value={}):
            result = planning_phase(state)
            # From INIT, retry_count stays at 0 (not a retry)
            assert result.get("planning_retry_count", 0) == 0

        # From PLANNING (critic rejected) — counter SHOULD increment
        state2 = SpineState(
            phase="PLANNING",
            previous_phase="PLANNING",
            requirement="test",
            plan=None,
            tasks={},
            completed_tasks=[],
            failed_tasks=[],
            swarm_state={},
            hive_cells={},
            swarm_events=[],
            variables={"debug_prompts": False},
            errors=[],
            providers={},
            agent_provider=None,
            critic_gate_result=None,
            error_state=None,
            error_history=[],
            planning_retry_count=0,
        )

        with patch("spine.core.state_machine._get_providers", return_value={}):
            result2 = planning_phase(state2)
            assert result2.get("planning_retry_count", 0) >= 1

    def test_error_recovery_does_not_increment_retry(self):
        """Re-entry from ERROR/REWORK should NOT inflate planning_retry_count."""
        from spine.core.state_machine import planning_phase
        from spine.models.types import SpineState

        state = SpineState(
            phase="PLANNING",
            previous_phase="ERROR",
            requirement="test",
            plan=None,
            tasks={},
            completed_tasks=[],
            failed_tasks=[],
            swarm_state={},
            hive_cells={},
            swarm_events=[],
            variables={"debug_prompts": False},
            errors=[],
            providers={},
            agent_provider=None,
            critic_gate_result=None,
            error_state=None,
            error_history=[],
            planning_retry_count=0,
        )

        with patch("spine.core.state_machine._get_providers", return_value={}):
            result = planning_phase(state)
            # From ERROR, retry_count stays at 0 (error recovery, not a critic retry)
            assert result.get("planning_retry_count", 0) == 0

    def test_retry_limit_routes_to_human_review(self):
        """When retry_count exceeds max, route to HUMAN_REVIEW, not EXECUTION."""
        from spine.core.state_machine import planning_phase
        from spine.models.types import SpineState

        # Set retry count above the default max (3) and previous_phase=PLANNING
        # so the counter increments to 4 > 3
        state = SpineState(
            phase="PLANNING",
            previous_phase="PLANNING",  # Must be PLANNING so counter increments
            requirement="test requirement",
            plan=None,
            tasks={},
            completed_tasks=[],
            failed_tasks=[],
            swarm_state={},
            hive_cells={},
            swarm_events=[],
            variables={"debug_prompts": False},
            errors=[],
            providers={},
            agent_provider=None,
            critic_gate_result=None,
            error_state=None,
            error_history=[],
            planning_retry_count=3,  # Already at max
        )

        with patch.dict(os.environ, {"SPINE_MAX_PLANNING_RETRIES": "3"}):
            result = planning_phase(state)

            # Should route to HUMAN_REVIEW, NOT auto-approve to EXECUTION
            assert result["phase"] == "HUMAN_REVIEW"
            assert result["critic_gate_result"] == "NEEDS_HUMAN_REVIEW"
            assert result["plan"] is not None
            assert result["variables"]["waiting_for_human"] is True
            assert result["variables"]["resume_phase"] == "planning"


class TestShouldContinueCriticRejection:
    """Tests for should_continue routing on critic gate rejection."""

    def test_needs_revision_routes_to_planning_retry(self):
        """NEEDS_REVISION is a soft rejection — should_continue allows retry."""
        from spine.core.state_machine import should_continue

        state = {
            "phase": "PLANNING",
            "critic_gate_result": "NEEDS_REVISION",
        }
        result = should_continue(state)
        assert result == "planning"

    def test_critic_rejected_routes_to_human_review(self):
        """REJECTED verdict routes to human_review (hard reject)."""
        from spine.core.state_machine import should_continue

        state = {
            "phase": "PLANNING",
            "critic_gate_result": "REJECTED",
        }
        result = should_continue(state)
        assert result == "human_review"

    def test_critic_needs_human_review_routes_to_human_review(self):
        """NEEDS_HUMAN_REVIEW verdict routes to human_review."""
        from spine.core.state_machine import should_continue

        state = {
            "phase": "PLANNING",
            "critic_gate_result": "NEEDS_HUMAN_REVIEW",
        }
        result = should_continue(state)
        assert result == "human_review"

    def test_critic_approved_routes_to_execution(self):
        """APPROVED verdict routes to execution."""
        from spine.core.state_machine import should_continue

        state = {
            "phase": "PLANNING",
            "critic_gate_result": "APPROVED",
        }
        result = should_continue(state)
        assert result == "execution"

    def test_critic_none_routes_to_planning(self):
        """No critic result (None) means planning hasn't run yet — route to planning."""
        from spine.core.state_machine import should_continue

        state = {
            "phase": "PLANNING",
            "critic_gate_result": None,
        }
        result = should_continue(state)
        assert result == "planning"


class TestPlanningPhaseCriticRouting:
    """Tests for the critic routing logic in planning_phase.

    The full _planning_phase_da path requires a DA agent to be created and
    invoked, which is too heavy for unit tests.  Instead we test the
    routing contract via should_continue and the retry-limit guard.
    """

    def test_needs_revision_allows_up_to_max_retries(self):
        """NEEDS_REVISION retries planning up to SPINE_MAX_PLANNING_RETRIES."""
        from spine.core.state_machine import should_continue

        # Within retry limit: retry
        for attempt in range(1, 4):
            state = {
                "phase": "PLANNING",
                "critic_gate_result": "NEEDS_REVISION",
                "planning_retry_count": attempt,
            }
            result = should_continue(state)
            assert result == "planning", (
                f"Attempt {attempt}/3 should retry, not go to human_review"
            )

    def test_needs_revision_then_approved_goes_to_execution(self):
        """After retrying, if critic approves, proceed to execution."""
        from spine.core.state_machine import should_continue

        state = {
            "phase": "PLANNING",
            "critic_gate_result": "APPROVED",
            "planning_retry_count": 2,
        }
        result = should_continue(state)
        assert result == "execution"

    def test_retry_limit_exceeded_routes_to_human_review(self):
        """When planning_phase retry limit is exceeded, it routes to HUMAN_REVIEW."""
        from spine.core.state_machine import planning_phase
        from spine.models.types import SpineState

        state = SpineState(
            phase="PLANNING",
            previous_phase="PLANNING",  # Must be PLANNING so counter increments
            requirement="test requirement",
            plan=None,
            tasks={},
            completed_tasks=[],
            failed_tasks=[],
            swarm_state={},
            hive_cells={},
            swarm_events=[],
            variables={"debug_prompts": False},
            errors=[],
            providers={},
            agent_provider=None,
            critic_gate_result=None,
            error_state=None,
            error_history=[],
            planning_retry_count=3,
        )

        with patch.dict(os.environ, {"SPINE_MAX_PLANNING_RETRIES": "3"}):
            result = planning_phase(state)
            assert result["phase"] == "HUMAN_REVIEW"
            assert result["critic_gate_result"] == "NEEDS_HUMAN_REVIEW"
            assert result["variables"]["waiting_for_human"] is True
