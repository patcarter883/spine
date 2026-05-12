"""Tests for state machine phase transitions, error handling, and hooks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.core.state_machine import (
    SubPhase, Phase,
    _evaluate_entry_conditions, _evaluate_exit_conditions,
    _run_pre_execute_hooks, _run_post_execute_hooks,
    _check_error_threshold,
)
from spine.core.constants import ErrorState


class TestSubPhaseStateTracking:
    """Tests for SubPhase-level state management."""

    def test_subphase_default_status_is_pending(self):
        from spine.core.constants import SubPhaseStatus
        sp = SubPhase(name="TEST")
        assert sp.status == SubPhaseStatus.PENDING

    def test_subphase_max_retries_default_is_3(self):
        sp = SubPhase(name="TEST")
        assert sp.max_retries == 3

    def test_subphase_fail_sets_status_and_error(self):
        from spine.core.constants import SubPhaseStatus
        sp = SubPhase(name="TEST")
        sp.fail("Something went wrong")
        assert sp.status == SubPhaseStatus.FAILED
        assert sp.error == "Something went wrong"

    def test_subphase_block_sets_blocked_by(self):
        from spine.core.constants import SubPhaseStatus
        sp = SubPhase(name="TEST")
        sp.block("UPSTREAM")
        assert sp.status == SubPhaseStatus.BLOCKED
        assert sp.blocked_by == "UPSTREAM"

    def test_subphase_mark_reworking_clears_error(self):
        from spine.core.constants import SubPhaseStatus
        sp = SubPhase(name="TEST")
        sp.fail("error", blocked_by="X")
        assert sp.status == SubPhaseStatus.FAILED
        sp.mark_reworking()
        assert sp.status == SubPhaseStatus.REWORKING
        assert sp.error is None

    def test_subphase_mark_success_clears_error(self):
        from spine.core.constants import SubPhaseStatus
        sp = SubPhase(name="TEST")
        sp.fail("error")
        sp.mark_success("result")
        assert sp.status == SubPhaseStatus.SUCCESS
        assert sp.error is None


class TestSubPhaseResultFactory:
    """Tests for SubPhaseResult factory methods."""

    def test_failed_result(self):
        from spine.core.state_machine import SubPhaseResult
        from spine.core.constants import SubPhaseStatus
        result = SubPhaseResult.failed("TEST", "error info")
        assert result.subphase_name == "TEST"
        assert result.result is None
        assert result.status == SubPhaseStatus.FAILED

    def test_blocked_result(self):
        from spine.core.state_machine import SubPhaseResult
        from spine.core.constants import SubPhaseStatus
        result = SubPhaseResult.blocked("TEST", "UPSTREAM")
        assert result.subphase_name == "TEST"
        assert result.result is None
        assert result.status == SubPhaseStatus.BLOCKED


class TestEntryPointConditions:
    """Tests for entry condition evaluation."""

    def test_evaluate_entry_conditions_all_pass(self):
        phase = Phase(
            name="TEST",
            entry_conditions=[
                lambda ctx: ctx.get("valid", False),
                lambda ctx: len(ctx.get("items", [])) > 0
            ]
        )
        context = {"valid": True, "items": [1, 2, 3]}
        result = _evaluate_entry_conditions(phase, context)
        assert result is True

    def test_evaluate_entry_conditions_one_fails(self):
        phase = Phase(
            name="TEST",
            entry_conditions=[
                lambda ctx: ctx.get("valid", False),
                lambda ctx: len(ctx.get("items", [])) > 0
            ]
        )
        context = {"valid": True, "items": []}
        result = _evaluate_entry_conditions(phase, context)
        assert result is False

    def test_evaluate_entry_conditions_no_conditions(self):
        phase = Phase(name="TEST", entry_conditions=[])
        context = {}
        result = _evaluate_entry_conditions(phase, context)
        assert result is True


class TestExitPointConditions:
    """Tests for exit condition evaluation."""

    def test_evaluate_exit_conditions_all_pass(self):
        phase = Phase(
            name="TEST",
            exit_criteria=[
                lambda ctx: ctx.get("complete", False),
                lambda ctx: ctx.get("success", False)
            ]
        )
        context = {"complete": True, "success": True}
        result = _evaluate_exit_conditions(phase, context)
        assert result is True

    def test_evaluate_exit_conditions_one_fails(self):
        phase = Phase(
            name="TEST",
            exit_criteria=[
                lambda ctx: ctx.get("complete", False),
                lambda ctx: ctx.get("success", False)
            ]
        )
        context = {"complete": True, "success": False}
        result = _evaluate_exit_conditions(phase, context)
        assert result is False


class TestDAGHooks:
    """Tests for DAG pre/post execution hooks."""

    def test_run_pre_execute_hooks_modifies_context(self):
        phase = Phase(
            name="TEST",
            pre_execute_hooks=[lambda ctx: {**ctx, "pre_hook_ran": True}]
        )
        context = {"initial": True}
        result = _run_pre_execute_hooks(phase, context)
        assert result.get("pre_hook_ran") is True
        assert result.get("initial") is True

    def test_run_post_execute_hooks_modifies_context(self):
        phase = Phase(
            name="TEST",
            post_execute_hooks=[lambda ctx: {**ctx, "post_hook_ran": True}]
        )
        context = {"initial": True}
        result = _run_post_execute_hooks(phase, context)
        assert result.get("post_hook_ran") is True
        assert result.get("initial") is True

    def test_multiple_hooks_execute_in_order(self):
        phase = Phase(
            name="TEST",
            pre_execute_hooks=[
                lambda ctx: {**ctx, "first": True},
                lambda ctx: {**ctx, "second": True}
            ]
        )
        context = {}
        result = _run_pre_execute_hooks(phase, context)
        assert result.get("first") is True
        assert result.get("second") is True


class TestErrorThreshold:
    """Tests for error threshold checking."""

    def test_check_error_threshold_no_errors(self):
        subphases = [
            SubPhase(name="A"),
            SubPhase(name="B"),
        ]
        error_state, failed = _check_error_threshold(subphases)
        assert error_state == ErrorState.INIT.value
        assert failed == []

    def test_check_error_threshold_exceeded(self):
        subphases = [
            SubPhase(name="A"),
        ]
        subphases[0].error_count = 5
        error_state, failed = _check_error_threshold(subphases, max_errors=3)
        assert error_state == ErrorState.FATAL.value
        assert failed == subphases


class TestSubPhaseErrorTracking:
    """Tests for SubPhase error state tracking."""

    def test_subphase_error_count_increments_on_fail(self):
        sp = SubPhase(name="TEST")
        sp.fail("first error")
        assert sp.error_count == 1
        sp.fail("second error")
        assert sp.error_count == 2

    def test_subphase_last_error_updates(self):
        sp = SubPhase(name="TEST")
        sp.fail("first error")
        assert sp.last_error == "first error"
        sp.fail("second error")
        assert sp.last_error == "second error"

    def test_has_exceeded_error_threshold(self):
        sp = SubPhase(name="TEST")
        sp.error_count = 2
        assert sp.has_exceeded_error_threshold(3) is False
        assert sp.has_exceeded_error_threshold(2) is False
        sp.error_count = 4
        assert sp.has_exceeded_error_threshold(3) is True


class TestErrorStateTransitions:
    """Tests for error state transitions."""

    def test_error_state_enum_exists(self):
        assert hasattr(ErrorState, "INIT")
        assert hasattr(ErrorState, "TRANSIENT")
        assert hasattr(ErrorState, "FATAL")
        assert hasattr(ErrorState, "HUMAN_REVIEW")
        assert hasattr(ErrorState, "TIMEOUT")

    def test_error_state_values(self):
        assert ErrorState.TRANSIENT.value == "TRANSIENT"
        assert ErrorState.FATAL.value == "FATAL"
        assert ErrorState.HUMAN_REVIEW.value == "HUMAN_REVIEW"
