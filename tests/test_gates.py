"""Tests for swarm gates."""

from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.swarm.gates import PreCheckBatch, CriticGate, CompletionGate
from spine.core.state_machine import SpineState


class TestPreCheckBatch:
    def test_init_has_default_checks(self):
        batch = PreCheckBatch()
        assert batch.name == "precheck_batch"
        assert batch.checks == ["lint_check", "secretscan", "sast_scan"]
        assert batch.required is True

    def test_evaluate_returns_results_structure(self):
        batch = PreCheckBatch()
        
        with patch.object(batch, '_run_lint_check', return_value={"check": "lint_check", "passed": True, "skipped": True}):
            with patch.object(batch, '_run_secretscan', return_value={"check": "secretscan", "passed": True, "skipped": True}):
                with patch.object(batch, '_run_sast_scan', return_value={"check": "sast_scan", "passed": True, "skipped": True}):
                    result = batch.evaluate({})
        
        assert "all_passed" in result
        assert "results" in result
        assert result["all_passed"] is True
        assert "lint_check" in result["results"]
        assert "secretscan" in result["results"]
        assert "sast_scan" in result["results"]

    def test_evaluate_fail_fast_on_failure(self):
        batch = PreCheckBatch()
        
        with patch.object(batch, '_run_lint_check', return_value={"check": "lint_check", "passed": False, "error": "lint failed"}):
            with patch.object(batch, '_run_secretscan', return_value={"check": "secretscan", "passed": True, "skipped": True}):
                with patch.object(batch, '_run_sast_scan', return_value={"check": "sast_scan", "passed": True, "skipped": True}):
                    result = batch.evaluate({})
        
        assert result["all_passed"] is False
        assert result["results"]["lint_check"]["passed"] is False


class TestCriticGate:
    def test_evaluate_no_plan_returns_not_approved(self):
        gate = CriticGate()
        result = gate.evaluate({})
        assert result["approved"] is False
        assert "No plan to review" in result["reason"]

    def test_evaluate_with_plan_returns_approved(self):
        gate = CriticGate()
        state = {"plan": {"tasks": ["a", "b"]}}
        result = gate.evaluate(state)
        assert result["approved"] is True


class TestCompletionGate:
    def test_evaluate_no_failed_tasks_returns_approved(self):
        gate = CompletionGate()
        state: SpineState = {
            "failed_tasks": [],
            "completed_tasks": ["task1", "task2"],
            "tasks": {},
            "phase": "VERIFICATION",
            "previous_phase": "EXECUTION",
            "requirement": "",
            "swarm_state": {},
            "hive_cells": {},
            "swarm_events": [],
            "variables": {},
            "errors": [],
            "providers": {},
        }
        result = gate.evaluate(state)
        assert result["approved"] is True

    def test_evaluate_with_failed_tasks_returns_not_approved(self):
        gate = CompletionGate()
        state: SpineState = {
            "failed_tasks": ["task1"],
            "completed_tasks": [],
            "tasks": {},
            "phase": "VERIFICATION",
            "previous_phase": "EXECUTION",
            "requirement": "",
            "swarm_state": {},
            "hive_cells": {},
            "swarm_events": [],
            "variables": {},
            "errors": [],
            "providers": {},
        }
        result = gate.evaluate(state)
        assert result["approved"] is False