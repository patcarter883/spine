"""Tests for swarm gates."""

import sys
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.swarm.gates import PreCheckBatch, CriticGate, CompletionGate, QualityGate
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

    def test_evaluate_with_llm_provider(self):
        """CriticGate should use LLM when available."""
        fake_llm = MagicMock()
        fake_llm.generate.return_value = '{"approved": true, "issues": [], "recommendations": []}'
        fake_llm.enabled = True
        
        gate = CriticGate(llm_provider=fake_llm)
        state = {"plan": {"tasks": ["task1"]}, "requirement": "Build API"}
        result = gate.evaluate(state)
        
        assert result["approved"] is True
        fake_llm.generate.assert_called_once()


class TestQualityGate:
    def test_init_default(self):
        """QualityGate should initialize with correct defaults."""
        gate = QualityGate()
        assert gate.name == "quality"
        assert gate.required is True

    def test_evaluate_no_plan_returns_not_approved(self):
        gate = QualityGate()
        result = gate.evaluate({})
        assert result["approved"] is False
        assert "No plan provided" in result["reason"]

    def test_evaluate_with_plan_no_llm(self):
        """QualityGate stub evaluation with plan."""
        gate = QualityGate()
        state: SpineState = {
            "requirement": "Build API",
            "plan": {"tasks": [{"id": "t1"}]},
            "phase": "PLANNING",
            "previous_phase": None,
            "tasks": {},
            "completed_tasks": [],
            "failed_tasks": [],
            "swarm_state": {},
            "hive_cells": {},
            "swarm_events": [],
            "variables": {},
            "errors": [],
            "providers": {},
        }
        result = gate.evaluate(state)
        assert result["approved"] is True

    def test_evaluate_with_llm_provider(self):
        """QualityGate should use LLM when available."""
        fake_llm = MagicMock()
        fake_llm.generate.return_value = '{"approved": true, "issues": [], "recommendations": ["Add tests"]}'
        fake_llm.enabled = True
        
        gate = QualityGate(llm_provider=fake_llm)
        state: SpineState = {
            "requirement": "Build API",
            "plan": {"tasks": []},
            "phase": "PLANNING",
            "previous_phase": None,
            "tasks": {},
            "completed_tasks": [],
            "failed_tasks": [],
            "swarm_state": {},
            "hive_cells": {},
            "swarm_events": [],
            "variables": {},
            "errors": [],
            "providers": {},
        }
        result = gate.evaluate(state)
        
        assert result["approved"] is True
        assert "Add tests" in result["recommendations"]
        fake_llm.generate.assert_called_once()

    def test_evaluate_plan_without_tasks(self):
        """QualityGate should fail for empty task list."""
        gate = QualityGate()
        state: SpineState = {
            "requirement": "Build API",
            "plan": {"tasks": []},
            "phase": "PLANNING",
            "previous_phase": None,
            "tasks": {},
            "completed_tasks": [],
            "failed_tasks": [],
            "swarm_state": {},
            "hive_cells": {},
            "swarm_events": [],
            "variables": {},
            "errors": [],
            "providers": {},
        }
        result = gate.evaluate(state)
        assert result["approved"] is False
        assert "no tasks" in result["reason"].lower()


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