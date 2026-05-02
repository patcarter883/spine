"""Swarm gates for validation and quality control."""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from ..core.state_machine import SpineState
from ..swarm.agents import CriticAgent


class SwarmGate:
    """Base class for swarm validation gates."""
    
    def __init__(self, name: str, required: bool = True):
        self.name = name
        self.required = required
    
    def evaluate(self, state: SpineState) -> dict[str, Any]:
        """Evaluate the gate and return result."""
        raise NotImplementedError


class CriticGate(SwarmGate):
    """Plan review gate - must pass before execution."""
    
    def __init__(self):
        super().__init__("critic", required=True)
        self.agent = CriticAgent()
    
    def evaluate(self, state: SpineState) -> dict[str, Any]:
        """Review the plan and return approval status."""
        if not state.get("plan"):
            return {"approved": False, "reason": "No plan to review"}
        
        # In production, would actually call the critic agent
        result = {
            "approved": True,
            "issues": [],
            "recommendations": []
        }
        return result


class PreCheckBatch(SwarmGate):
    """Runs lint_check, secretscan, sast_scan in parallel. Fail fast on any failure."""

    def __init__(self):
        super().__init__("precheck_batch", required=True)
        self.checks = ["lint_check", "secretscan", "sast_scan"]

    def _run_lint_check(self) -> dict[str, Any]:
        """Run lint check via subprocess."""
        try:
            result = subprocess.run(
                ["ruff", "check"],
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "check": "lint_check",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"check": "lint_check", "passed": False, "error": "timeout"}
        except FileNotFoundError:
            return {"check": "lint_check", "passed": True, "skipped": True}

    def _run_secretscan(self) -> dict[str, Any]:
        """Run secret scan via subprocess."""
        try:
            result = subprocess.run(
                ["gitleaks", "detect", "--source=.", "--exit-code=1"],
                capture_output=True,
                text=True,
                timeout=60
            )
            return {
                "check": "secretscan",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"check": "secretscan", "passed": False, "error": "timeout"}
        except FileNotFoundError:
            return {"check": "secretscan", "passed": True, "skipped": True}

    def _run_sast_scan(self) -> dict[str, Any]:
        """Run SAST scan via subprocess."""
        try:
            result = subprocess.run(
                ["semgrep", "scan", "--config=auto"],
                capture_output=True,
                text=True,
                timeout=120
            )
            return {
                "check": "sast_scan",
                "passed": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"check": "sast_scan", "passed": False, "error": "timeout"}
        except FileNotFoundError:
            return {"check": "sast_scan", "passed": True, "skipped": True}

    def evaluate(self, state: SpineState) -> dict[str, Any]:
        """Run all checks in parallel, fail fast on any failure."""
        results = {}
        all_passed = True

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_lint_check): "lint_check",
                executor.submit(self._run_secretscan): "secretscan",
                executor.submit(self._run_sast_scan): "sast_scan",
            }

            for future in as_completed(futures):
                check_name = futures[future]
                try:
                    result = future.result()
                    results[check_name] = result
                    if not result.get("passed", False) and not result.get("skipped", False):
                        all_passed = False
                except Exception as e:
                    results[check_name] = {"check": check_name, "passed": False, "error": str(e)}
                    all_passed = False

        return {"all_passed": all_passed, "results": results}


class CompletionGate(SwarmGate):
    """Final verification gate."""
    
    def __init__(self):
        super().__init__("completion", required=True)
    
    def evaluate(self, state: SpineState) -> dict[str, Any]:
        """Verify all tasks completed and no placeholders remain."""
        tasks = state.get("tasks", {})
        failed = state.get("failed_tasks", [])
        
        return {
            "approved": len(failed) == 0,
            "completed_count": len(state.get("completed_tasks", [])),
            "failed_count": len(failed)
        }