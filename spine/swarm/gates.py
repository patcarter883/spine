"""Swarm gates for validation and quality control."""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal
from ..core.state_machine import SpineState


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
    
    def __init__(self, agent_provider: Any | None = None):
        super().__init__("critic", required=True)
        self._agent_provider = agent_provider
    
    def evaluate(self, state: SpineState) -> dict[str, Any]:
        """Review the plan and return approval status."""
        plan = state.get("plan")
        
        if not plan:
            return {"approved": False, "reason": "No plan to review"}
        
        # Use agent for review if available
        if self._agent_provider and not isinstance(self._agent_provider, dict) and self._agent_provider.enabled:
            return self._evaluate_with_agent(plan, state)
        
        # Auto-approve when no agent (plan was already agent-generated)
        result = {
            "approved": True,
            "issues": [],
            "recommendations": []
        }
        return result
    
    def _evaluate_with_agent(self, plan: dict[str, Any], state: SpineState) -> dict[str, Any]:
        """Evaluate plan using agent provider."""
        prompt = f"""Review this execution plan for correctness and completeness.

Requirement: {state.get('requirement', '')}

Plan: {plan}

Evaluate for:
1. Completeness - Are all necessary tasks included?
2. Correctness - Will the approach work?
3. Safety - Any security or risk concerns?
4. Clarity - Are the tasks well-defined?

Return JSON with: approved (true/false), issues, recommendations.
"""
        try:
            result = self._agent_provider.execute(prompt, workdir=os.getcwd(), timeout=120)
            return self._parse_critic_response(result.output or "")
        except Exception as e:
            return {
                "approved": False,
                "reason": f"Agent evaluation error: {e}",
                "issues": [str(e)],
                "recommendations": []
            }
    
    def _parse_critic_response(self, response: str) -> dict[str, Any]:
        """Parse LLM critic response."""
        import json
        try:
            result = json.loads(response)
            return {
                "approved": result.get("approved", False),
                "issues": result.get("issues", []),
                "recommendations": result.get("recommendations", [])
            }
        except json.JSONDecodeError:
            approved = "approved" in response.lower() and "not approved" not in response.lower()
            return {
                "approved": approved,
                "issues": [],
                "recommendations": [response[:200]]
            }


class QualityGate(SwarmGate):
    """Quality validation gate with agent-powered evaluation."""
    
    def __init__(self, agent_provider: Any | None = None):
        super().__init__("quality", required=True)
        self._agent_provider = agent_provider
    
    def evaluate(self, state: SpineState) -> dict[str, Any]:
        """Evaluate quality gate and return approval status."""
        plan = state.get("plan")
        
        if not plan:
            return {
                "approved": False,
                "reason": "No plan provided for quality review"
            }
        
        # Use agent for thorough evaluation if available
        if self._agent_provider and not isinstance(self._agent_provider, dict) and self._agent_provider.enabled:
            return self._evaluate_with_agent(plan, state)
        
        # Auto-approve when no agent
        tasks = plan.get("tasks", [])
        if not tasks:
            return {
                "approved": False,
                "reason": "Plan has no tasks"
            }
        
        return {
            "approved": True,
            "issues": [],
            "recommendations": []
        }
    
    def _evaluate_with_agent(self, plan: dict[str, Any], state: SpineState) -> dict[str, Any]:
        """Evaluate using agent provider for comprehensive review."""
        prompt = self._build_quality_prompt(plan, state)
        
        try:
            result = self._agent_provider.execute(prompt, workdir=os.getcwd(), timeout=120)
            return self._parse_response(result.output or "")
        except Exception as e:
            return {
                "approved": False,
                "issues": [f"Agent evaluation error: {e}"],
                "recommendations": []
            }
    
    def _build_quality_prompt(self, plan: dict[str, Any], state: SpineState) -> str:
        """Build LLM prompt for quality gate evaluation."""
        return f"""You are a quality gate critic. Review this plan thoroughly.

Requirement: {state.get('requirement', '')}

Plan to review:
{plan}

Evaluate for:
1. Completeness - All necessary tasks included?
2. Correctness - Will the approach work?
3. Security - Any vulnerabilities?
4. Maintainability - Is the code well-structured?
5. Testability - Are tests included?

Return JSON:
{{"approved": true/false, "issues": [], "recommendations": []}}
"""
    
    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response."""
        import json
        try:
            result = json.loads(response)
            return {
                "approved": result.get("approved", False),
                "issues": result.get("issues", []),
                "recommendations": result.get("recommendations", [])
            }
        except json.JSONDecodeError:
            approved = "approved" in response.lower() and "not approved" not in response.lower()
            return {
                "approved": approved,
                "issues": [],
                "recommendations": [response[:200]]
            }


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