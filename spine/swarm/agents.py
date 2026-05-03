"""Swarm agents with role-specific capabilities and messaging."""

from typing import Any, Optional
from ..core.state_machine import SpineState
from ..providers.llm import LLMProvider
from .mail import SwarmMail
from .learning import LearningIntegration


class MessageTypes:
    """Standard message types for agent-to-agent communication."""
    PLAN_FOR_REVIEW = "PLAN_FOR_REVIEW"
    PLAN_REVIEWED = "PLAN_REVIEWED"
    TASK_ASSIGNMENT = "TASK_ASSIGNMENT"
    TASK_COMPLETED = "TASK_COMPLETED"
    RESEARCH_FINDINGS = "RESEARCH_FINDINGS"
    NEED_HELP = "NEED_HELP"
    WORK_REQUEST = "WORK_REQUEST"
    STATUS_UPDATE = "STATUS_UPDATE"


class SwarmAgent:
    """Base class for swarm agents with role-specific capabilities."""

    def __init__(self, role: str, capabilities: list[str], llm_provider: Optional[LLMProvider] = None):
        self.role = role
        self.capabilities = capabilities
        self._llm_provider = llm_provider
        self._mail: Optional[SwarmMail] = None
        self._learning: Optional[LearningIntegration] = None

    def set_llm_provider(self, provider: LLMProvider) -> None:
        """Set the LLM provider for this agent."""
        self._llm_provider = provider

    def set_mail(self, mail: SwarmMail) -> None:
        """Set the SwarmMail instance for this agent."""
        self._mail = mail

    def set_learning(self, learning: LearningIntegration) -> None:
        """Set the LearningIntegration instance for this agent."""
        self._learning = learning
    
    def execute(self, state: SpineState, capability: str, **kwargs) -> dict[str, Any]:
        """Execute a capability within the given state context using LLM if available."""
        if self._llm_provider:
            prompt = self._build_prompt(state, capability, **kwargs)
            response = self._llm_provider.generate(prompt)
            return self._parse_response(response, capability)
        return self._execute_stub(state, capability, **kwargs)

    def _build_prompt(self, state: SpineState, capability: str, **kwargs) -> str:
        """Build LLM prompt for capability execution."""
        return f"""You are a {self.role} agent with capabilities: {self.capabilities}.
Current task: {state.get('requirement', 'Unknown')}
Capability: {capability}
State: {state}
Additional context: {kwargs}

Provide your response:"""

    def _parse_response(self, response: str, capability: str) -> dict[str, Any]:
        """Parse LLM response into structured result."""
        return {"type": capability, "result": response, "from_role": self.role}

    def _execute_stub(self, state: SpineState, capability: str, **kwargs) -> dict[str, Any]:
        """Fallback stub execution when no LLM is available."""
        raise NotImplementedError
    
    def send_message(self, to: str, subject: str, body: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Send a message via SwarmMail if available."""
        if self._mail:
            return self._mail.send(to, subject, body)
        return None
    
    def request_plan_review(self, plan: dict[str, Any], critic_role: str = "critic") -> None:
        """Send plan for review to critic agent."""
        self.send_message(
            to=critic_role,
            subject=MessageTypes.PLAN_FOR_REVIEW,
            body={"plan": plan, "from_role": self.role}
        )
    
    def assign_task(self, to: str, task_id: str, task_data: dict[str, Any]) -> None:
        """Assign a task to another agent."""
        self.send_message(
            to=to,
            subject=MessageTypes.TASK_ASSIGNMENT,
            body={"task_id": task_id, "task": task_data, "from_role": self.role}
        )
    
    def report_task_completed(self, to: str, task_id: str, result: dict[str, Any]) -> None:
        """Report task completion to coordinator."""
        self.send_message(
            to=to,
            subject=MessageTypes.TASK_COMPLETED,
            body={"task_id": task_id, "result": result, "from_role": self.role}
        )
    
    def report_pattern_completion(
        self,
        pattern: Any,
        task_id: str,
        work_item_id: str,
        success: bool,
        context: dict[str, Any] | None = None
    ) -> None:
        """Report pattern completion for learning."""
        if self._learning:
            self._learning.record_pattern_completion(
                pattern, task_id, work_item_id, success, context
            )


class ExplorerAgent(SwarmAgent):
    """Analyzes requirements and extracts key information."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        super().__init__("explorer", ["parse", "identify_constraints"], llm_provider)

    def _execute_stub(self, state: SpineState, capability: str, **kwargs) -> dict[str, Any]:
        if capability == "parse":
            return {
                "type": "analysis",
                "requirement": state["requirement"],
                "components": ["core", "constraints", "success_criteria"]
            }
        elif capability == "identify_constraints":
            return {
                "type": "constraints",
                "technical": [],
                "business": []
            }
        return {}


class SMEAgent(SwarmAgent):
    """Researches best practices and existing solutions."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        super().__init__("sme", ["search", "analyze", "synthesize"], llm_provider)

    def _execute_stub(self, state: SpineState, capability: str, **kwargs) -> dict[str, Any]:
        if capability == "search":
            return {
                "type": "research",
                "patterns": [],
                "references": []
            }
        return {}


class PlannerAgent(SwarmAgent):
    """Creates detailed execution plans."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        super().__init__("planner", ["draft", "refine", "finalise"], llm_provider)

    def _execute_stub(self, state: SpineState, capability: str, **kwargs) -> dict[str, Any]:
        if capability == "draft":
            return {
                "type": "plan",
                "tasks": [],
                "dependencies": {}
            }
        return {}


class CriticAgent(SwarmAgent):
    """Reviews plans and implementations."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        super().__init__("critic", ["review", "verify_drift", "scan_placeholders"], llm_provider)

    def _execute_stub(self, state: SpineState, capability: str, **kwargs) -> dict[str, Any]:
        if capability == "review":
            return {
                "type": "review",
                "approved": True,
                "issues": []
            }
        return {}