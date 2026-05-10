"""Swarm agents with role-specific capabilities and messaging."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, AsyncIterator

from ..core.state_machine import SpineState
from ..providers.llm import LLMProvider
from ..providers.agents import AgentProvider, AgentResult
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


class InvalidAgentRoleError(Exception):
    """Raised when an invalid agent role is specified."""
    pass


class AgentRoleValidator:
    """Validates agent roles against allowed roles."""
    
    VALID_ROLES = frozenset([
        "explorer", "sme", "planner", "critic", 
        "coder", "reviewer", "test_engineer", "analyst", "designer"
    ])
    
    @classmethod
    def validate(cls, role: str) -> bool:
        """Check if a role is valid."""
        return role in cls.VALID_ROLES
    
    @classmethod
    def get_valid_roles(cls) -> list[str]:
        """Return list of valid agent roles."""
        return sorted(cls.VALID_ROLES)


class SwarmAgent:
    """Base class for swarm agents with role-specific capabilities.

    Uses structured prompts from the spine.prompts module for consistent,
    instructional prompts that follow DeepAgents/LangChain best practices.
    
    Key features:
    - Role-specific structured prompts with process guidance
    - Tool instructions for available capabilities
    - Output format specifications for structured responses
    - Workflow context injection for agent coordination
    """

    # Roles that use AgentProvider (implementation) vs LLMProvider (decision-making)
    IMPLEMENTATION_ROLES = frozenset(["coder", "test_engineer", "reviewer"])

    def __init__(
        self,
        role: str,
        capabilities: list[str],
        llm_provider: Optional[LLMProvider] = None,
        agent_provider: Optional[AgentProvider] = None,
        prompt_config: Optional[Any] = None,  # PromptConfig, avoid circular import
    ):
        # Validate role
        if not AgentRoleValidator.validate(role):
            raise InvalidAgentRoleError(
                f"Invalid agent role: '{role}'. Valid roles: {AgentRoleValidator.get_valid_roles()}"
            )
        self.role = role
        self.capabilities = capabilities
        self._llm_provider = llm_provider
        self._agent_provider = agent_provider
        self._mail: Optional[SwarmMail] = None
        self._learning: Optional[LearningIntegration] = None
        self._prompt_config = prompt_config
        self._prompt_builder: Optional[Any] = None  # PromptBuilder, lazy init

    def _get_prompt_builder(self):
        """Get or create the PromptBuilder for this agent."""
        if self._prompt_builder is None:
            from ..prompts import PromptBuilder, PromptConfig
            config = self._prompt_config or PromptConfig()
            self._prompt_builder = PromptBuilder(role=self.role, config=config)
        return self._prompt_builder

    def set_llm_provider(self, provider: LLMProvider) -> None:
        """Set the LLM provider for this agent."""
        self._llm_provider = provider

    def set_agent_provider(self, provider: AgentProvider) -> None:
        """Set the agent provider for this agent.

        Agent providers delegate implementation work to external coding
        agents (OpenCode, Codex CLI, Claude Code).  Only implementation
        roles (coder, test_engineer, reviewer) use agent providers.
        Decision-making roles (planner, critic, explorer) use LLM directly.
        """
        self._agent_provider = provider

    def set_mail(self, mail: SwarmMail) -> None:
        """Set the SwarmMail instance for this agent."""
        self._mail = mail

    def set_learning(self, learning: LearningIntegration) -> None:
        """Set the LearningIntegration instance for this agent."""
        self._learning = learning
    
    def execute(self, state: SpineState, capability: str, **kwargs) -> dict[str, Any]:
        """Execute a capability within the given state context.

        For implementation roles (coder, test_engineer, reviewer) with an
        agent_provider set, delegates to the external agent.  Otherwise uses
        the LLM provider for decision-making tasks.
        """
        # Implementation roles with agent provider: delegate to external agent
        if (self._agent_provider
                and self.role in self.IMPLEMENTATION_ROLES
                and self._agent_provider.enabled):
            prompt = self._build_prompt(state, capability, **kwargs)
            workdir = kwargs.pop("workdir", None) or state.get("variables", {}).get("workdir")
            agent_result: AgentResult = self._agent_provider.execute(
                prompt, workdir=workdir, **kwargs,
            )
            return self._parse_agent_result(agent_result, capability)

        # Decision-making roles or no agent provider: use LLM directly
        if self._llm_provider:
            prompt = self._build_prompt(state, capability, **kwargs)
            response = self._llm_provider.generate(prompt)
            return self._parse_response(response, capability)
        return self._execute_stub(state, capability, **kwargs)
    
    async def execute_streaming(self, state: SpineState, capability: str, **kwargs) -> AsyncIterator[str]:
        """Execute a capability with streaming support.
        
        Yields chunks of the LLM response as they become available.
        Falls back to non-streaming execution if LLM doesn't support streaming.
        """
        if self._llm_provider:
            prompt = self._build_prompt(state, capability, **kwargs)
            try:
                if hasattr(self._llm_provider, 'stream'):
                    async for chunk in self._llm_provider.stream(prompt):
                        yield chunk
                else:
                    # Fallback to non-streaming
                    response = self._llm_provider.generate(prompt)
                    yield response
            except Exception as e:
                yield f"[Agent {self.role} streaming error: {e}]"
        else:
            # No LLM provider - return stub result
            result = self._execute_stub(state, capability, **kwargs)
            yield str(result)
    
    def _build_prompt(self, state: SpineState, capability: str, **kwargs) -> str:
        """Build LLM prompt for capability execution.

        Uses the PromptBuilder to compose a structured prompt with:
        - Role-specific instructions
        - Tool instructions (if applicable)
        - Workflow context
        - Output format specification
        - Task-specific context
        """
        builder = self._get_prompt_builder()
        
        # Determine available tools based on role and providers
        tools = self._get_available_tools()
        
        # Get previous outputs for workflow context
        previous_outputs = kwargs.pop("previous_outputs", None)
        
        # Build the complete prompt
        return builder.build(
            state=dict(state) if hasattr(state, '__iter__') else state,
            capability=capability,
            tools=tools,
            previous_outputs=previous_outputs,
            **kwargs,
        )
    
    def _get_available_tools(self) -> list[str]:
        """Determine which tools are available for this agent."""
        tools = []
        
        # All agents potentially have filesystem access
        tools.append("filesystem")
        
        # Implementation roles with agent provider have external agent access
        if self.role in self.IMPLEMENTATION_ROLES and self._agent_provider:
            tools.append("agent_provider")
        
        return tools

    def _parse_response(self, response: str, capability: str) -> dict[str, Any]:
        """Parse LLM response into structured result."""
        # Try to parse as JSON first
        import json
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                parsed["from_role"] = self.role
                parsed["type"] = capability
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Fall back to wrapping in result
        return {"type": capability, "result": response, "from_role": self.role}

    def _parse_agent_result(self, result: AgentResult, capability: str) -> dict[str, Any]:
        """Parse AgentResult into the standard swarm result dict."""
        return {
            "type": capability,
            "result": result.output,
            "from_role": self.role,
            "success": result.success,
            "exit_code": result.exit_code,
            "files_changed": result.files_changed,
            "error": result.error,
            "metadata": result.metadata,
        }

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

    def evaluate_gate(self, gate_type: str, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a gate for quality control. Override in subclasses."""
        return {"status": "not_implemented", "gate_type": gate_type}


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
    """Reviews plans and implementations with LLM-powered evaluation."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        super().__init__("critic", ["review", "verify_drift", "scan_placeholders"], llm_provider)
        self.gate_name = "quality"

    def _execute_stub(self, state: SpineState, capability: str, **kwargs) -> dict[str, Any]:
        if capability == "review":
            return {
                "type": "review",
                "approved": True,
                "issues": []
            }
        return {}
    
    def evaluate_gate(self, gate_type: str, context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate quality gate with proper gate result.
        
        Uses structured prompts from the prompts module for consistent
        review behavior.
        
        Args:
            gate_type: Type of gate (e.g., "quality", "critic")
            context: Context containing plan and variables
            
        Returns:
            Gate evaluation result with approval status
        """
        plan = context.get("plan", {})
        
        if not plan:
            return {
                "status": "failed",
                "approved": False,
                "reason": "No plan provided for review"
            }
        
        if self._llm_provider and self._llm_provider.enabled:
            try:
                # Use the prompt builder for structured review
                state = {
                    "requirement": context.get("requirement", ""),
                    "current_phase": "REVIEW",
                    "variables": context.get("variables", {}),
                }
                prompt = self._build_prompt(
                    state, 
                    capability="review",
                    extra_context=f"Plan to review:\n{plan}",
                )
                response = self._llm_provider.generate(prompt)
                return self._parse_critic_response(response)
            except Exception as e:
                return {
                    "status": "failed",
                    "approved": False,
                    "reason": f"LLM evaluation error: {e}"
                }
        else:
            # Stub evaluation - check basic plan structure
            tasks = plan.get("tasks", [])
            if not tasks:
                return {
                    "status": "failed",
                    "approved": False,
                    "reason": "Plan has no tasks"
                }
            return {
                "status": "passed",
                "approved": True,
                "issues": [],
                "recommendations": []
            }
    
    def _parse_critic_response(self, response: str) -> dict[str, Any]:
        """Parse LLM critic response into structured result."""
        import json
        try:
            result = json.loads(response)
            return {
                "status": "passed" if result.get("approved", False) else "failed",
                "approved": result.get("approved", False),
                "issues": result.get("issues", []),
                "recommendations": result.get("recommendations", [])
            }
        except json.JSONDecodeError:
            # Non-JSON response - try to extract meaning
            approved = "approved" in response.lower() and "not approved" not in response.lower()
            return {
                "status": "passed" if approved else "failed",
                "approved": approved,
                "issues": [],
                "recommendations": [response[:200]]
            }