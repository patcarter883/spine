"""Swarm supervisor for coordinating multiple agents."""

import asyncio
from typing import Any, Optional, AsyncIterator
from langgraph.graph import MessagesState

from ..swarm.agents import SwarmAgent, AgentRoleValidator, InvalidAgentRoleError
from ..swarm.gates import QualityGate, PreCheckBatch


# Import the real langgraph_supervisor at call time to avoid import issues
def _get_create_supervisor():
    """Lazily import langgraph_supervisor to handle missing package gracefully."""
    try:
        from langgraph_supervisor import create_supervisor
        return create_supervisor
    except ImportError:
        return None


class AgentRole:
    """Standard agent roles in SPINE swarm."""
    EXPLORER = "explorer"   # Analyzes requirements
    SME = "sme"             # Subject matter expert, researches
    PLANNER = "planner"     # Creates execution plans
    CRITIC = "critic"       # Reviews and validates
    CODER = "coder"         # Implements code
    REVIEWER = "reviewer"   # Code review
    TEST_ENGINEER = "test_engineer"  # Testing
    ANALYST = "analyst"     # Risk assessment
    DESIGNER = "designer"   # UI/UX specifications


class SupervisorSwarmAgent(SwarmAgent):
    """A specialized agent in the swarm with additional features."""
    
    def __init__(self, role: str, name: str, system_prompt: str, agent_provider: Any | None = None):
        # Validate role before initializing
        if not AgentRoleValidator.validate(role):
            raise InvalidAgentRoleError(
                f"Invalid agent role: '{role}'. Valid roles: {AgentRoleValidator.get_valid_roles()}"
            )
        super().__init__(role, [name], agent_provider=agent_provider)
        self.name = name
        self.system_prompt = system_prompt
    
    async def execute_streaming(self, state: dict[str, Any]) -> AsyncIterator[str]:
        """Execute with streaming support."""
        result = self.execute(state, "analyze")
        if "error" in result:
            yield result["error"]
        else:
            yield str(result.get("output", result))
    
    def create_node(self):
        """Create a LangGraph node from this agent.
        
        Returns a node function that delegates to the agent provider.
        """
        def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            # Use the parent execute method
            result = self.execute(state, "analyze")
            # execute() returns dict with 'result' key from _parse_agent_result
            output = result.get("result", result.get("output", str(result)))
            return {
                "agent_output": output,
                "agent_role": self.role,
                "agent_name": self.name
            }
        return agent_node


def create_explorer_agent(agent_provider: Any | None = None) -> SupervisorSwarmAgent:
    """Create the explorer agent for requirement analysis."""
    from ..prompts.roles import EXPLORER_PROMPT
    return SupervisorSwarmAgent(
        role=AgentRole.EXPLORER,
        name="explorer",
        system_prompt=EXPLORER_PROMPT,
        agent_provider=agent_provider,
    )


def create_sme_agent(agent_provider: Any | None = None) -> SupervisorSwarmAgent:
    """Create the SME agent for research."""
    from ..prompts.roles import SME_PROMPT
    return SupervisorSwarmAgent(
        role=AgentRole.SME,
        name="sme",
        system_prompt=SME_PROMPT,
        agent_provider=agent_provider,
    )


def create_planner_agent(agent_provider: Any | None = None) -> SupervisorSwarmAgent:
    """Create the planner agent for execution planning."""
    from ..prompts.roles import PLANNER_PROMPT
    return SupervisorSwarmAgent(
        role=AgentRole.PLANNER,
        name="planner",
        system_prompt=PLANNER_PROMPT,
        agent_provider=agent_provider,
    )


def create_critic_agent() -> SupervisorSwarmAgent:
    """Create the critic agent for validation.
    
    Uses structured prompts from spine.prompts.roles for comprehensive
    review and validation instructions.
    """
    from ..prompts.roles import CRITIC_PROMPT
    return SupervisorSwarmAgent(
        role=AgentRole.CRITIC,
        name="critic",
        system_prompt=CRITIC_PROMPT,
    )


def create_coder_agent() -> SupervisorSwarmAgent:
    """Create the coder agent for implementation.
    
    Uses structured prompts from spine.prompts.roles for implementation
    instructions with tool usage guidance.
    """
    from ..prompts.roles import CODER_PROMPT
    return SupervisorSwarmAgent(
        role=AgentRole.CODER,
        name="coder",
        system_prompt=CODER_PROMPT,
    )


def create_supervisor(agents: list[SupervisorSwarmAgent], state_schema: type = MessagesState):
    """
    Create a supervisor that coordinates multiple swarm agents.
    
    Uses LangGraph supervisor for multi-agent orchestration when available.
    Falls back to configuration dict when langgraph_supervisor is not installed.
    
    Args:
        agents: List of SupervisorSwarmAgent instances to coordinate
        state_schema: The state schema for the supervisor graph
        
    Returns:
        A compiled supervisor graph or configuration dict
    """
    create_supervisor_fn = _get_create_supervisor()
    
    if create_supervisor_fn is not None:
        # Use the real LangGraph supervisor
        agent_graphs = {}
        for agent in agents:
            node = agent.create_node()
            agent_graphs[agent.name] = node
        
        supervisor = create_supervisor_fn(
            agents=agents,
            state_schema=state_schema
        )
        return supervisor
    else:
        # Fallback: return configuration for manual orchestration
        return {
            "agents": agents,
            "state_schema": state_schema,
            "supervisor_name": "spine_supervisor",
            "mode": "config"
        }


class Supervisor:
    """
    High-level supervisor for managing swarm execution.
    
    Coordinates parallel agents and enforces swarm gates.
    """
    
    def __init__(self, agents: Optional[list[SupervisorSwarmAgent]] = None):
        self.agents = agents or []
        self._agent_map = {a.role: a for a in self.agents}
    
    def spawn_agent(self, role: str) -> Optional[SupervisorSwarmAgent]:
        """Spawn an agent by role."""
        return self._agent_map.get(role)
    
    def run_gates(self, gate_names: list[str], context: dict[str, Any]) -> dict[str, Any]:
        """Run swarm gates with enforcement.
        
        Executes each gate by:
        1. Looking up the agent by gate name (or using predefined gate)
        2. Creating a LangGraph node from the agent
        3. Executing the node with the provided context
        4. Recording the result
        5. Enforcing gate requirements (fail if required gate fails)
        
        Args:
            gate_names: List of gate names (agent roles) to execute
            context: Execution context passed to each gate agent
            
        Returns:
            Dict mapping gate names to their results and status
            
        Raises:
            GateEnforcementError: When a required gate fails
        """
        results = {}
        gate_errors = []
        
        for gate_name in gate_names:
            # Check if this is a predefined gate (quality, precheck_batch)
            gate_result = self._run_named_gate(gate_name, context)
            
            if gate_result is None:
                # Not a predefined gate, try to find agent
                agent = self.spawn_agent(gate_name)
                if agent:
                    # Create and execute the agent node
                    node = agent.create_node()
                    try:
                        gate_result = node(context)
                        results[gate_name] = {
                            "status": "completed",
                            "result": gate_result
                        }
                    except Exception as e:
                        results[gate_name] = {
                            "status": "failed",
                            "error": str(e)
                        }
                        gate_errors.append(gate_name)
                else:
                    # Gate agent not found - mark as failed
                    results[gate_name] = {
                        "status": "failed",
                        "error": f"No agent found for gate: {gate_name}"
                    }
                    gate_errors.append(gate_name)
            else:
                results[gate_name] = gate_result
        
        # Enforce gate requirements - raise error if any gate failed
        if gate_errors:
            failed_gate_info = {g: results[g] for g in gate_errors}
            raise GateEnforcementError(
                f"Gate enforcement failed for: {gate_errors}",
                gate_results=failed_gate_info
            )
        
        return results
    
    def _run_named_gate(self, gate_name: str, context: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Run a predefined gate by name.
        
        Returns None if gate_name is not a predefined gate.
        """
        if gate_name == "quality":
            gate = QualityGate()
            result = gate.evaluate(context)
            return {
                "status": "passed" if result.get("approved", False) else "failed",
                "result": result
            }
        elif gate_name == "precheck_batch":
            gate = PreCheckBatch()
            result = gate.evaluate(context)
            return {
                "status": "passed" if result.get("all_passed", False) else "failed",
                "result": result
            }
        return None


class GateEnforcementError(Exception):
    """Raised when gate enforcement fails."""
    
    def __init__(self, message: str, gate_results: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.gate_results = gate_results or {}