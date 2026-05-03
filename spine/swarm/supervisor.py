"""Swarm supervisor for coordinating multiple agents."""

from typing import Any, Optional
from langgraph.graph import MessagesState

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


class SwarmAgent:
    """A specialized agent in the swarm."""
    
    def __init__(self, role: str, name: str, system_prompt: str):
        self.role = role
        self.name = name
        self.system_prompt = system_prompt
        self._llm_provider = None  # Optional LLM provider
    
    def set_llm_provider(self, provider: Any) -> None:
        """Set the LLM provider for this agent."""
        self._llm_provider = provider
    
    def create_node(self):
        """Create a LangGraph node from this agent that executes LLM-backed.
        
        Returns a node function that:
        1. Extracts the requirement/context from state
        2. Builds a prompt using the agent's system prompt and current state
        3. Calls the LLM provider if available
        4. Returns updated state with agent output
        """
        def agent_node(state: dict[str, Any]) -> dict[str, Any]:
            # Extract the requirement and context from state
            requirement = state.get("requirement", "")
            current_output = state.get("agent_output", "")
            
            # Build the prompt
            prompt = f"{self.system_prompt}\n\nCurrent requirement: {requirement}"
            if current_output:
                prompt += f"\n\nPrevious output: {current_output}"
            
            # Call LLM provider if available
            if self._llm_provider:
                try:
                    llm_output = self._llm_provider.generate(prompt)
                    output = llm_output
                except Exception as e:
                    output = f"[Agent {self.name} error: {e}]"
            else:
                # Fallback stub output
                output = f"[{self.name} agent completed: {requirement[:100]}]"
            
            # Return updated state with agent output
            return {
                "agent_output": output,
                "agent_role": self.role,
                "agent_name": self.name
            }
        return agent_node


def create_explorer_agent() -> SwarmAgent:
    """Create the explorer agent for requirement analysis."""
    return SwarmAgent(
        role=AgentRole.EXPLORER,
        name="explorer",
        system_prompt=(
            "You analyze user requirements and extract key information. "
            "Identify the core problem, constraints, and success criteria. "
            "Output should be structured and actionable."
        )
    )


def create_sme_agent() -> SwarmAgent:
    """Create the SME agent for research."""
    return SwarmAgent(
        role=AgentRole.SME,
        name="sme",
        system_prompt=(
            "You are a subject matter expert. Research best practices, "
            "existing solutions, and technical patterns. Synthesize findings "
            "into actionable recommendations."
        )
    )


def create_planner_agent() -> SwarmAgent:
    """Create the planner agent for execution planning."""
    return SwarmAgent(
        role=AgentRole.PLANNER,
        name="planner",
        system_prompt=(
            "You create detailed execution plans from requirements and research. "
            "Break down work into discrete tasks with clear dependencies. "
            "Include testing and verification considerations."
        )
    )


def create_critic_agent() -> SwarmAgent:
    """Create the critic agent for validation."""
    return SwarmAgent(
        role=AgentRole.CRITIC,
        name="critic",
        system_prompt=(
            "You review plans and implementations for correctness, security, and completeness. "
            "Identify potential issues, gaps, and improvement opportunities. "
            "Be thorough but constructive."
        )
    )


def create_supervisor(agents: list[SwarmAgent], state_schema: type = MessagesState):
    """
    Create a supervisor that coordinates multiple swarm agents.
    
    Uses LangGraph supervisor for multi-agent orchestration when available.
    Falls back to configuration dict when langgraph_supervisor is not installed.
    
    Args:
        agents: List of SwarmAgent instances to coordinate
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
    
    def __init__(self, agents: Optional[list[SwarmAgent]] = None):
        self.agents = agents or []
        self._agent_map = {a.role: a for a in self.agents}
    
    def spawn_agent(self, role: str) -> Optional[SwarmAgent]:
        """Spawn an agent by role."""
        return self._agent_map.get(role)
    
    def run_gates(self, gate_names: list[str], context: dict[str, Any]) -> dict[str, Any]:
        """Run swarm gates and return results.
        
        Executes each gate by:
        1. Looking up the agent by gate name
        2. Creating a LangGraph node from the agent
        3. Executing the node with the provided context
        4. Recording the result
        
        Args:
            gate_names: List of gate names (agent roles) to execute
            context: Execution context passed to each gate agent
            
        Returns:
            Dict mapping gate names to their results and status
        """
        results = {}
        for gate_name in gate_names:
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
            else:
                # Gate agent not found - mark as failed
                results[gate_name] = {
                    "status": "failed",
                    "error": f"No agent found for gate: {gate_name}"
                }
        return results