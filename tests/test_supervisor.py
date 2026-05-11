"""Tests for spine.swarm.supervisor - swarm coordinator with agent roles and gates."""

import sys
import pytest
from pathlib import Path

# Ensure spine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.swarm.supervisor import (
    AgentRole,
    SupervisorSwarmAgent,
    Supervisor,
    create_supervisor,
    create_explorer_agent,
    create_sme_agent,
    create_planner_agent,
    create_critic_agent,
    _get_create_supervisor,
    GateEnforcementError,
)


# --- Fixtures ---

@pytest.fixture
def sample_agents():
    """Create a set of sample agents for testing."""
    return [
        SupervisorSwarmAgent(role=AgentRole.EXPLORER, name="explorer", system_prompt="Analyze"),
        SupervisorSwarmAgent(role=AgentRole.PLANNER, name="planner", system_prompt="Plan"),
        SupervisorSwarmAgent(role=AgentRole.CODER, name="coder", system_prompt="Code"),
        SupervisorSwarmAgent(role=AgentRole.CRITIC, name="critic", system_prompt="Review"),
    ]


@pytest.fixture
def supervisor(sample_agents):
    """Create a Supervisor with sample agents."""
    return Supervisor(agents=sample_agents)


# --- AgentRole tests ---

class TestAgentRole:
    """Test AgentRole constants."""

    def test_all_roles_defined(self):
        """All standard roles should be defined."""
        assert AgentRole.EXPLORER == "explorer"
        assert AgentRole.SME == "sme"
        assert AgentRole.PLANNER == "planner"
        assert AgentRole.CRITIC == "critic"
        assert AgentRole.CODER == "coder"
        assert AgentRole.REVIEWER == "reviewer"
        assert AgentRole.TEST_ENGINEER == "test_engineer"
        assert AgentRole.ANALYST == "analyst"
        assert AgentRole.DESIGNER == "designer"

    def test_roles_are_unique(self):
        """All role values should be unique."""
        roles = [
            AgentRole.EXPLORER,
            AgentRole.SME,
            AgentRole.PLANNER,
            AgentRole.CRITIC,
            AgentRole.CODER,
            AgentRole.REVIEWER,
            AgentRole.TEST_ENGINEER,
            AgentRole.ANALYST,
            AgentRole.DESIGNER,
        ]
        assert len(roles) == len(set(roles))


# --- SwarmAgent tests ---

class TestSwarmAgentInit:
    """Test SupervisorSwarmAgent initialization."""

    def test_agent_basic_fields(self):
        """SupervisorSwarmAgent should store role, name, and system_prompt."""
        agent = SupervisorSwarmAgent(
            role=AgentRole.CODER,
            name="dev",
            system_prompt="Write code"
        )
        assert agent.role == "coder"
        assert agent.name == "dev"
        assert agent.system_prompt == "Write code"
        assert agent._agent_provider is None

    def test_agent_agent_provider_none(self):
        """SupervisorSwarmAgent should initialize with _agent_provider=None."""
        agent = SupervisorSwarmAgent(role="coder", name="c", system_prompt="code")
        assert agent._agent_provider is None

    def test_invalid_role_raises_error(self):
        """SupervisorSwarmAgent should raise InvalidAgentRoleError for invalid role."""
        from spine.swarm.agents import InvalidAgentRoleError
        with pytest.raises(InvalidAgentRoleError):
            SupervisorSwarmAgent(role="invalid_role", name="x", system_prompt="test")


class TestSwarmAgentSetLLM:
    """Test SupervisorSwarmAgent.set_agent_provider()."""

    def test_set_agent_provider_stores(self):
        """set_agent_provider should store the provider."""
        agent = SupervisorSwarmAgent(role="coder", name="c", system_prompt="code")
        fake_llm = type("FakeLLM", (), {"generate": lambda self, p: "output"})()
        agent.set_agent_provider(fake_llm)
        assert agent._agent_provider == fake_llm


class TestSwarmAgentCreateNode:
    """Test SupervisorSwarmAgent.create_node()."""

    def test_create_node_returns_callable(self):
        """create_node should return a callable node function."""
        agent = SupervisorSwarmAgent(role="explorer", name="e", system_prompt="Analyze")
        node = agent.create_node()
        assert callable(node)

    def test_create_node_no_agent_returns_error(self):
        """create_node without agent should return error dict (no fallback)."""
        agent = SupervisorSwarmAgent(role="explorer", name="e", system_prompt="Analyze")
        node = agent.create_node()
        state = {"requirement": "Build web app"}
        result = node(state)

        assert "agent_output" in result
        assert "error" in result["agent_output"].lower() or "No agent provider" in result["agent_output"]

    def test_create_node_with_agent(self):
        """create_node with agent should call provider.execute()."""
        from spine.providers.agents import AgentResult
        agent = SupervisorSwarmAgent(role="coder", name="c", system_prompt="Code")
        fake_agent = type("FakeAgent", (), {
            "enabled": True,
            "execute": lambda self, p, **kw: AgentResult(output="def hello(): pass", exit_code=0),
        })()
        agent.set_agent_provider(fake_agent)
        node = agent.create_node()
        state = {"requirement": "Write hello world"}
        result = node(state)
        assert result["agent_output"] == "def hello(): pass"

    def test_create_node_include_previous_output(self):
        """create_node should include previous output in prompt when agent available."""
        from spine.providers.agents import AgentResult
        agent = SupervisorSwarmAgent(role="explorer", name="e", system_prompt="Analyze")
        captured_prompt = []
        class FakeAgent:
            enabled = True
            def execute(self_self, prompt, **kw):
                captured_prompt.append(prompt)
                return AgentResult(output="new output", exit_code=0)
        agent.set_agent_provider(FakeAgent())
        node = agent.create_node()
        state = {
            "requirement": "Build thing",
            "agent_output": "previous analysis",
        }
        result = node(state)
        assert result["agent_output"] == "new output"
        # Verify the prompt was built and agent was called
        assert len(captured_prompt) > 0

    def test_create_node_agent_error_handling(self):
        """create_node should handle agent provider errors gracefully."""
        agent = SupervisorSwarmAgent(role="coder", name="c", system_prompt="Code")
        class FakeAgent:
            enabled = True
            def execute(self_self, prompt, **kw):
                raise ValueError("Agent service unavailable")
        agent.set_agent_provider(FakeAgent())
        node = agent.create_node()
        state = {"requirement": "test"}
        result = node(state)
        assert "error" in result["agent_output"].lower() or "c agent error" in result["agent_output"]

    def test_create_node_state_keys(self):
        """create_node should return state with agent_output, agent_role, agent_name."""
        agent = SupervisorSwarmAgent(role="sme", name="researcher", system_prompt="Research")
        node = agent.create_node()
        state = {"requirement": "Research topic"}
        result = node(state)
        assert "agent_output" in result
        assert "agent_role" in result
        assert "agent_name" in result
        assert result["agent_role"] == "sme"
        assert result["agent_name"] == "researcher"


# --- Factory function tests ---

class TestAgentFactoryFunctions:
    """Test create_*_agent factory functions."""

    def test_create_explorer_agent(self):
        """create_explorer_agent should return configured explorer."""
        agent = create_explorer_agent()
        assert isinstance(agent, SupervisorSwarmAgent)
        assert agent.role == AgentRole.EXPLORER
        assert agent.name == "explorer"
        assert "analyze" in agent.system_prompt.lower()
        assert "requirements" in agent.system_prompt.lower()

    def test_create_sme_agent(self):
        """create_sme_agent should return configured SME."""
        agent = create_sme_agent()
        assert agent.role == AgentRole.SME
        assert agent.name == "sme"
        assert "subject matter expert" in agent.system_prompt.lower()

    def test_create_planner_agent(self):
        """create_planner_agent should return configured planner."""
        agent = create_planner_agent()
        assert agent.role == AgentRole.PLANNER
        assert agent.name == "planner"
        assert "execution plans" in agent.system_prompt.lower()

    def test_create_critic_agent(self):
        """create_critic_agent should return configured critic."""
        agent = create_critic_agent()
        assert agent.role == AgentRole.CRITIC
        assert agent.name == "critic"
        assert "review" in agent.system_prompt.lower()


# --- Supervisor tests ---

class TestSupervisorInit:
    """Test Supervisor initialization."""

    def test_supervisor_empty(self):
        """Supervisor should work with no agents."""
        sup = Supervisor()
        assert sup.agents == []
        assert sup._agent_map == {}

    def test_supervisor_with_agents(self, sample_agents):
        """Supervisor should build agent map from agents list."""
        sup = Supervisor(agents=sample_agents)
        assert len(sup.agents) == 4
        assert sup._agent_map["explorer"] is sample_agents[0]
        assert sup._agent_map["planner"] is sample_agents[1]

    def test_supervisor_agent_map_by_role(self, sample_agents):
        """Supervisor agent map should key by role, not name."""
        sup = Supervisor(agents=sample_agents)
        assert "explorer" in sup._agent_map
        assert "planner" in sup._agent_map
        assert "coder" in sup._agent_map
        assert "critic" in sup._agent_map


class TestSupervisorSpawnAgent:
    """Test Supervisor.spawn_agent()."""

    def test_spawn_existing_agent(self, supervisor):
        """spawn_agent should return agent for existing role."""
        agent = supervisor.spawn_agent("explorer")
        assert agent is not None
        assert agent.role == "explorer"

    def test_spawn_nonexistent_agent(self, supervisor):
        """spawn_agent should return None for unknown role."""
        agent = supervisor.spawn_agent("unknown_role")
        assert agent is None

    def test_spawn_all_roles(self, supervisor):
        """spawn_agent should find all registered roles."""
        for role in ["explorer", "planner", "coder", "critic"]:
            agent = supervisor.spawn_agent(role)
            assert agent is not None
            assert agent.role == role


class TestSupervisorRunGates:
    """Test Supervisor.run_gates()."""

    def test_run_gates_success(self, supervisor):
        """run_gates should execute gates successfully."""
        context = {"requirement": "Build feature"}
        results = supervisor.run_gates(["explorer", "coder"], context)
        assert "explorer" in results
        assert "coder" in results
        assert results["explorer"]["status"] == "completed"
        assert results["coder"]["status"] == "completed"
        assert "agent_output" in results["explorer"]["result"]

    def test_run_gates_single_agent(self, supervisor):
        """run_gates should work with a single gate."""
        results = supervisor.run_gates(["planner"], {"req": "test"})
        assert results["planner"]["status"] == "completed"

    def test_run_gates_missing_agent(self, supervisor):
        """run_gates should mark missing agent as failed and raise."""
        with pytest.raises(GateEnforcementError):
            supervisor.run_gates(["nonexistent"], {})

    def test_run_gates_mixed_success_failure(self, supervisor):
        """run_gates should handle mix of existing and missing agents."""
        with pytest.raises(GateEnforcementError):
            supervisor.run_gates(["explorer", "ghost", "coder"], {"req": "test"})

    def test_run_gates_empty_list(self, supervisor):
        """run_gates with empty list should return empty dict."""
        results = supervisor.run_gates([], {})
        assert results == {}

    def test_run_gates_no_agents_supervisor(self):
        """run_gates on empty supervisor should fail all gates."""
        sup = Supervisor()
        with pytest.raises(GateEnforcementError):
            sup.run_gates(["explorer"], {})

    def test_run_gates_agent_error_handling(self):
        """run_gates should handle agent errors gracefully."""
        agent = SupervisorSwarmAgent(role="coder", name="te", system_prompt="Test")
        sup = Supervisor(agents=[agent])
        results = sup.run_gates(["coder"], {"requirement": "test"})
        assert results["coder"]["status"] == "completed"


# --- create_supervisor tests ---

class TestCreateSupervisor:
    """Test create_supervisor() function."""

    def test_create_supervisor_returns_config_without_langgraph(self):
        """create_supervisor should return config dict when langgraph_supervisor unavailable."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("spine.swarm.supervisor._get_create_supervisor", lambda: None)
            agents = [
                SupervisorSwarmAgent(role=AgentRole.EXPLORER, name="e", system_prompt="Analyze"),
                SupervisorSwarmAgent(role=AgentRole.PLANNER, name="p", system_prompt="Plan"),
            ]
            result = create_supervisor(agents)

        assert isinstance(result, dict)
        assert result["mode"] == "config"
        assert result["supervisor_name"] == "spine_supervisor"
        assert len(result["agents"]) == 2
        assert result["state_schema"] is not None

    def test_create_supervisor_with_state_schema(self):
        """create_supervisor should accept state_schema parameter."""
        from langgraph.graph import MessagesState
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("spine.swarm.supervisor._get_create_supervisor", lambda: None)
            agents = [SupervisorSwarmAgent(role=AgentRole.ANALYST, name="a", system_prompt="x")]
            result = create_supervisor(agents, state_schema=MessagesState)
        assert result["state_schema"] is MessagesState

    def test_get_create_supervisor_import_success(self):
        """_get_create_supervisor should return create_supervisor when available."""
        fn = _get_create_supervisor()
        # May succeed or return None depending on langgraph_supervisor install
        if fn is not None:
            assert callable(fn)
        # If it returns None, that's also valid (package not installed)


# --- Integration tests ---

class TestSupervisorIntegration:
    """Integration tests for Supervisor."""

    def test_full_supervisor_workflow(self):
        """Test full workflow: create agents, supervisor, run gates."""
        agents = [
            create_explorer_agent(),
            create_planner_agent(),
            create_critic_agent(),
        ]
        supervisor = Supervisor(agents=agents)

        # Spawn and verify
        explorer = supervisor.spawn_agent("explorer")
        assert explorer is not None

        # Run gates
        context = {
            "requirement": "Build a REST API with authentication",
            "context": {"stack": "Python"},
        }
        results = supervisor.run_gates(["explorer", "planner"], context)

        assert results["explorer"]["status"] == "completed"
        assert results["planner"]["status"] == "completed"
        assert "agent_output" in results["explorer"]["result"]
        assert "agent_output" in results["planner"]["result"]

    def test_supervisor_multiple_gate_runs(self):
        """Supervisor should support multiple gate runs."""
        agents = [
            SupervisorSwarmAgent(role=AgentRole.ANALYST, name="r1", system_prompt="First"),
            SupervisorSwarmAgent(role=AgentRole.DESIGNER, name="r2", system_prompt="Second"),
        ]
        sup = Supervisor(agents=agents)

        run1 = sup.run_gates(["analyst"], {"step": 1})
        assert run1["analyst"]["status"] == "completed"

        run2 = sup.run_gates(["designer"], {"step": 2})
        assert run2["designer"]["status"] == "completed"

        # Agents persist across runs
        run3 = sup.run_gates(["analyst", "designer"], {"step": 3})
        assert len(run3) == 2
    
    def test_quality_gate_enforcement(self, supervisor):
        """Supervisor should support quality gate."""
        context = {"plan": {"tasks": ["task1", "task2"]}}
        results = supervisor.run_gates(["quality"], context)
        assert "quality" in results
        assert results["quality"]["status"] in ["passed", "failed"]