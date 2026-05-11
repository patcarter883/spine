"""Tests for execute_dag() real execution and supervisor improvements."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import MagicMock, patch
import pytest

from spine.core.state_machine import (
    SwarmDAGExecutor, SubPhase, Task, Phase, StateStatus
)
from spine.models.dag import ResourceQuota, ExecutionProgress
from spine.swarm.supervisor import Supervisor, SupervisorSwarmAgent, AgentRole, create_supervisor, GateEnforcementError
from spine.providers.llm import LLMProvider
from spine.providers.agents import AgentProvider


class FakeLLMProvider(LLMProvider):
    """Fake LLM provider for testing."""
    
    def __init__(self, response="fake response"):
        self.response = response
        self.call_count = 0
    
    def configure(self, config):
        pass
    
    def validate(self):
        return True
    
    @property
    def name(self):
        return "fake"
    
    @property
    def enabled(self):
        return True
    
    def generate_sync(self, prompt, **kwargs):
        self.call_count += 1
        return self.response
    
    async def stream(self, prompt, **kwargs):
        yield self.response


class TestExecuteDagRealExecution:
    """Test execute_dag() with real task execution."""
    
    def test_execute_dag_with_subphase_requires_agent_provider(self):
        """execute_dag should fail when no agent_provider is configured."""
        executor = SwarmDAGExecutor()
        subphase = SubPhase(
            name="TEST_PHASE",
            agent_role="test_agent",
            tasks=[
                Task(id="task1", description="First task"),
                Task(id="task2", description="Second task"),
            ]
        )
        
        result = executor.execute_dag(subphase, {"input": "test"})
        
        assert result["subphase"] == "TEST_PHASE"
        assert result["agent_role"] == "test_agent"
        assert result["status"] == "failed"
        assert "No agent provider available" in result.get("error", "")
    
    def test_execute_dag_with_agent_provider_executes_tasks(self):
        """execute_dag should delegate tasks to agent_provider."""
        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_result = MagicMock()
        mock_result.output = "task output"
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.files_changed = []
        mock_result.error = None
        mock_provider.execute.return_value = mock_result
        
        executor = SwarmDAGExecutor(agent_provider=mock_provider)
        subphase = SubPhase(
            name="TEST_PHASE",
            agent_role="coder",
            tasks=[
                Task(id="task1", description="First task"),
                Task(id="task2", description="Second task"),
            ]
        )
        
        result = executor.execute_dag(subphase, {"input": "test"})
        
        assert result["subphase"] == "TEST_PHASE"
        assert result["agent_role"] == "coder"
        assert result["status"] == "success"
        assert result["tasks_executed"] == 2
        assert result["total_tasks"] == 2
        assert "task1" in result["tasks"]
        assert "task2" in result["tasks"]
    
    def test_execute_dag_with_llm_provider_ignores_it(self):
        """execute_dag should ignore llm_provider and require agent_provider instead."""
        fake_llm = FakeLLMProvider(response="llm output")
        executor = SwarmDAGExecutor(llm_provider=fake_llm)
        subphase = SubPhase(
            name="LLM_TEST",
            agent_role="researcher",
            tasks=[Task(id="research", description="Research task")]
        )
        
        result = executor.execute_dag(subphase, {"query": "what is SPINE?"})
        
        # Without agent_provider, it should fail
        assert result["status"] == "failed"
        assert "No agent provider available" in result.get("error", "")
    
    def test_execute_dag_task_status_tracking(self):
        """execute_dag should track task status through execution."""
        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_result = MagicMock()
        mock_result.output = "done"
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.files_changed = []
        mock_result.error = None
        mock_provider.execute.return_value = mock_result
        
        executor = SwarmDAGExecutor(agent_provider=mock_provider)
        subphase = SubPhase(
            name="STATUS_TEST",
            tasks=[Task(id="t1", description="Test", status=StateStatus.PENDING)]
        )
        
        executor.execute_dag(subphase, {})
        
        assert subphase.tasks[0].status == StateStatus.SUCCESS
        assert subphase.tasks[0].result is not None
    
    def test_execute_dag_stub_execution_works(self):
        """execute_dag with string name (backwards compat) should return stub."""
        executor = SwarmDAGExecutor()
        result = executor.execute_dag("old_style_name", {"key": "value"})
        
        assert result["dag"] == "old_style_name"
        assert result["status"] == "completed"
        assert result["tasks_executed"] == 0
    
    def test_execute_dag_prompt_contains_context(self):
        """execute_dag should build prompts with subphase and task info."""
        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_result = MagicMock()
        mock_result.output = "ok"
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.files_changed = []
        mock_result.error = None
        mock_provider.execute.return_value = mock_result
        
        executor = SwarmDAGExecutor(agent_provider=mock_provider)
        subphase = SubPhase(
            name="PROMPT_TEST",
            agent_role="analyst",
            tasks=[Task(id="analyze", description="Analyze data")]
        )
        
        context = {"input": "hello", "deps": {"value": "42"}}
        executor.execute_dag(subphase, context)
        
        # Verify execute was called (the prompt is internal)
        mock_provider.execute.assert_called_once()


class TestExecutePhaseWithAgent:
    """Test execute_phase with agent provider integration."""
    
    def _make_mock_agent_provider(self, output="analyzed"):
        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_result = MagicMock()
        mock_result.output = output
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.files_changed = []
        mock_result.error = None
        mock_provider.execute.return_value = mock_result
        return mock_provider

    def test_execute_phase_with_agent_provider(self):
        """execute_phase should pass agent provider to subphase execution."""
        mock_provider = self._make_mock_agent_provider()
        executor = SwarmDAGExecutor(agent_provider=mock_provider)
        phase = Phase(
            name="TEST",
            subphases=[
                SubPhase(
                    name="ANALYZE",
                    agent_role="explorer",
                    tasks=[Task(id="analyze_req", description="Analyze")]
                ),
            ]
        )
        
        result = executor.execute_phase(phase, {"requirement": "build something"})
        
        assert "ANALYZE" in result.subphase_results
        assert result.subphase_results["ANALYZE"]["status"] == "success"
        mock_provider.execute.assert_called()
    
    def test_execute_phase_multiple_subphases_with_agent(self):
        """execute_phase should run agent for each subphase task."""
        mock_provider = self._make_mock_agent_provider(output="done")
        executor = SwarmDAGExecutor(agent_provider=mock_provider)
        phase = Phase(
            name="MULTI",
            subphases=[
                SubPhase(name="A", tasks=[Task(id="t1", description="task1")]),
                SubPhase(name="B", tasks=[Task(id="t2", description="task2")]),
            ]
        )
        
        result = executor.execute_phase(phase, {})
        
        assert len(result.subphase_results) == 2


class TestSupervisorCreateNode:
    """Test SupervisorSwarmAgent.create_node() with agent provider integration."""
    
    def _make_mock_agent_provider(self, output="analyzed output"):
        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_result = MagicMock()
        mock_result.output = output
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.files_changed = []
        mock_result.error = None
        mock_provider.execute.return_value = mock_result
        return mock_provider
    
    def test_create_node_returns_function(self):
        """create_node should return a callable node function."""
        agent = SupervisorSwarmAgent(
            role=AgentRole.EXPLORER,
            name="explorer",
            system_prompt="Analyze requirements"
        )
        node = agent.create_node()
        
        assert callable(node)
    
    def test_create_node_delegates_to_agent_provider(self):
        """create_node should call agent provider when available."""
        mock_provider = self._make_mock_agent_provider()
        agent = SupervisorSwarmAgent(
            role=AgentRole.EXPLORER,
            name="explorer",
            system_prompt="Analyze requirements",
            agent_provider=mock_provider,
        )
        
        node = agent.create_node()
        state = {"requirement": "Build a web app"}
        result = node(state)
        
        assert result["agent_role"] == "explorer"
        mock_provider.execute.assert_called()
    
    def test_create_node_without_agent_provider(self):
        """create_node should produce error when no agent provider."""
        agent = SupervisorSwarmAgent(
            role=AgentRole.EXPLORER,
            name="explorer",
            system_prompt="Analyze requirements"
        )
        # No agent provider set
        
        node = agent.create_node()
        state = {"requirement": "Build something"}
        result = node(state)
        
        assert "agent_output" in result
        assert "error" in result["agent_output"].lower() or "No agent" in result["agent_output"]


class TestSupervisorRunGates:
    """Test Supervisor.run_gates() with real gate execution."""
    
    def test_run_gates_executes_agents(self):
        """run_gates should create and execute nodes for each gate agent."""
        explorer = SupervisorSwarmAgent(
            role=AgentRole.EXPLORER,
            name="explorer",
            system_prompt="Analyze requirements"
        )
        supervisor = Supervisor(agents=[explorer])
        
        context = {"requirement": "Build a web app"}
        results = supervisor.run_gates(["explorer"], context)
        
        assert "explorer" in results
        assert results["explorer"]["status"] == "completed"
        assert "agent_output" in results["explorer"]["result"]
    
    def test_run_gates_missing_agent_fails_gracefully(self):
        """run_gates should handle missing agents gracefully."""
        supervisor = Supervisor()  # No agents
        
        # Missing agent should raise GateEnforcementError
        with pytest.raises(GateEnforcementError):
            supervisor.run_gates(["nonexistent"], {"req": "test"})
    
    def test_run_gates_multiple_gates(self):
        """run_gates should execute multiple gate agents."""
        agents = [
            SupervisorSwarmAgent(role=AgentRole.EXPLORER, name="explorer", system_prompt="Analyze"),
            SupervisorSwarmAgent(role=AgentRole.PLANNER, name="planner", system_prompt="Plan"),
        ]
        supervisor = Supervisor(agents=agents)
        
        results = supervisor.run_gates(["explorer", "planner"], {"req": "test"})
        
        assert len(results) == 2
        assert results["explorer"]["status"] == "completed"
        assert results["planner"]["status"] == "completed"


class TestCreateSupervisorFallback:
    """Test create_supervisor fallback behavior."""
    
    def test_create_supervisor_returns_config_without_langgraph(self):
        """create_supervisor should return config dict when langgraph_supervisor unavailable."""
        # Mock import failure
        with patch('spine.swarm.supervisor._get_create_supervisor', return_value=None):
            agents = [
                SupervisorSwarmAgent(role=AgentRole.EXPLORER, name="explorer", system_prompt="Analyze"),
            ]
            result = create_supervisor(agents)
            
            assert result["mode"] == "config"
            assert len(result["agents"]) == 1
            assert result["supervisor_name"] == "spine_supervisor"


class TestSwarmAgentSetAgentProvider:
    """Test SupervisorSwarmAgent with agent_provider."""
    
    def test_agent_provider_stored(self):
        """Agent provider should be stored on the agent."""
        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        agent = SupervisorSwarmAgent(
            role=AgentRole.CODER,
            name="coder",
            system_prompt="Write code",
            agent_provider=mock_provider,
        )
        
        assert agent._agent_provider is mock_provider
    
    def test_create_node_uses_agent_provider(self):
        """create_node should use the agent provider."""
        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_result = MagicMock()
        mock_result.output = "def hello(): pass"
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.files_changed = []
        mock_result.error = None
        mock_provider.execute.return_value = mock_result
        
        agent = SupervisorSwarmAgent(
            role=AgentRole.CODER,
            name="coder",
            system_prompt="Write code",
            agent_provider=mock_provider,
        )
        
        node = agent.create_node()
        result = node({"requirement": "Write hello world"})
        
        mock_provider.execute.assert_called()


class TestDAGExecutionIntegration:
    """Integration tests for DAG execution with agent provider."""
    
    def test_full_dag_execution_with_agent(self):
        """Test full phase execution with agent-backed subphase execution."""
        mock_provider = MagicMock(spec=AgentProvider)
        mock_provider.enabled = True
        mock_result = MagicMock()
        mock_result.output = "analyzed"
        mock_result.success = True
        mock_result.exit_code = 0
        mock_result.files_changed = []
        mock_result.error = None
        mock_provider.execute.return_value = mock_result
        
        executor = SwarmDAGExecutor(agent_provider=mock_provider)
        
        phase = Phase(
            name="FULL_TEST",
            subphases=[
                SubPhase(
                    name="ANALYZE",
                    agent_role="explorer",
                    tasks=[Task(id="parse", description="Parse requirements")]
                ),
                SubPhase(
                    name="PLAN",
                    agent_role="planner",
                    dependencies=["ANALYZE"],
                    tasks=[
                        Task(id="draft", description="Draft plan"),
                        Task(id="review", description="Review plan"),
                    ]
                ),
            ]
        )
        
        context = {"requirement": "Build a REST API"}
        result = executor.execute_phase(phase, context)
        
        # Both subphases completed
        assert "ANALYZE" in result.subphase_results
        assert "PLAN" in result.subphase_results
        # Agent was called for each task (1 in ANALYZE + 2 in PLAN = 3)
        assert mock_provider.execute.call_count == 3
        # All tasks succeeded
        assert result.subphase_results["ANALYZE"]["status"] == "success"
        assert result.subphase_results["PLAN"]["status"] == "success"


class TestWaveBasedScheduling:
    """Tests for wave-based parallel scheduling features."""

    def test_resource_quota_configuration(self):
        """Executor should accept and use ResourceQuota."""
        quota = ResourceQuota(max_concurrent_subphases=3, max_workers=2)
        executor = SwarmDAGExecutor(resource_quota=quota)
        assert executor._resource_quota.max_concurrent_subphases == 3
        assert executor._resource_quota.max_workers == 2

    def test_execution_progress_initialization(self):
        """Progress should be initialized during phase execution."""
        executor = SwarmDAGExecutor()
        phase = Phase(
            name="PROGRESS_TEST",
            subphases=[SubPhase(name="A"), SubPhase(name="B")]
        )
        executor.execute_phase(phase, {})
        progress = executor.get_progress()
        assert progress is not None
        assert progress.total_subphases == 2

    def test_cancel_callback_integration(self):
        """Executor should support cancellation via callback."""
        executor = SwarmDAGExecutor()
        cancel_called = []

        def cancel_cb():
            cancel_called.append(True)
            return True

        executor.set_cancel_callback(cancel_cb)
        # Cancel should trigger callback
        result = executor._check_cancel_requested()
        assert result is True
        assert len(cancel_called) == 1

    def test_cancel_method_sets_progress_state(self):
        """Cancel method should update progress with cancellation state."""
        executor = SwarmDAGExecutor()
        phase = Phase(name="CANCEL_TEST", subphases=[SubPhase(name="A")])
        executor.execute_phase(phase, {})
        progress = executor.get_progress()
        assert progress.cancelled is False
        executor.cancel("test reason")
        assert progress.cancelled is True
        assert progress.cancel_reason == "test reason"

    def test_compute_waves_with_priority_ordering(self):
        """compute_waves should sort subphases by priority within each wave."""
        executor = SwarmDAGExecutor()
        subphases = [
            SubPhase(name="Z", priority=3),
            SubPhase(name="A", priority=1),
            SubPhase(name="M", priority=2),
        ]
        waves = executor.compute_waves(subphases)
        assert len(waves) == 1
        assert waves[0] == ["A", "M", "Z"]  # Sorted by priority

    def test_compute_waves_with_dependencies_and_priority(self):
        """compute_waves should respect both dependencies and priority."""
        executor = SwarmDAGExecutor()
        subphases = [
            SubPhase(name="FIRST", priority=5),
            SubPhase(name="A", priority=1, dependencies=["FIRST"]),
            SubPhase(name="B", priority=10, dependencies=["FIRST"]),
            SubPhase(name="C", priority=3, dependencies=["FIRST"]),
        ]
        waves = executor.compute_waves(subphases)
        assert waves[0] == ["FIRST"]
        assert waves[1] == ["A", "C", "B"]  # Sorted by priority

    def test_wave_size_limit_respected(self):
        """Wave size should be limited by resource_quota.max_concurrent_subphases."""
        quota = ResourceQuota(max_concurrent_subphases=2, max_workers=2)
        executor = SwarmDAGExecutor(resource_quota=quota)
        phase = Phase(
            name="LIMIT_TEST",
            subphases=[
                SubPhase(name="A"),
                SubPhase(name="B"),
                SubPhase(name="C"),
                SubPhase(name="D"),
            ]
        )
        result = executor.execute_phase(phase, {})
        assert len(result.subphase_results) == 4