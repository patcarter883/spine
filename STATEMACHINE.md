# State Machine Workflow Engine - Detailed Design

## Core Concept

The state machine defines explicit phases with deterministic transitions. Each phase can contain parallel sub-phases that execute concurrently, with a DAG of parallel agent tasks within each sub-phase. Phases incorporate swarm-style decomposition with gated execution and specialized agents.

---

## 1. State Definitions

### 1.1 States

| State | Description | Success Transition | Failure Transition |
|-------|-------------|------------------|-------------------|
| `INIT` | Initial state, work item created | → PLANNING | → ERROR |
| `PLANNING` | Create detailed plan with validation gates | → EXECUTION | → BLOCKED (human review) |
| `EXECUTION` | Execute plan tasks in parallel DAG | → VERIFICATION | → REWORK |
| `VERIFICATION` | Validate deliverables against criteria | → COMPLETE / → REWORK |
| `REWORK` | Loop back for corrections | → EXECUTION | → BLOCKED |
| `BLOCKED` | Human intervention required | → PLANNING/EXECUTION | → CANCELLED |
| `COMPLETE` | Work item finished successfully | (terminal) | → REWORK |
| `ERROR` | Unrecoverable error | → INIT | (terminal) |
| `CANCELLED` | Work item cancelled | (terminal) | (terminal) |

---

## 2. Phase Specification with Parallel Sub-Phases + Swarm Decomposition

### 2.1 Phase Schema (Updated with Swarm Patterns)

```yaml
phases:
  PLANNING:
    description: "Create and validate execution plan"
    timeout: "1h"
    
    # Swarm agent roles available in this phase
    swarm_agents:
      - explorer    # Analyze requirements
      - sme         # Research domain patterns
      - planner     # Synthesize plan
      - critic      # Validate plan
    
    # NEW: Parallel substates within the phase
    subphases:
      - name: ANALYZE
        description: "Analyze requirements and context"
        weight: 0.4
        priority: 1  # Executes first
        agent_role: explorer
        dag:
          - id: parse_requirement
            capability: "parse"
            input: "{{context.requirement}}"
          - id: identify_constraints
            capability: "identify_constraints"
            depends_on: [parse_requirement]
            
      - name: RESEARCH 
        description: "Research similar solutions and best practices"
        weight: 0.6
        priority: 1
        parallel: true
        agent_role: sme
        max_parallel: 3  # Limit concurrent SMEs
        dag:
          - id: search_existing
            capability: "search"
            input: "{{context.requirement}}"
          - id: analyze_patterns
            capability: "analyze"
            depends_on: [search_existing]
            
      - name: SYNTHESIZE
        description: "Combine analysis and research into plan"
        weight: 0.3
        priority: 2  # Executes after ANALYZE + RESEARCH complete
        agent_role: planner
        dependencies: [ANALYZE, RESEARCH]
        swarm_gates:
          - critic  # Plan must pass critic review
        dag:
          - id: draft_plan
            capability: "draft"
            input: "{{subphase.outputs.ANALYZE}}, {{subphase.outputs.RESEARCH}}"
          - id: critic_review
            capability: "review"
            depends_on: [draft_plan]
            
    entry_conditions:
      - "state == INIT"
    exit_criteria:
      - "plan_document.exists == True"
      - "plan_gates.verify_completion() == True"
      - "swarm_gates.critic.approved == True"  # NEW: Swarm gate
      - "all_subphases.status == 'success'"
      
  EXECUTION:
    # Similar structure with parallel sub-phases
    swarm_agents:
      - coder          # implement tasks
      - reviewer       # validate correctness
      - test_engineer  # write/run tests
      - designer       # UI/UX specifications
      
    pre_check_batch:  # NEW: Swarm pre-check pattern
      - lint_check
      - secretscan
      - sast_scan
      parallel: true
      fail_fast: true  # Stop on any failure
      
    subphases:
      - name: BACKEND
        weight: 0.5
        parallel: true
        agent_role: coder
        resources:
          exclusive_paths: ["src/backend/**"]
        dag: [tasks for backend implementation]
        
      - name: FRONTEND
        weight: 0.5
        parallel: true
        agent_role: coder
        resources:
          exclusive_paths: ["src/frontend/**"]
        dag: [tasks for frontend implementation]
        
      - name: INTEGRATION
        weight: 0.2
        agent_role: reviewer
        dependencies: [BACKEND, FRONTEND]
        dag:
          - id: integration_test
            capability: "test_integration"
          - id: security_review
            capability: "security_review"

  VERIFICATION:
    # Sequential execution with swarm gates
    subphases:
      - name: QUALITY_GATES
        parallel: true  # Can run in parallel
        dag:
          - id: syntax_check
            capability: "syntax_verify"
          - id: lint_check
            capability: "lint_check"
          - id: security_audit
            capability: "security_audit"
            
      - name: DRIFT_VERIFICATION
        dependencies: [QUALITY_GATES]
        agent_role: critic
        dag:
          - id: plan_drift_check
            capability: "verify_drift"
          - id: placeholder_scan
            capability: "scan_placeholders"
            
    exit_criteria:
      - "swarm_gates.all_passed == True"
```

### 2.2 Sub-Phase Execution Model

```python
class SubPhase:
    """A parallelizable unit within a phase with swarm patterns"""
    name: str
    weight: float  # Relative importance for parallel scheduling
    priority: int  # Execution order (lower = earlier)
    dependencies: List[str]  # Other subphase names this depends on
    parallel: bool = True  # Can execute concurrently with siblings
    agent_role: str  # Swarm agent role for this subphase
    dag: DAG
    
    # NEW: Swarm-specific fields
    swarm_gates: List[str] = []  # Gates that must pass
    max_parallel: int = 10  # Max concurrent task executions
    resources: Dict[str, Any] = None  # File reservations, compute needs

class Phase:
    """A phase containing potentially parallel sub-phases"""
    name: str
    subphases: List[SubPhase]
    
    # NEW: Swarm gates at phase level
    swarm_agents: List[str]  # Available agent roles
    pre_check_batch: List[str] = None  # Parallel verification tools
    
    def execute(self, context: Context) -> PhaseResult:
        # Build execution graph of subphases
        subphase_graph = self.build_dependency_graph()
        
        # Execute subphases respecting dependencies
        results = {}
        for subphase in self.topological_order(subphase_graph):
            # Run DAG within subphase using swarming agents
            result = DAGExecutor().execute(subphase.dag, context)
            results[subphase.name] = result
            
        return PhaseResult(results)
```

---

## 3. Parallel Sub-Phase Decision Matrix

### 3.1 When to Use Sub-Phases

| Scenario | Use Parallel Sub-Phases | Reason |
|----------|------------------------|--------|
| Independent analysis tasks | Yes | ANALYZE + RESEARCH can run concurrently |
| Frontend + Backend dev | Yes | No shared resources, can parallelize |
| Sequential validation | No | REVIEW depends on PLAN output |
| Database migrations | No | Must be sequential for consistency |

### 3.2 Conflict Resolution

```yaml
conflict_resolution:
  shared_resource_locks:
    - "database_schema"  # Only one subphase can modify at a time
    
  output_merging:
    strategy: "merge_by_agent_role"  # Planner outputs merge, researcher outputs merge
    on_conflict: "concatenate"  # join with newlines
    
  dependency_propagation:
    cross_subphase_access: "{{subphase.ANALYZE.output}}"
```

---

## 4. DAG Execution Model (Updated)

### 4.1 Enhanced Task Definition

```python
@dataclass
class Task:
    id: str
    agent: str
    capability: str
    input: Any = None
    depends_on: List[str] = None
    
    # NEW: Sub-phase context
    subphase: str = None  # Which subphase this belongs to
    
    # Swarm fields
    swarm_role: str = None  # Agent role (coder, reviewer, sme, etc.)
    exclusive_paths: List[str] = None  # File reservations
    timeout: str = "5m"
    retry: int = 3
    
    # Gate requirements
    pre_checks: List[str] = None  # Check batch requirements
    post_gates: List[str] = None  # Validation gates after task
```

### 4.2 Sub-Phase DAG Executor

```python
class SwarmDAGExecutor:
    """Executes a phase with potential parallel sub-phases using swarm agents"""
    
    def execute_phase(self, phase: Phase, context: Context) -> PhaseResult:
        # 1. Build dependency graph of subphases
        subphase_deps = self.build_subphase_deps(phase.subphases)
        
        # 2. Execute in waves based on dependencies
        wave_results = []
        remaining = set(sp.name for sp in phase.subphases)
        
        while remaining:
            # Find subphases with no unmet dependencies
            ready = self.find_ready_subphases(subphase_deps, remaining)
            
            # NEW: Reserve resources for parallel execution
            self.reserve_resources(ready, context)
            
            # Execute ready subphases in parallel
            wave_results.extend(
                self.execute_subphase_wave(ready, context)
            )
            
            # Remove completed
            remaining -= {r.subphase_name for r in wave_results[-len(ready):]}
            
        # NEW: Run phase-level gates
        gate_results = self.run_swarm_gates(phase.swarm_gates, context)
        
        return PhaseResult.from_waves(wave_results, gates=gate_results)
    
    def execute_subphase_wave(self, subphases: List[SubPhase], context: Context):
        """Execute multiple subphases concurrently using swarm agents"""
        with ThreadPoolExecutor(max_workers=len(subphases)) as executor:
            futures = {
                executor.submit(self.execute_dag, sp.dag, context, sp.swarm_role): sp.name
                for sp in subphases
            }
            return [
                SubPhaseResult(
                    subphase_name=name,
                    result=f.result()
                )
                for f, name in futures.items()
            ]
    
    def run_swarm_gates(self, gates: List[str], context: Context):
        """Run swarm-specific gates like critic review"""
        results = {}
        for gate in gates:
            agent = self.spawn_agent(gate, context)
            results[gate] = agent.execute(context)
        return results
```

---

## 5. Example: Complete PLANNING Phase with Swarm

```yaml
PLANNING:
  swarm_agents:
    - explorer   # Analyzes requirements
    - sme        # Researches domain
    - planner    # Creates plan
    - critic     # Reviews plan
    
  subphases:
    # Wave 1: Can run in parallel (swarm decomposition)
    - name: ANALYZE
      priority: 1
      parallel: true
      agent_role: explorer
      dag:
        - id: parse_input
          capability: "parse"
          
    - name: TECH_RESEARCH
      priority: 1
      parallel: true
      agent_role: sme
      dag:
        - id: research_stack_options
          capability: "research"
          
    - name: RISK_ASSESSMENT
      priority: 1
      parallel: true
      agent_role: analyst
      dag:
        - id: identify_risks
          capability: "assess_risks"
          
    # Wave 2: Depends on Wave 1
    - name: SYNTHESIZE
      priority: 2
      dependencies: [ANALYZE, TECH_RESEARCH, RISK_ASSESSMENT]
      agent_role: planner
      swarm_gates:
        - critic  # Plan must pass critic review
      dag:
        - id: combine_findings
          capability: "synthesize"
          input: "{{subphase.outputs.*}}"
        - id: critic_review
          capability: "review"
          depends_on: [combine_findings]
          
  exit_criteria:
    - "swarm_gates.critic.approved == true"
    - "all_subphases.status == 'success'"
```

---

## 6. Progress Tracking with Sub-Phases

```json
{
  "phase": "PLANNING",
  "subphases": {
    "ANALYZE": {"status": "success", "progress": 1.0, "agent": "explorer"},
    "TECH_RESEARCH": {"status": "success", "progress": 1.0, "agent": "sme"},
    "RISK_ASSESSMENT": {"status": "running", "progress": 0.65, "agent": "analyst"},
    "SYNTHESIZE": {"status": "pending", "progress": 0.0, "agent": "planner"}
  },
  "swarm_gates": {
    "critic": {"status": "pending", "required": true}
  },
  "phase_progress": 0.78
}
```

---

## 7. Swarm Gates and Quality Checks

### 7.1 Critic Gate (Required Before EXECUTION)

```yaml
swarm_gates:
  critic:
    description: "Reviews plan before any code is written"
    agent_role: critic
    exit_on: [APPROVED, NEEDS_REVISION, REJECTED]
    on_revision:
      - create_rework_tasks  # Loop back to PLANNING
      - notify_human
```

### 7.2 Pre-Check Batch (EXECUTION Entry)

```yaml
pre_check_batch:
  description: "Parallel verification before reviewer/test engineer"
  tools:
    - lint_check
    - secretscan
    - sast_scan
  execution: parallel
  fail_fast: true
  on_failure:
    - return_to_coder_with_feedback
    - log_failure
```

### 7.3 Completion Verify (Phase Exit)

```yaml
completion_verify:
  description: "Deterministic check that plan tasks exist in source"
  verification:
    - cross_reference_planned_tasks_with_files
    - estimate_completion_percentage
    - check_for_placeholder_code
```

---

## Next Steps

1. Define swarm agent roles and capabilities in PROVIDERS.md
2. Implement Hive for durable task tracking
3. Add critic gate pattern to phase executor
4. Create pre-check batch executor
5. Build file reservation system for parallel sub-phases