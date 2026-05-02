# Swarm Decomposition for SPINE Phases

## Executive Summary

This document analyzes two swarm frameworks (swarm-tools and opencode-swarm) and proposes how to integrate their core patterns into SPINE's phase/sub-phase execution model.

---

## 1. Key Patterns from Reference Implementations

### 1.1 swarm-tools (joelhooks)

**Core Concepts:**
- **Hive**: Git-backed task tracking (`.hive/` directory) - durable, syncable via git
- **Hivemind**: Semantic memory with embeddings for learning
- **Swarm Mail**: Actor-model coordination via DurableMailbox/DurableLock/DurableDeferred
- **Learning System**: Patterns mature (candidate → established → proven), anti-patterns auto-generate at >60% failure rate
- **File Reservations**: Prevent conflicting writes from parallel workers

**Key Workflow:**
```
Coordinator decomposes → Creates Hive cells → Spawns workers with file reservations → 
Coordinates via Swarm Mail → Reviews completions → Learns from outcomes
```

### 1.2 opencode-swarm (zaxbysauce)

**Core Concepts:**
- **Hub-and-Spoke Architecture**: Architect owns decisions, specialists execute
- **Gated Pipeline**: One task at a time through QA pipeline (coder → reviewer → test_engineer → architect)
- **Phase Gates**: critic review before coding, completion-verify after phases
- **Specialized Agents**: coder, reviewer, test_engineer, critic, SME, docs, designer
- **Anti-Tempation Rules**: "decompose, don't execute" - route, don't do

**Phase Execution:**
```
PLAN → critic-gate → EXECUTE (coder → pre-check-batch → reviewer → test_engineer) → 
completion-verify → Next phase
```

### 1.3 SPINE Current State (DESIGN.md + STATEMACHINE.md)

**Core Concepts:**
- **State Machine Phases**: INIT → PLANNING → EXECUTION → VERIFICATION → COMPLETE
- **Sub-Phases**: Parallel execution within phases (ANALYZE + RESEARCH concurrently)
- **DAG within Sub-Phases**: Tasks with dependencies, parallel safe
- **Three-Layer Persistence**: Durable Truth / Workflow State / Judgment Cache

---

## 2. Integration Strategy

### 2.1 Swarm Decomposition Into Phases

Each SPINE phase should decompose into swarm-style specialized agents:

```yaml
PLANNING:
  subphases:
    - ANALYZE: analyst agent parses requirements
    - RESEARCH: researcher agents gather context in parallel
    - SYNTHESIZE: planner synthesizes findings → outputs to .spine/artifacts/plans/
  
  swarm_gates:
    - "critic" reviews plan before exit
    - "completion-verify" ensures all tasks exist in spec

EXECUTION:
  subphases:
    - BACKEND: backend-eng implements in parallel
    - FRONTEND: frontend-eng implements in parallel
    - INTEGRATION: depends on BACKEND + FRONTEND
  
  swarm_gates:
    - reviewer validates correctness
    - test_engineer writes/runs tests
    - sast_scan + secretscan run pre-check-batch
```

### 2.2 Agent Roles Mapping

| SPINE Agent | swarm-tools Equivalent | opencode-swarm Equivalent | Role |
|-------------|------------------------|---------------------------|------|
| `analyst` | coordinator (analyze) | explorer | Parse requirements, identify constraints |
| `researcher` | coordinator (gather) | SME/explorer | Gather context, find patterns |
| `planner` | coordinator (synthesize) | architect | Create structured plans |
| `coder` | worker | coder | Implement tasks |
| `reviewer` | reviewer | reviewer | Correctness, security review |
| `test_engineer` | - | test_engineer | Write/run tests |
| `critic` | reviewer | critic | Challenge plan validity |

### 2.3 Swarm Mail Pattern for SPINE

Replace in-memory coordination with persistent event log:

```python
# Current SPINE (in-memory context passing)
context.subphase_outputs["ANALYZE"] = result

# Proposed SPHRE with Swarm Mail pattern
class SwarmMail:
    """Actor-model coordination with durable state"""
    
    def send(self, to: AgentRole, subject: str, body: dict):
        """Send message between agents, persisted to .spine/events/"""
        event = {
            "type": "message_sent",
            "from": self.agent_id,
            "to": to,
            "subject": subject,
            "body": body,
            "timestamp": iso_now()
        }
        self.event_store.append(event)
    
    def reserve(self, paths: List[str], exclusive: bool = True):
        """File reservations to prevent conflicts"""
        lock = {"paths": paths, "agent": self.agent_id, "timestamp": iso_now()}
        self.lock_store.set(f"lock:{paths}", lock)
        return lock
```

### 2.4 Learning System Integration

Adapt swarm-tools learning for SPINE's Judgment Cache:

```python
# In .spine/knowledge/lessons.json
{
  "patterns": [
    {
      "id": "p-auth-jwt",
      "status": "proven",
      "context": "auth system implementation",
      "solution": "Use RS256, separate key rotation, 15min expiry",
      "confidence": 0.95,
      "first_seen": "2024-01-15",
      "last_confirmed": "2024-01-20"
    }
  ],
  "anti_patterns": [
    {
      "pattern": "implement auth before reviewing plan",
      "failure_rate": 0.73,
      "avoidance": "always run critic gate before EXECUTION"
    }
  ]
}
```

---

## 3. Phase-Specific Swarm Integration

### 3.1 PLANNING Phase with Swarm Decomposition

```yaml
phase: PLANNING
entry: "state == INIT"
exit_criteria:
  - "plan_document.exists == True"
  - "critic_gate.approved == True"
  - "all_subphases.status == 'success'"

subphases:
  - name: ANALYZE
    priority: 1
    parallel: true
    agent_role: explorer
    dag:
      - id: parse_requirement
        capability: "parse"
      - id: identify_constraints
        capability: "identify_constraints"
        depends_on: [parse_requirement]
        
  - name: RESEARCH
    priority: 1
    parallel: true
    agent_role: sme
    dag:
      - id: search_patterns
        capability: "search"
      - id: analyze_findings
        capability: "analyze"
        depends_on: [search_patterns]
        
  - name: SYNTHESIZE
    priority: 2
    dependencies: [ANALYZE, RESEARCH]
    agent_role: planner
    dag:
      - id: draft_plan
        capability: "draft"
      - id: critic_review
        capability: "review"  # Internal critic gate
```

### 3.2 EXECUTION Phase with Swarm Decomposition

```yaml
phase: EXECUTION
entry: "state == PLANNING and critic_gate.approved"
exit_criteria:
  - "all_tasks.status == 'success'"
  - "reviewer.approved == True"
  - "test_engineer.all_tests_pass == True"

subphases:
  - name: IMPLEMENTATION
    priority: 1
    parallel_safe: false  # Sequential by default for consistency
    agent_role: coder
    dag:
      - id: task_1_1
        capability: "implement"
        # Pre-check batch runs automatically
      - id: task_1_2
        capability: "implement"
        depends_on: [task_1_1]
        
  - name: VERIFICATION
    priority: 2
    dependencies: [IMPLEMENTATION]
    parallel: true
    dag:
      - id: code_review
        agent: reviewer
      - id: test_creation
        agent: test_engineer
      - id: security_scan
        agent: automated
```

### 3.3 VERIFICATION Phase with Swarm Decomposition

```yaml
phase: VERIFICATION
subphases:
  - name: QUALITY_GATES
    dag:
      - id: syntax_check
        tool: "syntax_check"
      - id: lint_check
        tool: "lint"
      - id: security_audit
        tool: "sast_scan + secretscan"
      - id: unit_tests
        agent: test_engineer
        
  - name: DRIFT_VERIFICATION
    dag:
      - id: spec_drift_check
        agent: critic_drift_verifier
      - id: placeholder_scan
        tool: "placeholder_scan"
```

---

## 4. Implementation Recommendations

### 4.1 Immediate Additions

1. **Hive-like task tracking** in `.spine/state/hive.json`
   - Durable task records that survive restarts
   - Git-syncable for cross-machine handoff

2. **Critic gate before EXECUTION** phase
   - Agent reviews plan before any code written
   - Prevents invalid plans from wasting compute

3. **Pre-check batch pattern**
   - lint:check, secretscan, sast_scan run in parallel
   - Fail fast before reviewer/test_engineer

4. **File reservation system**
   - Track which agent owns which files
   - Prevent merge conflicts in parallel execution

### 4.2 Phase 2 Enhancements

1. **Hivemind integration**
   - Store successful patterns in `.spine/knowledge/`
   - Use embeddings for similarity search on past work

2. **Swarm Mail event log**
   - `.spine/events/` directory with JSONL event stream
   - Agent-to-agent messaging for complex workflows

3. **Learning from outcomes**
   - Track pattern success rates
   - Auto-generate anti-patterns at 60%+ failure rate

### 4.3 Phase 3 Optimizations

1. **Parallel task execution** with conflict detection
2. **Automatic plan refinement** based on learned patterns
3. **Cross-session memory** via semantic similarity

---

## 5. Concrete Changes to SPINE

### 5.1 Add to Phase Schema

```yaml
# In phase definitions, add:
swarm_agents:  # Available roles for this phase
  - explorer
  - sme
  - planner
  - critic

gates:  # Pre-exit verification
  - critic  # Required before phase exit
  - completion-verify  # Ensure plan tasks exist

anti_patterns_check: true  # Validate against .spine/knowledge/anti_patterns.json
```

### 5.2 New SPINE Directory Structure

```
.spine/
├── ... (existing)
├── events/               # Swarm Mail event log
│   └── swarm.log         # JSONL: messages, reservations, checkpoints
├── hive/                 # Hive task tracking
│   ├── cells.json        # Durable task records
│   └── reservations.json # Active file locks
└── knowledge/            # Enhanced judgment cache
    ├── patterns.json     # Learned successful patterns
    └── anti_patterns.json # Failed pattern avoidance
```

### 5.3 Agent Communication Protocol

```python
# Agent sends message
swarm_mail.send(
    to="planner",
    subject="RESEARCH_FINDINGS",
    body={
        "task_id": "research_patterns",
        "findings": [...],
        "confidence": 0.85
    }
)

# Planner receives and synthesizes
inbox = swarm_mail.inbox(agent="planner")
for msg in inbox:
    if msg.subject == "RESEARCH_FINDINGS":
        plan.add_research_context(msg.body)
```

---

## Next Steps

1. Implement Hive task tracking in `.spine/state/hive.json`
2. Add critic gate to PLANNING phase transition
3. Create pre-check batch for EXECUTION phase entry
4. Build file reservation system for parallel sub-phases
5. Add pattern learning to VERIFICATION phase exit