# AI Agent Harness - Architecture Design

**Codename: SPINE** - Deterministic workflow harness with modular provider abstraction

## Core Vision

A plugin-first agent harness where structured workflows drive model behavior through a state machine engine that coordinates parallel DAG execution within phases, incorporating swarm-style decomposition and gated verification.

---

## 1. Architecture Overview

```
+-------------------------------------------------------------+
|                    SPINE HARNESS                            |
+-------------------------------------------------------------+
||  CLI / API  Gateway  Scheduler  Controller  StateStore        |
+-------------------------------------------------------------+
||           |            |           |            |           ||
||           v            v           v            v           ||
||  +----------------+  +----------------+          +----------------+||
||  | Workflow       |  | Agent          |          | Provider       ||
||  | Engine         |  | Registry       |          | Adapters       ||
||  | StateMachine   |  | (specialized   |          | (LLM, Tools,   ||
||  |                |  |  agents)       |          |  Memory, etc)  ||
||  +----------------+  +----------------+          +----------------+||
||           |            |           |            |           ||
||           v            v           v            v           ||
||  +----------------+  +----------------+          +----------------+||
||  | Phase          |  | DAG Executor   |          | External       ||
||  | Context        |  | (parallel      |          | Services       ||
||  | Checkpoint     |  |  agents)       |          | (APIs, DBs)    ||
||  +----------------+  +----------------+          +----------------+||
+-------------------------------------------------------------+
```

---

## 2. State Machine Workflow Engine

### 2.1 States & Transitions

```
[INIT] --> [PLANNING] --> [EXECUTION] --> [VERIFICATION] --> [COMPLETE]
              |              |                |
              v              v                v
          [BLOCKED] <----+ [ERROR] <--------+ [REWORK]
              |
              v
        [HUMAN_REVIEW]
              |
              v
        (back to any state)
```

### 2.2 Phase Definition

Each phase has:
- **Entry Conditions**: What must be true to enter
- **Exit Criteria**: What must be satisfied to exit
- **DAG of Tasks**: Parallel agent execution plan
- **Swarm Gates**: Validation gates (critic, pre-check batch)
- **Checkpoint Data**: State serialized for continuity
- **Timeout Policy**: Escalation behavior

```yaml
phase: PLANNING
entry:
  - state == INIT or (state == REWORK and needs_plan_update)
exit_criteria:
  - plan_document.exists == true
  - plan_gates.verify_completion() == true
  - swarm_gates.critic.approved == true  # NEW: Swarm critic gate
dag:
  - planner_agent: create_initial_plan
  - researcher_agent: gather_context
  - reviewer_agent: validate_plan
swarm_gates:  # NEW: Swarm validation gates
  - critic: reviews_plan_before_execution
checkpoint:
  - .spine/.checkpoints/planning.json
timeout: 30m
on_timeout:
  - notify_human()
  - state = BLOCKED
```

---

## 3. Modular Provider Architecture

### 3.1 Provider Interface

```python
class Provider(ABC):
    """Base interface for all pluggable providers"""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Health check"""
        pass

class LLMProvider(Provider):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        pass
```

### 3.2 Supported Provider Types

| Provider | Interface | Examples |
|----------|-----------|----------|
| LLM | `LLMProvider` | OpenAI, Anthropic, Ollama, vLLM, SGLang |
| Memory | `MemoryProvider` | SQLite, Redis, Vector DB, File-based |
| Tools | `ToolsProvider` | MCP servers, REST APIs, CLI tools |
| Storage | `StorageProvider` | Local, S3, Git, Database |
| Notification | `NotifyProvider` | Slack, Email, Discord, SMS |

### 3.3 Plugin Registration

```python
# spine.yaml
providers:
  llm:
    - name: primary
      type: openai
      model: gpt-4.1
      api_key: ${OPENAI_API_KEY}
    - name: reasoning
      type: ollama
      model: qwen3:32b
      
  memory:
    - name: session
      type: sqlite
      path: .spine/sessions.db
    - name: vector
      type: chromadb
      path: .spine/vectors
      
plugins:
  - name: browser
    source: mcp:playwright
    capabilities: [browser_use, pdf_read]
```

---

## 4. Persistence & Continuity

### 4.1 Three-Layer Model (inspired by Workspine + swarm-tools)

1. **Durable Truth**: Specs, roadmap, design decisions in `.spine/spec/`
2. **Workflow State**: Active phase, pending tasks, checkpoints in `.spine/state/`
3. **Judgment Cache**: Constraints, anti-patterns, lessons learned in `.spine/knowledge/`
4. **Hive Tasks**: Durable task tracking (swarm-tools pattern) in `.spine/state/hive.json`
5. **Swarm Events**: Agent communication log in `.spine/events/`

### 4.2 Checkpoint Format

```json
{
  "version": "1.0",
  "phase": "EXECUTION",
  "timestamp": "2024-01-15T10:30:00Z",
  "completed_tasks": ["setup_env", "write_tests"],
  "pending_tasks": ["implement_feature"],
  "state": {
    "current_task": "implement_feature",
    "dependencies_resolved": true
  },
  "providers": {
    "llm": {"name": "primary", "last_request_id": "req_123"},
    "memory": {"last_checkpoint": "chk_456"}
  },
  "swarm_state": {
    "active_subphases": ["BACKEND", "FRONTEND"],
    "file_reservations": {"worker-a": ["src/backend/**"]},
    "pending_gates": ["reviewer", "test_engineer"]
  }
}
```

---

## 5. Multi-Agent DAG Execution with Swarm Patterns

### 5.1 Agent Types (Swarm Roster)

- **Planning Phase**: explorer, sme, planner, critic
- **Execution Phase**: coder, reviewer, test_engineer, designer
- **Verification Phase**: critic_drift_verifier, syntax_verifier

### 5.2 Example DAG for PLANNING Phase

```python
dag = DAG(
    name="planning_dag",
    tasks=[
        Task(id="analyze_requirements", agent="explorer", capability="parse"),
        Task(id="research_context", agent="sme", capability="search", depends_on=["analyze_requirements"]),
        Task(id="draft_plan", agent="planner", capability="draft", depends_on=["research_context"]),
        Task(id="critic_review", agent="critic", capability="review", depends_on=["draft_plan"]),  # Swarm gate
    ]
)

# Swarm gate configuration
swarm_gates = {
    "critic": {
        "required": True,
        "on_failure": "return_to_planning"
    }
}
```

---

## 6. Comparison with Inspiration Projects

| Feature | SPINE | Workspine | Hermes | Hive (swarm-tools) | Swarm (opencode) |
|---------|-------|-----------|--------|-------------------|------------------|
| State Machine Phases | Core | Checkpoints | Session-based | DAG-oriented | Phase Loop |
| Provider Abstraction | First-class | Limited | Model switching | Model support | Not structured |
| Multi-Agent DAG | Built-in | Sequential | Delegation | Core feature | Core feature |
| Swarm Agents | Built-in | No | No | Core | Core |
| Swarm Gates | Built-in | No | No | Via plugins | Core |
| Task Tracking | Checkpoints | Checkpoints | Session | Hive (git-sync) | .swarm/ |
| Persistence | Three-layer | Phase checkpoints | Skills/memory | Hive + events | .swarm/ |
| Learning System | Pattern cache | No | Skills | Hivemind | Lessons learned |

---

## Next Steps

1. Define concrete state machine transitions with swarm gates
2. Specify provider plugin API in detail
3. Design persistence schema with Hive integration
4. Create prototype implementation with critic gate
5. Add pre-check batch executor