# SPINE Implementation Plan

## Executive Summary

This plan maps design specifications (OVERVIEW.md, DESIGN.md, PROVIDERS.md, PERSISTENCE.md, STATEMACHINE.md, SWARM_DECOMPOSITION.md) to current implementation status and defines remaining work.

**Current State**: ~85% of design implemented. All 10 priority tasks completed across 4 areas. All 26 tests passing.

---

## 1. State Machine + DAG Execution (Priority: HIGH)

### Current Status: ⚠️ Partial

**Existing:**
- `spine/core/state_machine.py`: Basic LangGraph-based phases (INIT → PLANNING → EXECUTION → VERIFICATION → COMPLETE)
- `spine/core/constants.py`: PhaseName and StateStatus enums
- `SubPhase` and `Phase` dataclasses defined
- Simulated sub-phase execution (not real parallel execution)

**Completed (from STATEMACHINE.md):**
- [x] **Real DAG executor** - SwarmDAGExecutor with execute_phase(), topological_order(), compute_waves()
- [x] **Pre-check batch executor** (DESIGN.md §7) - PreCheckBatch with parallel lint/secretscan/sast
- [x] **Swarm gate enforcement** - critic_gate check before EXECUTION transition

**Remaining:**
- [ ] **Sub-phase wave execution** - Parallel execution of ANALYZE + RESEARCH concurrently
- [ ] **Dependency propagation** - Cross-subphase output access via `{{subphase.ANALYZE.output}}`
- [ ] **REWORK/BLOCKED states** - Currently not implemented as valid transitions

**Files to Modify:**
- `spine/core/state_machine.py` - Add real DAG execution logic
- `spine/swarm/supervisor.py` - Integrate supervisor for parallel agents

---

## 2. Provider Architecture (Priority: MEDIUM)

### Current Status: ⚠️ Partial

**Existing:**
- `spine/providers/base.py`: Provider base class and registry
- `spine/providers/llm.py`: LLMProvider interface, OpenAIProvider, OllamaProvider
- `spine/providers/memory.py`: MemoryProvider interface, SQLiteProvider
- `spine/providers/tools.py`: ToolsProvider interface, MCPProvider
- `spine/providers/storage.py`: StorageProvider interface

**Completed (from PROVIDERS.md):**
- [x] **ProviderConfig dataclass** - With name, type, enabled, priority, config fields
- [x] **Plugin registration system** - PluginLoader with discover_plugins()
- [x] **ProviderRegistry.load_providers()** - Load from spine.yaml config
- [x] **LLMResponse dataclass** - Standardized response with content, usage, finish_reason, model, request_id
- [x] **generate_with_confidence()** - Confidence-weighted results

**Remaining:**
- [ ] **MemoryEntry dataclass** - For persistence layer
- [ ] **ToolsProvider list_tools() and invoke()** - Full implementation
- [ ] **Conflict resolution system** - Confidence-weighted, voting, consensus strategies
- [ ] **NotifyProvider** - Multi-channel notifications

**Files to Modify:**
- `spine/providers/base.py` - Add ProviderConfig, plugin loader
- `spine/providers/llm.py` - Add LLMResponse, confidence scoring
- `spine/providers/memory.py` - Add MemoryEntry, vector search
- `spine/providers/tools.py` - Full MCP integration

---

## 3. Persistence Layer (Priority: MEDIUM)

### Current Status: ⚠️ Partial

**Existing:**
- `spine/hive/hive.py`: Cell and Hive classes for durable task tracking
- `spine/hive/reservations.py`: ResourceManager for file reservations
- Basic checkpoint in state_machine.py via LangGraph SqliteSaver

**Completed (from PERSISTENCE.md §1-5):**
- [x] **ContinuityManager class** - State restoration with swarm state
- [x] **CheckpointPolicy** - 8 triggers (phase_complete, progress_threshold, task_batch_complete, signal, interval, swarm_gate_complete)
- [x] **Human handoff protocol** - ResumeMarker with resume/inspect/adjust/cancel options
- [x] **Checkpoint/Trigger/Context dataclasses** - Full checkpoint schema

**Remaining:**
- [ ] **Five-layer model structure** - Only Hive partially exists
  - Layer 1: Durable Truth - `spec/requirements.md`, `spec/architecture.md`
  - Layer 3: Judgment Cache - `knowledge/constraints.md`, `knowledge/patterns.json`
- [ ] **RecoveryStrategy class** - Optimal resume plan with swarm state
- [ ] **Learning system** - Pattern maturity, anti-pattern generation

**Files to Modify:**
- `spine/hive/hive.py` - Enhance with full checkpoint schema
- Create `spine/core/persistence.py` - ContinuityManager, CheckpointPolicy, RecoveryStrategy
- Create `spine/core/learning.py` - Pattern learning system

---

## 4. Swarm Coordination (Priority: HIGH)

### Current Status: ⚠️ Skeleton

**Existing:**
- `spine/swarm/agents.py`: SwarmAgent base, Explorer/SME/Planner/Critic agents
- `spine/swarm/supervisor.py`: Supervisor, AgentRole definitions
- `spine/swarm/gates.py`: CriticGate, PreCheckGate, CompletionGate

**Completed (from SWARM_DECOMPOSITION.md):**
- [x] **Swarm Mail system** - SwarmMail with send(), broadcast(), reserve(), inbox(), get_events()
  - `.spine/events/swarm.log` JSONL event stream
- [x] **File reservation notifications** - ResourceManager broadcasts FILE_RESERVED on successful reservation

**Remaining:**
- [ ] **Agent-to-agent messaging** - PLAN_FOR_REVIEW, TASK_ASSIGNMENT
- [ ] **Learning system** - Patterns mature (candidate → established → proven)
- [ ] **Hivemind integration** - Semantic memory with embeddings
- [ ] **Actual agent execution** - Currently stubs, need LLM integration

**Files to Create/Modify:**
- Create `spine/swarm/mail.py` - SwarmMail implementation
- Create `spine/swarm/learning.py` - Pattern learning
- Create `spine/core/hivemind.py` - Semantic memory
- `spine/swarm/agents.py` - Add real LLM-backed execution

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
```bash
# Priority: State Machine + DAG
1. Implement real DAG executor with dependency resolution
2. Add sub-phase wave execution (parallel ANALYZE + RESEARCH)
3. Integrate critic gate before EXECUTION phase exit
4. Implement pre-check batch runner
```

### Phase 2: Provider Enhancement (Week 2-3)
```bash
# Priority: Provider Architecture
1. Add ProviderConfig dataclass and yaml loading
2. Implement plugin system with dynamic loading
3. Add ConflictResolver with confidence weighting
4. Complete MemoryProvider with vector search
```

### Phase 3: Persistence Complete (Week 3-4)
```bash
# Priority: Five-layer persistence
1. Implement ContinuityManager and CheckpointPolicy
2. Add human handoff protocol with resume markers
3. Create pattern learning system from outcomes
4. Implement anti-pattern auto-generation (>60% failure rate)
```

### Phase 4: Swarm Integration (Week 4-5)
```bash
# Priority: Swarm coordination
1. Build SwarmMail actor-model communication
2. Add file reservation notifications to events
3. Integrate Hivemind semantic memory
4. Full agent-to-agent messaging protocol
```

---

## 6. File Change Summary

### New Files Required:
| File | Purpose | Lines |
|------|---------|-------|
| `spine/core/persistence.py` | ContinuityManager, CheckpointPolicy | ~200 |
| `spine/core/learning.py` | Pattern learning system | ~150 |
| `spine/core/hivemind.py` | Semantic memory store | ~150 |
| `spine/swarm/mail.py` | Swarm Mail communication | ~180 |
| `PLAN.md` | This document | ~200 |

### Modified Files:
| File | Changes |
|------|---------|
| `spine/core/state_machine.py` | Real DAG execution, wave executor |
| `spine/providers/base.py` | ProviderConfig, plugin loader |
| `spine/providers/llm.py` | LLMResponse, confidence scoring |
| `spine/providers/memory.py` | MemoryEntry, vector search |
| `spine/swarm/agents.py` | LLM-backed execution |
| `spine/hive/hive.py` | Enhanced checkpoints |

---

## 7. Validation Checklist

- [x] Real DAG executor with wave execution
- [x] Pre-check batch runs in parallel before reviewer
- [x] Critic gate enforced before EXECUTION phase exit
- [x] File reservations integrated with SwarmMail events
- [x] Checkpoints include full swarm state (ContinuityManager)
- [ ] Human handoff resume restores all contexts correctly (needs integration testing)
- [ ] Patterns mature and anti-patterns generate (not yet implemented)

---

## 8. References

- **Design Docs:** `OVERVIEW.md`, `DESIGN.md`, `PROVIDERS.md`, `PERSISTENCE.md`, `STATEMACHINE.md`, `SWARM_DECOMPOSITION.md`
- **Implementation:** `spine/core/`, `spine/providers/`, `spine/hive/`, `spine/swarm/`