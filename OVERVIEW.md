# SPINE - AI Agent Harness
## Complete Design Overview

**SPINE** (State-driven Parallel Intelligent NEuron-g) is a deterministic AI agent harness with:
- State machine workflow engine driving phase-based execution
- DAG-powered parallel agent coordination within phases
- Modular plugin architecture for all external services
- Repo-native persistence for cross-session continuity

---

## Quick Start

```bash
# Install spine core
pip install spine-harness

# Initialize in repo
spine init

# Start work
spine work "Build authentication system"

# Resume from checkpoint
spine resume
```

---

## Architecture at a Glance

```
User Request
      │
      ▼
┌─────────────────────────────────────────┐
│            SPINE CORE                   │
│  ┌───────────────────────────────────┐  │
│  │ State Machine Workflow Engine   │  │
│  │  INIT → PLANNING → EXECUTION      │  │
│  │  → VERIFICATION → COMPLETE        │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │ DAG Executor (parallel agents)    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
      │        │        │        │
      ▼        ▼        ▼        ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ LLM     │ │ Memory  │ │ Tools   │ │ Storage │
│ Provider│ │ Provider│ │ Provider│ │ Provider│
└─────────┘ └─────────┘ └─────────┘ └─────────┘
      │        │        │        │
      ▼        ▼        ▼        ▼
┌─────────────────────────────────────────┐
│  .spine/                                  │
│  ├── state/checkpoints/              │
│  ├── knowledge/constraints.md         │
│  └── artifacts/                        │
└─────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. State Machine + DAG Hybrid with Parallel Sub-Phases
- **Phases** provide deterministic checkpoints and human handoff points
- **Sub-Phases** enable parallel execution within phases (ANALYZE + RESEARCH concurrently)
- **DAGs** coordinate agents within each sub-phase
- State transitions are explicit and auditable

### 2. Provider-First Architecture
- Everything external is a provider (LLM, Memory, Tools, Storage, Notify)
- Plugins add new provider types dynamically
- Fallback chains for reliability

### 3. Three-Layer Persistence
- **Durable Truth**: Human-authored specs and decisions
- **Workflow State**: Machine-managed phase checkpoints
- **Judgment Cache**: Evolving learned constraints/preferences

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| `DESIGN.md` | Core architecture overview |
| `STATEMACHINE.md` | Phase definitions, DAG execution |
| `PROVIDERS.md` | Plugin interfaces, configuration |
| `PERSISTENCE.md` | Continuity layer, checkpoints |

---

## Next Implementation Steps

1. **Core State Machine** (`src/spine/core/state_machine.py`)
   - Implement phase transitions
   - Build checkpoint save/restore

2. **Provider Registry** (`src/spine/core/providers.py`)
   - Create base interfaces
   - Implement OpenAI, Ollama adapters

3. **DAG Executor** (`src/spine/core/dag.py`)
   - Parallel task execution
   - Dependency resolution

4. **Persistence Layer** (`src/spine/core/persistence.py`)
   - Three-layer model implementation
   - Git integration

---

## Final Design Decisions

| Question | Decision |
|----------|----------|
| Should phases support parallel sub-phase execution? | ✓ Yes - Added sub-phases with dependency graphs |
| How to handle conflicting provider results? | ✓ Confidence-weighted resolution strategies |
| Optimal checkpoint frequency? | ✓ Adaptive checkpointing based on work type/risk |
| Constraints management? | ✓ Hybrid - AI proposes, human approves/modifies |
| Checkpoint schema versioning? | ✓ Semantic versioning + automatic migration |

---

## Comparison: SPINE vs Inspiration

| Aspect | SPINE | Workspine | Hermes | Hive | Learnship |
|--------|-------|-----------|--------|------|-----------|
| Deterministic phases | ✓ | Phases | ✗ | DAG | Phase loop |
| Parallel agents | ✓ | Sequential | Delegates | Core | Single |
| Provider abstraction | ✓ | Limited | Model swap | Model support | Context |
| Persistence | 3-layer | Phase checkpoints | Skills/memory | Role-based | Session |
| Human handoff | Built-in | Checkpoints | Session resume | Review required | Phase loop |