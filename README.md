# SPINE

Deterministic AI agent harness with state machine workflow engine.

## Installation

```bash
pip install spine-harness
```

## Quick Start

```bash
# Initialize in repo
spine init

# Start work
spine work "Build authentication system"

# Resume from checkpoint
spine resume
```

## Architecture

SPINE uses a state machine with parallel sub-phases:

```
INIT → PLANNING → EXECUTION → VERIFICATION → COMPLETE
```

Built on LangGraph for:
- State persistence with SQLite checkpoints
- DAG-based parallel execution
- Swarm agent orchestration