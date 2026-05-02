# Persistence & Continuity Layer - Design Specification

## Core Concept

Five-layer persistence model that survives cold starts, runtime switches, and session loss. Incorporates swarm-tools patterns for durable task tracking and agent coordination.

---

## 1. Five-Layer Model

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: DURABLE TRUTH                                     │
│  (Survives forever, human-authored)                         │
│  .spine/spec/                                               │
│  ├── requirements.md      - What we're building            │
│  ├── architecture.md      - Design decisions                │
│  └── roadmap.md           - Milestones and phases           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Layer 2: WORKFLOW STATE                                    │
│  (Restored on resume, machine-managed)                      │
│  .spine/state/                                              │
│  ├── current_work.json    - Active work item               │
│  ├── checkpoints/         - Phase snapshots                │
│  └── execution_log.jsonl  - Audit trail                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Layer 3: JUDGMENT CACHE                                      │
│  (Evolving knowledge, compressed)                           │
│  .spine/knowledge/                                          │
│  ├── constraints.md       - Active rules/anti-patterns     │
│  ├── patterns.json        - Learned successful patterns     │
│  └── anti_patterns.json   - Failed pattern avoidance        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Layer 4: HIVE TASK TRACKING (swarm-tools pattern)          │
│  (Durable task records, git-syncable)                       │
│  .spine/state/hive/                                         │
│  ├── cells.json           - Durable task records            │
│  └── reservations.json    - Active file locks               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Layer 5: SWARM EVENTS (swarm-tools pattern)                │
│  (Agent communication log)                                  │
│  .spine/events/                                             │
│  └── swarm.log            - JSONL: messages, reservations   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
.spine/
├── config.yaml              # Main configuration
├── spine.yaml -> ../spine.yaml  # Symlink for discovery
│
├── spec/                    # Layer 1: Durable truth
│   ├── requirements.md
│   ├── architecture.md
│   ├── roadmap.md
│   └── decisions/
│       └── adr-001-use-state-machine.md
│
├── state/                   # Layer 2: Workflow state
│   ├── current_work.json    # Active work item reference
│   ├── phases/
│   │   ├── PLANNING.json    # Phase-specific state
│   │   └── EXECUTION.json
│   ├── checkpoints/
│   │   ├── plan_20240115_1030.json
│   │   └── exec_20240115_1422.json
│   ├── execution.log        # JSONL audit trail
│   │
│   └── hive/                # Layer 4: Hive task tracking
│       ├── cells.json       # Durable task records
│       └── reservations.json # Active file locks
│
├── knowledge/               # Layer 3: Judgment cache
│   ├── constraints.md
│   ├── patterns.json        # Learned patterns
│   ├── anti_patterns.json   # Anti-patterns (>60% failure)
│   └── preferences.json
│
├── events/                  # Layer 5: Swarm events (NEW)
│   └── swarm.log            # JSONL: agent messages, locks
│
├── artifacts/               # Generated outputs
│   ├── plans/
│   ├── solutions/
│   └── verification/
│
├── memory/                  # Memory provider data
│   ├── sessions.db
│   └── vectors/
│
└── logs/                    # Operational logs
    ├── spine.log
    └── provider-debug.log
```

---

## 3. Checkpoint Schema (Enhanced)

### 3.1 Phase Checkpoint

```json
{
  "checkpoint_version": "1.0",
  "checkpoint_id": "exec_20240115_1422",
  "created_at": "2024-01-15T14:22:33Z",
  "work_item_id": "feat_auth_20240115",
  
  "phase": {
    "name": "EXECUTION",
    "entered_at": "2024-01-15T10:15:00Z",
    "progress": 0.65
  },
  
  "state": {
    "current_state": "EXECUTION",
    "completed_tasks": ["task_1", "task_2"],
    "pending_tasks": ["task_3"],
    "failed_tasks": [],
    "retry_counts": {"task_2": 1}
  },
  
  "dag": {
    "execution_plan": ["task_1", "task_2", "task_3"],
    "dependencies": {
      "task_3": ["task_1", "task_2"]
    },
    "results": {
      "task_1": {
        "status": "success",
        "output_ref": "artifacts/solutions/task_1_result.json",
        "duration_ms": 15420
      },
      "task_2": {
        "status": "success", 
        "output_ref": "artifacts/solutions/task_2_result.json",
        "duration_ms": 8230
      }
    }
  },
  
  "context": {
    "requirement": "Build authentication system",
    "plan_ref": "artifacts/plans/auth_system_v1.md",
    "variables": {
      "api_endpoint": "https://api.example.com",
      "auth_type": "oauth2"
    }
  },
  
  "providers": {
    "llm": {
      "name": "primary",
      "last_request_id": "req_a1b2c3d4"
    },
    "memory": {
      "last_stored": "mem_e5f6g7h8"
    }
  },
  
  "swarm_state": {
    "active_subphases": ["BACKEND", "FRONTEND"],
    "file_reservations": {
      "worker-a": ["src/backend/**"],
      "worker-b": ["src/frontend/**"]
    },
    "pending_gates": ["reviewer", "test_engineer"],
    "hive_cell_refs": ["cell_001", "cell_002"]
  },
  
  "checksum": "sha256:9f8e7d6c5b4a3210"
}
```

---

## 4. Hive Task Tracking Schema

### 4.1 Hive Cell (Durable Task Record)

```json
{
  "cell_id": "cell_001",
  "title": "Implement JWT auth middleware",
  "type": "task",
  "status": "in_progress",
  "assignee": "coder",
  
  "phase": "EXECUTION",
  "priority": "high",
  
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:35:00Z",
  
  "dependencies": ["cell_000"],  // task IDs
  "dependents": ["cell_002"],
  
  "file_reservation": {
    "paths": ["src/auth/middleware.py"],
    "exclusive": true,
    "agent_id": "worker-a"
  },
  
  "swarm_events": ["ev_001", "ev_002"],  // links to swarm.log
  
  "result": {
    "status": "success",
    "output_ref": "artifacts/solutions/jwt_middleware.py",
    "duration_ms": 45230
  }
}
```

### 4.2 Swarm Event Log Entry

```json
{
  "event_id": "ev_001",
  "type": "message_sent",
  "timestamp": "2024-01-15T10:35:00Z",
  
  "from": "architect",
  "to": "coder",
  "subject": "TASK_ASSIGNMENT",
  
  "body": {
    "task_id": "cell_001",
    "instructions": "Implement JWT auth middleware",
    "files": ["src/auth/middleware.py"]
  },
  
  "hive_cell_ref": "cell_001"
}
```

---

## 5. Continuity Mechanisms

### 5.1 State Restoration Flow

```python
class ContinuityManager:
    def restore_session(self, work_item_id: str = None) -> Context:
        """Restore state from checkpoint with swarm state"""
        
        # 1. Check for explicit work item
        if work_item_id:
            return self._restore_work_item(work_item_id)
            
        # 2. Check for auto-resume marker
        marker = self._check_resume_marker()
        if marker:
            return self._restore_checkpoint(marker.checkpoint_ref)
            
        # 3. Find most recent work item
        latest = self._find_most_recent_work()
        if latest:
            return self._prompt_resume(latest)
            
        # 4. Fresh start
        return Context.fresh()
    
    def _restore_checkpoint(self, checkpoint_path: str) -> Context:
        """Load checkpoint and rebuild state including swarm state"""
        checkpoint = load_json(checkpoint_path)
        
        context = Context()
        context.work_item = checkpoint.get("work_item_id")
        context.phase = checkpoint["phase"]["name"]
        context.variables = checkpoint["context"]["variables"]
        context.dag = self._rebuild_dag(checkpoint["dag"])
        
        # Restore swarm state
        if "swarm_state" in checkpoint:
            context.swarm_state = checkpoint["swarm_state"]
            context.file_reservations = checkpoint["swarm_state"].get("file_reservations", {})
            
        return context
```

### 5.2 Checkpoint Triggers

```python
class CheckpointPolicy:
    def __init__(self):
        self.triggers = [
            # On phase completion
            Trigger(type="phase_complete", action=save_checkpoint),
            
            # On significant progress
            Trigger(type="progress_threshold", percent=25, action=save_checkpoint),
            Trigger(type="progress_threshold", percent=50, action=save_checkpoint),
            Trigger(type="progress_threshold", percent=75, action=save_checkpoint),
            
            # On task batch completion
            Trigger(type="task_batch_complete", batch_size=5, action=save_checkpoint),
            
            # On timeout/interrupt
            Trigger(type="signal", signal="SIGINT", action=save_checkpoint),
            
            # Periodic background saves
            Trigger(type="interval", minutes=10, action=save_checkpoint),
            
            # NEW: After swarm gate completion
            Trigger(type="swarm_gate_complete", action=save_checkpoint),
        ]
```

---

## 6. Human Handoff Protocol

### 6.1 Resume Marker

```json
{
  "resume_version": "1.0",
  "work_item_id": "feat_auth_20240115",
  "checkpoint_ref": ".spine/state/checkpoints/exec_20240115_1422.json",
  "saved_at": "2024-01-15T14:22:33Z",
  "reason": "timeout",
  "next_action": "resume_execution",
  
  "human_instructions": {
    "message": "Work paused due to timeout. Ready to resume EXECUTION phase.",
    "swarm_state": {
      "active_subphases": ["BACKEND"],
      "pending_gates": ["reviewer", "test_engineer"]
    },
    "options": [
      {"label": "Resume execution", "action": "resume"},
      {"label": "Review current state", "action": "inspect"},
      {"label": "Adjust plan", "action": "adjust"},
      {"label": "Cancel work", "action": "cancel"}
    ]
  }
}
```

---

## 7. Cross-Session Memory with Swarm Patterns

### 7.1 Pattern Learning (swarm-tools adaptation)

```python
class SwarmLearningManager:
    """Pattern learning adapted from swarm-tools"""
    
    def __init__(self):
        self.pattern_status = {
            "candidate": {"min_confirmations": 1, "required_confidence": 0.6},
            "established": {"min_confirmations": 3, "required_confidence": 0.8},
            "proven": {"min_confirmations": 10, "required_confidence": 0.9}
        }
    
    def record_completion(self, task: Task, success: bool):
        """Record task outcome for pattern learning"""
        pattern = self._extract_pattern(task)
        
        record = {
            "pattern": pattern,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "task_id": task.id,
            "work_item_id": task.work_item_id
        }
        
        self._append_to_events("completion_record", record)
        
        # Update pattern maturity
        self._update_pattern_status(pattern, success)
        
        # Check for anti-pattern (60%+ failure rate)
        if self._failure_rate(pattern) > 0.6:
            self._generate_anti_pattern(pattern)
    
    def _update_pattern_status(self, pattern: str, success: bool):
        """Mature patterns based on success rate"""
        stats = self._get_pattern_stats(pattern)
        
        if stats["success_rate"] > 0.9 and stats["count"] >= 10:
            status = "proven"
        elif stats["success_rate"] > 0.8 and stats["count"] >= 3:
            status = "established"
        elif stats["success_rate"] > 0.6:
            status = "candidate"
        else:
            status = "anti_pattern_candidate"
            
        self._update_pattern(pattern, {"status": status})

# In .spine/knowledge/patterns.json
{
  "patterns": {
    "auth-jwt-rs256": {
      "status": "proven",
      "first_seen": "2024-01-10",
      "confirmations": 12,
      "successes": 11,
      "context": "JWT authentication implementation",
      "solution": "Use RS256 with separate key rotation, 15min expiry",
      "confidence": 0.95
    }
  }
}
```

---

## 8. Git Integration Strategy

### 8.1 Commit Structure

```
feat(spine): checkpoint EXECUTION phase for feat_auth

- Completed tasks: task_1, task_2
- Pending tasks: task_3
- Swarm state: BACKEND active, reviewer pending
- Resume: spine resume feat_auth_20240115
```

### 8.2 Branch Strategy

```
main                    # Stable, completed work
├── spine/wip           # Active work in progress
└── spine/checkpoints   # Checkpoint branches for long work
```

---

## 9. Recovery Strategy

```python
class RecoveryStrategy:
    def resume(self, checkpoint: Checkpoint) -> ExecutionPlan:
        """Build optimal resume plan including swarm state"""
        # Identify in-flight tasks
        in_flight = [t for t in checkpoint.tasks if t.status == "running"]
        
        # Rebuild DAG excluding completed tasks
        remaining_dag = self.rebuild_dag(
            checkpoint.dag, 
            exclude=[t.id for t in checkpoint.completed_tasks]
        )
        
        # Restore swarm state if present
        swarm_info = checkpoint.get("swarm_state", {})
        
        return ExecutionPlan(
            tasks=remaining_dag,
            in_flight_recovery=in_flight,
            verification_needed=self.needs_verification(checkpoint),
            file_reservations=swarm_info.get("file_reservations", {}),
            pending_gates=swarm_info.get("pending_gates", [])
        )
```

---

## Summary

This persistence layer ensures:
- **Survivable state**: Any interruption can be recovered via checkpoints
- **Durable tasks**: Hive pattern tracks tasks git-syncably across sessions
- **Agent coordination**: Swarm Mail pattern logs inter-agent communication
- **Learning**: Patterns mature, anti-patterns auto-generate
- **Human-readable**: Git-based with clear commit history
- **Portable**: Works across different runtimes and sessions