# Specification: Phase-End Artifact Writeout

## Overview

Work task artifacts are currently persisted only after the entire workflow completes. This specification defines a change to write artifacts to disk at the end of each phase, enabling human operators to track task progress in real-time as phases execute.

## Requirements

### Functional Requirements

1. **FR-1**: Artifacts MUST be written to disk immediately after each phase completes, not just at workflow completion.
2. **FR-2**: Artifact files MUST be stored in the existing `.spine/artifacts/{work_id}/{phase}/{name}` structure.
3. **FR-3**: Metadata sidecar files (`.meta.json`) MUST be created for each artifact.
4. **FR-4**: The existing `_update_work_progress()` function MUST continue to update the work_entries database.
5. **FR-5**: Artifact writes MUST not block the workflow if the filesystem is unavailable (graceful degradation).
6. **FR-6**: The UI MUST be able to display artifacts as they become available during workflow execution.

### Non-Functional Requirements

1. **NFR-1**: Artifact writes MUST NOT significantly slow down phase execution (async where possible).
2. **NFR-2**: Failed artifact writes MUST log warnings but NOT fail the phase or workflow.
3. **NFR-3**: Artifact content MUST be serialized safely (encoding, size limits).
4. **NFR-4**: Concurrent writes to the same artifact MUST be handled safely.

## Architecture

### Current Architecture (Problem)

```
┌─────────────────────────────────────────────────────────────┐
│                    submit_work()                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Workflow Graph Execution                           │   │
│  │  (phases execute, state accumulates in memory)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  POST-WORKFLOW: Save ALL artifacts to disk          │   │
│  │  (Only if workflow succeeds)                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Proposed Architecture (Solution)

```
┌─────────────────────────────────────────────────────────────┐
│                    submit_work()                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Workflow Graph Execution (streamed)                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Phase →      │  │ Phase →      │  │ Phase →  │ │   │
│  │  │ artifacts    │  │ artifacts    │  │ artifacts│ │   │
│  │  │ save to disk │  │ save to disk │  │ save to  │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Component Changes

#### 1. `spine/work/dispatcher.py`

**Current**: Artifact saves happen only after `graph.astream()` completes (lines 229-233).

**Change**: Move artifact saving into the streaming loop:

```python
# Current code (lines 184-216):
async for chunk in graph.astream(initial_state, thread_config):
    for node_name, node_output in chunk.items():
        # ... merging logic ...
        result.update(node_output)
        # ... progress update ...

# After: Write artifacts as they become available
async for chunk in graph.astream(initial_state, thread_config):
    for node_name, node_output in chunk.items():
        # ... merging logic ...
        result.update(node_output)
        # ... progress update ...
        
        # NEW: Save artifacts immediately
        node_artifacts = node_output.get("artifacts")
        if node_artifacts:
            for phase, phase_artifacts in node_artifacts.items():
                for name, content in phase_artifacts.items():
                    if content is not None:
                        artifacts.save_artifact(work_id, phase, name, str(content))
```

#### 2. `spine/persistence/artifacts.py`

**No changes required** - the `save_artifact()` method already exists and handles:
- Directory creation
- Content writing
- Metadata sidecar creation

#### 3. `spine/ui_api/api.py`

**No changes required** - `get_artifacts()` and `read_artifact()` already query from disk.

## Interfaces

### Modified Interface: `submit_work()` in `spine/work/dispatcher.py`

```python
async def submit_work(
    description: str,
    work_type: str = "spec",
    config: SpineConfig | None = None,
) -> dict[str, Any]:
    """Submit a new work item for processing.

    Changes:
    - Artifacts are now saved incrementally during workflow execution
    - Phase completion triggers immediate artifact persistence
    """
```

### Data Flow

```
1. Phase node executes → returns `artifacts` dict in state update
2. `astream` yields chunk with node output
3. Dispatcher merges artifacts into `result` dict
4. For each artifact: `artifacts.save_artifact(work_id, phase, name, content)`
5. UI can now query and display the artifact via `UIApi.get_artifacts()`
```

## Implementation Plan

### Phase 1: Modify `dispatcher.py` Artifact Saving Logic

1. Extract artifact saving logic into a helper method `_save_node_artifacts()`
2. Call this method inside the `astream` loop for each node output
3. Add error handling to prevent filesystem failures from breaking the workflow

### Phase 2: Add Error Resilience

1. Wrap `save_artifact()` calls in try/except
2. Log warnings on failure, continue workflow
3. Consider adding retry logic for transient filesystem errors

### Phase 3: Testing

1. Unit test for artifact saving during phased execution
2. Integration test verifying UI can display artifacts mid-workflow
3. Error scenario tests (filesystem unavailable, permissions)

## Success Criteria

1. **SC-1**: Artifacts are visible in the UI after each phase completes (verified by refreshing work_detail page during workflow execution)
2. **SC-2**: Artifact files exist on disk after each phase (verified by checking `.spine/artifacts/{work_id}/{phase}/` directory)
3. **SC-3**: Metadata sidecars are created correctly (verified by checking `.meta.json` files)
4. **SC-4**: Workflow completion time is not significantly impacted (<5% overhead)
5. **SC-5**: Failed artifact writes do not cause workflow failure

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Filesystem permissions prevent writes | Log warning, continue workflow; error visible in audit log |
| Concurrent writes to same artifact | `save_artifact()` uses separate files per phase, no conflict |
| Large artifacts slow down streaming | Write is synchronous but fast; consider async file I/O future |
| Partial workflow failures lose artifacts | Artifacts written immediately, survive workflow crash |

## Estimated Effort

- Lines of code changed: ~20 (dispatcher.py)
- Testing: 1-2 days
- Total effort: 2-3 days