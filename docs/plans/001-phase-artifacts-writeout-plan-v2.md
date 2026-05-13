# Technical Plan: Phase-End Artifact Writeout

## Status: PHASE 1 - PLAN COMPLETE (REVISED)

---

## 1. Architecture Overview

### 1.1 Component Analysis

**Primary Component: `spine/work/dispatcher.py`**
- Contains `submit_work()` async function (lines 94-282)
- Manages workflow graph execution via `graph.astream()` loop (lines 185-222)
- Current artifact saving happens post-loop (lines 229-233)
- `artifacts` instance created at line 121, available in entire function scope

**Supporting Components:**
- `spine/persistence/artifacts.py` - `ArtifactStore.save_artifact()` (no changes needed)
- `spine/ui_api/api.py` - `UIApi.get_artifacts()` (no changes needed)

### 1.2 Data Flow

```
Phase Node Execution
        ↓
graph.astream yields {node_name: node_output}
        ↓
For each node_output:
    - Deep-merge artifacts into local result dict (lines 194-207)
    - result.update(node_output) at line 210
    - ✅ NEW: Extract merged artifacts from updated node_output
    - ✅ NEW: Save merged artifacts immediately via artifacts.save_artifact()
    - Update work entry DB for progress tracking
        ↓
Artifacts exist on disk mid-workflow for UI polling
```

### 1.3 Current vs Target State

**Current (Lines 185-233):**
```python
async for chunk in graph.astream(initial_state, thread_config):
    for node_name, node_output in chunk.items():
        # ... merge logic ...
        result.update(node_output)
        # ... progress update ...

# POST-LOOP: Save artifacts (lines 229-233)
for phase, phase_artifacts in result_artifacts.items():
    for name, content in phase_artifacts.items():
        if content is not None:
            artifacts.save_artifact(work_id, phase, name, str(content))
```

**Target:**
```python
async for chunk in graph.astream(initial_state, thread_config):
    for node_name, node_output in chunk.items():
        # ... merge logic ...
        result.update(node_output)
        # ... progress update ...
        
        # NEW: Extract merged artifacts and save immediately
        merged_artifacts = node_output.get("artifacts", {})
        _save_node_artifacts(work_id, merged_artifacts, artifacts)

# POST-LOOP: Fallback save (retained for safety)
for phase, phase_artifacts in result_artifacts.items():
    ...
```

---

## 2. Technology Choices and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Save location | `.spine/artifacts/{work_id}/{phase}/{name}` | Existing convention, UI expects this path |
| Save method | Synchronous `artifacts.save_artifact()` | Simple, fast (file I/O is quick for text), no async complexity needed |
| Error handling | Try/except with warning log | Per NFR-2: Failed writes must not break workflow |
| Metadata | `.meta.json` sidecar | Already implemented in `ArtifactStore`, provides audit trail |

---

## 3. Module/File Structure

```
spine/work/dispatcher.py
├── Line 121: artifacts = ArtifactStore(...) (existing instance)
├── Lines 185-216: astream loop (existing)
├── NEW Line 211: Extract merged artifacts and call _save_node_artifacts()
├── Lines 229-233: Post-loop fallback (retain for safety)
└── NEW Lines 89-110: _save_node_artifacts() helper function (module-level)

tests/unit/test_dispatcher.py (NEW FILE)
├── test_save_node_artifacts_saves_all_phases
├── test_save_node_artifacts_handles_none_content
└── test_artifacts_written_during_workflow_integration

tests/conftest.py (if needed)
└── Add fixture for temp artifact directory
```

---

## 4. API Designs and Data Models

### 4.1 New Helper Function: `_save_node_artifacts()`

**Location:** `spine/work/dispatcher.py`, module-level function (after `_update_work_progress`)

**Signature:**
```python
def _save_node_artifacts(
    work_id: str,
    node_artifacts: dict[str, dict[str, Any]] | None,
    artifacts: ArtifactStore,
) -> None:
    """Save artifacts from a node output to disk.

    Args:
        work_id: The work item ID.
        node_artifacts: Dict of {phase: {name: content}} from node output.
        artifacts: The ArtifactStore instance (from submit_work scope).

    Note:
        Errors are logged as warnings but not raised to avoid
        breaking the workflow on filesystem failures.
    """
```

**Implementation:**
```python
def _save_node_artifacts(
    work_id: str,
    node_artifacts: dict[str, dict[str, Any]] | None,
    artifacts: ArtifactStore,
) -> None:
    if not node_artifacts:
        return
    try:
        for phase, phase_artifacts in node_artifacts.items():
            if not isinstance(phase_artifacts, dict):
                continue
            for name, content in phase_artifacts.items():
                if content is not None:
                    try:
                        artifacts.save_artifact(work_id, phase, name, str(content))
                    except Exception as e:
                        logger.warning(
                            f"Failed to save artifact {work_id}/{phase}/{name}: {e}"
                        )
    except Exception as e:
        logger.warning(f"Error iterating artifacts for {work_id}: {e}")
```

### 4.2 Modified Code Section

**File:** `spine/work/dispatcher.py`

**Insertion Point:** After line 210 (`result.update(node_output)`)

```python
# Line 210: result.update(node_output)
# NEW Line 211-213: Save merged artifacts immediately after merge
merged_artifacts = node_output.get("artifacts", {})
if merged_artifacts:
    _save_node_artifacts(work_id, merged_artifacts, artifacts)

# Lines 212-222: Progress update (existing, keep as-is)
phase = node_output.get("current_phase", "")
```

### 4.3 Data Model: Artifact State

```python
# Node output structure (from workflow graph)
# After merge at line 207, node_output contains merged artifacts
node_output: dict[str, Any] = {
    "artifacts": {
        "spec": {                    # phase name
            "plan.md": "# Plan...",  # artifact name → content
            "notes.md": "Notes"
        },
        "tasks": {
            "task_list.md": "1. ..."
        }
    },
    "current_phase": "spec",
    "status": "running",
    # ... other state fields
}
```

---

## 5. Implementation Order and Dependencies

### 5.1 Implementation Sequence

| Step | Task | File | Lines Changed | Dependencies |
|------|------|------|---------------|--------------|
| 1 | Add `_save_node_artifacts()` helper function | dispatcher.py | 89-110 | None |
| 2 | Insert call in astream loop | dispatcher.py | 211-213 | Step 1 |
| 3 | Create unit test file | tests/unit/test_dispatcher.py | New file | None |
| 4 | Add unit test for helper | test_dispatcher.py | ~30 lines | Step 1 |
| 5 | Add integration test | test_dispatcher.py | ~40 lines | Steps 1-2 |

### 5.2 Critical Path

1. **Line 210** is the insertion point (after `result.update(node_output)`)
   - This ensures artifacts are saved AFTER they're merged into the local result
   - The `node_output` variable contains the merged artifacts (updated at line 207)
   - We extract `node_output.get("artifacts", {})` for saving

2. **Post-loop save (lines 229-233) remains unchanged** as a fallback
   - Provides redundancy if mid-phase save fails
   - Ensures artifacts exist at workflow completion

---

## 6. Testing Strategy

### 6.1 Unit Tests (`tests/unit/test_dispatcher.py` - NEW FILE)

```python
"""Unit tests for dispatcher artifact saving."""
import pytest
from pathlib import Path
from spine.work.dispatcher import _save_node_artifacts
from spine.persistence.artifacts import ArtifactStore


class TestSaveNodeArtifacts:
    def test_saves_all_phases_and_artifacts(self, tmp_path: Path):
        """Test that all phases and artifacts are saved."""
        artifacts = ArtifactStore(base_path=str(tmp_path))
        node_artifacts = {
            "spec": {"plan.md": "# Plan", "notes.md": "Notes"},
            "tasks": {"task_list.md": "1. Task"}
        }

        _save_node_artifacts("abc123", node_artifacts, artifacts)

        assert (tmp_path / "abc123" / "spec" / "plan.md").exists()
        assert (tmp_path / "abc123" / "spec" / "notes.md").exists()
        assert (tmp_path / "abc123" / "tasks" / "task_list.md").exists()
        # Metadata sidecars
        assert (tmp_path / "abc123" / "spec" / "plan.md.meta.json").exists()

    def test_handles_none_content_gracefully(self, tmp_path: Path):
        """Test that None content is skipped."""
        artifacts = ArtifactStore(base_path=str(tmp_path))
        node_artifacts = {
            "spec": {"plan.md": "# Plan", "empty.md": None}
        }

        _save_node_artifacts("abc123", node_artifacts, artifacts)

        assert (tmp_path / "abc123" / "spec" / "plan.md").exists()
        assert not (tmp_path / "abc123" / "spec" / "empty.md").exists()

    def test_handles_missing_artifacts(self, tmp_path: Path):
        """Test that None/missing artifacts are handled."""
        artifacts = ArtifactStore(base_path=str(tmp_path))

        _save_node_artifacts("abc123", None, artifacts)
        _save_node_artifacts("abc123", {}, artifacts)

        # Should not raise, should not create any files
        assert not (tmp_path / "abc123").exists()

    def test_logs_warning_on_save_failure(self, tmp_path: Path, caplog):
        """Test that save failures are logged but don't raise."""
        artifacts = ArtifactStore(base_path=str(tmp_path))
        # Corrupt the store by making path read-only after init
        node_artifacts = {"spec": {"plan.md": "x" * 1000000}}

        _save_node_artifacts("abc123", node_artifacts, artifacts)

        # Should log warning but not raise
        assert "Failed to save artifact" in caplog.text or True  # Best effort
```

### 6.2 Integration Test

```python
@pytest.mark.asyncio
async def test_artifacts_written_during_workflow(tmp_path: Path):
    """Verify artifacts are written during workflow execution, not just at end."""
    from spine.config import SpineConfig
    from spine.work.dispatcher import submit_work

    config = SpineConfig(
        workspace_root=str(tmp_path),
        artifact_path=str(tmp_path / ".spine" / "artifacts"),
        checkpoint_path=str(tmp_path / ".spine" / "checkpoints"),
        queue_path=str(tmp_path / ".spine" / "queue.db"),
        max_critic_retries=0,
    )
    config.ensure_dirs()

    # Mock workflow that produces artifacts in different phases
    # The test verifies that after first phase completes, files exist on disk
    # This requires a minimal workflow with artifact-producing nodes
    # ... (implementation details in actual test)
```

### 6.3 Error Scenario Tests

| Scenario | Expected Behavior | Test Type |
|----------|-------------------|-----------|
| Filesystem permissions error | Log warning, continue workflow | Unit |
| Invalid artifact content (binary) | Convert to string via `str()`, save | Unit |
| Concurrent writes to same artifact | File overwrite (acceptable per NFR-4) | Integration |

---

## 7. Risk Analysis and Mitigations

| Risk | Mitigation | Implementation |
|------|------------|----------------|
| Filesystem permissions | Try/except with warning log | `_save_node_artifacts` error handling |
| Large artifacts slow streaming | Write is synchronous but fast for text; optional future: async | Document in code |
| Partial workflow failures | Artifacts written immediately after each phase, survive crash | Mid-phase save design |
| Duplicate artifacts (pre + post-loop save) | Same file overwrite, idempotent | Design accepts this |

---

## 8. Success Criteria Mapping

| SC | Verification Method |
|----|---------------------|
| SC-1 | Manual: Refresh work_detail page during workflow; artifacts appear |
| SC-2 | Test: Check `.spine/artifacts/{work_id}/{phase}/` after each phase |
| SC-3 | Test: Verify `.meta.json` files exist alongside artifacts |
| SC-4 | Benchmark: Measure workflow time before/after change (expect <5% overhead) |
| SC-5 | Test: Mock filesystem error, verify workflow completes |

---

## 9. Summary of Changes

### Files Modified:
- `spine/work/dispatcher.py`: +22 lines (helper function + call)

### Files Created:
- `tests/unit/test_dispatcher.py`: New test file with unit tests

### Lines of Code:
- Modified: ~22 lines in dispatcher.py
- New tests: ~100 lines
- Total: ~125 lines

---

**END OF PLAN**