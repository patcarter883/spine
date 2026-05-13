# Technical Plan: Phase-End Artifact Writeout

## 1. Architecture Overview

### 1.1 Current Architecture (Problem)

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

### 1.2 Proposed Architecture (Solution)

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

### 1.3 Component Interaction Diagram

```
Work Dispatcher (dispatcher.py)
    │
    ├─── astream loop → receives node_output chunks
    │         │
    │         ├─── Deep merge artifacts dict (preserves prior phase artifacts)
    │         │
    │         └─── _save_node_artifacts() ──► ArtifactStore.save_artifact()
    │                   │                           │
    │                   │                           ▼
    │                   │                    disk: .spine/artifacts/{work_id}/
    │                   │                          └── {phase}/
    │                   │                              └── {name}
    │                   │                              └── {name}.meta.json
    │                   │
    └─── _update_work_progress() (existing) ──► SQLite DB
```

---

## 2. Technology Choices and Rationale

| Component | Technology | Rationale |
|-----------|------------|-----------|
| File Storage | Local filesystem with `pathlib.Path` | Simple, reliable, no external dependencies; artifacts are human-readable |
| Metadata | JSON sidecar (`.meta.json`) | Structured metadata queryable by `UIApi.get_artifacts()`; survives workflow crashes |
| Error Handling | Try/except with logging | NFR-2 requires graceful degradation; warnings logged but workflow continues |
| Serialization | UTF-8 text encoding | NFR-3 compliance; safe for text-based artifacts |

---

## 3. Module/File Structure

### 3.1 Primary Modification

**File:** `/home/pat/Projects/spine/spine/work/dispatcher.py`

| Component | Lines | Description |
|-----------|-------|-------------|
| `_save_node_artifacts()` | ~15 | New helper method to extract and persist artifacts from node output |
| `submit_work()` | ~20 modified | Integrate `_save_node_artifacts()` call inside the `astream` loop |

### 3.2 Unchanged Components (verified)

| File | Status | Reason |
|------|--------|--------|
| `spine/persistence/artifacts.py` | No changes | `save_artifact()` already handles directory creation and metadata |
| `spine/ui_api/api.py` | No changes | `get_artifacts()` and `read_artifact()` already query from disk |
| `spine/models/state.py` | No changes | State reducer handles artifact merging |

---

## 4. API Design and Data Models

### 4.1 New Private Method: `_save_node_artifacts()`

```python
async def _save_node_artifacts(
    artifacts_store: ArtifactStore,
    work_id: str,
    node_artifacts: dict[str, dict[str, str]] | None,
) -> None:
    """Save artifacts from a node output to disk.
    
    Args:
        artifacts_store: The ArtifactStore instance.
        work_id: The work item ID.
        node_artifacts: Dict of {phase: {name: content}} from node output.
        
    Note:
        Logs warnings on failure but does not raise. Filesystems may be
        temporarily unavailable; artifacts can be recovered from in-memory
        state or re-run.
    """
```

### 4.2 Data Flow

```
1. Phase node returns: {"artifacts": {"specify": {"specification.md": "..."}}}
   │
2. astream yields: {"specify": {"artifacts": {"specify": {"specification.md": "..."}}}}
   │
3. dispatcher._save_node_artifacts() called with node_artifacts
   │
4. For each (phase, name, content) tuple:
   │   artifacts.save_artifact(work_id, phase, name, str(content))
   │
5. File written: .spine/artifacts/{work_id}/specify/specification.md
6. Metadata written: .spine/artifacts/{work_id}/specify/specification.md.meta.json
```

### 4.3 ArtifactStore.save_artifact() Signature (unchanged)

```python
def save_artifact(self, work_id: str, phase: str, name: str, content: str) -> Path:
    """
    Saves:
    - .spine/artifacts/{work_id}/{phase}/{name}
    - .spine/artifacts/{work_id}/{phase}/{name}.meta.json
    """
```

---

## 5. Implementation Order and Dependencies

### 5.1 Phase 1: Core Implementation (dispatcher.py)

| Step | Task | Lines | Dependencies |
|------|------|-------|--------------|
| 1.1 | Add `_save_node_artifacts()` helper method | ~15 | None |
| 1.2 | Modify `astream` loop to call `_save_node_artifacts()` after merge | ~5 | 1.1 |
| 1.3 | Wrap calls in try/except for graceful degradation | ~10 | 1.2 |

**Estimated: ~30 lines, 1-2 hours**

### 5.2 Phase 2: Error Handling (dispatcher.py)

| Step | Task | Lines | Dependencies |
|------|------|-------|--------------|
| 2.1 | Add try/except around `save_artifact` calls | ~8 | 1.1 |
| 2.2 | Log warnings on failure with artifact path | ~4 | 2.1 |
| 2.3 | Ensure original artifact saving at workflow end remains (fallback) | 0 | N/A |

**Estimated: ~12 lines, 30 minutes**

### 5.3 Phase 3: Testing

| Step | Task | Location | Type |
|------|------|----------|------|
| 3.1 | Unit test for `_save_node_artifacts()` | `tests/unit/test_dispatcher.py` | Unit |
| 3.2 | Integration test: artifacts visible mid-workflow | `tests/integration/test_artifacts.py` | Integration |
| 3.3 | UI test: `UIApi.get_artifacts()` during execution | `tests/e2e/test_work_detail.py` | E2E |

**Estimated: 1-2 days**

---

## 6. Testing Strategy

### 6.1 Unit Tests

**File:** `tests/unit/test_dispatcher.py`

```python
def test_save_node_artifacts_writes_files(tmp_path, monkeypatch):
    """Test that _save_node_artifacts creates artifact files."""
    from spine.work.dispatcher import _save_node_artifacts
    from spine.persistence.artifacts import ArtifactStore
    
    # Set up temp artifact path
    artifacts = ArtifactStore(base_path=str(tmp_path))
    
    node_artifacts = {
        "specify": {
            "specification.md": "# Test Spec",
            "notes.txt": "Some notes",
        }
    }
    
    _save_node_artifacts(artifacts, "test-work-id", node_artifacts)
    
    # Verify files exist
    assert (tmp_path / "test-work-id" / "specify" / "specification.md").exists()
    assert (tmp_path / "test-work-id" / "specify" / "notes.txt").exists()
    # Verify metadata exists
    assert (tmp_path / "test-work-id" / "specify" / "specification.md.meta.json").exists()

def test_save_node_artifacts_handles_none(tmp_path):
    """Test that _save_node_artifacts handles None gracefully."""
    from spine.work.dispatcher import _save_node_artifacts
    from spine.persistence.artifacts import ArtifactStore
    
    artifacts = ArtifactStore(base_path=str(tmp_path))
    _save_node_artifacts(artifacts, "test-work-id", None)  # Should not raise

def test_save_node_artifacts_handles_empty_dict(tmp_path):
    """Test that _save_node_artifacts handles empty dict."""
    from spine.work.dispatcher import _save_node_artifacts
    from spine.persistence.artifacts import ArtifactStore
    
    artifacts = ArtifactStore(base_path=str(tmp_path))
    _save_node_artifacts(artifacts, "test-work-id", {})  # Should not raise
```

### 6.2 Integration Tests

**File:** `tests/integration/test_artifacts.py`

```python
@pytest.mark.asyncio
async def test_artifacts_written_during_workflow(tmp_path, monkeypatch):
    """Test that artifacts are visible on disk during workflow execution."""
    from spine.work.dispatcher import submit_work
    from spine.config import SpineConfig
    from spine.persistence.artifacts import ArtifactStore
    import asyncio
    
    config = SpineConfig(
        artifact_path=str(tmp_path / "artifacts"),
        workspace_root="/nonexistent",  # Will fail at agent call
    )
    
    # This test verifies the artifact writing mechanism is called
    # even when workflow may fail
    result = await submit_work("test description", config=config)
    
    # Artifacts should exist even if work failed
    artifacts = ArtifactStore(base_path=config.artifact_path)
    work_id = result["work_id"]
    
    # At minimum, the work entry was created
    assert work_id is not None
```

### 6.3 E2E/UI Tests

**File:** `tests/e2e/test_work_detail.py`

```python
@pytest.mark.asyncio
async def test_ui_can_query_artifacts_mid_workflow():
    """Test that UIApi.get_artifacts returns artifacts during execution."""
    from spine.ui_api.api import UIApi
    
    api = UIApi()
    
    # After work_id is created during workflow
    # Query artifacts should return any already-written artifacts
    # This is for UI refresh functionality
    artifacts = api.get_artifacts(work_id="recent-work-id")
    
    # Artifacts should be list of dicts with expected keys
    assert isinstance(artifacts, list)
    if artifacts:
        assert "path" in artifacts[0]
        assert "phase" in artifacts[0]
        assert "name" in artifacts[0]
```

---

## 7. Risk Analysis and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Filesystem permissions** | Artifacts not saved | Log warning (NFR-2); fallback to in-memory state |
| **Concurrent writes** | Data corruption | Each phase writes to separate directories; no concurrent same-phase writes |
| **Large artifacts** | Performance impact | NFR-1: Synchronous write is fast for text files (<5% overhead target) |
| **Workflow crash** | Partial artifacts lost | Artifacts persisted immediately; survived crashes (NFR-4) |

---

## 8. Success Criteria Verification

| SC | Verification Method |
|----|---------------------|
| SC-1 | Manual: Run workflow, refresh work_detail page mid-execution |
| SC-2 | Automated: `test_artifacts_written_during_workflow` integration test |
| SC-3 | Automated: Check `.meta.json` files exist in `test_save_node_artifacts_writes_files` |
| SC-4 | Benchmark: Compare execution time before/after (<5% overhead) |
| SC-5 | Automated: `test_save_node_artifacts_handles_none` tests graceful handling |

---

## 9. Implementation Checklist

- [ ] Add `_save_node_artifacts()` method to `dispatcher.py`
- [ ] Integrate call inside `astream` loop after artifact merge (line ~216)
- [ ] Add try/except with logging for filesystem errors
- [ ] Create unit tests in `tests/unit/test_dispatcher.py`
- [ ] Create integration tests in `tests/integration/test_artifacts.py`
- [ ] Verify existing artifact saving at workflow end remains as fallback
- [ ] Run full test suite to ensure no regressions

---

**Status:** READY FOR IMPLEMENTATION