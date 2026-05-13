# Feature Slices: Phase-End Artifact Writeout

## Status: READY FOR IMPLEMENTATION

---

## Wave Structure

| Wave | Slices | Description |
|------|--------|-------------|
| 1 | SLICE-1, SLICE-3 | Parallel: Add helper function + Create test file |
| 2 | SLICE-2, SLICE-4 | Parallel: Integration + Unit tests |
| 3 | SLICE-5 | Integration test for mid-workflow writeout |

---

## Slice Definitions

---

### SLICE-1: Add `_save_node_artifacts` helper function

**Name:** SLICE-1  
**Description:** Create module-level helper function to extract and save artifacts from node output. This function encapsulates the artifact saving logic with proper error handling and logging.

**Files to modify:**
- `spine/work/dispatcher.py` - Add function after `_update_work_progress` (around line 89)

**Dependencies:** None

**Acceptance Criteria:**
- [ ] Function signature: `def _save_node_artifacts(work_id: str, node_artifacts: dict[str, dict[str, Any]] | None, artifacts: ArtifactStore) -> None`
- [ ] Returns early if `node_artifacts` is None or empty
- [ ] Iterates over phases and artifact names to call `artifacts.save_artifact()`
- [ ] Converts content to string via `str(content)` before saving
- [ ] Skips artifacts where content is None
- [ ] Logs warning on individual save failures without raising
- [ ] Has docstring documenting the function purpose and parameters

**Estimated Complexity:** small (~20 lines)

---

### SLICE-2: Integrate artifact saving in astream loop

**Name:** SLICE-2  
**Description:** Insert call to `_save_node_artifacts` after `result.update(node_output)` in the streaming loop, extracting merged artifacts from the updated node output.

**Files to modify:**
- `spine/work/dispatcher.py` - Insert after line 210 (`result.update(node_output)`)

**Dependencies:** SLICE-1

**Acceptance Criteria:**
- [ ] Extracts `merged_artifacts = node_output.get("artifacts", {})`
- [ ] Checks if `merged_artifacts` has content before calling save
- [ ] Calls `_save_node_artifacts(work_id, merged_artifacts, artifacts)`
- [ ] Post-loop fallback save (lines 229-233) remains unchanged for redundancy

**Estimated Complexity:** small (~4 lines)

---

### SLICE-3: Create unit test file structure

**Name:** SLICE-3  
**Description:** Create the new test file `tests/unit/test_dispatcher.py` with proper imports, fixtures, and test class structure.

**Files to create:**
- `tests/unit/test_dispatcher.py` (NEW FILE)

**Dependencies:** None

**Acceptance Criteria:**
- [ ] File created with proper module docstring
- [ ] Imports: `pytest`, `Path`, `_save_node_artifacts`, `ArtifactStore`
- [ ] Test class `TestSaveNodeArtifacts` defined
- [ ] Uses `tmp_path` fixture for temp directory isolation

**Estimated Complexity:** small (~15 lines)

---

### SLICE-4: Add unit tests for `_save_node_artifacts`

**Name:** SLICE-4  
**Description:** Implement comprehensive unit tests for the helper function covering normal operation, edge cases, and error handling.

**Files to modify:**
- `tests/unit/test_dispatcher.py`

**Dependencies:** SLICE-1, SLICE-3

**Acceptance Criteria:**
- [ ] `test_saves_all_phases_and_artifacts` - Verifies files created in correct structure
- [ ] `test_handles_none_content_gracefully` - Verifies None content is skipped
- [ ] `test_handles_missing_artifacts` - Verifies None/{} inputs don't create files
- [ ] `test_logs_warning_on_save_failure` - Verifies warnings logged without exception

**Estimated Complexity:** medium (~60 lines)

---

### SLICE-5: Add integration test for mid-workflow artifact writeout

**Name:** SLICE-5  
**Description:** Integration test verifying artifacts are written to disk during workflow execution, not just at the end. Uses a valid temp directory configuration.

**Files to modify:**
- `tests/unit/test_dispatcher.py`

**Dependencies:** SLICE-1, SLICE-2

**Acceptance Criteria:**
- [ ] Test function `test_artifacts_written_during_workflow` defined
- [ ] Uses `tmp_path` fixture for valid workspace directories
- [ ] Creates `SpineConfig` with valid `workspace_root`, `artifact_path`, `checkpoint_path`, `queue_path`
- [ ] Calls `config.ensure_dirs()` to create required directories
- [ ] Verifies artifacts exist on disk mid-workflow (after phases complete)

**Estimated Complexity:** medium (~40 lines)

---

## Dependency Graph (DAG)

```
SLICE-1 ──┐
           ├── SLICE-2 ──┐
SLICE-3 ──┤             ├── SLICE-5
           └── SLICE-4 ──┘
```

---

## Implementation Order

1. **Wave 1 (Parallel):**
   - SLICE-1: Add helper function to dispatcher.py
   - SLICE-3: Create test file structure

2. **Wave 2 (Parallel):**
   - SLICE-2: Integrate artifact saving call
   - SLICE-4: Add unit tests for helper function

3. **Wave 3:**
   - SLICE-5: Add integration test

---

## Total Estimates

| Metric | Value |
|--------|-------|
| Total slices | 5 |
| Parallel capacity | 2 per wave |
| Code changes | ~80 lines |
| Test code | ~100 lines |