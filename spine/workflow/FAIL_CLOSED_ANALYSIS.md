# Fail-Closed State Verification Gaps Analysis

## Executive Summary

Analysis of workflow nodes in SPINE revealed several patterns where state access occurs without proper prerequisite validation, potentially leading to silent failures or fail-open behavior that contradicts the intended fail-closed architecture.

---

## 1. Nodes Accessing State Without Prerequisite Validation

### 1.1 SPECIFY Phase - Missing Required Input Validation

**File:** `spine/phases/specify.py`

**Issue:** Lines 63-68 use `.get()` with empty string defaults for REQUIRED fields:
```python
description = state.get("description", "")  # REQUIRED - first phase
work_id = state.get("work_id", "unknown")    # REQUIRED
work_type = state.get("work_type", "")        # Should be validated
```

**Risk:** Empty description leads to meaningless specification. No validation that these critical fields exist.

**Code Pattern:** Same pattern in `spine/workflow/subgraphs/specify_subgraph.py` lines 36-39.

### 1.2 PLAN Phase - Missing SPECIFY Artifact Validation

**File:** `spine/phases/plan.py`

**Issue:** Lines 181-183 check for spec artifacts but don't fail if missing:
```python
has_spec = True  # Default true
if state.get("artifacts", {}).get(PhaseName.SPECIFY.value):
    has_spec = True  # But no else branch to handle missing spec!
```

**Risk:** PLAN proceeds without validated specification artifacts.

### 1.3 IMPLEMENT Phase - Missing Wave/Plan Validation

**File:** `spine/phases/implement.py`

**Issue:** Lines 185-186 access execution_waves without validating they're properly populated for wave-based workflows:
```python
execution_waves = state.get("execution_waves")
has_waves = bool(execution_waves)  # Empty list is falsy - silent fallback
```

**Risk:** Wave-based dispatch silently falls back to legacy mode without logging, potentially missing ordering constraints.

---

## 2. Missing Guard Checks That Could Cause Silent Failures

### 2.1 Artifact Gate Fail-Open Behavior (CRITICAL)

**File:** `spine/workflow/artifact_gate.py`

**Issue 1 (Lines 335-339):** Disk validation failure causes gate to proceed anyway:
```python
except Exception:
    has_disk_artifacts = True  # FAIL-OPEN!
```

**Issue 2 (Lines 363-368):** Quality check exception forces pass:
```python
except Exception as exc:
    quality_ok = True  # FAIL-OPEN!
```

**Issue 3 (Lines 394-400):** Same fail-open in plan quality check.

**Risk:** Artifact gates can pass when they should fail, allowing downstream phases to receive invalid input.

### 2.2 PLAN Validation Node Silent Skip

**File:** `spine/workflow/subgraphs/critic_subgraph.py`

**Issue (Lines 70-72):** Plan validation silently skipped if plan directory doesn't exist:
```python
if not plan_dir.is_dir():
    return {"phase_status": "passed"}  # Should be NEEDS_REVISION
```

**Risk:** Invalid plan artifacts bypass validation entirely.

### 2.3 Phase Result Mappers Lack Hard Checks

**File:** `spine/workflow/compose.py`

**Issue:** Result mappers (e.g., `_plan_result_mapper` lines 266-279) only check phase_status strings, not actual artifact presence. No validation that `plan.md` or `plan.json` actually exist.

---

## 3. Where Fail-Closed Pattern Should Be Injected

### 3.1 SPECIFY Phase Input Validation

**Required:** Add explicit validation at the start of `call_specify`:
```python
if not state.get("description"):
    return {"status": "needs_review", "feedback": [{"tier": "structural", "reason": "description is required"}]}
if not state.get("work_id"):
    return {"status": "needs_review", "feedback": [{"tier": "structural", "reason": "work_id is required"}]}
```

### 3.2 PLAN Phase Output Validation

**Required:** Convert silent skip to explicit failure:
```python
if not plan_dir.is_dir():
    return {"phase_status": ReviewStatus.NEEDS_REVISION.value, ...}
```

### 3.3 Artifact Gate Error Handling

**Required:** Replace fail-open exception handling with fail-closed:
```python
except Exception as exc:
    logger.error(f"Disk validation failed: {exc}")
    return {"status": "needs_review", "feedback": [...]}
```

---

## 4. Summary Table of Findings

| File | Line(s) | Issue | Severity |
|------|---------|-------|----------|
| artifact_gate.py | 335-339 | Disk check exception causes pass | Critical |
| artifact_gate.py | 363-368 | Quality check exception causes pass | Critical |
| artifact_gate.py | 394-400 | Plan quality exception causes pass | Critical |
| critic_subgraph.py | 70-72 | Plan dir missing returns passed | High |
| specify.py | 63-68 | No required input validation | High |
| verify_subgraph.py | 82-83 | No IMPLEMENT output validation | Medium |
| implement_subgraph.py | 185-186 | No wave validation for spec workflows | Medium |
2. Improve phase result mappers to validate artifact presence
3. Consider adding state schema validation at workflow entry points