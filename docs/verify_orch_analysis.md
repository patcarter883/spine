# Verify Orchestrator System Prompt Analysis

## Executive Summary

The verify orchestrator system prompt (in `/home/pat/Projects/spine/docs/verify_orch_sys_prompt.md`) contains a significant amount of runtime-varying information that could be moved out of the system prompt. The current prompt mixes:

- **Runtime-varying data** (work_id-specific paths, slice counts, file lists) that changes per work-item
- **Constant workflow rules** (tool restrictions, subagent types, dispatch patterns) that are identical across all runs
- **Phase-generic content** (interpreter environment, tool descriptions, core behavior) that belongs in the base prompt or skills

By moving runtime-varying info into injected runtime state and leveraging existing patterns, the system prompt could be reduced by approximately **60-70%** (from ~700 lines to ~200 lines).

---

## 1. Runtime-Varying Information (Per Work-Item)

These elements vary per work-item and should be moved out of the system prompt:

### Work-ID Specific Paths
| Current Location | Runtime Value |
|-----------------|---------------|
| Line 64: `.spine/artifacts/da9cfc33/verify/` | `.spine/artifacts/{work_id}/verify/` |
| Line 289: `/home/pat/Projects/spine/AGENTS.md` | Workspace-root-relative path discovery |
| Lines 76-77: Phase-specific paths (`spec_path`, `plan_path`, `tasks_path`, `impl_path`, `verify_path`) | Computed via `artifact_path(work_id, phase)` |

### Slice Inventory (Lines 75-86 in verify_agent.py)
```python
slice_count = len(slice_files)  # per-work-item
slice_inventory = f"{slice_count} slice file(s) found in `{tasks_dir}/`:..."
```

### Work Type Detection (Line 60 in verify_subgraph.py)
```python
has_spec = "spec" in work_type  # varies per work type
```

### Slice File List (Line 75 in verify_agent.py)
```python
slice_files = list_slice_files(workspace_root, work_id)  # per-work-item
```

---

## 2. Constant Information (Across All Verify Orchestrator Runs)

These elements are identical across all verify orchestrator runs:

### Tool Restrictions
- **Allowed tools**: `ls`, `read_file`, `glob`, `grep`, `write_file`
- **Forbidden tools**: `edit_file`, `execute` (by design)
- **Subagent-only tools**: `execute` available only to `slice-verifier` subagents

### Workflow Steps
1. Read context (codebase-map.md, implementation files)
2. Dispatch `slice-verifier` subagents in parallel via `eval`
3. Synthesize `verification.md` report

### Subagent Types
- **Only valid type**: `slice-verifier`
- **Subagent description** (from subagents.py lines 99-104): Verifies a single feature slice against acceptance criteria

### Output Format Rules
- First line MUST be `VERIFIED`, `PASSED`, or `FAILED`
- Must include per-slice sections with verdict and checklist
- Must include aggregated gaps and recommendations

### Dispatch Pattern (Lines 21-36 in verify_orch_sys_prompt.md)
- JavaScript code for parallel dispatch via `Promise.allSettled`
- Use of `globalThis.verifyResults` for storing results
- Subagent self-contained task descriptions

### Interpreter Environment Rules
- QuickJS sandbox, not Node.js
- PTC tool bindings
- `globalThis.context` access for session-specific context

---

## 3. Migration Recommendations

### A. Move to SubAgentMiddleware's system_prompt (Fixed at Build Time)

**Current location**: `subagents.py` lines 264-287 (`slice-verifier` prompt)

The subagent system prompt (`SUBAGENT_PROMPTS["slice-verifier"]`) already contains the bulk of verification instructions. **No changes needed** - it's already properly structured.

### B. Move to User Message Prompt in `_run_verify_agent()` (Runtime Injection)

**File**: `verify_subgraph.py` lines 66-112

The `_run_verify_agent` function already builds a substantial user prompt with:
- Artifact paths (lines 66-86)
- Where to write output (lines 97-102)
- RLM parallel verify pattern (lines 104-109)

**Additional items to inject**:
- Slice inventory should be computed at runtime and embedded
- Phase paths should use `artifact_path(work_id, phase)` function

### C. Move to Custom Tools (Read Work Context Pattern)

**Pattern reference**: `specify_tools.py` `ReadWorkContextTool`

Create a `ReadVerifyContextTool` that returns:
```python
{
    "work_id": work_id,
    "verify_dir": artifact_path(work_id, "verify"),
    "tasks_dir": artifact_path(work_id, "tasks"),
    "impl_dir": artifact_path(work_id, "implement"),
    "slice_files": list_slice_files(workspace_root, work_id),
    "has_spec": "spec" in work_type,
}
```

This eliminates the need for:
- Lines 64-69 in the system prompt (artifact paths)
- Lines 75-86 in verify_agent.py (slice inventory injection)

### D. Move to globalThis.context (Eval Context Seed)

**Current location**: Line 59 in verify_orch_sys_prompt.md mentions `globalThis.context`

The `SpineContext` already provides (context.py):
- `work_id`
- `workspace_root`
- `artifact_paths` (phase → path mapping)

**Enhancement**: Add `slice_files` to `SpineContext` so eval can access via `globalThis.context.slice_files`.

### E. Move to Memory Files (Project AGENTS.md)

**Already implemented**: The project's AGENTS.md is loaded via `resolve_memory()` for phases except TASKS/CRITIC.

**Proposal**: Create `.spine/AGENTS.md` with SPINE-specific conventions that are small and relevant to VERIFY.

### F. Move to Skills System

**Current**: `code-review` skill is loaded for VERIFY phase (skills_resolver.py line 40)

**Proposal**: Create a `verify-workflow` skill containing:
- Dispatch pattern code
- Output format rules
- Parallel verification guidance

This would:
- Reduce system prompt size
- Enable progressive disclosure (loaded only when needed)
- Allow skill evolution independent of prompt

---

## 4. Specific Recommendations for Prompt Reduction

### Phase 1: Immediate (Low Risk)

1. **Remove hardcoded artifact paths** from system prompt (lines 64-69)
   - Replace with reference to `globalThis.context` or user message

2. **Move slice inventory to user message** (verify_agent.py lines 75-86)
   - Already computed at runtime; can be injected into the user prompt

3. **Consolidate interpreter guidance**
   - Move QuickJS rules to RLM skill (already loaded for phases with interpreter)

### Phase 2: Medium Term

4. **Create `ReadVerifyContextTool`** following the `ReadWorkContextTool` pattern
   - Injects work-specific context at invocation time
   - Eliminates per-work-item path injection in system prompt

5. **Move dispatch pattern to skill**
   - Extract JavaScript dispatch code into `verify-workflow` skill
   - Reduces system prompt by ~50 lines

### Phase 3: Optimization

6. **Remove redundant workflow context**
   - Lines 121-133 about SPINE workflow are duplicated from base prompt
   - Lines 139-147 about write_todos can be skill-loaded

7. **Simplify tool documentation**
   - Filesystem tool descriptions (lines 196-220) duplicate SPINE_FILESYSTEM_PROMPT
   - Task/subagent documentation (lines 237-267) can be skill-loaded

---

## 5. Estimated Token Savings

| Component | Current Tokens | After Optimization | Savings |
|-----------|---------------|-------------------|---------|
| System Prompt (current) | ~700 lines (~65K chars) | ~200 lines (~25K chars) | ~40K chars |
| Artifact paths | ~500 chars per work-item | 0 (in user message) | 500 |
| Slice inventory | ~200 chars per work-item | 0 (in user message) | 200 |
| Dispatch code | ~500 chars | moved to skill | 500 |
| **Total estimated** | | | **~41K chars saved** |

---

## 6. Implementation Priority

| Priority | Action | File | Risk |
|----------|--------|------|------|
| 1 (HIGH) | Move artifact paths to user message/injection | verify_subgraph.py | Low |
| 2 (HIGH) | Create ReadVerifyContextTool | new file: verify_tools.py | Low |
| 3 (MEDIUM) | Extract dispatch pattern to verify-workflow skill | spine/skills/verify-workflow/ | Medium |
| 4 (LOW) | Consolidate interpreter guidance into RLM skill | spine/skills/rlm-pattern/ | Low |
| 5 (LOW) | Remove redundant workflow context lines | verify_agent.py | Low |

---

## Appendix: Current vs. Optimized Prompt Structure

### Current Structure (verify_orch_sys_prompt.md)
```
1. Role definition (3 lines)
2. Tool restrictions (4 lines) 
3. Expected tool errors (6 lines)
4. Workflow steps (33 lines)
5. Strict Rules (17 lines)
6. Eval Context Seed (4 lines)
7. Where to Write This Phase's Artifacts (8 lines)
8. Core Behaviour + Interpreter (64 lines) - DUPLICATES BASE PROMPT
9. Tools (85 lines) - DUPLICATES FILESYSTEM PROMPT
10. Batch reads (4 lines)
11. Large Tool Results (5 lines)
12. Execute Tool (4 lines) - DUPLICATES FILESYSTEM PROMPT
13. task tool (30 lines) - COULD BE SKILL
14. Interpreter (10 lines) - DUPLICATES BASE PROMPT
15. PTC Note (3 lines)
16. AGENTS.md documentation (481 lines) - PROJECT MEMORY
```

### Optimized Structure
```
1. Role definition (3 lines)
2. Workflow steps (20 lines)
3. Dispatch pattern reference (5 lines) - LINK TO SKILL
4. Output format rules (15 lines)
5. Total: ~43 lines vs 700+ lines
```