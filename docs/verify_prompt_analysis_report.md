# Verify Orchestrator System Prompt Analysis Report

**File Analyzed:** `/home/pat/Projects/spine/docs/verify_orch_sys_prompt.md`  
**Date:** 2026-05-23  
**Purpose:** Identify negative/ambiguous prompting patterns and micro-tooling gaps

---

## 1. Negative Prompting Instances (Do NOT / Must NOT / Do not try)

| Line | Original Text | Category |
|------|---------------|----------|
| 1 | "You do NOT inspect source code yourself" | Negative prohibition |
| 2 | "You do NOT have `edit_file` or `execute`" | Negative prohibition |
| 5 | "BY DESIGN — do not recover" | Negative guardrail |
| 6 | "Do not try alternative tools or attempt to work around the restriction" | Negative prohibition |
| 52 | "You MUST NOT call `edit_file`, `execute`, or `write_file` on anything other than `verification.md`" | Negative prohibition |
| 53 | "Do not attempt to verify slices inline" | Negative prohibition |
| 54 | "Do NOT request `general-purpose` — it does not exist" | Negative prohibition |
| 203 | "**NEVER use absolute Linux paths**" | Negative prohibition |
| 207 | "**Path traversal** (`..`, `~`) is BLOCKED" | Negative guardrail |
| 125 | "Do NOT ask follow-up questions" | Negative prohibition |
| 126 | "Do NOT seek user approval" | Negative prohibition |

**Total Negative Prompting Instances: 11**

---

## 2. Philosophical Fluff / Ambiguous Guardrails

These are statements that a 30B model would struggle to interpret correctly because they lack concrete, actionable guidance:

| Line | Original Text | Issue |
|------|---------------|-------|
| 5 | "Expected tool errors (BY DESIGN — do not recover)" | Vague - doesn't explain WHAT to do instead |
| 6 | "Dispatch a `slice-verifier` subagent instead — they can run tests" | Vague - doesn't specify HOW to structure the dispatch |
| 11 | "batch-read the codebase-map.md tasks file and implementation files" | Ambiguous - which exact files? What order? |
| 12 | "Refer to Step 1 & Step 2 guidelines preloaded in your user context pre-prompt" | Circular reference - guidelines don't exist in a separate pre-prompt |
| 20-36 | The dispatch pattern JS code | Contains errors (`tools.read_file` instead of `tools.readFile`), incomplete variable setup (`implExcerpt` not defined), and doesn't load implementation.md |
| 82 | "Reserve verbosity for the final artifact" | Subjective - what constitutes "verbosity"? |
| 83 | "Batch independent operations" | Ambiguous - doesn't specify threshold or example |
| 107 | "Use the interpreter (eval) for orchestration" | Vague - doesn't specify when to use vs when not to |
| 108-114 | "Context is L1 cache; conversation history is swap" | Metaphorical - confusing for LLMs without systems background |
| 111-114 | "Never re-read a file in the same phase" | Vague - what if context is lost? What if file changed? |
| 115-119 | Token budget guidance | Theoretical - doesn't give concrete steps |

**Total Philosophical Fluff Items: 11**

---

## 3. Concrete Rewrite Suggestions

### Instance 1 — Line 1
**Original:** "You do NOT inspect source code yourself — you dispatch one `slice-verifier` subagent per feature slice and synthesize their verdicts into a single report."

**Rewrite Suggestion:**
```
You ARE an orchestrator only. Do these three steps in order:

Step 1: Load context via `read_verification_context()` (one call loads tasks.md, codebase-map.md, implementation.md)

Step 2: For each slice in tasks.md, dispatch exactly one `slice-verifier` subagent inside `eval` using this template:
  - Embed the full slice definition
  - Include relevant excerpts from implementation.md
  - Include relevant codebase-map.md entries

Step 3: Write verification.md with VERIFIED/PASSED/FAILED status plus per-slice verdicts
```

---

### Instance 2 — Lines 5-6
**Original:** "If you attempt to call `edit_file` or `execute`, you will see a 'tool not found' error. Do not try alternative tools..."

**Rewrite Suggestion:**
```
TOOL RESTRICTION ENFORCED AT BUILD TIME:
- Your tools are: read_verification_context, write_verification_report, task, eval
- Generic filesystem tools (ls, read_file, glob, grep, write_file) are NOT available
- If you need to dispatch a subagent, use `task` inside `eval`
- If you receive "tool not found", check you're using the correct tool name from the list above
```

---

### Instance 3 — Lines 20-36 (JS Dispatch Pattern)
**Original:** Contains multiple errors (tool name, missing implExcerpt setup, incomplete logic)

**Rewrite Suggestion:**
```javascript
// Step 1: Load all context
const context = JSON.parse(await tools.read_verification_context());
const sliceFiles = context.slices; // array of slice filenames

// Step 2: Dispatch all slice-verifiers in parallel
const dispatches = sliceFiles.map(async (name) => {
  const slice = context.slice_contents[name];
  const implExcerpt = context.implementation_excerpts[name] || "";
  return tools.task({
    subagent_type: "slice-verifier",
    description: `Verify slice: ${name}\n\n` +
      `## Slice Definition\n${slice}\n\n` +
      `## Implementation Report Excerpt\n${implExcerpt}\n`,
  });
});
const results = await Promise.allSettled(dispatches);

// Step 3: Extract verdicts for synthesis
const verdicts = results.map((r, i) => ({
  slice: sliceFiles[i],
  status: r.status === "fulfilled" ? r.value.verdict : "EXCEPTION",
  error: r.status === "rejected" ? String(r.reason) : null
}));
```

---

### Instance 4 — Line 203 (Path Convention)
**Original:** "**NEVER use absolute Linux paths** like `/home/user/project/spine/ui/pages.py`..."

**Rewrite Suggestion:**
```
PATH FORMAT (REQUIRED):
- Use relative paths from workspace root: `spine/ui/pages.py`
- NOT absolute paths: `/home/pat/Projects/spine/spine/ui/pages.py`
- NOT double-nested paths that won't resolve

VALID EXAMPLES:
- `spine/agents/factory.py`
- `.spine/artifacts/da9cfc33/verify/verification.md`
```

---

### Instance 5 — Line 111-114 (Never Re-read)
**Original:** "Never re-read a file in the same phase. If a file is already cached, use the cached summary..."

**Rewrite Suggestion:**
```
FILE READING RULE:
1. On your first turn, call `read_verification_context()` ONCE
2. The return value contains ALL files you need:
   - tasks.md content
   - codebase-map.md content  
   - implementation.md content
   - slice file contents
3. Store needed values in `globalThis` (e.g., `globalThis.verificationContext = ...`)
4. Do NOT call any read tool again - use the stored values
```

---

## 4. Micro-Tooling Gap Analysis

### Current State (verify_agent.py)

```python
_VERIFY_ORCHESTRATOR_TOOLS: list[str] = [
    "ls",        # Generic filesystem
    "read_file", # Generic filesystem
    "glob",      # Generic filesystem
    "grep",      # Generic filesystem
    "write_file",  # for verification.md only
]
```

**Problems Identified:**
1. **Exposes generic filesystem tools** — allows agent to fall back to manual file exploration
2. **No structured read tool** — agent must manually discover and read multiple files
3. **write_file restriction is prompt-only** — agent could accidentally write wrong file
4. **No skip_filesystem_middleware=True** — generic tools are always present as fallback

### Best Practice Pattern (implement_tools.py)

```python
def build_implement_orchestrator_tools(workspace_root: str, work_id: str) -> list[BaseTool]:
    """Returns exactly two tools:
    - read_slice_files: loads all slice definitions + codebase map in one call
    - write_implementation_report: writes the implementation.md artifact
    """
    # No generic filesystem tools exposed
```

### Recommended Verify Tools (TODO)

Following the implement pattern, create `verify_tools.py`:

```python
# verify_tools.py
def build_verify_orchestrator_tools(workspace_root: str, work_id: str) -> list[BaseTool]:
    """Returns exactly two tools:
    - read_verification_context: loads tasks.md, codebase-map.md, implementation.md, slice files
    - write_verification_report: writes verification.md from structured verdicts
    """
    
    return [
        ReadVerificationContextTool(
            workspace_root=workspace_root,
            work_id=work_id,
        ),
        WriteVerificationReportTool(
            workspace_root=workspace_root,
            verify_dir=verify_dir,
        ),
    ]
```

### Required Changes to verify_agent.py

1. **Add skip_filesystem_middleware=True** to remove generic tool fallback
2. **Inject custom tools** via extra_tools parameter
3. **Remove `_VERIFY_ORCHESTRATOR_TOOLS` allowlist** since generic tools won't exist

---

## 5. Summary of Required Actions

| Priority | Action | File(s) to Modify |
|----------|--------|-------------------|
| HIGH | Create `verify_tools.py` with `ReadVerificationContextTool` and `WriteVerificationReportTool` | New file |
| HIGH | Modify `build_verify_agent()` to use `skip_filesystem_middleware=True` and inject custom tools | `verify_agent.py` |
| MEDIUM | Rewrite system prompt with positive, step-by-step directives | `verify_agent.py` `_build_orchestrator_prompt()` |
| MEDIUM | Fix JS dispatch pattern in system prompt (correct tool names, add missing variables) | `verify_agent.py` |
| LOW | Remove negative prompting, replace with positive "what to do" instructions | `verify_agent.py` |

---

## 6. Pattern Consistency Check

| Phase | Uses Custom Tools | skip_filesystem_middleware | Tool Count |
|-------|-------------------|---------------------------|------------|
| Specify | ✅ Yes (read_work_context, write_specification) | ✅ Yes | 2 + task + eval |
| Plan | ✅ Yes (read_prior_artifacts, search_codebase, write_plan) | ✅ Yes | 3 + task + eval |
| Tasks | ✅ Yes (read_prior_artifacts, search_codebase, write_tasks_artifacts) | ✅ Yes | 3 (+task + eval) |
| Implement | ✅ Yes (read_slice_files, write_implementation_report) | ✅ Yes | 2 + task + eval |
| **Verify** | ❌ No (uses generic ls, read_file, glob, grep, write_file) | ❌ No | 5 + task + eval |

**Gap:** Verify is the only phase NOT using the custom tools pattern, making it a micro-tooling outlier.