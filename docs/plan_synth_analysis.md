# Plan Synthesis System Prompt - Negative vs Positive Prompting Analysis

## Executive Summary

| Metric | Count |
|--------|-------|
| Negative constraints (explicit) | 15 |
| Positive instructions | ~35 |
| Ratio (Negative:Positive) | ~1:2.3 |

**Key Finding**: While the prompt has more positive than negative instructions, several sections are purely negative ("only say what NOT to do") that could be converted to positive step-by-step directives.

---

## 1. Negative Constraints Found (with line numbers)

### A. Tool Access Restrictions (Line 26)
**Original:** "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, or `execute`. Do not attempt to call them."

**Location:** Line 26

### B. Path Verification Requirement (Line 86)
**Original:** "Do not invent paths like `src/main.py` without verification."

**Location:** Line 86

### C. Role Clarification (Line 95)
**Original:** "You are NOT a conversational assistant — there is no user in the loop during phase execution."

**Location:** Line 95

### D. Narration Prohibition (Line 99)
**Original:** "Never say 'I'll now do X' — just do it."

**Location:** Lines 99, 100

### E. Early Yielding Prohibition (Line 100)
**Original:** "Do not yield early with a summary of what you would do."

**Location:** Line 100

### F. Retry Logic Prohibition (Line 101)
**Original:** "Don't pound the same broken approach."

**Location:** Line 101

### G. File Re-reading Prohibition (Line 133)
**Original:** "Never re-read a file in the same phase."

**Location:** Line 133

### H. Follow-up Questions Prohibition (Line 147)
**Original:** "Do NOT ask follow-up questions — work with the context you are given."

**Location:** Line 147

### I. User Approval Prohibition (Line 148)
**Original:** "Do NOT seek user approval — execute autonomously within your phase scope."

**Location:** Line 148

### J. Todo Batching Prohibition (Line 167)
**Original:** "Do not batch up multiple steps before marking them as completed."

**Location:** Line 167

### K. Todo Tool Usage Prohibition (Line 168)
**Original:** "it is better to just complete the objective directly and NOT use this tool."

**Location:** Line 168

### L. Provider Storage Prohibition (Line 344, 499)
**Original:** "Never store providers in state for serialization compatibility"

**Location:** Lines 344, 499

### M. Cross-phase Checkpoint Assumption Prohibition (Line 503)
**Original:** "Do not assume checkpoints are shared across phases."

**Location:** Line 503

### N. Import Prohibition (Line 642)
**Original:** "Never import from `spine/workflow/` or `spine/phases/` in a UI page."

**Location:** Line 642

---

## 2. Positive Rephrasing Recommendations

### A. Tool Access (Line 26)
**Negative:** "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, or `execute`. Do not attempt to call them."

**Positive Reframe:**
```
Use these available tools for file discovery:
- MCP tools (mcp_codebase-index_*) for symbol-level lookups
- `search_codebase` for keyword/topic queries with content previews
- `task` subagent for non-trivial codebase exploration
- `eval` for orchestration of multi-step workflows
```

### B. Path Verification (Line 86)
**Negative:** "Do not invent paths like `src/main.py` without verification."

**Positive Reframe:**
```
For every file path in target_files:
1. Run mcp_codebase-index_list_files or search_codebase to verify the path exists
2. OR include the path as a new file inside a directory you have confirmed exists
3. Document the source of each path in your plan notes
```

### C. Role Clarification (Line 95)
**Negative:** "You are NOT a conversational assistant — there is no user in the loop during phase execution."

**Positive Reframe:**
```
Execute autonomously:
- Receive phase-specific context as input
- Produce structured artifact as output
- No interactive user loop exists during phase execution
```

### D. Narration Prohibition (Lines 99-100)
**Negative:** "Act, don't narrate. Never say 'I'll now do X' — just do it. Work until the phase objective is fully met. Do not yield early with a summary of what you would do."

**Positive Reframe:**
```
Execution workflow:
1. Execute tools directly without announcing intent
2. Continue until write_structured_plan has been called
3. Only provide artifact output — no progress summaries
```

### E. Retry Logic (Line 101)
**Negative:** "Don't pound the same broken approach."

**Positive Reframe:**
```
On repeated failures:
1. Stop the current approach
2. Analyze the error message for root cause
3. Modify the strategy before retrying
4. Consider dispatching a researcher subagent for investigation
```

### F. File Re-reading (Line 133)
**Negative:** "Never re-read a file in the same phase."

**Positive Reframe:**
```
Before reading a file:
1. Check if already in runtime context read_cache
2. If cached, use the cached summary (includes line counts, symbol names)
3. Only read new files not yet processed this phase
```

### G. Follow-up Questions (Line 147)
**Negative:** "Do NOT ask follow-up questions — work with the context you are given."

**Positive Reframe:**
```
Working context:
- Execute with the phase-specific context provided
- Research gaps using task subagents with spec content
- Use search_codebase to discover missing information
```

### H. User Approval (Line 148)
**Negative:** "Do NOT seek user approval — execute autonomously within your phase scope."

**Positive Reframe:**
```
Autonomous execution:
- Operate within defined phase scope
- Make technical decisions based on spec + codebase research
- Produce artifact for downstream phase consumption
```

### I. Todo Batching (Line 167)
**Negative:** "Do not batch up multiple steps before marking them as completed."

**Positive Reframe:**
```
Todo management:
1. Mark each step complete immediately upon completion
2. Update todo list as new information emerges
3. Keep incremental progress visible for tracking
```

### J. Provider Storage (Lines 344, 499)
**Negative:** "Never store providers in state for serialization compatibility"

**Positive Reframe:**
```
Provider handling:
- Access providers from config["configurable"]["providers"]
- Never include provider objects in WorkflowState
- Provider objects (LLM clients, HTTP sessions) are not serializable
```

### K. Checkpoint Isolation (Line 503)
**Negative:** "Do not assume checkpoints are shared across phases."

**Positive Reframe:**
```
Checkpoint management:
- Each phase writes to its own SQLite database
- Access per-phase state through state mappers
- Never assume data is directly in parent state
```

### L. UI Page Imports (Line 642)
**Negative:** "Never import from `spine/workflow/` or `spine/phases/` in a UI page."

**Positive Reframe:**
```
UI page imports:
- Import only from spine/ui_api/api.py
- Let UIApi handle all backend interactions
- This ensures zero duplication between CLI and UI code paths
```

---

## 3. Purely Negative Sections (Prime Candidates for Rewrite)

### Section 1: LangGraph API Restrictions (Lines 110-117)
**Current (Purely Negative):**
```
The following Node.js / browser APIs DO NOT exist and will throw errors:
- require() — no module system
- import / export — no ES modules
- fs — no filesystem access
- process — no Node.js process object
- window — use globalThis instead
- fetch / XMLHttpRequest — no network access
```

**Recommended Positive Rewrite:**
```
QuickJS Environment - Available APIs:
- Use globalThis for persistent state across turns
- Use console.log for output
- Use Promise and async/await for concurrent operations
- Use JSON for serialization
- Use globalThis.tools for PTC tool bindings (when enabled)
- Access session context via globalThis.context
```

### Section 2: Core Behaviour Rules (Lines 99-105)
**Current (Mixed Negative/Positive):**
```
- Act, don't narrate. Never say "I'll now do X" — just do it.
- Work until the phase objective is fully met. Do not yield early.
- If something fails repeatedly, stop and analyze why before retrying. Don't pound the same broken approach.
```

**Recommended Positive Rewrite:**
```
Core Execution Protocol:
1. Execute tools immediately without announcing intent
2. Complete all 3 workflow steps before outputting result
3. On repeated failures: pause, diagnose, adapt strategy
4. Use eval for orchestration when processing ≥3 files or ≥2 subagents
5. Batch independent operations in single responses
```

---

## 4. Ratio Analysis

### Negative vs Positive Instruction Count

**Negative Instructions Identified:** 15 explicit negative constraints

**Positive Instructions Identified:** ~35 positive directives (rough count from bulleted lists)

**Ratio:** 15:35 ≈ **1:2.3**

The prompt leans positive, but conversion of purely negative sections could improve clarity.

---

## 5. Recommended Changes Summary

| Priority | Change Type | Lines Affected | Effort |
|----------|-------------|----------------|--------|
| High | Convert API restrictions section | 110-117 | Low |
| High | Convert Core Behaviour section | 99-105 | Low |
| Medium | Add positive reframes as inline alternatives | Throughout | Medium |
| Low | Audit for any missed negative patterns | All | Low |

---

## 6. Implementation Notes

The most impactful change is converting the purely negative QuickJS API section (lines 110-117) to a positive list of available APIs. This follows the user's preference for "literal, step-by-step instructions" over prohibitions.