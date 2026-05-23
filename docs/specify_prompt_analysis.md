# Specify Phase System Prompt Analysis: Negative vs Positive Prompting Patterns

## Executive Summary

| Section | Negative Instructions | Positive Instructions | Ambiguous Guidance |
|---------|----------------------|----------------------|-------------------|
| Tool Surface | 2 | 0 | 0 |
| Workflow | 2 | 3 | 0 |
| Core Behaviour | 1 | 6 | 1 |
| Interpreter Environment | 5 | 2 | 0 |
| Tools | 2 | 6 | 1 |
| Workflow Context | 2 | 0 | 0 |
| Output | 0 | 3 | 0 |
| Codebase Navigation (MCP) | 0 | 2 | 0 |
| write_todos | 0 | 4 | 0 |
| Skills System | 0 | 5 | 0 |
| task (subagent spawner) | 1 | 6 | 0 |
| Interpreter | 3 | 3 | 0 |
| PTC Note | 2 | 0 | 0 |
| Project Documentation (Pitfalls) | 10+ | 5 | 0 |

---

## Detailed Section Analysis

### 1. Tool Surface (Lines 3-9)

**Negative Instructions:**
1. `You do NOT have 'ls', 'read_file', 'glob', 'grep', 'write_file', 'edit_file', or 'execute'.`
2. `Do not attempt to call them — they do not exist in your session.`

**Rewrite Suggestions:**
```
Original: "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, or `execute`. Do not attempt to call them — they do not exist in your session."

Improved: "Your available tools are: `read_work_context`, `write_specification`, and `task` (via eval). Use only these tools for all operations. Codebase exploration is performed by `researcher` subagents."
```

---

### 2. Workflow (Lines 11-56)

**Positive Instructions (Preserve):**
- "Call `read_work_context` first — always."
- "Dispatch researchers before writing."
- "Call `write_specification` exactly once, with all required fields."
- "Total turns: ~3. More than 5 turns without calling `write_specification` means something has gone wrong."

**Negative Instructions:**
1. `Do not write the spec from the description alone without codebase research (unless trivial).`
2. `Never say "I'll now do X" — just do it.` (in Core Behaviour)

**Rewrite Suggestions:**
```
Original: "Do not write the spec from the description alone without codebase research (unless trivial)."

Improved: "Write the specification only after dispatching researcher subagents to gather codebase context. Exception: trivial tasks that require no existing code knowledge."
```

---

### 3. Core Behaviour (Lines 62-72)

**Positive Instructions (Preserve):**
- "Act, don't narrate."
- "Work until the phase objective is fully met."
- "Be concise in reasoning. Reserve verbosity for the final artifact."
- "Batch independent operations."
- "Use the interpreter (eval) for orchestration."

**Negative Instructions:**
1. `Never say "I'll now do X" — just do it.`

**Ambiguous/Philosophical:**
- `Your first attempt is rarely correct — iterate.` (vague - no guidance on how/what to iterate)

**Rewrite Suggestions:**
```
Original: "Never say 'I'll now do X' — just do it."

Improved: "Execute tool calls directly without preamble statements like 'I'll now do X'. Begin your response with the tool call."

---

Original: "Your first attempt is rarely correct — iterate."

Improved: "If a subagent returns empty results or findings don't match expectations, revise the task description with more specific file paths and investigation criteria, then retry."
```

---

### 4. Interpreter Environment (Lines 74-88)

**Negative Instructions:**
1. `require() — no module system`
2. `import / export — no ES modules`
3. `fs — no filesystem access`
4. `process — no Node.js process object`
5. `window — use globalThis instead`
6. `fetch / XMLHttpRequest — no network access`

**Positive Instructions (Preserve):**
- Available: `globalThis`, `console.log`, `Promise`, `async/await`, `JSON`, `globalThis.tools`

**Rewrite Suggestions:**
```
Original format listing what DOESN'T exist:
"``require()`` — no module system
``import`` / ``export`` — no ES modules
``fs`` — no filesystem access..."

Improved format listing what IS available:
"Available APIs: ``globalThis``, ``Promise``, ``async/await``, ``JSON``, ``console.log``, ``globalThis.tools``.
Note: Module system (require/import), filesystem (fs), process, window, and network (fetch) are not available — use PTC tools instead."
```

---

### 5. Tools Section (Lines 90-108)

**Positive Instructions (Preserve):**
- "Read before write — inspect existing code before modifying it."
- "Test after write — run tests immediately after making changes."
- "Use `task` subagents for parallel work on independent slices."
- "Use `eval` to orchestrate multi-step workflows in code, not conversation."
- "Batch reads, use eval for multi-step orchestration, and produce compact artifacts."

**Negative Instructions:**
1. "Never re-read a file in the same phase."
2. "Do not use old filesystem tool names like `ls`, `glob`, `grep`, `readFile`, `writeFile` — they do not exist in PTC on orchestrator phases."

**Ambiguous/Philosophical:**
- "Context is L1 cache; conversation history is swap." (metaphor unclear to models)

**Rewrite Suggestions:**
```
Original: "Never re-read a file in the same phase. If a file is already cached, use the cached summary..."

Improved: "Check the read cache before reading files — each file read is already cached with line counts and symbol names. Use this cached metadata instead of re-reading the same file."

---

Original: "Do not use old filesystem tool names like `ls`, `glob`, `grep`..."

Improved: "Use only the tools listed in your tool surface: `read_work_context`, `write_specification`, `task`, and `eval`. Filesystem tools (ls, glob, grep, readFile, writeFile) are not available."
```

---

### 6. Workflow Context (Lines 110-115)

**Negative Instructions:**
1. `Do NOT ask follow-up questions — work with the context you are given.`
2. `Do NOT seek user approval — execute autonomously within your phase scope.`

**Rewrite Suggestions:**
```
Original: "Do NOT ask follow-up questions — work with the context you are given."

Improved: "Execute the phase objective using only the provided work description, feedback, and prior specification. Do not request additional information."

---

Original: "Do NOT seek user approval — execute autonomously within your phase scope."

Improved: "Complete the phase artifact autonomously. Your output will be reviewed by the critic phase automatically."
```

---

### 7. output Section (Lines 117-121)

**Positive Instructions (Preserve):**
- "Produce the artifact your phase requires"
- "Structure your output clearly with headers"
- "End with a clear status indicator"

---

### 8. Codebase Navigation Tools (MCP) (Lines 124-127)

**Positive Instructions (Preserve):**
- "Use these for symbol lookup, dependency analysis, and change impact assessment."
- "They are MUCH more token-efficient than reading entire files... use them FIRST"

---

### 9. write_todos Section (Lines 128-140)

**Positive Instructions (Preserve):**
- "Use this tool for complex objectives to ensure that you are tracking each necessary step"
- "It is critical that you mark todos as completed as soon as you are done with a step"
- "Don't be afraid to revise the To-Do list as you go"

---

### 10. Skills System (Lines 144-182)

**Positive Instructions (Preserve):**
- "Check if the user's task matches a skill's description"
- "Read the skill's full instructions"
- "Follow the skill's instructions"
- "Access supporting files"

---

### 11. task (subagent spawner) (Lines 185-215)

**Positive Instructions (Preserve):**
- "When a task is complex and multi-step, and can be fully delegated in isolation"
- "When a task is independent of other tasks and can run in parallel"
- "When you only care about the output of the subagent"

**Negative Instructions:**
1. "If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)"

**Rewrite Suggestions:**
```
Original: "If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)"

Improved: "For tasks requiring step-by-step review during execution, use eval with direct tool calls instead of the task tool."
```

---

### 12. Interpreter Section (Lines 216-226)

**Negative Instructions:**
1. "no filesystem"
2. "no stdlib"
3. "no network"
4. "no real clock"
5. "no `fetch`"
6. "no `require`"

**Positive Instructions (Preserve):**
- "State (variables, functions) persists across tool calls"
- "Top-level `await` works"
- "Timeout: 10.0s per call"
- "Memory: 64 MB total"

---

### 13. PTC Note (Lines 225-226)

**Negative Instructions:**
1. "Do NOT call `require()` or access `fs`"
2. "Do NOT use old filesystem tool names"

**Rewrite Suggestions:**
```
Original: "Do NOT call `require()` or access `fs`. Do NOT use old filesystem tool names..."

Improved: "Use PTC tools: `tools.read_work_context`, `tools.write_specification`, `tools.task`. All tool names are camelCase and available via globalThis.tools."
```

---

### 14. Project Documentation - Pitfalls Section (Lines 493-514)

**Negative Instructions (Philosophical "Never/Do NOT" patterns):**
1. "Never break its resolution"
2. "Never store providers in WorkflowState"
3. "Never assume data is directly in the parent state"
4. "Never assume checkpoints are shared across phases" (implied)
5. "Never assume..." (multiple similar patterns)

**Positive Instructions (Preserve):**
- Technical specifics about what TO do

---

## Summary of Key Patterns Found

### Pattern: "Do NOT have X, do not call them"
**Found:** Line 9
**Issue:** Lists absence rather than presence
**Fix:** List available tools prominently

### Pattern: "Never say X, just do Y"
**Found:** Line 66
**Issue:** Prohibits without enabling alternative
**Fix:** Directly state the desired behavior

### Pattern: "Do NOT ask, Do NOT seek"
**Found:** Lines 114-115
**Issue:** Focuses on prohibition
**Fix:** State what to do instead (work autonomously with given context)

### Pattern: "no X, no Y, no Z" Lists
**Found:** Lines 77-84, 219-222, 226
**Issue:** Defines by absence, overwhelming negative framing
**Fix:** Lead with available APIs, mention unavailable as exception

### Pattern: "Never X — Y" Warnings
**Found:** Multiple in Pitfalls section
**Issue:** Fear-based motivation
**Fix:** Explain the constraint and give the correct approach