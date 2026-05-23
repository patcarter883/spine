# PLAN Phase System Prompt — Optimization Report

> Synthesized from 5 parallel analyses: negative→positive prompting, injected state, hyper-literal cleanup, micro-tooling, and structured I/O.

---

## Executive Summary

| Analysis Area | Key Finding | Token Savings |
|---|---|---|
| **Negative→Positive** | 15 negative constraints, ratio 1:2.3 (positive-leanable but improvable) | Quality ↑ |
| **Injected State** | ~2,450 tokens of redundant content duplicating runtime-injected data | **~2,450** |
| **Hyper-Literal** | 6 fluff phrases, 4 ambiguous guardrails, 1 critical duplicate block (75+ lines), 3 implicit assumptions | **~1,500** |
| **Micro-Tooling** | 13-15 tools exposed; 3 redundant search mechanisms; prompt-side tool docs bloat | **~500** |
| **Structured I/O** | SPECIFY→PLAN handoff uses fragile `split('## Architecture')` string parsing | Reliability ↑ |
| **Total** | | **~4,450** |

---

## 1. Negative → Positive Prompting

### Critical Issues

**Lines 110-117 — QuickJS API Restrictions (Purely Negative)**
```
BEFORE: "The following Node.js / browser APIs DO NOT exist and will throw errors:
- require() — no module system
- import / export — no ES modules
- fs — no filesystem access
- process — no Node.js process object
- window — use globalThis instead
- fetch / XMLHttpRequest — no network access"

AFTER: "QuickJS available APIs:
1. globalThis — persistent state across turns
2. console.log — output
3. Promise, async/await — concurrent operations
4. JSON — serialization
5. globalThis.tools — PTC tool bindings (when enabled)
6. globalThis.context — session context (work_id, plan_dir)"
```

**Lines 99-105 — Core Behaviour (Mixed Negative/Positive)**
```
BEFORE: "Act, don't narrate. Never say 'I'll now do X' — just do it.
Work until the phase objective is fully met. Do not yield early.
If something fails repeatedly, stop and analyze why before retrying.
Don't pound the same broken approach."

AFTER: "Execution protocol:
1. Call tools directly. Do not output 'I will...' statements.
2. Complete all 3 workflow steps. Output only the final artifact.
3. On 3 consecutive failures: stop, diagnose root cause, change approach.
4. Batch independent tool calls in single responses.
5. Use eval for orchestration when dispatching ≥2 subagents."
```

**Line 26 — Tool Restrictions (Purely Negative)**
```
BEFORE: "You do NOT have ls, read_file, glob, grep, write_file, edit_file, or execute.
Do not attempt to call them."

AFTER: "Available tools: read_prior_artifacts, task (via eval), eval,
MCP codebase-index tools, search_codebase, write_structured_plan.
Call only these."
```

### All 15 Negative Constraints → Positive Reframes

| # | Location | Negative | Positive Reframe |
|---|----------|----------|-----------------|
| 1 | L26 | "Do NOT have X, do not attempt" | "Available tools: [list]. Call only these." |
| 2 | L86 | "Do not invent paths" | "Every target_file must come from: (1) MCP output, (2) search_codebase output, or (3) new file in verified existing dir" |
| 3 | L95 | "You are NOT a conversational assistant" | "Execute autonomously. Receive context as input, produce artifact as output." |
| 4 | L99 | "Never say I'll now do X" | "Call tools directly without announcing intent." |
| 5 | L100 | "Do not yield early" | "Complete all steps. Output only the final artifact." |
| 6 | L101 | "Don't pound the same approach" | "On 3 failures: diagnose, change approach, retry." |
| 7 | L133 | "Never re-read a file" | "Before reading: check read_cache. Use cached summary if available." |
| 8 | L147 | "Do NOT ask follow-up questions" | "Answer using provided context. Research gaps via task subagents." |
| 9 | L148 | "Do NOT seek user approval" | "Execute all steps autonomously within phase scope." |
| 10 | L167 | "Do not batch todos" | "Mark each step complete immediately upon finishing." |
| 11 | L344,499 | "Never store providers in state" | "Access providers from config[configurable][providers]." |
| 12 | L503 | "Do not assume shared checkpoints" | "Each phase has its own SQLite DB. Access via state mappers." |
| 13 | L642 | "Never import from workflow/phases" | "Import only from spine/ui_api/api.py" |

---

## 2. Injected State — Eliminate Redundancy

### Critical: Double Injection of SPINE_BASE_PROMPT

The `plan_synth_sys_prompt.md` file contains an embedded `<project_documentation>` block (lines 226-660) with the full AGENTS.md. **Separately**, `profile.py` injects `SPINE_BASE_PROMPT` (~550 tokens) via the HarnessProfile CUSTOM slot. The factory then concatenates: `system_prompt + "\n\n" + base_prompt`. This means AGENTS.md content appears **twice** in the context.

**Fix**: Remove lines 226-660 from `plan_synth_sys_prompt.md`. The AGENTS.md content is already injected by `SpineProjectMemoryMiddleware` in the factory.

**Savings: ~1,000 tokens**

### Tool Descriptions Already Auto-Injected

The prompt hand-documents every tool (lines 3-24, ~800 tokens). Deep Agents runtime already injects tool schemas. This is pure duplication.

**Fix**: Replace the hand-written tool list with a single line:
```
Tool usage patterns: read_prior_artifacts FIRST, dispatch researchers via eval+task,
MCP tools for narrow lookups, write_structured_plan LAST.
```
Tool schemas provide the detailed signatures.

**Savings: ~700 tokens**

### `plan_agent.py` `_build_plan_prompt()` Duplicates `plan_synth_sys_prompt.md`

The `_build_plan_prompt()` function in `plan_agent.py` contains ~2,130 tokens that are verbatim copies of the docs file. This creates a maintenance divergence risk.

**Fix**: Replace `_build_plan_prompt()` with a minimal phase-framing prompt (~150 tokens). Move all workflow instructions to the docs file (single source of truth) or rely on injected tool schemas.

**Savings: ~2,000 tokens**

### MCP Call Patterns in Prompt

Lines 9-18 document MCP tool call syntax (`Call: {"name": "symbol_name"}`). This is already in the MCP tool schemas.

**Fix**: Remove. Replace with: "MCP tools: call with native kwargs from schema."

**Savings: ~200 tokens**

### Token Budget Impact

| Component | Current | After | Savings |
|-----------|---------|-------|---------|
| Phase system_prompt | ~800 | ~150 | 650 |
| Embedded AGENTS.md | ~1,000 | 0 | 1,000 |
| Duplicate tool descriptions | ~500 | 0 | 500 |
| MCP call patterns | ~200 | 0 | 200 |
| **Total** | **~2,500** | **~150** | **~2,350** |

---

## 3. Hyper-Literal Prompt Cleanup

### Philosophical Fluff → Concrete Directives

| Before (Fluff) | After (Literal) |
|---|---|
| "Be concise in reasoning. Reserve verbosity for the final artifact." | "Reasoning: ≤3 sentences. Full detail: only in write_structured_plan output." |
| "Your first attempt is rarely correct — iterate." | "Revise until correct. Each attempt must improve on the previous." |
| "If something fails repeatedly, stop and analyze why before retrying." | "On 3 consecutive failures: (1) stop, (2) read error message, (3) use different tool or approach." |
| "Work until the phase objective is fully met." | "Complete all 3 steps. Call write_structured_plan. Then stop." |

### Ambiguous Guardrails → Explicit Criteria

| Before (Ambiguous) | After (Explicit) |
|---|---|
| "Aim for 2–8 slices per plan." | "Create 2–8 feature_slices. Count before calling write_structured_plan." |
| "Each slice should be completable in a single implementation turn (~30 files, ~2000 lines)." | "Each slice: ≤30 files changed, ≤2000 lines. If larger, split into two slices." |
| "complexity: One of 'small', 'medium', or 'large'." | "complexity criteria: small=≤1 file ≤50 lines, medium=2-3 files ≤200 lines, large=4+ files or >200 lines." |
| "Dependencies must form a DAG (no cycles)." | "Verify DAG: no slice depends on itself. For each dependency chain, no slice appears twice." |

### Implicit Knowledge → Explicit Instructions

1. **DAG validation**: Add explicit cycle-checking instruction
2. **Feature slice definition**: "Each slice must contain all information needed for a subagent to implement without reading the spec."
3. **Complexity levels**: Add explicit criteria (see above)

### Over-Specific Example → Minimal Template

Lines 44-59 contain an ~80-line JS dispatch pattern with hardcoded section names. This is both brittle and bloated.

**Before (80 lines)**: Full JS code with `spec.split('## Architecture')`, `spec.split('## Interfaces')`, etc.

**After (15 lines)**:
```js
// Extract spec sections for researcher dispatch:
const spec = globalThis.ctx.artifacts.specify['specification.md'];
// Identify 2-4 spec areas needing codebase mapping
const results = await Promise.allSettled([
  tools.task({subagent_type: "researcher",
    description: `SPEC: [paste relevant section]\nTASK: Find exact file paths, existing patterns, conventions to follow.`}),
]);
globalThis.reports = results.map(r => r.value);
```

**Savings: ~750 tokens**

### Contradictory Instructions

Lines 95-215 in `plan_synth_sys_prompt.md` duplicate `SPINE_BASE_PROMPT` from `profile.py`. If the two diverge, the model receives conflicting instructions. **Remove lines 95-215 from the docs file** — the profile injection handles this.

---

## 4. Micro-Tooling — Reduce Cognitive Load

### Current Tool Surface (13-15 tools)

**Purpose-built (4)**: read_prior_artifacts, search_codebase, write_plan, write_structured_plan
**Middleware (2)**: eval, task
**MCP codebase-index (11)**: find_symbol, get_function_source, get_dependencies, get_dependents, get_change_impact, get_call_chain, search_codebase, list_files, get_project_summary, get_functions, get_classes

### Redundancy Analysis

| Mechanism | Scope | Speed | Token Cost |
|---|---|---|---|
| `task` (researcher subagent) | Broad exploration | Slow (LLM turn) | High |
| `search_codebase` | Content/keyword search | Medium (rg+read) | Medium |
| `mcp_codebase-index_search_codebase` | Regex search | Fast (indexed) | Low |
| `mcp_codebase-index_find_symbol` | Symbol lookup | Fast (indexed) | Low |
| `mcp_codebase-index_get_function_source` | Single function | Fast (indexed) | Low |

**Problem**: `search_codebase` and `mcp_codebase-index_search_codebase` overlap. The prompt says MCP is "supplemental" but lists it first, confusing 30B models.

### Recommendations

1. **Remove `write_plan` tool** — redundant with `write_structured_plan`. Only `write_structured_plan` is referenced in the workflow.

2. **Filter MCP tools code-side for PLAN phase** — expose only 4 essential MCP tools:
   - `mcp_codebase-index_find_symbol` — primary symbol lookup
   - `mcp_codebase-index_get_function_source` — pattern reference
   - `mcp_codebase-index_list_files` — file discovery
   - `mcp_codebase-index_get_project_summary` — high-level overview
   
   Hide: get_dependencies, get_dependents, get_change_impact, get_call_chain, search_codebase, get_functions, get_classes (available via `find_symbol` + `get_function_source`)

3. **Clarify the 3-tier exploration hierarchy** in the prompt:
   ```
   Exploration strategy (in order):
   1. eval + task (researchers) — PRIMARY for broad codebase mapping
   2. MCP tools — SECONDARY for narrow symbol-level lookups after research
   3. search_codebase — FALLBACK for content queries not covered by MCP
   ```

4. **Shorten prompt-side tool documentation** — replace lines 3-24 with a single reference block.

---

## 5. Structured I/O — SPECIFY→PLAN Handoff

### Current Problem

PLAN consumes the SPECIFY artifact via fragile string splitting:
```js
const archSection = spec.split('## Architecture')[1]?.split('## ')[0] || '';
const ifaceSection = spec.split('## Interfaces')[1]?.split('## ')[0] || '';
```

This breaks if: headers vary, markdown formatting changes, sections are reordered, or content contains `##` in code blocks.

### Recommendations

**Short-term (low effort)**:
- Add a `specification.json` alongside `specification.md` in the SPECIFY output
- PLAN reads JSON first, falls back to markdown parsing
- JSON structure matches the `WriteSpecificationTool` input fields: `overview`, `requirements`, `architecture`, `interfaces`, `success_criteria`, `open_questions`

**Medium-term (medium effort)**:
- Add `Specification` Pydantic model to `spine/models/types.py`
- `read_prior_artifacts` loads and parses `specification.json` automatically
- PLAN receives structured spec fields instead of a markdown string
- Eliminates all string splitting

**Long-term (higher effort)**:
- Pre-parse work description before SPECIFY into structured fields (domain, scope, entities)
- Inject structured spec fields into `WorkflowState` as `specification_structured: dict`
- All phases access typed spec data without parsing

---

## 6. Implementation Plan

### Phase 1: Quick Wins (Low Risk, High Impact)

| # | Change | File | Effort | Savings |
|---|--------|------|--------|---------|
| 1 | Remove AGENTS.md embedding from prompt | `plan_synth_sys_prompt.md` | 5 min | ~1,000 tokens |
| 2 | Convert QuickJS section to positive | `plan_synth_sys_prompt.md` L110-117 | 5 min | Quality ↑ |
| 3 | Convert Core Behaviour to numbered steps | `plan_synth_sys_prompt.md` L99-105 | 5 min | Quality ↑ |
| 4 | Convert tool restrictions to positive | `plan_synth_sys_prompt.md` L26 | 5 min | Quality ↑ |
| 5 | Simplify JS dispatch example | `plan_synth_sys_prompt.md` L44-59 | 10 min | ~750 tokens |
| 6 | Add explicit complexity criteria | `plan_synth_sys_prompt.md` L72 | 5 min | Quality ↑ |
| 7 | Add explicit DAG validation | `plan_synth_sys_prompt.md` L71 | 5 min | Quality ↑ |
| **Total** | | | **~45 min** | **~1,750 tokens** |

### Phase 2: Structural Changes (Medium Risk, High Impact)

| # | Change | File | Effort | Savings |
|---|--------|------|--------|---------|
| 8 | Simplify `_build_plan_prompt()` to minimal framing | `plan_agent.py` | 30 min | ~2,000 tokens |
| 9 | Remove hand-written tool docs from prompt | `plan_synth_sys_prompt.md` L3-24 | 15 min | ~700 tokens |
| 10 | Enhance tool descriptions with usage patterns | `plan_tools.py` | 30 min | Quality ↑ |
| 11 | Remove `write_plan` tool (redundant) | `plan_tools.py` | 15 min | ~100 tokens |
| 12 | Filter MCP tools to 4 essentials for PLAN | `plan_agent.py` or `factory.py` | 30 min | ~300 tokens |
| **Total** | | | **~2 hours** | **~3,100 tokens** |

### Phase 3: Structured I/O (Higher Effort, Reliability Impact)

| # | Change | File | Effort | Impact |
|---|--------|------|--------|--------|
| 13 | Add `specification.json` output to SPECIFY | `specify_tools.py` | 30 min | Reliability ↑ |
| 14 | Add `Specification` Pydantic model | `types.py` | 30 min | Type safety ↑ |
| 15 | Update `read_prior_artifacts` to load JSON | `plan_tools.py` | 20 min | Reliability ↑ |
| 16 | Remove string splitting from PLAN prompt | `plan_synth_sys_prompt.md` | 15 min | Reliability ↑ |
| **Total** | | | **~1.5 hours** | **Reliability ↑↑** |

### Phase 4: Validation

| # | Action | Details |
|---|--------|---------|
| 17 | Run `spine run` end-to-end | Verify PLAN phase still produces valid feature_slices |
| 18 | Token count comparison | Measure actual token reduction |
| 19 | Test with local 30B model | Verify quantized model performance improves |
| 20 | Update `plan_synth_sys_prompt.md` docs | Reflect all changes |

---

## 7. Recommended Prompt Structure (After Optimization)

The optimized `plan_synth_sys_prompt.md` should be ~90 lines instead of ~260:

```
# PLAN Phase Agent

## Role
Create a technical plan from the specification, grounded in codebase structure.
Output: flat array of feature_slices with explicit dependencies.

## Workflow (3 steps, 4 turns)

### Step 1 — Load context (turn 1)
Call read_prior_artifacts with no arguments. Store:
  globalThis.ctx = JSON.parse(result);

### Step 2 — Dispatch researchers (turn 2)
Identify 2-4 spec areas needing codebase mapping. For each:
  - Extract the relevant spec section
  - Dispatch a researcher subagent via eval + Promise.allSettled
  - Each description MUST be ≥300 chars and include the spec section verbatim

### Step 3 — Write plan (turn 3-4)
Synthesize spec + research into feature_slices. Call write_structured_plan.

## feature_slices requirements
Each slice MUST have: id, title, target_files, execution_requirements,
  dependencies, acceptance_criteria, complexity.

## Slice design rules
- Count: 2-8 slices
- Size: ≤30 files, ≤2000 lines per slice
- Complexity: small (≤1 file, ≤50 lines), medium (2-3 files, ≤200 lines), large (4+ files)
- DAG: no slice depends on itself. Verify no cycles.
- target_files: MUST come from MCP or search_codebase output, or be new files in verified dirs

## Rework handling
If feedback is non-empty: address EVERY item before calling write_structured_plan.

## Exploration hierarchy
1. eval + task (researchers) — PRIMARY
2. MCP tools (find_symbol, get_function_source) — SECONDARY
3. search_codebase — FALLBACK

## Strict rules
- read_prior_artifacts FIRST
- write_structured_plan LAST, exactly once
- By turn 5: call write_structured_plan with what you have
```

---

## 8. Files to Modify

| File | Changes |
|------|---------|
| `docs/plan_synth_sys_prompt.md` | Remove AGENTS.md, convert negatives→positives, simplify examples, add explicit criteria |
| `spine/agents/plan_agent.py` | Simplify `_build_plan_prompt()`, filter MCP tools |
| `spine/agents/plan_tools.py` | Enhance tool descriptions, remove `write_plan` |
| `spine/agents/specify_tools.py` | Add `specification.json` output |
| `spine/models/types.py` | Add `Specification` Pydantic model |
| `spine/agents/profile.py` | No changes needed |

---

*Report generated: 2026-05-23*
*Analyses: 5 parallel subagents, ~14 total LLM calls*
