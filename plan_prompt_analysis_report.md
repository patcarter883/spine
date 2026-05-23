# Plan System Prompt Injection Analysis Report

## Executive Summary

This report identifies information in the plan synthesis system prompt (`/home/pat/Projects/spine/docs/plan_synth_sys_prompt.md`) that duplicates data already available via injected state mechanisms in SPINE. Significant token savings (~1,000+ tokens) are achievable by leveraging runtime injection instead of inline prompt text.

---

## 1. Information Available via Injected State (Not in Prompt)

### Tool Descriptions (Auto-injected by Runtime)
| Information | Injected Source | Token Estimate |
|-------------|-----------------|----------------|
| `read_prior_artifacts` tool description | `plan_tools.py`: ReadPriorArtifactsTool.description | ~50 tokens |
| `search_codebase` tool description | `plan_tools.py`: SearchCodebaseTool.description | ~50 tokens |
| `write_structured_plan` tool description | `plan_tools.py`: StructuredWritePlanTool.description | ~50 tokens |
| `write_plan` tool description | `plan_tools.py`: WritePlanTool.description | ~30 tokens |
| MCP tool descriptions | `spine/mcp/client.py` via `get_mcp_tools()` | ~200 tokens |

**Total potential savings**: ~380 tokens

### Context Properties (via SpineContext)
| Property | Injected Source | Current Prompt Usage |
|----------|-----------------|---------------------|
| `work_id` | `SpineContext.work_id` | In prompt: "Access session-specific context properties via `globalThis.context`" |
| `plan_dir` | `SpineContext.plan_dir` (derived) | Same as above |
| `feedback` | `SpineContext.critic_feedback` | Inlined as "## Previous Review Feedback" |
| `retry_count` | `SpineContext.retry_count` | Inlined as rework handling |
| `workspace_root` | `SpineContext.workspace_root` | Not in prompt |

**Total potential savings**: ~100 tokens

### Workflow Metadata (via globalThis)
| Information | Injected Source | Current Prompt Usage |
|-------------|-----------------|---------------------|
| Phase workflow sequence | `SpineContext.phase` + AGENTS.md memory | "## Workflow Context" section |
| Work type info | `SpineContext.work_type` | Generic text |

---

## 2. Verbatim Duplication Between `plan_agent.py` and `plan_synth_sys_prompt.md`

### Exact Match Analysis

The `_build_plan_prompt()` function in `plan_agent.py` (lines 108-261) contains substantial text that appears verbatim in `docs/plan_synth_sys_prompt.md`:

| Section | Lines in plan_agent.py | Lines in plan_synth_sys_prompt.md | Token Count |
|---------|------------------------|-----------------------------------|-------------|
| Opening role description | 109-114 | 1-1 | ~50 tokens |
| Tool surface (complete list) | 115-153 | 3-26 | ~800 tokens |
| "You do NOT have ls..." section | 154-157 | 26-26 | ~50 tokens |
| Workflow section | 158-207 | 28-59 | ~600 tokens |
| feature_slices structure | 212-231 | 64-80 | ~300 tokens |
| Slice design rules | 232-241 | 74-79 | ~150 tokens |
| Rework handling | 242-246 | 81-82 | ~50 tokens |
| Strict rules | 247-256 | 84-89 | ~100 tokens |
| Eval context seed | 257-260 | 90-91 | ~30 tokens |

**Total duplicated content**: ~2,130 tokens

**Savings opportunity**: The entire prompt content in `plan_agent.py` could be replaced by loading from `plan_synth_sys_prompt.md` OR the prompt could be dramatically simplified since most tool descriptions are auto-injected.

---

## 3. Content Suitable for MCP Tool Descriptions

### `read_prior_artifacts` Tool
**Current prompt text** (lines 30-35 in sys_prompt.md):
```
### Step 1 — Call read_prior_artifacts (1 turn)
Call `read_prior_artifacts` with no arguments, store results:
```js
globalThis.ctx = JSON.parse(result);
// ctx.description, ctx.artifacts.specify, ctx.feedback
```

This usage pattern could be documented in the tool's `description` field instead, saving ~100 tokens.

### MCP Tool Call Patterns
**Current prompt text** (lines 9-18 in sys_prompt.md):
```
Call with native kwargs (no tool_input wrapper):
- `mcp_codebase-index_find_symbol` — locate symbol. Call: `{"name": "symbol_name"}`
- `mcp_codebase-index_get_function_source` — get function source. Call: `{"name": "func_name"}`
... etc
```

These call patterns are already documented in MCP tool schemas. The prompt duplicates ~200 tokens that could be removed.

---

## 4. Content Suitable for SPINE_BASE_PROMPT (profile.py)

### Current Redundancies

The `SPINE_BASE_PROMPT` in `profile.py` (lines 72-148) contains content that overlaps with `plan_synth_sys_prompt.md`:

| Content | SPINE_BASE_PROMPT | plan_synth_sys_prompt.md | Status |
|---------|-------------------|--------------------------|--------|
| Core Behaviour section | Lines 78-89 | Lines 99-106 | **DUPLICATE** - Already injected via profile |
| Interpreter Environment | Lines 96-110 | Lines 107-122 | **DUPLICATE** - Already injected via profile + skill |
| Tools section | Lines 112-130 | Lines 123-141 | **DUPLICATE** - Already injected via profile |
| Workflow Context | Lines 132-139 | Lines 143-150 | **DUPLICATE** - Already injected via profile |
| Output section | Lines 141-147 | Lines 150-155 | **DUPLICATE** - Already injected via profile |

**Key Issue**: The `plan_synth_sys_prompt.md` file re-includes the entire `SPINE_BASE_PROMPT` content (embedded via `<project_documentation path=".../AGENTS.md">`), then the `profile.py` also injects `SPINE_BASE_PROMPT` as the CUSTOM slot in prompt assembly.

**Savings opportunity**: Remove AGENTS.md embedding from plan_synth_sys_prompt.md and rely solely on profile injection. This would save ~400 tokens from the embedded AGENTS.md section alone.

---

## 5. Specific Recommendations

### Recommendation 1: Move Tool Usage Patterns to Tool Descriptions
**Action**: Enhance `read_prior_artifacts` and `write_structured_plan` tool descriptions with usage examples.
```python
# In plan_tools.py
description = (
    "Load all prior phase artifacts... "
    "Usage: Call FIRST with no arguments. "
    "Returns work_id, work_type, description, feedback, plan_dir, and artifacts dict."
)
```
**Estimated savings**: ~300 tokens

### Recommendation 2: Remove AGENTS.md Embedding from Prompt File
**Action**: Delete lines 226-660 from `plan_synth_sys_prompt.md` (the `<project_documentation>` section containing AGENTS.md).
**Estimated savings**: ~1,000 tokens

### Recommendation 3: Simplify `_build_plan_prompt()` in `plan_agent.py`
**Action**: Replace the ~2,100 token prompt with a minimal phase-framing message:
```python
def _build_plan_prompt() -> str:
    return (
        "You are the PLAN phase agent. Create a technical plan from the specification, "
        "grounded in codebase structure. Use read_prior_artifacts FIRST, then dispatch "
        "researcher subagents via eval + Promise.allSettled, then write_structured_plan."
    )
```
**Estimated savings**: ~2,000 tokens (replaced by injected tool descriptions)

### Recommendation 4: Move MCP Guidance to Interpreter Skill
**Action**: The detailed MCP tool call patterns could be moved to the `rlm-pattern` skill that's already loaded when the interpreter is available.
**Estimated savings**: ~200 tokens

### Recommendation 5: Use `WriteStructuredPlanTool` Schema for Slice Requirements
**Action**: The slice field requirements (lines 64-71 in sys_prompt.md) are already specified in `_FeatureSliceInput` in `plan_tools.py`. Reference the tool schema instead.
**Estimated savings**: ~150 tokens

---

## 6. Token Budget Impact

| Component | Current Tokens | After Changes | Savings |
|-----------|---------------|---------------|---------|
| Phase system_prompt | ~800 | ~150 | ~650 |
| SPINE_BASE_PROMPT (embedded AGENTS.md) | ~0 (already injected) | 0 | +1,000 |
| Duplicate tool descriptions | ~500 | 0 | 500 |
| MCP tool guidance | ~300 | 0 | 300 |
| Unused interpreter guidance | ~200 | 0 | 200 |
| **Total** | **~2,600** | **~150** | **~2,450** |

**Net token reduction**: ~2,450 tokens per PLAN phase invocation

---

## 7. Implementation Priority

| Priority | Change | Effort | Risk |
|----------|--------|--------|------|
| High | Remove AGENTS.md embedding | Low | Low |
| High | Simplify `_build_plan_prompt()` | Low | Medium |
| Medium | Enhance tool descriptions | Medium | Low |
| Low | Move MCP guidance to skill | Medium | Low |

---

## Files to Modify

1. `/home/pat/Projects/spine/docs/plan_synth_sys_prompt.md` - Remove AGENTS.md embedded content
2. `/home/pat/Projects/spine/spine/agents/plan_agent.py` - Simplify `_build_plan_prompt()`
3. `/home/pat/Projects/spine/spine/agents/plan_tools.py` - Enhance tool descriptions
4. `/home/pat/Projects/spine/spine/agents/profile.py` - No changes needed (already correct)