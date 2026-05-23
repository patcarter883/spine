# Redundancy Analysis: Specify Phase System Prompt

## Executive Summary

The specify phase system prompt includes ~38KB of project documentation that is largely redundant. The specify agent:
1. **Cannot write code** - it only dispatches researchers and writes specifications
2. **Receives all needed context via `read_work_context`** tool
3. **Does not need provider configs, CLI commands, or code style details**

**Estimated token savings: ~25,000-30,000 tokens** by removing/reducing redundant sections.

---

## Category Analysis

### 1. Project Overview Information

**Content in AGENTS.md (lines 1-265):**
- SPINE architecture overview (LangGraph, Deep Agents, Streamlit)
- Work types (Task, Critical Task, Reviewed Task, Critical Reviewed Task)
- Workflow sequences (SPECIFY -> PLAN -> CRITIC_PLAN -> IMPLEMENT -> VERIFY)
- CLI commands (run, status, list, resume, restart, worker, ui)
- Provider resolution mechanics
- Subgraph architecture details
- Key modules table (55 entries)

**Analysis:**
- **Injected:** Work type is injected via `work_type` in `read_work_context`
- **Needed:** NO - The specify agent doesn't need to know CLI commands or the full architecture
- **Token cost:** ~8,000 tokens
- **Recommendation:** **REMOVE** - Completely unnecessary for a specification-writing agent

---

### 2. Phase-Specific Context (Already Injected)

**Content that duplicates runtime injection:**

| Field | Injected via `read_work_context` | Repeated in AGENTS.md |
|-------|----------------------------------|----------------------|
| `work_id` | ✅ Yes | Implied in workflow context |
| `description` | ✅ Yes | Repeated in work types section |
| `work_type` | ✅ Yes | Repeated in work types section |
| `feedback` | ✅ Yes | Not in AGENTS.md (good) |
| `workspace_root` | ✅ Yes | Injected via tool, not prompt |
| `spec_dir` | ✅ Yes | Injected via tool, not prompt |

**Analysis:**
- All phase-specific context is **already injected** via `read_work_context` tool
- AGENTS.md doesn't duplicate this directly (good), but the system prompt wrapper does somewhat
- **Recommendation:** **NO CHANGE NEEDED** - Already properly injected

---

### 3. Provider Configuration Details

**Content (lines 43-108 in AGENTS.md):**
```yaml
providers:
  llm:
    - name: glm
      type: deepagents-model
      model: openrouter:z-ai/glm-5.1
      ... (8 more providers with full config)
  phases:
    specify:
      provider: deepseek-v4-pro
    ... (6 more phases)
```

**Analysis:**
- **Injected:** Provider selection happens at runtime via `build_phase_agent()`
- **Needed:** NO - The agent doesn't choose providers; the system does
- **Token cost:** ~2,500 tokens
- **Recommendation:** **REMOVE** - Irrelevant to specification writing

---

### 4. Design Principles (12 Principles)

**Content (lines 236-263 in AGENTS.md):**

| # | Principle | Relevant to Specify? |
|---|-----------|---------------------|
| 1 | LangGraph is workflow engine | NO |
| 2 | Zero duplication: CLI/UI share code | NO |
| 3 | One Deep Agent Per Phase | NO (implementation detail) |
| 4 | Critic Gate Is Structural | NO |
| 5 | Artifact Gates Prevent Empty Progression | NO |
| 6 | SPINE Base Prompt Replaces DA Default | NO |
| 7 | OpenRouter Session Tracking | NO |
| 8 | Phase Nodes Must Return Complete State Updates | NO |
| 9 | Subgraph State Isolation | NO |
| 10 | WebSocket Push Notifications | NO |
| 11 | Feature Flag Controlled Rollout | NO |
| 12 | MCP Tools Are Namespaced | NO |

**Analysis:**
- **Injected:** Not applicable
- **Needed:** **NONE** - All 12 principles are implementation details for developers, not specification writers
- **Token cost:** ~1,200 tokens
- **Recommendation:** **REMOVE ALL 12** - None are relevant

---

### 5. Code Style, Naming, Testing Patterns

**Content (lines 289-337 in AGENTS.md):**
- Linting: `ruff check`
- Type annotations: modern syntax
- Import grouping rules
- Naming conventions (PascalCase, snake_case)
- Testing patterns (pytest, asyncio, mock)
- What NOT to test

**Analysis:**
- **Injected:** Not applicable
- **Needed:** **NO** - The specify agent DOES NOT WRITE CODE, ONLY SPECIFICATIONS
- **Token cost:** ~1,500 tokens
- **Recommendation:** **REMOVE** - Entirely irrelevant to a specification phase

---

### 6. Directory Layout & Dependencies

**Content (lines 341-385 in AGENTS.md):**
- Source tree structure (`spine/agents/`, `spine/workflow/`, etc.)
- `.spine/` runtime directories
- Dependency list by category
- Environment variables table

**Analysis:**
- **Injected:** `workspace_root` is injected
- **Needed:** 
  - **Directory layout:** NO - Researchers explore, not the specify agent directly
  - **Dependencies:** NO - Irrelevant to writing specs
  - **Environment variables:** NO - Already configured at runtime
- **Token cost:** ~1,200 tokens
- **Recommendation:** **REMOVE**

---

### 7. "Adding a New Phase/Page" Sections

**Content (lines 400-428 in AGENTS.md):**
- Adding new workflow phase (9 steps)
- Adding new UI page
- Git hooks

**Analysis:**
- **Injected:** Not applicable
- **Needed:** NO - The specify agent doesn't add phases or pages
- **Token cost:** ~600 tokens
- **Recommendation:** **REMOVE**

---

## Detailed Section Breakdown

| Section | Lines | Tokens | Recommendation |
|---------|-------|--------|----------------|
| Project Overview | 1-43 | ~500 | REMOVE |
| Work Types | 14-432 | ~800 | REMOVE |
| Configuration/YAML | 43-108 | ~2,500 | REMOVE |
| Provider Resolution | 110-114 | ~200 | REMOVE |
| CLI Commands | 124-137 | ~300 | REMOVE |
| Provider Setup Notes | 133-143 | ~400 | REMOVE |
| Subgraph Architecture | 145-232 | ~1,500 | REMOVE |
| Key Modules Table | 166-232 | ~2,000 | REMOVE |
| Design Principles (12) | 236-263 | ~1,200 | REMOVE |
| Pitfalls | 267-285 | ~600 | REMOVE |
| Code Style | 289-309 | ~600 | REMOVE |
| Testing | 313-337 | ~900 | REMOVE |
| Directory Layout | 341-360 | ~400 | REMOVE |
| Dependencies | 375-384 | ~400 | REMOVE |
| Environment Variables | 388-396 | ~200 | REMOVE |
| Adding Phase/Page | 400-428 | ~600 | REMOVE |
| License | 432-434 | ~100 | REMOVE |

**Total removable tokens: ~14,100 tokens**

---

## What SHOULD Remain

**Keep in system prompt:**
1. **Phase-specific instructions** (lines 1-166 of specify_synth_sys_prompt.md)
   - Tool surface description
   - Workflow steps
   - Strict rules
   - Eval context seed
   - Core behavior

2. **Minimal project context** (if any):
   - Nothing fundamental - the work description + researcher findings provide all needed context

---

## Critical Discovery: Memory Injection Mechanism

The AGENTS.md content is injected via `SpineProjectMemoryMiddleware` (line 177-221 in factory.py), which loads the full AGENTS.md file as "memory" for every phase ** EXCEPT** TASKS and CRITIC phases (see `resolve_memory` in skills_resolver.py lines 123-128).

The `_SKIP_AGENTS_MD` set currently only contains `{PhaseName.TASKS.value, PhaseName.CRITIC.value}`.

**The SPECIFY phase is NOT in this set, so it currently loads the full AGENTS.md!**

This is exactly the problem the task identifies - the specify phase loads ~38KB of redundant documentation.

---

## Recommendations

### Option A: Aggressive Reduction (Recommended)
- **Remove SPECIFY from AGENTS.md injection** by adding it to `_SKIP_AGENTS_MD` in skills_resolver.py
- Keep only the `_build_specify_prompt()` content (~7.7KB, ~1,200 tokens)
- **Savings: ~37,000 tokens (38KB of documentation)**

### Option B: Minimal Retention
If some project context is deemed necessary for specify:
- Create a trimmed `SPECIFY_CONTEXT.md` with only relevant information
- Update resolve_memory to load this smaller file instead
- Or use phase-specific memory filtering

---

## Implementation Plan

1. **Short term:** Remove AGENTS.md injection from specify phase prompt
2. **Medium term:** Create `SPECIFY_CONTEXT.md` with phase-specific guidance if needed
3. **Long term:** Consider separate trimmed prompts for each phase (plan, implement, verify)