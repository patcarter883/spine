# SPINE Phase Agent Prompt Optimization Plan

**Date:** 2026-05-23
**Scope:** Implement orchestrator, specify orchestrator, and all subagent prompts
**Target models:** Local vLLM (Qwen3.6-35B quantized), OpenRouter (deepseek-v4-pro)

---

## Executive Summary

Analysis of all SPINE phase agent prompts identified ~2000-3000 tokens of redundant content per phase invocation, 15+ negative-constraint instructions that should be rewritten as positive directives, hyper-literal compliance issues specific to quantized 30B models, and micro-tooling density imbalances. This plan provides concrete before/after code for each change.

**Estimated impact:** 25-35% reduction in prompt tokens, measurably better 30B model compliance through positive instructions, ~200 tokens saved in researcher tool surface.

---

## Change 1: Remove Duplicated Content

### Problem
Phase system prompts contain content already provided via other injection mechanisms:
- `SPINE_BASE_PROMPT` via `HarnessProfile` (Core Behaviour, Interpreter Environment, Tools guidance, Workflow Context, Output format)
- `SpineContext` auto-injected (work_id, phase, workspace_root, etc.)
- `SkillsMiddleware` auto-loading
- `TodoListMiddleware` auto-loading
- `skip_filesystem_middleware=True` enforcing tool absence

### Sections to Remove (all from `spine/agents/implement_agent.py` and `spine/agents/specify_agent.py`)

| Section | Tokens | Injection source | Action |
|---|---|---|---|
| Core Behaviour ("Act, don't narrate") | ~200 | SPINE_BASE_PROMPT | Remove |
| Interpreter Environment (QuickJS) | ~300 | SPINE_BASE_PROMPT + rlm-pattern skill | Remove |
| Tools guidance ("Read before write") | ~150 | SPINE_BASE_PROMPT | Remove |
| Workflow Context | ~100 | SPINE_BASE_PROMPT | Remove |
| Output format guidance | ~80 | SPINE_BASE_PROMPT | Remove |
| Skills System explanation | ~250 | SkillsMiddleware | Remove |
| `task` subagent spawner docs | ~200 | SPINE_BASE_PROMPT | Remove |
| `write_todos` tool docs | ~120 | TodoListMiddleware | Remove |
| Eval context seed | ~60 | SpineContext | Remove |
| MCP tools list | ~80 | MCP tools auto-injected | Remove |
| Negative constraint about missing tools | ~50 | `skip_filesystem_middleware=True` | Remove |

### Code Changes

**`spine/agents/implement_agent.py` — `_build_wave_orchestrator_prompt()` rewrite:**

```python
def _build_wave_orchestrator_prompt() -> str:
    """Orchestrator prompt for wave-based dispatch (plan.json mode)."""
    return (
        "You are the IMPLEMENT phase orchestrator. Your job: dispatch "
        "slice-implementer subagents and synthesize their results into "
        "implementation.md.\n\n"
        "## Tools\n"
        "1. `read_slice_files` — load slice definitions + codebase map (call first)\n"
        "2. `write_implementation_report` — write the phase artifact (call last)\n"
        "3. `eval` — run JavaScript for subagent dispatch\n\n"
        "## Workflow (3 turns)\n"
        "Turn 1: Call `read_slice_files()`. Store result in `globalThis.planData`.\n"
        "Turn 2: Dispatch subagents via `eval` — dispatch pattern below.\n"
        "Turn 3: Call `write_implementation_report({slice_results, summary})`.\n\n"
        "## Dispatch pattern\n"
        "Read `globalThis.planData.slices` for slice definitions. For each wave,\n"
        "dispatch slices in parallel. Embed full slice definition, codebase map,\n"
        "and acceptance criteria in each task description.\n\n"
        "```js\n"
        "const waves = globalThis.execution_waves || [];\n"
        "const sliceDefs = globalThis.planData.slices;\n"
        "const results = [];\n"
        "for (const wave of waves) {\n"
        "  const waveResults = await Promise.allSettled(\n"
        "    wave.map(sliceId => tools.task({\n"
        "      subagent_type: 'slice-implementer',\n"
        "      description: buildDispatchDescription(sliceDefs[sliceId])\n"
        "    }))\n"
        "  );\n"
        "  results.push(...waveResults);\n"
        "}\n"
        "globalThis.sliceResults = results;\n"
        "\n"
        "function buildDispatchDescription(slice) {\n"
        "  return `Implement slice '${slice.id}': ${slice.execution_requirements}` +\n"
        "    `\\n\\nFiles to modify: ${slice.target_files.join(', ')}` +\n"
        "    `\\n\\nAcceptance criteria: ${slice.acceptance_criteria.join('; ')}`;\n"
        "}\n"
        "```<|tool_calls_end|>\n\n"
        "## Completion criteria\n"
        "Call `write_implementation_report` with: slice_results (list of {slice_name, status, files_modified, files_created, test_results, issues}), summary (str).\n"
        "Phase is complete only after write_implementation_report returns successfully.\n"
    )
```

**`spine/agents/specify_agent.py` — `_build_specify_prompt()` rewrite:**

```python
def _build_specify_prompt() -> str:
    return (
        "You are the SPECIFY phase orchestrator. Your job: dispatch researcher "
        "subagents to explore the codebase, then synthesize findings into a "
        "structured specification.md.\n\n"
        "## Tools\n"
        "1. `read_work_context` — load work description, feedback, prior spec (call first)\n"
        "2. `write_specification` — write specification.md with 5 required sections (call last)\n"
        "3. `eval` — run JavaScript for parallel subagent dispatch\n\n"
        "## Workflow (3 turns)\n"
        "Turn 1: Call `read_work_context()`. Store in `globalThis.ctx`.\n"
        "Turn 2: Identify 2-4 codebase areas relevant to the work. Dispatch researcher\n"
        "  subagents via `eval` — dispatch pattern below.\n"
        "  Each description must be >=200 characters. Include work context, specific\n"
        "  file paths, and what to look for.\n"
        "Turn 3: Synthesize research into 5 sections and call `write_specification`.\n\n"
        "## Dispatch pattern\n"
        "```js\n"
        "const desc = globalThis.ctx.description;\n"
        "const topics = [\n"
        "  {area: 'phase_architecture', files: ['spine/workflow/registry.py', 'spine/phases/']},\n"
        "  // add 2-4 topics based on the work description\n"
        "];\n"
        "globalThis.research = await Promise.allSettled(\n"
        "  topics.map(t => tools.task({\n"
        "    subagent_type: 'researcher',\n"
        "    description: `Research ${t.area} for: ${desc}\\n` +\n"
        "      `Investigate: ${t.files.join(', ')}\\n` +\n"
        "      `Look for: conventions, patterns, APIs, dependencies`\n"
        "  }))\n"
        ");\n"
        "```<|tool_calls_end|>\n\n"
        "## Specification sections\n"
        "- overview: what to build (2-4 paragraphs)\n"
        "- requirements: functional + non-functional, measurable list\n"
        "- architecture: design decisions + rationale\n"
        "- interfaces: APIs, data models, contracts\n"
        "- success_criteria: verifiable outcomes for VERIFY phase\n\n"
        "## Completion criteria\n"
        "Call `write_specification` once with all 5 required fields.\n"
        "Phase is complete only after write_specification returns successfully.\n"
    )
```

---

## Change 2: Rewrite Negative → Positive Constraints

### Problem
15+ negative-constraint instructions in prompts confuse quantized 30B models. Models understand direct positive instructions far better than "do not X" patterns.

### Change Table

| File | Negative phrasing | Positive rewrite |
|---|---|---|
| implement_agent.py | "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, or `execute`." | Remove — tool design enforces this |
| implement_agent.py | "Do not attempt to call them." | Remove — not possible without tools |
| implement_agent.py | "There is nothing to explore" | Remove |
| implement_agent.py | "Do not skip `read_slice_files`" | "Call `read_slice_files` first" |
| implement_agent.py | "Do not attempt to implement slices yourself" | Remove — no write tools available |
| implement_agent.py | "More than 5 turns without dispatching subagents means something has gone wrong" | "Expected workflow: read_slice_files → eval (dispatch) → write_implementation_report" |
| implement_agent.py | "Do NOT ask follow-up questions" | "Execute autonomously — produce the artifact with given context" |
| implement_agent.py | "Do NOT seek user approval" | Remove — no user in the loop |
| implement_agent.py | "Do NOT load everything into context at once" | "Read files selectively — use offset/limit for large files" |
| specify_agent.py | "Do not attempt to call them — they do not exist in your session" | Remove — same enforcement |
| specify_agent.py | "Do not write the spec from the description alone without codebase research" | "Dispatch researchers before writing the spec" |
| specify_agent.py | "More than 5 turns without calling write_specification means something has gone wrong" | "Expected workflow: read_work_context → eval (research) → write_specification" |
| subagents.py | "Do not describe changes — make them with write_file" | "Use write_file and edit_file to make changes, then verify with execute" |
| subagents.py | "Do not return empty results" | "Report findings with actual file reads and evidence" |
| subagents.py | "Do not touch files outside its scope" | "Modify only files listed in the slice definition" |
| subagents.py | "Do not try to implement the dependency yourself" | "Report the blocking dependency in issues" |
| subagents.py | "Do not fix issues you find" | "Record failures in the checklist and gaps — do not repair" |
| subagents.py | "Do not verify from memory" | "Inspect actual files and run actual tests" |
| subagents.py | "Do NOT return empty results again" | "Report findings with actual file reads and evidence" |

---

## Change 3: Hyper-Literal Restructuring for 30B Models

### Problem
Quantized 30B models understand direct step-by-step instructions far better than abstract guidance, cross-references, and implicit tool routing.

### Implement Orchestrator — Inline Complete Dispatch Pattern

**Before:** "Refer to Step 2 guidelines preloaded in your user prompt."
**After:** Complete JavaScript dispatch pattern inline (see Change 1 code).

### Researcher Subagent — Reduce Tool Surface

**Before:** 18 MCP tools listed in a table with individual call syntax.

**After:**
```markdown
Use these tools first:
1. `mcp_codebase-index_find_symbol({"name": "X"})` — locate any symbol definition
2. `mcp_codebase-index_get_function_source({"name": "X"})` — read function source
3. `mcp_codebase-index_get_dependencies({"name": "X"})` — what a symbol calls/uses
4. `mcp_codebase-index_get_project_summary()` — file count, packages, top symbols
5. `mcp_codebase-index_search_codebase({"pattern": "regex", "max_results": 20})` — search all files

For other needs: list_files, get_classes, get_functions, get_imports,
get_dependents, get_change_impact, get_call_chain.

Fallback: search_codebase, read_file, ls, glob, grep.
```

### Slice-Implementer Subagent — Simplify Workflow

**Before:** "Exploration budget: Maximum 3 turns of read/search before your first write/edit. If you haven't changed code by turn 4, you're over-exploring"

**After:** "Read files → make changes → run tests → fix if needed. Aim for 4-5 turns total."

---

## Change 4: Specify Workflow Structured I/O

### Current State
The specify orchestrator already has good tool design. `read_work_context` returns structured JSON, `write_specification` accepts 5 required fields.

### Enhancement 1: Pydantic Schema Validation

**File:** `spine/agents/specify_tools.py`

```python
class _WriteSpecificationInput(BaseModel):
    overview: str = Field(..., min_length=100, description="Summary of what needs to be built (2-4 paragraphs).")
    requirements: str = Field(..., min_length=200, description="Functional and non-functional requirements as a markdown list.")
    architecture: str = Field(..., min_length=300, description="High-level design decisions: components, data flow, key patterns.")
    interfaces: str = Field(..., min_length=200, description="API endpoints, data models, and contracts.")
    success_criteria: str = Field(..., min_length=100, description="Measurable outcomes that define completion.")
    open_questions: str = Field(default="", description="Any open questions or risks. Optional.")
```

### Enhancement 2: Explicit Research Topic Mapping

The orchestrator already composes task descriptions as free-text. The improvement is making the topic-listing pattern explicit in the prompt rather than leaving it to model inference (see Change 1 code for the concrete JS pattern).

---

## Change 5: Micro-Tooling Refinements

### Tool Surface Analysis

| Phase | Current Tools | Count | Suggested | Impact |
|---|---|---|---|---|
| Implement orchestrator | 2 custom + task + eval | 4 | No change | Already optimal |
| Specify orchestrator | 2 custom + task + eval | 4 | No change | Already optimal |
| Researcher | 8 MCP explicit + 3 fallback | 11 | Top 5 MCP + discovery pattern | ~200 tokens saved |
| Slice-implementer | 8 generic tools | 8 | No change | All needed for implementation |
| Slice-verifier | 6 read+execute tools | 6 | No change | All needed for verification |

### Why Implement/Specify Are Already Optimal
The implement orchestrator's tool surface (`read_slice_files`, `write_implementation_report`, `task`, `eval`) enforces a 3-step workflow at the tool level. There is literally nothing else the model can do. This is the strongest form of tool restriction — no prompt guidance needed because the tools make the wrong action impossible.

---

## Priority Order & Implementation Effort

| Priority | Change | Effort | Risk |
|---|---|---|---|
| P0 | Remove duplicated prompt sections | Medium — edit 2 files | Low — content already injected elsewhere |
| P0 | Negative → positive rewrites | Small — find-and-replace | Very low — purely stylistic |
| P1 | Hyper-literal restructuring | Medium — rewrite prompts | Medium — test with actual vLLM |
| P1 | Specify schema validation | Small — add min_length to 5 fields | Low — runtime validation |
| P2 | Researcher tool table reduction | Small — rewrite subagent prompt | Low |

---

## Files to Modify

1. **`spine/agents/implement_agent.py`** — `_build_wave_orchestrator_prompt()`, `_build_legacy_orchestrator_prompt()`
2. **`spine/agents/specify_agent.py`** — `_build_specify_prompt()`
3. **`spine/agents/subagents.py`** — `SUBAGENT_PROMPTS` (researcher, slice-implementer, slice-verifier)
4. **`spine/agents/specify_tools.py`** — `_WriteSpecificationInput` min_length validation
5. **`spine/agents/profile.py`** — (optional) consolidate SPINE_BASE_PROMPT to absorb remaining duplicates

---

## Verification Checklist

- [ ] Remove all identified duplicate sections from both orchestrator prompts
- [ ] Replace all negative constraints with positive directives
- [ ] Inline complete dispatch patterns (no cross-references)
- [ ] Reduce researcher tool table from 18 to 5 + discovery
- [ ] Add min_length validation to WriteSpecificationInput fields
- [ ] Run existing implement phase work items through vLLM and compare token counts
- [ ] Verify no functional regression in workflow progression
