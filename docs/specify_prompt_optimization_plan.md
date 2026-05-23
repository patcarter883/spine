# Specify Phase System Prompt Optimization — Consolidated Plan

## Executive Summary

The specify phase system prompt (662 lines, ~38KB) has five classes of issues that degrade performance on quantized 30B models:

| Issue | Impact | Fix Effort |
|-------|--------|------------|
| Negative prompting dominates | Model ignores or misinterprets ~30 negative constraints | Medium |
| AGENTS.md injection wasteful | 37K tokens of irrelevant project docs per run | Low (1 code change) |
| Philosophical/ambiguous language | 30B models need literal directives, not "iterate, be concise" | Medium |
| Tool surface too dense | ~22 tools mentioned; cross-contamination between phase and subagent | Medium |
| Unstructured data flow | Free-text research → free-text spec; downstream phases must parse markdown | Medium |

**Total estimated token savings: ~25K-37K tokens per specify run** (mostly from removing AGENTS.md).
**Quality improvements**: ~30 rewritten instructions, 4 conflicting rules resolved, structured intermediate artifact added.

---

## Change Set Overview

### P0 — High Impact, Low Effort (implement first)
| # | Change | Tokens Saved | Effort |
|---|--------|-------------|--------|
| A | Remove SPECIFY from AGENTS.md injection | ~22K tokens | 1 line of code |
| B | Rewrite negative instructions as positive | 0 (same size) | Medium |
| C | Remove irrelevant prompt sections (write_todos, generic skills, generic task docs) | ~2,500 tokens | Medium |

### P1 — Medium Impact, Medium Effort
| # | Change | Tokens Saved | Effort |
|---|--------|-------------|--------|
| D | Restructure to hyper-literal, numbered workflow | 0 (same size) | Medium |
| E | Reduce researcher tool surface from 17 to 6 essential tools | ~1,500 tokens (prompt) | Medium |
| F | Remove MCP tool list from specify agent prompt (belongs to researcher only) | ~500 tokens | Low |

### P2 — Structured Data Flow (medium effort, downstream benefits)
| # | Change | Complexity | Effort |
|---|--------|------------|--------|
| G | Extend ResearchFindings with 3 new fields | Low | 30 min |
| H | Add structured `specification.json` parallel output | Medium | 1 hour |
| I | Add intermediate `ResearchBrief` artifact | Medium | 2 hours |

---

## Change A: Remove AGENTS.md Injection for SPECIFY Phase

### Problem
The `resolve_memory()` function in `spine/agents/skills_resolver.py` injects the full 662-line AGENTS.md (~22K chars, ~5K tokens per call, bloated by repeated injection into the prompt) for every phase except TASKS and CRITIC. The specify agent:
- Does **not write code** (needs no linting/naming/testing conventions)
- Does **not choose providers** (provider resolution is runtime)
- Does **not add phases** (needs no "adding new phase" instructions)
- Already gets work_id, description, work_type, feedback, spec_dir via `read_work_context`

### Current Code
```python
# spine/agents/skills_resolver.py line 128
_SKIP_AGENTS_MD: set[str] = {PhaseName.TASKS.value, PhaseName.CRITIC.value}
```

### Fix
```python
_SKIP_AGENTS_MD: set[str] = {
    PhaseName.SPECIFY.value,
    PhaseName.TASKS.value,
    PhaseName.CRITIC.value,
}
```

### Token Savings
**~22,000 chars (~5K tokens)** per specify run, and **~22K chars** removed from the system prompt itself (the prompt file duplicates the AGENTS.md content).

### Risk
Minimal. The specify agent's workflow instructions, tool surface, and output format are all in `_build_specify_prompt()` (lines 1-167 of the prompt file), which is separate from AGENTS.md injection.

---

## Change B: Rewrite Negative Instructions as Positive Guidance

### Problem
The prompt contains ~30 negative instructions ("Do NOT", "Never", "Do not", "You do NOT have"). Quantized 30B models respond better to positive, constructive instructions that tell them what **to** do rather than what **not** to do.

### Detailed Rewrite Table

#### Section: Tool Surface (lines 3-9)

| Original | Improved |
|----------|----------|
| `You do NOT have ls, read_file, glob, grep, write_file, edit_file, or execute. Do not attempt to call them — they do not exist in your session.` | `Your available tools are: read_work_context, write_specification, task (researcher subagent dispatch), and eval (JavaScript REPL). Use only these tools. Codebase exploration is performed by researcher subagents.` |

#### Section: Core Behaviour (lines 66-69)

| Original | Improved |
|----------|----------|
| `Never say "I'll now do X" — just do it.` | `Begin every response with a tool call. Do not include preamble text like "I will now..." or "Let me..." before tool calls.` |
| `Your first attempt is rarely correct — iterate.` | `If a tool call result is empty or unsatisfactory, revise your approach with more specific file paths or investigation criteria, then retry.` |

#### Section: Interpreter Environment (lines 77-84)

| Original | Improved |
|----------|----------|
| 6 items starting with `require() — no module system`, `import/export — no ES modules`, `fs — no filesystem access`, `process — no Node.js process`, `window — use globalThis`, `fetch — no network` | `Available QuickJS APIs: globalThis (persistent state), console.log (output), Promise, async/await, JSON, globalThis.tools (tool bindings).` <br>`Note: The sandbox has no module system, no filesystem, no network, and no process/window objects.` |

#### Section: Tools (lines 97-103)

| Original | Improved |
|----------|----------|
| `Context is L1 cache; conversation history is swap. ... Never re-read a file in the same phase.` | `Check the read cache before reading files — the runtime context stores a metadata summary of every file read. Use cached line counts and symbol names to determine if a file is already available.` |

#### Section: Workflow Context (lines 114-115)

| Original | Improved |
|----------|----------|
| `Do NOT ask follow-up questions — work with the context you are given.` | `Execute the phase objective using only the provided work description, feedback, and prior specification. Do not request additional information.` |
| `Do NOT seek user approval — execute autonomously within your phase scope.` | `Complete the phase artifact autonomously. Your output will be reviewed by the critic phase automatically.` |

#### Section: PTC Note (line 226)

| Original | Improved |
|----------|----------|
| `Do NOT call require() or access fs. Do NOT use old filesystem tool names like ls, glob, grep, readFile, writeFile — they do not exist in PTC on orchestrator phases.` | `Use PTC tools: tools.read_work_context, tools.write_specification, tools.task. All tool names are camelCase and available via globalThis.tools.` |

---

## Change C: Remove Irrelevant Prompt Sections

### Section: write_todos (lines 128-142)
**Why remove**: The specify agent has a rigid 3-turn workflow (read → dispatch → write). Dynamic task planning is unnecessary.

**Fix**: Remove the entire section. Add `skip_todolist_middleware=True` to `build_specify_agent()` in `specify_agent.py`.

### Section: Skills System (lines 144-182)
**Why remove**: The section contains 39 lines of generic skill instructions. At runtime, the skill list shows "(No skills available yet)" — the instructions are dead weight. The skills resolution (`resolve_skills()` in `skills_resolver.py`) already loads spec-writing and rlm-pattern skills if they exist.

**Fix**: Replace with 2 lines:
```
Phase skills are loaded automatically: spec-writing (for specification conventions), rlm-pattern (for interpreter usage).
```

### Section: Generic task tool docs (lines 185-206)
**Why remove**: 22 lines of generic advice about when to use `task` vs `eval`. The specify agent only uses `task` to dispatch researchers — the "when NOT to use" section (lines 202-206) creates confusion.

**Fix**: Replace with 4 lines:
```
Use `task` with `subagent_type: "researcher"` to dispatch parallel codebase research.
Use `eval` to dispatch multiple researchers in a single call via `Promise.allSettled`.
```

### Section: Project Documentation — Pitfalls (lines 493-514)
**Why remove**: After removing AGENTS.md injection, this section disappears. If AGENTS.md is kept for other phases, these pitfalls should be moved to phase-specific docs, not injected into every agent.

### Section: Code Style, Naming, Testing (lines 517-566)
**Why remove**: The specify agent writes specifications, not Python code. Linting rules, naming conventions, and testing patterns are irrelevant.

---

## Change D: Restructure to Hyper-Literal, Numbered Workflow

### Problem
The current workflow section (lines 11-56) mixes step-by-step instructions with philosophical guidance. Quantized 30B models need a clear, numbered sequence: do this, then this, then this.

### Proposed Restructured Workflow

Replace lines 11-56 with:

```markdown
## SPECIFY PHASE — STEP-BY-STEP WORKFLOW

Execute these steps in order. Complete each step fully before proceeding.

### Step 1 — Read Context (Turn 1)
1. Call `read_work_context` with no arguments.
2. Store the JSON result in `globalThis.ctx`.
3. Extract: `ctx.description`, `ctx.feedback`, `ctx.prior_spec`, `ctx.spec_dir`.

### Step 2 — Dispatch Researchers (Turn 2, conditional)
4. Check: Does `ctx.description` contain any file path (e.g. `*.py`, `spine/...`) or any codebase term ("SPINE", "spine", "module", "function", "class")?
5. IF yes → identify 3 codebase areas, then for each area:
   a. Write a task description that includes: work description, specific file paths, 3-4 investigation questions.
   b. Ensure each description is ≥200 characters (count the characters).
   c. Dispatch all researchers in a single `eval` call using `Promise.allSettled`.
   d. Store results in `globalThis.research`.
6. IF no file paths and no codebase terms → set `globalThis.research = null`, proceed to Step 3.

### Step 3 — Write Specification (Turn 3)
7. Synthesize research findings into these 5 required sections:
   - `overview`: 2-3 sentences of what needs to be built.
   - `requirements`: numbered list of functional and non-functional requirements.
   - `architecture`: bullet list of design decisions with rationale.
   - `interfaces`: bullet list of APIs, data models, contracts with types.
   - `success_criteria`: numbered list of 3-5 measurable, verifiable outcomes.
8. Call `write_specification` with all 5 fields. The tool produces specification.md.

### Turn Budget
- Expected: 3 turns. If you exceed 5 turns without calling `write_specification`, check Step 2 for errors.
```

### Benefits for 30B Models
- **Numbered steps** → deterministic execution order
- **If/then conditionals** → replaces ambiguous "skip if trivial"
- **Character counting** → concrete metric instead of "make it detailed"
- **Explicit field lists** → no guesswork about required output structure

---

## Change E: Reduce Researcher Tool Surface

### Problem
The researcher subagent currently sees ~17 tools (12 MCP + 5 fallback). A quantized 30B model will struggle with this choice density — it wastes attention on tools it doesn't need.

### Current Researcher Tools (subagents.py)
```python
_READ_ONLY_TOOLS = ["ls", "read_file", "glob", "grep", "search_codebase"]
# + 12 MCP tools from codebase-index
# Total: ~17 tools
```

### Proposed Researcher Tool Set

```python
# Priority-ordered tool list for researcher subagents
_RESEARCHER_ESSENTIALS = [
    "mcp_codebase-index_find_symbol",           # Where is X defined?
    "mcp_codebase-index_get_function_source",   # What does function X look like?
    "mcp_codebase-index_get_dependencies",      # What does X call?
    "mcp_codebase-index_get_dependents",        # What calls X?
    "mcp_codebase-index_search_codebase",       # Find pattern X everywhere
]

# Minimal fallback (content-level search when MCP doesn't cover)
_RESEARCHER_FALLBACK = [
    "search_codebase",  # Keyword search with content previews
    "read_file",        # Last resort — read specific file contents
]
```

### Researcher Workflow Guidance

Replace lines 115-165 of the researcher prompt with:

```markdown
## Research Tools (use in this order)

1. **Always start with:** `mcp_codebase-index_get_project_summary` (orientation, no arguments)
2. **To find symbols:** `mcp_codebase-index_find_symbol` with `{"name": "symbol_name"}`
3. **To see code:** `mcp_codebase-index_get_function_source` or `get_class_source`
4. **To trace relationships:** `get_dependencies` (what a symbol calls) or `get_dependents` (what calls a symbol)
5. **For content search:** `mcp_codebase-index_search_codebase` with regex pattern
6. **Fallback only** (when MCP doesn't find what you need): `search_codebase` or `read_file`

**Batch rule:** If you need to look up 3 functions, call all 3 `get_function_source` in ONE response. Do NOT wait for each result before issuing the next call.

**Hard limit:** Maximum 5 MCP tool calls in your first turn. If you can't cover your research scope in 5 calls, prioritize the most important symbols and use `search_codebase` for the rest.
```

### Token Savings
- Removes 12 lines of tool descriptions from researcher prompt (~800 tokens)
- Reduces cognitive load: 17 options → 5 essential + 2 fallback

---

## Change F: Remove MCP Tool List from Specify Agent Prompt

### Problem
Lines 123-126 of the specify prompt lists MCP tools ("mcp_codebase-index_get_project_summary, ... and 8 more"). The specify orchestrator agent **never uses MCP tools directly** — only researcher subagents do. This creates confusion: the specify agent sees MCP tools in its prompt but has no way to call them.

### Fix
Remove lines 123-126 entirely. The MCP tool list belongs only in the researcher subagent prompt (`SUBAGENT_PROMPTS["researcher"]` in `subagents.py`).

### Token Savings
**~500 tokens** (1 line of MCP tool names + "and 8 more")

---

## Change G: Extend ResearchFindings Schema

### Problem
The current `ResearchFindings` model (subagents.py lines 45-55) has 4 fields: `summary`, `patterns`, `file_map`, `dependencies`. This misses information that would directly help spec synthesis.

### Proposed Extension

```python
class ResearchFindings(BaseModel):
    summary: str = Field(description="Concise summary of findings (2-3 paragraphs)")
    patterns: list[str] = Field(description="Notable patterns, conventions, or idioms discovered")
    file_map: dict[str, str] = Field(description="Mapping of important file paths to brief descriptions")
    dependencies: list[str] = Field(description="Key dependencies, imports, or external services found")
    relevant_conventions: list[str] = Field(
        default=[],
        description="Coding conventions, naming patterns, or architectural patterns found in existing code"
    )
    implementation_risks: list[str] = Field(
        default=[],
        description="Potential risks, edge cases, or known issues discovered during research"
    )
    existing_extensions: list[str] = Field(
        default=[],
        description="Existing implementations, extensions, or similar features that the new spec should build on"
    )
```

### Researcher Prompt Update
Add to the researcher system prompt (after line 200):
```markdown
### Additional Fields to Report
- `relevant_conventions`: List any coding conventions you observed (e.g., "PhaseConfig uses snake_case", "Error handling uses TransientAPIError")
- `implementation_risks`: Note any risks or edge cases (e.g., "Shared mutable state in X could cause race conditions")
- `existing_extensions`: Note similar features or extensions that already exist (e.g., "A similar phase exists in Y")
```

### Effort
- `spine/agents/subagents.py`: Add 3 fields to `ResearchFindings` (~30 lines)
- `subagents.py`: Update researcher prompt (~10 lines)

---

## Change H: Add Structured specification.json Output

### Problem
`write_specification` currently writes only `specification.md` (markdown). The downstream plan phase must parse this markdown to extract requirements, architecture, and interfaces. This is fragile and adds cognitive load.

### Solution
Mirror the plan phase's dual-output pattern (`plan.md` + `plan.json`):

```python
class WriteSpecificationTool(BaseTool):
    name: str = "write_specification"
    
    def _run(overview, requirements, architecture, interfaces, success_criteria, open_questions="") -> str:
        # ... existing markdown write ...
        
        # NEW: Also write specification.json
        spec_json = {
            "overview": overview.strip(),
            "requirements": [r.strip() for r in requirements.strip().split("\n") if r.strip()],
            "architecture": architecture.strip(),
            "interfaces": interfaces.strip(),
            "success_criteria": [s.strip() for s in success_criteria.strip().split("\n") if s.strip()],
            "open_questions": open_questions.strip() if open_questions.strip() else [],
            "research_dependencies": dependencies_from_context,  # optional
        }
        json_path = output.with_suffix(".json")
        json_path.write_text(json.dumps(spec_json, ensure_ascii=False, indent=2))
```

### Downstream Benefits
- Plan phase's `plan_resolver.py` can parse JSON instead of markdown regex
- Structured requirements → easier validation in VERIFY phase
- Machine-parseable spec → better downstream phase planning

### Effort
- `spine/agents/specify_tools.py`: Add JSON output to `WriteSpecificationTool` (~30 lines)
- `spine/work/plan_resolver.py`: Add JSON parsing path alongside markdown parsing (~1 hour)

---

## Change I: Add Intermediate ResearchBrief Artifact

### Problem
Currently: researcher results (raw JSON) → specify agent receives all 3-4 research findings simultaneously → agent must read, compare, synthesize all of them. This creates a large context window requirement and cognitive load.

### Solution
Add a structured intermediate artifact: `ResearchBrief`. This is generated by the specify agent after receiving research results but before writing the spec. It compresses the findings into a focused brief.

```python
class ResearchBrief(BaseModel):
    research_scope: str
    key_findings: list[str]  # Top 5-10 findings across all researchers
    relevant_conventions: list[str]  # Consolidated from all researchers
    file_references: dict[str, str]  # Consolidated file_map
    open_questions: list[str]  # Gaps identified by research
    risk_flags: list[str]  # Consensus risks across researchers
```

### Workflow Change

```
Step 2: Dispatch researchers → receive 3-4 ResearchFindings
Step 2.5: Synthesize into ResearchBrief (1 turn)
  - Merge patterns from all researchers
  - Extract consensus conventions
  - Identify gaps that need clarification
Step 3: Write specification using ResearchBrief as primary reference
```

### Effort
- `spine/agents/subagents.py`: Add `ResearchBrief` model (~50 lines)
- `spine/agents/specify_tools.py`: Add `consolidate_research()` tool (~1 hour)
- `docs/specify_synth_sys_prompt.md`: Add Step 2.5 to workflow (~20 lines)

---

## Implementation Order

### Sprint 1: Low-Hanging Fruit (30 min - 2 hours)
1. **Change A**: Add SPECIFY to `_SKIP_AGENTS_MD` — 1 line
2. **Change F**: Remove MCP list from specify prompt — 1 line
3. **Change C (partial)**: Remove write_todos, replace skills section — 2 patches
4. **Change G**: Extend ResearchFindings — 3 new fields

### Sprint 2: Prompt Restructure (2-4 hours)
5. **Change B**: Rewrite all negative instructions — patch entire prompt
6. **Change D**: Restructure to numbered workflow — replace lines 11-56
7. **Change C (complete)**: Remove generic task docs, pitfalls, code style sections

### Sprint 3: Structured Data Flow (3-6 hours)
8. **Change H**: Add specification.json output — extend WriteSpecificationTool
9. **Change I**: Add ResearchBrief intermediate artifact — new model + tool
10. **Change E**: Reduce researcher tool surface — update subagent prompt

### Sprint 4: Integration (1-2 hours)
11. Update `plan_resolver.py` to prefer `specification.json` when available
12. Test the full specify → plan flow with the new structured outputs
13. Update any test fixtures that reference the old specification format

---

## Expected Outcomes

### Token Reduction
| Change | Tokens Saved |
|--------|-------------|
| A: Remove AGENTS.md injection | ~22,000 chars (~5,000 tokens) per run |
| B: Rewrite negative → positive | ~0 (same size) |
| C: Remove irrelevant sections | ~2,500 tokens |
| F: Remove MCP list | ~500 tokens |
| E: Reduce researcher prompt | ~800 tokens |
| **Total** | **~26,000 chars (~5,800 tokens) per specify run** |

### Quality Improvements
- **30 negative instructions** rewritten to positive guidance
- **4 conflicting rules** resolved (researcher count, "trivial" definition, skill availability, task tool scope)
- **Structured intermediate artifact** (ResearchBrief) reduces cognitive load
- **Dual-format spec output** (markdown + JSON) improves downstream phase reliability
- **Numbered workflow** with explicit if/then conditionals replaces ambiguous prose

### Model-Specific Impact
For quantized 30B models:
- **Hyper-literal workflow** (numbered steps, explicit conditionals) replaces philosophical guidance
- **Reduced tool surface** (researcher: 17 → 7 tools) reduces choice confusion
- **Structured inputs** (extended ResearchFindings) reduce the need for the model to infer missing information
- **~5,800 fewer prompt tokens** means more context window for actual research findings and synthesis
