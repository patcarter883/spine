# Structured I/O Expansion: IMPLEMENT & VERIFY Phases

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Expand the manager/router/call pattern (already in SPECIFY and PLAN) to IMPLEMENT and VERIFY
phases, replacing all agent-driven markdown-parsing with structured JSON I/O for feature slices and
implementation results.

**Architecture:** Each phase gets a self-contained data contract. PLAN writes `plan.json` (already done).
IMPLEMENT reads `plan.json` → writes `implementation.json` (new). VERIFY reads `plan.json` +
`implementation.json` → writes `verification.json` (new). The markdown artifacts remain as
human-readable companions, but agents never parse them — they consume only JSON.

**Tech Stack:** Python 3.12+, Pydantic v2, LangChain BaseTool, LangGraph subgraphs.

---

## Current State Analysis

The "manager, router, call" pattern consists of three layers:

1. **Manager** — Phase agent builder (e.g., `build_specify_agent()`). Defines a 3-step
   workflow: read context → dispatch subagents → write structured output.
2. **Router Tools** — Purpose-built LangChain tools that replace generic filesystem access.
   Each phase has exactly: one "read context" tool (loads everything in one call) and one
   "write output" tool (accepts structured Pydantic model, writes to fixed path).
3. **Call** — Parallel subagent dispatch via `eval` + `tools.task()` + `Promise.allSettled`.

| Phase | Manager | Router (Read) | Router (Write) | Call | Structured? |
|-------|---------|---------------|----------------|------|-------------|
| SPECIFY | ✅ | `read_work_context` | `write_specification` | researcher subagents | ✅ JSON |
| PLAN | ✅ | `read_prior_artifacts` + `search_codebase` | `write_structured_plan` | researcher subagents | ✅ `plan.json` |
| IMPLEMENT | ✅ | `read_slice_files` | `write_implementation_report` | slice-implementer subagents | ⚠️ Partial |
| VERIFY | ✅ | `read_verify_context` | `write_verification_report` | slice-verifier subagents | ❌ Markdown |

### Specific Gaps

**IMPLEMENT gaps:**
- `read_slice_files` reads `plan.json` in wave mode but falls back to `slice-*.md` markdown in legacy mode
- Subgraph prompt (`implement_subgraph.py`) references stale markdown paths the agent cannot read
- No `implementation.json` output — downstream phases must parse `implementation.md`
- Slice-implementer subagents receive raw markdown text in `task.description`, not structured data

**VERIFY gaps (more severe):**
- `read_verify_context` reads `slice-*.md` markdown files from `tasks/` — the primary markdown-parsing
  antipattern the user wants eliminated
- Subgraph prompt (`verify_subgraph.py`) references `read_file` and `grep` tools the orchestrator
  doesn't even have (it uses `read_verify_context`)
- No structured implementation results input — reads raw `implementation.md`
- Slice-verifier subagents receive ALL context (slice + codebase_map + implementation)
  as raw text in `task.description`, causing context bloat

---

## Design: Structured Data Contracts

### Data Flow (post-expansion)

```
PLAN ──[plan.json]──▶ IMPLEMENT ──[implementation.json]──▶ VERIFY
         feature_slices              slice_results             verification_results
         codebase_map                (per-slice status,        (per-slice verdict,
                                      files, tests,             checklist, gaps,
                                      issues)                   recommendations)
```

Each phase reads the upstream JSON, not markdown. Markdown files remain as human companions
but are never consumed by agent code.

### New/Modified Schemas

**`implementation.json`** (new — IMPLEMENT output, VERIFY input):
```json
{
  "summary": "string",
  "slice_results": [
    {
      "slice_name": "add-user-model",
      "status": "implemented|partial|blocked",
      "files_modified": ["spine/models/user.py"],
      "files_created": ["spine/models/__init__.py"],
      "test_results": "3 tests passed, ruff clean",
      "issues": []
    }
  ]
}
```

**`verification.json`** (new — VERIFY output):
```json
{
  "summary": "string",
  "overall_status": "VERIFIED|FAILED",
  "verification_results": [
    {
      "slice_name": "add-user-model",
      "verdict": "VERIFIED|NOT_VERIFIED",
      "checklist": [{"criterion": "string", "passed": true, "detail": "string"}],
      "gaps": [],
      "recommendations": []
    }
  ]
}
```

---

## Implementation Plan

---

### Task 1: Add `implementation.json` write side to WriteImplementationReportTool

**Objective:** Make the IMPLEMENT orchestrator's write tool also emit `implementation.json`
alongside `implementation.md`.

**Files:**
- Modify: `spine/agents/implement_tools.py:176-288`

**Step 1: Add JSON serialization to `WriteImplementationReportTool._run`**

After writing `implementation.md`, also write `implementation.json` with the same structured data:

```python
# Build implementation.json data
json_data = {
    "summary": summary,
    "slice_results": [
        {
            "slice_name": name,
            "status": status,
            "files_modified": modified,
            "files_created": created,
            "test_results": test_res,
            "issues": issues,
        }
        for name, status, modified, created, test_res, issues in ...
    ],
}
json_path = impl_path / "implementation.json"
json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
```

**Step 2: Update return message**

Append ` + implementation.json ({n} bytes)` to the return string.

**Verification:** Run existing implement tool tests, verify `implementation.json` is written alongside `implementation.md`.

---

### Task 2: Add `verification.json` write side to WriteVerificationReportTool

**Objective:** Mirror Task 1 for the VERIFY phase.

**Files:**
- Modify: `spine/agents/verify_tools.py:217-310`

**Step 1: Add JSON serialization to `WriteVerificationReportTool._run`**

After writing `verification.md`, also write `verification.json`:

```python
verify_json_path = verify_path / "verification.json"
json_data = {
    "summary": summary,
    "overall_status": overall_status,
    "verification_results": [
        {
            "slice_name": name,
            "verdict": verdict,
            "checklist": checklist,
            "gaps": gaps,
            "recommendations": recommendations,
        }
        for name, verdict, checklist, gaps, recommendations in ...
    ],
}
verify_json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
```

**Verification:** Run existing verify tool tests, verify `verification.json` is written.

---

### Task 3: Remove legacy pathway + make `execution_waves` a Fail-Closed invariant

**Objective:** Eliminate the legacy `slice-*.md` fallback AND hook the `execution_waves`
presence check into the existing Fail-Closed prerequisite gate system. If IMPLEMENT runs
without `execution_waves`, it's a structural invariant violation — halt at `needs_review`.

**Architecture:** The prerequisite gate (`_check_plan_prerequisite`) is the single check-point.
`build_implement_agent()` no longer has a legacy fallback — by the time IMPLEMENT starts,
`execution_waves` is guaranteed by the gate. The agent builder simply reads it from state.

**Files:**
- Modify: `spine/workflow/artifact_gate.py:578-600` (`_check_plan_prerequisite`)
- Modify: `spine/agents/implement_agent.py:97-165` (remove legacy path from builder)
- Modify: `spine/agents/implement_agent.py:171-299` (remove `_build_legacy_orchestrator_prompt`, `_build_orchestrator_prompt` with `has_waves`, `_common_prompt_header`, `_prompt_step_1_template`, `_prompt_step_3_template`)
- Modify: `spine/workflow/subgraphs/implement_subgraph.py:109-127` (remove has_waves branch in subgraph prompt — always wave-based)
- Modify: `spine/models/state.py:109` (add `execution_waves_present: bool` to WorkflowState)

**Step 1: Extend `_check_plan_prerequisite` to also check `execution_waves`**

```python
def _check_plan_prerequisite(state: WorkflowState) -> tuple[bool, str]:
    plan_completed = state.get("plan_completed", False)
    execution_waves = state.get("execution_waves", [])
    
    if not plan_completed:
        return (False,
            "IMPLEMENT phase requires PLAN to have completed successfully. "
            "The plan artifact is missing or the PLAN phase did not finish. "
            "Re-run PLAN or resolve the prior failure before proceeding.")
    
    if not execution_waves or len(execution_waves) == 0:
        return (False,
            "IMPLEMENT phase requires structured execution_waves from the PLAN phase. "
            "The PLAN phase completed but did not produce feature_slices with "
            "execution wave scheduling. Re-run PLAN — the plan agent must call "
            "`write_structured_plan` to produce plan.json with feature_slices.")
    
    return True, ""
```

**Step 2: Remove legacy branch from `build_implement_agent`**

Remove the entire `else:` block (lines 123-135) that calls `list_slice_files`. Remove
`tasks_dir` (line 99). Remove the `has_waves` check — it's always True by this point.

Before (123-135):
```python
else:
    # Legacy fallback: discover slice-*.md files from tasks/ dir
    slice_files = list_slice_files(workspace_root, work_id)
    ...
```

After: No `else` branch. If `execution_waves` is empty (shouldn't happen due to gate),
return `needs_review`:

```python
execution_waves = state.get("execution_waves", [])
if not execution_waves:
    # Prerequisite gate should prevent this, but be defensive
    logger.error(f"[{work_id}] IMPLEMENT: execution_waves missing from state — "
                  "prerequisite gate bypassed?")
    return build_phase_agent(
        state=state, config=config, phase=PhaseName.IMPLEMENT,
        system_prompt="Error: No execution waves. Human review required.",
        skip_filesystem_middleware=True,
    )
```

**Step 3: Remove legacy prompt builders**

Delete from `implement_agent.py`:
- `_build_orchestrator_prompt()` (lines 171-185 — the `has_waves` dispatcher)
- `_build_legacy_orchestrator_prompt()` (lines 276-299)
- `_common_prompt_header()` (lines 188-202)
- `_prompt_step_1_template()` (lines 205-223)
- `_prompt_step_3_template()` (lines 226-239)

Inline the wave prompt directly in `build_implement_agent()` — no branching needed.
The `_build_wave_orchestrator_prompt()` becomes the only prompt, renamed to
`_build_implement_prompt()`.

**Step 4: Simplify subgraph prompt**

Remove the `has_waves` branch from `implement_subgraph.py` (lines 118-127). Always include
the wave-based dispatch guidance. The "Read `plan/plan.md`" references become "Your
`read_slice_files` tool loads everything from `plan.json`."

**Step 5: Remove dead code**

- `list_slice_files` import from `implement_agent.py`
- `ReadSliceFilesVerifyTool` from `verify_tools.py` (separate but related dead code)
- Remove `tasks_dir` import and usage

**Verification:**
1. `pytest tests/` — ensure no regression
2. Config test: submit a work item where PLAN completes but produces no `execution_waves` → prerequisite gate should route to `needs_review` (not proceed to IMPLEMENT)
3. `rg '_build_legacy|list_slice_files' spine/` → zero matches

---

### Task 4: Refactor `read_verify_context` to use structured JSON, not markdown

**Objective:** The verify orchestrator currently reads `slice-*.md` markdown files from the tasks
directory. Replace with reading `plan.json` (structured slice definitions) and `implementation.json`
(structured results). This is the **primary fix** for the user's core requirement.

**Files:**
- Modify: `spine/agents/verify_tools.py:38-118` (`ReadVerifyContextTool._run`)
- Modify: `spine/agents/verify_agent.py:91-136` (orchestrator prompt)

**Step 1: Rewrite `ReadVerifyContextTool._run`**

Replace the `slice-*.md` glob loop (lines 96-106) with:

```python
# Load plan.json for structured slice definitions + codebase_map
plan_path = workspace / self.plan_dir / "plan.json"
if plan_path.exists():
    try:
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        # Store structured slices (dicts, not markdown strings)
        for sl in plan_data.get("feature_slices", []):
            if isinstance(sl, dict) and sl.get("id"):
                result["slices"][sl["id"]] = sl  # structured dict, not markdown
    except (json.JSONDecodeError, OSError) as exc:
        result["plan_error"] = str(exc)

# Load implementation.json for structured results (not implementation.md)
impl_json_path = workspace / self.impl_dir / "implementation.json"
if impl_json_path.exists():
    result["implementation"] = json.loads(impl_json_path.read_text(encoding="utf-8"))
```

**Step 2: Update orchestrator prompt**

Change the verify orchestrator prompt to reference structured data, not markdown:

```
### Step 1 — Call read_verify_context
Call `read_verify_context` with no arguments. It returns:
```json
{
  "slices": {"slice-id": {structured slice dict}, ...},
  "codebase_map": "<string>",
  "implementation": {structured results dict}
}
```

### Step 2 — Dispatch Subagents In Parallel (1 eval turn)
Each subagent receives structured data:
```js
const {slices, codebase_map, implementation} = globalThis.verifyContext;
const results = await Promise.allSettled(
  Object.entries(slices).map(([id, slice]) =>
    tools.task({
      subagent_type: 'slice-verifier',
      description: `Verify slice: ${id}\n\n` +
        `Slice Definition: ${JSON.stringify(slice)}\n` +
        `Codebase Map: ${codebase_map}\n` +
        `Implementation Result: ${JSON.stringify(implementation?.slice_results?.find(r => r.slice_name === id) || {})}\n`
    })
  )
);
```

**Step 3: Remove `tasks_dir` from build_verify_orchestrator_tools**

The `ReadVerifyContextTool` no longer reads from `tasks/`. Remove `tasks_dir` from the factory
and `ReadVerifyContextTool` fields. Remove `ReadSliceFilesVerifyTool` entirely (dead code).

**Verification:** Run verify tests. Verify orchestrator reads `plan.json` + `implementation.json`.

---

### Task 5: Fix stale subgraph prompts

**Objective:** The `implement_subgraph.py` and `verify_subgraph.py` user prompts still tell
the agent to read markdown files using `read_file`/`grep` — but the agent doesn't have those
tools. Update the prompts to match the actual tool surface.

**Files:**
- Modify: `spine/workflow/subgraphs/implement_subgraph.py:64-146`
- Modify: `spine/workflow/subgraphs/verify_subgraph.py:66-111`

**Step 1: Fix implement subgraph prompt**

Remove references to `specification.md`, `plan.md`, `tasks.md`, `codebase-map.md`. The
orchestrator's `read_slice_files` tool loads everything it needs from `plan.json`. Replace
the artifact path references with:

```python
prompt_lines = [
    "Implement the feature slices from the plan. Use your `read_slice_files` tool "
    "to load all slice definitions and the codebase map in one call — do not read "
    "individual files manually.",
    "",
]
# Add wave-mode specific guidance if has_waves
```

**Step 2: Fix verify subgraph prompt**

Remove references to `specification.md`, `plan.md`, `tasks.md`, `codebase-map.md`,
`read_file`, `grep`. Replace with:

```python
prompt_lines = [
    "Verify the implementation using your `read_verify_context` tool to load "
    "structured slice definitions and implementation results in one call. "
    "Dispatch a `slice-verifier` subagent per slice, then synthesize results "
    "with `write_verification_report`.",
    "",
]
```

**Verification:** Visual inspection. Run a test workflow to confirm prompts are correct.

---

### Task 6: Update `save_artifacts` in verify subgraph to use structured data

**Objective:** The verify subgraph's `_save_verify_artifacts` currently parses the markdown
`verification.md` to determine `VERIFIED`/`PASSED` status via string matching. After Task 2,
`verification.json` exists — use it for authoritative status.

**Files:**
- Modify: `spine/workflow/subgraphs/verify_subgraph.py:140-201`

**Step 1: Read `verification.json` for status**

Replace the string-matching heuristic (lines 186-187):

```python
# Before: is_verified = "VERIFIED" in verify_text.upper() or "PASSED" in verify_text.upper()
# After: read verification.json
verify_json = Path(workspace_root) / artifact_path(work_id, PhaseName.VERIFY.value) / "verification.json"
is_verified = False
if verify_json.exists():
    vdata = json.loads(verify_json.read_text())
    is_verified = vdata.get("overall_status") == "VERIFIED"
```

**Verification:** Run verify tests. Confirm phase status is correctly determined from JSON.

---

### Task 7: Add tests for new JSON output

**Objective:** Ensure the new `implementation.json` and `verification.json` outputs are correct.

**Files:**
- Create: `tests/agents/test_implement_tools.py` (if not existing, add JSON test)
- Create: `tests/agents/test_verify_tools.py` (if not existing, add JSON test)

**Step 1: Test `write_implementation_report` writes JSON**

```python
def test_write_implementation_report_writes_json(tmp_path):
    from spine.agents.implement_tools import WriteImplementationReportTool
    tool = WriteImplementationReportTool(
        workspace_root=str(tmp_path),
        impl_dir="test/implement",
    )
    result = tool._run(
        slice_results=[
            {"slice_name": "s1", "status": "implemented", "files_modified": ["a.py"],
             "files_created": [], "test_results": "ok", "issues": []},
        ],
        summary="All done",
    )
    assert "implementation.json" in result
    json_path = tmp_path / "test" / "implement" / "implementation.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["summary"] == "All done"
    assert len(data["slice_results"]) == 1
```

**Step 2: Test `write_verification_report` writes JSON**

Same pattern as above, verifying `verification.json`.

**Step 3: Test `read_verify_context` reads structured data**

```python
def test_read_verify_context_reads_plan_json(tmp_path):
    # Set up plan.json
    plan_dir = tmp_path / "test" / "plan"
    plan_dir.mkdir(parents=True)
    plan_dir.joinpath("plan.json").write_text(json.dumps({
        "feature_slices": [{"id": "s1", "title": "Test", ...}],
        "codebase_map": "mapping here",
    }))

    from spine.agents.verify_tools import ReadVerifyContextTool
    tool = ReadVerifyContextTool(
        workspace_root=str(tmp_path), plan_dir="test/plan",
        tasks_dir="", impl_dir="test/implement", verify_dir="test/verify",
    )
    result = json.loads(tool._run())
    assert "s1" in result["slices"]
    assert isinstance(result["slices"]["s1"], dict)  # structured, not string
```

**Verification:** `pytest tests/agents/test_implement_tools.py tests/agents/test_verify_tools.py -v`

---

### Task 8: Clean up dead code

**Objective:** Remove tools and references that are no longer needed after the refactor.

**Files:**
- Modify: `spine/agents/verify_tools.py` (remove `ReadSliceFilesVerifyTool`, lines 121-167)
- Modify: `spine/agents/implement_agent.py` (remove `_build_legacy_orchestrator_prompt`)
- Modify: `spine/agents/verify_tools.py` (remove `ReadSliceFilesVerifyInput`, unused imports)

**Step 1: Remove `ReadSliceFilesVerifyTool`**

This was an alternate tool for reading `slice-*.md` from tasks/. After Task 4, it's dead code.

**Step 2: Remove legacy imports**

Remove `list_slice_files` import from `verify_tools.py` if it was only used by the removed tool.

**Step 3: Remove `_build_legacy_orchestrator_prompt`**

After Task 3, the legacy dispatch path is gone. Remove the function and its prompt template parts.

**Verification:** `rg "ReadSliceFilesVerifyTool|_build_legacy|list_slice_files" spine/` — zero matches.

---

### Task 9: Integration test — full workflow with structured I/O

**Objective:** Run a complete SPECIFY → PLAN → IMPLEMENT → VERIFY workflow and verify
all JSON artifacts are generated and consumed correctly.

**Files:**
- Create: `tests/integration/test_structured_io_flow.py`

**Step 1: Write integration test**

```python
@pytest.mark.asyncio
async def test_full_flow_structured_io(tmp_workspace, mock_model):
    """SPECIFY writes spec.json, PLAN writes plan.json, IMPLEMENT writes
    implementation.json and reads plan.json, VERIFY writes verification.json
    and reads plan.json + implementation.json."""
    # Mock all LLM calls, verify tool schemas are correct at each phase
    ...
```

**Verification:** `pytest tests/integration/test_structured_io_flow.py -v`

---

## Key Pattern: LangGraph Send API (from exploration_subgraph.py)

The SPECIFY/PLAN exploration subgraph already uses the superior pattern for parallel subagent dispatch:

```
manager → router → [Send(node, state), ...] → parallel execution → aggregate
```

**How it works:**
1. `research_manager_node` — single LLM call (no tools, no loop) → `{"decision": "explore", "topics": [...]}`
2. `_research_router(state)` — returns `[Send("explore", {"topic": t}) for t in topics]` OR `"synthesize"`
3. LangGraph executes all `Send` targets in **parallel within the same super-step**
4. Results accumulated via `operator.add` reducer on the `findings` field
5. `_aggregate_node` — fan-in checkpoint, then `sufficiency_router` → loop or synthesize

**This is what IMPLEMENT and VERIFY should become:**

| Phase | Manager | Router | Parallel Nodes | Aggregation |
|-------|---------|--------|----------------|-------------|
| SPECIFY | `build_specify_agent()` | `task` + `eval` (current, suboptimal) | researcher subagents | implicit via prompt |
| PLAN | `build_plan_agent()` | `task` + `eval` (current, suboptimal) | researcher subagents | implicit via prompt |
| IMPLEMENT | **Should use Send API** | `_implement_router(state)` → `[Send("slice-implementer", slice)]` | `run_slice_implementer` nodes | `operator.add` on results |
| VERIFY | **Should use Send API** | `_verify_router(state)` → `[Send("slice-verifier", slice)]` | `run_slice_verifier` nodes | `operator.add` on results |

**Critical differences from `eval`+`task` pattern:**
- **Send API** is native LangGraph — no JavaScript, no middleware
- **Deterministic parallel fan-out** — explicit graph structure
- **Type-safe state injection** — each slice is injected as structured state
- **No string parsing** — `task.description` string concatenation antipattern eliminated

| Phase | Manager | Router (Read) | Router (Write) | Call Pattern |
|-------|---------|---------------|----------------|--------------|
| SPECIFY | `build_specify_agent()` | `read_work_context` → `task` → `write_specification` | `task` + `eval` + `Promise.allSettled` | ✅ JSON |
| PLAN | `build_plan_agent()` | `read_prior_artifacts` + `search_codebase` → `write_structured_plan` | `task` + `eval` + `Promise.allSettled` | ✅ JSON |
| IMPLEMENT | `build_implement_agent()` | `read_slice_files` (reads `plan.json`) → `task` → `write_implementation_report` + `implementation.json` | `task` + `eval` + `Promise.allSettled` | ✅ **after fix** |
| VERIFY | `build_verify_agent()` | `read_verify_context` (reads `plan.json` + `implementation.json`) → `task` → `write_verification_report` + `verification.json` | `task` + `eval` + `Promise.allSettled` | ✅ **after fix** |

**Critical Change:** Subagents receive structured JSON in `task.description`, not markdown strings. The orchestrator passes:

```js
// Before (markdown antipattern):
description: `## Slice Definition\n${slices[name]}\n\n## Codebase Map\n${codebase_map}...`

// After (structured JSON):
description: `Slice ID: ${id}\n\nSlice Data: ${JSON.stringify(slice)}\n\nCodebase Map: ${codebase_map}\n\nImplementation: ${JSON.stringify(impl_results?.find(r => r.slice_name === id))}`
```

**Files touched:** `implement_tools.py`, `verify_tools.py`, `implement_agent.py`,
`verify_agent.py`, `implement_subgraph.py`, `verify_subgraph.py`, + test files.

**Zero-breaking:** All existing markdown artifacts (`plan.md`, `implementation.md`,
`verification.md`) continue to be written alongside the new JSON files. The markdown
files remain for human review; agents consume only JSON.

---

## Summary of Changes Table

| # | What | Impact |
|---|------|--------|
| 1 | `WriteImplementationReportTool` writes `implementation.json` | New structured output |
| 2 | `WriteVerificationReportTool` writes `verification.json` | New structured output |
| 3 | `_check_plan_prerequisite` checks `execution_waves`; remove legacy path | Fail-Closed invariant |
| 4 | `read_verify_context` uses `plan.json` + `implementation.json` | **Core fix** — no markdown |
| 5 | Fix stale subgraph prompts | Correct tool surface |
| 6 | `_save_verify_artifacts` uses `verification.json` | Deterministic status |
| 7 | Tests for new JSON output | Regression safety |
| 8 | Remove dead code | Cleaner codebase |
| 9 | Integration test | End-to-end verification |

**Total: ~9 tasks, ~45-90 min estimated.**
