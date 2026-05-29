# Plan-quality fixes: push critic checks upstream + give the critic the spec it needs

> **Status:** PROPOSED — not yet implemented. Saved here so it can be reviewed
> from another machine. To execute, re-enter plan mode locally and approve, or
> just hand it to the next session as "implement the levers in
> `.spine/reviews/plan-quality-fixes-proposal.md`".

## Context

Trace `019e726f` (a `task` workflow that SHOULD have reached IMPLEMENT/VERIFY) terminated at the critic-plan gate after **3 plan rework cycles**, all rejected with substantively similar feedback. The goal: improve plan quality without increasing token burn.

Investigation surfaced **three compounding issues**:

1. **The critic reviewing PLAN never receives the SPEC.** `_build_review_prompt` in `spine/workflow/critic_review.py:41-68` inlines only `plan.json` + the original user description. The critic agent is built with `allowed_tools=[]` + `skip_filesystem_middleware=True` (`spine/critic/agent.py:151-156`), so it can't read `specification.json` from disk either. But the critic's system prompt at `_PLAN_REVIEW_INSTRUCTIONS:48-58` explicitly instructs it to compare each slice's `target_files` / `execution_requirements` against the spec's `scope_inclusions` / `scope_exclusions` — fields it never sees. The critic hallucinates that those fields belong on the plan and rejects on phantom scope-creep concerns. Synthesizer rework can't fix this because `_StructuredWritePlanInput` (`spine/agents/plan_tools.py:392`) has no scope fields by design.

2. **The SPECIFY synthesizer often leaves soft fields empty.** Direct artifact inspection:
   - `019e726f` (failed 3×): `scope_inclusions=[]`, `scope_exclusions=[]`, `constraints=[]`, `known_risks=[]` — all empty
   - `019e721d` (`7d2143d7`): same — all empty
   - `019e723c` (`3c400fa6`, passed): all populated with substantive content
   - Older `e97b6df7` (passed): all populated
   - Roughly half the SPECIFY runs are producing empty scope/constraint fields. Even fixing issue #1, the critic would correctly reject plans where the spec offers nothing to validate against.

3. **`write_structured_plan` has no pre-write validation** beyond Pydantic shape. The critic LLM is currently the only line of defense against dependency-graph problems (unknown IDs, cycles, duplicates) — even though `validate_feature_slices` in `spine/workflow/slice_scheduler.py:31` already implements all of these checks deterministically and is currently only called downstream of the critic when computing execution waves.

**Outcome:** push as many checks UPSTREAM as possible (cheap tool-loop self-correction in the same agent invocation) and give the critic the data its prompt already tells it to use. Same models, same workflows, dramatically fewer cross-phase rework cycles → fewer LLM calls → fewer tokens.

## Approach

Three surgical fixes. Each is small. They compound.

- **A. Give the PLAN critic the spec it's instructed to compare against.** Refactor `_build_review_prompt` through the existing XML-tagged hostage-layout helpers (`spine.agents.prompt_format`), accept an optional `spec_payload`, and have `agent_critic_check` pull `state.get("specification_json")` when `reviewed_phase == "plan"`. ~+2 K tokens per critic call, saves ~5 K per avoided rework cycle.
- **B. Pre-validate the plan inside `write_structured_plan`.** Call the existing `validate_feature_slices()` from `slice_scheduler.py` before writing. On failure, return a tool-error string the synthesizer's agent loop catches and self-corrects against — within the SAME invocation. Catches dependency-ID typos, cycles, duplicate slice IDs, empty required fields. Eliminates whole categories of critic-LLM rejection.
- **D. Pre-validate the spec inside `write_specification`.** For non-trivial work descriptions, require `scope_inclusions` and `scope_exclusions` to be non-empty. (Trivial heuristic: description length ≥ 200 chars or contains "implement"/"design"/"refactor"/"build" — matches the existing SPECIFY critic's proportionality rule at `_SPECIFY_REVIEW_INSTRUCTIONS:78-86`.) Return tool-error → synthesizer self-corrects. Addresses the root cause for ~50 % of empty-scope specs.

**Critically:** every prompt-change touches the `prompt_format` helpers (Tag enum + `xml_blocks` + `hostage_layout`) per the project convention shipped in `9300a28`. The current `_build_review_prompt` predates that refactor and uses raw markdown headers — refactoring it is part of lever A.

## Changes

### A — Critic receives spec in PLAN-review

**File:** `spine/workflow/critic_review.py`

1. `_build_review_prompt` (line 41) — add optional `spec_payload: str | None = None` param. Refactor body to use `hostage_layout` + `xml_blocks`:

```python
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks

def _build_review_prompt(
    *,
    reviewed_phase: str,
    structured_payload: str,
    description: str,
    spec_payload: str | None = None,
) -> str:
    return hostage_layout(
        xml_blocks(
            (Tag.OBJECTIVE, description),
            (Tag.SPECIFICATION, spec_payload or ""),  # PLAN review only
            (Tag.FINDINGS, f"```json\n{structured_payload}\n```"),
        ),
        (
            f"Review the {reviewed_phase}-phase structured output above. "
            "Do not attempt to read files or run commands; everything you "
            "need is in the tagged blocks. Respond with PASSED, "
            "NEEDS_REVISION, or NEEDS_REVIEW and include concrete reasons "
            "and suggestions."
        ),
    )
```

2. `agent_critic_check` (line 143) — when `reviewed_phase == PhaseName.PLAN.value`, pull the spec via `state.get("specification_json")` and thread it through:

```python
spec_payload: str | None = None
if reviewed_phase == PhaseName.PLAN.value:
    spec_payload = state.get("specification_json")

prompt = _build_review_prompt(
    reviewed_phase=reviewed_phase,
    structured_payload=structured_payload,
    description=state.get("description") or "",
    spec_payload=spec_payload,
)
```

3. Tighten `_PLAN_REVIEW_INSTRUCTIONS` in `spine/critic/agent.py` — replace the orphan reference to `scope_inclusions` with explicit mention of the `<specification>` block ("The user message includes the spec in a `<specification>` block — pull the lists from there").

### B — Pre-validate the plan inside the write tool

**File:** `spine/agents/plan_tools.py` — `StructuredWritePlanTool._run` (line 456)

After the Pydantic coercion of `validated_slices` (~line 470) and BEFORE the file writes (~line 526), call `validate_feature_slices`. On `ValueError`, return a tool-error string so the synthesizer's agent loop sees it and retries the tool call with corrected inputs (within the same `agent.ainvoke`):

```python
from spine.workflow.slice_scheduler import validate_feature_slices
from spine.models.types import FeatureSlice as _SchedFeatureSlice
from dataclasses import asdict

# Pydantic → dataclass shape the scheduler validator expects.
scheduler_slices = [
    _SchedFeatureSlice(
        id=sl.id,
        title=sl.title,
        target_files=list(sl.target_files),
        execution_requirements=sl.execution_requirements,
        dependencies=list(sl.dependencies),
        acceptance_criteria=list(sl.acceptance_criteria),
        complexity=sl.complexity,
    )
    for sl in validated_slices
]
try:
    validate_feature_slices(scheduler_slices)
except ValueError as exc:
    return (
        f"VALIDATION_ERROR: plan rejected before writing.\n{exc}\n"
        "Fix the structural issue and call write_structured_plan again."
    )
```

The downstream `_compute_waves` in `spine/workflow/subgraphs/exploration_subgraph.py:873` already calls `compute_execution_waves` (which calls the same validator) — so this only pulls EXISTING validation upstream. No new validation logic.

### D — Pre-validate the spec inside `write_specification`

**File:** `spine/agents/specify_tools.py` — `WriteSpecificationTool._run` (line 187)

Mirror lever B's shape. Before constructing the `Specification` model, check non-triviality and reject if scope fields are empty:

```python
def _is_nontrivial_description(description: str) -> bool:
    """Same heuristic as _SPECIFY_REVIEW_INSTRUCTIONS proportionality rule."""
    if len(description) > 200:
        return True
    keywords = ("implement", "design", "refactor", "rebuild", "architect", "build")
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in keywords)


# Inside _run, after collecting inputs:
if _is_nontrivial_description(self.description):
    missing: list[str] = []
    if not (scope_inclusions and any(s.strip() for s in scope_inclusions)):
        missing.append("scope_inclusions")
    if not (scope_exclusions and any(s.strip() for s in scope_exclusions)):
        missing.append("scope_exclusions")
    if missing:
        return (
            f"VALIDATION_ERROR: specification rejected before writing.\n"
            f"For non-trivial work ('{self.description[:80]}…'), the "
            f"following fields MUST be non-empty: {missing}. "
            f"Without these, the downstream PLAN critic cannot perform "
            f"scope-creep validation and will reject every plan. "
            f"Re-call write_specification with these fields populated."
        )
```

(`self.description` is already an injected field on `WriteSpecificationTool` — confirmed in `specify_tools.py` near the BaseTool class definition; if not, thread it through `build_specify_orchestrator_tools` the same way `description` flows.)

### Tests

New unit-test file `tests/unit/test_critic_plan_includes_spec.py` (lever A):
- `test_plan_critic_prompt_includes_spec_block_in_hostage_layout` — stub `agent_critic_check`'s model invocation, drive a plan review with `specification_json` on state, assert the user message has `<specification>`, `<objective>`, `<findings>` tags via `parse_tags` and `assert_hostage_layout`.
- `test_plan_critic_prompt_omits_spec_when_absent` — same but state.specification_json is empty/None → no `<specification>` block but `<objective>` + `<findings>` still present.
- `test_specify_critic_prompt_does_not_inject_spec` — sanity: SPECIFY critic still works (it's reviewing the spec itself; no inject path).

Extend `tests/unit/test_structured_write_plan_tool.py` (or create if absent) (lever B):
- `test_write_structured_plan_rejects_unknown_dependency` — feature_slices reference a non-existent ID → returns `VALIDATION_ERROR`, no files written.
- `test_write_structured_plan_rejects_cycle` — A depends on B, B depends on A → `VALIDATION_ERROR`.
- `test_write_structured_plan_rejects_duplicate_ids` — same ID twice.
- `test_write_structured_plan_writes_when_valid` — happy path stays unchanged.

Extend `tests/unit/test_specify_tools.py` (or create) (lever D):
- `test_write_specification_rejects_empty_scope_for_nontrivial_description` — long description + empty scope_inclusions → `VALIDATION_ERROR`.
- `test_write_specification_allows_empty_scope_for_trivial_description` — short description with no implementation keywords → writes successfully even with empty scope.
- `test_write_specification_happy_path_writes_files` — populated fields → writes.

## Critical files

- `spine/workflow/critic_review.py` — `_build_review_prompt` refactor + `agent_critic_check` threading of `specification_json`
- `spine/critic/agent.py` — tighten `_PLAN_REVIEW_INSTRUCTIONS` to reference the `<specification>` block explicitly (drops the "missing scope fields" hallucination vector)
- `spine/agents/plan_tools.py` — `StructuredWritePlanTool._run` gains the `validate_feature_slices` pre-check
- `spine/agents/specify_tools.py` — `WriteSpecificationTool._run` gains the empty-scope pre-check for non-trivial descriptions
- `spine/workflow/slice_scheduler.py` — reuse `validate_feature_slices` (already imports cleanly, no changes)
- Tests: 3 new files (or 3 new test cases in existing files if present)

## What we are deliberately NOT doing

- **Not adding `scope_inclusions` / `scope_exclusions` to the PLAN schema.** Those are spec-level concepts. The plan's slices RESPECT scope; they don't redefine it.
- **Not changing the model** for any phase. The goal is quality per token, not bigger model.
- **Not bumping `max_retries`** above 3. If the synthesizer can't fix a structural defect in 3 cycles, more cycles won't help — the bug is in the validation gating, not the rework budget.
- **Not removing the critic LLM check** for scope creep. The deterministic pre-checks in B catch structural defects; the LLM is still needed for semantic checks (slice granularity, coverage, the LLM-only judgements).
- **Not refactoring the SPECIFY rework loop's prompt** — that's already using the hostage layout via the prior session's `9300a28` work.
- **Not modifying the critic's system prompt structure** beyond updating the one paragraph that references `scope_inclusions`/`scope_exclusions` to reference the new `<specification>` block. The XML-tagged role/constraints/workflow blocks from the prior refactor stay intact.

## Verification

1. **Unit:** the new test files above must all pass. Stub at the same layer as existing critic / write-tool tests.
2. **Unit regression:** the full `tests/unit/` sweep should match the post-`23565e6` baseline (854 passing, 12 pre-existing failures unrelated to this work).
3. **Smoke (read-only):** import-time check — for an empty-scope payload, `WriteSpecificationTool._run` returns a string starting with `"VALIDATION_ERROR"` and does NOT create files. Same for an invalid-deps plan and `StructuredWritePlanTool._run`.
4. **End-to-end:** re-run the same `task` workflow ("Project Onboarding Engine") that failed at `019e726f`. Audit the resulting trace via `/langsmith-trace-analysis`:
   - **Spec side**: `specification.json` has non-empty `scope_inclusions` and `scope_exclusions` (the write-tool now refuses to persist without them).
   - **Plan side**: the critic's PLAN-review prompt contains a `<specification>` block (eyeball via the trace's user message).
   - **Rework count**: the critic-plan rework cycle count drops below 3 (target: ≤ 1, ideally 0 for this work item).
   - **Reaches IMPLEMENT/VERIFY**: the workflow successfully exits the critic-plan gate and dispatches at least one slice. This finally unlocks the deferred IMPLEMENT/VERIFY trace audit and validates the `23565e6` changes from the prior session.
   - **P:C ratio**: should stay in the current 13-21:1 healthy band. Lever A adds ~2 K per critic call; B + D save 5-15 K per avoided rework. Net target: P:C ≤ 20:1 unchanged or slightly improved on the same workload.
5. **Failure-mode check:** intentionally construct a description that's clearly trivial ("Add a `--verbose` flag to spine CLI") and confirm `WriteSpecificationTool._run` does NOT reject empty scope fields (proportionality rule — short, no implementation keywords).
</content>
