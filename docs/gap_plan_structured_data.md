# Gap Plan — SPINE Combined Implementation

**Generated:** 2026-05-23  
**Based on:** `docs/implementation_audit.md`  
**Reference Plan:** `docs/combined_implementation_plan.md`

---

## Verification Summary

Audit of the combined implementation plan found **64% completion** (21/33 items pass). The structural foundation is solid — all models, config schemas, prerequisite check functions, and state invariants are correctly defined. However, four key areas need remediation:

1. **🔴 Routing bug:** `prereq_gate_implement` bypassed from `critic_plan` path (all work types)
2. **🟡 Structured output gap:** `SliceResult`/`VerificationResult` Pydantic models exist but aren't used at inference time
3. **🟡 SPECIFY `specification.json`:** `Specification` model exists but tool doesn't produce JSON
4. **🟡 Constrained decoding:** Backend wire exists but not activated in config or subagent guard

---

## Fix Instructions

### Fix 1: 🔴 `prereq_gate_implement` Routing Bypass

- **Slice ID:** `routing-bug-prereq-gate`
- **File path:** `spine/workflow/compose.py`
- **Change type:** modify
- **Specific change:** In the `gate_edges` loop (line 793), change the artifact gate's `"proceed"` target. When the destination is `PhaseName.IMPLEMENT.value`, route `"proceed"` to `"prereq_gate_implement"` instead of directly to `IMPLEMENT`. This ensures ALL paths to IMPLEMENT go through the prerequisite invariant check.

```python
# Current (lines 793-806):
for (src, dst), required_phase in gate_edges.items():
    gate_name = _gate_node_name(src, dst)
    graph.add_node(gate_name, make_artifact_gate_node(required_phase, dst))
    graph.add_conditional_edges(gate_name, artifact_gate_router, {
        "proceed": dst,  # ← directly to IMPLEMENT, bypassing prereq gate
        "needs_review": "human_review",
    })

# Fixed:
for (src, dst), required_phase in gate_edges.items():
    gate_name = _gate_node_name(src, dst)
    graph.add_node(gate_name, make_artifact_gate_node(required_phase, dst))
    # Route through prerequisite gate when target is IMPLEMENT
    actual_dst = f"prereq_gate_{dst}" if prereq_gate_exists(dst) else dst
    graph.add_conditional_edges(gate_name, artifact_gate_router, {
        "proceed": actual_dst,  # ← now routes through prereq gate
        "needs_review": "human_review",
    })
```

- **Acceptance criteria:**
  - [ ] All 4 work types route `critic_plan` → `gate_*` → `prereq_gate_implement` → `IMPLEMENT`
  - [ ] `plan_completed` invariant checked before IMPLEMENT runs in every path
  - [ ] Existing workflow tests pass without regression
  - [ ] Gap-fix loop (`gap_plan → implement`) still works correctly

- **Estimated complexity:** small

---

### Fix 2: Enable `response_format` for Slice Implementers

- **Slice ID:** `structured-slice-implementer`
- **File path:** `spine/agents/subagents.py`
- **Change type:** modify
- **Specific change:** On line 593, replace the `name == "researcher"` restriction with a more permissive check. All subagents with a `SUBAGENT_RESPONSE_MODELS` entry should get `response_format`. Replace `_supports_forced_tool_choice` with `_supports_guided_decoding` to match the constrained decoding strategy.

```python
# Current (line 593):
if name == "researcher" and _supports_forced_tool_choice(model):
    spec["response_format"] = SUBAGENT_RESPONSE_MODELS[name]

# Fixed:
if name in SUBAGENT_RESPONSE_MODELS and _supports_forced_tool_choice(model):
    spec["response_format"] = SUBAGENT_RESPONSE_MODELS[name]
```

- **Acceptance criteria:**
  - [ ] `slice-implementer` subagent receives `response_format=SliceResult`
  - [ ] `slice-verifier` subagent receives `response_format=VerificationResult`
  - [ ] `researcher` subagent still receives `response_format=ResearchFindings`
  - [ ] Thinking models (Qwen3, QwQ, DeepSeek-R) still skip `response_format` via the guard
  - [ ] Structured results are accessible via `result["structured_response"]`

- **Estimated complexity:** small

---

### Fix 3: SPECIFY `specification.json` Structured Output

- **Slice ID:** `structured-specify-output`
- **File path:** `spine/agents/specify_tools.py`
- **Change type:** modify
- **Specific change:** Add `specification_json: str | None` field to `_WriteSpecificationInput`. In `WriteSpecificationTool._run()`, write `specification.json` alongside `specification.md` to the artifacts directory.

```python
class _WriteSpecificationInput(BaseModel):
    # ... existing fields ...
    specification_json: str | None = Field(
        default=None,
        description="Structured JSON specification (must be valid JSON). "
                    "Use the Specification schema with keys: title, summary, "
                    "objectives, requirements, constraints, scope_inclusions, "
                    "scope_exclusions, known_risks."
    )
```

And in `_run()`:
```python
if specification_json:
    json_path = Path(output_dir) / "specification.json"
    json_path.write_text(specification_json, encoding="utf-8")
```

- **File path:** `spine/agents/specify_agent.py`
- **Change type:** modify
- **Specific change:** Add instruction in `_build_specify_prompt()` to call `write_specification` with `specification_json` set to a valid `Specification` JSON object.

- **File path:** `spine/workflow/subgraphs/exploration_subgraph.py`
- **Change type:** modify
- **Specific change:** In `_synthesize_specify` and `_save_exploration_artifacts`, scan for `specification.json` like `plan.json` is already scanned.

- **File path:** `spine/workflow/subgraph_state.py`
- **Change type:** modify
- **Specific change:** Add `specification_json: str` field to `SpecifySubgraphState` for state persistence.

- **Acceptance criteria:**
  - [ ] SPECIFY produces both `specification.md` and `specification.json`
  - [ ] `specification.json` is valid JSON matching the `Specification` Pydantic schema
  - [ ] `specification.json` is captured in artifacts state and persisted to disk
  - [ ] PLAN phase can read `specification.json` from disk
  - [ ] Backward compatible — existing workflows without JSON field still work

- **Estimated complexity:** medium

---

### Fix 4: Enable Constrained Decoding

- **Slice ID:** `enable-guided-decoding`
- **File path:** `.spine/config.yaml`
- **Change type:** modify
- **Specific change:** Add `guided_decoding: true` to the `local` provider config and to phase configs that use it.

```yaml
providers:
  llm:
  - name: local
    type: deepagents-model
    model: openai:model
    base_url: 'http://localhost:8000/v1'
    api_key: 'vllm'
    enabled: true
    guided_decoding: true  # ← NEW
  phases:
    implement:
      provider: local
      temperature: 0.6
      guided_decoding: true  # ← NEW
    implement/subagents/slice-implementer:
      provider: local
      guided_decoding: true  # ← NEW
```

- **File path:** `spine/agents/subagents.py`
- **Change type:** modify
- **Specific change:** Add import of `_supports_guided_decoding` from `spine.agents.helpers` and use it alongside `_supports_forced_tool_choice`:

```python
from spine.agents.helpers import _supports_guided_decoding

# Line 593 — add guided_decoding awareness:
if name in SUBAGENT_RESPONSE_MODELS and (
    _supports_forced_tool_choice(model) or _supports_guided_decoding(model)
):
    spec["response_format"] = SUBAGENT_RESPONSE_MODELS[name]
```

- **Acceptance criteria:**
  - [ ] `guided_decoding: true` present in `.spine/config.yaml` for local provider
  - [ ] `guided_json` injected into model kwargs when provider supports it
  - [ ] `_supports_guided_decoding` used alongside `_supports_forced_tool_choice` in subagents
  - [ ] Local vLLM produces schema-constrained output for subagents
  - [ ] OpenRouter models fall back gracefully (they don't support `guided_json`)

- **Estimated complexity:** small

---

### Fix 5: Set `critic_*_completed` Invariant Flags

- **Slice ID:** `critic-completion-flags`
- **File path:** `spine/workflow/compose.py`
- **Change type:** modify
- **Specific change:** In `_critic_result_mapper` (line 306 function body), add setting of `critic_specify_completed` and `critic_plan_completed` flags based on `reviewed_phase`.

```python
# After line 345 (phase_status == "error" check):
# Set critic completion invariants
if reviewed_phase == PhaseName.SPECIFY.value:
    base["critic_specify_completed"] = True
elif reviewed_phase == PhaseName.PLAN.value:
    base["critic_plan_completed"] = True
```

- **Acceptance criteria:**
  - [ ] `critic_specify_completed` set to `True` after `critic_specify` runs
  - [ ] `critic_plan_completed` set to `True` after `critic_plan` runs
  - [ ] Flags preserved across rework loops

- **Estimated complexity:** small

---

### Fix 6: Add `verification_findings` to VerifySubgraphState

- **Slice ID:** `verify-findings-state`
- **File path:** `spine/workflow/subgraph_state.py`
- **Change type:** modify
- **Specific change:** Add `verification_findings` field to `VerifySubgraphState` for programmatic access to structured verification results.

```python
class VerifySubgraphState(BaseSubgraphState, total=False):
    # ... existing fields ...
    verification_findings: list[dict]  # Structured VerificationResult objects
```

- **File path:** `spine/workflow/subgraphs/verify_subgraph.py`
- **Change type:** modify
- **Specific change:** In `_save_artifacts` node, populate `verification_findings` from subagent results.

- **Acceptance criteria:**
  - [ ] `verification_findings` populated when verify agent completes
  - [ ] GAP_PLAN can access structured verification data
  - [ ] Backward compatible — existing workflows work without the field

- **Estimated complexity:** small

---

## Re-Verify Slices

After applying fixes, verify these slices:

- `routing-bug-prereq-gate` — trace all 4 work type graphs for correct prereq gate routing
- `structured-slice-implementer` — verify SliceResult appears in structured_response
- `structured-specify-output` — verify specification.json written to disk
- `enable-guided-decoding` — verify guided_json appears in vLLM request

---

## Implementation Order

1. **Fix 1** (routing bug) — 🔴 P0, must fix first
2. **Fix 2** (response_format for subagents) — 🟡 P1, enables structured data
3. **Fix 4** (guided decoding) — 🟡 P1, enables constrained output
4. **Fix 3** (specification.json) — 🟡 P1, completes SPECIFY→PLAN handoff
5. **Fix 5** (critic flags) — 🟢 P2, minor
6. **Fix 6** (verification_findings) — 🟢 P2, minor

---

*Gap plan produced from `docs/implementation_audit.md`. See `docs/combined_implementation_plan.md` for the full reference plan.*
