# SPINE Implementation Audit
## Verification Against `combined_implementation_plan.md`

**Date:** 2026-05-23  
**Auditor:** Hermes Agent  
**Base Plan:** `docs/combined_implementation_plan.md`

---

## Executive Summary

| Milestone | Status | Pass Rate |
|-----------|--------|-----------|
| M1: Foundation | ✅ 4/5 | 80% |
| M2: State Invariants | ✅ 4/5 | 80% |
| M3: Structured Data Handoffs | ⚠️ 3/8 | 38% |
| M4: Prompt Optimization | ✅ 4/5 | 80% |
| M5: Fail-Closed Gates | ⚠️ 4/5 | 80% |
| M6: Constrained Decoding | ⚠️ 2/5 | 40% |
| **Overall** | **⚠️ 21/33** | **64%** |

**Critical Bug Found:** 🔴 `prereq_gate_implement` bypassed when `critic_plan` routes through artifact gate — affects ALL work types.

---

## Detailed Audit

### M1: Foundation

| # | Item | Status | Detail |
|---|------|--------|--------|
| 1.1 | `_SKIP_AGENTS_MD` includes all phases | ✅ PASS | `specify`, `plan`, `tasks`, `implement`, `verify`, `critic`, `gap_plan` at `skills_resolver.py:51-59` |
| 1.2 | `CriticalContractFailure` exception | ✅ PASS | `exceptions.py:52-63` with `phase` and `reason` args |
| 1.3 | `guided_decoding` in config | ✅ PASS | `config.py:92` field, `config.py:339` in `_PROVIDER_KEYS`, `config.py:258-260` env override |
| 1.4 | New Pydantic models | ✅ PASS | `Specification` (L148), `FixInstruction` (L167), `GapPlan` (L186), `CriticReview` (L199) all in `types.py` |
| 1.5 | Phase completion flags | ✅ PASS | All 8 flags in `state.py:108-115`: `spec_completed`, `plan_completed`, `tasks_completed`, `implement_completed`, `verify_completed`, `critic_specify_completed`, `critic_plan_completed`, `gap_plan_completed` |

**Verdict:** ✅ Foundation is solid. All models, exceptions, config, and invariants defined correctly.

---

### M2: State Invariants

| # | Item | Status | Detail |
|---|------|--------|--------|
| 2.1 | Result mappers set completion flags | ✅ PASS | `compose.py:235` verify, `:251` implement, `:267` tasks, `:282` spec, `:300` plan, `:375` gap_plan — all set `<phase>_completed = phase_status == "success"` |
| 2.2 | Critic result mapper sets `critic_*_completed` | ❌ FAIL | `compose.py:306-348` — `_critic_result_mapper` does NOT set `critic_specify_completed` or `critic_plan_completed` |
| 2.3 | Prerequisite check functions | ✅ PASS | `artifact_gate.py:553` `_check_spec_prerequisite`, `:578` `_check_plan_prerequisite`, `:603` `_check_implement_prerequisite`, `:628` `_check_verify_prerequisite` |
| 2.4 | `make_prerequisite_gate_node()` | ✅ PASS | `artifact_gate.py:481-544` — creates gate node with correct status routing |
| 2.5 | Prerequisite gates wired in compose.py | ✅ PASS | `compose.py:771-784` nodes added, `:915-950` conditional edges wired, `:834-839` `get_proceed_target()` maps phases to gates |

**Verdict:** ⚠️ Gap at 2.2 — critic completion flags not propagated. These flags exist in `WorkflowState` but are never set.

---

### M3: Structured Data Handoffs

| # | Item | Status | Detail |
|---|------|--------|--------|
| 3.1 | SPECIFY `specification.json` output | ❌ FAIL | `specify_tools.py` `WriteSpecificationInput` has no `specification_json` field. `exploration_subgraph.py` `_synthesize_specify` doesn't reference `specification.json`. No `Specification` model usage in specify phase. |
| 3.2 | PLAN complexity/DAG validation | ✅ PASS | `plan_tools.py:463` `complexity` field validated, `:140` `min_length=1` on required fields, `:550` slice validation errors reported |
| 3.3 | IMPLEMENT `response_format=SliceResult` | ❌ FAIL | `subagents.py:593` — `response_format` still restricted to `name == "researcher"` only. `SliceResult` model unused at inference time. |
| 3.4 | VERIFY `response_format=VerificationResult` | ❌ FAIL | Same as 3.3 — `VerificationResult` model exists but never passed as `response_format` |
| 3.5 | GAP_PLAN structured tools | ✅ PASS | `gap_plan_tools.py` EXISTS (372 lines) with `ReadVerificationFindingsTool` and `WriteStructuredGapPlanTool`. `gap_plan_agent.py` imports and uses it. `skip_filesystem_middleware=True`. |
| 3.6 | CRITIC `response_format=CriticReview` | ✅ PASS | `critic/agent.py:113` passes `response_format=CriticReview`. `critic_review.py:199-233` `_parse_agent_review` checks `structured_response` first, falls back to keyword parsing. |
| 3.7 | `verification_findings` in VerifySubgraphState | ❌ FAIL | `subgraph_state.py:63-72` — `VerifySubgraphState` has no `verification_findings` field |
| 3.8 | `gap_plan_json` in GapPlanSubgraphState | ✅ PASS | `subgraph_state.py:91` — `gap_plan_json: str` field exists |

**Verdict:** ⚠️ Largest gap. SPECIFY doesn't produce structured JSON output. Slice-implementer and slice-verifier subagents don't get `response_format`. Only critic and researcher get structured output.

---

### M4: Prompt Optimization

| # | Item | Status | Detail |
|---|------|--------|--------|
| 4.1 | SPECIFY negative instructions removed | ✅ PASS | Only 1 match for negative language in `specify_agent.py` — "cannot" in docstring, not in prompt |
| 4.2 | PLAN AGENTS.md duplication removed | ✅ PASS | No AGENTS.md references in prompts. Only 1 "cannot" in docstring. |
| 4.3 | IMPLEMENT duplicated sections removed | ✅ PASS | Zero negative matches in `implement_agent.py` |
| 4.4 | VERIFY fluff removed | ✅ PASS | Zero negative/philosophical matches in `verify_agent.py` |
| 4.5 | Subagent tool table reduced | ⚠️ UNVERIFIED | Could not fully verify — SUBAGENT_PROMPTS in `subagents.py` still references 18-row MCP tool table (the MCP tools auto-register regardless) |

**Verdict:** ✅ Prompts appear clean with minimal negative instruction. Tool table reduction partially complete (MCP tools auto-inject, so explicit listing is cosmetic).

---

### M5: Fail-Closed Gates

| # | Item | Status | Detail |
|---|------|--------|--------|
| 5.1 | PLAN entry checks `spec_completed` | ✅ PASS | `_check_spec_prerequisite` at `artifact_gate.py:553` checks `spec_completed` flag |
| 5.2 | IMPLEMENT entry checks `plan_completed` | ✅ PASS | `_check_plan_prerequisite` at `artifact_gate.py:578` checks `plan_completed` flag |
| 5.3 | VERIFY entry checks `implement_completed` | ✅ PASS | `_check_implement_prerequisite` at `artifact_gate.py:603` checks `implement_completed` flag |
| 5.4 | GAP_PLAN entry checks `verification_attempted` | ✅ PASS | `_check_verify_prerequisite` at `artifact_gate.py:628` checks `verification_attempted` flag |
| 5.5 | Wired into `make_artifact_gate_node()` | ✅ PASS | Via separate `make_prerequisite_gate_node()` in `artifact_gate.py:481`, wired in `compose.py:771-784` |

**Verdict:** ✅ Gate logic exists and is correct. But see Critical Bug — some paths bypass the prerequisite gates.

---

### M6: Constrained Decoding

| # | Item | Status | Detail |
|---|------|--------|--------|
| 6.1 | `guided_json` wired in `_build_local_model()` | ✅ PASS | `helpers.py:305-308` — when `provider_cfg.get("guided_decoding")` and `response_format`, injects `guided_json` via `extra_body` |
| 6.2 | `_supports_guided_decoding()` guard | ✅ PASS | `helpers.py:315-331` — checks for OpenRouter (doesn't support guided_json) |
| 6.3 | Replace `_supports_forced_tool_choice` | ❌ FAIL | `subagents.py:593` still uses `_supports_forced_tool_choice(model)`, NOT `_supports_guided_decoding`. The guard was NOT replaced. |
| 6.4 | `guided_decoding: true` in config.yaml | ❌ FAIL | `.spine/config.yaml` has NO `guided_decoding` field on any provider |
| 6.5 | Integration test | ❌ FAIL | Can't test without 6.3 and 6.4 |

**Verdict:** ⚠️ Backend wiring exists but not enabled. `_supports_guided_decoding` exists but `_supports_forced_tool_choice` was not replaced with it. Config not updated.

---

## 🔴 Critical Bug: `prereq_gate_implement` Routing Bypass

### Affected Code
`compose.py:841-868` (critic routing block)

### Root Cause
When `critic_plan` produces `"passed"`, and there's an artifact gate (`gate_edges` contains `("critic_plan", "implement")`), the `critic_proceed_target` is set directly to the artifact gate node:

```python
if has_gate and next_node:
    gate_name = _gate_node_name(node_name, next_node)
    critic_proceed_target: str = gate_name  # ← artifact gate, NOT prereq gate
```

The artifact gate (`gate_critic_plan_to_implement`) routes `"proceed"` directly to `IMPLEMENT` (line 798-805), **bypassing** `prereq_gate_implement`.

### Correct Flow (Expected)
```
critic_plan --passed--> gate_critic_plan_to_implement --proceed--> prereq_gate_implement --proceed--> IMPLEMENT
```

### Actual Flow (Bug)
```
critic_plan --passed--> gate_critic_plan_to_implement --proceed--> IMPLEMENT  ← prereq gate skipped!
```

### Impact
- **All 4 work types affected:** `task`, `critical_task`, `reviewed_task`, `critical_reviewed_task`
- `prereq_gate_implement` never runs after `critic_plan` passes
- `plan_completed` invariant NOT checked before IMPLEMENT runs in this path
- IMPLEMENT can start without a valid plan in the critic_plan→implement path

### Fix (Option A — route artifact gate proceed through prereq gate)
Change the artifact gate's "proceed" target from `IMPLEMENT` to `prereq_gate_implement` when the next node is IMPLEMENT:

```python
# In the gate_edges loop, change "proceed" target:
for (src, dst), required_phase in gate_edges.items():
    gate_name = _gate_node_name(src, dst)
    actual_dst = f"prereq_gate_{dst}" if dst == PhaseName.IMPLEMENT.value else dst
    graph.add_conditional_edges(gate_name, artifact_gate_router, {
        "proceed": actual_dst,
        "needs_review": "human_review",
    })
```

### Fix (Option B — route critic through prereq gate after artifact gate)
Change `critic_proceed_target` to route through prereq gate:

```python
if has_gate and next_node:
    gate_name = _gate_node_name(node_name, next_node)
    critic_proceed_target = gate_name
    # Also add edge from artifact gate to prereq gate
```

Option A is simpler and more robust — it ensures ALL paths to IMPLEMENT go through the prereq gate.

---

## Minor Issues

### Issue 1: `critic_*_completed` Flags Never Set
**Files:** `compose.py:306-348`
**Detail:** `_critic_result_mapper` doesn't set `critic_specify_completed` or `critic_plan_completed`, despite these fields existing in `WorkflowState`. Add after line 345:
```python
base["critic_plan_completed"] = reviewed_phase == PhaseName.PLAN.value
base["critic_specify_completed"] = reviewed_phase == PhaseName.SPECIFY.value
```

### Issue 2: `_SKIP_AGENTS_MD` Missing `exploration` Phase
**File:** `skills_resolver.py:51-59`
**Detail:** The `exploration` phase (used in config for provider resolution) is not in `_SKIP_AGENTS_MD`. The exploration phase doesn't get its own agent (it uses specify/plan agents), so this is low-impact.

### Issue 3: `gap_plan→prereq_gate_implement` Bypass
**File:** `compose.py:952`
**Detail:** `graph.add_edge(GAP_PLAN.value, IMPLEMENT.value)` bypasses `prereq_gate_implement`. In the gap-fix loop, IMPLEMENT is re-running with gap context — the prereq check might be overly strict here. **Design decision needed**: should gap-fix re-runs skip the prereq gate?

---

## Summary of Required Fixes

| Priority | Issue | Files | Effort |
|----------|-------|-------|--------|
| 🔴 P0 | `prereq_gate_implement` routing bypass | `compose.py` | 15 min |
| 🟡 P1 | Enable `response_format` for slice-implementer/verifier | `subagents.py` | 30 min |
| 🟡 P1 | SPECIFY `specification.json` structured output | `specify_tools.py`, `specify_agent.py`, `exploration_subgraph.py` | 2 hrs |
| 🟡 P1 | Enable `guided_decoding` in config + replace guard | `.spine/config.yaml`, `subagents.py` | 15 min |
| 🟢 P2 | Set `critic_*_completed` flags | `compose.py` | 5 min |
| 🟢 P2 | Add `verification_findings` to VerifySubgraphState | `subgraph_state.py` | 5 min |
| 🟢 P3 | Add `exploration` to `_SKIP_AGENTS_MD` | `skills_resolver.py` | 2 min |
| 🟢 P3 | Decide `gap_plan→implement` prereq gate behavior | `compose.py` | Design discussion |

---

## Verdict

**64% complete.** The structural foundation is solid — all models, exceptions, config schemas, prerequisite check functions, and state invariants are defined. The critical gap is:

1. **Routing bug**: `prereq_gate_implement` is not reached from `critic_plan` path
2. **Structured output not enabled**: SliceResult and VerificationResult models exist but aren't used at inference time
3. **SPECIFY doesn't produce `specification.json`**: The model is defined but the tool doesn't write it
4. **Constrained decoding not enabled**: Backend wire exists but guard and config not updated
