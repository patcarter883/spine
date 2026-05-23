# Constrained Decoding, Fail-Closed Gates, State Invariants

**Date:** 2026-05-23
**Scope:** SPINE workflow engine — inference layer, phase entry points, state schema

---

## 1. Hardcoded Constrained Decoding (Grammar Control)

### Problem
When models generate tool calls under load, they occasionally drop commas, hallucinate escaped characters, or wrap output in unsolicited markdown. Current post-hoc validation (`ToolSchemaValidator` rebound loop) wastes tokens on self-correction.

### Solution
Inject schema-constrained sampling at the inference engine level so the model **cannot** generate tokens that violate the JSON schema.

### Changes

#### 1a. Extend provider config schema — `spine/config.py`

Add a `_PROVIDER_KEYS` entry and config field for guided decoding opt-in:

```yaml
providers:
  llm:
    - name: local
      type: deepagents-model
      model: openai:model
      base_url: "http://localhost:8000/v1"
      api_key: "vllm"
      guided_decoding: true    # NEW
```

In `SpineConfig`, add:
```python
guided_decoding: bool = False  # class field
```

In `load()`, parse:
```python
guided_decoding=os.getenv(
    "SPINE_GUIDED_DECODING",
    str(spine.get("guided_decoding", False)).lower(),
) in ("1", "true", "yes"),
```

Also add `"guided_decoding"` to the `_PROVIDER_KEYS` tuple so phases can override:
```python
_PROVIDER_KEYS: tuple[str, ...] = (
    "base_url", "api_key", "temperature",
    "max_tokens", "max_completion_tokens",
    "request_timeout", "max_retries",
    "guided_decoding",  # NEW
)
```

#### 1b. Wire guided JSON into `_build_local_model()` — `spine/agents/helpers.py`

When the provider is vLLM-compatible (`openai:model`) and `guided_decoding` is enabled, pass `extra_body` with `guided_json` extracted from the agent's `response_format`.

The key change: `_build_local_model()` must accept an optional `response_format` parameter (a Pydantic model class) and convert it to a vLLM-compatible schema:

```python
def _build_local_model(
    model_spec: str,
    provider_cfg: dict[str, Any],
    response_format: type | None = None,  # NEW param
) -> BaseChatModel:
    # ... existing kwargs setup ...

    # Inject guided decoding for vLLM endpoints
    if provider_cfg.get("guided_decoding") and response_format is not None:
        try:
            json_schema = response_format.model_json_schema()
            kwargs.setdefault("extra_body", {})["guided_json"] = json_schema
        except Exception:
            logger.debug("Could not extract JSON schema from response_format")

    return ChatOpenAI(**kwargs)
```

#### 1c. Wire response_format through `build_phase_agent()` — `spine/agents/factory.py`

Pass `response_format` through to `_build_local_model()` so subagent Pydantic models are enforced at the engine level:

```python
# In build_phase_agent(), after resolving provider_cfg:
if provider_cfg.get("guided_decoding") and phase in SUBAGENT_RESPONSE_MODELS:
    model = _build_local_model(model_spec, provider_cfg,
                               response_format=SUBAGENT_RESPONSE_MODELS[phase])
```

#### 1d. Guard: detection logic for engine support

Not all models/endpoints support guided decoding. Add a helper:

```python
def _supports_guided_decoding(model_spec: str, provider_cfg: dict) -> bool:
    """Check if the model/engine supports guided JSON decoding."""
    if not provider_cfg.get("guided_decoding"):
        return False
    # vLLM, SGLang, llama.cpp support it; OpenRouter does not
    if model_spec.startswith("openrouter:"):
        return False
    if model_spec.startswith("openai:") and provider_cfg.get("base_url"):
        return True
    return False
```

### Files Changed
| File | Change |
|------|--------|
| `spine/config.py` | Add `guided_decoding` field + `_PROVIDER_KEYS` entry |
| `spine/agents/helpers.py` | Accept `response_format`, inject `guided_json` into `extra_body` |
| `spine/agents/factory.py` | Wire `response_format` through to model builder |
| `.spine/config.yaml` | Add `guided_decoding: true` to local provider entries |

---

## 2. Fail-Closed State Verification (The Hard Gate)

### Problem
Nodes currently access state loosely (`state.get("artifacts", {}).get(...)`) and silently proceed with empty prerequisites. A refactor that breaks spec-to-plan data passing silently produces empty plans instead of crashing.

### Solution
Add hard precondition checks at every phase entry point. Use `CriticalContractFailure` (new) that cost $0 and zero tokens.

### Changes

#### 2a. New exception — `spine/exceptions.py`

```python
class CriticalContractFailure(Exception):
    """Raised when a phase's required prerequisites are not met.

    This is a deterministic, code-level failure — not an LLM error.
    It costs $0 and zero tokens, and pins the exact bug to the middleware.
    """
```

#### 2b. PLAN phase guard — `spine/phases/plan.py`

Replace loose state access at line 180 with a hard check **before** building the agent:

```python
async def call_plan(state, config=None):
    work_id = state.get("work_id", "unknown")

    # ── Fail-closed: SPECIFY artifact must exist and be non-empty ──
    spec_artifact = (
        state.get("artifacts", {})
        .get(PhaseName.SPECIFY.value, {})
    )
    spec_content = spec_artifact.get("specification.md", "") if isinstance(spec_artifact, dict) else ""
    if not spec_content or len(spec_content.strip()) < 50:
        raise CriticalContractFailure(
            f"PLAN phase triggered for {work_id}, but SPECIFY artifact is missing "
            f"or empty (≤50 chars). spec.md content: {spec_content!r}"
        )

    # Also verify on-disk presence
    spec_path = Path(artifact_path(work_id, PhaseName.SPECIFY.value)) / "specification.md"
    if not spec_path.exists() or spec_path.stat().st_size < 50:
        raise CriticalContractFailure(
            f"PLAN phase triggered for {work_id}, but specification.md does not exist "
            f"on disk at {spec_path} or is too small."
        )
```

#### 2c. IMPLEMENT phase guard — `spine/phases/implement.py`

Verify `execution_waves` is non-empty before dispatching:

```python
execution_waves = state.get("execution_waves", [])
if not execution_waves or not any(wave for wave in execution_waves):
    raise CriticalContractFailure(
        f"IMPLEMENT phase triggered for {work_id}, but execution_waves is empty. "
        "PLAN phase must produce valid feature_slices with dependencies."
    )
```

#### 2d. Add artifact gate: plan → critic_plan — `spine/workflow/compose.py`

Currently only `tasks → implement` is gated. Add a gate before critic:

```python
# In build_workflow_graph(), after the plan phase node:
if not has_critic_before_plan:
    # Ensure plan artifacts exist before critic reviews them
    graph.add_node("gate_plan_to_critic",
                    make_artifact_gate_node(PhaseName.PLAN.value, f"{PhaseName.CRITIC.value}_plan"))
    graph.add_conditional_edges(
        "gate_plan_to_critic",
        artifact_gate_router,
        {"proceed": f"{PhaseName.CRITIC.value}_plan", "needs_review": "human_review"},
    )
```

#### 2e. VERIFY phase guard — `spine/phases/verify.py`

```python
implement_artifact = (
    state.get("artifacts", {})
    .get(PhaseName.IMPLEMENT.value, {})
)
if not implement_artifact:
    raise CriticalContractFailure(
        f"VERIFY phase triggered for {work_id}, but IMPLEMENT produced no artifacts."
    )
```

### Files Changed
| File | Change |
|------|--------|
| `spine/exceptions.py` | Add `CriticalContractFailure` exception class |
| `spine/phases/plan.py` | Add fail-closed spec check before agent build |
| `spine/phases/implement.py` | Add fail-closed execution_waves check |
| `spine/phases/verify.py` | Add fail-closed implement artifact check |
| `spine/workflow/compose.py` | Add gate_plan_to_critic artifact gate |

---

## 3. Explicit State Invariants

### Problem
The system has some invariant booleans (`exploration_executed`, `gap_plan_produced`) but they are not consistently set, not wired into all state schemas, and not checked by any gate. An empty artifact payload can be misinterpreted as intentional.

### Solution
Add explicit phase completion booleans to `WorkflowState` and enforce that result mappers set them. Split "the agent ran" from "the agent produced output."

### Changes

#### 3a. Extend `WorkflowState` — `spine/models/state.py`

Replace the ad-hoc invariant section with a comprehensive completion tracker:

```python
class WorkflowState(TypedDict, total=False):
    # ... existing fields ...

    # ── Phase Completion Flags ──
    # Set by result mappers on success. Consumed by downstream gates
    # and fail-closed checks to distinguish "intentional empty" from "bug".
    spec_completed: bool
    plan_completed: bool
    tasks_completed: bool
    implement_completed: bool
    verify_completed: bool
    critic_plan_completed: bool
    critic_specify_completed: bool
    gap_plan_completed: bool
```

Remove the old ad-hoc fields (`gap_plan_produced`, `exploration_executed`) — the new flags replace them with a consistent naming convention (`<phase>_completed`).

#### 3b. Update all result mappers — `spine/workflow/compose.py`

Each `_<phase>_result_mapper` must set its completion flag. Update all seven:

**`_specify_result_mapper`** (line 254):
```python
base["spec_completed"] = phase_status not in ("error", "needs_review")
```

**`_plan_result_mapper`** (line 266):
```python
base["plan_completed"] = phase_status not in ("error", "needs_review")
base["exploration_executed"] = True  # backward compat, will remove
```

**`_tasks_result_mapper`** (line 242):
```python
base["tasks_completed"] = phase_status not in ("error", "needs_review")
```

**`_implement_result_mapper`** (line 230):
```python
base["implement_completed"] = phase_status not in ("error", "needs_review")
```

**`_verify_result_mapper`** (line 200):
```python
base["verify_completed"] = phase_status not in ("error", "needs_review")
```

**`_critic_result_mapper`** (line 282):
```python
# Set the appropriate critic flag based on reviewed_phase
if reviewed_phase == PhaseName.PLAN.value:
    base["critic_plan_completed"] = phase_status == ReviewStatus.PASSED.value
elif reviewed_phase == PhaseName.SPECIFY.value:
    base["critic_specify_completed"] = phase_status == ReviewStatus.PASSED.value
```

**`_gap_plan_result_mapper`** (line 342):
```python
base["gap_plan_completed"] = phase_status not in ("error", "needs_review")
```

#### 3c. Add invariant checks to artifact gates — `spine/workflow/artifact_gate.py`

In `make_artifact_gate_node()`, after the basic presence check passes, add an invariant consistency check:

```python
# ── Invariant: if the phase claims completion, artifacts must exist ──
phase_completed_key = f"{required_phase}_completed"
if state.get(phase_completed_key) and not has_state_artifacts:
    reason = (
        f"Invariant violation: state.{phase_completed_key}=True but "
        f"no artifacts found for {required_phase}. This indicates a "
        f"state serialization bug — the phase reported success but "
        f"the artifacts were not persisted."
    )
    logger.error("[%s] %s", work_id, reason)
    return {
        "current_phase": required_phase,
        "status": "needs_review",
        "feedback": [{
            "status": "needs_review",
            "tier": "structural",
            "reason": reason,
            "suggestions": [],
        }],
        "prompt_request": None,
    }
```

#### 3d. Extend subgraph states — `spine/workflow/subgraph_state.py`

Add completion flags to each subgraph state for internal tracking:

```python
class PlanSubgraphState(BaseSubgraphState, total=False):
    # ... existing fields ...
    spec_read: bool  # True when spec was actually read from disk
    plan_validated: bool  # True when plan.json passed quality checks

class TasksSubgraphState(BaseSubgraphState, total=False):
    plan_read: bool
    tasks_validated: bool

class ImplementSubgraphState(BaseSubgraphState, total=False):
    # ... existing fields ...
    plan_read: bool
    slices_validated: bool  # True when slices were validated before dispatch
```

### Files Changed
| File | Change |
|------|--------|
| `spine/models/state.py` | Add `<phase>_completed` booleans, remove ad-hoc fields |
| `spine/workflow/compose.py` | Set completion flags in all 7 result mappers |
| `spine/workflow/artifact_gate.py` | Add invariant consistency check in gate node |
| `spine/workflow/subgraph_state.py` | Add per-phase validation flags to subgraph states |

---

## Implementation Order

1. **Phase 1 — State Invariants** (no behavior change, just tracking)
   - Add completion booleans to `WorkflowState`
   - Update result mappers to set them
   - Add invariant checks to artifact gates

2. **Phase 2 — Fail-Closed Guards** (crash loudly on bugs)
   - Add `CriticalContractFailure` exception
   - Add precondition checks in `plan.py`, `implement.py`, `verify.py`
   - Add `gate_plan_to_critic` in compose.py

3. **Phase 3 — Constrained Decoding** (eliminate parsing crashes)
   - Add `guided_decoding` to config schema
   - Wire `response_format` → `guided_json` in `_build_local_model()`
   - Add `_supports_guided_decoding()` guard
   - Enable in `.spine/config.yaml` for local providers

---

## Verification

- **Phase 1**: Run `spine run "test work"` and verify `phase_results` includes completion flags
- **Phase 2**: Intentionally corrupt state (empty spec) and verify `CriticalContractFailure` is raised with clear message
- **Phase 3**: Enable `guided_decoding: true` and verify tool calls never produce schema violations (check LangSmith traces)
