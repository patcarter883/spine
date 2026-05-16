# SPINE Subgraph Decomposition — Architectural & Technical Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan phase-by-phase. Start with Phase 1 (VERIFY PoC) and validate before proceeding to Phase 2.

**Goal:** Decompose the monolithic SPINE LangGraph StateGraph into a parent orchestrator graph with per-phase subgraphs, improving resilience to CancelledError, enabling per-phase checkpoint granularity, independent timeouts, state isolation, and true human-in-the-loop via `interrupt()`.

**Architecture:** Each SPINE phase (SPECIFY, PLAN, TASKS, IMPLEMENT, VERIFY, CRITIC) becomes a compiled LangGraph StateGraph added as a node in the parent orchestrator graph. The parent graph's `WorkflowState` shrinks to routing fields only (status, phase, feedback, artifact summaries). Each subgraph has its own TypedDict state schema, its own checkpointer, and its own timeout. Phase transitions preserve the existing critic rework pattern and artifact gate logic.

**Tech Stack:** Python 3.12+, LangGraph (StateGraph, interrupt, Command, CompiledGraph subgraph-as-node), Deep Agents 0.5.9+, Pydantic (state schemas), SQLite (checkpointing), LangSmith (tracing)

**Motivation (from trace analysis):**
- 1/10 recent traces completed successfully
- 30% errored with CancelledError (deep nesting: ls_run_depth=12, langgraph_step=50 inside a single phase node)
- 20% are stuck orphans (end_time null, 3.51M tokens burned)
- Single 120s stall timer resets on every LLM token but can't detect hung agents
- No checkpoint granularity inside phases — a CancelledError at step 47 loses all prior work

---

## Architecture Overview

### Before (Current — Monolith)

```
┌──── StateGraph(WorkflowState) ──────────────────────────────────────┐
│  START                                                            │
│    │                                                              │
│    ▼                                                              │
│  specify ──► plan ──► critic_plan ──► tasks ──► [gate] ──►        │
│                                                    │              │
│    implement ──► verify ──► END                                   │
│                                                                    │
│  Each node = async function that:                                  │
│    1. Builds a Deep Agent (create_deep_agent)                     │
│    2. Calls agent.invoke() / ainvoke_with_retry()                 │
│    3. Blocks until the entire agent loop completes                │
│    4. Returns partial state update                                │
│                                                                    │
│  Problems:                                                         │
│  • DA agent is a nested StateGraph inside a node function          │
│    → ls_run_depth=12, opaque to parent                            │
│  • Single asyncio.wait() for entire workflow → CancelledError     │
│    kills everything                                                │
│  • One checkpointer for all phases → no mid-phase resume          │
│  • One WorkflowState → all phases share all data                  │
│  • needs_review → END → resume_work() restarts from scratch       │
└────────────────────────────────────────────────────────────────────┘
```

### After (Target — Subgraph Decomposition)

```
┌──── OrchestratorGraph(ParentState) ────────────────────────────────┐
│  START                                                            │
│    │                                                              │
│    ▼                                                              │
│  specify_subgraph ──► plan_subgraph ──► critic ──►                │
│                                                    │              │
│  tasks_subgraph ──► [artifact_gate] ──► implement_subgraph ──►    │
│                                                                    │
│  verify_subgraph ──► END                                          │
│                                                                    │
│  ParentState = routing-only fields:                                │
│    • work_id, work_type, description, workspace_root              │
│    • status: str  (routing signal)                                │
│    • current_phase: str                                           │
│    • phase_results: dict[phase, PhaseResult]  (summary + paths)   │
│    • feedback: list  (critic + human)                             │
│    • retry_count: dict[phase, int]                                │
│    • needs_review_phase: str | None  (for interrupt target)       │
│                                                                    │
│  Each subgraph has its own TypedDict state schema:                 │
│    • messages: Annotated[list, add_messages]  (agent conversation)│
│    • artifacts_output: dict  (files produced by this phase)       │
│    • phase_status: str                                           │
│                                                                    │
│  Each subgraph has its own checkpointer + timeout.                 │
│  CancelledError in plan_subgraph → parent catches → retry or      │
│  route to needs_review. Spec artifacts preserved.                 │
└────────────────────────────────────────────────────────────────────┘
```

### Subgraph Integration Pattern

Two patterns from LangGraph docs:

**Pattern A: "Add subgraph as a node"** — parent and subgraph share state keys. The compiled subgraph is passed directly to `graph.add_node()`.

**Pattern B: "Call subgraph inside a node"** — parent and subgraph have different state schemas. A wrapper function maps parent state to subgraph input, invokes the subgraph, and maps output back to parent state.

**SPINE uses Pattern B** because each phase subgraph needs its own `messages` channel (DA agent conversation state) that the parent shouldn't see. The wrapper function:
1. Extracts relevant fields from `ParentState` → `PhaseSubgraphState`
2. Invokes the subgraph with its own checkpointer
3. Catches exceptions (CancelledError, MaxTokenBudgetExceeded)
4. Maps subgraph output → `ParentState` update

---

## Phase 0: Foundation — Subgraph Infrastructure (Days 1-2)

Before converting any phase, build the shared infrastructure that all subgraphs will use.

### Task 0.1: Define phase-specific subgraph state schemas

**Files:**
- Create: `spine/workflow/subgraph_state.py`

Each phase gets its own TypedDict. The schemas share common fields but can diverge where needed.

```python
# spine/workflow/subgraph_state.py
"""Per-phase subgraph state schemas for the SPINE orchestrator.

Each subgraph has its own TypedDict so DA agent message history
and phase-internal state don't leak into the parent graph's state.
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class BaseSubgraphState(TypedDict, total=False):
    """Fields shared by all phase subgraphs."""
    phase: str
    work_id: str
    work_type: str
    description: str
    workspace_root: str
    retry_count: int
    feedback: list
    messages: Annotated[list, add_messages]
    artifacts_output: dict  # {filename: content} — what this phase produced
    phase_status: str       # "success" | "needs_review" | "error"


class SpecifySubgraphState(BaseSubgraphState, total=False):
    """SPECIFY phase — produces specification.md."""
    pass  # No additional fields needed


class PlanSubgraphState(BaseSubgraphState, total=False):
    """PLAN phase — reads spec, produces plan.md."""
    spec_path: str


class TasksSubgraphState(BaseSubgraphState, total=False):
    """TASKS phase — reads plan, produces tasks.md + slice-*.md."""
    plan_path: str
    spec_path: str  # Only for spec/critical_spec workflows
    skill_paths: list[str]


class ImplementSubgraphState(BaseSubgraphState, total=False):
    """IMPLEMENT phase — reads tasks, writes code."""
    tasks_path: str


class VerifySubgraphState(BaseSubgraphState, total=False):
    """VERIFY phase — confirms implementation."""
    tasks_path: str
    spec_path: str | None  # Only for spec/critical_spec workflows
    plan_path: str | None


class CriticSubgraphState(BaseSubgraphState, total=False):
    """CRITIC phase — reviews a preceding phase's output."""
    reviewed_phase: str
    reviewed_phase_path: str
```

### Task 0.2: Define parent orchestrator state schema

**Files:**
- Modify: `spine/models/state.py`

Shrink `WorkflowState` to routing-only fields. Add `PhaseResult` as a lightweight summary.

```python
# spine/models/state.py (new fields to add)

from typing import NotRequired

class PhaseResult(TypedDict, total=False):
    """Lightweight summary of a phase subgraph's output."""
    phase: str
    status: str  # "success" | "needs_review" | "error"
    artifact_count: int
    artifact_names: list[str]
    error: str | None


# WorkflowState shrinks to:
class WorkflowState(TypedDict, total=False):
    """Parent orchestrator state — routing and coordination only."""
    work_id: str
    work_type: str
    description: str
    current_phase: str
    phase_index: int
    retry_count: Annotated[dict, _merge_dicts]
    max_retries: int
    phase_results: Annotated[dict, _merge_dicts]  # phase → PhaseResult
    feedback: Annotated[list, operator.add]
    status: str
    prompt_request: dict | None
    critic_reviewing: str
    workspace_root: str
    needs_review_phase: str | None  # Which phase needs human review (for interrupt)
```

### Task 0.3: Build subgraph wrapper factory

**Files:**
- Create: `spine/workflow/subgraph_wrapper.py`

A factory function that creates a LangGraph node wrapper for any phase subgraph. Handles state mapping, exception isolation, and timeout.

```python
# spine/workflow/subgraph_wrapper.py
"""Factory for wrapping phase subgraphs as parent graph nodes.

Each wrapper:
1. Maps ParentState → SubgraphState
2. Invokes the subgraph with its own checkpointer + timeout
3. Catches CancelledError, MaxTokenBudgetExceeded, and other exceptions
4. Maps subgraph output → ParentState update
"""

import asyncio
import logging
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from spine.models.state import WorkflowState
from spine.agents.retry import MaxTokenBudgetExceeded

logger = logging.getLogger(__name__)

# Per-phase timeout overrides (seconds). Default: _DEFAULT_PHASE_TIMEOUT.
_PHASE_TIMEOUTS: dict[str, int] = {
    "specify": 600,    # 10 min — produces a spec doc
    "plan": 600,       # 10 min — produces a plan doc
    "tasks": 900,      # 15 min — exploration + decomposition + subagents
    "implement": 1800, # 30 min — code generation + file writes + tests
    "verify": 600,     # 10 min — review + file reads
    "critic": 300,     # 5 min — review existing output
}
_DEFAULT_PHASE_TIMEOUT = 900
```

```python
def make_subgraph_node(
    subgraph: StateGraph,
    phase_name: str,
    state_mapper: Callable[[WorkflowState, RunnableConfig | None], dict],
    result_mapper: Callable[[dict, WorkflowState], dict[str, Any]],
) -> Callable:
    """Create a LangGraph node that wraps a phase subgraph.

    Args:
        subgraph: Compiled StateGraph for this phase.
        phase_name: Phase enum value (e.g. "verify").
        state_mapper: Function (parent_state, config) → subgraph_input.
        result_mapper: Function (subgraph_output, parent_state) → parent_state_update.

    Returns:
        An async node function with signature (WorkflowState, config) → dict.
    """

    async def subgraph_node(
        parent_state: WorkflowState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        work_id = parent_state.get("work_id", "unknown")
        timeout = _PHASE_TIMEOUTS.get(phase_name, _DEFAULT_PHASE_TIMEOUT)

        logger.info(f"[{work_id}] [{phase_name}] subgraph starting (timeout={timeout}s)")

        try:
            # Map parent state to subgraph input
            subgraph_input = state_mapper(parent_state, config)

            # Build subgraph config with its own thread_id for independent checkpointing
            subgraph_config = {
                **(config or {}),
                "configurable": {
                    **(config.get("configurable", {}) if config else {}),
                    "thread_id": f"{work_id}_{phase_name}",
                },
            }

            # Invoke with timeout
            result = await asyncio.wait_for(
                subgraph.ainvoke(subgraph_input, subgraph_config),
                timeout=timeout,
            )

            # Map subgraph result back to parent state
            parent_update = result_mapper(result, parent_state)
            logger.info(
                f"[{work_id}] [{phase_name}] subgraph completed: "
                f"phase_status={result.get('phase_status', 'unknown')}"
            )
            return parent_update

        except asyncio.TimeoutError:
            logger.error(
                f"[{work_id}] [{phase_name}] subgraph timed out after {timeout}s"
            )
            return _error_update(parent_state, phase_name, f"Timed out after {timeout}s")

        except asyncio.CancelledError:
            logger.error(f"[{work_id}] [{phase_name}] subgraph cancelled")
            return _error_update(
                parent_state,
                phase_name,
                "Cancelled — subgraph did not complete. Prior phases preserved.",
            )

        except MaxTokenBudgetExceeded as e:
            logger.error(f"[{work_id}] [{phase_name}] token budget exceeded: {e}")
            return _needs_review_update(
                parent_state,
                phase_name,
                str(e),
                suggestions=[
                    "Reduce task scope or break into smaller work items",
                    "Use a cheaper model for this phase",
                ],
            )

        except Exception as e:
            logger.error(
                f"[{work_id}] [{phase_name}] subgraph failed: {e}", exc_info=True
            )
            return _error_update(parent_state, phase_name, str(e))

    # Name for LangSmith Studio / debug
    subgraph_node.__name__ = f"{phase_name}_subgraph"
    return subgraph_node


def _error_update(
    state: WorkflowState, phase: str, error: str
) -> dict[str, Any]:
    """Build a parent state update for subgraph errors."""
    return {
        "current_phase": phase,
        "status": "needs_review",
        "prompt_request": None,
        "feedback": [
            {
                "status": "needs_review",
                "tier": "structural",
                "reason": f"[{phase}] subgraph error: {error}",
                "suggestions": ["Review logs and retry", "Reduce scope"],
            }
        ],
        "phase_results": {
            phase: {
                "phase": phase,
                "status": "error",
                "artifact_count": 0,
                "artifact_names": [],
                "error": error,
            }
        },
    }


def _needs_review_update(
    state: WorkflowState, phase: str, reason: str, suggestions: list[str] | None = None
) -> dict[str, Any]:
    """Build a parent state update for needs_review."""
    return {
        "current_phase": phase,
        "status": "needs_review",
        "prompt_request": None,
        "feedback": [
            {
                "status": "needs_review",
                "tier": "structural",
                "reason": reason,
                "suggestions": suggestions or [],
            }
        ],
        "phase_results": {
            phase: {
                "phase": phase,
                "status": "needs_review",
                "artifact_count": 0,
                "artifact_names": [],
                "error": reason,
            }
        },
        "needs_review_phase": phase,
    }
```

### Task 0.4: Build result mapper helpers

**Files:**
- Modify: `spine/workflow/subgraph_wrapper.py`

State mapper functions are phase-specific (they know what fields each subgraph needs). Result mappers follow a common pattern.

```python
def make_success_result_mapper(phase: str) -> Callable:
    """Create a standard result mapper for a successful phase completion.

    Extracts artifacts_output from the subgraph result and maps to the
    parent graph's phase_results and status fields.
    """

    def map_success(subgraph_result: dict, parent_state: dict) -> dict[str, Any]:
        artifacts = subgraph_result.get("artifacts_output", {})
        artifact_names = list(artifacts.keys()) if isinstance(artifacts, dict) else []

        return {
            "current_phase": phase,
            "status": "running",
            "prompt_request": None,
            "phase_results": {
                phase: {
                    "phase": phase,
                    "status": "success",
                    "artifact_count": len(artifact_names),
                    "artifact_names": artifact_names,
                    "error": None,
                }
            },
            "artifacts": {
                phase: {
                    name: content[:_MAX_ARTIFACT_STATE_CHARS]
                    for name, content in artifacts.items()
                    if isinstance(content, str)
                }
            },
        }

    return map_success
```

### Task 0.5: Migrate artifact handling to parent graph

**Files:**
- Modify: `spine/work/dispatcher.py`

Currently `submit_work()` persists artifacts from node outputs. With subgraphs, each subgraph handles its own artifact persistence via `materialize_phase_artifacts()`. The parent graph only needs `phase_results` summaries. Remove the in-dispatcher artifact persistence loop and delegate to subgraphs.

### Task 0.6: Write foundation tests

**Files:**
- Create: `tests/unit/test_subgraph_state.py`
- Create: `tests/unit/test_subgraph_wrapper.py`

Tests for:
- State schemas validate correctly (required fields, defaults)
- State mapper transforms parent → subgraph correctly
- Result mapper transforms subgraph → parent correctly
- Error update includes proper feedback and phase_results
- Timeout update sets needs_review, not error
- CancelledError update sets needs_review (preserves prior phases)

---

## Phase 1: Proof of Concept — VERIFY Subgraph (Days 3-4)

Convert VERIFY as the PoC because:
- It runs last — failure doesn't waste prior phase work
- Simplest agent (reads files, produces report, no code writes)
- Best isolation: VERIFY subgraph failing should never affect IMPLEMENT artifacts

### Task 1.1: Build verify subgraph

**Files:**
- Create: `spine/workflow/subgraphs/__init__.py`
- Create: `spine/workflow/subgraphs/verify_subgraph.py`

Extract `call_verify()` logic into a compiled StateGraph with two internal nodes:
1. `run_agent` — builds the verify agent, invokes it with retry
2. `save_artifacts` — scans disk for artifacts, saves to `artifacts_output`

```python
# spine/workflow/subgraphs/verify_subgraph.py
"""VERIFY phase as a LangGraph subgraph."""

from langgraph.graph import StateGraph, START, END

from spine.workflow.subgraph_state import VerifySubgraphState


def build_verify_subgraph() -> StateGraph:
    """Build the VERIFY phase subgraph.

    The subgraph has two nodes:
    1. run_agent: Builds and invokes the verify Deep Agent.
    2. save_artifacts: Scans disk for artifacts written by the agent
       and populates artifacts_output.
    """
    builder = StateGraph(VerifySubgraphState)

    builder.add_node("run_agent", _verify_agent_node)
    builder.add_node("save_artifacts", _save_verify_artifacts)

    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder.compile()
```

```python
async def _verify_agent_node(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the verify Deep Agent within the subgraph."""
    from spine.agents.verify_agent import build_verify_agent
    from spine.agents.helpers import extract_response
    from spine.agents.retry import ainvoke_with_retry
    from spine.agents.context import build_context
    from spine.agents.artifacts import _artifact_path
    from spine.models.enums import PhaseName

    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    agent = build_verify_agent(state, config)

    has_spec = "spec" in work_type
    tasks_path = _artifact_path(work_id, PhaseName.TASKS.value)
    impl_path = _artifact_path(work_id, PhaseName.IMPLEMENT.value)

    prompt_lines = [
        "Verify that the implementation meets the requirements. ...",
        # (same prompt as current call_verify)
    ]
    prompt = "\n".join(prompt_lines)

    ctx = build_context(state, PhaseName.VERIFY)

    result = await ainvoke_with_retry(
        agent,
        {"messages": [{"role": "user", "content": prompt}]},
        phase_name=PhaseName.VERIFY.value,
        work_id=work_id,
        work_type=work_type,
        context=ctx,
    )

    return {"messages": result.get("messages", []), "agent_response": extract_response(result)}


async def _save_verify_artifacts(
    state: VerifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the verify agent to disk and state."""
    from spine.agents.artifacts import (
        scan_artifact_dir,
        materialize_phase_artifacts,
    )
    from spine.models.enums import PhaseName

    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")

    _MAX_ARTIFACT_STATE_CHARS = 500

    # Scan what the agent wrote to disk
    disk_artifacts = scan_artifact_dir(
        workspace_root, work_id, PhaseName.VERIFY.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    # Fallback: use agent response
    if not disk_artifacts:
        verify_content = agent_response
        if not verify_content or len(verify_content.strip()) < 20:
            verify_content = (
                "Verification could not produce a meaningful report. "
                "The agent returned insufficient output. Manual review required."
            )
        materialize_phase_artifacts(
            PhaseName.VERIFY.value,
            {"verification.md": verify_content},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"verification.md": verify_content[:_MAX_ARTIFACT_STATE_CHARS]}

    # Determine status
    verify_text = next(iter(disk_artifacts.values()), "") if disk_artifacts else ""
    is_verified = "VERIFIED" in verify_text.upper() or "PASSED" in verify_text.upper()

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if is_verified else "needs_review",
    }
```

### Task 1.2: Wire verify subgraph into parent graph

**Files:**
- Modify: `spine/workflow/compose.py`

Add the verify subgraph as a node instead of `call_verify`. Create the state mapper and result mapper.

```python
# In compose.py, replace:
#   graph.add_node("verify", phase_def.call_fn)
# With:
from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph
from spine.workflow.subgraph_wrapper import make_subgraph_node, make_success_result_mapper

def _verify_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    has_spec = "spec" in parent_state.get("work_type", "")
    return {
        "phase": PhaseName.VERIFY.value,
        "work_id": parent_state.get("work_id", "unknown"),
        "work_type": parent_state.get("work_type", ""),
        "description": parent_state.get("description", ""),
        "workspace_root": parent_state.get("workspace_root", "."),
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.VERIFY.value, 0),
        "feedback": parent_state.get("feedback", []),
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "tasks_path": f".spine/artifacts/{work_id}/tasks",
        "spec_path": f".spine/artifacts/{work_id}/specify" if has_spec else None,
        "plan_path": f".spine/artifacts/{work_id}/plan" if has_spec else None,
    }

verify_subgraph = build_verify_subgraph()
verify_result_mapper = make_success_result_mapper(PhaseName.VERIFY.value)

graph.add_node(
    "verify",
    make_subgraph_node(
        verify_subgraph,
        PhaseName.VERIFY.value,
        _verify_state_mapper,
        verify_result_mapper,
    ),
)
```

### Task 1.3: Update registry to accept subgraphs

**Files:**
- Modify: `spine/workflow/registry.py`
- Modify: `spine/phases/verify.py` (registration)

The registry currently stores `call_fn` (a bare async function). Update it to accept either a call function or a subgraph wrapper. For backward compatibility during migration, support both.

```python
@dataclass
class PhaseDefinition:
    name: str
    call_fn: Callable | None = None  # Legacy — None for subgraph phases
    build_agent_fn: Callable | None = None
    subgraph_node_fn: Callable | None = None  # New — subgraph wrapper
    description: str = ""
```

### Task 1.4: Write VERIFY subgraph tests

**Files:**
- Create: `tests/unit/test_verify_subgraph.py`

Tests:
- `test_verify_subgraph_compiles`: graph compiles without error
- `test_verify_subgraph_has_correct_nodes`: "run_agent" and "save_artifacts" nodes exist
- `test_verify_agent_node_produces_artifacts`: mocking the agent, verifies `artifacts_output` populated
- `test_verify_save_artifacts_sets_status`: verifies `phase_status` is set based on content
- `test_verify_state_mapper`: verifies parent → subgraph state mapping
- `test_verify_result_mapper`: verifies subgraph → parent result mapping
- `test_verify_subgraph_timeout_returns_needs_review`: timeout produces proper error state
- `test_verify_subgraph_cancelled_returns_needs_review`: CancelledError produces proper error

### Task 1.5: Integration test — end-to-end with verify subgraph

**Files:**
- Create: `tests/integration/test_subgraph_workflow.py`

```python
@pytest.mark.asyncio
async def test_quick_workflow_with_verify_subgraph():
    """A quick workflow runs tasks→implement→verify, verify as subgraph."""
    # Build graph with verify subgraph
    # Mock DA agents to return predefined outputs
    # Stream graph, verify all phases complete
    # Verify parent state has phase_results for verify
```

---

## Phase 2: Roll Out to Remaining Phases (Days 5-10)

Once VERIFY works end-to-end, convert the remaining phases following the same pattern.

### Task 2.1: Convert IMPLEMENT to subgraph

**Files:**
- Create: `spine/workflow/subgraphs/implement_subgraph.py`

Three internal nodes:
1. `materialize_prior_artifacts` — writes tasks artifacts to disk
2. `run_agent` — builds and invokes the implement Deep Agent
3. `save_artifacts` — scans disk for implementation artifacts

State mapper passes `tasks_path` to subgraph. Result mapper extracts `artifacts_output`.

### Task 2.2: Convert TASKS to subgraph

**Files:**
- Create: `spine/workflow/subgraphs/tasks_subgraph.py`

Most complex subgraph — tasks has subagents, exploration, and file discovery.

Three internal nodes:
1. `materialize_prior_artifacts` — writes spec/plan to disk (if present)
2. `run_agent` — builds the tasks agent with researchers + RLM
3. `collect_slice_files` — scans disk for `slice-*.md`, preserves per-file artifacts

State mapper conditionally includes `spec_path`/`plan_path` based on `work_type`.

### Task 2.3: Convert SPECIFY to subgraph

**Files:**
- Create: `spine/workflow/subgraphs/specify_subgraph.py`

Two internal nodes:
1. `run_agent` — builds the specify agent
2. `save_artifacts` — writes specification.md

### Task 2.4: Convert PLAN to subgraph

**Files:**
- Create: `spine/workflow/subgraphs/plan_subgraph.py`

Two internal nodes:
1. `run_agent` — builds the plan agent (reads spec)
2. `save_artifacts` — writes plan.md

### Task 2.5: Convert CRITIC to subgraph

**Files:**
- Create: `spine/workflow/subgraphs/critic_subgraph.py`

Critic is unique — it's used multiple times in the same workflow (critic_specify, critic_plan, critic_tasks). The subgraph is parameterized by `reviewed_phase`.

Two internal nodes:
1. `structural_check` — same as current `structural_critic_check()`, no LLM
2. `agent_check` — runs the critic Deep Agent for quality review

```python
def build_critic_subgraph(reviewed_phase: str) -> StateGraph:
    """Build a critic subgraph for a specific reviewed phase."""
    builder = StateGraph(CriticSubgraphState)

    builder.add_node("structural_check", _structural_check_node)
    builder.add_node("agent_check", _agent_check_node)

    builder.add_edge(START, "structural_check")
    builder.add_conditional_edges(
        "structural_check",
        _critic_subgraph_router,
        {"passed": "agent_check", "needs_revision": END, "needs_review": END},
    )
    builder.add_edge("agent_check", END)

    return builder.compile()
```

### Task 2.6: Simplify parent graph compose.py

**Files:**
- Modify: `spine/workflow/compose.py`

After all phases are subgraphs, `compose.py` becomes much simpler:

```python
def build_workflow_graph(work_type: str, checkpointer=None):
    # ...same phase_seq lookup...

    graph = StateGraph(WorkflowState)

    for node_name, reviewed_phase in phase_seq:
        if node_name.startswith(PhaseName.CRITIC.value):
            # Critic — parameterize by reviewed_phase
            critic_subgraph = build_critic_subgraph(reviewed_phase or "unknown")
            graph.add_node(node_name, make_subgraph_node(
                critic_subgraph, node_name,
                _critic_state_mapper(reviewed_phase),
                _critic_result_mapper(reviewed_phase),
            ))
        else:
            # Regular phase subgraph
            subgraph = PHASE_SUBGRAPH_BUILDERS[node_name]()
            graph.add_node(node_name, make_subgraph_node(
                subgraph, node_name,
                PHASE_STATE_MAPPERS[node_name],
                make_success_result_mapper(node_name),
            ))

    # Wire edges (identical to current compose.py)
    # ...same edge wiring with critic routers and artifact gates...
```

### Task 2.7: Write subgraph tests for all phases

**Files:**
- Create: `tests/unit/test_implement_subgraph.py`
- Create: `tests/unit/test_tasks_subgraph.py`
- Create: `tests/unit/test_specify_subgraph.py`
- Create: `tests/unit/test_plan_subgraph.py`
- Create: `tests/unit/test_critic_subgraph.py`

Each test file follows the VERIFY pattern: compilation, node existence, state mapping, result mapping, error handling.

### Task 2.8: Integration tests for all work types

**Files:**
- Modify: `tests/integration/test_subgraph_workflow.py`

Test each work type end-to-end with mocked agents:
- `test_quick_workflow_subgraphs`: tasks → [gate] → implement → verify
- `test_critical_quick_workflow_subgraphs`: tasks → critic → implement → verify
- `test_spec_workflow_subgraphs`: specify → plan → critic → tasks → implement → verify
- `test_critical_spec_workflow_subgraphs`: full 8-node sequence
- `test_subgraph_timeout_isolation`: a timeout in plan doesn't lose spec artifacts
- `test_subgraph_cancelled_isolation`: CancelledError in implement doesn't lose tasks artifacts
- `test_critic_rework_loop_with_subgraphs`: needs_revision routes back to previous subgraph

### Task 2.9: Remove legacy phase call functions

**Files:**
- Modify: `spine/phases/verify.py` — remove `call_verify()` (or keep as thin wrapper)
- Modify: `spine/phases/implement.py` — remove `call_implement()`
- Modify: `spine/phases/tasks.py` — remove `call_tasks()`
- Modify: `spine/phases/specify.py` — remove `call_specify()`
- Modify: `spine/phases/plan.py` — remove `call_plan()`
- Modify: `spine/phases/critic.py` — remove `call_critic()`

Or keep them as thin wrappers that delegate to the subgraph node for backward compatibility during migration. The `_make_critic_node` wrapper in `compose.py` can be removed — each critic instance gets its own subgraph.

---

## Phase 3: True Human-in-the-Loop with interrupt() (Days 11-12)

Replace the `needs_review → END → resume_work()` pattern with LangGraph's native `interrupt()` + `Command(resume=...)`.

### Task 3.1: Add interrupt points between phases

**Files:**
- Modify: `spine/workflow/compose.py`

Add an interrupt node between each phase pair. When a critic or artifact gate sets `needs_review_phase`, the workflow pauses instead of routing to END.

```python
from langgraph.types import interrupt

def _phase_review_node(state: WorkflowState) -> dict:
    """Pause for human review between phases.

    The workflow stops here. A human (via UI or CLI) reviews the current
    state and calls Command(resume={"action": "approve"|"rework", "feedback": "..."}).
    """
    needs_review_phase = state.get("needs_review_phase")
    feedback = state.get("feedback", [])

    # Build a review prompt
    last_fb = feedback[-1] if feedback else {}
    review_info = {
        "phase": needs_review_phase or state.get("current_phase", ""),
        "reason": last_fb.get("reason", "No reason provided"),
        "suggestions": last_fb.get("suggestions", []),
        "artifacts": state.get("phase_results", {}),
    }

    # interrupt() pauses the graph. Human response comes back via Command(resume=...)
    human_decision = interrupt(review_info)

    return {
        "human_feedback": human_decision,
        "needs_review_phase": None,  # Clear the flag
    }
```

### Task 3.2: Update compose.py routing

**Files:**
- Modify: `spine/workflow/compose.py`

Replace `"needs_review": END` in conditional edge maps with routing to a `_phase_review_node`, then a conditional edge from that node:

```python
# After critic node:
graph.add_conditional_edges(
    node_name,
    critic_router,
    {
        "passed": critic_proceed_target,
        "needs_revision": pre_critic,
        "needs_review": "human_review",  # Route to interrupt node, not END
    },
)

# Human review node — interrupt + conditional edge
graph.add_node("human_review", _phase_review_node)
graph.add_conditional_edges(
    "human_review",
    _human_review_router,
    {
        "rework": _get_rework_target(state),  # Back to the failed phase
        "approve": _get_approve_target(state),  # Skip to next phase
        "abort": END,
    },
)
```

### Task 3.3: Update dispatcher for interrupt-based resume

**Files:**
- Modify: `spine/work/dispatcher.py`

`submit_work()` no longer needs `resume_work()`. When a workflow hits `interrupt()`, LangGraph pauses. The UI calls `Command(resume={"action": "rework", "feedback": "add more detail"})` to continue.

```python
async def resume_interrupted_work(
    work_id: str,
    action: str,  # "rework" | "approve" | "abort"
    feedback: str = "",
    config: SpineConfig | None = None,
) -> dict[str, Any]:
    """Resume a workflow that hit an interrupt() for human review.

    Uses LangGraph's Command(resume=...) to signal the interrupt.
    """
    checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
    checkpointer = await checkpoint_store.get_checkpointer()

    # Load existing state from checkpointer
    config_obj = {"configurable": {"thread_id": work_id}}
    state = await checkpointer.aget_tuple(config_obj)

    # Resume with Command
    from langgraph.types import Command

    # Re-build the graph and stream with resume command
    graph = build_workflow_graph(state.config["configurable"]["work_type"])
    command = Command(resume={"action": action, "feedback": feedback})

    async for chunk in graph.astream(command, config_obj, ...):
        # Same streaming logic as submit_work()
        ...
```

### Task 3.4: Update UI for interrupt-based resume

**Files:**
- Modify: `spine/ui/_pages/human_review.py`
- Modify: `spine/ui/_pages/work_detail.py`

The UI's "Resume with feedback" button calls `resume_interrupted_work()` instead of `resume_work()`. The review page shows the `interrupt()` message with artifact summaries.

### Task 3.5: Write interrupt tests

**Files:**
- Create: `tests/unit/test_interrupt_workflow.py`

- `test_interrupt_fires_on_needs_review`: verify interrupt() is called
- `test_resume_with_approve`: Command(resume={"action": "approve"}) continues past the flagged phase
- `test_resume_with_rework`: Command(resume={"action": "rework"}) routes back to redo
- `test_resume_with_abort`: Command(resume={"action": "abort"}) routes to END

---

## Phase 4: Per-Phase Checkpointing and Timeout Isolation (Days 13-14)

### Task 4.1: Per-subgraph checkpointers

**Files:**
- Modify: `spine/workflow/compose.py`
- Modify: `spine/workflow/subgraph_wrapper.py`

Each subgraph gets its own SQLite checkpointer file to enable independent checkpoint persistence and mid-phase resume.

```python
# In subgraph_wrapper.py
from spine.persistence.checkpoint import CheckpointStore

async def _get_subgraph_checkpointer(work_id: str, phase: str) -> BaseCheckpointSaver:
    """Get a dedicated checkpointer for a phase subgraph."""
    store = CheckpointStore(db_path=f".spine/checkpoints/{work_id}/{phase}.db")
    return await store.get_checkpointer()
```

### Task 4.2: Per-phase timeout configuration

**Files:**
- Modify: `spine/config.py`
- Modify: `.spine/config.yaml`

Add timeout configuration to SpineConfig:

```yaml
# .spine/config.yaml
spine:
  timeouts:
    specify: 600
    plan: 600
    tasks: 900
    implement: 1800
    verify: 600
    critic: 300
  default_timeout: 900
```

```python
# spine/config.py
@dataclass
class SpineConfig:
    phase_timeouts: dict[str, int] = field(default_factory=lambda: {
        "specify": 600,
        "plan": 600,
        "tasks": 900,
        "implement": 1800,
        "verify": 600,
        "critic": 300,
    })
    default_timeout: int = 900
```

### Task 4.3: Mid-phase resume from checkpoint

**Files:**
- Create: `spine/work/subgraph_resume.py`

If a subgraph fails at internal step 47/50, resume from that checkpoint rather than restarting the subgraph.

```python
async def resume_subgraph(
    work_id: str,
    phase: str,
    config: SpineConfig,
) -> dict[str, Any]:
    """Resume a failed phase subgraph from its last checkpoint."""
    # Load subgraph-specific checkpointer
    store = CheckpointStore(db_path=f".spine/checkpoints/{work_id}/{phase}.db")
    checkpointer = await store.get_checkpoint()

    # Get last saved state
    config_obj = {"configurable": {"thread_id": f"{work_id}_{phase}"}}
    state = await checkpointer.aget_tuple(config_obj)

    # Re-invoke subgraph from checkpoint
    subgraph = PHASE_SUBGRAPH_BUILDERS[phase]()
    result = await subgraph.ainvoke(None, config_obj)  # None = resume from checkpoint

    return result
```

---

## Phase 5: Agent Quality Fixes (Parallel Track, Days 1-10)

These fixes are independent of the subgraph decomposition and can be done in parallel.

### Task 5.1: Thinking model content fallback

**Files:**
- Modify: `spine/agents/helpers.py`

`extract_response()` reads only `last.content`. For thinking models (qwen3.6-27b), `content` is empty during tool-calling turns while `reasoning_content` has the actual output. Fall back to `reasoning_content` when `content` is empty.

```python
def extract_response(result: dict) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        if content and len(content.strip()) > 10:
            return content.strip()
        # Fall back to reasoning_content for thinking models
        reasoning = (
            getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
            or ""
        )
        if reasoning and len(reasoning.strip()) > 10:
            return reasoning.strip()
    return ""
```

### Task 5.2: Retry with fresh agent (not exhausted agent)

**Files:**
- Modify: `spine/phases/tasks.py`

When the tasks agent produces empty output, the retry logic calls `ainvoke_with_retry()` with the **same agent instance**. The agent's internal step budget is already exhausted, so the retry produces 0 tokens. Fix: create a fresh agent with no prior conversation state for the retry.

```python
# In call_tasks() retry path:
if not tasks_content or len(tasks_content.strip()) < 50:
    # Create a FRESH agent, not reuse the exhausted one
    retry_agent = build_tasks_agent(state, config)
    retry_result = await ainvoke_with_retry(
        retry_agent,
        {"messages": [{"role": "user", "content": retry_prompt}]},
        ...
    )
```

### Task 5.3: Verify can find implement artifacts

**Files:**
- Modify: `spine/agents/artifacts.py`

The artifact persistence gap: agent writes files via `write_file` to paths that don't match what the next phase expects. After each subgraph run, validate artifact paths:

```python
def validate_artifact_dir(
    workspace_root: str, work_id: str, phase: str
) -> bool:
    """Return True if artifacts exist at the expected path."""
    expected = Path(workspace_root) / ".spine" / "artifacts" / work_id / phase
    if not expected.exists():
        return False
    files = list(expected.glob("*"))
    if not files:
        return False
    logger.info(f"Validated {len(files)} artifacts at {expected}")
    return True
```

### Task 5.4: Budget enforcement for all code paths

**Files:**
- Audit: all phase files for `ainvoke_with_retry` calls
- Ensure `work_type=work_type` is passed to every call (some may be missing)
- Ensure `MaxTokenBudgetExceeded` catch block exists in every phase's try/except

---

## Migration Strategy

### Rollback Safety

Every phase conversion is reversible:
1. The old `call_*()` function remains in `spine/phases/` during migration
2. `compose.py` can use either the old call function or the new subgraph via a feature flag
3. `PhaseDefinition` supports both `call_fn` and `subgraph_node_fn`

```python
# compose.py
_SUBGRAPH_ENABLED: dict[str, bool] = {
    "verify": True,   # Phase 1: enable verify subgraph
    "implement": False,
    "tasks": False,
    "specify": False,
    "plan": False,
    "critic": False,
}

for node_name, reviewed_phase in phase_seq:
    if _SUBGRAPH_ENABLED.get(node_name, False):
        # Use subgraph
        subgraph = PHASE_SUBGRAPH_BUILDERS[node_name]()
        graph.add_node(node_name, make_subgraph_node(...))
    else:
        # Use legacy call function
        phase_def = registry.require(node_name)
        graph.add_node(node_name, phase_def.call_fn)
```

### Validation Gates

Before moving to the next phase:
1. All existing tests pass (unit + integration)
2. New subgraph tests pass
3. Quick workflow end-to-end succeeds with mocked agents
4. LangSmith trace shows subgraph boundaries (checkpoint_ns reflects subgraph nesting)
5. Manual test: `spine run --type quick "Fix the col2 bug in human_review.py"` completes

### Order of Rollout

1. **VERIFY** (Phase 1) — lowest risk, runs last
2. **CRITIC** (Phase 2) — parameterized, reused by all workflows
3. **SPECIFY** (Phase 2) — simple, first in spec workflows
4. **PLAN** (Phase 2) — simple, reads spec
5. **TASKS** (Phase 2) — most complex, has subagents
6. **IMPLEMENT** (Phase 2) — code generation, test runner
7. **interrupt()** (Phase 3) — only after all subgraphs stable
8. **Per-phase checkpoints** (Phase 4) — only after interrupt() works

---

## Files Changed Summary

### New Files
| File | Purpose |
|------|---------|
| `spine/workflow/subgraphs/__init__.py` | Subgraph package |
| `spine/workflow/subgraphs/verify_subgraph.py` | VERIFY phase subgraph |
| `spine/workflow/subgraphs/implement_subgraph.py` | IMPLEMENT phase subgraph |
| `spine/workflow/subgraphs/tasks_subgraph.py` | TASKS phase subgraph |
| `spine/workflow/subgraphs/specify_subgraph.py` | SPECIFY phase subgraph |
| `spine/workflow/subgraphs/plan_subgraph.py` | PLAN phase subgraph |
| `spine/workflow/subgraphs/critic_subgraph.py` | CRITIC phase subgraph (parameterized) |
| `spine/workflow/subgraph_state.py` | Per-phase subgraph TypedDict schemas |
| `spine/workflow/subgraph_wrapper.py` | Subgraph wrapper factory (state mapping, timeout, error isolation) |
| `spine/work/subgraph_resume.py` | Mid-phase checkpoint resume |
| `tests/unit/test_subgraph_state.py` | State schema tests |
| `tests/unit/test_subgraph_wrapper.py` | Wrapper factory tests |
| `tests/unit/test_verify_subgraph.py` | VERIFY subgraph tests |
| `tests/unit/test_implement_subgraph.py` | IMPLEMENT subgraph tests |
| `tests/unit/test_tasks_subgraph.py` | TASKS subgraph tests |
| `tests/unit/test_specify_subgraph.py` | SPECIFY subgraph tests |
| `tests/unit/test_plan_subgraph.py` | PLAN subgraph tests |
| `tests/unit/test_critic_subgraph.py` | CRITIC subgraph tests |
| `tests/unit/test_interrupt_workflow.py` | Interrupt workflow tests |
| `tests/integration/test_subgraph_workflow.py` | End-to-end subgraph workflow tests |

### Modified Files
| File | Changes |
|------|---------|
| `spine/models/state.py` | Add `PhaseResult`, `needs_review_phase`, remove `artifacts` from parent (moved to phase_results), keep `_merge_dicts` and `_merge_artifacts` for backward compat |
| `spine/workflow/compose.py` | Use subgraph nodes instead of bare call functions; add `_SUBGRAPH_ENABLED` feature flags; add `human_review` interrupt node in Phase 3 |
| `spine/workflow/registry.py` | Add `subgraph_node_fn` field to `PhaseDefinition` |
| `spine/workflow/studio.py` | Update to build graphs with subgraph nodes |
| `spine/work/dispatcher.py` | Remove inline artifact persistence loop (delegated to subgraphs); add `resume_interrupted_work()` in Phase 3 |
| `spine/phases/verify.py` | Keep `call_verify()` as backward-compat wrapper; remove agent-building logic (moved to subgraph) |
| `spine/phases/implement.py` | Same — keep wrapper, move logic |
| `spine/phases/tasks.py` | Same + fix retry with fresh agent (Task 5.2) |
| `spine/phases/specify.py` | Same |
| `spine/phases/plan.py` | Same |
| `spine/phases/critic.py` | Same |
| `spine/agents/helpers.py` | Fix `extract_response()` for thinking models (Task 5.1) |
| `spine/agents/artifacts.py` | Add `validate_artifact_dir()` (Task 5.3) |
| `spine/config.py` | Add `phase_timeouts` and `default_timeout` fields (Task 4.2) |
| `spine/ui/_pages/human_review.py` | Update resume button to use `resume_interrupted_work()` |
| `spine/ui/_pages/work_detail.py` | Show subgraph checkpoints; update resume flow |
| `.spine/config.yaml` | Add timeouts section |
| `langgraph.json` | Add subgraph entry points for LangSmith Studio |
| `pyproject.toml` | No new dependencies (all LangGraph built-in) |

### Unchanged Files
| File | Reason |
|------|--------|
| `spine/agents/factory.py` | No changes — `build_phase_agent()` is still called inside subgraph nodes |
| `spine/agents/retry.py` | No changes — `ainvoke_with_retry()` still used inside subgraphs |
| `spine/agents/context.py` | No changes — `SpineContext` still passed at invoke time |
| `spine/agents/skills_resolver.py` | No changes |
| `spine/agents/backend.py` | No changes |
| `spine/agents/interpreter.py` | No changes |
| `spine/workflow/critic_review.py` | No changes — structural check and critic router logic unchanged, just moved inside subgraph |
| `spine/workflow/artifact_gate.py` | No changes — gate stays in parent graph between subgraphs |
| `spine/models/enums.py` | No changes |

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Subgraph invocation increases latency (extra graph overhead) | Medium | LangGraph subgraph-as-node is designed for this — overhead is minimal. The DA agent loop is already a nested graph; adding one more nesting level is negligible compared to LLM latency. |
| State serialization between parent and subgraph becomes a bottleneck | Low | PhaseResults are lightweight dicts. Full artifact content stays on disk. Only paths and summaries cross the boundary. |
| Interrupt-based resume requires UI changes that break existing users | Medium | Keep legacy `resume_work()` as fallback during migration. Add interrupt-based resume as new endpoint. |
| Per-phase checkpoints increase disk usage | Low | SQLite WAL files are small (~KB per checkpoint). Only recent checkpoints are kept. |
| Migration takes too long — 15 days estimated | Medium | Each phase conversion is independent. Can stop after any phase and ship partial benefits. VERIFY-only (Phase 1) already provides CancelledError isolation for the most common failure mode. |

---

## Success Metrics

After Phase 1 (VERIFY subgraph):
- **CancelledError in verify no longer loses implement artifacts** — verify fails in its own subgraph, parent state has implement's PhaseResult
- **LangSmith traces show verify as a subgraph boundary** — checkpoint_ns shows `verify:uuid` instead of just `verify`

After Phase 2 (all subgraphs):
- **Any single phase failure preserved prior phases** — CancelledError in plan doesn't lose spec
- **Isolated timeouts** — plan gets 10 minutes, implement gets 30 minutes
- **Trace success rate > 50%** — up from current 10%

After Phase 3 (interrupt):
- **Human review is instant** — no graph restart from scratch
- **No more needs_review → END → resume_work()** — replaced by interrupt() + Command(resume)

After Phase 4 (checkpoints):
- **Mid-phase resume works** — a plan subgraph that fails at step 47 can resume from step 46
- **Zero work loss on transient errors** — RemoteProtocolError in tasks subgraph can resume from last checkpoint

---

## Verification

After each phase, verify:

```bash
# Unit tests
pytest tests/unit/test_subgraph_*.py -v

# Integration tests
pytest tests/integration/test_subgraph_workflow.py -v

# All existing tests still pass
pytest tests/ -v

# Lint
ruff check spine/ tests/
ruff format --check spine/ tests/

# Type check
mypy spine/workflow/subgraph*.py

# Manual end-to-end
spine run --type quick "Add a docstring to spine/workflow/compose.py"

# LangSmith trace inspection
# Verify subgraph boundaries visible in traces
# Verify CancelledError in one subgraph doesn't lose prior phase artifacts
```
