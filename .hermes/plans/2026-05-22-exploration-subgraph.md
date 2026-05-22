# Exploration Subgraph — Implementation Plan

> **For Hermes:** Implement this plan task-by-task, committing after each.

**Goal:** Replace the linear `run_agent → save_artifacts` specify/plan subgraphs with a multi-node exploration loop: `research_manager ⇄ [explore×N → aggregate → sufficiency] → synthesize → save_artifacts`.

**Architecture:** A new `ExplorationSubgraphState` TypedDict with an `operator.add` reducer for accumulated findings. The `research_manager` is a lightweight single LLM call (no Deep Agent tools). Parallel `explore` nodes run existing `researcher` subagents via the LangGraph `Send` API. A `sufficiency` gate (conditional edge function, not prompted) routes back for more exploration or forward to synthesis. The synthesis node is a full Deep Agent that writes the spec/plan artifact. Feature-flagged behind `_SUBGRAPH_ENABLED["specify"]` (already exists).

**Tech Stack:** LangGraph StateGraph, Send API, existing `build_subagent_spec`, `build_phase_agent`, `ResearchFindings` model, `operator.add` reducer.

---

### Task 1: Add ExplorationSubgraphState to subgraph_state.py

**Objective:** Add the new state TypedDict with accumulator fields.

**File:** Modify `spine/workflow/subgraph_state.py`

Add after the existing state classes:

```python
from operator import add as _op_add


class ExplorationSubgraphState(BaseSubgraphState, total=False):
    """Multi-node exploration → synthesis subgraph (SPECIFY, PLAN).

    Accumulates findings across parallel explore rounds via ``operator.add``,
    then routes to synthesis when research is sufficient.
    """

    # Exploration loop control
    research_round: int  # Current round number (0-based)
    max_rounds: int  # Safety valve — max exploration rounds (default 3)
    manager_decision: str  # "explore" | "done" — set by research_manager

    # Accumulated research (operator.add reducer merges per-round findings)
    topics: Annotated[list[str], _op_add]  # Areas being explored this round
    findings: Annotated[list[dict], _op_add]  # ResearchFindings dicts from explore nodes

    # Synthesis output
    agent_response: str  # Final spec/plan text from synthesizer
    plan_json: str  # Only used by PLAN — raw plan.json content
    execution_waves: list  # Only used by PLAN — computed waves
```

**Verification:** `python -c "from spine.workflow.subgraph_state import ExplorationSubgraphState; print('OK')"`

---

### Task 2: Create exploration agent builders

**Objective:** Build lightweight agents for `research_manager` and `explore` nodes. The manager gets a single LLM call to decide topics. The explore node runs the existing `researcher` subagent.

**Files:** Create `spine/agents/exploration_agents.py`

The file provides two async functions:

1. `run_research_manager(state, config)` — single LLM `ainvoke()` call. Prompt asks: "Given the work description and accumulated findings so far, what areas still need research? Return JSON with {decision: "explore"|"done", topics: ["area1", "area2"]}." The research manager has NO tools — it's a pure reasoning call.

2. `run_explore_node(state, config, topic)` — builds a `researcher` subagent via `build_subagent_spec("researcher", ...)`, injects it via `build_phase_agent(is_subagent=True, ...)`, and invokes it with the topic. Returns `ResearchFindings` structured output.

**Key design:** 
- `run_research_manager` uses `model.ainvoke([SystemMessage(prompt), HumanMessage(context)])` — no agent loop, no tools. One turn.
- `run_explore_node` reuses the existing subagent infrastructure entirely — same `build_subagent_spec`, same `_inject_mcp_tools`, same structured output.

```python
"""SPINE exploration agents — research_manager and explore nodes.

These are lightweight agents for the exploration subgraph.
They do NOT use the full middleware stack — they are single-purpose
LLM calls, not orchestrator agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from spine.agents.helpers import resolve_model
from spine.agents.subagents import (
    build_subagent_spec,
    SUBAGENT_RESPONSE_MODELS,
)
from spine.agents.factory import build_phase_agent
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)

# ── Research manager prompt ──────────────────────────────────────────

_RESEARCH_MANAGER_SYSTEM = """\
You are a research planning assistant. Your job is to decide what areas
of a codebase still need investigation before writing a specification or plan.

Given:
1. The work description
2. A list of research topics already explored
3. The findings accumulated so far

Decide:
- Are we done? (decision: "done") — all key areas have been investigated
- Or do we need more? (decision: "explore") — return the next 2-4 topics

Respond with ONLY a JSON object:
{"decision": "explore" | "done", "topics": ["area1", "area2"]}

Rules:
- Never return more than 4 topics in a single round.
- If you've already explored a topic, don't return it again.
- Prefer targeted, specific topics over broad ones.
- If the work description is self-contained (no codebase needed), decide "done".
"""


async def run_research_manager(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the research manager — a single LLM call to decide next topics.

    Args:
        state: The ExplorationSubgraphState.
        config: LangGraph runtime config.

    Returns:
        Dict with ``manager_decision`` and ``topics`` keys.
    """
    description = state.get("description", "")
    existing_topics = state.get("topics", [])
    findings = state.get("findings", [])
    round_num = state.get("research_round", 0)
    max_rounds = state.get("max_rounds", 3)
    work_id = state.get("work_id", "unknown")

    # Safety valve — if we've hit max rounds, force done
    if round_num >= max_rounds:
        logger.info("[%s] Research manager: max rounds (%d) reached — forcing done",
                     work_id, max_rounds)
        return {"manager_decision": "done", "topics": []}

    model = resolve_model(config, session_id=work_id, phase="exploration/manager")

    # Build the context for the manager
    findings_summary = _summarize_findings(findings)
    context = (
        f"## Work Description\n{description}\n\n"
        f"## Round\n{round_num + 1} of max {max_rounds}\n\n"
        f"## Topics Already Explored\n{json.dumps(existing_topics)}\n\n"
        f"## Findings So Far\n{findings_summary}\n\n"
        f"Decide: are we done, or do we need more research?"
    )

    try:
        response = await model.ainvoke([
            SystemMessage(content=_RESEARCH_MANAGER_SYSTEM),
            HumanMessage(content=context),
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        result = json.loads(raw)
        decision = result.get("decision", "done")
        topics = result.get("topics", [])
        if not isinstance(topics, list):
            topics = []
        logger.info("[%s] Research manager: decision=%s topics=%s",
                     work_id, decision, topics)
        return {"manager_decision": decision, "topics": topics}
    except Exception as e:
        logger.warning("[%s] Research manager LLM call failed: %s — defaulting to done",
                       work_id, e)
        return {"manager_decision": "done", "topics": []}


def _summarize_findings(findings: list[dict]) -> str:
    """Create a compact summary of accumulated research findings."""
    if not findings:
        return "(no findings yet)"
    parts = []
    for i, f in enumerate(findings):
        if isinstance(f, dict):
            summary = f.get("summary", "")
            patterns = f.get("patterns", [])
            deps = f.get("dependencies", [])
            entry = f"Finding {i+1}: {summary[:200]}"
            if patterns:
                entry += f"\n  Patterns: {', '.join(patterns[:5])}"
            if deps:
                entry += f"\n  Dependencies: {', '.join(deps[:5])}"
            parts.append(entry)
    return "\n\n".join(parts[:10])  # Keep compact


# ── Explore node ─────────────────────────────────────────────────────


async def run_explore_node(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """Run an explore node — invokes a researcher subagent for one topic.

    Uses ``build_subagent_spec("researcher", ...)`` to get the full
    subagent configuration (model, tools, MCP, response_format), then
    wraps it in a minimal agent for invocation.

    Args:
        state: The ExplorationSubgraphState.
        config: LangGraph runtime config.
        topic: The specific research topic (set by Send API).

    Returns:
        Dict with ``findings`` key containing a list with one
        ResearchFindings dict (merged by operator.add).
    """
    from spine.agents.retry import ainvoke_with_retry

    work_id = state.get("work_id", "unknown")
    topic_str = topic or "general codebase investigation"

    logger.info("[%s] Explore node: researching topic=%r", work_id, topic_str)

    try:
        # Build the researcher subagent
        subagent_spec = build_subagent_spec(
            name="researcher",
            phase=PhaseName.SPECIFY,  # Use SPECIFY for model resolution
            state=state,  # type: ignore[arg-type]
            config=config,
        )

        # Build a minimal agent for this subagent
        agent = build_phase_agent(
            state=state,  # type: ignore[arg-type]
            config=config,
            phase=PhaseName.SPECIFY,
            system_prompt=subagent_spec["system_prompt"],
            is_subagent=True,
            extra_tools=subagent_spec.get("tools", []),
            response_format=subagent_spec.get("response_format"),
        )

        prompt = (
            f"## Research Topic\n{topic_str}\n\n"
            f"Investigate this specific area of the codebase. "
            f"Use MCP tools first for structural navigation, "
            f"then fall back to read_file/search_codebase if needed. "
            f"Return your findings as a structured ResearchFindings response."
        )

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name="explore",
            work_id=work_id,
        )

        # Extract findings from the result
        findings = _extract_findings(result)
        logger.info("[%s] Explore node: topic=%r — %d findings entries",
                     work_id, topic_str, len(findings))

    except Exception as e:
        logger.error("[%s] Explore node failed for topic=%r: %s",
                     work_id, topic_str, e, exc_info=True)
        findings = [{
            "summary": f"Research failed for topic '{topic_str}': {e}",
            "patterns": [],
            "file_map": {},
            "dependencies": [],
        }]

    return {"findings": findings}


def _extract_findings(result: dict) -> list[dict]:
    """Extract ResearchFindings from an agent result.

    If the agent returned structured output (via response_format),
    it'll be in the structured_response key. Otherwise fall back to
    parsing the final message.
    """
    # Try structured output first
    structured = result.get("structured_response")
    if structured:
        if isinstance(structured, dict):
            return [structured]
        if hasattr(structured, "model_dump"):
            return [structured.model_dump()]

    # Fall back to messages
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return [{"summary": content, "patterns": [], "file_map": {}, "dependencies": []}]

    return [{"summary": "(no findings)", "patterns": [], "file_map": {}, "dependencies": []}]
```

**Verification:** `python -c "from spine.agents.exploration_agents import run_research_manager, run_explore_node; print('OK')"`

---

### Task 3: Create the exploration subgraph builder

**Objective:** Build the multi-node StateGraph with Send API fan-out.

**File:** Create `spine/workflow/subgraphs/exploration_subgraph.py`

```python
"""SPECIFY/PLAN exploration subgraph — multi-node research loop.

Nodes:
- research_manager: single LLM call to decide next topics or done
- explore: researcher subagent (runs in parallel via Send API)
- aggregate: deterministic merge of findings into accumulator
- synthesize: Deep Agent that writes the spec/plan artifact
- save_artifacts: scans disk, materializes to state

Edges:
START → research_manager
research_manager → Send("explore", {topic}) × N  OR  → synthesize
explore → aggregate
aggregate → sufficiency check → research_manager (loop) OR synthesize
synthesize → save_artifacts → END
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import ExplorationSubgraphState
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    _artifact_path,
)
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500
_DEFAULT_MAX_ROUNDS = 3


# ── Node: research_manager ───────────────────────────────────────────

async def _research_manager_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Call the research manager LLM to decide next topics or done."""
    from spine.agents.exploration_agents import run_research_manager

    result = await run_research_manager(dict(state), config)
    round_num = state.get("research_round", 0)
    return {
        **result,
        "research_round": round_num + 1,
    }


# ── Router: research_manager → explore (Send) or synthesize ─────────

def _research_router(
    state: ExplorationSubgraphState,
) -> list[Send] | Literal["synthesize"]:
    """Fan-out to explore nodes via Send API, or proceed to synthesis."""
    decision = state.get("manager_decision", "done")
    topics = state.get("topics", [])

    if decision == "done" or not topics:
        logger.info("Research complete — routing to synthesize")
        return "synthesize"

    sends = [Send("explore", {"topic": t}) for t in topics]
    logger.info("Dispatching %d explore nodes: %s", len(sends), topics)
    return sends


# ── Node: explore ───────────────────────────────────────────────────

async def _explore_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
    *,
    topic: str = "",
) -> dict[str, Any]:
    """Run a researcher subagent for one topic."""
    from spine.agents.exploration_agents import run_explore_node

    return await run_explore_node(dict(state), config, topic=topic)


# ── Node: aggregate ─────────────────────────────────────────────────

async def _aggregate_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Deterministic merge — findings are already accumulated via operator.add.

    This node exists as a routing checkpoint so the sufficiency gate
    can inspect the fully accumulated state after all parallel explore
    nodes have completed (fan-in point).
    """
    findings = state.get("findings", [])
    logger.info("Aggregated %d findings across all rounds", len(findings))
    return {}


# ── Router: aggregate → loop (research_manager) or done (synthesize) ─

def _sufficiency_router(
    state: ExplorationSubgraphState,
) -> Literal["loop", "done"]:
    """Check whether research is sufficient to proceed to synthesis."""
    decision = state.get("manager_decision", "done")
    max_rounds = state.get("max_rounds", _DEFAULT_MAX_ROUNDS)
    round_num = state.get("research_round", 0)

    if decision == "done":
        return "done"
    if round_num >= max_rounds:
        logger.info("Max rounds (%d) reached — proceeding to synthesis", max_rounds)
        return "done"
    return "loop"


# ── Node: synthesize ────────────────────────────────────────────────

async def _synthesize_specify(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Synthesize research findings into a specification.

    Uses the existing specify agent infrastructure — builds a Deep Agent
    with the write_specification tool and researcher findings as context.
    """
    from spine.agents.specify_agent import build_specify_agent

    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")
    findings = state.get("findings", [])

    logger.info("[%s] Synthesize (specify): %d findings available", work_id, len(findings))

    try:
        agent = build_specify_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        findings_text = _format_findings(findings)
        prompt = (
            f"Create a detailed specification for the following work, "
            f"incorporating the codebase research findings below.\n\n"
            f"## Work Description\n{description}\n\n"
            f"## Codebase Research Findings\n{findings_text}\n\n"
            f"Write the specification to `.spine/artifacts/{work_id}/specify/specification.md` "
            f"using `write_specification`."
        )

        ctx = build_context(dict(state), PhaseName.SPECIFY)
        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.SPECIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
        }

    except Exception as e:
        logger.error("[%s] Synthesize (specify) failed: %s", work_id, e, exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Synthesis error: {e}",
            "phase_status": "error",
        }


def _format_findings(findings: list[dict]) -> str:
    """Format accumulated findings for the synthesizer prompt."""
    if not findings:
        return "(no codebase research was performed — the work is self-contained)"
    parts = []
    for i, f in enumerate(findings):
        if isinstance(f, dict):
            summary = f.get("summary", "")
            patterns = f.get("patterns", [])
            file_map = f.get("file_map", {})
            deps = f.get("dependencies", [])
            parts.append(f"### Finding {i+1}\n{summary}")
            if patterns:
                parts.append(f"Patterns: {', '.join(patterns)}")
            if file_map:
                parts.append(f"Key files: {json.dumps(file_map)}")
            if deps:
                parts.append(f"Dependencies: {', '.join(deps)}")
    return "\n\n".join(parts)


# ── Note: PLAN synthesizer follows the same pattern ─────────────────
# For now, we focus on SPECIFY. PLAN can be added in Phase 4.


# ── Node: save_artifacts ─────────────────────────────────────────────

async def _save_exploration_artifacts(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the exploration subgraph."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    phase = state.get("phase", PhaseName.SPECIFY.value)
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {"artifacts_output": {}, "phase_status": existing_phase_status}

    disk_artifacts = scan_artifact_dir(
        workspace_root, work_id, phase,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        artifact_name = "specification.md" if phase == PhaseName.SPECIFY.value else "plan.md"
        materialize_phase_artifacts(
            phase,
            {artifact_name: agent_response},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {artifact_name: agent_response[:_MAX_ARTIFACT_STATE_CHARS]}

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }


# ── Builder ──────────────────────────────────────────────────────────

def build_exploration_subgraph(
    phase: str = PhaseName.SPECIFY.value,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
) -> Any:
    """Build the multi-node exploration → synthesis subgraph.

    Args:
        phase: Which phase this subgraph is for ("specify" or "plan").
        max_rounds: Maximum number of research_manager rounds (safety valve).

    Returns:
        Uncompiled StateGraph builder. Call ``.compile()`` to get a
        runnable graph.
    """
    builder = StateGraph(ExplorationSubgraphState)

    # Choose the synthesizer based on phase
    if phase == PhaseName.SPECIFY.value:
        synthesizer = _synthesize_specify
    elif phase == PhaseName.PLAN.value:
        # PLAN synthesizer — to be added in Phase 4
        raise NotImplementedError("PLAN exploration subgraph not yet implemented")
    else:
        raise ValueError(f"Unsupported phase for exploration subgraph: {phase!r}")

    builder.add_node("research_manager", _research_manager_node)
    builder.add_node("explore", _explore_node)
    builder.add_node("aggregate", _aggregate_node)
    builder.add_node("synthesize", synthesizer)
    builder.add_node("save_artifacts", _save_exploration_artifacts)

    builder.add_edge(START, "research_manager")

    # research_manager → Send("explore", ...) or → synthesize
    builder.add_conditional_edges(
        "research_manager",
        _research_router,
        {"explore": "explore", "synthesize": "synthesize"},
    )

    # Explore → aggregate (fan-in — LangGraph waits for ALL Send targets)
    builder.add_edge("explore", "aggregate")

    # Aggregate → loop to research_manager or done → synthesize
    builder.add_conditional_edges(
        "aggregate",
        _sufficiency_router,
        {"loop": "research_manager", "done": "synthesize"},
    )

    builder.add_edge("synthesize", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder
```

**Verification:** `python -c "from spine.workflow.subgraphs.exploration_subgraph import build_exploration_subgraph; g = build_exploration_subgraph(); print('OK')"`

---

### Task 4: Wire exploration subgraph into compose.py

**Objective:** Add the exploration subgraph builder to the registry and wire it as an alternative to the current linear specify subgraph. Feature-flag with `_SUBGRAPH_ENABLED`.

**File:** Modify `spine/workflow/compose.py`

1. Add import:
```python
from spine.workflow.subgraphs.exploration_subgraph import build_exploration_subgraph
```

2. Add a new feature flag dict for exploration mode:
```python
# Feature flags for exploration subgraph rollout
# When True, SPECIFY/PLAN use the multi-node exploration loop
# instead of the linear run_agent → save_artifacts subgraph.
_USE_EXPLORATION_SUBGRAPH: dict[str, bool] = {
    PhaseName.SPECIFY.value: True,   # Enable for SPECIFY
    PhaseName.PLAN.value: False,     # PLAN not yet implemented
}
```

3. In `build_workflow_graph()`, before the phase node loop, add exploration subgraph handling:

```python
# ── Exploration subgraph registration ──
# When exploration mode is enabled for a phase, use the multi-node
# research_manager → explore → synthesize subgraph instead of the
# linear run_agent → save_artifacts subgraph.
# Register the exploration builder so per-phase checkpointer can recompile.
if _USE_EXPLORATION_SUBGRAPH.get(PhaseName.SPECIFY.value, False):
    register_subgraph_builder(
        PhaseName.SPECIFY.value,
        lambda: build_exploration_subgraph(phase=PhaseName.SPECIFY.value),
    )
```

4. The existing `build_specify_subgraph()` builder registration (line 85) will be shadowed by the exploration builder when `_USE_EXPLORATION_SUBGRAPH["specify"] = True`. Actually, looking at the code more carefully, the registration at line 85 happens at import time, before `build_workflow_graph()` is called. We need to handle this differently — the exploration builder should REPLACE the existing registration when the flag is on.

Simpler approach: In `compose.py`, check the flag inside `build_workflow_graph()` and use the exploration builder instead of the standard one for the specify node. The `_SUBGRAPH_BUILDER_REGISTRY` lookup happens at line 582 — we can override it before the lookup.

Actually, the cleanest approach: override the registration in `build_workflow_graph()`:

```python
# Override SPECIFY builder with exploration subgraph when feature-flagged
if _USE_EXPLORATION_SUBGRAPH.get(PhaseName.SPECIFY.value, False):
    register_subgraph_builder(
        PhaseName.SPECIFY.value,
        lambda: build_exploration_subgraph(phase=PhaseName.SPECIFY.value),
    )
```

This replaces the existing registration from the import-time call.

**Verification:** `python -c "from spine.workflow.compose import build_workflow_graph; g = build_workflow_graph('spec'); print('OK')"`

---

### Task 5: Add `from __future__ import annotations` import

**Objective:** `exploration_subgraph.py` uses `list[dict]` syntax — need `from __future__ import annotations`.

**File:** Already included in the template above.

---

### Task 6: Verify with existing tests

**Objective:** Run the existing test suite to make sure nothing is broken.

```bash
cd /home/pat/projects/spine && python -m pytest tests/unit/ -x -q 2>&1 | tail -20
```

Expected: All existing tests should pass (the exploration subgraph only activates for SPECIFY new-style subgraph mode, which is already gated behind `_SUBGRAPH_ENABLED["specify"]` being True).

---

### Task 7: Integration test for exploration subgraph

**Objective:** Write a minimal integration test that exercises the exploration subgraph.

**File:** Create test in `tests/integration/` or add to existing specify tests.

```python
"""Integration test for exploration subgraph."""

import pytest
from spine.workflow.subgraphs.exploration_subgraph import build_exploration_subgraph
from spine.models.enums import PhaseName


@pytest.mark.asyncio
async def test_exploration_subgraph_builds():
    """The exploration subgraph should compile without errors."""
    builder = build_exploration_subgraph(phase=PhaseName.SPECIFY.value)
    graph = builder.compile()
    assert graph is not None


@pytest.mark.asyncio
async def test_exploration_state_schema():
    """The ExplorationSubgraphState should accept valid input."""
    from spine.workflow.subgraph_state import ExplorationSubgraphState
    
    state: ExplorationSubgraphState = {
        "phase": "specify",
        "work_id": "test-1",
        "work_type": "spec",
        "description": "Test work description",
        "workspace_root": "/tmp",
        "retry_count": 0,
        "feedback": [],
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "research_round": 0,
        "max_rounds": 3,
        "manager_decision": "",
        "topics": [],
        "findings": [],
        "agent_response": "",
        "plan_json": "",
        "execution_waves": [],
    }
    assert state["research_round"] == 0
```

**Verification:** `pytest tests/integration/ -k "exploration" -v`

---

## Implementation Order

1. Task 1: State schema (no deps)
2. Task 2: Agent builders (depends on Task 1)
3. Task 3: Subgraph builder (depends on Tasks 1, 2)
4. Task 4: Wire into compose.py (depends on Task 3)
5. Task 5: `from __future__` import — already included
6. Task 6: Run existing tests
7. Task 7: Integration test

Each task is 2-5 min of focused work. Commit after each.
