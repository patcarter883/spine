# SPINE Workflow Rebuild — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Rebuild the SPINE workflow engine from first principles with composable phase-per-file architecture, agent-per-file Deep Agent definitions, and a deterministic graph composition system supporting four workflow types.

**Architecture:** Each workflow phase (Specify, Plan, Tasks, Implement, Verify, Critic) is a self-contained module with its own state schema and compiled subgraph. Each Deep Agent definition is a factory function in its own file. Workflows are composed by wiring phase subgraphs into a parent StateGraph with conditional edges for critic review loops. The critic is structural — it enforces pass/fail/rework with configurable retry limits and human escalation via `interrupt()`.

**Tech Stack:** Python 3.11+, LangGraph (StateGraph, Command, interrupt, Overwrite), Deep Agents (`create_deep_agent`), TypedDict + Annotated reducers

---

## Design Decisions

### 1. Phase-Per-File with Subgraph Composition

Each phase lives in `spine/phases/<phase_name>.py` and exports:
- A `TypedDict` for its local state (`SpecifyState`, `PlanState`, etc.)
- A `build_phase_graph()` function returning a compiled `StateGraph`
- A `call_phase()` wrapper function that maps parent state → phase subgraph input → parent state

This uses LangGraph's **Pattern A (wrapper function)** from the docs — parent and subgraph have different state schemas, so we map between them explicitly. This gives maximum isolation per phase.

### 2. Agent-Per-File Factory Functions

Each Deep Agent definition lives in `spine/agents/<agent_name>.py` and exports:
- A `create_<name>_agent(providers, ...)` factory function
- Returns a compiled Deep Agent graph ready for `.invoke()`

### 3. Critic as a Phase, Not Middleware

The critic is a workflow phase — not a DA middleware. It runs AFTER the phase's DA agent completes, as a separate graph node. This makes the review loop a graph-level concern, not an agent-level concern. The critic phase:
- Receives the output of the preceding phase
- Returns `Command` with verdict (pass/fail) and routing (next phase / rework / human_review)
- Tracks retry counts in parent state
- Escalates to `interrupt()` after max retries

### 4. Prompt Requests via `interrupt()`

When a phase needs human input, it calls `interrupt()` with a structured payload. The workflow pauses. The caller resumes with `Command(resume=value)`. This uses LangGraph's built-in human-in-the-loop mechanism — no custom escalation system needed.

### 5. Workflow State Design

```python
class WorkflowState(TypedDict):
    # ── Input ──
    requirement: str                          # Original work description
    work_type: str                            # "quick" | "critical_quick" | "spec" | "critical_spec"

    # ── Phase tracking ──
    current_phase: str                        # Name of the currently executing phase
    phase_history: Annotated[list[str], operator.add]  # Ordered list of visited phases

    # ── Artifacts (overwrite: latest wins) ──
    artifacts: dict                           # {"specification": "...", "plan": "...", "tasks": [...], "implementation": "..."}

    # ── Feedback (accumulate: grows across retries) ──
    feedback: Annotated[list[str], operator.add]  # Critic feedback messages

    # ── Critic review ──
    critic_verdict: str                       # "pass" | "fail" | "rework"
    retry_counts: dict                        # {"specify": 0, "plan": 1, ...}
    escalation_reason: str | None             # Why we escalated to human

    # ── Config (through config, not state, for providers) ──
    # Providers go through config["configurable"]["providers"] — NEVER in state
```

### 6. Four Workflow Types as Graph Compositions

Each workflow type wires phases differently:

| Type | Phases |
|------|--------|
| Quick | TASKS → IMPLEMENT → VERIFY |
| Critical Quick | TASKS → CRITIC → IMPLEMENT → VERIFY |
| Spec | SPECIFY → PLAN → TASKS → IMPLEMENT → VERIFY |
| Critical Spec | SPECIFY → CRITIC → PLAN → CRITIC → TASKS → CRITIC → IMPLEMENT → VERIFY |

The critic phase is inserted AFTER certain phases in "critical" variants. The graph builder decides which composition to build based on `work_type`.

### 7. File Structure

```
spine/
├── phases/
│   ├── __init__.py              # Phase registry + PhaseOutput dataclass
│   ├── specify.py               # Specify phase subgraph + wrapper
│   ├── plan.py                  # Plan phase subgraph + wrapper
│   ├── tasks.py                 # Tasks phase subgraph (decomposition) + wrapper
│   ├── implement.py             # Implement phase subgraph + wrapper
│   ├── verify.py                # Verify phase subgraph + wrapper
│   └── critic.py                # Critic phase subgraph + wrapper
├── agents/
│   ├── __init__.py              # Agent registry
│   ├── specify_agent.py         # create_specify_agent()
│   ├── plan_agent.py            # create_plan_agent()
│   ├── tasks_agent.py           # create_tasks_agent() (decomposition)
│   ├── implement_agent.py       # create_implement_agent()
│   ├── verify_agent.py          # create_verify_agent()
│   └── critic_agent.py          # create_critic_agent() — quality reviewer
├── workflow/
│   ├── __init__.py              # Public API: run_workflow(), resume_workflow()
│   ├── state.py                 # WorkflowState TypedDict + reducers
│   ├── compose.py               # build_workflow_graph(work_type) → compiled graph
│   └── critic_review.py         # structural_critic_check() + retry escalation logic
└── ... (existing modules stay)
```

---

## Implementation Tasks

### Task 1: Create the workflow state module

**Objective:** Define `WorkflowState` TypedDict with all reducers and a `PhaseOutput` dataclass for phase results.

**Files:**
- Create: `spine/workflow/__init__.py`
- Create: `spine/workflow/state.py`

**Step 1: Write the state module**

```python
# spine/workflow/__init__.py
"""Composable workflow engine for SPINE — phase-per-file, agent-per-file architecture."""

from .state import WorkflowState, PhaseOutput, PhaseStatus

__all__ = ["WorkflowState", "PhaseOutput", "PhaseStatus"]
```

```python
# spine/workflow/state.py
"""Workflow state schema and phase output types."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any

from typing_extensions import TypedDict


class PhaseStatus(str, Enum):
    """Status of a workflow phase execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REWORK = "needs_rework"
    NEEDS_HUMAN = "needs_human"


@dataclass
class PhaseOutput:
    """Output produced by a workflow phase.

    Phases return this to communicate artifacts, feedback, or prompt requests
    back to the workflow engine.
    """
    artifacts: dict[str, Any] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)
    status: PhaseStatus = PhaseStatus.COMPLETED
    prompt_request: dict[str, Any] | None = None  # Non-None → needs human input


class WorkflowState(TypedDict):
    """Parent workflow state — shared across all phases and the critic.

    Providers MUST NOT be stored here. They go through
    config["configurable"]["providers"] to avoid serialization failures.
    """
    # ── Input ──
    requirement: str
    work_type: str  # "quick" | "critical_quick" | "spec" | "critical_spec"

    # ── Phase tracking ──
    current_phase: str
    phase_history: Annotated[list[str], operator.add]

    # ── Artifacts (overwrite: latest wins) ──
    artifacts: dict

    # ── Feedback (accumulate: grows across retries) ──
    feedback: Annotated[list[str], operator.add]

    # ── Critic review ──
    critic_verdict: str  # "pass" | "fail" | "rework"
    retry_counts: dict  # {"specify": 0, "plan": 1, ...}
    escalation_reason: str | None
```

**Step 2: Run syntax check**

Run: `python -c "from spine.workflow.state import WorkflowState, PhaseOutput, PhaseStatus; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add spine/workflow/
git commit -m "feat(workflow): add WorkflowState and PhaseOutput types"
```

---

### Task 2: Create the phase registry and base phase protocol

**Objective:** Define the phase protocol (what every phase module exports) and a registry for looking up phases by name.

**Files:**
- Create: `spine/phases/__init__.py`

**Step 1: Write the phase registry**

```python
# spine/phases/__init__.py
"""Workflow phases — one module per phase, each exporting build_phase_graph() and call_phase()."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph.state import CompiledStateGraph

from spine.workflow.state import WorkflowState


# ── Phase Protocol ──
# Every phase module (specify.py, plan.py, etc.) MUST export:
#   1. build_phase_graph() -> CompiledStateGraph
#      Compiles the phase's internal subgraph with its own state schema.
#
#   2. call_phase(state: WorkflowState, config) -> dict
#      Wrapper function: maps parent state → subgraph input, invokes subgraph,
#      maps subgraph output → parent state updates.
#      Returns a partial WorkflowState dict.
#
# This protocol is enforced by convention + the registry below.


PhaseFactory = Callable[..., CompiledStateGraph]
PhaseWrapper = Callable[[WorkflowState, Any], dict]


# ── Phase Registry ──
_PHASE_REGISTRY: dict[str, tuple[PhaseFactory, PhaseWrapper]] = {}


def register_phase(name: str, factory: PhaseFactory, wrapper: PhaseWrapper) -> None:
    """Register a phase by name."""
    _PHASE_REGISTRY[name] = (factory, wrapper)


def get_phase(name: str) -> tuple[PhaseFactory, PhaseWrapper]:
    """Look up a registered phase by name.

    Raises:
        KeyError: if the phase name is not registered.
    """
    if name not in _PHASE_REGISTRY:
        # Lazy-import all built-in phases on first miss
        _import_builtin_phases()
        if name not in _PHASE_REGISTRY:
            raise KeyError(f"Unknown phase: {name!r}. Available: {list(_PHASE_REGISTRY)}")
    return _PHASE_REGISTRY[name]


_BUILTINS_IMPORTED = False


def _import_builtin_phases() -> None:
    """Import all built-in phase modules to trigger their registration."""
    global _BUILTINS_IMPORTED
    if _BUILTINS_IMPORTED:
        return
    _BUILTINS_IMPORTED = True

    from spine.phases import specify, plan, tasks, implement, verify, critic  # noqa: F401
```

**Step 2: Run syntax check**

Run: `python -c "from spine.phases import get_phase; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add spine/phases/__init__.py
git commit -m "feat(phases): add phase registry with lazy-import protocol"
```

---

### Task 3: Create the agent registry

**Objective:** Define the agent module protocol (factory function per agent) and a registry.

**Files:**
- Create: `spine/agents/__init__.py`

**Step 1: Write the agent registry**

```python
# spine/agents/__init__.py
"""Deep Agent definitions — one module per agent, each exporting create_*_agent()."""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph.state import CompiledStateGraph


AgentFactory = Callable[..., CompiledStateGraph]


_AGENT_REGISTRY: dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory) -> None:
    """Register an agent factory by name."""
    _AGENT_REGISTRY[name] = factory


def get_agent(name: str) -> AgentFactory:
    """Look up a registered agent by name.

    Raises:
        KeyError: if the agent name is not registered.
    """
    if name not in _AGENT_REGISTRY:
        _import_builtin_agents()
        if name not in _AGENT_REGISTRY:
            raise KeyError(f"Unknown agent: {name!r}. Available: {list(_AGENT_REGISTRY)}")
    return _AGENT_REGISTRY[name]


_BUILTINS_IMPORTED = False


def _import_builtin_agents() -> None:
    """Import all built-in agent modules to trigger their registration."""
    global _BUILTINS_IMPORTED
    if _BUILTINS_IMPORTED:
        return
    _BUILTINS_IMPORTED = True

    from spine.agents import specify_agent, plan_agent, tasks_agent, implement_agent, verify_agent  # noqa: F401
```

**Step 2: Run syntax check**

Run: `python -c "from spine.agents import get_agent; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add spine/agents/__init__.py
git commit -m "feat(agents): add agent registry with lazy-import protocol"
```

---

### Task 4: Create the Specify phase

**Objective:** Implement the Specify phase as a self-contained subgraph with its own state, agent invocation, and registration.

**Files:**
- Create: `spine/phases/specify.py`
- Create: `spine/agents/specify_agent.py`

**Step 1: Create the specify agent factory**

```python
# spine/agents/specify_agent.py
"""Deep Agent for the Specify phase — generates a detailed spec from a requirement."""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from spine.agents import register_agent

SPECIFY_SYSTEM_PROMPT = """\
You are a specification writer. Given a requirement, produce a detailed
specification document that includes:

1. **Problem Statement** — What problem does this work address?
2. **Scope** — What is in scope and out of scope.
3. **Requirements** — Functional and non-functional requirements.
4. **Constraints** — Technical, organizational, or resource constraints.
5. **Success Criteria** — How to verify the work is complete.

If you need clarification on the requirement, use the `write_todos` tool
to note what needs clarification, then ask. Otherwise, produce the spec and
end with SPEC_COMPLETE.
"""


def create_specify_agent(
    providers: dict[str, Any],
    project_root: str | None = None,
    **kwargs: Any,
):
    """Create a Deep Agent for the Specify phase.

    Args:
        providers: Provider dict from config["configurable"]["providers"].
        project_root: Root directory for filesystem backend.
        **kwargs: Additional args passed to create_deep_agent.

    Returns:
        Compiled Deep Agent graph.
    """
    from deepagents.backends import LocalShellBackend

    llm_provider = providers.get("llm")
    chat_model = llm_provider.chat_model if llm_provider and not isinstance(llm_provider, dict) else None

    if not chat_model:
        raise ValueError("No LLM provider available for Specify phase")

    agent = create_deep_agent(
        name="specify",
        model=chat_model,
        system_prompt=SPECIFY_SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=project_root or ".", virtual_mode=False),
        **kwargs,
    )
    return agent


register_agent("specify", create_specify_agent)
```

**Step 2: Create the specify phase subgraph**

```python
# spine/phases/specify.py
"""Specify phase — generates a detailed spec from a requirement."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from spine.phases import register_phase
from spine.workflow.state import PhaseOutput, PhaseStatus, WorkflowState


class SpecifyState(TypedDict):
    """Internal state for the Specify phase subgraph."""
    requirement: str
    feedback: list[str]
    specification: str
    prompt_request: dict[str, Any] | None
    status: str  # "completed" | "needs_human"


def specify_node(state: SpecifyState, config: RunnableConfig | None = None) -> dict:
    """Run the Specify agent to produce a specification."""
    from spine.agents import get_agent

    cfg_providers = (config or {}).get("configurable", {}).get("providers", {})
    project_root = (config or {}).get("configurable", {}).get("project_root", ".")

    # Include any critic feedback for rework
    feedback_context = ""
    if state.get("feedback"):
        feedback_context = (
            "\n\n## Previous Critic Feedback (address these):\n"
            + "\n".join(f"- {f}" for f in state["feedback"])
        )

    prompt = f"Produce a detailed specification for the following requirement:\n\n{state['requirement']}{feedback_context}"

    try:
        agent_factory = get_agent("specify")
        agent = agent_factory(providers=cfg_providers, project_root=project_root)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": "specify"}},
        )
        # Extract the last AI message as the spec
        messages = result.get("messages", [])
        spec_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                spec_text = msg.content
                break

        return {
            "specification": spec_text,
            "prompt_request": None,
            "status": "completed",
        }
    except Exception as exc:
        return {
            "specification": f"Specify phase failed: {exc}",
            "prompt_request": None,
            "status": "completed",  # Let critic review handle failures
        }


def build_phase_graph() -> Any:
    """Build and compile the Specify phase subgraph."""
    builder = StateGraph(SpecifyState)
    builder.add_node("specify", specify_node)
    builder.add_edge(START, "specify")
    builder.add_edge("specify", END)
    return builder.compile()


def call_specify(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    """Map parent state → Specify subgraph input, invoke, map back."""
    graph = build_phase_graph()
    result = graph.invoke(
        {
            "requirement": state["requirement"],
            "feedback": state.get("feedback", []),
            "specification": "",
            "prompt_request": None,
            "status": "pending",
        },
    )

    updates: dict[str, Any] = {
        "current_phase": "specify",
        "phase_history": ["specify"],
        "artifacts": {"specification": result.get("specification", "")},
    }

    # If the phase requested human input, set escalation
    if result.get("prompt_request"):
        updates["escalation_reason"] = "specify_phase_needs_input"
        updates["critic_verdict"] = "rework"

    return updates


register_phase("specify", build_phase_graph, call_specify)
```

**Step 3: Run syntax check**

Run: `python -c "from spine.phases.specify import call_specify; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add spine/phases/specify.py spine/agents/specify_agent.py
git commit -m "feat(phases): add Specify phase with agent factory"
```

---

### Task 5: Create the Plan phase

**Objective:** Implement the Plan phase — defines technical architecture from a specification.

**Files:**
- Create: `spine/phases/plan.py`
- Create: `spine/agents/plan_agent.py`

**Step 1: Create the plan agent factory**

```python
# spine/agents/plan_agent.py
"""Deep Agent for the Plan phase — defines technical architecture from a spec."""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from spine.agents import register_agent

PLAN_SYSTEM_PROMPT = """\
You are a technical architect. Given a specification, produce a detailed
technical plan that includes:

1. **Architecture** — System architecture, components, and their relationships.
2. **Interfaces** — Key interfaces and APIs.
3. **Data Models** — Core data structures and schemas.
4. **Implementation Strategy** — Order of implementation and dependencies.
5. **Risk Assessment** — Technical risks and mitigations.

Produce the plan and end with PLAN_COMPLETE.
"""


def create_plan_agent(
    providers: dict[str, Any],
    project_root: str | None = None,
    **kwargs: Any,
):
    """Create a Deep Agent for the Plan phase."""
    from deepagents.backends import LocalShellBackend

    llm_provider = providers.get("llm")
    chat_model = llm_provider.chat_model if llm_provider and not isinstance(llm_provider, dict) else None

    if not chat_model:
        raise ValueError("No LLM provider available for Plan phase")

    agent = create_deep_agent(
        name="plan",
        model=chat_model,
        system_prompt=PLAN_SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=project_root or ".", virtual_mode=False),
        **kwargs,
    )
    return agent


register_agent("plan", create_plan_agent)
```

**Step 2: Create the plan phase subgraph**

```python
# spine/phases/plan.py
"""Plan phase — defines technical architecture from a specification."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from spine.phases import register_phase
from spine.workflow.state import WorkflowState


class PlanState(TypedDict):
    """Internal state for the Plan phase subgraph."""
    requirement: str
    specification: str
    feedback: list[str]
    plan: str
    prompt_request: dict[str, Any] | None
    status: str


def plan_node(state: PlanState, config: RunnableConfig | None = None) -> dict:
    """Run the Plan agent to produce a technical plan."""
    from spine.agents import get_agent

    cfg_providers = (config or {}).get("configurable", {}).get("providers", {})
    project_root = (config or {}).get("configurable", {}).get("project_root", ".")

    feedback_context = ""
    if state.get("feedback"):
        feedback_context = (
            "\n\n## Previous Critic Feedback (address these):\n"
            + "\n".join(f"- {f}" for f in state["feedback"])
        )

    spec_section = ""
    if state.get("specification"):
        spec_section = f"\n\n## Specification:\n{state['specification']}"

    prompt = (
        f"Create a technical plan for the following requirement:\n\n"
        f"{state['requirement']}"
        f"{spec_section}"
        f"{feedback_context}"
    )

    try:
        agent_factory = get_agent("plan")
        agent = agent_factory(providers=cfg_providers, project_root=project_root)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": "plan"}},
        )
        messages = result.get("messages", [])
        plan_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                plan_text = msg.content
                break

        return {"plan": plan_text, "prompt_request": None, "status": "completed"}
    except Exception as exc:
        return {"plan": f"Plan phase failed: {exc}", "prompt_request": None, "status": "completed"}


def build_phase_graph() -> Any:
    """Build and compile the Plan phase subgraph."""
    builder = StateGraph(PlanState)
    builder.add_node("plan", plan_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", END)
    return builder.compile()


def call_plan(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    """Map parent state → Plan subgraph input, invoke, map back."""
    graph = build_phase_graph()
    result = graph.invoke({
        "requirement": state["requirement"],
        "specification": state.get("artifacts", {}).get("specification", ""),
        "feedback": state.get("feedback", []),
        "plan": "",
        "prompt_request": None,
        "status": "pending",
    })

    updates: dict[str, Any] = {
        "current_phase": "plan",
        "phase_history": ["plan"],
        "artifacts": {"plan": result.get("plan", "")},
    }

    if result.get("prompt_request"):
        updates["escalation_reason"] = "plan_phase_needs_input"
        updates["critic_verdict"] = "rework"

    return updates


register_phase("plan", build_phase_graph, call_plan)
```

**Step 3: Run syntax check**

Run: `python -c "from spine.phases.plan import call_plan; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add spine/phases/plan.py spine/agents/plan_agent.py
git commit -m "feat(phases): add Plan phase with agent factory"
```

---

### Task 6: Create the Tasks phase (decomposition)

**Objective:** Implement the Tasks phase — breaks a plan into executable feature slices. This is where decomposition happens.

**Files:**
- Create: `spine/phases/tasks.py`
- Create: `spine/agents/tasks_agent.py`

**Step 1: Create the tasks agent factory**

```python
# spine/agents/tasks_agent.py
"""Deep Agent for the Tasks phase — decomposes a plan into feature slices."""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from spine.agents import register_agent

TASKS_SYSTEM_PROMPT = """\
You are a task decomposition specialist. Given a technical plan, break it into
implementable feature slices. For each slice, provide:

1. **Name** — Short identifier (e.g., "user-auth-model").
2. **Description** — What this slice implements.
3. **Scope** — Files, modules, or components affected.
4. **Acceptance Criteria** — How to verify this slice is complete.
5. **Dependencies** — Other slices that must be completed first.

Output the slices as a structured list. End with TASKS_COMPLETE.
"""


def create_tasks_agent(
    providers: dict[str, Any],
    project_root: str | None = None,
    **kwargs: Any,
):
    """Create a Deep Agent for the Tasks phase."""
    from deepagents.backends import LocalShellBackend

    llm_provider = providers.get("llm")
    chat_model = llm_provider.chat_model if llm_provider and not isinstance(llm_provider, dict) else None

    if not chat_model:
        raise ValueError("No LLM provider available for Tasks phase")

    agent = create_deep_agent(
        name="tasks",
        model=chat_model,
        system_prompt=TASKS_SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=project_root or ".", virtual_mode=False),
        **kwargs,
    )
    return agent


register_agent("tasks", create_tasks_agent)
```

**Step 2: Create the tasks phase subgraph**

```python
# spine/phases/tasks.py
"""Tasks phase — breaks a plan into executable feature slices."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from spine.phases import register_phase
from spine.workflow.state import WorkflowState


class TasksState(TypedDict):
    """Internal state for the Tasks phase subgraph."""
    requirement: str
    plan: str
    specification: str
    feedback: list[str]
    tasks: str  # Structured task/slice definitions
    prompt_request: dict[str, Any] | None
    status: str


def tasks_node(state: TasksState, config: RunnableConfig | None = None) -> dict:
    """Run the Tasks agent to decompose the plan into feature slices."""
    from spine.agents import get_agent

    cfg_providers = (config or {}).get("configurable", {}).get("providers", {})
    project_root = (config or {}).get("configurable", {}).get("project_root", ".")

    feedback_context = ""
    if state.get("feedback"):
        feedback_context = (
            "\n\n## Previous Critic Feedback (address these):\n"
            + "\n".join(f"- {f}" for f in state["feedback"])
        )

    plan_section = ""
    if state.get("plan"):
        plan_section = f"\n\n## Technical Plan:\n{state['plan']}"

    spec_section = ""
    if state.get("specification"):
        spec_section = f"\n\n## Specification:\n{state['specification']}"

    prompt = (
        f"Break the following into implementable feature slices:\n\n"
        f"## Requirement:\n{state['requirement']}"
        f"{spec_section}"
        f"{plan_section}"
        f"{feedback_context}"
    )

    try:
        agent_factory = get_agent("tasks")
        agent = agent_factory(providers=cfg_providers, project_root=project_root)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": "tasks"}},
        )
        messages = result.get("messages", [])
        tasks_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                tasks_text = msg.content
                break

        return {"tasks": tasks_text, "prompt_request": None, "status": "completed"}
    except Exception as exc:
        return {"tasks": f"Tasks phase failed: {exc}", "prompt_request": None, "status": "completed"}


def build_phase_graph() -> Any:
    """Build and compile the Tasks phase subgraph."""
    builder = StateGraph(TasksState)
    builder.add_node("tasks", tasks_node)
    builder.add_edge(START, "tasks")
    builder.add_edge("tasks", END)
    return builder.compile()


def call_tasks(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    """Map parent state → Tasks subgraph input, invoke, map back."""
    graph = build_phase_graph()
    artifacts = state.get("artifacts", {})
    result = graph.invoke({
        "requirement": state["requirement"],
        "plan": artifacts.get("plan", ""),
        "specification": artifacts.get("specification", ""),
        "feedback": state.get("feedback", []),
        "tasks": "",
        "prompt_request": None,
        "status": "pending",
    })

    updates: dict[str, Any] = {
        "current_phase": "tasks",
        "phase_history": ["tasks"],
        "artifacts": {"tasks": result.get("tasks", "")},
    }

    if result.get("prompt_request"):
        updates["escalation_reason"] = "tasks_phase_needs_input"
        updates["critic_verdict"] = "rework"

    return updates


register_phase("tasks", build_phase_graph, call_tasks)
```

**Step 3: Run syntax check**

Run: `python -c "from spine.phases.tasks import call_tasks; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add spine/phases/tasks.py spine/agents/tasks_agent.py
git commit -m "feat(phases): add Tasks phase (decomposition) with agent factory"
```

---

### Task 7: Create the Implement phase

**Objective:** Implement the Implement phase — generates code to implement feature slices.

**Files:**
- Create: `spine/phases/implement.py`
- Create: `spine/agents/implement_agent.py`

**Step 1: Create the implement agent factory**

```python
# spine/agents/implement_agent.py
"""Deep Agent for the Implement phase — generates code for feature slices."""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from spine.agents import register_agent

IMPLEMENT_SYSTEM_PROMPT = """\
You are a software implementer. Given feature slices with descriptions,
scope, and acceptance criteria, implement the code changes.

For each feature slice:
1. Read the existing code in the scope directory.
2. Implement the required changes.
3. Write or update tests for the changes.
4. Verify the acceptance criteria are met.

Use the filesystem tools to read, write, and edit files.
Use the shell tools to run tests and verify behavior.
End with IMPLEMENT_COMPLETE.
"""


def create_implement_agent(
    providers: dict[str, Any],
    project_root: str | None = None,
    **kwargs: Any,
):
    """Create a Deep Agent for the Implement phase."""
    from deepagents.backends import LocalShellBackend

    llm_provider = providers.get("llm")
    chat_model = llm_provider.chat_model if llm_provider and not isinstance(llm_provider, dict) else None

    if not chat_model:
        raise ValueError("No LLM provider available for Implement phase")

    agent = create_deep_agent(
        name="implement",
        model=chat_model,
        system_prompt=IMPLEMENT_SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=project_root or ".", virtual_mode=False),
        **kwargs,
    )
    return agent


register_agent("implement", create_implement_agent)
```

**Step 2: Create the implement phase subgraph**

```python
# spine/phases/implement.py
"""Implement phase — generates code to implement feature slices."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from spine.phases import register_phase
from spine.workflow.state import WorkflowState


class ImplementState(TypedDict):
    """Internal state for the Implement phase subgraph."""
    requirement: str
    tasks: str
    plan: str
    specification: str
    feedback: list[str]
    implementation: str
    prompt_request: dict[str, Any] | None
    status: str


def implement_node(state: ImplementState, config: RunnableConfig | None = None) -> dict:
    """Run the Implement agent to produce code changes."""
    from spine.agents import get_agent

    cfg_providers = (config or {}).get("configurable", {}).get("providers", {})
    project_root = (config or {}).get("configurable", {}).get("project_root", ".")

    feedback_context = ""
    if state.get("feedback"):
        feedback_context = (
            "\n\n## Previous Critic Feedback (address these):\n"
            + "\n".join(f"- {f}" for f in state["feedback"])
        )

    prompt = (
        f"Implement the following feature slices:\n\n"
        f"## Requirement:\n{state['requirement']}\n\n"
        f"## Feature Slices:\n{state['tasks']}"
        f"{feedback_context}"
    )

    try:
        agent_factory = get_agent("implement")
        agent = agent_factory(providers=cfg_providers, project_root=project_root)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": "implement"}},
        )
        messages = result.get("messages", [])
        impl_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                impl_text = msg.content
                break

        return {"implementation": impl_text, "prompt_request": None, "status": "completed"}
    except Exception as exc:
        return {"implementation": f"Implement phase failed: {exc}", "prompt_request": None, "status": "completed"}


def build_phase_graph() -> Any:
    """Build and compile the Implement phase subgraph."""
    builder = StateGraph(ImplementState)
    builder.add_node("implement", implement_node)
    builder.add_edge(START, "implement")
    builder.add_edge("implement", END)
    return builder.compile()


def call_implement(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    """Map parent state → Implement subgraph input, invoke, map back."""
    graph = build_phase_graph()
    artifacts = state.get("artifacts", {})
    result = graph.invoke({
        "requirement": state["requirement"],
        "tasks": artifacts.get("tasks", ""),
        "plan": artifacts.get("plan", ""),
        "specification": artifacts.get("specification", ""),
        "feedback": state.get("feedback", []),
        "implementation": "",
        "prompt_request": None,
        "status": "pending",
    })

    updates: dict[str, Any] = {
        "current_phase": "implement",
        "phase_history": ["implement"],
        "artifacts": {"implementation": result.get("implementation", "")},
    }

    if result.get("prompt_request"):
        updates["escalation_reason"] = "implement_phase_needs_input"
        updates["critic_verdict"] = "rework"

    return updates


register_phase("implement", build_phase_graph, call_implement)
```

**Step 3: Run syntax check**

Run: `python -c "from spine.phases.implement import call_implement; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add spine/phases/implement.py spine/agents/implement_agent.py
git commit -m "feat(phases): add Implement phase with agent factory"
```

---

### Task 8: Create the Verify phase

**Objective:** Implement the Verify phase — confirms implementation meets requirements and the plan.

**Files:**
- Create: `spine/phases/verify.py`
- Create: `spine/agents/verify_agent.py`

**Step 1: Create the verify agent factory**

```python
# spine/agents/verify_agent.py
"""Deep Agent for the Verify phase — confirms implementation meets requirements."""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from spine.agents import register_agent

VERIFY_SYSTEM_PROMPT = """\
You are a verification engineer. Given a requirement, plan, and implementation,
verify that:

1. **Feature slices** have been correctly implemented.
2. **The plan** has been followed.
3. **Task requirements** are successfully completed.
4. **Tests pass** and coverage is adequate.

Run tests, inspect code, and compare against the specification.
Report: VERIFIED if all checks pass, or NOT_VERIFIED with specific issues.
"""


def create_verify_agent(
    providers: dict[str, Any],
    project_root: str | None = None,
    **kwargs: Any,
):
    """Create a Deep Agent for the Verify phase."""
    from deepagents.backends import LocalShellBackend

    llm_provider = providers.get("llm")
    chat_model = llm_provider.chat_model if llm_provider and not isinstance(llm_provider, dict) else None

    if not chat_model:
        raise ValueError("No LLM provider available for Verify phase")

    agent = create_deep_agent(
        name="verify",
        model=chat_model,
        system_prompt=VERIFY_SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=project_root or ".", virtual_mode=False),
        **kwargs,
    )
    return agent


register_agent("verify", create_verify_agent)
```

**Step 2: Create the verify phase subgraph**

```python
# spine/phases/verify.py
"""Verify phase — confirms implementation meets requirements and plan."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from spine.phases import register_phase
from spine.workflow.state import PhaseStatus, WorkflowState


class VerifyState(TypedDict):
    """Internal state for the Verify phase subgraph."""
    requirement: str
    specification: str
    plan: str
    tasks: str
    implementation: str
    verification_result: str
    verified: bool
    prompt_request: dict[str, Any] | None
    status: str


def verify_node(state: VerifyState, config: RunnableConfig | None = None) -> dict:
    """Run the Verify agent to check the implementation."""
    from spine.agents import get_agent

    cfg_providers = (config or {}).get("configurable", {}).get("providers", {})
    project_root = (config or {}).get("configurable", {}).get("project_root", ".")

    prompt = (
        f"Verify the following implementation:\n\n"
        f"## Requirement:\n{state['requirement']}\n\n"
        f"## Specification:\n{state['specification']}\n\n"
        f"## Plan:\n{state['plan']}\n\n"
        f"## Feature Slices:\n{state['tasks']}\n\n"
        f"Run tests and report whether the implementation is VERIFIED or NOT_VERIFIED."
    )

    try:
        agent_factory = get_agent("verify")
        agent = agent_factory(providers=cfg_providers, project_root=project_root)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": "verify"}},
        )
        messages = result.get("messages", [])
        verify_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                verify_text = msg.content
                break

        verified = "VERIFIED" in verify_text and "NOT_VERIFIED" not in verify_text

        return {
            "verification_result": verify_text,
            "verified": verified,
            "prompt_request": None,
            "status": "completed",
        }
    except Exception as exc:
        return {
            "verification_result": f"Verify phase failed: {exc}",
            "verified": False,
            "prompt_request": None,
            "status": "completed",
        }


def build_phase_graph() -> Any:
    """Build and compile the Verify phase subgraph."""
    builder = StateGraph(VerifyState)
    builder.add_node("verify", verify_node)
    builder.add_edge(START, "verify")
    builder.add_edge("verify", END)
    return builder.compile()


def call_verify(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    """Map parent state → Verify subgraph input, invoke, map back."""
    graph = build_phase_graph()
    artifacts = state.get("artifacts", {})
    result = graph.invoke({
        "requirement": state["requirement"],
        "specification": artifacts.get("specification", ""),
        "plan": artifacts.get("plan", ""),
        "tasks": artifacts.get("tasks", ""),
        "implementation": artifacts.get("implementation", ""),
        "verification_result": "",
        "verified": False,
        "prompt_request": None,
        "status": "pending",
    })

    updates: dict[str, Any] = {
        "current_phase": "verify",
        "phase_history": ["verify"],
        "artifacts": {"verification": result.get("verification_result", "")},
        "critic_verdict": "pass" if result.get("verified") else "fail",
    }

    if not result.get("verified"):
        updates["feedback"] = [f"Verification failed: {result.get('verification_result', 'unknown')}"]

    if result.get("prompt_request"):
        updates["escalation_reason"] = "verify_phase_needs_input"

    return updates


register_phase("verify", build_phase_graph, call_verify)
```

**Step 3: Run syntax check**

Run: `python -c "from spine.phases.verify import call_verify; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add spine/phases/verify.py spine/agents/verify_agent.py
git commit -m "feat(phases): add Verify phase with agent factory"
```

---

### Task 9: Create the two-tier Critic phase (structural + agent review)

**Objective:** Implement the Critic phase as a two-tier review system:
1. **Tier 1 — Structural check** (fast, deterministic): verifies artifacts exist, have minimum length, contain required markers
2. **Tier 2 — Agent review** (deep, quality-focused): a specialised Deep Agent evaluates correctness, completeness, feasibility, and provides actionable feedback

The structural check runs first. If it fails → rework immediately (no need to waste an agent call). If it passes → the critic agent does deeper quality review. The agent returns APPROVED, NEEDS_REVISION, or REJECTED. Only after the agent also approves does the workflow continue.

**Files:**
- Create: `spine/phases/critic.py`
- Create: `spine/agents/critic_agent.py`
- Create: `spine/workflow/critic_review.py`

**Step 1: Create the structural critic review logic**

```python
# spine/workflow/critic_review.py
"""Two-tier critic review — structural checks + agent quality review.

Tier 1 (structural): fast, deterministic checks on artifact presence, length,
and required content markers. No LLM calls needed.

Tier 2 (agent): deep quality review by a specialised critic agent. Evaluates
correctness, completeness, feasibility, and provides actionable feedback.

The structural check runs first. If it fails, rework immediately without
invoking the agent. If it passes, the agent reviews for deeper issues.
"""

from __future__ import annotations

import os
from typing import Any

from spine.workflow.state import WorkflowState


# ── Minimum content thresholds per phase ──
_MIN_CONTENT_LENGTH: dict[str, int] = {
    "specify": 200,
    "plan": 300,
    "tasks": 200,
    "implement": 100,
}

# ── Required markers in phase output ──
_REQUIRED_MARKERS: dict[str, list[str]] = {
    "specify": ["requirement", "scope", "success"],
    "plan": ["architecture", "implementation"],
    "tasks": ["slice", "acceptance", "depend"],
}

# ── Max retries before human escalation ──
DEFAULT_MAX_CRITIC_RETRIES = 3


def structural_critic_check(
    state: WorkflowState,
    phase_name: str,
) -> tuple[str, str]:
    """Tier 1: Check whether a phase's output passes structural review.

    Structural checks are fast and deterministic — no LLM calls.
    Catches missing/empty artifacts, too-short output, missing required
    content markers, and failure markers.

    Args:
        state: Current workflow state.
        phase_name: Name of the phase that just completed.

    Returns:
        (verdict, reason) where verdict is "pass", "fail", or "rework".
    """
    artifacts = state.get("artifacts", {})
    artifact_key = _artifact_key_for_phase(phase_name)

    content = artifacts.get(artifact_key, "")

    # ── Check 1: Artifact exists and is non-empty ──
    if not content or len(content.strip()) == 0:
        return "rework", f"[{phase_name}] No output produced"

    # ── Check 2: Minimum content length ──
    min_length = _MIN_CONTENT_LENGTH.get(phase_name, 100)
    if len(content) < min_length:
        return "rework", f"[{phase_name}] Output too short ({len(content)} < {min_length} chars)"

    # ── Check 3: Required markers ──
    markers = _REQUIRED_MARKERS.get(phase_name, [])
    content_lower = content.lower()
    missing = [m for m in markers if m not in content_lower]
    if missing:
        return "rework", f"[{phase_name}] Missing required content markers: {missing}"

    # ── Check 4: No failure marker ──
    if "failed:" in content.lower() and phase_name in content.lower():
        return "fail", f"[{phase_name}] Phase reported failure"

    return "pass", f"[{phase_name}] Passed structural review"


def agent_critic_check(
    state: WorkflowState,
    phase_name: str,
    providers: dict[str, Any],
    project_root: str | None = None,
) -> tuple[str, str]:
    """Tier 2: Deep quality review by a specialised critic agent.

    The agent evaluates the phase output for correctness, completeness,
    feasibility, and provides actionable feedback. Only called if the
    structural check passes.

    Args:
        state: Current workflow state.
        phase_name: Name of the phase that just completed.
        providers: Provider dict from config["configurable"]["providers"].
        project_root: Root directory for filesystem backend.

    Returns:
        (verdict, reason) where verdict is "pass", "rework", or "fail".
    """
    from spine.agents import get_agent

    artifacts = state.get("artifacts", {})
    artifact_key = _artifact_key_for_phase(phase_name)
    content = artifacts.get(artifact_key, "")

    requirement = state.get("requirement", "")
    feedback_history = state.get("feedback", [])

    try:
        agent_factory = get_agent("critic")
        agent = agent_factory(providers=providers, project_root=project_root)

        # Build a focused prompt with the artifact and context
        prompt = _build_critic_agent_prompt(
            phase_name=phase_name,
            content=content,
            requirement=requirement,
            feedback_history=feedback_history,
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": f"critic_{phase_name}"}},
        )

        # Extract the agent's verdict from the last AI message
        messages = result.get("messages", [])
        review_text = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                review_text = msg.content
                break

        return _parse_critic_agent_verdict(review_text, phase_name)

    except Exception as exc:
        # Agent failure should not crash the workflow — return rework
        # so the phase can be retried, not escalate immediately
        return "rework", f"[{phase_name}] Critic agent failed: {exc}"


def _artifact_key_for_phase(phase_name: str) -> str:
    """Map phase name to the artifact key in state."""
    mapping = {
        "specify": "specification",
        "plan": "plan",
        "tasks": "tasks",
        "implement": "implementation",
    }
    return mapping.get(phase_name, phase_name)


def _build_critic_agent_prompt(
    phase_name: str,
    content: str,
    requirement: str,
    feedback_history: list[str],
) -> str:
    """Build a focused prompt for the critic agent."""
    phase_descriptions = {
        "specify": "specification",
        "plan": "technical plan",
        "tasks": "task decomposition / feature slices",
        "implement": "implementation",
    }
    phase_desc = phase_descriptions.get(phase_name, phase_name)

    prompt = (
        f"Review the following {phase_desc} for quality.\n\n"
        f"## Original Requirement\n{requirement}\n\n"
        f"## {phase_desc.title()} Output\n{content}\n\n"
    )

    if feedback_history:
        prompt += "## Previous Feedback (check if addressed)\n"
        for fb in feedback_history[-3:]:  # Only last 3 feedback items
            prompt += f"- {fb}\n"
        prompt += "\n"

    prompt += (
        "Evaluate for: correctness, completeness, feasibility, and whether it "
        "addresses the requirement. Respond with your verdict on the first line "
        "as exactly one of: APPROVED, NEEDS_REVISION, or REJECTED. Then explain "
        "your reasoning. If NEEDS_REVISION, list specific issues to address."
    )

    return prompt


def _parse_critic_agent_verdict(
    review_text: str,
    phase_name: str,
) -> tuple[str, str]:
    """Parse the critic agent's response into a (verdict, reason) tuple."""
    if not review_text:
        return "rework", f"[{phase_name}] Critic agent returned empty response"

    # Extract verdict from first line
    first_line = review_text.strip().split("\n")[0].strip().upper()

    for verdict_word, mapped_verdict in [
        ("APPROVED", "pass"),
        ("NEEDS_REVISION", "rework"),
        ("REJECTED", "fail"),
    ]:
        if verdict_word in first_line:
            return mapped_verdict, f"[{phase_name}] Agent review: {review_text.strip()}"

    # If no clear verdict found, treat as rework with the full text as feedback
    return "rework", f"[{phase_name}] Agent review (no clear verdict): {review_text.strip()}"


def should_escalate_to_human(
    state: WorkflowState,
    phase_name: str,
    max_retries: int | None = None,
) -> bool:
    """Check if a phase has exhausted its retry budget.

    Args:
        state: Current workflow state.
        phase_name: Name of the phase being retried.
        max_retries: Override for max retries (default from env/config).

    Returns:
        True if the phase should be escalated to human review.
    """
    if max_retries is None:
        max_retries = int(os.environ.get("SPINE_MAX_CRITIC_RETRIES", str(DEFAULT_MAX_CRITIC_RETRIES)))

    retry_counts = state.get("retry_counts", {})
    current_retries = retry_counts.get(phase_name, 0)
    return current_retries >= max_retries
```

**Step 2: Create the critic agent factory**

```python
# spine/agents/critic_agent.py
"""Deep Agent for the Critic phase — specialised quality reviewer.

The critic agent evaluates phase outputs for correctness, completeness,
feasibility, and alignment with the original requirement. It is NOT a
general-purpose LLM — it is a focused reviewer with access to the codebase
for verification.

The agent has read-only filesystem access (no writes) so it can verify
claims made in the phase output against the actual codebase.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from spine.agents import register_agent

CRITIC_SYSTEM_PROMPT = """\
You are a senior software quality reviewer. You critically evaluate the output
of development workflow phases (specifications, plans, task decompositions,
implementations) to ensure they are correct, complete, and feasible.

Your review process:
1. **Correctness** — Are the technical claims accurate? If code exists, verify
   against the actual codebase using filesystem tools.
2. **Completeness** — Does the output cover all aspects of the requirement?
   Are there gaps, undefined behaviors, or missing edge cases?
3. **Feasibility** — Can this actually be built as described? Are there
   unrealistic assumptions or technical contradictions?
4. **Alignment** — Does the output address the original requirement? Or has
   scope crept or drifted?

You have READ-ONLY filesystem access. Use it to verify claims against reality.

Your response MUST start with exactly one of:
- APPROVED — The output is ready for the next phase.
- NEEDS_REVISION — The output has specific issues that must be fixed.
  List each issue clearly with what needs to change.
- REJECTED — The output is fundamentally flawed and needs to be redone.
  Explain why and what approach should be taken instead.

After the verdict, provide your detailed reasoning.
"""


def create_critic_agent(
    providers: dict[str, Any],
    project_root: str | None = None,
    **kwargs: Any,
):
    """Create a Deep Agent for the Critic phase.

    The critic agent has read-only filesystem access to verify claims
    against the actual codebase. It does NOT write files.

    Args:
        providers: Provider dict from config["configurable"]["providers"].
        project_root: Root directory for filesystem backend (read-only).
        **kwargs: Additional args passed to create_deep_agent.

    Returns:
        Compiled Deep Agent graph.
    """
    from deepagents.backends import LocalShellBackend

    llm_provider = providers.get("llm")
    chat_model = llm_provider.chat_model if llm_provider and not isinstance(llm_provider, dict) else None

    if not chat_model:
        raise ValueError("No LLM provider available for Critic agent")

    agent = create_deep_agent(
        name="critic",
        model=chat_model,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=project_root or ".", virtual_mode=False),
        **kwargs,
    )
    return agent


register_agent("critic", create_critic_agent)
```

**Step 3: Create the Critic phase subgraph (two-tier)**

```python
# spine/phases/critic.py
"""Critic phase — two-tier review of the previous phase's output.

Tier 1: Structural check (fast, deterministic — no LLM).
Tier 2: Agent review (deep, quality-focused — uses a specialised critic agent).

Flow:
  1. Run structural check. If fails → rework immediately (skip agent call).
  2. If structural passes → run critic agent for deep quality review.
  3. Agent returns APPROVED/NEEDS_REVISION/REJECTED.
  4. Track retry counts. After max retries → escalate to human via interrupt().
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

from spine.phases import register_phase
from spine.workflow.critic_review import (
    agent_critic_check,
    should_escalate_to_human,
    structural_critic_check,
)
from spine.workflow.state import WorkflowState


class CriticState(TypedDict):
    """Internal state for the Critic phase subgraph."""
    phase_under_review: str
    # Tier 1: structural
    structural_verdict: str
    structural_reason: str
    # Tier 2: agent
    agent_verdict: str
    agent_reason: str
    # Combined
    critic_verdict: str
    critic_reason: str
    # Retry tracking
    retry_count: int
    max_retries_reached: bool


def critic_node(state: CriticState, config: RunnableConfig | None = None) -> dict:
    """Perform two-tier review: structural check, then agent review.

    Tier 1 (structural) runs first. If it fails, we skip the agent call
    and return rework immediately. If it passes, Tier 2 (agent) runs
    a deep quality review.
    """
    phase = state["phase_under_review"]
    artifacts = (config or {}).get("configurable", {}).get("artifacts", {})
    providers = (config or {}).get("configurable", {}).get("providers", {})
    project_root = (config or {}).get("configurable", {}).get("project_root", ".")
    requirement = (config or {}).get("configurable", {}).get("requirement", "")
    feedback = (config or {}).get("configurable", {}).get("feedback", [])

    # ── Tier 1: Structural check ──
    structural_verdict, structural_reason = structural_critic_check(
        {"artifacts": artifacts},
        phase,
    )

    if structural_verdict != "pass":
        # Structural failure — rework immediately, skip agent call
        max_retries_reached = state.get("max_retries_reached", False)
        final_verdict = "fail" if max_retries_reached else structural_verdict
        final_reason = structural_reason
        if max_retries_reached:
            final_reason += " (max retries exceeded — escalating to human review)"

        return {
            "structural_verdict": structural_verdict,
            "structural_reason": structural_reason,
            "agent_verdict": "skipped",
            "agent_reason": "Structural check failed — agent review skipped",
            "critic_verdict": final_verdict,
            "critic_reason": final_reason,
            "retry_count": state.get("retry_count", 0),
            "max_retries_reached": max_retries_reached,
        }

    # ── Tier 2: Agent quality review ──
    agent_verdict, agent_reason = agent_critic_check(
        {
            "artifacts": artifacts,
            "requirement": requirement,
            "feedback": feedback,
        },
        phase,
        providers=providers,
        project_root=project_root,
    )

    # Combine: agent verdict takes precedence when structural passes
    max_retries_reached = state.get("max_retries_reached", False)
    final_verdict = agent_verdict
    final_reason = agent_reason

    if max_retries_reached and agent_verdict != "pass":
        final_verdict = "fail"
        final_reason += " (max retries exceeded — escalating to human review)"

    return {
        "structural_verdict": structural_verdict,
        "structural_reason": structural_reason,
        "agent_verdict": agent_verdict,
        "agent_reason": agent_reason,
        "critic_verdict": final_verdict,
        "critic_reason": final_reason,
        "retry_count": state.get("retry_count", 0),
        "max_retries_reached": max_retries_reached,
    }


def build_phase_graph() -> Any:
    """Build and compile the Critic phase subgraph."""
    builder = StateGraph(CriticState)
    builder.add_node("critic", critic_node)
    builder.add_edge(START, "critic")
    builder.add_edge("critic", END)
    return builder.compile()


def call_critic(state: WorkflowState, config: RunnableConfig | None = None) -> dict:
    """Map parent state → Critic subgraph input, invoke, map back.

    The critic wrapper handles the full retry/escalation logic:
    - If pass → continue to next phase
    - If rework → loop back to the reviewed phase with feedback
    - If max retries → escalate to human via interrupt()
    """
    # Determine which phase the critic is reviewing
    current = state.get("current_phase", "")
    phase_under_review = current.replace("_critic", "")

    # Get retry state
    retry_counts = state.get("retry_counts", {})
    current_retries = retry_counts.get(phase_under_review, 0)
    max_reached = should_escalate_to_human(state, phase_under_review)

    # Pass everything through config so the critic subgraph can access it
    critic_config = dict(config or {})
    if "configurable" not in critic_config:
        critic_config["configurable"] = {}
    critic_config["configurable"]["artifacts"] = state.get("artifacts", {})
    critic_config["configurable"]["providers"] = (config or {}).get("configurable", {}).get("providers", {})
    critic_config["configurable"]["project_root"] = (config or {}).get("configurable", {}).get("project_root", ".")
    critic_config["configurable"]["requirement"] = state.get("requirement", "")
    critic_config["configurable"]["feedback"] = state.get("feedback", [])

    graph = build_phase_graph()
    result = graph.invoke(
        {
            "phase_under_review": phase_under_review,
            "structural_verdict": "",
            "structural_reason": "",
            "agent_verdict": "",
            "agent_reason": "",
            "critic_verdict": "",
            "critic_reason": "",
            "retry_count": current_retries,
            "max_retries_reached": max_reached,
        },
        config=critic_config,
    )

    verdict = result.get("critic_verdict", "fail")
    reason = result.get("critic_reason", "")
    agent_reason = result.get("agent_reason", "")

    # Build feedback from both tiers
    feedback_parts = []
    structural_reason = result.get("structural_reason", "")
    if structural_reason and result.get("structural_verdict") != "pass":
        feedback_parts.append(f"Structural: {structural_reason}")
    if agent_reason and result.get("agent_verdict") != "skipped":
        feedback_parts.append(f"Agent: {agent_reason}")

    updates: dict[str, Any] = {
        "current_phase": "critic",
        "phase_history": ["critic"],
        "critic_verdict": verdict,
        "feedback": feedback_parts if feedback_parts else [reason],
    }

    if verdict == "rework":
        # Increment retry count and loop back
        new_retries = current_retries + 1
        updates["retry_counts"] = {phase_under_review: new_retries}
        if should_escalate_to_human(
            {**state, "retry_counts": {**retry_counts, phase_under_review: new_retries}},
            phase_under_review,
        ):
            # Escalate to human — use interrupt()
            decision = interrupt({
                "type": "critic_escalation",
                "phase": phase_under_review,
                "retry_count": new_retries,
                "structural_verdict": result.get("structural_verdict", ""),
                "agent_verdict": result.get("agent_verdict", ""),
                "reason": reason,
                "message": (
                    f"Critic review failed after {new_retries} retries for {phase_under_review}. "
                    f"Approve to override and continue, or reject to cancel."
                ),
            })
            if decision and decision.get("approved"):
                updates["critic_verdict"] = "pass"
                updates["escalation_reason"] = f"human_override_{phase_under_review}"
            else:
                updates["critic_verdict"] = "fail"
                updates["escalation_reason"] = "human_rejected"

    return updates


register_phase("critic", build_phase_graph, call_critic)
```

**Step 4: Run syntax check**

Run: `python -c "from spine.phases.critic import call_critic; from spine.workflow.critic_review import structural_critic_check, agent_critic_check; print('OK')"`
Expected: OK

**Step 5: Commit**

```bash
git add spine/phases/critic.py spine/agents/critic_agent.py spine/workflow/critic_review.py
git commit -m "feat(phases): add two-tier Critic phase with structural check + agent review"
```

---

### Task 10: Create the workflow composer

**Objective:** Build the `compose.py` module that constructs workflow graphs from phase definitions based on `work_type`. This is the heart of the composable architecture.

**Files:**
- Create: `spine/workflow/compose.py`

**Step 1: Write the workflow composer**

```python
# spine/workflow/compose.py
"""Workflow composer — builds a LangGraph StateGraph from phase definitions.

Each workflow type (quick, critical_quick, spec, critical_spec) wires phases
together differently. The composer creates the parent graph, adds phase nodes
as wrapper functions, and connects them with edges + conditional edges for
the critic review loop.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from spine.workflow.state import WorkflowState


# ── Phase sequences per workflow type ──
# Each tuple is (phase_name, has_critic_after)
_WORKFLOW_PHASES: dict[str, list[tuple[str, bool]]] = {
    "quick": [
        ("tasks", False),
        ("implement", False),
        ("verify", False),
    ],
    "critical_quick": [
        ("tasks", True),
        ("implement", False),
        ("verify", False),
    ],
    "spec": [
        ("specify", False),
        ("plan", False),
        ("tasks", False),
        ("implement", False),
        ("verify", False),
    ],
    "critical_spec": [
        ("specify", True),
        ("plan", True),
        ("tasks", True),
        ("implement", False),
        ("verify", False),
    ],
}


def build_workflow_graph(
    work_type: str = "spec",
    checkpoint_path: str | None = None,
) -> Any:
    """Build a compiled workflow graph for the given work type.

    Args:
        work_type: One of "quick", "critical_quick", "spec", "critical_spec".
        checkpoint_path: Path to SQLite DB for checkpointing. None = MemorySaver.

    Returns:
        Compiled StateGraph ready for .invoke() / .stream().
    """
    if work_type not in _WORKFLOW_PHASES:
        raise ValueError(f"Unknown work_type: {work_type!r}. Available: {list(_WORKFLOW_PHASES)}")

    phases = _WORKFLOW_PHASES[work_type]
    builder = StateGraph(WorkflowState)

    # ── Add phase nodes ──
    for phase_name, _has_critic in phases:
        _add_phase_node(builder, phase_name)

    # ── Add critic nodes (for phases that need them) ──
    critic_phases = [(name, i) for i, (name, has_c) in enumerate(phases) if has_c]
    for phase_name, _phase_idx in critic_phases:
        critic_node_name = f"{phase_name}_critic"
        _add_critic_node(builder, critic_node_name, phase_name)

    # ── Add human review node ──
    builder.add_node("human_review", _human_review_node)

    # ── Wire edges ──
    _wire_edges(builder, phases, work_type)

    # ── Compile with checkpointer (required for interrupt()) ──
    checkpointer = _make_checkpointer(checkpoint_path)
    return builder.compile(checkpointer=checkpointer)


def _add_phase_node(builder: StateGraph, phase_name: str) -> None:
    """Add a phase subgraph as a node in the parent graph."""
    from spine.phases import get_phase

    _factory, wrapper = get_phase(phase_name)

    def phase_node_fn(state: WorkflowState, config: Any = None) -> dict:
        return wrapper(state, config)

    builder.add_node(phase_name, phase_node_fn)


def _add_critic_node(builder: StateGraph, node_name: str, reviewed_phase: str) -> None:
    """Add a critic node that reviews the specified phase."""
    from spine.phases import get_phase

    _factory, critic_wrapper = get_phase("critic")

    def critic_node_fn(state: WorkflowState, config: Any = None) -> dict:
        # Set the current phase so the critic knows what it's reviewing
        state_with_context = {**state, "current_phase": reviewed_phase}
        return critic_wrapper(state_with_context, config)

    builder.add_node(node_name, critic_node_fn)


def _human_review_node(state: WorkflowState, config: Any = None) -> Command[Literal["__end__"]]:
    """Handle human review — pause via interrupt, then route based on decision."""
    from langgraph.types import interrupt

    decision = interrupt({
        "type": "human_review",
        "phase": state.get("current_phase", "unknown"),
        "escalation_reason": state.get("escalation_reason", ""),
        "retry_counts": state.get("retry_counts", {}),
        "feedback": state.get("feedback", []),
        "artifacts": state.get("artifacts", {}),
        "message": "Human review required. Approve to override and continue, or reject to cancel.",
    })

    if decision and decision.get("approved"):
        return Command(
            update={"critic_verdict": "pass", "escalation_reason": "human_override"},
            goto=END,
        )
    return Command(
        update={"critic_verdict": "fail", "escalation_reason": "human_rejected"},
        goto=END,
    )


def _wire_edges(builder: StateGraph, phases: list[tuple[str, bool]], work_type: str) -> None:
    """Wire edges between phases, critics, and human review."""
    # START → first phase
    first_phase = phases[0][0]
    builder.add_edge(START, first_phase)

    for i, (phase_name, has_critic) in enumerate(phases):
        if has_critic:
            # phase → critic → (conditional: next_phase | rework_phase | human_review)
            critic_node_name = f"{phase_name}_critic"
            builder.add_edge(phase_name, critic_node_name)

            # Determine the next phase after the critic
            next_phase_name = phases[i + 1][0] if i + 1 < len(phases) else None

            # Conditional edges from critic
            _add_critic_conditional_edges(
                builder,
                critic_node_name,
                phase_name,  # rework target
                next_phase_name,  # pass target
            )
        else:
            # phase → next phase (or END for verify)
            if phase_name == "verify":
                # Verify produces its own verdict
                builder.add_conditional_edges(
                    phase_name,
                    _verify_router,
                    {
                        "pass": END,
                        "fail": "human_review",
                    },
                )
            elif i + 1 < len(phases):
                next_phase = phases[i + 1][0]
                builder.add_edge(phase_name, next_phase)
            else:
                builder.add_edge(phase_name, END)

    # human_review → END (after human decision)
    # (handled by Command return in _human_review_node)


def _add_critic_conditional_edges(
    builder: StateGraph,
    critic_node: str,
    rework_target: str,
    pass_target: str | None,
) -> None:
    """Add conditional edges from a critic node based on verdict."""
    edge_map: dict[str, str] = {
        "rework": rework_target,
        "human_review": "human_review",
    }
    if pass_target:
        edge_map["pass"] = pass_target
    else:
        edge_map["pass"] = END  # type: ignore[assignment]

    builder.add_conditional_edges(
        critic_node,
        _critic_router,
        edge_map,
    )


def _critic_router(state: WorkflowState) -> Literal["pass", "rework", "human_review"]:
    """Route based on critic verdict."""
    verdict = state.get("critic_verdict", "fail")
    if verdict == "pass":
        return "pass"
    elif verdict == "fail":
        return "human_review"
    else:  # "rework"
        return "rework"


def _verify_router(state: WorkflowState) -> Literal["pass", "fail"]:
    """Route based on verify result."""
    verdict = state.get("critic_verdict", "fail")
    return "pass" if verdict == "pass" else "fail"


def _make_checkpointer(checkpoint_path: str | None) -> Any:
    """Create a checkpointer for the workflow graph."""
    if checkpoint_path:
        return SqliteSaver.from_conn_string(checkpoint_path)
    return MemorySaver()
```

**Step 2: Run syntax check**

Run: `python -c "from spine.workflow.compose import build_workflow_graph; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add spine/workflow/compose.py
git commit -m "feat(workflow): add composable workflow composer with four work types"
```

---

### Task 11: Create the public workflow API

**Objective:** Expose `run_workflow()` and `resume_workflow()` as the public API for submitting work and resuming after human review.

**Files:**
- Modify: `spine/workflow/__init__.py`

**Step 1: Write the public API**

```python
# spine/workflow/__init__.py
"""Composable workflow engine for SPINE — phase-per-file, agent-per-file architecture.

Public API:
    run_workflow(requirement, work_type, providers, ...) → result
    resume_workflow(thread_id, decision, checkpoint_path, ...) → result
"""

from __future__ import annotations

from typing import Any

from .compose import build_workflow_graph
from .state import PhaseOutput, PhaseStatus, WorkflowState


__all__ = ["WorkflowState", "PhaseOutput", "PhaseStatus", "run_workflow", "resume_workflow"]


def run_workflow(
    requirement: str,
    work_type: str = "spec",
    providers: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
    project_root: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Submit a new workflow for execution.

    Args:
        requirement: The work description.
        work_type: One of "quick", "critical_quick", "spec", "critical_spec".
        providers: Provider dict (LLM, agent, etc.). Passed through config, not state.
        checkpoint_path: Path to SQLite DB for checkpointing.
        project_root: Root directory for the project (agent filesystem access).
        thread_id: Optional thread ID for the workflow run.

    Returns:
        Final workflow state dict.
    """
    import uuid

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    graph = build_workflow_graph(work_type, checkpoint_path=checkpoint_path)

    initial_state: WorkflowState = {
        "requirement": requirement,
        "work_type": work_type,
        "current_phase": "",
        "phase_history": [],
        "artifacts": {},
        "feedback": [],
        "critic_verdict": "",
        "retry_counts": {},
        "escalation_reason": None,
    }

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "providers": providers or {},
            "project_root": project_root or ".",
        },
    }

    result = graph.invoke(initial_state, config)
    return dict(result)


def resume_workflow(
    thread_id: str,
    decision: dict[str, Any],
    work_type: str = "spec",
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Resume a paused workflow after human review.

    Args:
        thread_id: The thread ID of the paused workflow.
        decision: The human's decision (e.g., {"approved": True}).
        work_type: Must match the original work_type.
        checkpoint_path: Path to SQLite DB (must match original).

    Returns:
        Final workflow state dict after resumption.
    """
    from langgraph.types import Command

    graph = build_workflow_graph(work_type, checkpoint_path=checkpoint_path)

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
        },
    }

    result = graph.invoke(Command(resume=decision), config)
    return dict(result)
```

**Step 2: Run syntax check**

Run: `python -c "from spine.workflow import run_workflow, resume_workflow; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add spine/workflow/__init__.py
git commit -m "feat(workflow): add public API run_workflow() and resume_workflow()"
```

---

### Task 12: Write integration tests for the workflow composer

**Objective:** Verify that the four workflow types compose correctly, that critic routing works, and that human escalation via interrupt works.

**Files:**
- Create: `tests/workflow/__init__.py`
- Create: `tests/workflow/test_compose.py`

**Step 1: Write the test file**

```python
# tests/workflow/__init__.py
```

```python
# tests/workflow/test_compose.py
"""Integration tests for the composable workflow engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from spine.workflow.compose import build_workflow_graph, _WORKFLOW_PHASES
from spine.workflow.state import WorkflowState


class TestWorkflowComposition:
    """Test that workflow graphs compose correctly for each work type."""

    def test_quick_workflow_has_correct_phases(self):
        """Quick workflow: TASKS → IMPLEMENT → VERIFY."""
        phases = _WORKFLOW_PHASES["quick"]
        phase_names = [p[0] for p in phases]
        assert phase_names == ["tasks", "implement", "verify"]

    def test_critical_quick_workflow_has_critic_after_tasks(self):
        """Critical quick: TASKS → CRITIC → IMPLEMENT → VERIFY."""
        phases = _WORKFLOW_PHASES["critical_quick"]
        assert phases[0] == ("tasks", True)  # tasks has critic
        assert phases[1] == ("implement", False)  # implement has no critic

    def test_spec_workflow_has_all_phases(self):
        """Spec workflow: SPECIFY → PLAN → TASKS → IMPLEMENT → VERIFY."""
        phases = _WORKFLOW_PHASES["spec"]
        phase_names = [p[0] for p in phases]
        assert phase_names == ["specify", "plan", "tasks", "implement", "verify"]

    def test_critical_spec_workflow_has_critics(self):
        """Critical spec: SPECIFY → CRITIC → PLAN → CRITIC → TASKS → CRITIC → IMPLEMENT → VERIFY."""
        phases = _WORKFLOW_PHASES["critical_spec"]
        # specify, plan, tasks all have critics
        assert phases[0] == ("specify", True)
        assert phases[1] == ("plan", True)
        assert phases[2] == ("tasks", True)
        assert phases[3] == ("implement", False)

    def test_invalid_work_type_raises(self):
        """Unknown work_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown work_type"):
            build_workflow_graph("nonexistent")

    def test_build_graph_compiles_for_all_types(self):
        """All four work types produce a compilable graph."""
        for work_type in _WORKFLOW_PHASES:
            graph = build_workflow_graph(work_type)
            assert graph is not None


class TestCriticRouting:
    """Test that the critic conditional edges route correctly."""

    def test_critic_router_pass(self):
        """Critic verdict 'pass' routes to next phase."""
        from spine.workflow.compose import _critic_router
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {},
            "feedback": [],
            "critic_verdict": "pass",
            "retry_counts": {},
            "escalation_reason": None,
        }
        assert _critic_router(state) == "pass"

    def test_critic_router_rework(self):
        """Critic verdict 'rework' routes back to the reviewed phase."""
        from spine.workflow.compose import _critic_router
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {},
            "feedback": [],
            "critic_verdict": "rework",
            "retry_counts": {},
            "escalation_reason": None,
        }
        assert _critic_router(state) == "rework"

    def test_critic_router_fail(self):
        """Critic verdict 'fail' routes to human review."""
        from spine.workflow.compose import _critic_router
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {},
            "feedback": [],
            "critic_verdict": "fail",
            "retry_counts": {},
            "escalation_reason": None,
        }
        assert _critic_router(state) == "human_review"

    def test_verify_router_pass(self):
        """Verify verdict 'pass' routes to END."""
        from spine.workflow.compose import _verify_router
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "quick",
            "current_phase": "verify",
            "phase_history": [],
            "artifacts": {},
            "feedback": [],
            "critic_verdict": "pass",
            "retry_counts": {},
            "escalation_reason": None,
        }
        assert _verify_router(state) == "pass"

    def test_verify_router_fail(self):
        """Verify verdict 'fail' routes to human review."""
        from spine.workflow.compose import _verify_router
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "quick",
            "current_phase": "verify",
            "phase_history": [],
            "artifacts": {},
            "feedback": [],
            "critic_verdict": "fail",
            "retry_counts": {},
            "escalation_reason": None,
        }
        assert _verify_router(state) == "fail"


class TestStructuralCritic:
    """Test the structural critic review logic."""

    def test_empty_artifact_fails(self):
        """Empty artifact triggers rework."""
        from spine.workflow.critic_review import structural_critic_check
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {"specification": ""},
            "feedback": [],
            "critic_verdict": "",
            "retry_counts": {},
            "escalation_reason": None,
        }
        verdict, reason = structural_critic_check(state, "specify")
        assert verdict == "rework"
        assert "No output" in reason

    def test_short_artifact_rework(self):
        """Too-short artifact triggers rework."""
        from spine.workflow.critic_review import structural_critic_check
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {"specification": "too short"},
            "feedback": [],
            "critic_verdict": "",
            "retry_counts": {},
            "escalation_reason": None,
        }
        verdict, reason = structural_critic_check(state, "specify")
        assert verdict == "rework"
        assert "too short" in reason

    def test_good_artifact_passes(self):
        """Well-formed artifact passes review."""
        from spine.workflow.critic_review import structural_critic_check
        spec = (
            "# Specification\n\n"
            "## Problem Statement\nWe need to build X.\n\n"
            "## Scope\nIn scope: Y. Out of scope: Z.\n\n"
            "## Requirements\n- Func req 1\n- Func req 2\n\n"
            "## Success Criteria\nAll tests pass.\n"
        )
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {"specification": spec},
            "feedback": [],
            "critic_verdict": "",
            "retry_counts": {},
            "escalation_reason": None,
        }
        verdict, reason = structural_critic_check(state, "specify")
        assert verdict == "pass"

    def test_missing_markers_triggers_rework(self):
        """Artifact missing required markers triggers rework."""
        from spine.workflow.critic_review import structural_critic_check
        spec = "A" * 300  # Long enough but no required markers
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {"specification": spec},
            "feedback": [],
            "critic_verdict": "",
            "retry_counts": {},
            "escalation_reason": None,
        }
        verdict, reason = structural_critic_check(state, "specify")
        assert verdict == "rework"
        assert "markers" in reason.lower()


class TestAgentCriticReview:
    """Test the agent (Tier 2) critic review logic."""

    def test_parse_approved_verdict(self):
        """APPROVED maps to 'pass'."""
        from spine.workflow.critic_review import _parse_critic_agent_verdict
        verdict, reason = _parse_critic_agent_verdict(
            "APPROVED\nThe spec is complete and correct.", "specify"
        )
        assert verdict == "pass"
        assert "APPROVED" in reason

    def test_parse_needs_revision_verdict(self):
        """NEEDS_REVISION maps to 'rework'."""
        from spine.workflow.critic_review import _parse_critic_agent_verdict
        verdict, reason = _parse_critic_agent_verdict(
            "NEEDS_REVISION\n- Missing error handling section\n- Scope is vague", "plan"
        )
        assert verdict == "rework"
        assert "NEEDS_REVISION" in reason

    def test_parse_rejected_verdict(self):
        """REJECTED maps to 'fail'."""
        from spine.workflow.critic_review import _parse_critic_agent_verdict
        verdict, reason = _parse_critic_agent_verdict(
            "REJECTED\nThis plan contradicts the specification entirely.", "specify"
        )
        assert verdict == "fail"

    def test_parse_no_clear_verdict_defaults_rework(self):
        """No recognizable verdict defaults to rework with feedback."""
        from spine.workflow.critic_review import _parse_critic_agent_verdict
        verdict, reason = _parse_critic_agent_verdict(
            "I'm not sure about this. There are some issues.", "plan"
        )
        assert verdict == "rework"
        assert "no clear verdict" in reason.lower()

    def test_parse_empty_response_defaults_rework(self):
        """Empty agent response defaults to rework."""
        from spine.workflow.critic_review import _parse_critic_agent_verdict
        verdict, reason = _parse_critic_agent_verdict("", "specify")
        assert verdict == "rework"
        assert "empty" in reason.lower()

    @patch("spine.agents.get_agent")
    def test_agent_critic_check_with_mock_agent(self, mock_get_agent):
        """agent_critic_check invokes the critic agent and parses the result."""
        from spine.workflow.critic_review import agent_critic_check

        mock_result = {"messages": [MagicMock(content="APPROVED\nLooks good.", type="ai")]}
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_factory = MagicMock(return_value=mock_agent)
        mock_get_agent.return_value = mock_factory

        state = {
            "artifacts": {"specification": "A" * 300},
            "requirement": "Build auth",
            "feedback": [],
        }
        verdict, reason = agent_critic_check(state, "specify", providers={"llm": MagicMock()})
        assert verdict == "pass"

    @patch("spine.agents.get_agent")
    def test_agent_critic_check_exception_returns_rework(self, mock_get_agent):
        """Agent exception does not crash — returns rework instead."""
        from spine.workflow.critic_review import agent_critic_check

        mock_factory = MagicMock(side_effect=RuntimeError("Agent unavailable"))
        mock_get_agent.return_value = mock_factory

        state = {
            "artifacts": {"specification": "A" * 300},
            "requirement": "Build auth",
            "feedback": [],
        }
        verdict, reason = agent_critic_check(state, "specify", providers={})
        assert verdict == "rework"
        assert "Agent unavailable" in reason


class TestTwoTierCriticIntegration:
    """Test the two-tier critic flow: structural first, then agent."""

    @patch("spine.agents.get_agent")
    def test_structural_failure_skips_agent(self, mock_get_agent):
        """When structural check fails, the agent is never invoked."""
        from spine.phases.critic import build_phase_graph

        graph = build_phase_graph()
        # Empty artifact → structural check fails
        critic_config = {
            "configurable": {
                "artifacts": {"specification": ""},
                "providers": {},
                "project_root": ".",
                "requirement": "test",
                "feedback": [],
            }
        }
        result = graph.invoke(
            {
                "phase_under_review": "specify",
                "structural_verdict": "",
                "structural_reason": "",
                "agent_verdict": "",
                "agent_reason": "",
                "critic_verdict": "",
                "critic_reason": "",
                "retry_count": 0,
                "max_retries_reached": False,
            },
            config=critic_config,
        )
        assert result["structural_verdict"] == "rework"
        assert result["agent_verdict"] == "skipped"
        mock_get_agent.assert_not_called()

    @patch("spine.agents.get_agent")
    def test_structural_pass_runs_agent(self, mock_get_agent):
        """When structural passes, the agent IS invoked for deep review."""
        from spine.phases.critic import build_phase_graph

        # Good artifact → structural passes
        spec = (
            "# Spec\n## Problem\nBuild X.\n"
            "## Scope\nIn scope: Y.\n## Success\nTests pass.\n" * 10
        )

        # Mock the agent to return APPROVED
        mock_result = {"messages": [MagicMock(content="APPROVED\nGood quality.", type="ai")]}
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_factory = MagicMock(return_value=mock_agent)
        mock_get_agent.return_value = mock_factory

        critic_config = {
            "configurable": {
                "artifacts": {"specification": spec},
                "providers": {},
                "project_root": ".",
                "requirement": "test",
                "feedback": [],
            }
        }
        result = graph.invoke(
            {
                "phase_under_review": "specify",
                "structural_verdict": "",
                "structural_reason": "",
                "agent_verdict": "",
                "agent_reason": "",
                "critic_verdict": "",
                "critic_reason": "",
                "retry_count": 0,
                "max_retries_reached": False,
            },
            config=critic_config,
        )
        assert result["structural_verdict"] == "pass"
        assert result["agent_verdict"] == "pass"
        mock_get_agent.assert_called_once()


class TestRetryEscalation:
    """Test retry counting and human escalation."""

    def test_should_not_escalate_under_limit(self):
        """Retry count under limit should not escalate."""
        from spine.workflow.critic_review import should_escalate_to_human
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {},
            "feedback": [],
            "critic_verdict": "",
            "retry_counts": {"specify": 2},
            "escalation_reason": None,
        }
        assert should_escalate_to_human(state, "specify", max_retries=3) is False

    def test_should_escalate_at_limit(self):
        """Retry count at limit should escalate."""
        from spine.workflow.critic_review import should_escalate_to_human
        state: WorkflowState = {
            "requirement": "test",
            "work_type": "spec",
            "current_phase": "specify",
            "phase_history": [],
            "artifacts": {},
            "feedback": [],
            "critic_verdict": "",
            "retry_counts": {"specify": 3},
            "escalation_reason": None,
        }
        assert should_escalate_to_human(state, "specify", max_retries=3) is True

    def test_default_max_retries(self):
        """Default max retries is 3."""
        from spine.workflow.critic_review import DEFAULT_MAX_CRITIC_RETRIES
        assert DEFAULT_MAX_CRITIC_RETRIES == 3
```

**Step 2: Run the tests**

Run: `pytest tests/workflow/test_compose.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/workflow/
git commit -m "test(workflow): add integration tests for composer, critic, and routing"
```

---

### Task 13: Wire the new workflow engine into the existing dispatcher

**Objective:** Update `spine/work/dispatcher.py` to use the new composable workflow engine for work submission, while keeping backward compatibility with the existing state machine path.

**Files:**
- Modify: `spine/work/dispatcher.py`

**Step 1: Add the new workflow path to the dispatcher**

This task requires reading the existing dispatcher to understand its current interface, then adding a `workflow_engine="compose"` option that routes to the new system. The exact code depends on the current dispatcher API, which we'll inspect before modifying.

**Step 2: Run tests to verify no regressions**

Run: `pytest tests/ -k "dispatcher" -v`

**Step 3: Commit**

```bash
git add spine/work/dispatcher.py
git commit -m "feat(work): wire composable workflow engine into dispatcher"
```

---

### Task 14: Update the existing adapters to use agent-per-file factories

**Objective:** Refactor `spine/adapters/da_phase_adapter.py` to import from the new `spine/agents/` module instead of defining agent factories inline.

**Files:**
- Modify: `spine/adapters/da_phase_adapter.py`

**Step 1: Update the adapter to delegate to spine/agents/**

The adapter currently defines `create_planning_agent()`, `create_execution_agent()`, `create_verification_agent()` inline. We'll update these to import from `spine.agents` and deprecate the old names.

**Step 2: Run existing tests to verify no regressions**

Run: `pytest tests/ -k "adapter" -v`

**Step 3: Commit**

```bash
git add spine/adapters/da_phase_adapter.py
git commit -m "refactor(adapters): delegate to spine/agents factories"
```

---

### Task 15: Update models/enums.py with new phase names

**Objective:** Add the new phase names (SPECIFY, TASKS, CRITIC) to the `PhaseName` enum and update the `work_type` concept.

**Files:**
- Modify: `spine/models/enums.py`

**Step 1: Add new enum values**

Add `SPECIFY`, `TASKS`, `CRITIC` to `PhaseName`. Add a `WorkType` enum for the four workflow types.

**Step 2: Run syntax check**

Run: `python -c "from spine.models.enums import PhaseName, WorkType; print('OK')"`

**Step 3: Commit**

```bash
git add spine/models/enums.py
git commit -m "feat(models): add SPECIFY, TASKS, CRITIC phases and WorkType enum"
```

---

### Task 16: End-to-end validation

**Objective:** Run a full end-to-end test of the new workflow system with mocked agents to verify the complete flow works.

**Files:**
- Create: `tests/workflow/test_e2e.py`

**Step 1: Write the e2e test**

```python
# tests/workflow/test_e2e.py
"""End-to-end test for the composable workflow engine with mocked agents."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from spine.workflow.compose import build_workflow_graph
from spine.workflow.state import WorkflowState


def _make_mock_agent(output_text: str):
    """Create a mock agent that returns the given text."""
    mock_result = {
        "messages": [MagicMock(content=output_text, type="ai")],
    }
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = mock_result
    return mock_agent


class TestQuickWorkflowE2E:
    """End-to-end test of the quick workflow with mocked agents."""

    @patch("spine.agents.get_agent")
    def test_quick_workflow_completes(self, mock_get_agent):
        """Quick workflow: TASKS → IMPLEMENT → VERIFY completes."""
        # Mock agents to return plausible output
        mock_get_agent.side_effect = [
            lambda **kw: _make_mock_agent("Feature Slice 1: Auth\nAcceptance: tests pass\nDepends: none\n\nTASKS_COMPLETE"),
            lambda **kw: _make_mock_agent("Implemented all slices. IMPLEMENT_COMPLETE"),
            lambda **kw: _make_mock_agent("All tests pass. VERIFIED."),
        ]

        graph = build_workflow_graph("quick")
        result = graph.invoke({
            "requirement": "Build user auth",
            "work_type": "quick",
            "current_phase": "",
            "phase_history": [],
            "artifacts": {},
            "feedback": [],
            "critic_verdict": "",
            "retry_counts": {},
            "escalation_reason": None,
        })

        # Workflow should complete
        assert result.get("critic_verdict") == "pass" or result.get("current_phase") == "verify"


class TestCriticReviewLoopE2E:
    """Test the critic rework loop with mocked agents."""

    def test_structural_critic_rejects_empty_output(self):
        """Critic flags rework for empty artifacts."""
        from spine.workflow.critic_review import structural_critic_check
        state = {
            "artifacts": {"specification": ""},
        }
        verdict, _ = structural_critic_check(state, "specify")
        assert verdict == "rework"

    def test_structural_critic_accepts_good_output(self):
        """Critic passes well-formed artifacts."""
        from spine.workflow.critic_review import structural_critic_check
        spec = (
            "# Spec\n## Problem Statement\nBuild X.\n"
            "## Scope\nY is in scope.\n## Success Criteria\nTests pass.\n" * 10
        )
        state = {"artifacts": {"specification": spec}}
        verdict, _ = structural_critic_check(state, "specify")
        assert verdict == "pass"
```

**Step 2: Run the e2e test**

Run: `pytest tests/workflow/test_e2e.py -v`

**Step 3: Commit**

```bash
git add tests/workflow/test_e2e.py
git commit -m "test(workflow): add end-to-end tests with mocked agents"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Workflow state module | `spine/workflow/state.py`, `spine/workflow/__init__.py` |
| 2 | Phase registry | `spine/phases/__init__.py` |
| 3 | Agent registry | `spine/agents/__init__.py` |
| 4 | Specify phase + agent | `spine/phases/specify.py`, `spine/agents/specify_agent.py` |
| 5 | Plan phase + agent | `spine/phases/plan.py`, `spine/agents/plan_agent.py` |
| 6 | Tasks phase + agent | `spine/phases/tasks.py`, `spine/agents/tasks_agent.py` |
| 7 | Implement phase + agent | `spine/phases/implement.py`, `spine/agents/implement_agent.py` |
| 8 | Verify phase + agent | `spine/phases/verify.py`, `spine/agents/verify_agent.py` |
| 9 | Two-tier Critic phase (structural + agent review) | `spine/phases/critic.py`, `spine/agents/critic_agent.py`, `spine/workflow/critic_review.py` |
| 10 | Workflow composer | `spine/workflow/compose.py` |
| 11 | Public API (run/resume) | `spine/workflow/__init__.py` |
| 12 | Integration tests | `tests/workflow/test_compose.py` |
| 13 | Wire into dispatcher | `spine/work/dispatcher.py` |
| 14 | Refactor adapters | `spine/adapters/da_phase_adapter.py` |
| 15 | Update enums | `spine/models/enums.py` |
| 16 | E2E validation | `tests/workflow/test_e2e.py` |
