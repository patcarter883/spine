# SPINE Hybrid Architecture: Deep Agents Integration Plan

**Version:** 1.0  
**Date:** 2026-05-11  
**Status:** Draft  

---

## 1. Executive Summary

SPINE will retain its phased state machine as the top-level orchestrator — this is
our differentiator and cannot be sacrificed. Beneath each phase, we will replace the
current ad-hoc LLM calling and OpenCode subprocess integration with Deep Agents'
`create_deep_agent()` as the execution runtime. This gives us:

- Production-grade middleware (summarization, tool-error handling, patch-tool-calls)
- A battle-tested subagent delegation system via the `task` tool
- Pluggable sandbox backends for future cloud isolation
- Automatic context compaction instead of manual prompt construction

The hybrid keeps what SPINE does uniquely well (enforced phases, critic gates,
FeatureSlice decomposition) while adopting Deep Agents' mature infrastructure for
everything SPINE currently builds by hand and gets wrong.

---

## 2. Problem Statement

### 2.1 Current Pain Points

| Pain Point | Root Cause | Deep Agents Solution |
|------------|-----------|----------------------|
| Context overflow on long tasks | Manual prompt building in `_build_task_prompt()` | `ToolOutputTrimmer` auto-evicts old tool results |
| Tool-call format errors from local models | No recovery — malformed JSON crashes the loop | `PatchToolCallsMiddleware` auto-corrects common malformations |
| OpenCode + vLLM returns 3-24 tokens | Protocol mismatch between OpenCode's ACP and vLLM's OpenAI endpoint | Use `create_deep_agent()` directly with `langchain.chat_models.init_chat_model()` — bypass OpenCode entirely for local models |
| Provider objects lost after checkpoint | Custom `ProviderSerializer` is fragile (msgpack format bugs) | Deep Agents passes providers through `config["configurable"]`, not state — same pattern we should adopt |
| No structured subagent delegation | `SwarmDAGExecutor` builds shell commands manually | `SubAgentMiddleware` + `task` tool handles delegation with isolated context windows |
| Middleware is ad-hoc | Hooks in `state_machine.py` are inline lambdas | Deep Agents middleware protocol: `@before_model`, `@after_agent`, `@on_tool_error` |

### 2.2 What We Must Not Lose

- **Phased enforcement**: PLANNING → EXECUTION → VERIFICATION is structurally
  guaranteed, not prompt-suggested
- **Critic gate**: Plan cannot proceed to execution without `APPROVED` status
- **FeatureSlice decomposition**: Work is decomposed into independent, delegatable
  units with scope, acceptance criteria, and dependency ordering
- **Wave-based DAG execution**: Subphases with dependencies execute in parallel
  waves, not sequentially
- **Three-layer persistence**: spec/, state/, knowledge/ separation

### 2.3 What Deep Agents Does Not Provide

- No concept of phases or enforced workflow transitions
- No critic gate or quality gate mechanism
- No FeatureSlice decomposition — its `write_todos` is a flat list
- No structured planning output — planning is whatever the LLM decides to put in todos
- No verification phase — "run tests" is a suggestion in the system prompt

---

## 3. Architecture

### 3.1 Layered Model

```
┌─────────────────────────────────────────────────────────┐
│                    SPINE STATE MACHINE                    │
│  INIT → PLANNING → EXECUTION → VERIFICATION → COMPLETE   │
│  (LangGraph StateGraph — retained as-is)                 │
│                                                          │
│  • should_continue() routing                             │
│  • critic_gate_result enforcement                        │
│  • FeatureSlice synthesis                               │
│  • Three-layer persistence                              │
│  • Entry/exit condition evaluation                       │
└────────────────────────┬────────────────────────────────┘
                         │ delegates work to
                         ▼
┌─────────────────────────────────────────────────────────┐
│                DEEP AGENTS RUNTIME LAYER                  │
│  create_deep_agent() instances — one per phase           │
│                                                          │
│  • ToolOutputTrimmer (auto tool result eviction)        │
│  • SubAgentMiddleware (task tool for delegation)         │
│  • FilesystemMiddleware (read/write/edit/ls/glob/grep)   │
│  • PatchToolCallsMiddleware (tool-call error recovery)   │
│  • SandboxBackend (pluggable execution environment)      │
│  • MemoryMiddleware (AGENTS.md injection)                │
│  • SkillsMiddleware (reusable prompt templates)         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Key Design Decisions

**Decision 1: State machine stays at the top level.**  
The LangGraph `StateGraph` with `should_continue()` routing remains the orchestrator.
Deep Agents is NOT the outer loop. Deep Agents instances are created and invoked
*inside* phase functions.

**Decision 2: One `create_deep_agent()` per phase.**  
Each SPINE phase (PLANNING, EXECUTION, VERIFICATION) constructs its own
`create_deep_agent()` with phase-specific system prompts, tools, and middleware.
This preserves phase isolation while giving each phase the full DA infrastructure.

**Decision 3: FeatureSlices map to SubAgent specs.**  
During the EXECUTION phase, each FeatureSlice becomes a declarative `SubAgent`
spec passed to `create_deep_agent(subagents=[...])`. The DA `task` tool lets the
main execution agent delegate each slice to an isolated subagent with its own
context window, middleware stack, and tool access.

**Decision 4: Replace OpenCode subprocess with direct DA execution.**  
For local models (vLLM, Ollama), bypass OpenCode entirely. Use
`langchain.chat_models.init_chat_model("openai:<model>", base_url=...)` 
directly. OpenCode becomes optional — only used when the user explicitly wants
its specialized agents or ACP protocol.

**Decision 5: Providers through config, not state.**  
Adopt Deep Agents' pattern of passing LLM client objects through
`config["configurable"]` rather than storing them in `SpineState`. This
eliminates the serialization bug class entirely.

---

## 4. Detailed Design

### 4.1 SPINE State Machine (Retained)

The state machine graph, transitions, and routing logic remain unchanged from the
current implementation:

```
Nodes: init, planning, execution, verification, rework, blocked, error, human_review
Edges: conditional based on should_continue()
```

What changes is the *body* of each phase function. Currently, each phase manually:

1. Resolves providers from state/config
2. Creates a `SwarmDAGExecutor`
3. Builds prompts by concatenating strings
4. Calls `executor.execute_phase()`
5. Post-processes results into state

In the hybrid, each phase will:

1. Resolve providers from config (not state)
2. Construct a `create_deep_agent()` with phase-specific configuration
3. Invoke the agent with structured input
4. Extract structured output from the agent's message history
5. Update state from the structured output

### 4.2 Phase-Specific Deep Agent Configurations

#### 4.2.1 PLANNING Phase Agent

```python
def planning_phase(state: SpineState, config: Optional[RunnableConfig] = None) -> SpineState:
    """PLANNING phase using Deep Agents for sub-phase execution."""
    providers = _get_providers_from_config(config)
    llm = providers.get("llm")
    
    # Create the planning agent with read-only tools
    planning_agent = create_deep_agent(
        model=llm.chat_model,  # langchain BaseChatModel
        system_prompt=construct_planning_prompt(
            requirement=state["requirement"],
        ),
        tools=[],  # Planning sub-agents get tools via SubAgent specs below
        subagents=[
            SubAgent(
                name="explorer",
                description="Analyzes requirements and existing codebase",
                system_prompt=construct_explorer_prompt(state["requirement"]),
                tools=[read_file, ls, glob, grep],  # read-only
            ),
            SubAgent(
                name="sme",
                description="Researches technology stack and dependencies",
                system_prompt=construct_sme_prompt(state["requirement"]),
                tools=[read_file, ls, glob, grep, web_search],
            ),
            SubAgent(
                name="analyst",
                description="Identifies risks, constraints, and edge cases",
                system_prompt=construct_analyst_prompt(state["requirement"]),
                tools=[read_file, ls, glob, grep],
            ),
        ],
        middleware=[
            # No FilesystemMiddleware on main agent — planning is read-only
            # Summarization handles long exploration output
        ],
        backend=StateBackend(),  # In-memory for planning
    )
    
    # Invoke: main planning agent delegates to subagents and synthesizes
    result = planning_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Analyze this requirement and create an execution plan: {state['requirement']}"
        }]
    }, config=config)
    
    # Extract structured output from the agent's response
    planning_context = _extract_planning_context(result)
    feature_slices = _extract_feature_slices(result)
    
    # Critic gate (SPINE-specific, not in DA)
    critic_result = _run_critic_gate(planning_context, llm)
    state["critic_gate_result"] = critic_result
    
    if critic_result != "APPROVED":
        state["phase"] = PhaseName.PLANNING  # Loop back
        return state
    
    # Update state with planning results
    state["plan"] = {
        "requirement": state["requirement"],
        "planning_context": planning_context,
        "feature_slices": [s.to_dict() for s in feature_slices],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    state["previous_phase"] = PhaseName.PLANNING
    state["phase"] = PhaseName.EXECUTION
    return state
```

#### 4.2.2 EXECUTION Phase Agent

```python
def execution_phase(state: SpineState, config: Optional[RunnableConfig] = None) -> SpineState:
    """EXECUTION phase using Deep Agents with FeatureSlice subagents."""
    providers = _get_providers_from_config(config)
    llm = providers.get("llm")
    plan = state.get("plan") or {}
    raw_slices = plan.get("feature_slices", [])
    planning_context = plan.get("planning_context", {})
    spec_content = _read_spec(state)
    
    # Build SubAgent specs from FeatureSlices
    subagents = []
    if raw_slices:
        for s_data in raw_slices:
            slice_obj = FeatureSlice.from_dict(s_data)
            subagents.append(SubAgent(
                name=slice_obj.id,
                description=slice_obj.description,
                system_prompt=construct_coder_prompt(
                    slice=slice_obj,
                    planning_context=planning_context,
                    spec_content=spec_content,
                ),
                # Execution subagents get full file + shell tools
                tools=[read_file, write_file, edit_file, execute, ls, glob, grep],
            ))
    
    # Create execution agent
    execution_agent = create_deep_agent(
        model=llm.chat_model,
        system_prompt=construct_execution_orchestrator_prompt(
            planning_context=planning_context,
            spec_content=spec_content,
        ),
        subagents=subagents,
        middleware=[
            # Summarization for long execution sessions
        ],
        backend=get_backend(config),  # Filesystem or sandbox
    )
    
    result = execution_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Implement the planned feature slices. Start with slices that have no dependencies."
        }]
    }, config=config)
    
    # Extract execution results
    execution_results = _extract_execution_results(result)
    state["completed_tasks"].extend(execution_results["completed"])
    state["failed_tasks"].extend(execution_results["failed"])
    
    state["previous_phase"] = PhaseName.EXECUTION
    state["phase"] = PhaseName.VERIFICATION
    return state
```

#### 4.2.3 VERIFICATION Phase Agent

```python
def verification_phase(state: SpineState, config: Optional[RunnableConfig] = None) -> SpineState:
    """VERIFICATION phase using Deep Agents for review and testing."""
    providers = _get_providers_from_config(config)
    llm = providers.get("llm")
    
    verification_agent = create_deep_agent(
        model=llm.chat_model,
        system_prompt=construct_verification_prompt(
            requirement=state["requirement"],
        ),
        subagents=[
            SubAgent(
                name="reviewer",
                description="Reviews code changes for correctness and quality",
                system_prompt=construct_reviewer_prompt(),
                tools=[read_file, execute, ls, glob, grep],
            ),
            SubAgent(
                name="test_engineer",
                description="Runs tests and validates acceptance criteria",
                system_prompt=construct_test_engineer_prompt(),
                tools=[read_file, execute, ls, glob, grep],
            ),
        ],
        middleware=[],
        backend=get_backend(config),
    )
    
    result = verification_agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Verify the implementation meets all acceptance criteria."
        }]
    }, config=config)
    
    verification_results = _extract_verification_results(result)
    
    if verification_results["needs_rework"]:
        state["failed_tasks"] = verification_results["failed_criteria"]
        state["previous_phase"] = PhaseName.VERIFICATION
        state["phase"] = PhaseName.REWORK
    else:
        state["previous_phase"] = PhaseName.VERIFICATION
        state["phase"] = PhaseName.COMPLETE
    
    return state
```

### 4.3 Model Integration: Bypassing OpenCode for Local Models

The current `OpenCodeAgentProvider` shells out to `opencode run` which wraps the LLM
in its own protocol. For local models served by vLLM, this creates a double-encoding
problem (OpenCode's ACP → vLLM's OpenAI-compatible endpoint → model).

**Solution:** Create a `DeepAgentsModelProvider` that directly uses LangChain's
`init_chat_model()`:

```python
# spine/providers/deepagents_model.py

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from .base import Provider, ProviderType

class DeepAgentsModelProvider(Provider):
    """Provider that creates LangChain chat models for direct DA integration.
    
    Replaces OpenCodeAgentProvider for local models where the subprocess
    wrapper adds latency and protocol fragility.
    """
    provider_type = ProviderType.LLM

    def __init__(self):
        self._config: dict = {}
        self._chat_model: BaseChatModel | None = None

    @property
    def name(self) -> str:
        return "deepagents-model"

    def configure(self, config: dict) -> None:
        self._config = config
        model_str = config.get("model", "openai:gpt-4")
        base_url = config.get("base_url")
        api_key = config.get("api_key", "dummy")
        
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        
        # Temperature, max_tokens, etc. passed through
        for key in ("temperature", "max_tokens", "reasoning", "top_p"):
            if key in config:
                kwargs[key] = config[key]
        
        self._chat_model = init_chat_model(model_str, **kwargs)

    @property
    def chat_model(self) -> BaseChatModel | None:
        return self._chat_model

    def validate(self) -> bool:
        if self._chat_model is None:
            return False
        try:
            self._chat_model.invoke("ping")
            return True
        except Exception:
            return False

    @property
    def enabled(self) -> bool:
        return self._chat_model is not None
```

**vLLM-specific configuration:**

```yaml
# spine.yaml
providers:
  llm:
    - name: local-vllm
      type: deepagents-model
      config:
        model: "openai:Qwen/Qwen3-32B"  # or whatever is served
        base_url: "http://localhost:8000/v1"
        api_key: "dummy"  # vLLM doesn't require real keys
        temperature: 0.3
        max_tokens: 8192
```

This completely bypasses OpenCode for local models, eliminating the 3-24 token
problem. OpenCode remains available as an optional agent_provider for users who
want its specialized agents.

### 4.4 Provider Resolution: Config Over State

Current SPINE stores providers in `SpineState["providers"]`, which LangGraph's
checkpointer serializes into plain dicts. The `_get_providers()` function has a
complex fallback chain to handle this.

Deep Agents passes providers through `config["configurable"]`, which is never
serialized. We adopt this pattern:

```python
# Before (state-based, fragile):
providers = _get_providers(state, config)  # Complex fallback chain

# After (config-based, clean):
def _get_providers_from_config(config: Optional[RunnableConfig]) -> dict[str, Any]:
    """Resolve providers exclusively from config — never from state."""
    if config:
        return config.get("configurable", {}).get("providers", {})
    return {}
```

The CLI injects providers at invocation time:

```python
# spine/cli/commands/work.py
config = {
    "configurable": {
        "providers": {
            "llm": llm_provider,
            "agent": agent_provider,  # optional: OpenCode or DA-native
            "memory": memory_provider,
            "storage": storage_provider,
        },
        "thread_id": thread_id,
    },
    "recursion_limit": 9999,
}
app = create_spine_workflow()
result = app.invoke(initial_state, config=config)
```

### 4.5 Middleware Integration

Deep Agents middleware runs inside each `create_deep_agent()` instance. SPINE's
phase-level hooks (pre_execute, post_execute) remain at the state machine level.
These are different concerns:

| Concern | Level | Mechanism |
|---------|-------|-----------|
| Tool-call error recovery | Agent loop | DA `PatchToolCallsMiddleware` |
| Context compaction | Agent loop | DA `ToolOutputTrimmer` |
| Subagent delegation | Agent loop | DA `SubAgentMiddleware` |
| File permission enforcement | Agent loop | DA `FilesystemMiddleware` |
| Phase entry/exit conditions | State machine | SPINE `_evaluate_entry_conditions()` |
| Critic gate validation | State machine | SPINE `_run_critic_gate()` |
| FeatureSlice synthesis | State machine | SPINE `synthesize_slices()` |
| Error state transitions | State machine | SPINE `should_continue()` |
| Step-limit notification | Agent loop | DA `ModelCallLimitMiddleware` |
| Message queue injection | Agent loop | DA `check_message_queue_before_model` |

SPINE-specific middleware (critic gate, step-limit, message queue) can be
implemented as DA-compatible middleware classes that run inside the agent loop.
This is cleaner than the current inline lambda hooks.

```python
# spine/middleware/critic_gate.py

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelResponse

class CriticGateMiddleware(AgentMiddleware):
    """SPINE critic gate as DA-compatible middleware.
    
    Inspects agent output after model calls. If the agent signals
    plan completion, runs critic review. Returns Command to interrupt
    if the plan is rejected.
    """
    
    name = "CriticGateMiddleware"
    
    def __init__(self, llm_provider):
        self._llm = llm_provider
    
    @property
    def after_model(self):
        return True
    
    async def after_model(self, state: AgentState, response: ModelResponse, runtime):
        # Only run critic during PLANNING phase
        if state.get("spine_phase") != "PLANNING":
            return state
        
        # Check if the agent has produced a plan
        last_msg = response.messages[-1] if response.messages else None
        if not last_msg or "PLAN_COMPLETE" not in (last_msg.content or ""):
            return state
        
        # Run critic review
        plan_text = last_msg.content
        critic_result = await self._run_critic(plan_text)
        
        if critic_result == "APPROVED":
            return state
        
        # Inject critic feedback for revision
        return {
            **state,
            "messages": state["messages"] + [{
                "role": "user",
                "content": f"Critic feedback: {critic_result}. Please revise the plan."
            }]
        }
```

### 4.6 Critic Gate: Dual-Level Design

The critic gate exists at two levels in the hybrid:

1. **Agent loop level** (DA middleware): Catches the agent's plan output,
   runs critic, and either approves or injects revision feedback into the
   conversation. This runs *inside* the DA agent loop, allowing iterative
   refinement without leaving the PLANNING phase.

2. **State machine level** (SPINE routing): After the planning DA agent
   completes, `should_continue()` checks `critic_gate_result`. If still
   not `APPROVED`, routes back to planning. This is the structural guarantee
   that prevents the workflow from proceeding without approval.

This dual-level design means:
- The agent can self-correct within a single planning invocation (fast iteration)
- The state machine enforces the invariant that EXECUTION never starts without
  an approved plan (structural guarantee)

### 4.7 Summarization: How It Replaces Manual Prompt Building

Current SPINE builds prompts manually in `_build_task_prompt()`:

```python
prompt = f"## Requirement\n{requirement}\n\n"
prompt += f"## Requirement Analysis\n{analysis}\n\n"
prompt += f"## Technology Research\n{tech}\n\n"
# ... etc
```

This breaks down when context exceeds the model's window. SPINE's
`ToolOutputTrimmer` handles this:

- Monitors token usage per model call
- When usage exceeds 85% of context window, compacts older messages:
  1. Calls an LLM to summarize the conversation so far
  2. Replaces old messages with the summary
  3. Offloads full history to the backend at `/conversation_history/{thread_id}.md`
- Keeps recent messages (last 10%) intact for continuity

**Migration path:**

1. Remove `_build_task_prompt()` and all manual prompt assembly functions
2. Pass structured context as the initial user message to the DA agent
3. Let ToolOutputTrimmer handle overflow automatically
4. Keep spec files on disk for persistence (backend=FilesystemBackend)

### 4.8 Subagent Delegation: FeatureSlices → SubAgent Specs

The current `SwarmDAGExecutor` runs subphases via `ThreadPoolExecutor`, each
calling `agent_provider.execute(prompt, role="coder")`. This creates separate
OpenCode processes.

In the hybrid, FeatureSlices become `SubAgent` specs:

```python
# Current: manual subprocess delegation
subphase = SubPhase(
    name=s.id.upper(),
    tasks=[Task(id=f"{s.id}-exec", description=task_desc)],
)
executor.execute_phase(phase, context)

# Hybrid: declarative subagent delegation
subagents.append(SubAgent(
    name=slice_obj.id,
    description=slice_obj.description,
    system_prompt=construct_coder_prompt(
        slice=slice_obj,
        planning_context=planning_context,
        spec_content=spec_content,
    ),
    tools=[read_file, write_file, edit_file, execute, ls, glob, grep],
))
```

The DA `SubAgentMiddleware` handles:
- Spawning the subagent with its own context window (no context pollution)
- Running subagent to completion with its own middleware stack
- Returning the result to the parent agent
- Parallel vs sequential execution (via `task` tool usage patterns)

**Key advantage:** Subagents in DA have *isolated context windows*. In the current
SPINE, all subphases share the same growing prompt string. In DA, each subagent
starts fresh with only its system prompt and task description. This dramatically
reduces token usage and prevents context pollution between slices.

### 4.9 Sandbox Backend

Deep Agents defines `BackendProtocol` with these implementations:

| Backend | Use Case |
|---------|----------|
| `StateBackend` | In-memory, no filesystem. For planning phases (read-only analysis) |
| `FilesystemBackend` | Local disk. For local development execution |
| `LangSmithSandbox` | Cloud isolation. For CI/CD and production |
| `DaytonaSandbox` | Cloud isolation via Daytona |
| `ModalSandbox` | Serverless execution via Modal |

**SPINE integration:**

```python
def get_backend(config: RunnableConfig) -> BackendProtocol:
    """Select backend based on execution context."""
    backend_type = os.environ.get("SPINE_BACKEND", "filesystem")
    
    if backend_type == "filesystem":
        return FilesystemBackend(root_dir=os.getcwd())
    elif backend_type == "state":
        return StateBackend()
    elif backend_type == "langsmith":
        return LangSmithSandbox(...)  # configured via env vars
    else:
        return FilesystemBackend(root_dir=os.getcwd())
```

For now, `FilesystemBackend` for execution and `StateBackend` for planning are
sufficient. Cloud sandboxes are a future option without code changes.

---

## 5. Migration Plan

### Phase 0: Foundation (Week 1)

**Goal:** Add Deep Agents as a dependency, verify basic integration.

| Task | Files | Est. |
|------|-------|------|
| Add `deepagents>=0.5.0` to pyproject.toml | `pyproject.toml` | 0.5h |
| Create `spine/providers/deepagents_model.py` | New file | 2h |
| Create `spine/middleware/` directory structure | New dir | 0.5h |
| Test: `create_deep_agent()` works with vLLM endpoint | Manual test | 2h |
| Test: `SubAgent` delegation works with local model | Manual test | 2h |
| Write `spine/adapters/da_phase_adapter.py` | New file | 3h |

**Deliverable:** A single phase function that constructs and invokes a
`create_deep_agent()` instance and returns structured output.

**Risk:** vLLM tool-calling may still fail. Mitigation: test with Ollama
(which has known-working tool-calling via `chat_models.init_chat_model`)
and OpenRouter (cloud fallback).

### Phase 1: Planning Phase Migration (Week 2-3)

**Goal:** Replace the planning phase body with DA agent invocation.

| Task | Files | Est. |
|------|-------|------|
| Rewrite `planning_phase()` to use DA agent | `state_machine.py` | 4h |
| Create planning-specific SubAgent specs (explorer, sme, analyst) | `spine/prompts/roles/` | 3h |
| Implement critic gate as DA middleware | `spine/middleware/critic_gate.py` | 3h |
| Move providers from state to config | `state_machine.py`, `cli.py` | 4h |
| Port planning prompts to DA system prompt format | `spine/prompts/` | 2h |
| Test: full planning phase works end-to-end | Integration test | 4h |
| Remove `SwarmDAGExecutor` usage from planning | `state_machine.py` | 1h |

**Deliverable:** Planning phase runs via DA with subagent delegation and
critic gate middleware. Context overflow handled by ToolOutputTrimmer.

**Exit criteria:** `spine work "build a REST API for todo items"` completes
PLANNING phase with APPROVED critic gate result, producing FeatureSlices.

### Phase 2: Execution Phase Migration (Week 3-4)

**Goal:** Replace execution phase with DA subagent delegation.

| Task | Files | Est. |
|------|-------|------|
| Rewrite `execution_phase()` to use DA agent | `state_machine.py` | 4h |
| Map FeatureSlices to SubAgent specs | `state_machine.py` | 2h |
| Construct coder SubAgent prompts with planning context | `spine/prompts/` | 3h |
| Add FilesystemBackend for execution | `state_machine.py` | 1h |
| Test: execution of single-slice plans | Integration test | 3h |
| Test: execution of multi-slice plans with dependencies | Integration test | 4h |
| Remove `SwarmDAGExecutor` from execution | `state_machine.py` | 1h |

**Deliverable:** Execution phase delegates FeatureSlices to DA subagents.
Each subagent has isolated context. FilesystemBackend writes to local disk.

**Exit criteria:** A 2-slice plan (e.g., "add auth middleware + add rate limiter")
executes with both slices producing file changes.

### Phase 3: Verification Phase Migration (Week 5)

**Goal:** Replace verification phase with DA agent + review subagents.

| Task | Files | Est. |
|------|-------|------|
| Rewrite `verification_phase()` to use DA agent | `state_machine.py` | 3h |
| Create reviewer and test_engineer SubAgent specs | `spine/prompts/` | 2h |
| Implement verification criteria extraction | `state_machine.py` | 2h |
| Wire rework routing (failed criteria → REWORK) | `should_continue()` | 1h |
| Test: verification passes for correct code | Integration test | 2h |
| Test: verification fails and routes to rework | Integration test | 2h |

**Deliverable:** Verification phase runs code review + test execution via DA
subagents. Failed criteria route to REWORK phase.

### Phase 4: Cleanup and Hardening (Week 6)

| Task | Files | Est. |
|------|-------|------|
| Remove `SwarmDAGExecutor` entirely | `models/dag.py` | 1h |
| Remove `OpenCodeAgentProvider` (make optional) | `providers/agents.py` | 2h |
| Remove `_build_task_prompt()` and manual prompt assembly | `models/dag.py` | 2h |
| Remove `ProviderSerializer` (providers now in config) | `state_machine.py` | 1h |
| Add `--debug-prompts` flag for DA agent inspection | `cli.py` | 2h |
| Add `--backend` flag (state/filesystem/langsmith) | `cli.py` | 1h |
| Update all tests to use config-based providers | `tests/` | 4h |
| Update DESIGN.md, README.md, PROVIDERS.md | Docs | 2h |
| End-to-end integration test: full workflow | New test | 4h |

**Total estimated effort: ~80 hours (2 weeks full-time or 4 weeks part-time)**

---

## 6. File Map

### New Files

| Path | Purpose |
|------|---------|
| `spine/providers/deepagents_model.py` | LangChain chat model provider for DA |
| `spine/middleware/__init__.py` | Middleware package |
| `spine/middleware/critic_gate.py` | Critic gate as DA middleware |
| `spine/middleware/step_limit.py` | Step limit notification middleware |
| `spine/middleware/message_queue.py` | Mid-run message injection middleware |
| `spine/adapters/__init__.py` | DA adapter package |
| `spine/adapters/da_phase_adapter.py` | Helper to construct DA agents per phase |

### Modified Files

| Path | Changes |
|------|---------|
| `spine/core/state_machine.py` | Phase bodies rewritten; providers from config; remove DAG executor calls |
| `spine/cli/commands/work.py` | Inject providers via config, not state |
| `spine/providers/base.py` | Add `ProviderType.DEEPAGENTS_MODEL` |
| `spine/models/types.py` | Remove provider fields from `SpineState` (keep `agent_provider` as optional) |
| `spine/models/dag.py` | Deprecate `SwarmDAGExecutor` (remove in Phase 4) |
| `spine/prompts/roles/__init__.py` | Update role prompts for DA SubAgent format |
| `pyproject.toml` | Add `deepagents>=0.5.0` dependency |
| `.spine/config.yaml` | Add `backend` option, `model` provider config |

### Removed Files (Phase 4)

| Path | Reason |
|------|--------|
| `spine/models/dag.py` | Replaced by DA subagent system |
| `spine/core/state_machine.py` (ProviderSerializer) | Replaced by config-based provider injection |
| `spine/core/state_machine.py` (_get_providers) | Replaced by `_get_providers_from_config` |

---

## 7. Risk Analysis

### 7.1 vLLM Tool-Calling Compatibility

**Risk:** vLLM may not support LangChain's tool-calling format, same as the
current OpenCode problem.

**Mitigation:**
1. Test Ollama first (known-working tool-calling via LangChain)
2. Test OpenRouter cloud models (guaranteed working)
3. If vLLM tool-calling fails, use DA's `PatchToolCallsMiddleware` to recover
   from partial/malformed tool calls
4. Worst case: use DA's `execute` tool (raw shell) instead of structured
   file tools — the agent can write files via shell commands

**Fallback:** Keep `OpenCodeAgentProvider` as an optional provider. Users who
can't get direct tool-calling working can fall back to `opencode run`.

### 7.2 Subagent Context Isolation

**Risk:** Subagents in DA have isolated context windows. This means a subagent
won't see changes made by a parallel subagent.

**Mitigation:**
1. FeatureSlices already define `depends_on` — DA's `task` tool respects this
   by only delegating when dependencies are complete
2. The main execution agent orchestrates: it only delegates a slice after its
   dependencies' subagents have returned
3. File changes are written to the shared FilesystemBackend, so subsequent
   subagents can `read_file` the outputs of earlier ones

### 7.3 Summarization Accuracy

**Risk:** LLM-generated summaries may lose important details when compacting
context.

**Mitigation:**
1. ToolOutputTrimmer keeps the most recent tool results intact as metadata
2. Full conversation history is offloaded to the backend for retrieval
3. SPINE's spec files remain on disk — the agent can always re-read them
4. Agent can re-read files from disk instead of relying on context history
   manually before critical steps

### 7.4 Breaking State Format

**Risk:** Changing `SpineState` fields breaks existing checkpoints.

**Mitigation:**
1. Migration is a clean break — old checkpoints are not forward-compatible
2. Add a `spine migrate` CLI command to convert old `.spine/spine.db` files
3. Or simply delete old state and start fresh (acceptable for pre-1.0)

---

## 8. Self-Improvement Path

The hybrid architecture enables self-improvement at two levels:

### 8.1 Agent-Level Learning (via DA)

Deep Agents' `SkillsMiddleware` supports reusable prompt templates (skills).
When SPINE identifies a pattern that works (e.g., "how to implement auth
middleware in Express"), it can save this as a skill:

```
.spine/skills/auth-middleware-express.md
```

Future runs with similar requirements can load this skill, avoiding re-discovery.

### 8.2 Workflow-Level Learning (via SPINE)

SPINE's `knowledge/` directory persists:
- Anti-patterns (things that failed)
- Constraints (project-specific rules)
- Lessons learned (from critic gate rejections)

The planning phase agent reads `knowledge/` via DA's `MemoryMiddleware`, 
incorporating past failures into new plans.

### 8.3 Feedback Loop Design

```
Task Execution → Verification Result → Classification
    ↓                                      ↓
Success        → Update skill template    → Knowledge base
Failure        → Classify root cause      → Anti-pattern entry
Partial        → Identify missing info     → Constraint entry

Periodic review:
  Knowledge base → Prompt template updates → Better planning → Fewer failures
```

This is the self-improvement loop. It requires:
1. **Structured task outcomes** — SPINE's phase results provide this
2. **Failure classification** — The critic gate + verification phase provide this
3. **Prompt/tool modification** — Skill templates and knowledge entries provide this
4. **Regression testing** — DA's evals (Harbor) can validate changes

---

## 9. Success Metrics

| Metric | Current | Target (Week 6) | Target (3 months) |
|--------|---------|-----------------|-------------------|
| Planning phase completion rate | ~60% (routing bugs) | 90% | 95% |
| Execution token usage per slice | Manual, unbounded | Auto-compacted | < 50% of current |
| Tool-call error recovery | Crash | Auto-recovery via PatchToolCalls | < 1% unrecoverable |
| Local model compatibility | Broken (3-24 tokens) | Working via direct LangChain | Working via all paths |
| End-to-end workflow completion | ~20% (many bugs) | 70% | 90% |
| Self-improvement feedback loop | Not implemented | Knowledge base writes | Skill template updates |

---

## 10. Open Questions

1. **Async execution:** DA's `create_deep_agent()` is synchronous by default.
   SPINE's current wave-based DAG executor runs subphases in parallel via
   `ThreadPoolExecutor`. Do we need DA's `AsyncSubAgentMiddleware` for parallel
   slice execution, or is sequential subagent delegation sufficient?

2. **Checkpoint compatibility:** Should we maintain backward compatibility
   with existing `.spine/spine.db` files, or treat the migration as a clean
   break? Recommendation: clean break with a `spine migrate` command.

3. **OpenCode as default or optional:** Should the default config use DA's
   native execution (shell + file tools) or continue to shell out to OpenCode
   for coding tasks? Recommendation: DA-native for local models, OpenCode
   optional for cloud models where its agents add value.

4. **Multi-model routing:** DA's `HarnessProfile` system auto-selects prompts
   based on model family. Should SPINE use this, or continue with its own
   role-based prompt system? Recommendation: use HarnessProfile for DA-native
   execution, keep SPINE role prompts for the state machine level.

5. **MCP integration:** DA supports MCP servers via `langchain-mcp-adapters`.
   Should SPINE expose its phases as MCP tools so external agents can invoke
   them? This would enable SPINE-as-a-service. Worth exploring post-migration.
