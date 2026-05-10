# SPINE Agent Prompting Improvement Plan

## Research Summary

### Best Practices from DeepAgents / OpenHarness / LangChain

#### 1. **Structured System Prompts** (DeepAgents Pattern)
DeepAgents uses **middleware-based prompt composition** where each capability adds its own section to the system prompt:

```python
# Filesystem middleware adds:
FILESYSTEM_SYSTEM_PROMPT = """## Following Conventions
- Read files before editing — understand existing content before making changes
- Mimic existing style, naming conventions, and patterns

## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
You have access to a filesystem which you can interact with using these tools.
..."""

# REPL middleware adds:
REPL_SYSTEM_PROMPT = """## REPL tool
You have access to a `repl` tool.
CRITICAL: The REPL does NOT retain state between calls...
- Write assignments like `user = lookup_fn("value")`.
- Use indexing like `items[0]` and `user["id"]`.
- Use `if cond then ... else ... end` for branching.
..."""
```

**Key insight**: Each middleware is responsible for teaching the agent how to use its tools effectively.

#### 2. **Task-Specific Agent Profiles** (LangChain "Tuning Deep Agents")
The blog post emphasizes **model-specific profiles** that adjust:
- System prompts per model family
- Tool descriptions optimized for each model
- Middleware behavior tuned for model strengths

This led to **10-20 point improvements** on benchmarks.

#### 3. **Research Agent Pattern** (DeepAgents Deep Research)
The research agent uses **layered, instructional prompts**:

```python
RESEARCHER_INSTRUCTIONS = """You are a research assistant...

<Task>
Your job is to use tools to gather information about the user's input topic.
You can call these tools in series or in parallel...
</Task>

<Available Research Tools>
You have access to two specific research tools:
1. **tavily_search**: For conducting web searches
2. **think_tool**: For reflection and strategic planning
**CRITICAL: Use think_tool after each search**
</Available Research Tools>

<Instructions>
Think like a human researcher with limited time...
</Instructions>

<Hard Limits>
**Tool Call Budgets**:
- Simple queries: Use 2-3 search tool calls maximum
- Complex queries: Use up to 5 search tool calls maximum
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze...
</Show Your Thinking>

<Final Response Format>
Structure your response: Organize findings with clear headings...
</Final Response Format>
"""
```

**Key patterns**:
- XML-style section tags for structure
- Tool budgets and limits
- Explicit thinking patterns
- Response format specifications

#### 4. **Better Agent Pattern** (Harness Engineering)
The "Better Agent" that improves other harnesses uses **role-specific rules**:

```python
DEFAULT_SYSTEM_PROMPT = """You are Better Agent...

Rules:
- Edit only files under /current.
- Do not edit train_cases, history, or bookkeeping files except /proposal.md.
- Prefer general harness fixes over case-specific hacks.
- Do not overfit to the visible examples.
- The files under /current are the actual harness surfaces.
- If you change tool or middleware behavior, update both implementation and registration.
- Use surface_manifest.json and task.md to understand mapping.
- Keep changes concise and coherent.
- Make the smallest set of edits needed.
- Stop as soon as /current and /proposal.md are updated.
"""
```

#### 5. **Sub-Agent Delegation Pattern**
DeepAgents provides explicit delegation instructions:

```python
SUBAGENT_DELEGATION_INSTRUCTIONS = """# Sub-Agent Research Coordination

## Delegation Strategy
**DEFAULT: Start with 1 sub-agent** for most queries
**ONLY parallelize when the query EXPLICITLY requires comparison...**

## Key Principles
- **Bias towards single sub-agent**: One comprehensive task is more token-efficient
- **Avoid premature decomposition**: Don't break "research X" into multiple narrow tasks
- **Parallelize only for clear comparisons**

## Parallel Execution Limits
- Use at most {max_concurrent_research_units} parallel sub-agents per iteration
"""
```

---

## Current SPINE Issues

### Issue 1: Prompts Are Too Generic
**Current (`spine/swarm/agents.py:143-151`)**:
```python
def _build_prompt(self, state: SpineState, capability: str, **kwargs) -> str:
    """Build LLM prompt for capability execution."""
    return f"""You are a {self.role} agent with capabilities: {self.capabilities}.
Current task: {state.get('requirement', 'Unknown')}
Capability: {capability}
State: {state}
Additional context: {kwargs}

Provide your response:"""
```

This tells the agent **nothing** about:
- How to approach the task
- What tools to use
- What output format to produce
- What constraints to follow
- How to decompose the problem

### Issue 2: Role Prompts Are One-Liners
**Current (`spine/swarm/supervisor.py:124-128`)**:
```python
system_prompt=(
    "You analyze user requirements and extract key information. "
    "Identify the core problem, constraints, and success criteria. "
    "Output should be structured and actionable."
)
```

Compare to DeepAgents research agent: **130+ lines** of structured instructions.

### Issue 3: No Tool Instructions
Agents don't know what tools they have access to or how to use them.

### Issue 4: No Output Format Specification
Agents don't know what format to return results in.

### Issue 5: No Workflow Context
Agents don't know their place in the larger workflow or what other agents have done.

---

## Proposed Improvements

### Phase 1: Create Prompt Templates Module

Create `spine/prompts/` directory with structured templates:

```
spine/prompts/
├── __init__.py
├── base.py           # Base prompt templates
├── roles/
│   ├── explorer.py   # Requirement analysis
│   ├── sme.py        # Research patterns
│   ├── planner.py    # Execution planning
│   ├── critic.py     # Review/validation
│   ├── coder.py      # Implementation
│   └── reviewer.py   # Code review
├── tools/
│   ├── filesystem.py # File tool instructions
│   └── agent.py      # Agent delegation instructions
└── formats/
    ├── json.py       # JSON output formats
    └── markdown.py   # Markdown output formats
```

### Phase 2: Implement Structured Role Prompts

**Example: Explorer Agent**

```python
EXPLORER_PROMPT = """# Requirement Analysis Agent

<Role>
You are an expert requirement analyst. Your job is to decompose user requests into structured, actionable specifications.
</Role>

<YourTask>
Analyze the provided requirement and produce a comprehensive decomposition that other agents can act upon.
</YourTask>

<Process>
1. **Parse the requirement** - Identify the core problem statement
2. **Extract constraints** - Technical, time, resource limitations
3. **Identify stakeholders** - Who benefits, who reviews, who approves
4. **Define success criteria** - How will we know when this is done?
5. **Map dependencies** - What must exist or happen first?
6. **Flag ambiguities** - What needs clarification before proceeding?
</Process>

<OutputFormat>
Return a JSON object with this structure:

```json
{
  "problem_statement": "One-sentence core problem",
  "context": "Background information",
  "constraints": {
    "technical": ["constraint1", ...],
    "time": "deadline or 'none'",
    "resources": "resource constraints"
  },
  "success_criteria": ["criterion1", "criterion2", ...],
  "dependencies": ["dep1", "dep2", ...],
  "ambiguities": ["question1", "question2", ...],
  "suggested_phases": ["phase1", "phase2", ...],
  "risk_areas": ["risk1", "risk2", ...]
}
```
</OutputFormat>

<HardLimits>
- Do NOT start implementation - that's for other agents
- Do NOT skip ambiguities - flag them for clarification
- Do NOT make assumptions about technology unless specified
- Keep the problem statement to ONE sentence
</HardLimits>
"""
```

### Phase 3: Implement Tool Instruction Middleware

**Example: Agent Provider Tool Instructions**

```python
AGENT_PROVIDER_INSTRUCTIONS = """## External Agent Delegation

You have access to an external coding agent (OpenCode, Codex, or Claude Code).

### When to use:
- Implementation tasks that require code changes
- Multi-file refactoring
- Test writing
- Documentation updates

### When NOT to use:
- Simple analysis or planning tasks
- Decision-making that requires human judgment
- Tasks that need clarification first

### How to use:
1. **Be specific** - Describe exactly what needs to be done
2. **Provide context** - Include relevant code snippets or file paths
3. **Set scope** - Specify which directories/files the agent should work in
4. **Define acceptance** - List specific criteria for completion

### Example prompt:
```
Implement user authentication in the `/src/auth` directory.

Requirements:
- JWT-based authentication
- Password hashing with bcrypt
- Session management with Redis

Files to modify:
- src/auth/login.py
- src/auth/session.py
- src/models/user.py

Acceptance criteria:
- [ ] Users can log in with email/password
- [ ] Sessions expire after 24 hours
- [ ] Failed attempts are rate-limited
```
"""
```

### Phase 4: Implement Workflow Context Injection

Each agent receives context about where they fit:

```python
def build_workflow_context(state: SpineState, role: str) -> str:
    """Build context about the current workflow state."""
    completed_phases = state.get("completed_phases", [])
    current_phase = state.get("current_phase", "unknown")
    
    return f"""## Workflow Context

**Your role**: {role}
**Current phase**: {current_phase}
**Completed phases**: {', '.join(completed_phases) or 'None yet'}

**Previous agent outputs**:
{format_previous_outputs(state)}

**Your task**: {get_task_for_role(role, state)}
"""
```

### Phase 5: Centralize Prompt Building

Create a `PromptBuilder` class that composes prompts from components:

```python
class PromptBuilder:
    """Compose agent prompts from multiple sources."""
    
    def __init__(self, role: str, config: PromptConfig):
        self.role = role
        self.config = config
    
    def build(
        self,
        state: SpineState,
        capability: str,
        tools: list[str] | None = None,
        **kwargs
    ) -> str:
        """Build complete prompt for agent execution."""
        parts = []
        
        # 1. Role-specific instructions
        parts.append(ROLE_PROMPTS[self.role])
        
        # 2. Tool instructions (if tools available)
        if tools:
            parts.append(self._build_tool_instructions(tools))
        
        # 3. Workflow context
        parts.append(build_workflow_context(state, self.role))
        
        # 4. Output format
        parts.append(OUTPUT_FORMATS.get(capability, DEFAULT_FORMAT))
        
        # 5. Task-specific context
        if task_context := self._build_task_context(state, capability, **kwargs):
            parts.append(task_context)
        
        return "\n\n".join(parts)
```

### Phase 6: Use Agent Provider Consistently

**Current issue**: Some code paths use LLM directly, others use agent provider.

**Proposed**: All implementation tasks go through agent provider. Decision-making/planning uses LLM provider.

```python
class SwarmAgent:
    IMPLEMENTATION_ROLES = {"coder", "test_engineer", "reviewer"}
    DECISION_ROLES = {"explorer", "planner", "critic", "analyst"}
    
    def execute(self, state: SpineState, capability: str, **kwargs):
        if self.role in self.IMPLEMENTATION_ROLES and self._agent_provider:
            # Delegate to external agent with structured prompt
            prompt = self._prompt_builder.build(state, capability, **kwargs)
            return self._agent_provider.execute(prompt, **kwargs)
        else:
            # Use LLM for decision-making
            prompt = self._prompt_builder.build(state, capability, **kwargs)
            return self._llm_provider.generate(prompt)
```

---

## Implementation Priority

### Critical (Do First)
1. Create `spine/prompts/` module structure
2. Implement 5 core role prompts (explorer, planner, coder, critic, sme)
3. Update `SwarmAgent._build_prompt()` to use structured templates

### High
4. Implement tool instruction middleware
5. Implement workflow context injection
6. Add output format specifications

### Medium
7. Create model-specific prompt profiles (like DeepAgents)
8. Add prompt versioning for A/B testing
9. Implement prompt metrics/tracking

### Low
10. Create prompt debugging/visualization tools
11. Add prompt optimization harness (like "Better Agent")

---

## Key Metrics to Track

1. **Task completion rate** - Are agents completing their assigned tasks?
2. **Clarification requests** - Are agents asking fewer questions?
3. **Output quality** - Is output structured and usable by downstream agents?
4. **Token efficiency** - Are we using tokens effectively?
5. **Phase transition success** - Are handoffs between agents clean?

---

## Example: Before/After Comparison

### Before (Current)
```
You are a coder agent with capabilities: ['implement', 'test'].
Current task: Add user authentication
Capability: implement
State: {'requirement': 'Add user authentication', ...}
Additional context: {}

Provide your response:
```

### After (Proposed)
```
# Implementation Agent

<Role>
You are an expert software engineer. Your job is to implement features according to specifications.
</Role>

<YourTask>
Implement the feature described below. You have access to an external coding agent.
</YourTask>

<AvailableTools>
## External Agent Delegation
You have access to an external coding agent. Use it for:
- Writing and modifying code
- Running tests
- Creating documentation

Instructions for delegation: [detailed instructions...]
</AvailableTools>

<Process>
1. Read the specification carefully
2. Identify files that need modification
3. Break into small, testable changes
4. Implement incrementally
5. Verify with tests
</Process>

<OutputFormat>
Return JSON:
{
  "files_changed": ["path1", "path2"],
  "summary": "What was implemented",
  "tests_passed": true/false,
  "remaining_work": ["item1", ...],
  "notes": "Any important notes"
}
</OutputFormat>

## Workflow Context
**Your role**: coder
**Current phase**: IMPLEMENTATION
**Completed phases**: ANALYSIS, PLANNING
**Previous outputs**: [structured data from previous agents...]

## Task
Implement user authentication with the following specification:
[detailed specification from explorer/planner...]

<HardLimits>
- Only modify files within approved scope
- All changes must have tests
- Follow existing code patterns
- No breaking changes to public APIs
</HardLimits>
```

---

## Next Steps

1. **Review this plan** - Does this align with your vision?
2. **Prioritize phases** - Which are most critical for your use case?
3. **Start implementation** - Begin with Phase 1 (prompt templates module)
