"""SPINE tasks agent — Deep Agent for the TASKS (decomposition) phase.

When interpreter support is enabled, the tasks agent can use the QuickJS
interpreter to orchestrate parallel slice decomposition via PTC. This is
the most impactful phase for the RLM pattern because:

1. The plan artifact can be large — store it in interpreter variables
   instead of re-reading from the model context each turn.
2. Slice decomposition is naturally parallel — ``tools.task(...)`` with
   ``Promise.all`` dispatches independent slices concurrently.
3. Dependency ordering is a structured-data problem — sorting waves,
   building adjacency lists, and tracking completion is deterministic
   work best done in code, not by prompting.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.state import WorkflowState
from spine.agents.helpers import resolve_model, debug_enabled


def build_tasks_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the TASKS phase.

    Creates a deep agent configured for decomposing plans into feature
    slices with dependency tracking. When the interpreter is enabled,
    the agent can use PTC to dispatch parallel research subagents and
    compose results in code.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    from deepagents import create_deep_agent

    from spine.agents.backend import build_backend
    from spine.agents.interpreter import build_interpreter_middleware, interpreter_enabled

    model = resolve_model(config, session_id=state.get("work_id"))
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)

    middleware: list[Any] = []
    if interpreter_enabled():
        middleware.append(build_interpreter_middleware("tasks"))

    system_prompt = (
        "You are a task decomposition specialist. Given a plan, "
        "break it into smaller, executable feature slices.\n\n"
        f"Your workspace root is: {workspace_root}\n\n"
        "For each feature slice, specify:\n"
        "1. Name and description\n"
        "2. Files to create or modify\n"
        "3. Dependencies (which slices must complete first)\n"
        "4. Acceptance criteria\n"
        "5. Estimated complexity (small/medium/large)\n\n"
        "Group slices by dependency waves — slices with no dependencies "
        "can run in parallel. Use a DAG structure to show ordering.\n\n"
        "Output the slices in a structured markdown format with clear "
        "dependency annotations."
    )

    if middleware:
        system_prompt += (

            "\n\n## Interpreter Workspace (RLM Pattern)\n\n"
            "You have a QuickJS interpreter available via the `eval` tool. "
            "Use it for structured decomposition tasks:\n\n"
            "### Storing the plan\n"
            "Read the plan with filesystem tools, then store key sections "
            "in interpreter variables for reference without re-reading:\n"
            "```js\n"
            "const plan = `... plan content ...`;\n"
            "const sections = plan.split(/^## /m).filter(Boolean);\n"
            "console.log(`Plan has ${sections.length} sections`);\n"
            "```\n\n"
            "### Parallel research (PTC)\n"
            "Dispatch independent research subagents concurrently:\n"
            "```js\n"
            "const areas = ['auth', 'api', 'database', 'testing'];\n"
            "const research = await Promise.all(\n"
            "  areas.map(area => tools.task({\n"
            "    description: `Analyze the existing ${area} implementation. `\n"
            "      + 'Report current patterns, files involved, and constraints.',\n"
            "    subagent_type: 'general-purpose',\n"
            "  }))\n"
            ");\n"
            "// Process research results in code\n"
            "const findings = research.map((r, i) => ({ area: areas[i], report: r }));\n"
            "```\n\n"
            "### Dependency sorting\n"
            "Build the dependency graph and sort into waves in code — "
            "this is deterministic work that doesn't need the model:\n"
            "```js\n"
            "const slices = [\n"
            "  { name: 'A', deps: [] },\n"
            "  { name: 'B', deps: ['A'] },\n"
            "  { name: 'C', deps: ['A'] },\n"
            "  { name: 'D', deps: ['B', 'C'] },\n"
            "];\n"
            "// Topological sort into waves\n"
            "const waves = [];\n"
            "const completed = new Set();\n"
            "let remaining = [...slices];\n"
            "while (remaining.length > 0) {\n"
            "  const wave = remaining.filter(s => s.deps.every(d => completed.has(d)));\n"
            "  if (wave.length === 0) break; // cycle\n"
            "  wave.forEach(s => completed.add(s.name));\n"
            "  waves.push(wave.map(s => s.name));\n"
            "  remaining = remaining.filter(s => !completed.has(s.name));\n"
            "}\n"
            "console.log('Waves:', JSON.stringify(waves));\n"
            "```\n\n"
            "Write the final slice definitions to a file using the "
            "filesystem tools."
        )

    agent = create_deep_agent(
        name="spine-tasks",
        model=model,
        backend=backend,
        middleware=middleware,
        debug=debug_enabled(),
        system_prompt=system_prompt,
    )

    return agent
