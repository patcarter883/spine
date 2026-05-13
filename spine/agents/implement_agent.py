"""SPINE implement agent — Deep Agent for the IMPLEMENT phase.

This is the highest-impact phase for the RLM interpreter pattern. The
IMPLEMENT phase receives FeatureSlices from the TASKS phase and must
execute them, respecting the dependency DAG. With the interpreter + PTC:

1. The agent stores the full slice list and dependency graph in interpreter
   variables — no need to re-read the plan from the model context.
2. Independent slices within a wave execute in parallel via
   ``await Promise.all(slices.map(s => tools.task(...)))``.
3. Failed slices get try/catch handling — the agent can retry, skip, or
   flag without consuming model context on each attempt.
4. Result aggregation (which slices passed, failed, need rework) happens
   deterministically in code, producing a compact summary for the model.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.state import WorkflowState
from spine.agents.helpers import resolve_model, debug_enabled


def build_implement_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the IMPLEMENT phase.

    Creates a deep agent configured for code generation. When the
    interpreter is enabled, the agent can orchestrate parallel slice
    implementation via PTC with proper error handling.

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
        middleware.append(build_interpreter_middleware("implement"))

    system_prompt = (
        "You are an implementation engineer. Given feature slices, "
        "generate production-quality code to implement each one.\n\n"
        "Guidelines:\n"
        "1. Write clean, well-documented code\n"
        "2. Follow the project's existing coding conventions and patterns\n"
        "3. Include appropriate type annotations and docstrings\n"
        "4. Handle errors gracefully\n"
        "5. Write code that is testable\n"
        "6. For independent slices, use the task tool to delegate "
        "to subagents for parallel execution\n\n"
        f"Your workspace root is: {workspace_root}\n"
        "All file paths should be relative to this root directory. "
        "Use the write_file tool to create files on disk.\n\n"
        "After implementing all slices, provide a summary of what "
        "was created and any decisions made during implementation."
    )

    if middleware:
        system_prompt += (

            "\n\n## Interpreter Workspace (RLM Pattern)\n\n"
            "You have a QuickJS interpreter available via the `eval` tool. "
            "Use it to orchestrate parallel implementation:\n\n"
            "### Loading the slice list\n"
            "Read the task definitions with filesystem tools, then store "
            "them in interpreter variables:\n"
            "```js\n"
            "const slices = [\n"
            "  { name: 'auth-middleware', deps: [], files: ['src/auth.ts'] },\n"
            "  { name: 'user-routes', deps: ['auth-middleware'], files: ['src/routes/users.ts'] },\n"
            "  // ... more slices\n"
            "];\n"
            "```\n\n"
            "### Parallel execution by wave\n"
            "Execute independent slices concurrently with PTC:\n"
            "```js\n"
            "// Wave 1: no dependencies\n"
            "const wave1 = slices.filter(s => s.deps.length === 0);\n"
            "const results1 = await Promise.all(\n"
            "  wave1.map(slice => tools.task({\n"
            "    description: `Implement the ${slice.name} feature slice. `\n"
            "      + `Files to modify: ${slice.files.join(', ')}. `\n"
            "      + 'Follow the spec and project conventions.',\n"
            "    subagent_type: 'general-purpose',\n"
            "  }))\n"
            ");\n"
            "```\n\n"
            "### Error handling\n"
            "Handle failures without consuming model context:\n"
            "```js\n"
            "const outcomes = await Promise.allSettled(\n"
            "  wave.map(slice => tools.task({\n"
            "    description: `Implement ${slice.name}`,\n"
            "    subagent_type: 'general-purpose',\n"
            "  }))\n"
            ");\n"
            "const succeeded = outcomes\n"
            "  .filter((r, i) => r.status === 'fulfilled')\n"
            "  .map((r, i) => ({ name: wave[i].name, result: r.value }));\n"
            "const failed = outcomes\n"
            "  .filter(r => r.status === 'rejected')\n"
            "  .map((r, i) => ({ name: wave[i].name, error: r.reason }));\n"
            "console.log(`Completed: ${succeeded.length}, Failed: ${failed.length}`);\n"
            "```\n\n"
            "### Progress tracking\n"
            "Build a completion map across waves:\n"
            "```js\n"
            "const completed = new Set(succeeded.map(s => s.name));\n"
            "// ... next wave filters by completed set\n"
            "const nextWave = slices.filter(\n"
            "  s => !completed.has(s.name) && s.deps.every(d => completed.has(d))\n"
            ");\n"
            "```\n\n"
            "Write all implementation files to disk using the filesystem "
            "tools (not the interpreter — it has no filesystem access)."
        )

    agent = create_deep_agent(
        name="spine-implement",
        model=model,
        backend=backend,
        middleware=middleware,
        debug=debug_enabled(),
        system_prompt=system_prompt,
    )

    return agent
