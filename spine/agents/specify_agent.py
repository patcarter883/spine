"""SPINE specify agent — Deep Agent for the SPECIFY phase.

When interpreter support is enabled (``SPINE_INTERPRETER=1``), the specify
agent receives a QuickJS interpreter workspace via
``CodeInterpreterMiddleware``. This enables the Recursive Language Model
(RLM) pattern: the agent writes code to inspect, chunk, and search the
project codebase rather than loading everything into the model context.

The interpreter is the **orchestration brain** — the agent writes JS to:
- Store codebase content as variables and search/filter it
- Spawn research subagents via ``tools.task(...)`` (PTC)
- Combine subagent results into a compact spec synthesis

Filesystem writes and shell commands still go through the
``LocalShellBackend`` — language-agnostic regardless of the target project
being TypeScript, PHP, Python, or anything else.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.state import WorkflowState
from spine.agents.helpers import resolve_model, debug_enabled


def build_specify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the SPECIFY phase.

    Creates a deep agent configured for specification generation. When the
    interpreter is enabled, the agent can use the QuickJS eval workspace to
    handle large codebases and orchestrate research subagents via PTC.

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
        middleware.append(build_interpreter_middleware("specify"))

    # Base system prompt — always used
    system_prompt = (
        "You are a technical specification writer. Given a work description, "
        "produce a detailed specification document.\n\n"
        f"Your workspace root is: {workspace_root}\n\n"
        "The specification should include:\n"
        "1. Overview — summary of what needs to be built\n"
        "2. Requirements — functional and non-functional requirements\n"
        "3. Architecture — high-level design decisions\n"
        "4. Interfaces — API endpoints, data models, contracts\n"
        "5. Success criteria — measurable outcomes\n\n"
        "Be specific and technical. Avoid vague language."
    )

    # Add RLM interpreter guidance when the interpreter is active
    if middleware:
        system_prompt += (

            "\n\n## Interpreter Workspace (RLM Pattern)\n\n"
            "You have a QuickJS interpreter available via the `eval` tool. "
            "Use it to handle large codebases without overflowing your context:\n\n"
            "### Inspecting the codebase\n"
            "Use filesystem tools to read files and `eval` to process them:\n"
            "```js\n"
            "// Read a file with the filesystem tool, then process it in eval\n"
            "const content = `... pasted from read_file tool ...`;\n"
            "const lines = content.split('\\n');\n"
            "const relevant = lines.filter(l => l.includes('interface'));\n"
            "console.log(relevant.join('\\n'));\n"
            "```\n\n"
            "### Orchestrating research (PTC)\n"
            "You can call `tools.task(...)` from inside eval to spawn "
            "subagents for parallel research:\n"
            "```js\n"
            "const topics = ['auth', 'api', 'data-model'];\n"
            "const reports = await Promise.all(\n"
            "  topics.map(t => tools.task({\n"
            "    description: `Research ${t} in this codebase and report findings.`,\n"
            "    subagent_type: 'general-purpose',\n"
            "  }))\n"
            ");\n"
            "reports.join('\\n\\n');\n"
            "```\n\n"
            "### Synthesis\n"
            "Combine research results in eval, then write the spec to a file "
            "using the filesystem tools."
        )

    agent = create_deep_agent(
        name="spine-specify",
        model=model,
        backend=backend,
        middleware=middleware,
        debug=debug_enabled(),
        system_prompt=system_prompt,
    )

    return agent
