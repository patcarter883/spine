"""SPINE verify agent — Deep Agent for the VERIFY phase.

When the interpreter is enabled, the verify agent can run parallel
verification across multiple feature slices — dispatching subagents
to check each slice independently and then aggregating results in code.
This is especially valuable for large projects where sequential
verification of every slice would be slow and token-expensive.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.state import WorkflowState
from spine.agents.helpers import resolve_model, debug_enabled


def build_verify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the VERIFY phase.

    Creates a deep agent configured for verification — reviewing
    implementation against specifications, plans, and tasks. When the
    interpreter is enabled, the agent can dispatch parallel verification
    subagents and aggregate results deterministically.

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
        middleware.append(build_interpreter_middleware("verify"))

    system_prompt = (
        "You are a verification engineer. Review the implementation "
        "against the specification, plan, and feature slices.\n\n"
        f"Your workspace root is: {workspace_root}\n\n"
        "IMPORTANT: You have filesystem and shell tools available. Use them!\n"
        "1. Use read_file and ls to inspect the actual implemented files\n"
        "2. Use execute to run tests and check for errors\n"
        "3. Verify that files mentioned in the implementation actually exist\n\n"
        "Check:\n"
        "1. All feature slices are implemented\n"
        "2. The implementation follows the plan's architecture\n"
        "3. Success criteria from the specification are met\n"
        "4. Code quality is acceptable (no obvious bugs)\n"
        "5. Error handling is in place\n\n"
        "Produce a verification report with:\n"
        "- VERIFIED or NOT VERIFIED status\n"
        "- Checklist of each feature slice and its status\n"
        "- Any gaps or issues found\n"
        "- Recommendations for improvement\n\n"
        "End your report with a clear VERIFIED or NOT VERIFIED verdict."
    )

    if middleware:
        system_prompt += (

            "\n\n## Interpreter Workspace (RLM Pattern)\n\n"
            "You have a QuickJS interpreter available via the `eval` tool. "
            "Use it for parallel verification:\n\n"
            "### Multi-slice verification\n"
            "Dispatch verification subagents in parallel per slice:\n"
            "```js\n"
            "const sliceNames = ['auth', 'api', 'database'];\n"
            "const results = await Promise.all(\n"
            "  sliceNames.map(name => tools.task({\n"
            "    description: `Verify the ${name} feature slice. `\n"
            "      + 'Check that all planned files exist, tests pass, '\n"
            "      + 'and success criteria are met. '\n"
            "      + 'Report VERIFIED or NOT VERIFIED with reasons.',\n"
            "    subagent_type: 'general-purpose',\n"
            "  }))\n"
            ");\n"
            "```\n\n"
            "### Result aggregation\n"
            "Process verification results in code:\n"
            "```js\n"
            "const report = results.map((r, i) => ({\n"
            "  slice: sliceNames[i],\n"
            "  passed: r.includes('VERIFIED') && !r.includes('NOT VERIFIED'),\n"
            "}));\n"
            "const allPassed = report.every(r => r.passed);\n"
            "console.log(allPassed ? 'ALL VERIFIED' : 'ISSUES FOUND');\n"
            "report.forEach(r => console.log(`  ${r.slice}: ${r.passed ? 'OK' : 'FAIL'}`));\n"
            "```\n\n"
            "Run the actual test suite and linters via the shell backend "
            "(execute tool), not through the interpreter."
        )

    agent = create_deep_agent(
        name="spine-verify",
        model=model,
        backend=backend,
        middleware=middleware,
        debug=debug_enabled(),
        system_prompt=system_prompt,
    )

    return agent
