"""SPINE plan synthesizer — single-tool Deep Agent for the PLAN phase.

PLAN runs as the exploration → synthesis subgraph: codebase research happens
UPSTREAM (the exploration subgraph's parallel researchers), and this synthesizer
turns the inlined specification + research findings into the structured plan via
a SINGLE forced tool, ``write_structured_plan`` (a flat array of feature_slices
with explicit dependencies).

Tool surface (complete list): ``write_structured_plan`` — nothing else.

There is deliberately no ``search_codebase`` and no ``read_prior_artifacts``.
The legacy 3-tool plan agent (search_codebase + read_prior_artifacts + write)
was removed: a finite-window local model spiralled on it, re-researching what
exploration already owns (~62 ``search_codebase`` calls, never converging).
With a single tool, ``tool_choice="any"`` is a structural pin — the only legal
move is the write, so the research spiral is impossible by construction. Do NOT
reintroduce a read/search tool here.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import artifact_path
from spine.agents.factory import build_phase_agent
from spine.agents.plan_tools import StructuredWritePlanTool
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def build_plan_synthesizer(
    state: WorkflowState,
    config: RunnableConfig | None = None,
    *,
    completion_cap_override: int | None = None,
) -> Any:
    """Build the synthesize-only Deep Agent for the PLAN phase.

    Tool surface is exactly ONE tool: ``write_structured_plan``. The
    specification and research findings are inlined into the prompt by
    ``_synthesize_plan``, so there is nothing to read first.

    ``completion_cap_override`` replaces the default synthesis completion
    clamp — used by the length-aware corrective retry to rebuild the agent
    with a raised cap after the plan truncated at the base clamp (trace
    019eb940: identical retries truncated identically at 8K).

    ``read_prior_artifacts`` was removed after trace 019eb52c: on the
    exploration path ``state["artifacts"]`` is never populated, so the
    tool always answered "No prior artifacts found" while the system
    prompt promised it would load the specification — the forced-tool
    loop then re-called it 23× chasing a spec that never came (the named
    tool_choice pin that would have broken the loop is unsupported on
    local providers). With a single tool, ``tool_choice="any"`` IS a pin:
    the loop is structurally impossible on every provider.
    """
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "")
    plan_dir = artifact_path(work_id, PhaseName.PLAN.value)

    synthesizer_tools = [
        StructuredWritePlanTool(
            workspace_root=workspace_root,
            plan_dir=plan_dir,
        ),
    ]

    from spine.agents.synthesis_budget import synthesis_completion_cap
    from spine.agents.tool_forcing import ForceToolUntilCalledMiddleware

    return build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.PLAN,
        system_prompt=_build_plan_synthesizer_prompt(),
        extra_tools=synthesizer_tools,
        skip_filesystem_middleware=True,
        # The plan JSON is 2-5K tokens; without a clamp the request inherits
        # the global max_completion_tokens (30K) and a finite-window model
        # 400s once prompt + completion budget exceed the window (019eb3dd).
        completion_token_cap=(
            completion_cap_override
            if completion_cap_override and completion_cap_override > 0
            else synthesis_completion_cap(PhaseName.PLAN.value)
        ),
        # Forcing with a single-tool surface: the model cannot end a turn
        # in prose/reasoning (trace 019eb412) and cannot stall on a read
        # tool (trace 019eb52c) — the only legal move is the write, and
        # forcing releases when it succeeds.
        extra_middleware=[
            ForceToolUntilCalledMiddleware(final_tool="write_structured_plan")
        ],
    )


def _build_plan_synthesizer_prompt() -> str:
    return (
        "You are the PLAN phase synthesizer. Codebase exploration was completed "
        "BEFORE you started — the findings are injected into your prompt below. "
        "Your job is to synthesize spec + findings into a structured plan and call "
        "`write_structured_plan` ONCE. Do NOT re-explore the codebase.\n\n"
        "## Proportionality: minimal slice COUNT, complete COVERAGE (read FIRST)\n"
        "The plan MUST be proportionate to the objective AND cover every "
        "specification requirement. These are two separate axes — do not trade "
        "one for the other:\n"
        "- **Minimal slice COUNT.** Prefer extending existing patterns over "
        "creating new modules/files. If the findings show an analogous mechanism "
        "already exists (e.g. an existing CLI flag the new one can mirror), "
        "follow it. Do NOT introduce a new module, file, or abstraction — or an "
        "extra slice — unless the specification explicitly requires it. A narrow "
        "objective (a single flag, a small behaviour tweak, a bugfix — short "
        "description, no 'design'/'refactor'/'rebuild'/'architect' verbs) is "
        "usually ONE slice touching existing files.\n"
        "- **Complete COVERAGE — minimal does NOT mean dropping requirements.** "
        "Fewer slices means FOLDING work into the existing slice, never omitting "
        "it. The single slice must still satisfy EVERY acceptance criterion in "
        "the specification. In particular, if the spec asks for test file "
        "targeting / verifiable acceptance criteria, the slice MUST list the "
        "test file(s) in its `target_files` and encode the spec's acceptance "
        "criteria as concrete, testable items in `acceptance_criteria` — inside "
        "the one slice, not as a second slice and not left out. A minimal plan "
        "that silently drops a spec requirement (e.g. tests) will be rejected "
        "just as surely as an over-scoped one.\n"
        "- Every slice must trace to a specification requirement, and every "
        "specification requirement must be covered by some slice.\n\n"
        "Before you call the tool, self-check: (1) is the slice COUNT as small "
        "as the objective allows, and (2) does the union of all slices cover "
        "EVERY spec acceptance criterion, including tests? Both must be yes.\n\n"
        "## Available tools (the ONLY tool you have)\n"
        "- `write_structured_plan` — writes both plan.md and plan.json.\n\n"
        "## Workflow (exactly 1 call)\n\n"
        "Everything you need is ALREADY in your prompt: the specification "
        "(<specification> block), the research findings (<findings> block), "
        "and the objective. There is nothing to read or load first. "
        "Synthesize spec + findings into structured fields and call "
        "`write_structured_plan` ONCE. The tool renders markdown and emits JSON for you — "
        "DO NOT author markdown, DO NOT hand-serialize JSON, DO NOT call write_file.\n\n"
        "Top-level fields:\n"
        "- architecture_overview: prose paragraph (string)\n"
        "- technology_choices: list of short strings, one item per choice\n"
        "- feature_slices: list of structured slice objects (see below). REQUIRED — at least one.\n"
        "- testing_strategy: prose paragraph (string)\n"
        "- risks: list of short strings, one item per risk\n"
        "- codebase_map: optional prose paragraph (string)\n\n"
        "Each feature_slices item:\n"
        "- id: unique short identifier (e.g., 'add-user-model')\n"
        "- title: human-readable one-line summary\n"
        "- target_files: list of every file path to create/modify\n"
        "- execution_requirements: detailed implementation instructions (string)\n"
        "- dependencies: list of slice IDs that must complete first. Every ID here "
        "MUST be the id of another slice in THIS plan — never reference a slice you did not define.\n"
        "- acceptance_criteria: list of concrete test/verification steps\n"
        "- complexity: 'small', 'medium', or 'large'\n"
    )
