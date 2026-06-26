"""SPINE plan agent — Deep Agent for the PLAN phase.

Reads the specification artifact and codebase structure (pre-explored by the
exploration subgraph) then writes the technical plan via the
``write_structured_plan`` tool, emitting a flat array of feature_slices with
explicit dependencies.

Tool surface (complete list):
- ``read_prior_artifacts`` — loads specification + context in one call
- ``search_codebase`` — multi-query codebase file search
- ``write_structured_plan`` — structured write with feature_slices (only)

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The plan agent has targeted read access via
``read_prior_artifacts`` + ``search_codebase``, and write access only
through ``write_structured_plan``. It cannot browse the filesystem
arbitrarily.

PLAN research strategy: codebase exploration is handled UPSTREAM by the
exploration subgraph (LangGraph Send API) before this agent runs.  The
agent synthesises those findings to produce the plan.  ``search_codebase``
is supplemental, used only for narrow targeted lookups that were not
covered by the upstream exploration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import artifact_path
from spine.agents.factory import build_phase_agent
from spine.agents.helpers import escalation_level_for_phase
from spine.agents.plan_tools import (
    StructuredWritePlanTool,
    build_plan_agent_tools,
)
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def build_plan_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the PLAN phase.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")
    workspace_root = state.get("workspace_root", ".")
    work_type = state.get("work_type", "")
    description = state.get("description", "")
    feedback_raw = state.get("feedback", [])
    feedback = [str(f) for f in feedback_raw] if feedback_raw else []

    # Build prior artifact dir mapping from state
    prior_phase_dirs = _resolve_prior_phase_dirs(state, work_id)

    agent_tools = build_plan_agent_tools(
        workspace_root=workspace_root,
        work_id=work_id,
        description=description,
        work_type=work_type,
        prior_phase_dirs=prior_phase_dirs,
        feedback=feedback,
    )

    system_prompt = _build_plan_prompt()

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.PLAN,
        system_prompt=system_prompt,
        extra_tools=agent_tools,
        skip_filesystem_middleware=True,
        # Escalate the model on critic-driven rework (no-op without a ladder).
        escalation_level=escalation_level_for_phase(state, PhaseName.PLAN),
    )

    return agent


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
    from spine.config import SpineConfig

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
            else synthesis_completion_cap(
                PhaseName.PLAN.value,
                phase_cap=SpineConfig.load().plan_synthesize_max_completion_tokens,
            )
        ),
        # Forcing with a single-tool surface: the model cannot end a turn
        # in prose/reasoning (trace 019eb412) and cannot stall on a read
        # tool (trace 019eb52c) — the only legal move is the write, and
        # forcing releases when it succeeds.
        extra_middleware=[
            ForceToolUntilCalledMiddleware(final_tool="write_structured_plan")
        ],
        # Escalate the model on critic-driven rework (no-op without a ladder).
        escalation_level=escalation_level_for_phase(state, PhaseName.PLAN),
    )


def _resolve_prior_phase_dirs(
    state: WorkflowState,
    work_id: str,
) -> dict[str, str]:
    """Map phase names to their artifact directories for phases with artifacts.

    Disk is the source of truth. ``state["artifacts"]`` is an unreliable carrier:
    subgraph state schemas that don't declare the channel silently drop it (e.g.
    ``PlanSubgraphState``), and the standalone plan path never populates it — so
    ``read_prior_artifacts`` answered "No prior artifacts found" while the spec
    sat on disk, confusing the model it told to call that tool first. We
    therefore discover prior-phase dirs by scanning the work item's artifact tree
    and union them with any state-declared phases. The current phase is excluded
    so a re-run that already wrote its own dir isn't treated as prior.
    """
    dirs: dict[str, str] = {}

    # State-declared phases (unchanged behaviour for callers that populate it).
    artifacts = state.get("artifacts", {}) or {}
    for phase, phase_artifacts in artifacts.items():
        if phase_artifacts and isinstance(phase_artifacts, dict):
            dirs[phase] = artifact_path(work_id, phase)

    # Disk-discovered phases (the fix): any phase subdir holding a real
    # (non-``.meta.json``) artifact file — found even when state never carried it.
    current_phase = state.get("phase", "")
    workspace_root = Path(state.get("workspace_root", ".") or ".")
    base = workspace_root / Path(artifact_path(work_id, ""))
    if base.is_dir():
        for phase_dir in sorted(base.iterdir()):
            phase = phase_dir.name
            if not phase_dir.is_dir() or phase == current_phase or phase in dirs:
                continue
            if any(
                f.is_file() and not f.name.endswith(".meta.json")
                for f in phase_dir.iterdir()
            ):
                dirs[phase] = artifact_path(work_id, phase)
    return dirs


def _build_plan_prompt() -> str:
    return (
        "You are the PLAN phase agent. Create a technical plan from the "
        "specification, grounded in codebase structure. Output: flat array of "
        "feature_slices with dependencies.\\n\\n"
        "## Available tools (use only these)\\n"
        "- `read_prior_artifacts` — loads spec and all prior artifacts. Call FIRST.\\n"
        "- `search_codebase` — targeted file search for narrow lookups not in exploration.\\n"
        "- `write_structured_plan` — emits feature_slices. Call LAST.\\n\\n"
        "## WORKFLOW (2 steps, ~3 turns)\\n\\n"
        "### Step 1 — Call read_prior_artifacts (Turn 1)\\n"
        "Call with no arguments to load the specification and prior exploration results.\\n\\n"
        "### Step 2 — Review exploration results (already available)\\n"
        "Codebase research was run BEFORE this agent started via the LangGraph "
        "exploration subgraph (Send API parallel dispatch). The research findings are "
        "injected into your context alongside the specification. Synthesise them — do "
        "NOT dispatch additional researcher subagents. Use `search_codebase` only for "
        "narrow targeted lookups (specific file paths, symbol names) not already covered.\\n\\n"
        "### Step 3 — Call write_structured_plan (Turn 2-3)\\n"
        "Synthesize spec + exploration into the structured fields below and call "
        "write_structured_plan ONCE. The tool renders markdown and emits JSON — do "
        "NOT author markdown.\\n\\n"
        "Top-level fields:\\n"
        "- architecture_overview: prose paragraph (string) on how the components fit together\\n"
        "- technology_choices: list of short strings, one item per choice (rationale inline)\\n"
        "- feature_slices: list of structured slice objects (see below). REQUIRED — at least one.\\n"
        "- testing_strategy: prose paragraph (string)\\n"
        "- risks: list of short strings, one item per risk\\n"
        "- codebase_map: optional prose paragraph (string)\\n\\n"
        "Each feature_slices item is a structured object with these fields:\\n"
        "- id: unique short identifier (e.g., 'add-user-model')\\n"
        "- title: human-readable one-line summary\\n"
        "- target_files: list of every file path to create/modify\\n"
        "- execution_requirements: detailed implementation instructions (string)\\n"
        "- reference_symbols: qualified names of EXISTING symbols the slice's "
        "code will call/extend/mimic (e.g. 'UIApi.update_mcp_server', "
        "'SpineConfig') — take these from your codebase research so the "
        "implementer reads exactly those instead of surveying files\\n"
        "- dependencies: list of slice IDs that must complete first\\n"
        "- acceptance_criteria: list of concrete test/verification steps\\n"
        "- complexity: 'small', 'medium', or 'large'\\n\\n"
        "## Rework handling\\n"
        "If feedback exists in prior artifacts, address every item before calling "
        "`write_structured_plan`.\\n\\n"
        "## Rules\\n"
        "- Call `read_prior_artifacts` first.\\n"
        "- Every target_file path must come from `search_codebase`, the "
        "exploration findings in your context, or a confirmed existing directory.\\n"
        "- Call `write_structured_plan` exactly once with all required fields.\\n"
        "- If not called by turn 4, write with current information.\\n"
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
