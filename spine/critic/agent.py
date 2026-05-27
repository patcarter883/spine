"""SPINE critic agent — Deep Agent for the CRITIC phase.

Uses the shared :func:`build_phase_agent` factory.  The critic gets a
preview of artifacts under review (short inline) with full content on
disk, rather than inlining everything.

When the reviewed phase is PLAN, the agent prompt is augmented with
instructions to validate the structured ``plan.json`` format.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.factory import build_phase_agent
from spine.workflow.critic_review import _get_reviewed_phase
from spine.models.types import CriticReview


# ── PLAN-specific review instructions appended to the system prompt ──────────
_PLAN_REVIEW_INSTRUCTIONS = (
    "## Structured Plan Validation (PLAN phase)\n\n"
    "Validate the following structure in the provided `plan.json` payload:\n\n"
    "1. **feature_slices present and non-empty** — the top-level\n"
    "   `feature_slices` array must exist and contain at least one slice.\n"
    "2. **Required slice fields** — every slice object must have:\n"
    "   - `id` — a unique string identifier\n"
    "   - `title` — a short human-readable name\n"
    "   - `target_files` — a non-empty list of file paths the slice will create\n"
    "     or modify\n"
    "   - `acceptance_criteria` — a non-empty list of verifiable conditions\n"
    "     that define when the slice is complete\n"
    "3. **Dependency integrity** — every ID listed in a slice's `dependencies`\n"
    "   array must correspond to an existing slice `id` in the same plan.\n"
    "   Flag typos, missing slices, or dangling references.\n"
    "4. **No circular dependencies** — the dependency graph must form a DAG.\n"
    "   If A depends on B and B depends on A (directly or transitively) the\n"
    "   plan is invalid.\n"
    "5. **Coverage** — the slices collectively address all requirements captured\n"
    "   in the specification. Flag any requirement that is not covered by at\n"
    "   least one slice.\n"
    "6. **Slice granularity** — each slice should be implementable in a single\n"
    "   pass (roughly one focused PR). Flag slices that look too large or\n"
    "   that bundle unrelated changes.\n\n"
    "## Scope Creep Detection\n\n"
    "The specification includes `scope_inclusions` and `scope_exclusions` lists.\n"
    "Review each slice's `target_files` and `execution_requirements` to ensure:\n\n"
    "- **Inclusions check**: Slices\n"
    "Every slice should primarily work within the areas listed in\n"
    "`scope_inclusions`. If a slice's target files or requirements go significantly\n"
    "beyond these inclusions, flag it as potential scope creep.\n\n"
    "- **Exclusions check**:\n"
    "If any slice's target files or requirements overlap with items in\n"
    "`scope_exclusions`, this is a **VIOLATION**. Flag it and suggest the slice\n"
    "be revised to exclude the forbidden areas.\n\n"
    "- **Reporting**: For each scope violation, list:\n"
    "  - The slice ID\n"
    "  - The excluded area being touched\n"
    "  - The specific target file or requirement that violates scope\n\n"
    "If any check fails, respond with NEEDS_REVISION and list the specific\n"
    "violations in your suggestions.\n\n"
)


# ── SPECIFY-specific review instructions appended to the system prompt ──────
_SPECIFY_REVIEW_INSTRUCTIONS = (
    "## Spec-to-Description Alignment (SPECIFY phase)\n\n"
    "The user message includes the original user description above the\n"
    "structured payload. Treat the description as the source of truth and\n"
    "verify that every requirement in the specification traces back to it.\n\n"
    "1. **Traceability** — every entry in the spec's requirements /\n"
    "   acceptance criteria / scope_inclusions must be derivable from the\n"
    "   original description. Requirements that introduce concepts the user\n"
    "   never mentioned are scope creep.\n"
    "2. **Proportionality** — for trivial descriptions (≤ 200 chars and no\n"
    "   verbs like 'design', 'refactor', 'rebuild', 'architect'), the spec\n"
    "   should be concise: a single feature area, no cross-cutting\n"
    "   architectural changes, no surveys of unrelated subsystems. Flag\n"
    "   specs that propose multi-module refactors for a trivial request.\n"
    "3. **Scope exclusions present** — when the user's intent is narrow,\n"
    "   the spec should explicitly list `scope_exclusions` for adjacent\n"
    "   areas the model considered but rejected. Missing exclusions on a\n"
    "   narrow description signal the critic should look harder for creep.\n\n"
    "If you find scope creep, respond with NEEDS_REVISION and set the\n"
    "`reason` field to a phrase starting with `scope_creep:` followed by\n"
    "the specific untraceable requirement(s). List each violating\n"
    "requirement separately in `suggestions`.\n\n"
)


def build_critic_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the CRITIC phase.

    Creates a deep agent configured for quality review of phase outputs.
    The critic can inspect actual files on disk when reviewing artifacts.

    When the reviewed phase is PLAN the system prompt is extended with
    structured-plan validation instructions (feature_slices format,
    dependency integrity, cycle detection, etc.) so the LLM-based tier-2
    review complements the deterministic tier-1.5 structural check.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config (may contain provider settings).

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    reviewed_phase = _get_reviewed_phase(state)

    system_prompt = (
        "You are a quality reviewer. Review the structured payload provided "
        "in the user message and determine if it meets quality standards.\n\n"
        "Your review must be based ONLY on the JSON payload inlined in the "
        "user message. Do not attempt to read files, run shell commands, or "
        "inspect anything on disk — the payload is the complete scope of "
        "your review.\n\n"
        "Evaluate:\n"
        "1. Completeness — all required elements are present\n"
        "2. Correctness — the content is technically accurate\n"
        "3. Clarity — the document is well-structured and understandable\n"
        "4. Actionability — the output can be used by the next phase\n\n"
        "Respond with one of:\n"
        "- PASSED — the phase output meets quality standards\n"
        "- NEEDS_REVISION — the output needs improvement (specify what)\n"
        "- NEEDS_REVIEW — the output requires human judgment\n\n"
        "Always explain your reasoning and provide specific suggestions "
        "for improvement when recommending revision."
    )

    # Append phase-specific validation instructions. PLAN gets structured-plan
    # rules + scope-creep detection; SPECIFY gets spec-to-description
    # traceability rules. Other phases keep the base prompt unchanged.
    if PhaseName(reviewed_phase) == PhaseName.PLAN:
        system_prompt = system_prompt + "\n\n" + _PLAN_REVIEW_INSTRUCTIONS
    elif PhaseName(reviewed_phase) == PhaseName.SPECIFY:
        system_prompt = system_prompt + "\n\n" + _SPECIFY_REVIEW_INSTRUCTIONS

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.CRITIC,
        system_prompt=system_prompt,
        response_format=CriticReview,
        allowed_tools=[],
        skip_filesystem_middleware=True,
    )

    return agent
