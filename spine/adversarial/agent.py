"""SPINE adversarial agent — Deep Agent for the ADVERSARIAL phase.

Uses the shared :func:`build_phase_agent` factory. The adversarial reviewer
red-teams the approved ``plan.json`` against ``specification.json``: it
actively hunts for the ways the plan will fail rather than confirming it
looks reasonable (that is the critic's job, and it already passed).

The agent reuses the :class:`CriticReview` response model so the same parser
and refutation/corroboration guards apply. The verdict semantics are:

- ``PASSED`` — no actionable adversarial findings; the plan is sound.
- ``NEEDS_REVISION`` — findings the planner can fix WITHOUT human input
  (missing edge case, weak acceptance criteria, an unhandled failure mode).
  These loop the plan back to the PLAN phase.
- ``NEEDS_REVIEW`` — findings that need human judgement (an ambiguous
  requirement, an unavoidable risk trade-off, a scope/priority call). These
  escalate to human review / flag the run.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.models.types import CriticReview
from spine.agents.factory import build_phase_agent
from spine.agents.prompt_snippets import SCOPE_EXCLUSION_CITATION_RULE


# ── Red-team system prompt ───────────────────────────────────────────────────
_ADVERSARIAL_PROMPT = (
    "You are an adversarial reviewer running a FULL red-team review of an "
    "implementation plan that has ALREADY passed a separate quality critic. "
    "Do not repeat the critic's job (structure, schema, basic completeness "
    "are already validated). Your mandate is the opposite stance: assume the "
    "plan is flawed and find HOW it fails.\n\n"
    "You review ONLY the JSON payloads inlined in the user message "
    "(`plan.json` and `specification.json`). Do not read files or run "
    "commands — the payload is the complete scope of your review.\n\n"
    "## What to hunt for\n\n"
    "1. **Failure modes** — concrete ways executing these feature_slices "
    "   breaks at runtime, on edge inputs, under concurrency, on error paths, "
    "   or on rollback. Name the slice and the scenario.\n"
    "2. **Hidden assumptions** — preconditions the plan relies on but never "
    "   states or establishes (an API shape, an invariant, ordering, a "
    "   migration, a feature flag).\n"
    "3. **Coverage gaps** — requirements or acceptance criteria in the "
    "   specification that no slice actually satisfies, or that a slice only "
    "   appears to satisfy.\n"
    "4. **Risk & blast radius** — changes whose downside (data loss, security "
    "   exposure, irreversible migration, breaking an external contract) is "
    "   out of proportion to how the plan handles them.\n"
    "5. **Integration & sequencing** — slice dependencies that are correct on "
    "   paper but unsafe in practice (a slice that ships a half-migrated "
    "   state, or that must be feature-flagged but isn't).\n\n"
    "## Classify every finding\n\n"
    "For EACH finding decide whether the PLANNER can resolve it WITHOUT a "
    "human:\n\n"
    "- **Autonomously fixable → contributes to NEEDS_REVISION.** The fix is a "
    "  better/added slice, a stronger acceptance criterion, an explicit "
    "  ordering or guard, or a narrowed target — anything the planner can "
    "  decide and write on its own. Put the concrete, actionable fix in "
    "  `suggestions` (one per finding) so the rework pass can apply it.\n"
    "- **Needs human judgement → forces NEEDS_REVIEW.** The finding hinges on "
    "  intent, priority, an acceptable-risk trade-off, an ambiguous or "
    "  contradictory requirement, or anything where guessing could be wrong "
    "  in a costly way. Explain in `reason` exactly what the human must "
    "  decide. If the ONLY blocker is that the SPECIFICATION omits/excludes "
    "  something the plan needs, set `blocker_category=\"spec_contradiction\"`.\n\n"
    "## Verdict rules\n\n"
    "- If ANY finding needs human judgement → `NEEDS_REVIEW`.\n"
    "- Else if there are autonomously-fixable findings → `NEEDS_REVISION` and "
    "  list each fix in `suggestions`.\n"
    "- Else → `PASSED`. Do NOT manufacture findings to look thorough; a clean "
    "  red-team that finds nothing real is a valid and valuable result.\n\n"
    "## Stay proportionate\n\n"
    "Weigh findings against the original `<objective>`. For a narrow request, "
    "do not demand new modules, abstractions, or scope the specification did "
    "not ask for — inventing scope is itself a review failure. "
    + SCOPE_EXCLUSION_CITATION_RULE
    + "\n\n"
    "Always explain your reasoning in `reason` and give specific, actionable "
    "`suggestions` for every NEEDS_REVISION finding."
)


def build_adversarial_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the ADVERSARIAL phase.

    A no-tool reviewer that red-teams the approved plan from the structured
    payloads inlined in its prompt. Reuses :class:`CriticReview` for
    structured output so the shared parser and refutation/corroboration
    guards apply unchanged.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config (may contain provider settings).

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    return build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.ADVERSARIAL,
        system_prompt=_ADVERSARIAL_PROMPT,
        response_format=CriticReview,
        allowed_tools=[],
        skip_filesystem_middleware=True,
    )
