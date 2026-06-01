"""SPINE skills resolver — locates skill directories for Deep Agents.

Deep Agents' ``skills`` parameter accepts a list of directory paths or file
paths.  The agent reads frontmatter from each ``SKILL.md`` at startup, then
loads the full content only when it determines the skill is relevant
(progressive disclosure).

This module provides :func:`resolve_skills` which builds the list of skill
paths appropriate for a given SPINE phase, combining:

1. **Phase-specific skills** — e.g. spec-writing for SPECIFY,
   feature-slice-decomposition for TASKS, code-review for VERIFY.
2. **Workspace skills** — if the target project has a ``.spine/skills/``
   directory, those are included too.

Skill paths are absolute so they work regardless of the agent's cwd.
"""

from __future__ import annotations

import logging
from pathlib import Path

from spine.models.enums import PhaseName

logger = logging.getLogger(__name__)

# ── Built-in skill directories (shipped with spine) ──────────────────────

_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"

# Maps phase name -> list of skill directory names (under _SKILLS_ROOT)
_PHASE_SKILLS: dict[str, list[str]] = {
    PhaseName.SPECIFY.value: ["spec-writing"],
    PhaseName.PLAN.value: [],
    PhaseName.TASKS.value: ["feature-slice-decomposition"],
    PhaseName.IMPLEMENT.value: [],
    PhaseName.VERIFY.value: ["code-review"],
    PhaseName.CRITIC.value: [],
    PhaseName.GAP_PLAN.value: [],
}

# Phases that don't need the full project AGENTS.md — it's ~22K chars
# of mostly irrelevant content (testing, config, deps, workflows).
# All phases skip AGENTS.md to save ~25K tokens total.
_SKIP_AGENTS_MD: set[str] = {
    PhaseName.SPECIFY.value,
    PhaseName.PLAN.value,
    PhaseName.TASKS.value,
    PhaseName.IMPLEMENT.value,
    PhaseName.VERIFY.value,
    PhaseName.CRITIC.value,
    PhaseName.GAP_PLAN.value,
}


def resolve_skills(
    phase: str,
    workspace_root: str | None = None,
) -> list[str]:
    """Build the list of skill paths for a SPINE phase agent.

    Args:
        phase: Phase name (e.g. "specify", "implement").
        workspace_root: Project workspace root (checked for .spine/skills/).

    Returns:
        List of absolute paths to skill directories, ready to pass to
        ``create_deep_agent(skills=[...])``.
    """
    skills: list[str] = []

    # Phase-specific built-in skills
    for skill_name in _PHASE_SKILLS.get(phase, []):
        skill_dir = _SKILLS_ROOT / skill_name
        if skill_dir.is_dir():
            skills.append(str(skill_dir))
        else:
            logger.warning("Built-in skill directory not found: %s", skill_dir)

    # Workspace-level skills (project-specific)
    if workspace_root:
        ws_skills = Path(workspace_root) / ".spine" / "skills"
        if ws_skills.is_dir():
            for child in sorted(ws_skills.iterdir()):
                if child.is_dir() and (child / "SKILL.md").exists():
                    skills.append(str(child))

    return skills


def resolve_memory(
    workspace_root: str | None = None,
    phase: str | None = None,
) -> list[str]:
    """Build the list of memory file paths for a SPINE phase agent.

    DA's ``memory`` parameter loads AGENTS.md files as "always injected"
    context.  We load:

    1. The target project's AGENTS.md (if it exists at the workspace root).
       **Skipped for phases in _SKIP_AGENTS_MD** — these phases don't need
       testing patterns, config formats, or dependency tables, and the
       full file costs ~22K chars (~5K tokens) per turn.
    2. The target project's .spine/AGENTS.md (if it exists — project-specific
       SPINE conventions).  Always loaded — this is typically small.

    Args:
        workspace_root: Project workspace root.
        phase: Phase name (e.g. "tasks", "specify"). When provided, phases
            that don't benefit from project-level AGENTS.md content skip it.

    Returns:
        List of absolute file paths, ready to pass to
        ``create_deep_agent(memory=[...])``.
    """
    if not workspace_root:
        return []

    memory_paths: list[str] = []
    root = Path(workspace_root)

    # Project root AGENTS.md
    if phase not in _SKIP_AGENTS_MD:
        agents_md = root / "AGENTS.md"
        if agents_md.exists():
            memory_paths.append(str(agents_md))

    # .spine/AGENTS.md — SPINE-specific conventions for this project
    # Always loaded — typically small and highly relevant.
    spine_agents = root / ".spine" / "AGENTS.md"
    if spine_agents.exists():
        memory_paths.append(str(spine_agents))

    return memory_paths


# ── Onboarding documents ─────────────────────────────────────────────────

# Per-phase PRIMARY onboarding document — the one injected in full into the
# system prompt (hybrid injection). The remaining documents are referenced by
# path for on-demand reading. The mapping reflects which document each phase
# most needs:
#   specify           — what to build      → PROJECT_DEFINITION
#   plan / tasks       — where it goes      → ARCHITECTURE_MAP
#   implement / verify — how to write it    → CODING_GUIDELINES
#   critic             — assist boundaries  → SPINE_ASSISTANCE_REQUIREMENTS
#   gap_plan           — structure          → ARCHITECTURE_MAP
_PHASE_PRIMARY_DOC: dict[str, str] = {
    PhaseName.SPECIFY.value: "PROJECT_DEFINITION.md",
    PhaseName.PLAN.value: "ARCHITECTURE_MAP.md",
    PhaseName.TASKS.value: "ARCHITECTURE_MAP.md",
    PhaseName.IMPLEMENT.value: "CODING_GUIDELINES.md",
    PhaseName.VERIFY.value: "CODING_GUIDELINES.md",
    PhaseName.CRITIC.value: "SPINE_ASSISTANCE_REQUIREMENTS.md",
    PhaseName.GAP_PLAN.value: "ARCHITECTURE_MAP.md",
}

# One-line purpose per document, shown in the reference block so an agent knows
# which file to read on demand.
_DOC_PURPOSE: dict[str, str] = {
    "PROJECT_DEFINITION.md": "what the project is and does — purpose, domains",
    "ARCHITECTURE_MAP.md": "module responsibilities, execution paths, dependencies",
    "CODING_GUIDELINES.md": "typing, error-handling, testing, and naming conventions",
    "SPINE_ASSISTANCE_REQUIREMENTS.md": "where to assist + context/token guardrails",
}

# Size cap (bytes) for the injected primary document. Above this it is demoted to
# reference-only so the always-on injection can never blow the bounded token
# budget that project AGENTS.md is itself skipped to protect (see
# ``_SKIP_AGENTS_MD``).
_ONBOARDING_INJECT_BYTE_CAP = 12_000


def resolve_onboarding_docs(
    workspace_root: str | None,
    phase: str,
) -> tuple[str | None, list[tuple[str, str]]]:
    """Resolve onboarding context for a phase agent (hybrid injection).

    Onboarding writes four markdown documents to a single stable location,
    ``<workspace_root>/.spine/onboarding/`` (see
    :func:`spine.work.onboarding.synthesis_tools.onboarding_docs_dir`). This
    selects the phase's PRIMARY document to inject in full and lists the rest as
    references the agent may read on demand.

    Args:
        workspace_root: Project workspace root (holds ``.spine/onboarding/``).
        phase: Phase name (e.g. ``"specify"``).

    Returns:
        ``(inject_path, reference)`` where ``inject_path`` is the absolute path to
        the primary document to inject in full — or ``None`` when it is absent or
        exceeds :data:`_ONBOARDING_INJECT_BYTE_CAP` (demoted to reference-only) —
        and ``reference`` is a list of ``(filename, abs_path)`` for the other
        existing onboarding documents.
    """
    from spine.work.onboarding.synthesis_tools import (
        ONBOARDING_DOC_NAMES,
        onboarding_docs_dir,
    )

    if not workspace_root:
        return None, []

    docs_dir = onboarding_docs_dir(workspace_root)
    if not docs_dir.is_dir():
        return None, []

    # Choose the primary doc to inject, subject to the size guard.
    inject_path: str | None = None
    primary = _PHASE_PRIMARY_DOC.get(phase)
    if primary:
        ppath = docs_dir / primary
        if ppath.is_file():
            try:
                within_cap = ppath.stat().st_size <= _ONBOARDING_INJECT_BYTE_CAP
            except OSError:
                within_cap = False
            if within_cap:
                inject_path = str(ppath)
            else:
                logger.debug(
                    "Phase %s: onboarding doc %s exceeds inject cap — "
                    "demoting to reference-only",
                    phase,
                    primary,
                )

    # Everything else (existing docs not injected in full) becomes a reference.
    reference: list[tuple[str, str]] = []
    for name in ONBOARDING_DOC_NAMES:
        fname = f"{name}.md"
        fpath = docs_dir / fname
        if not fpath.is_file():
            continue
        if inject_path is not None and str(fpath) == inject_path:
            continue
        reference.append((fname, str(fpath)))

    return inject_path, reference


def build_onboarding_reference(reference: list[tuple[str, str]]) -> str:
    """Build the ``<onboarding_documentation>`` reference block.

    Lists each referenced onboarding document by absolute path plus a one-line
    purpose, mirroring the "files on disk — read what you need" pattern of
    :func:`spine.agents.artifacts.build_artifact_prompt`. Returns ``""`` when
    there is nothing to reference (so the block elides cleanly).
    """
    from spine.agents.prompt_format import Tag, xml_block

    if not reference:
        return ""

    lines = [
        "The project's onboarding documentation is available on disk. Read a "
        "file with your read tool only if you need it for this task:",
    ]
    for fname, path in reference:
        purpose = _DOC_PURPOSE.get(fname, "")
        suffix = f" — {purpose}" if purpose else ""
        lines.append(f"- `{path}`{suffix}")

    return xml_block(Tag.ONBOARDING_DOCS, "\n".join(lines))
