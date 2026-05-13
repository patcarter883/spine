"""SPINE skills resolver — locates skill directories for Deep Agents.

Deep Agents' ``skills`` parameter accepts a list of directory paths or file
paths.  The agent reads frontmatter from each ``SKILL.md`` at startup, then
loads the full content only when it determines the skill is relevant
(progressive disclosure).

This module provides :func:`resolve_skills` which builds the list of skill
paths appropriate for a given SPINE phase, combining:

1. **Always-loaded skills** — the RLM pattern skill (when interpreter is
   available), which applies to any phase that has the eval tool.
2. **Phase-specific skills** — e.g. spec-writing for SPECIFY,
   feature-slice-decomposition for TASKS, code-review for VERIFY.
3. **Workspace skills** — if the target project has a ``.spine/skills/``
   directory, those are included too.

Skill paths are absolute so they work regardless of the agent's cwd.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
}

# RLM pattern skill — included for all phases that have interpreter support
_RLM_SKILL = "rlm-pattern"


def resolve_skills(
    phase: str,
    workspace_root: str | None = None,
    include_rlm: bool = False,
) -> list[str]:
    """Build the list of skill paths for a SPINE phase agent.

    Args:
        phase: Phase name (e.g. "specify", "implement").
        workspace_root: Project workspace root (checked for .spine/skills/).
        include_rlm: Whether to include the RLM pattern skill
            (set True when interpreter is enabled).

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

    # RLM pattern skill (when interpreter is available)
    if include_rlm:
        rlm_dir = _SKILLS_ROOT / _RLM_SKILL
        if rlm_dir.is_dir():
            skills.append(str(rlm_dir))

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
) -> list[str]:
    """Build the list of memory file paths for a SPINE phase agent.

    DA's ``memory`` parameter loads AGENTS.md files as "always injected"
    context.  We load:

    1. The target project's AGENTS.md (if it exists at the workspace root).
    2. The target project's .spine/AGENTS.md (if it exists — project-specific
       SPINE conventions).

    Args:
        workspace_root: Project workspace root.

    Returns:
        List of absolute file paths, ready to pass to
        ``create_deep_agent(memory=[...])``.
    """
    if not workspace_root:
        return []

    memory_paths: list[str] = []
    root = Path(workspace_root)

    # Project root AGENTS.md
    agents_md = root / "AGENTS.md"
    if agents_md.exists():
        memory_paths.append(str(agents_md))

    # .spine/AGENTS.md — SPINE-specific conventions for this project
    spine_agents = root / ".spine" / "AGENTS.md"
    if spine_agents.exists():
        memory_paths.append(str(spine_agents))

    return memory_paths
