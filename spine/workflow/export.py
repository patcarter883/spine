"""SPINE export — gathers work item data for external analysis.

Exports specification, plan, research data, and prompts to a
structured markdown document suitable for external review and
analysis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from spine.config import SpineConfig
from spine.persistence.artifacts import ArtifactStore

logger = logging.getLogger(__name__)

_EXPLORATION_PHASES = ("specify", "plan")


def export_work_item(work_id: str, config: SpineConfig | None = None) -> dict[str, Any]:
    """Gather all export data for a work item.

    Args:
        work_id: The work item ID to export.
        config: Optional SpineConfig (loads default if None).

    Returns:
        A dict with keys: ``work_id``, ``description``, ``work_type``,
        ``status``, ``phases`` (dict of phase -> artifacts/research/prompt),
        and ``error`` (set on failure).
    """
    if config is None:
        config = SpineConfig.load()

    try:
        from spine.work.dispatcher import get_work_status
    except ImportError:
        return {"error": "Could not import dispatcher"}

    entry = get_work_status(work_id, config)
    if entry is None:
        return {"error": f"Work item '{work_id}' not found"}

    store = ArtifactStore(base_path=config.artifact_path)
    workspace_root = config.workspace_root

    result: dict[str, Any] = {
        "work_id": work_id,
        "description": entry.get("description", ""),
        "work_type": entry.get("work_type", ""),
        "status": entry.get("status", ""),
        "created_at": entry.get("created_at", ""),
        "phases": {},
    }

    for phase in _EXPLORATION_PHASES:
        phase_data = _export_phase_data(work_id, phase, store, workspace_root)
        if phase_data:
            result["phases"][phase] = phase_data

    return result


def _export_phase_data(
    work_id: str,
    phase: str,
    store: ArtifactStore,
    workspace_root: str,
) -> dict[str, Any] | None:
    """Export data for a single phase.

    Returns None if no artifacts or research data exist for the phase.
    """
    phase_data: dict[str, Any] = {}

    research_log = _load_research_log(work_id, phase, store)
    if research_log:
        phase_data["research"] = research_log

    artifacts = _load_phase_artifacts(work_id, phase, store, workspace_root)
    if artifacts:
        phase_data["artifacts"] = artifacts

    prompt = _build_phase_prompt(phase)
    if prompt:
        phase_data["prompt"] = prompt

    if not phase_data:
        return None
    return phase_data


def _load_research_log(
    work_id: str,
    phase: str,
    store: ArtifactStore,
) -> dict[str, Any] | None:
    """Load the research_log.json artifact if it exists."""
    raw = store.load_artifact(work_id, phase, "research_log.json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[%s] research_log.json is malformed", work_id)
        return None


def _load_phase_artifacts(
    work_id: str,
    phase: str,
    store: ArtifactStore,
    workspace_root: str,
) -> dict[str, str | None]:
    """Load key artifacts for a phase from disk.

    Loads the markdown and json artifacts if present. Returns the full
    content (not truncated previews).
    """
    artifacts: dict[str, str | None] = {}
    artifact_names = _get_phase_artifact_names(phase)

    for name in artifact_names:
        content = store.load_artifact(work_id, phase, name)
        if content:
            artifacts[name] = content

    return artifacts


def _get_phase_artifact_names(phase: str) -> tuple[str, ...]:
    """Return the artifact filenames to load for a phase."""
    if phase == "specify":
        return ("specification.md", "specification.json")
    if phase == "plan":
        return ("plan.md", "plan.json")
    return ()


def _build_phase_prompt(phase: str) -> str | None:
    """Return the agent system prompt for a phase.

    Uses the same prompt builder functions that the phase agents use,
    ensuring the exported prompt matches what was sent to the LLM.
    """
    try:
        if phase == "specify":
            from spine.agents.specify_agent import _build_specify_prompt

            return _build_specify_prompt()
        if phase == "plan":
            from spine.agents.plan_agent import _build_plan_prompt

            return _build_plan_prompt()
    except ImportError as exc:
        logger.warning("Could not build prompt for phase '%s': %s", phase, exc)
    return None


def format_export_markdown(data: dict[str, Any]) -> str:
    """Format export data as a markdown document.

    Args:
        data: The dict returned by :func:`export_work_item`.

    Returns:
        A markdown-formatted string.
    """
    if "error" in data:
        return f"# Export Error\n\n{data['error']}\n"

    lines: list[str] = []
    lines.append(f"# Work Item: {data['work_id']}\n")
    lines.append(f"**Status:** {data.get('status', 'N/A')}")
    lines.append(f"**Type:** {data.get('work_type', 'N/A')}")
    lines.append(f"**Created:** {data.get('created_at', 'N/A')}")
    lines.append(f"\n## Description\n\n{data.get('description', '')}\n")

    for phase_key in ("specify", "plan"):
        phase_data: dict[str, Any] = data.get("phases", {}).get(phase_key, {})
        if not phase_data:
            continue

        lines.append(f"## {phase_key.upper()} Phase\n")

        prompt = phase_data.get("prompt")
        if prompt:
            lines.append("### Prompt\n")
            lines.append("```")
            lines.append(prompt)
            lines.append("```\n")

        research = phase_data.get("research")
        if research:
            lines.append("### Research\n")
            topics = research.get("topics", [])
            findings = research.get("findings", [])
            if topics:
                lines.append("#### Topics\n")
                for topic in topics:
                    lines.append(f"- {topic}")
                lines.append("")
            if findings:
                # Drop error-sentinel findings: they carry raw exception
                # text like "Research failed for topic ...: GraphRecursionError"
                # in their `summary` field and must never reach a human-facing
                # export. The salvage path in spine/agents/exploration_agents.py
                # sets `error=True` on these sentinels for exactly this filter.
                real_findings = [
                    f for f in findings
                    if isinstance(f, dict) and not f.get("error")
                ]
                if real_findings:
                    lines.append("#### Findings\n")
                    for i, f in enumerate(real_findings, 1):
                        lines.append(f"**Finding {i}**")
                        lines.append(f"- Topic: {f.get('topic', '')}")
                        lines.append(f"- Summary: {f.get('summary', '')}")
                        patterns = f.get("patterns", [])
                        if patterns:
                            lines.append(f"- Patterns: {', '.join(patterns)}")
                        file_map = f.get("file_map", {})
                        if file_map:
                            lines.append(f"- File Map: {json.dumps(file_map)}")
                        dependencies = f.get("dependencies", [])
                        if dependencies:
                            lines.append(f"- Dependencies: {', '.join(dependencies)}")
                        lines.append("")
                    lines.append("")

        artifacts = phase_data.get("artifacts", {})
        if artifacts:
            lines.append("### Artifacts\n")
            for name, content in sorted(artifacts.items()):
                lines.append(f"#### {name}\n")
                if content:
                    if name.endswith(".json"):
                        lines.append("```json")
                        try:
                            parsed = json.loads(content)
                            lines.append(json.dumps(parsed, indent=2))
                        except json.JSONDecodeError:
                            lines.append(content)
                        lines.append("```")
                    else:
                        lines.append(content)
                else:
                    lines.append("*(not available)*")
                lines.append("")

    return "\n".join(lines)
