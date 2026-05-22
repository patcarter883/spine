"""SPINE Plan Resolver — decomposes approved plans into work units for execution.

This module provides the plan resolution capability that bridges the planning
workflow with the execution workflow. Given an approved plan artifact, it
produces a decomposition into individual work units that can be spawned as
separate execution work items.
"""

from __future__ import annotations

from pathlib import Path

from spine.models.enums import WorkType
from spine.models.types import PlanDecomposition, WorkUnit


def resolve_plan(plan_artifact_path: str, work_id: str) -> PlanDecomposition:
    """Resolve a plan artifact into work units for execution.

    Reads the plan markdown file and extracts structured work units from
    the Feature Slice section. Each work unit represents a discrete piece
    of work that can be spawned as a separate execution work item.

    Args:
        plan_artifact_path: Path to the planning.md artifact file.
        work_id: The parent work item ID for spawned items.

    Returns:
        A PlanDecomposition containing the extracted work units.

    Raises:
        FileNotFoundError: If the plan artifact doesn't exist.
        ValueError: If the plan artifact is empty or malformed.
    """
    path = Path(plan_artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan artifact not found: {plan_artifact_path}")

    content = path.read_text()
    if not content.strip():
        raise ValueError(f"Plan artifact is empty: {plan_artifact_path}")

    units = _extract_work_units(content, work_id)
    return PlanDecomposition(units=units)


async def resolve_plan_to_units(
    plan_content: str,
    work_type: str = "task",
    state: dict | None = None,
) -> PlanDecomposition:
    """Async wrapper for plan resolution compatible with Deep Agent patterns.

    Args:
        plan_content: The plan markdown content.
        work_type: The work type for spawned items (default: "task").
        state: Optional state dict for Deep Agent compatibility.

    Returns:
        A PlanDecomposition containing the extracted work units.
    """
    work_id = state.get("work_id", "unknown") if state else "unknown"
    units = _extract_work_units(plan_content, work_id)
    return PlanDecomposition(units=units)


def create_work_spawn_specs(
    decomposition: PlanDecomposition,
    plan_id: str,
    base_description: str = "",
) -> list[dict]:
    """Create work spawn specifications from a plan decomposition.

    Converts a PlanDecomposition into a list of dicts suitable for spawning
    execution work items.

    Args:
        decomposition: The plan decomposition containing work units.
        plan_id: The ID of the planning work item that produced this.
        base_description: Base description for spawned items.

    Returns:
        List of dicts with description, work_type, and plan_id keys.
    """
    specs: list[dict] = []

    for i, unit in enumerate(decomposition.units):
        spec = {
            "title": unit.title,
            "description": unit.description or base_description,
            "work_type": WorkType.TASK,
            "plan_id": plan_id,
            "priority": unit.priority,
        }
        specs.append(spec)

    return specs


def _extract_work_units(content: str, work_id: str) -> list[WorkUnit]:
    """Extract work units from plan content.

    Looks for sections like:
    - ## Work Units
    - ## Tasks
    - ## Feature Slices

    Each work unit should be a markdown heading (### or ####) or a list item
    under these sections.

    Args:
        content: The plan markdown content.
        work_id: Parent work ID for spawned items.

    Returns:
        List of WorkUnit objects extracted from the plan.
    """
    units: list[WorkUnit] = []
    lines = content.split("\n")

    in_work_section = False

    for line in lines:
        # Check for work section headers
        if line.startswith("## Work Units") or line.startswith("## Tasks"):
            in_work_section = True
            continue
        if (
            line.startswith("## ")
            and not line.startswith("## Work")
            and not line.startswith("## Tasks")
        ):
            in_work_section = False
            continue

        if in_work_section:
            # Look for work unit entries (### headings or - list items)
            if line.startswith("### "):
                unit_title = line[4:].strip()
                if unit_title:
                    units.append(
                        WorkUnit(
                            title=unit_title,
                            description=f"Work unit '{unit_title}' from plan {work_id}",
                            priority="medium",
                        )
                    )
            elif line.startswith("#### "):
                # Sub-work unit, treat as a unit
                unit_title = line[5:].strip()
                if unit_title:
                    units.append(
                        WorkUnit(
                            title=unit_title,
                            description=f"Sub-work unit from plan {work_id}",
                            priority="medium",
                        )
                    )
            elif line.startswith("- ") and not line.startswith("- ["):
                # List item that might be a work unit
                item = line[2:].strip()
                if item and not item.startswith("["):  # Skip checkbox items
                    units.append(
                        WorkUnit(
                            title=item,
                            description=f"Task from plan {work_id}",
                            priority="medium",
                        )
                    )

    # If no structured work units found, create a default one
    if not units:
        units.append(
            WorkUnit(
                title=f"Execute plan {work_id}",
                description=f"Implement the plan defined in work item {work_id}",
                priority="high",
            )
        )

    return units
