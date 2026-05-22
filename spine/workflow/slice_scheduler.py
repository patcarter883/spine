"""SPINE slice scheduler — topological sorting for feature slice execution.

Groups feature slices into parallel execution waves using Python's
``graphlib.TopologicalSorter``. Each wave contains slices whose
dependencies have all been satisfied by prior waves, enabling maximum
parallelism during the IMPLEMENT phase.

Public API:
    - ``compute_execution_waves(feature_slices)`` — returns ordered waves
    - ``validate_feature_slices(slices)`` — structural validation
    - ``slices_to_state_dict(waves)`` — JSON-serializable state dicts

All exceptions are ``ValueError`` with clear, actionable messages.
"""

from __future__ import annotations

from dataclasses import asdict
from graphlib import CycleError, TopologicalSorter
from typing import Any

from spine.models.types import FeatureSlice

__all__ = [
    "compute_execution_waves",
    "validate_feature_slices",
    "slices_to_state_dict",
]


def validate_feature_slices(slices: list[FeatureSlice]) -> None:
    """Validate a collection of feature slices for structural correctness.

    Checks performed:
        1. Non-empty list (plan must have >= 1 slice).
        2. Unique IDs — no duplicates.
        3. Non-empty required fields (``id``, ``title``).
        4. Valid dependency references — every dependency must point to an
           existing slice ID.
        5. No dependency cycles.

    Args:
        slices: The feature slices to validate.

    Raises:
        ValueError: If any validation check fails. The message identifies
            the specific problem (duplicate IDs, missing dependencies, etc.).
    """
    if not slices:
        raise ValueError("Slice plan must contain at least one feature slice.")

    # ── Check 1: Unique IDs ────────────────────────────────────────────
    seen_ids: dict[str, int] = {}
    for slice_ in slices:
        seen_ids.setdefault(slice_.id, 0)
        seen_ids[slice_.id] += 1

    duplicates = {sid: count for sid, count in seen_ids.items() if count > 1}
    if duplicates:
        dup_list = ", ".join(f"'{sid}' (x{c})" for sid, c in duplicates.items())
        raise ValueError(f"Duplicate slice IDs found: {dup_list}")

    # ── Check 2: Non-empty required fields ─────────────────────────────
    all_ids = set(seen_ids.keys())
    for slice_ in slices:
        if not slice_.id.strip():
            raise ValueError("Slice has an empty id.")
        if not slice_.title.strip():
            raise ValueError(f"Slice '{slice_.id}' has an empty title.")

    # ── Check 3: Valid dependency references ───────────────────────────
    for slice_ in slices:
        invalid_deps = [d for d in slice_.dependencies if d not in all_ids]
        if invalid_deps:
            raise ValueError(
                f"Slice '{slice_.id}' references unknown dependencies: "
                f"{invalid_deps}. Valid IDs: {sorted(all_ids)}"
            )

    # ── Check 4: No cycles ─────────────────────────────────────────────
    graph: dict[str, set[str]] = {s.id: set(s.dependencies) for s in slices}
    try:
        sorter = TopologicalSorter(graph)
        sorter.prepare()  # Raises CycleError if cycle exists
    except CycleError as exc:
        raise ValueError(f"Dependency cycle detected among slices: {exc}") from exc


def compute_execution_waves(
    feature_slices: list[FeatureSlice],
) -> list[list[FeatureSlice]]:
    """Group feature slices into parallel execution waves.

    Uses topological sorting to determine the earliest wave each slice
    can execute in. Slices within the same wave have no mutual
    dependencies and can run in parallel.

    Args:
        feature_slices: The slices to schedule. Must pass
            ``validate_feature_slices()`` first.

    Returns:
        Ordered list of waves, where each wave is a list of slices
        that can execute concurrently. Wave 0 has no dependencies;
        wave *n* depends only on slices from waves 0..*n*-1.

    Raises:
        ValueError: If slices are invalid (empty, duplicates, bad
            references, or cycles).
    """
    validate_feature_slices(feature_slices)

    # Build lookup from ID -> FeatureSlice for wave assembly.
    slice_by_id: dict[str, FeatureSlice] = {s.id: s for s in feature_slices}

    # Build dependency graph: {node: set of predecessors}.
    graph: dict[str, set[str]] = {s.id: set(s.dependencies) for s in feature_slices}

    sorter = TopologicalSorter(graph)
    sorter.prepare()

    waves: list[list[FeatureSlice]] = []
    while sorter.is_active():
        # All currently-ready nodes can run in parallel (this wave).
        ready = sorter.get_ready()
        wave = [slice_by_id[node_id] for node_id in sorted(ready)]
        waves.append(wave)
        # Mark all wave members as done so their dependents become ready.
        for node_id in ready:
            sorter.done(node_id)

    return waves


def slices_to_state_dict(
    waves: list[list[FeatureSlice]],
) -> list[list[dict[str, Any]]]:
    """Convert execution waves to JSON-serializable dicts for WorkflowState.

    Produces a list-of-lists structure where the outer list represents
    waves and the inner lists contain serialized ``FeatureSlice`` dicts.
    This format is suitable for storing in ``WorkflowState`` fields or
    serializing directly to JSON.

    Example output::

        [
            [
                {"id": "add-models", "title": "...", "dependencies": [], ...},
                {"id": "add-config", "title": "...", "dependencies": [], ...},
            ],
            [
                {"id": "add-api", "title": "...",
                 "dependencies": ["add-models", "add-config"], ...},
            ],
        ]

    Args:
        waves: Ordered execution waves from ``compute_execution_waves()``.

    Returns:
        A list of waves, where each wave is a list of JSON-serializable
        dicts representing the feature slices in that wave.
    """
    result: list[list[dict[str, Any]]] = []

    for wave in waves:
        wave_dicts: list[dict[str, Any]] = [asdict(slice_) for slice_ in wave]
        result.append(wave_dicts)

    return result
