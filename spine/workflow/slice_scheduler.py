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


_COMPLEXITY_ORDER = {"small": 0, "medium": 1, "large": 2}


def _as_text(value: Any) -> str:
    """execution_requirements may be a str or list[str]; render to str."""
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value or "")


def merge_file_overlapping_slices(slices: list[FeatureSlice]) -> list[FeatureSlice]:
    """Merge slices that share a target file into one slice.

    A plan that puts several slices over the SAME file(s) (e.g. three feature
    slices all editing ``config_view.py`` + ``api.py``) forces the IMPLEMENT
    loop to do that file 3× — overlapping per-file sub-slices that conflict and
    re-decompose, blowing the token budget (trace r2: 34 invocations / ~1M
    tokens). Slices touching a shared file cannot run in parallel anyway, so
    union them (transitively) and combine their fields into a single slice.
    Fully defensive: any failure returns the input unchanged.
    """
    try:
        if len(slices) < 2:
            return slices
        parent = {s.id: s.id for s in slices}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        file_owner: dict[str, str] = {}
        for s in slices:
            for f in s.target_files or []:
                if f in file_owner:
                    parent[find(s.id)] = find(file_owner[f])
                else:
                    file_owner[f] = s.id

        groups: dict[str, list[FeatureSlice]] = {}
        for s in slices:
            groups.setdefault(find(s.id), []).append(s)
        if len(groups) == len(slices):
            return slices  # nothing shares files

        remap: dict[str, str] = {}
        merged: list[FeatureSlice] = []
        for members in groups.values():
            if len(members) == 1:
                merged.append(members[0])
                remap[members[0].id] = members[0].id
                continue
            members = sorted(members, key=lambda s: s.id)
            new_id = members[0].id
            member_ids = {m.id for m in members}
            tf: list[str] = []
            ac: list[str] = []
            rs: list[str] = []
            ep: list = []
            er: list[str] = []
            deps: set[str] = set()
            comp = 0
            for m in members:
                for f in m.target_files or []:
                    if f not in tf:
                        tf.append(f)
                for a in m.acceptance_criteria or []:
                    if a not in ac:
                        ac.append(a)
                for r in m.reference_symbols or []:
                    if r not in rs:
                        rs.append(r)
                ep.extend(m.edit_plan or [])
                er.append(f"[{m.title}]\n{_as_text(m.execution_requirements)}".strip())
                deps |= set(m.dependencies or [])
                comp = max(comp, _COMPLEXITY_ORDER.get(m.complexity, 1))
            for mid in member_ids:
                remap[mid] = new_id
            merged.append(FeatureSlice(
                id=new_id,
                title=" + ".join(m.title for m in members),
                target_files=tf,
                execution_requirements="\n\n".join(er),
                dependencies=sorted(deps - member_ids),
                acceptance_criteria=ac,
                complexity=next(k for k, v in _COMPLEXITY_ORDER.items() if v == comp),
                reference_symbols=rs,
                edit_plan=ep,
            ))
        # Remap any dependency references to the merged representative ids.
        for s in merged:
            s.dependencies = sorted(
                {remap.get(d, d) for d in (s.dependencies or [])} - {s.id}
            )
        return merged
    except Exception:  # noqa: BLE001 — never let merging break scheduling
        return slices


def serialize_file_overlapping_slices(slices: list[FeatureSlice]) -> list[FeatureSlice]:
    """Chain slices that share a target file so they run sequentially.

    Same-file slices can't run in parallel (they'd conflict), but merging them
    into one big slice overwhelms the implementer. Instead, within each group of
    slices touching a shared file, add a dependency from each slice to the
    previous one — so they land in consecutive waves (the sandbox carries each
    edit forward) while staying small and individually actionable. Independent
    files still run in parallel. Fully defensive: returns the input on any error.
    """
    try:
        if len(slices) < 2:
            return slices
        parent = {s.id: s.id for s in slices}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        file_owner: dict[str, str] = {}
        for s in slices:
            for f in s.target_files or []:
                if f in file_owner:
                    parent[find(s.id)] = find(file_owner[f])
                else:
                    file_owner[f] = s.id

        groups: dict[str, list[FeatureSlice]] = {}
        for s in slices:
            groups.setdefault(find(s.id), []).append(s)
        if len(groups) == len(slices):
            return slices  # nothing shares files

        ids = {s.id for s in slices}
        for members in groups.values():
            if len(members) < 2:
                continue
            # Order the group consistently with any dependencies ALREADY declared
            # between its members, then chain forward in THAT order. Sorting by id
            # (the old behaviour) chained alphabetically, which reversed an
            # existing edge when the later-alphabetical slice was a dependency of
            # the earlier one — injecting a 2-cycle that compute_execution_waves
            # then rejected (two api.py slices, trace 55ae1919). A topological
            # order over intra-group edges makes "chain forward" cycle-free.
            member_ids = {s.id for s in members}
            by_id = {s.id: s for s in members}
            ts: TopologicalSorter[str] = TopologicalSorter()
            for s in sorted(members, key=lambda s: s.id):
                intra = [d for d in (s.dependencies or []) if d in member_ids]
                ts.add(s.id, *intra)
            try:
                ordered = [by_id[i] for i in ts.static_order()]
            except CycleError:
                # Authored cycle within the group (the critic should reject these);
                # leave deps untouched and let validation surface it.
                continue
            for prev, cur in zip(ordered, ordered[1:]):
                deps = set(cur.dependencies or [])
                # Chain forward only; the ordering guarantees this never reverses
                # an existing edge.
                if prev.id != cur.id and prev.id in ids:
                    deps.add(prev.id)
                cur.dependencies = sorted(deps)
        return slices
    except Exception:  # noqa: BLE001 — never let serialization break scheduling
        return slices


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
    # Serialize slices that share a target file BEFORE scheduling — they cannot
    # run in parallel (same-file edit conflict), but MERGING them into one big
    # slice overwhelms the implementer (trace 019ede24: a merged 27-reference
    # slice → the model read 24 symbols and never edited). Instead chain them
    # with dependencies so they run in separate waves, each small and
    # actionable, with the sandbox carrying earlier edits forward.
    feature_slices = serialize_file_overlapping_slices(feature_slices)
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
