"""Pure, graph-free context-projection helpers for distributed onboarding.

These two functions are the core context fix for the distributed synthesis
hierarchy (design Revision 2, §6.4). They guarantee that **no LLM ever receives
the whole manifest**:

- :func:`manifest_index` builds a *compact* index (names + roles + counts only,
  NO ``key_symbols`` / pattern ``evidence`` / ``raw_code``), ranked and capped so
  it stays bounded (~2-3k tokens) *regardless of repo size*. It drives the
  documentation manager's single planning call.
- :func:`resolve_fragment` projects exactly the slice of the manifest one section
  worker needs (per doc kind), then **hard-truncates to ``token_cap``** —
  degrading ``key_symbols`` -> names -> truncate so the returned fragment NEVER
  exceeds the cap, even for a pathologically large single module.

Both functions are pure: they take a :class:`RepoManifest`, return plain dicts,
make no LLM/tool calls, and do no I/O. Token metering uses
:func:`spine.agents._tokens.count_tokens`.
"""

from __future__ import annotations

import json
from typing import Any

from spine.agents._tokens import count_tokens
from spine.work.onboarding.manifest import RepoManifest
from spine.work.onboarding.synthesis_tools import ONBOARDING_DOC_NAMES

# Default ceiling for the number of ranked modules surfaced in the index. The
# real engine passes ``onboarding_max_sections``; this keeps the helper usable
# (and bounded) when called directly. The tail beyond this is grouped into a
# single synthetic "other modules" entry so the index is O(K), not O(repo).
_DEFAULT_MAX_MODULES: int = 32

# First path components that indicate a filesystem-derived artifact domain
# rather than a real code module. These arise when absolute paths outside the
# workspace root (e.g. /home/pat/Projects/spine/spine/log.py) leak into the
# AST indexer and get turned into dotted names like ``home.pat`` instead of
# the correct ``spine.log.py``. Filtering them prevents hallucinated sections.
_FILESYSTEM_ARTIFACT_PREFIXES: frozenset[str] = frozenset(
    {
        "home",
        "users",
        "tmp",
        "temp",
        "var",
        "etc",
        "usr",
        "opt",
        "root",
        "mnt",
        "media",
        "srv",
        "volumes",
        "private",
    }
)


def _is_path_artifact(name: str) -> bool:
    """Return True if *name* looks like an absolute-path-derived artifact domain.

    Checks whether the first dotted or slash-delimited component of *name* is a
    known filesystem root directory (``home``, ``users``, ``tmp``, …). Boundaries
    with such names should be excluded from the index and fragment projection so
    section workers never receive them.
    """
    if not name:
        return False
    first = name.split(".")[0].split("/")[0].lower()
    return first in _FILESYSTEM_ARTIFACT_PREFIXES


def _module_symbol_count(boundary: dict[str, Any]) -> int:
    """Number of key symbols recorded for a serialised module boundary."""
    return len(boundary.get("key_symbols", []) or [])


def _out_degree_by_module(
    boundaries: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, int]:
    """Compute out-degree (count of outgoing edges) per module name.

    Edges reference symbol/module names in ``src``; we attribute an edge to a
    module when ``src`` equals the module name or is prefixed by it (dotted or
    path-style). Only module *names* and integer counts are returned — never the
    edge list itself — so the index stays compact.
    """
    names = [b.get("name", "") for b in boundaries if b.get("name")]
    counts: dict[str, int] = {name: 0 for name in names}
    # Longest names first so "spine.work.onboarding" wins over "spine.work".
    ordered = sorted(names, key=len, reverse=True)
    for edge in edges:
        src = edge.get("src", "") or ""
        for name in ordered:
            if src == name or src.startswith(f"{name}.") or src.startswith(f"{name}/"):
                counts[name] = counts.get(name, 0) + 1
                break
    return counts


def manifest_index(
    manifest: RepoManifest,
    max_modules: int = _DEFAULT_MAX_MODULES,
) -> dict[str, Any]:
    """Build a compact, bounded index of *manifest* for the documentation manager.

    The index lists **names, one-line roles, and counts only** — it deliberately
    excludes ``key_symbols``, pattern ``evidence``, and any raw code so it stays
    bounded (~2-3k tokens) even for a 5,000-module repository. Modules are ranked
    by symbol count (descending); the top ``max_modules`` are kept verbatim and
    the remaining tail is collapsed into a single synthetic
    ``"__other_modules__"`` entry carrying aggregate counts.

    Args:
        manifest: The full :class:`RepoManifest` from the analysis stage.
        max_modules: Maximum number of individual modules surfaced before the
            tail is grouped. Defaults to :data:`_DEFAULT_MAX_MODULES`.

    Returns:
        A JSON-serialisable dict with keys ``mode``, ``tech_stack``,
        ``core_domains``, ``modules``, ``pattern_categories``, ``edge_counts``,
        ``totals``, and ``notes``.
    """
    data = manifest.to_dict()
    boundaries: list[dict[str, Any]] = [
        b for b in (data.get("module_boundaries", []) or [])
        if not _is_path_artifact(b.get("name", ""))
    ]
    edges: list[dict[str, Any]] = list(data.get("dependency_chains", []) or [])
    patterns: list[dict[str, Any]] = list(data.get("patterns", []) or [])

    out_degree = _out_degree_by_module(boundaries, edges)

    # Rank by symbol count desc, then name asc for a deterministic tail split.
    ranked = sorted(
        boundaries,
        key=lambda b: (-_module_symbol_count(b), b.get("name", "")),
    )
    kept = ranked[: max(max_modules, 0)]
    tail = ranked[max(max_modules, 0) :]

    modules: list[dict[str, Any]] = [
        {
            "name": b.get("name", ""),
            "path": b.get("path", ""),
            "role": b.get("role", ""),
            "symbol_count": _module_symbol_count(b),
        }
        for b in kept
    ]
    if tail:
        modules.append(
            {
                "name": "__other_modules__",
                "path": "",
                "role": f"{len(tail)} smaller modules grouped (tail)",
                "symbol_count": sum(_module_symbol_count(b) for b in tail),
            }
        )

    # edge_counts mirrors the kept modules only (plus a tail aggregate) so it
    # cannot grow without bound either.
    edge_counts: dict[str, int] = {
        b.get("name", ""): out_degree.get(b.get("name", ""), 0) for b in kept
    }
    if tail:
        edge_counts["__other_modules__"] = sum(
            out_degree.get(b.get("name", ""), 0) for b in tail
        )

    pattern_categories: list[str] = []
    seen: set[str] = set()
    for p in patterns:
        cat = p.get("category", "")
        if cat and cat not in seen:
            seen.add(cat)
            pattern_categories.append(cat)

    return {
        "mode": data.get("mode", ""),
        "tech_stack": list(data.get("tech_stack", []) or []),
        "core_domains": [
            d for d in (data.get("core_domains", []) or [])
            if not _is_path_artifact(d)
        ],
        "modules": modules,
        "pattern_categories": pattern_categories,
        "edge_counts": edge_counts,
        "totals": {
            "symbol_count": int(data.get("symbol_count", 0) or 0),
            "file_count": int(data.get("file_count", 0) or 0),
            "module_count": len(boundaries),
            "pattern_count": len(patterns),
            "edge_count": len(edges),
        },
        "notes": data.get("notes", ""),
    }


# ── resolve_fragment ────────────────────────────────────────────────────────


def _fragment_token_count(fragment: dict[str, Any]) -> int:
    """Token count of a fragment as the worker would see it (compact JSON)."""
    return count_tokens(json.dumps(fragment, ensure_ascii=False))


def _doc_kind_of(fragment_keys: dict[str, Any]) -> str:
    """Resolve the onboarding doc kind a fragment-key set targets.

    Requires an explicit, valid ``doc_id`` key. An unknown / missing ``doc_id``
    raises ``ValueError`` so a hollow plan never silently resolves to the
    architecture map (design finding #5). Callers that need a tolerant default
    must validate up front via :func:`validate_fragment_keys`.
    """
    doc_id = fragment_keys.get("doc_id", "")
    if doc_id in ONBOARDING_DOC_NAMES:
        return doc_id
    raise ValueError(f"unknown onboarding doc_id: {doc_id!r}")


def validate_fragment_keys(
    index: dict[str, Any],
    fragment_keys: dict[str, Any],
) -> list[str]:
    """Return the list of unresolvable selectors in *fragment_keys*.

    A refined :class:`SectionPlan` references manifest entries by stable key.
    This checks those keys against the compact *index* (the only repo view the
    documentation manager ever sees) and returns a list of human-readable
    reasons for every selector that cannot be resolved:

    - a ``doc_id`` that is not one of :data:`ONBOARDING_DOC_NAMES`;
    - any ``modules`` name absent from the index modules (the synthetic
      ``"__other_modules__"`` tail entry is accepted);
    - any ``categories`` name absent from the index ``pattern_categories``;
    - any ``domains`` name absent from the index ``core_domains``.

    An empty selector list (e.g. ``"modules": []``) means "the full set for this
    doc kind" and is always valid. The returned list is empty when every
    selector resolves; a non-empty list means the plan must be rejected.
    """
    reasons: list[str] = []

    doc_id = fragment_keys.get("doc_id", "")
    if doc_id not in ONBOARDING_DOC_NAMES:
        reasons.append(f"unknown doc_id: {doc_id!r}")

    known_modules = {
        m.get("name", "") for m in (index.get("modules", []) or []) if isinstance(m, dict)
    }
    known_categories = set(index.get("pattern_categories", []) or [])
    known_domains = set(index.get("core_domains", []) or [])

    for name in fragment_keys.get("modules", []) or []:
        if name not in known_modules:
            reasons.append(f"unknown module: {name!r}")
    for name in fragment_keys.get("categories", []) or []:
        if name not in known_categories:
            reasons.append(f"unknown category: {name!r}")
    for name in fragment_keys.get("domains", []) or []:
        if name not in known_domains:
            reasons.append(f"unknown domain: {name!r}")

    return reasons


def _matching_modules(
    boundaries: list[dict[str, Any]],
    wanted: list[str],
) -> list[dict[str, Any]]:
    """Boundaries whose name is in *wanted* (order follows *wanted*)."""
    by_name = {b.get("name", ""): b for b in boundaries}
    out: list[dict[str, Any]] = []
    for name in wanted:
        if name in by_name:
            out.append(by_name[name])
    return out


def _edges_for_modules(
    edges: list[dict[str, Any]],
    module_names: list[str],
) -> list[dict[str, Any]]:
    """Dependency edges whose ``src`` belongs to one of *module_names*."""
    ordered = sorted(module_names, key=len, reverse=True)
    out: list[dict[str, Any]] = []
    for edge in edges:
        src = edge.get("src", "") or ""
        for name in ordered:
            if src == name or src.startswith(f"{name}.") or src.startswith(f"{name}/"):
                out.append(edge)
                break
    return out


def _strip_key_symbols_to_names(boundary: dict[str, Any]) -> dict[str, Any]:
    """A copy of *boundary* whose ``key_symbols`` are reduced to bare names."""
    return {
        **boundary,
        "key_symbols": [
            s.get("symbol_name", "") for s in boundary.get("key_symbols", []) or []
        ],
    }


def _enforce_token_cap(
    fragment: dict[str, Any],
    token_cap: int,
) -> dict[str, Any]:
    """Guarantee ``count_tokens(json(fragment)) <= token_cap``.

    Degradation ladder (per design §6.4): if the projected fragment exceeds the
    cap, first reduce module ``key_symbols`` to names-only, then progressively
    drop list elements (modules / edges / findings / domains), then truncate the
    largest string field, and finally fall back to a minimal stub. The returned
    fragment is GUARANTEED to be at or under the cap.
    """
    if token_cap <= 0:
        return {}

    if _fragment_token_count(fragment) <= token_cap:
        return fragment

    work = dict(fragment)

    # Step 1: degrade key_symbols -> names everywhere they appear.
    if isinstance(work.get("modules"), list):
        work["modules"] = [
            _strip_key_symbols_to_names(b) if isinstance(b, dict) else b
            for b in work["modules"]
        ]
        work["degraded"] = True
        if _fragment_token_count(work) <= token_cap:
            return work

    # Step 2: shrink the largest list channel by BINARY SEARCH on its length so
    # large fragments (thousands of modules/edges) converge in O(log n)
    # tokenizations rather than O(n) — re-tokenizing per element is too slow.
    list_keys = ("modules", "edges", "findings", "module_roles", "domains")
    for _ in range(len(list_keys) + 1):
        if _fragment_token_count(work) <= token_cap:
            return work
        # Pick the longest remaining list to trim.
        longest_key: str | None = None
        longest_len = 0
        for key in list_keys:
            val = work.get(key)
            if isinstance(val, list) and len(val) > longest_len:
                longest_key = key
                longest_len = len(val)
        if longest_key is None:
            break

        full = work[longest_key]
        # Largest prefix length that keeps the fragment under the cap.
        lo, hi = 0, len(full)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            work[longest_key] = full[:mid]
            if _fragment_token_count(work) <= token_cap:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        work[longest_key] = full[:best]
        work["truncated"] = True
        if best < len(full) and _fragment_token_count(work) <= token_cap:
            return work

    if _fragment_token_count(work) <= token_cap:
        return work

    # Step 3: hard character-truncate any remaining oversized string fields.
    work["truncated"] = True
    for key, val in list(work.items()):
        if isinstance(val, str) and len(val) > 200:
            # token_cap*4 chars is a safe upper bound; shrink iteratively.
            limit = max(token_cap * 3, 200)
            work[key] = val[:limit]
    if _fragment_token_count(work) <= token_cap:
        return work

    # Step 4: minimal stub — progressively stripped so it can NEVER exceed the
    # cap, even for a pathologically tiny ``token_cap``. The fallbacks shrink
    # from a descriptive stub down to an empty dict (1 token).
    doc_id = fragment.get("doc_id", "")
    for stub in (
        {
            "doc_id": doc_id,
            "truncated": True,
            "note": "fragment exceeded token cap and was reduced to a stub",
        },
        {"doc_id": doc_id, "truncated": True},
        {"truncated": True},
        {},
    ):
        if _fragment_token_count(stub) <= token_cap:
            return stub
    return {}


def resolve_fragment(
    manifest: RepoManifest,
    fragment_keys: dict[str, Any],
    token_cap: int,
) -> dict[str, Any]:
    """Project the bounded manifest slice one section worker needs.

    The projection depends on the doc kind (``fragment_keys["doc_id"]``):

    - ``ARCHITECTURE_MAP`` — the named module boundaries (full ``key_symbols``)
      plus the dependency edges originating from those modules.
    - ``CODING_GUIDELINES`` — the named pattern categories' findings (with
      evidence).
    - ``PROJECT_DEFINITION`` — the module roles for the named core domains.
    - ``SPINE_ASSISTANCE_REQUIREMENTS`` — size/budget signals only (symbol/file
      counts, largest modules, notes).

    Whatever the projection, the result is then passed through
    :func:`_enforce_token_cap`, which degrades ``key_symbols`` -> names ->
    truncate so the returned fragment **never** exceeds *token_cap*.

    This is a thin wrapper over :func:`resolve_fragment_from_dict` for callers
    that hold a :class:`RepoManifest`; the graph's ``_section_router`` already
    holds the manifest *dict* (``state["manifest"]``) and calls the dict variant
    directly, avoiding a per-section ``from_dict``/``to_dict`` round-trip
    (finding #8).

    Args:
        manifest: The full :class:`RepoManifest`.
        fragment_keys: ``{"doc_id": <NAME>, "modules"|"categories"|"domains":
            [...]}`` selecting what to project. Missing selectors default to the
            full set for that doc kind.
        token_cap: Hard ceiling on the fragment's token count (e.g.
            ``onboarding_section_token_cap``, default 6000).

    Returns:
        A small, JSON-serialisable dict at or under *token_cap* tokens.
    """
    return resolve_fragment_from_dict(manifest.to_dict(), fragment_keys, token_cap)


def resolve_fragment_from_dict(
    data: dict[str, Any],
    fragment_keys: dict[str, Any],
    token_cap: int,
) -> dict[str, Any]:
    """Project the bounded manifest slice from an already-serialised manifest.

    Identical projection to :func:`resolve_fragment` but operates directly on a
    :meth:`RepoManifest.to_dict` dict (the form the graph carries in
    ``state["manifest"]``). This eliminates the per-section
    ``RepoManifest.from_dict`` -> ``manifest.to_dict`` deep-copy round-trip the
    router previously paid once per section (finding #8). The output is
    byte-identical to :func:`resolve_fragment` for the same manifest + keys.

    Args:
        data: A manifest dict as produced by :meth:`RepoManifest.to_dict`.
        fragment_keys: See :func:`resolve_fragment`.
        token_cap: See :func:`resolve_fragment`.

    Returns:
        A small, JSON-serialisable dict at or under *token_cap* tokens.
    """
    boundaries: list[dict[str, Any]] = [
        b for b in (data.get("module_boundaries", []) or [])
        if not _is_path_artifact(b.get("name", ""))
    ]
    edges: list[dict[str, Any]] = list(data.get("dependency_chains", []) or [])
    patterns: list[dict[str, Any]] = list(data.get("patterns", []) or [])

    kind = _doc_kind_of(fragment_keys)
    fragment: dict[str, Any] = {"doc_id": kind}

    if kind == "ARCHITECTURE_MAP":
        wanted = list(fragment_keys.get("modules", []) or [])
        mods = (
            _matching_modules(boundaries, wanted) if wanted else boundaries
        )
        names = [m.get("name", "") for m in mods]
        fragment["modules"] = mods
        fragment["edges"] = _edges_for_modules(edges, names)

    elif kind == "CODING_GUIDELINES":
        wanted = list(fragment_keys.get("categories", []) or [])
        if wanted:
            wanted_set = set(wanted)
            findings = [p for p in patterns if p.get("category", "") in wanted_set]
        else:
            findings = patterns
        fragment["findings"] = findings

    elif kind == "PROJECT_DEFINITION":
        wanted = list(fragment_keys.get("domains", []) or [])
        if wanted:
            domains = [d for d in wanted if not _is_path_artifact(d)]
        else:
            domains = [
                d for d in (data.get("core_domains", []) or [])
                if not _is_path_artifact(d)
            ]
        domain_set = {d for d in domains}
        # A domain maps to a module of the same name (core_domains are derived
        # from boundary names); surface those modules' roles only.
        roles = [
            {
                "name": b.get("name", ""),
                "path": b.get("path", ""),
                "role": b.get("role", ""),
            }
            for b in boundaries
            if (not domain_set) or b.get("name", "") in domain_set
        ]
        fragment["domains"] = domains
        fragment["module_roles"] = roles
        fragment["tech_stack"] = list(data.get("tech_stack", []) or [])

    else:  # SPINE_ASSISTANCE_REQUIREMENTS
        largest = sorted(
            boundaries,
            key=lambda b: (-_module_symbol_count(b), b.get("name", "")),
        )[:10]
        fragment["totals"] = {
            "symbol_count": int(data.get("symbol_count", 0) or 0),
            "file_count": int(data.get("file_count", 0) or 0),
            "module_count": len(boundaries),
            "pattern_count": len(patterns),
        }
        fragment["largest_modules"] = [
            {
                "name": b.get("name", ""),
                "symbol_count": _module_symbol_count(b),
            }
            for b in largest
        ]
        fragment["tech_stack"] = list(data.get("tech_stack", []) or [])
        fragment["notes"] = data.get("notes", "")

    return _enforce_token_cap(fragment, token_cap)
