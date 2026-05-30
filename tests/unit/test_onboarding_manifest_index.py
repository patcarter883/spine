"""Tests for the bounded context-projection helpers.

Builds a *pathologically large* synthetic :class:`RepoManifest` (thousands of
modules, plus one giant single module) and asserts:

- :func:`manifest_index` stays bounded (token count well under a sane ceiling)
  and excludes ``key_symbols`` / pattern ``evidence`` entirely; and
- :func:`resolve_fragment` output token count is ``<= token_cap`` for EVERY doc
  kind, including for the giant single module.
"""

from __future__ import annotations

import json

from spine.agents._tokens import count_tokens
from spine.work.onboarding.manifest import (
    DependencyEdge,
    ModuleBoundary,
    PatternFinding,
    RepoManifest,
    SymbolRef,
)
from spine.work.onboarding.manifest_index import (
    manifest_index,
    resolve_fragment,
    resolve_fragment_from_dict,
)
from spine.work.onboarding.synthesis_tools import ONBOARDING_DOC_NAMES

_INDEX_TOKEN_CEILING = 6000
_TOKEN_CAP = 6000


def _giant_symbol(i: int) -> SymbolRef:
    return SymbolRef(
        file_path=f"pkg/mod/file_{i}.py",
        symbol_name=f"symbol_{i}",
        symbol_type="function",
        lang="python",
        summary="A fairly long summary line that exists to inflate token cost. " * 4,
    )


def _build_large_manifest(
    n_modules: int = 5000,
    giant_module_symbols: int = 4000,
) -> RepoManifest:
    """A 5000-module manifest plus one module with thousands of key symbols."""
    boundaries: list[ModuleBoundary] = []
    edges: list[DependencyEdge] = []

    # The giant single module — pathologically large key_symbols list.
    giant_name = "giant.module"
    boundaries.append(
        ModuleBoundary(
            name=giant_name,
            path="giant/module",
            role="An enormous module with thousands of symbols " * 3,
            key_symbols=[_giant_symbol(i) for i in range(giant_module_symbols)],
        )
    )

    for m in range(n_modules):
        name = f"pkg.module_{m}"
        boundaries.append(
            ModuleBoundary(
                name=name,
                path=f"pkg/module_{m}",
                role=f"Module {m} owns feature area {m}.",
                key_symbols=[_giant_symbol(j) for j in range(3)],
            )
        )
        # A couple of outgoing edges per module.
        edges.append(DependencyEdge(src=name, dst=giant_name, kind="imports"))
        edges.append(DependencyEdge(src=f"{name}.func", dst="pkg.module_0", kind="calls"))

    patterns = [
        PatternFinding(
            category=cat,
            description=f"The {cat} convention used pervasively across the repo. " * 3,
            evidence=[_giant_symbol(k) for k in range(3)],
        )
        for cat in ("logging", "config", "data_model", "error_handling", "testing")
    ]

    return RepoManifest(
        workspace_root="/tmp/huge",
        mode="brownfield",
        tech_stack=["python", "langgraph"],
        core_domains=[f"pkg.module_{m}" for m in range(10)],
        module_boundaries=boundaries,
        dependency_chains=edges,
        patterns=patterns,
        symbol_count=giant_module_symbols + n_modules * 3,
        file_count=n_modules + 1,
        generated_at="2026-05-29T00:00:00+00:00",
        notes="huge synthetic repo",
    )


def test_manifest_index_is_bounded_and_excludes_bodies() -> None:
    manifest = _build_large_manifest()
    index = manifest_index(manifest)

    serialized = json.dumps(index, ensure_ascii=False)
    tokens = count_tokens(serialized)
    assert tokens <= _INDEX_TOKEN_CEILING, f"index too large: {tokens} tokens"

    # No key_symbols / evidence / raw bodies anywhere in the index.
    assert "key_symbols" not in serialized
    assert "evidence" not in serialized
    assert "symbol_name" not in serialized

    # Modules are capped + tail grouped, not one entry per 5000 modules.
    assert len(index["modules"]) <= 64
    assert any(m["name"] == "__other_modules__" for m in index["modules"])

    # Pattern categories are names only.
    assert index["pattern_categories"] == [
        "logging",
        "config",
        "data_model",
        "error_handling",
        "testing",
    ]
    assert index["totals"]["module_count"] == 5001


def test_resolve_fragment_respects_cap_for_every_doc_kind() -> None:
    manifest = _build_large_manifest()

    cases = [
        {"doc_id": "ARCHITECTURE_MAP", "modules": ["giant.module"]},  # huge module
        {"doc_id": "ARCHITECTURE_MAP", "modules": []},  # all modules
        {"doc_id": "CODING_GUIDELINES", "categories": ["logging"]},
        {"doc_id": "CODING_GUIDELINES", "categories": []},
        {"doc_id": "PROJECT_DEFINITION", "domains": ["pkg.module_0"]},
        {"doc_id": "PROJECT_DEFINITION", "domains": []},
        {"doc_id": "SPINE_ASSISTANCE_REQUIREMENTS"},
    ]

    for fragment_keys in cases:
        fragment = resolve_fragment(manifest, fragment_keys, _TOKEN_CAP)
        tokens = count_tokens(json.dumps(fragment, ensure_ascii=False))
        assert tokens <= _TOKEN_CAP, (
            f"fragment for {fragment_keys} exceeded cap: {tokens} > {_TOKEN_CAP}"
        )


def test_resolve_fragment_giant_module_degrades_key_symbols() -> None:
    """The giant module alone exceeds the cap, so it must be degraded."""
    manifest = _build_large_manifest()
    fragment = resolve_fragment(
        manifest, {"doc_id": "ARCHITECTURE_MAP", "modules": ["giant.module"]}, _TOKEN_CAP
    )
    tokens = count_tokens(json.dumps(fragment, ensure_ascii=False))
    assert tokens <= _TOKEN_CAP
    # Degradation/truncation markers should be present given the size.
    assert fragment.get("degraded") or fragment.get("truncated")


def test_resolve_fragment_tiny_cap_never_exceeds() -> None:
    """Even an absurdly small cap yields a fragment under that cap.

    Includes caps smaller than the descriptive stub (~30 tokens) and the
    degenerate ``token_cap <= 0`` case, so the progressive-stub fallback is
    exercised down to an empty dict.
    """
    manifest = _build_large_manifest()
    for fragment_keys in (
        {"doc_id": "ARCHITECTURE_MAP", "modules": ["giant.module"]},
        {"doc_id": "CODING_GUIDELINES", "categories": []},
        {"doc_id": "SPINE_ASSISTANCE_REQUIREMENTS"},
    ):
        # Caps below the descriptive stub (~32 tokens) force the progressive
        # fallback; cap=1 is the floor an empty dict ("{}") satisfies.
        for cap in (50, 10, 3, 1):
            fragment = resolve_fragment(manifest, fragment_keys, cap)
            tokens = count_tokens(json.dumps(fragment, ensure_ascii=False))
            assert tokens <= cap, f"{fragment_keys} @cap={cap} -> {tokens} tokens > {cap}"
        # Degenerate non-positive cap: no JSON has 0 tokens, so return the
        # smallest possible value ({}), never the multi-token stub.
        assert resolve_fragment(manifest, fragment_keys, 0) == {}


def test_resolve_fragment_doc_ids_all_supported() -> None:
    manifest = _build_large_manifest(n_modules=10, giant_module_symbols=10)
    for doc_id in ONBOARDING_DOC_NAMES:
        fragment = resolve_fragment(manifest, {"doc_id": doc_id}, _TOKEN_CAP)
        assert fragment["doc_id"] == doc_id


def test_from_dict_projection_byte_identical_to_resolve_fragment() -> None:
    """Finding #8: the in-state-dict projection equals the old round-trip output.

    ``_section_router`` now slices the in-state manifest DICT once per run via
    :func:`resolve_fragment_from_dict` instead of paying a per-section
    ``from_dict``/``to_dict`` deep copy. The projected fragment must be
    byte-identical to what :func:`resolve_fragment` (which round-trips through a
    ``RepoManifest``) produces for the same selectors and cap.
    """
    manifest = _build_large_manifest(n_modules=200, giant_module_symbols=400)
    data = manifest.to_dict()

    cases = [
        {"doc_id": "ARCHITECTURE_MAP", "modules": ["giant.module"]},
        {"doc_id": "ARCHITECTURE_MAP", "modules": ["pkg.module_0", "pkg.module_1"]},
        {"doc_id": "ARCHITECTURE_MAP", "modules": []},
        {"doc_id": "CODING_GUIDELINES", "categories": ["logging"]},
        {"doc_id": "CODING_GUIDELINES", "categories": []},
        {"doc_id": "PROJECT_DEFINITION", "domains": ["pkg.module_0"]},
        {"doc_id": "PROJECT_DEFINITION", "domains": []},
        {"doc_id": "SPINE_ASSISTANCE_REQUIREMENTS"},
    ]
    for cap in (_TOKEN_CAP, 800, 50, 3):
        for keys in cases:
            via_manifest = resolve_fragment(manifest, keys, cap)
            via_dict = resolve_fragment_from_dict(data, keys, cap)
            assert via_dict == via_manifest, f"mismatch for {keys} @cap={cap}"
            # Byte-identical at the JSON level too (what the worker actually sees).
            assert json.dumps(via_dict, ensure_ascii=False) == json.dumps(
                via_manifest, ensure_ascii=False
            )


def test_from_dict_does_not_mutate_input() -> None:
    """The dict projection must not mutate the caller's shared in-state manifest."""
    manifest = _build_large_manifest(n_modules=50, giant_module_symbols=80)
    data = manifest.to_dict()
    before = json.dumps(data, ensure_ascii=False)
    for keys in (
        {"doc_id": "ARCHITECTURE_MAP", "modules": []},
        {"doc_id": "ARCHITECTURE_MAP", "modules": ["giant.module"]},
        {"doc_id": "CODING_GUIDELINES", "categories": []},
    ):
        resolve_fragment_from_dict(data, keys, 50)
    assert json.dumps(data, ensure_ascii=False) == before


# ── Field-coverage guard (finding #10) ───────────────────────────────────────
#
# Every top-level ``RepoManifest`` field must reach the synthesis LLM tiers
# through SOME bounded surface — either the compact :func:`manifest_index` (the
# only repo view the documentation manager sees) or at least one
# :func:`resolve_fragment` projection (the only repo view a section worker
# sees). If a new manifest field is added and wired into neither, it would
# silently never influence the generated docs. This guard fails on that drift.
#
# Fields that are DELIBERATELY excluded from both surfaces (with the reason)
# must be listed here, so adding a field forces an explicit decision: project
# it, or document why it is excluded.
_EXCLUDED_MANIFEST_FIELDS: dict[str, str] = {
    # Absolute path of the analysed repo — provenance metadata, not content the
    # onboarding docs should describe; deliberately never sent to any LLM.
    "workspace_root": "provenance path; not doc content",
    # ISO timestamp of the analysis run — provenance metadata only.
    "generated_at": "provenance timestamp; not doc content",
}


def _coverage_manifest() -> RepoManifest:
    """A small manifest with a distinctive sentinel value in every field."""
    sym = SymbolRef(
        file_path="pkg/auth/login.py",
        symbol_name="authenticate",
        symbol_type="function",
        lang="python",
        summary="performs login",
    )
    return RepoManifest(
        workspace_root="/abs/sentinel/root",
        mode="brownfield",
        tech_stack=["python", "langgraph"],
        core_domains=["auth.domain"],
        module_boundaries=[
            ModuleBoundary(
                name="auth.domain",
                path="pkg/auth",
                role="owns authentication",
                key_symbols=[sym],
            ),
        ],
        dependency_chains=[
            DependencyEdge(src="auth.domain", dst="db.layer", kind="calls"),
        ],
        patterns=[
            PatternFinding(
                category="error_handling",
                description="raise typed errors",
                evidence=[sym],
            ),
        ],
        symbol_count=4242,
        file_count=314,
        generated_at="2026-01-01T00:00:00",
        notes="analysis-caveat-sentinel",
    )


def test_every_manifest_field_is_indexed_or_projected_or_excluded() -> None:
    """Guard: each RepoManifest field surfaces in the index or a fragment.

    Asserts (a) every top-level field is classified as either projected or an
    explicit documented exclusion (so a NEW field can't be silently ignored),
    and (b) the projected fields actually appear in the index JSON or at least
    one fragment projection (so a wiring regression that drops a field is
    caught).
    """
    import dataclasses

    manifest = _coverage_manifest()
    field_names = {f.name for f in dataclasses.fields(RepoManifest)}

    # Build the index plus a fragment for every doc kind (default selectors =
    # the full set for that kind), then concatenate their JSON. A field is
    # "covered" if its sentinel data appears anywhere in that combined text.
    index = manifest_index(manifest)
    fragments = [
        resolve_fragment(manifest, {"doc_id": doc}, _TOKEN_CAP)
        for doc in ONBOARDING_DOC_NAMES
    ]
    combined = json.dumps(index, ensure_ascii=False) + "".join(
        json.dumps(fr, ensure_ascii=False) for fr in fragments
    )

    # Per-field sentinel signal that must appear in some bounded surface.
    signals: dict[str, str] = {
        "mode": "brownfield",
        "tech_stack": "langgraph",
        "core_domains": "auth.domain",
        "module_boundaries": "owns authentication",  # module role text
        "dependency_chains": "db.layer",  # edge dst surfaced in ARCHITECTURE_MAP
        "patterns": "error_handling",  # pattern category
        "symbol_count": "4242",
        "file_count": "314",
        "notes": "analysis-caveat-sentinel",
    }

    projected = set(signals)
    excluded = set(_EXCLUDED_MANIFEST_FIELDS)

    # (a) Classification is exhaustive: a new field forces an explicit choice.
    unclassified = field_names - projected - excluded
    assert not unclassified, (
        f"RepoManifest field(s) {sorted(unclassified)} are neither projected "
        f"into manifest_index/resolve_fragment nor listed in "
        f"_EXCLUDED_MANIFEST_FIELDS. Project them or document the exclusion."
    )
    assert projected | excluded == field_names, (
        "field-coverage guard lists a field that no longer exists on "
        f"RepoManifest: {sorted((projected | excluded) - field_names)}"
    )

    # (b) Every projected field's sentinel really surfaces in a bounded view.
    for field, signal in signals.items():
        assert signal in combined, (
            f"RepoManifest.{field} (sentinel {signal!r}) does not appear in the "
            f"manifest_index or any resolve_fragment projection — it would never "
            f"reach the synthesis LLMs."
        )


# ── Path-artifact filtering (trace 019e7855) ─────────────────────────────────


def _artifact_manifest() -> RepoManifest:
    """Manifest with both real domains and path-artifact entries like home.pat."""
    sym = SymbolRef(
        file_path="spine/log.py",
        symbol_name="configure_logging",
        symbol_type="function",
        lang="python",
        summary="sets up logging",
    )
    return RepoManifest(
        workspace_root="/home/pat/Projects/spine",
        mode="brownfield",
        tech_stack=["python"],
        core_domains=["spine.agents", "home.pat", "users.runner", "tmp.scratch"],
        module_boundaries=[
            ModuleBoundary(
                name="spine.agents",
                path="spine/agents",
                role="orchestration layer",
                key_symbols=[sym],
            ),
            ModuleBoundary(
                name="home.pat",
                path="home/pat",
                role="path artifact",
                key_symbols=[sym],
            ),
            ModuleBoundary(
                name="users.runner",
                path="users/runner",
                role="another artifact",
                key_symbols=[],
            ),
            ModuleBoundary(
                name="tmp.scratch",
                path="tmp/scratch",
                role="temp artifact",
                key_symbols=[],
            ),
        ],
        dependency_chains=[],
        patterns=[],
        symbol_count=1,
        file_count=4,
        generated_at="2026-05-30T00:00:00",
        notes="",
    )


def test_manifest_index_excludes_path_artifact_domains() -> None:
    """manifest_index must drop home.*, users.*, tmp.* from core_domains and modules."""
    manifest = _artifact_manifest()
    index = manifest_index(manifest)

    assert "spine.agents" in index["core_domains"]
    assert "home.pat" not in index["core_domains"]
    assert "users.runner" not in index["core_domains"]
    assert "tmp.scratch" not in index["core_domains"]

    module_names = {m["name"] for m in index["modules"]}
    assert "spine.agents" in module_names
    assert "home.pat" not in module_names
    assert "users.runner" not in module_names
    assert "tmp.scratch" not in module_names


def test_resolve_fragment_project_definition_excludes_path_artifacts() -> None:
    """PROJECT_DEFINITION fragment must not include path-artifact domains."""
    manifest = _artifact_manifest()

    # Explicit domains list: path artifacts are filtered out.
    fragment = resolve_fragment(
        manifest,
        {"doc_id": "PROJECT_DEFINITION", "domains": ["spine.agents", "home.pat"]},
        _TOKEN_CAP,
    )
    domain_names = fragment.get("domains", [])
    assert "spine.agents" in domain_names
    assert "home.pat" not in domain_names

    role_names = {r["name"] for r in fragment.get("module_roles", [])}
    assert "spine.agents" in role_names
    assert "home.pat" not in role_names

    # Default (empty) domains selector: same filtering applies.
    fragment_default = resolve_fragment(
        manifest,
        {"doc_id": "PROJECT_DEFINITION", "domains": []},
        _TOKEN_CAP,
    )
    assert "home.pat" not in fragment_default.get("domains", [])
    assert "users.runner" not in fragment_default.get("domains", [])


def test_resolve_fragment_architecture_map_excludes_path_artifact_modules() -> None:
    """ARCHITECTURE_MAP fragment must not include path-artifact module boundaries."""
    manifest = _artifact_manifest()

    # Empty selector = full set: path artifacts filtered from boundaries.
    fragment = resolve_fragment(
        manifest,
        {"doc_id": "ARCHITECTURE_MAP", "modules": []},
        _TOKEN_CAP,
    )
    module_names = {m.get("name", "") for m in fragment.get("modules", [])}
    assert "spine.agents" in module_names
    assert "home.pat" not in module_names
    assert "users.runner" not in module_names


def test_is_path_artifact_recognises_prefixes() -> None:
    """_is_path_artifact should match known filesystem prefixes and nothing else."""
    from spine.work.onboarding.manifest_index import _is_path_artifact

    assert _is_path_artifact("home.pat")
    assert _is_path_artifact("home.pat.Projects.spine")
    assert _is_path_artifact("users.runner")
    assert _is_path_artifact("HOME.PAT")  # case-insensitive
    assert _is_path_artifact("tmp.scratch")
    assert _is_path_artifact("var.log")

    assert not _is_path_artifact("spine.agents")
    assert not _is_path_artifact("alembic.env.py")
    assert not _is_path_artifact("tests.unit")
    assert not _is_path_artifact("")
    assert not _is_path_artifact("src.utils")
