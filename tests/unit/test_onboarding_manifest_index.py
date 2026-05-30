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
from spine.work.onboarding.manifest_index import manifest_index, resolve_fragment
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
