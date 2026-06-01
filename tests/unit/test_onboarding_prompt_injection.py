"""Unit tests for onboarding-document injection into phase prompts.

Covers the resolver + reference-block builder in
:mod:`spine.agents.skills_resolver` that feed each phase agent the relevant
onboarding document (hybrid injection: the most-relevant doc per phase in full,
the rest referenced by path). See the workflow plan
``workflow-plan-integration-of-dynamic-nest``.
"""

from __future__ import annotations

from pathlib import Path

from spine.agents.skills_resolver import (
    _ONBOARDING_INJECT_BYTE_CAP,
    _PHASE_PRIMARY_DOC,
    build_onboarding_reference,
    resolve_onboarding_docs,
)
from spine.models.enums import PhaseName
from spine.work.onboarding.synthesis_tools import (
    ONBOARDING_DOC_NAMES,
    onboarding_docs_dir,
)


def _write_docs(root: Path, *, sizes: dict[str, int] | None = None) -> Path:
    """Write the four onboarding docs to ``<root>/.spine/onboarding``."""
    sizes = sizes or {}
    docs_dir = onboarding_docs_dir(str(root))
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ONBOARDING_DOC_NAMES:
        fname = f"{name}.md"
        body = "x" * sizes.get(fname, 64)
        (docs_dir / fname).write_text(f"# {name}\n{body}", encoding="utf-8")
    return docs_dir


# ── resolve_onboarding_docs ──────────────────────────────────────────────


def test_no_workspace_root_returns_empty() -> None:
    assert resolve_onboarding_docs(None, PhaseName.SPECIFY.value) == (None, [])
    assert resolve_onboarding_docs("", PhaseName.SPECIFY.value) == (None, [])


def test_missing_docs_dir_returns_empty(tmp_path: Path) -> None:
    # No .spine/onboarding directory at all.
    assert resolve_onboarding_docs(str(tmp_path), PhaseName.SPECIFY.value) == (None, [])


def test_specify_injects_project_definition(tmp_path: Path) -> None:
    docs_dir = _write_docs(tmp_path)
    inject, reference = resolve_onboarding_docs(str(tmp_path), PhaseName.SPECIFY.value)

    assert inject == str(docs_dir / "PROJECT_DEFINITION.md")
    # The injected doc is excluded from the references; the other three remain.
    ref_names = {name for name, _ in reference}
    assert "PROJECT_DEFINITION.md" not in ref_names
    assert ref_names == {
        "CODING_GUIDELINES.md",
        "ARCHITECTURE_MAP.md",
        "SPINE_ASSISTANCE_REQUIREMENTS.md",
    }


def test_each_phase_injects_its_primary(tmp_path: Path) -> None:
    docs_dir = _write_docs(tmp_path)
    for phase, primary in _PHASE_PRIMARY_DOC.items():
        inject, reference = resolve_onboarding_docs(str(tmp_path), phase)
        assert inject == str(docs_dir / primary)
        assert primary not in {name for name, _ in reference}


def test_unknown_phase_injects_nothing_but_references_all(tmp_path: Path) -> None:
    _write_docs(tmp_path)
    inject, reference = resolve_onboarding_docs(str(tmp_path), "no_such_phase")
    assert inject is None
    # Every existing doc is referenced when none is injected.
    assert {name for name, _ in reference} == {f"{n}.md" for n in ONBOARDING_DOC_NAMES}


def test_size_guard_demotes_oversized_primary_to_reference(tmp_path: Path) -> None:
    # Make PROJECT_DEFINITION.md exceed the inject byte cap.
    _write_docs(
        tmp_path,
        sizes={"PROJECT_DEFINITION.md": _ONBOARDING_INJECT_BYTE_CAP + 1024},
    )
    inject, reference = resolve_onboarding_docs(str(tmp_path), PhaseName.SPECIFY.value)

    # Too large to inject → demoted to reference-only.
    assert inject is None
    assert "PROJECT_DEFINITION.md" in {name for name, _ in reference}


def test_only_existing_docs_are_referenced(tmp_path: Path) -> None:
    docs_dir = onboarding_docs_dir(str(tmp_path))
    docs_dir.mkdir(parents=True, exist_ok=True)
    # Only two of the four docs exist.
    (docs_dir / "PROJECT_DEFINITION.md").write_text("# pd\nbody", encoding="utf-8")
    (docs_dir / "ARCHITECTURE_MAP.md").write_text("# am\nbody", encoding="utf-8")

    inject, reference = resolve_onboarding_docs(str(tmp_path), PhaseName.PLAN.value)
    # PLAN's primary is ARCHITECTURE_MAP → injected; only PROJECT_DEFINITION left.
    assert inject == str(docs_dir / "ARCHITECTURE_MAP.md")
    assert {name for name, _ in reference} == {"PROJECT_DEFINITION.md"}


# ── build_onboarding_reference ───────────────────────────────────────────


def test_reference_block_empty_when_no_refs() -> None:
    assert build_onboarding_reference([]) == ""


def test_reference_block_lists_paths_in_xml(tmp_path: Path) -> None:
    reference = [
        ("CODING_GUIDELINES.md", str(tmp_path / "CODING_GUIDELINES.md")),
        ("ARCHITECTURE_MAP.md", str(tmp_path / "ARCHITECTURE_MAP.md")),
    ]
    block = build_onboarding_reference(reference)

    assert block.startswith("<onboarding_documentation>")
    assert block.endswith("</onboarding_documentation>")
    for _, path in reference:
        assert path in block
    # Carries the one-line purpose hints.
    assert "conventions" in block
