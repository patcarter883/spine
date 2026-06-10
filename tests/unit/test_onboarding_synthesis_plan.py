"""Tests for the deterministic section skeleton + plan schemas.

Asserts the per-document skeleton shape (design §2.2), the greenfield minimal
plan, and that every section carries non-empty ``fragment_keys`` + ``instruction``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spine.work.onboarding.synthesis_plan import (
    SectionContent,
    SectionEntry,
    SectionPlan,
    SectionPlanSet,
    SectionResult,
    deterministic_section_plan,
    render_section_markdown,
)
from spine.work.onboarding.synthesis_tools import ONBOARDING_DOC_NAMES


def _brownfield_index() -> dict:
    return {
        "mode": "brownfield",
        "tech_stack": ["python"],
        "core_domains": ["spine.work", "spine.agents"],
        "modules": [
            {"name": "spine.work", "path": "spine/work", "role": "work engine", "symbol_count": 40},
            {"name": "spine.agents", "path": "spine/agents", "role": "agents", "symbol_count": 30},
            {"name": "spine.ui", "path": "spine/ui", "role": "ui", "symbol_count": 12},
        ],
        "pattern_categories": ["logging", "config", "error_handling"],
        "edge_counts": {"spine.work": 5, "spine.agents": 3, "spine.ui": 1},
        "totals": {
            "symbol_count": 82,
            "file_count": 20,
            "module_count": 3,
            "pattern_count": 3,
            "edge_count": 9,
        },
        "notes": "",
    }


def _assert_every_section_complete(plan: list[dict]) -> None:
    for section in plan:
        # Validates shape against the schema too.
        sp = SectionPlan(**section)
        assert sp.doc_id in ONBOARDING_DOC_NAMES
        assert sp.fragment_keys, f"empty fragment_keys: {section}"
        assert sp.instruction.strip(), f"empty instruction: {section}"
        assert sp.title.strip()


def test_brownfield_skeleton_shape_per_doc() -> None:
    index = _brownfield_index()
    plan = deterministic_section_plan(index, "brownfield")

    by_doc: dict[str, list[dict]] = {}
    for s in plan:
        by_doc.setdefault(s["doc_id"], []).append(s)

    # All four docs represented.
    assert set(by_doc) == set(ONBOARDING_DOC_NAMES)

    # ARCHITECTURE_MAP: one section per module.
    assert len(by_doc["ARCHITECTURE_MAP"]) == 3
    assert [s["fragment_keys"]["modules"] for s in by_doc["ARCHITECTURE_MAP"]] == [
        ["spine.work"],
        ["spine.agents"],
        ["spine.ui"],
    ]

    # CODING_GUIDELINES: one per pattern category.
    assert len(by_doc["CODING_GUIDELINES"]) == 3
    assert [s["fragment_keys"]["categories"][0] for s in by_doc["CODING_GUIDELINES"]] == [
        "logging",
        "config",
        "error_handling",
    ]

    # PROJECT_DEFINITION: one per core domain.
    assert len(by_doc["PROJECT_DEFINITION"]) == 2
    assert [s["fragment_keys"]["domains"][0] for s in by_doc["PROJECT_DEFINITION"]] == [
        "spine.work",
        "spine.agents",
    ]

    # SPINE_ASSISTANCE_REQUIREMENTS: 1-2 sections.
    assert 1 <= len(by_doc["SPINE_ASSISTANCE_REQUIREMENTS"]) <= 2

    _assert_every_section_complete(plan)


def test_section_orders_are_per_doc_contiguous() -> None:
    plan = deterministic_section_plan(_brownfield_index(), "brownfield")
    by_doc: dict[str, list[int]] = {}
    for s in plan:
        by_doc.setdefault(s["doc_id"], []).append(s["order"])
    for doc_id, orders in by_doc.items():
        assert orders == list(range(len(orders))), f"{doc_id} orders not contiguous: {orders}"


def test_greenfield_minimal_plan() -> None:
    index = {
        "mode": "greenfield",
        "tech_stack": ["python"],
        "core_domains": [],
        "modules": [],
        "pattern_categories": [],
        "edge_counts": {},
        "totals": {"symbol_count": 0, "file_count": 0, "module_count": 0, "pattern_count": 0},
        "notes": "",
    }
    plan = deterministic_section_plan(index, "greenfield")

    # Exactly one section per document.
    assert len(plan) == len(ONBOARDING_DOC_NAMES)
    assert {s["doc_id"] for s in plan} == set(ONBOARDING_DOC_NAMES)
    for s in plan:
        assert s["fragment_keys"].get("greenfield") is True
    _assert_every_section_complete(plan)


def test_brownfield_empty_index_falls_back_to_overview_sections() -> None:
    """A brownfield repo with no modules/patterns/domains still yields complete sections."""
    empty = {
        "mode": "brownfield",
        "tech_stack": [],
        "core_domains": [],
        "modules": [],
        "pattern_categories": [],
        "edge_counts": {},
        "totals": {"symbol_count": 0, "file_count": 0, "module_count": 0, "pattern_count": 0},
        "notes": "",
    }
    plan = deterministic_section_plan(empty, "brownfield")
    assert {s["doc_id"] for s in plan} == set(ONBOARDING_DOC_NAMES)
    _assert_every_section_complete(plan)


def test_large_index_caps_architecture_sections() -> None:
    """One ARCHITECTURE_MAP section per surfaced (already-capped) module."""
    index = _brownfield_index()
    index["modules"] = [
        {"name": f"m{i}", "path": f"m{i}", "role": "r", "symbol_count": 1}
        for i in range(32)
    ] + [{"name": "__other_modules__", "path": "", "role": "tail", "symbol_count": 99}]
    plan = deterministic_section_plan(index, "brownfield")
    arch = [s for s in plan if s["doc_id"] == "ARCHITECTURE_MAP"]
    assert len(arch) == 33  # one per surfaced module incl. the grouped tail
    _assert_every_section_complete(plan)


def test_schemas_roundtrip() -> None:
    plan = deterministic_section_plan(_brownfield_index(), "brownfield")
    plan_set = SectionPlanSet(sections=[SectionPlan(**s) for s in plan])
    assert len(plan_set.sections) == len(plan)

    result = SectionResult(doc_id="ARCHITECTURE_MAP", order=0, markdown="# x", status="ok")
    assert result.status == "ok"
    # Defaults.
    blank = SectionResult(doc_id="CODING_GUIDELINES", order=1)
    assert blank.markdown == ""
    assert blank.status == "ok"


# ── Degenerate-domain merge ─────────────────────────────────────────────────


def test_small_domains_merge_into_supporting_section() -> None:
    """Domains below _MIN_DOMAIN_SYMBOLS (trace 019eaf55: 'app.Console' with 4
    symbols, 'tests.Architecture' with 1) get ONE merged section, not one each."""
    index = _brownfield_index()
    index["core_domains"] = ["spine.work", "app.Console", "tests.Architecture"]
    index["modules"].extend(
        [
            {"name": "app.Console", "path": "app/Console", "role": "console", "symbol_count": 4},
            {"name": "tests.Architecture", "path": "tests/Architecture", "role": "arch tests", "symbol_count": 1},
        ]
    )
    plan = deterministic_section_plan(index, "brownfield")
    pd = [s for s in plan if s["doc_id"] == "PROJECT_DEFINITION"]

    assert [s["title"] for s in pd] == ["Domain: spine.work", "Supporting Domains"]
    merged = pd[-1]
    assert merged["fragment_keys"]["domains"] == ["app.Console", "tests.Architecture"]
    assert "app.Console" in merged["instruction"]
    assert [s["order"] for s in pd] == [0, 1]
    _assert_every_section_complete(plan)


def test_all_small_domains_yield_single_supporting_section() -> None:
    index = _brownfield_index()
    index["core_domains"] = ["tiny.a", "tiny.b"]
    index["modules"] = [
        {"name": "tiny.a", "path": "a", "role": "a", "symbol_count": 1},
        {"name": "tiny.b", "path": "b", "role": "b", "symbol_count": 2},
    ]
    plan = deterministic_section_plan(index, "brownfield")
    pd = [s for s in plan if s["doc_id"] == "PROJECT_DEFINITION"]
    assert len(pd) == 1
    assert pd[0]["title"] == "Supporting Domains"
    assert pd[0]["order"] == 0


def test_domain_missing_from_index_modules_treated_as_small() -> None:
    """A domain whose module fell into the collapsed tail has no symbol count
    in the index — it must merge rather than get a dedicated thin section."""
    index = _brownfield_index()
    index["core_domains"] = ["spine.work", "ghost.domain"]
    plan = deterministic_section_plan(index, "brownfield")
    pd = [s for s in plan if s["doc_id"] == "PROJECT_DEFINITION"]
    assert [s["title"] for s in pd] == ["Domain: spine.work", "Supporting Domains"]
    assert pd[-1]["fragment_keys"]["domains"] == ["ghost.domain"]


# ── SectionContent schema + renderer ────────────────────────────────────────


def test_section_content_requires_nonempty_overview() -> None:
    """The content schema must reject the envelope-only response shape (trace
    019eaf55): no overview and an empty overview are both invalid."""
    with pytest.raises(ValidationError):
        SectionContent()
    with pytest.raises(ValidationError):
        SectionContent(overview="")
    assert "overview" in SectionContent.model_json_schema()["required"]


def test_section_entry_requires_name_and_description() -> None:
    with pytest.raises(ValidationError):
        SectionEntry(name="", description="x")
    with pytest.raises(ValidationError):
        SectionEntry(name="x", description="")


def test_render_section_markdown_full_shape() -> None:
    content = SectionContent(
        overview="The work engine dispatches jobs.",
        entries=[
            SectionEntry(
                name="spine.work",
                path="spine/work",
                description="Owns `submit_work` and the dispatch loop.",
            ),
            SectionEntry(name="spine.agents", description="Agent factories."),
        ],
        notes=["Avoid loading the full manifest.", "  "],
    )
    md = render_section_markdown("Domain: spine.work", content)

    assert md.startswith("## Domain: spine.work")
    assert "The work engine dispatches jobs." in md
    assert "### spine.work (`spine/work`)" in md
    assert "### spine.agents\n" in md  # no path → no parenthetical
    assert "**Notes:**\n- Avoid loading the full manifest." in md
    # The whitespace-only note is dropped.
    assert md.count("- ") == 1


def test_render_section_markdown_prose_only() -> None:
    md = render_section_markdown("Overview", SectionContent(overview="Just prose."))
    assert md == "## Overview\n\nJust prose."


def test_render_section_markdown_skips_blank_entries() -> None:
    content = SectionContent(
        overview="Body.",
        entries=[SectionEntry(name=" ", description="orphan")],
    )
    md = render_section_markdown("T", content)
    assert "###" not in md
    assert "orphan" not in md
