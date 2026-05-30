"""Tests for the deterministic section skeleton + plan schemas.

Asserts the per-document skeleton shape (design §2.2), the greenfield minimal
plan, and that every section carries non-empty ``fragment_keys`` + ``instruction``.
"""

from __future__ import annotations

from spine.work.onboarding.synthesis_plan import (
    SectionPlan,
    SectionPlanSet,
    SectionResult,
    deterministic_section_plan,
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
