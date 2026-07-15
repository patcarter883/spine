"""Dual-facet recall: structural query + diversity-capped merge.

Work d8bc459c: recall queried with the raw schema-heavy objective returned
6/10 unrelated migrations/seeders (plus triple-FarmPolicy crowding), while
the same index queried with DDL noise stripped + structural terms returned
the exact farm-scoped CRUD exemplars. pre_research_gate now recalls both
facets and merges with per-file/per-directory caps.
"""

from __future__ import annotations

from spine.workflow.subgraphs.exploration_subgraph import (
    _merge_recall_facets,
    _structural_recall_query,
)


def _hit(path: str, symbol: str) -> dict:
    return {"file_path": path, "symbol_name": symbol, "enriched_summary": "s"}


class TestStructuralQuery:
    def test_keeps_intent_drops_schema_detail(self):
        q = _structural_recall_query(
            "Add farm-scoped RainGauge entities to app/Domain/Farm (standard "
            "slice). RainGauge: uuid PK, farm_id, code nullable, amount_mm "
            "decimal(8,2), UNIQUE (rain_gauge_id, date) constraint. "
            "CRUD under /farms/{farm_id}/rain-gauges. "
            "Prerequisite: migration foundation.",
            "Backend/API",
        )
        low = q.lower()
        # Intent sentence survives; schema-detail sentences are dropped.
        assert "raingauge entities" in low
        assert "amount_mm" not in low
        assert "decimal" not in low
        assert "nullable" not in low
        # Route/path mentions from ANY sentence survive.
        assert "/farms/{farm_id}/rain-gauges" in low
        assert "app/domain/farm" in low

    def test_appends_category_terms(self):
        q = _structural_recall_query("Add RainGauge entities.", "Backend/API")
        assert "controller" in q and "policy" in q and "routes" in q

    def test_unknown_category_gets_default_terms(self):
        q = _structural_recall_query("Add RainGauge entities.", "Generic")
        assert "controller" in q and "test" in q

    def test_none_category(self):
        assert "controller" in _structural_recall_query("x", None)


class TestMergeFacets:
    def test_primary_leads_and_dedupes(self):
        primary = [_hit("app/A.php", "A"), _hit("app/B.php", "B")]
        secondary = [_hit("app/A.php", "A"), _hit("app/C.php", "C")]
        out = _merge_recall_facets(primary, secondary, k=10)
        assert [(h["file_path"], h["symbol_name"]) for h in out] == [
            ("app/A.php", "A"), ("app/B.php", "B"), ("app/C.php", "C"),
        ]

    def test_per_directory_cap_blocks_migration_monopoly(self):
        migrations = [
            _hit(f"database/migrations/2026_{i}.php", "up") for i in range(8)
        ]
        exemplars = [
            _hit("app/Http/Controllers/FarmController.php", "FarmController"),
            _hit("app/Domain/Farm/Policies/FarmPolicy.php", "FarmPolicy"),
        ]
        # k=5 fits exactly within the caps (2 exemplars + 3 migrations) so
        # the uncapped backfill pass stays out of this assertion.
        out = _merge_recall_facets(exemplars, migrations, k=5, per_dir=3)
        migration_count = sum(
            1 for h in out if h["file_path"].startswith("database/migrations")
        )
        assert migration_count <= 3
        assert any("FarmController" == h["symbol_name"] for h in out)

    def test_per_file_cap_blocks_symbol_crowding(self):
        crowd = [
            _hit("app/Policies/FarmPolicy.php", s)
            for s in ("FarmPolicy", "FarmPolicy.view", "FarmPolicy.create", "FarmPolicy.update")
        ]
        out = _merge_recall_facets(crowd, [], k=10, per_file=2)
        assert len([h for h in out if h["file_path"] == "app/Policies/FarmPolicy.php"]) >= 2
        # capped set first, backfill may re-add — but the CAPPED window (first
        # pass) holds 2; overall dedup keeps 4 unique max
        assert len(out) <= 4

    def test_backfill_when_caps_starve(self):
        same_dir = [_hit(f"app/X/{i}.php", f"S{i}") for i in range(6)]
        out = _merge_recall_facets(same_dir, [], k=5, per_dir=2)
        assert len(out) == 5  # caps allow 2, backfill tops up to k

    def test_k_bound_respected(self):
        many = [_hit(f"app/{i}/f.php", f"S{i}") for i in range(30)]
        assert len(_merge_recall_facets(many, many, k=10)) == 10
