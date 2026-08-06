"""Verdict sanity gate and Laravel migration guard (agripath probe 27, cc9c2611).

That run parked needs_review flat at 3 VERIFIED / 3 NOT_VERIFIED across FOUR
gap cycles. Two distinct defects:

* the ``database-migrations-farm`` judge returned ONE self-contradictory reason
  — "The file is named ContactController but the class is actually named
  ContactController in the file." — for 14 of 15 criteria, seven of which
  belonged to a different slice. A slice is never re-judged within a cycle, so
  the fabrication became the rework loop's instruction for three more cycles
  and inflated ``_total_gap_count``'s convergence arithmetic.
* the contacts migration used ``foreignId('farm_id')->constrained('farms')``
  where ``farms.id`` is a uuid — a hard Postgres DDL error (SQLSTATE 42804)
  that ``php -l`` and the ``php_syntax`` landing gate are both blind to. Four
  rework cycles never fixed the one-token error.
"""

from __future__ import annotations

from pathlib import Path

from spine.agents.tools.php_migrations import (
    check_migration_filename,
    check_migration_ordering,
    migration_pk_types,
    repair_foreign_id,
)
from spine.workflow.verdict_gate import (
    is_self_contradictory,
    reject_unreliable_verdict,
)

# The probe-27 verdict, verbatim.
PROBE27_DETAIL = (
    "The file is named ContactController but the class is actually named "
    "ContactController in the file."
)


def _verdict(detail: str, n: int) -> dict:
    return {
        "verdict": "NOT_VERIFIED",
        "gaps": [detail],
        "recommendations": [detail],
        "checklist": [
            {"criterion": f"criterion {i}", "passed": False, "detail": detail}
            for i in range(n)
        ],
    }


# ── verdict gate ───────────────────────────────────────────────────────────


def test_detects_the_probe27_tautology():
    assert is_self_contradictory(PROBE27_DETAIL)


def test_a_genuine_contrast_is_not_flagged():
    assert not is_self_contradictory(
        "The file is named ContactController but the class is actually named "
        "FooBar in the file."
    )


def test_unrelated_prose_is_not_flagged():
    assert not is_self_contradictory("No route registration found in routes/api.php.")


def test_probe27_details_are_rewritten():
    v = _verdict(PROBE27_DETAIL, 15)
    rewritten = reject_unreliable_verdict(v, "cc9c2611", "database-migrations-farm")
    assert rewritten == 15
    assert all(c["detail"].startswith("UNRELIABLE VERDICT") for c in v["checklist"])


def test_failing_count_is_preserved_for_the_ratchet():
    """The gate must not change how many criteria are failing.

    compose._total_gap_count counts failing entries and feeds the best-state
    ratchet. An earlier version collapsed 15 -> 1, which made the FIRST,
    unfixed cycle the ratchet's "best" state: every honest later cycle then
    scored as a regression and restore_best deleted the real fixes from disk.
    Measured against the production mapper, totals went [1, 6, 4] with the run
    parked on cycle-1 code, versus [15, 6, 4, 2] converging with the gate off.
    """
    v = _verdict(PROBE27_DETAIL, 15)
    reject_unreliable_verdict(v)
    assert len(v["checklist"]) == 15
    assert sum(1 for c in v["checklist"] if not c["passed"]) == 15


def test_gate_never_promotes():
    v = _verdict(PROBE27_DETAIL, 15)
    reject_unreliable_verdict(v)
    assert v["verdict"] == "NOT_VERIFIED"
    assert all(not c["passed"] for c in v["checklist"])


def test_genuine_concurrent_findings_are_untouched():
    """Only entries carrying the boilerplate are rewritten."""
    v = _verdict(PROBE27_DETAIL, 10)
    v["checklist"].append(
        {"criterion": "route", "passed": False, "detail": "No route in routes/api.php"}
    )
    v["checklist"].append({"criterion": "ok", "passed": True, "detail": "present"})
    reject_unreliable_verdict(v)
    details = [c["detail"] for c in v["checklist"]]
    assert "No route in routes/api.php" in details
    assert "present" in details


def test_negated_contrast_is_coherent_and_not_flagged():
    """'a column named X exists but there is no property named X' is a real
    finding — seven realistic Laravel findings were wrongly flagged before."""
    for detail in (
        "The migration creates a column named farm_id but the model has no "
        "property named farm_id.",
        "A factory named ContactFactory exists but no test named "
        "ContactFactory was found.",
    ):
        assert not is_self_contradictory(detail)


def test_healthy_verdict_is_untouched():
    v = {
        "verdict": "NOT_VERIFIED",
        "gaps": ["missing route"],
        "checklist": [
            {"criterion": "a", "passed": False, "detail": "No route in routes/api.php"},
            {"criterion": "b", "passed": False, "detail": "foreignId used, needs uuid"},
            {"criterion": "c", "passed": True, "detail": "present"},
        ],
    }
    before = list(v["checklist"])
    assert reject_unreliable_verdict(v) == 0
    assert v["checklist"] == before


def test_repeated_but_coherent_reason_is_untouched():
    """Two criteria can legitimately share one real cause — repetition alone
    is not evidence of a fabricated verdict."""
    v = _verdict("The pest suite failed to run.", 6)
    assert reject_unreliable_verdict(v) == 0
    assert len(v["checklist"]) == 6


def test_short_checklist_is_untouched():
    v = _verdict(PROBE27_DETAIL, 2)
    assert reject_unreliable_verdict(v) == 0


def test_missing_checklist_is_safe():
    assert reject_unreliable_verdict({"verdict": "NOT_VERIFIED"}) == 0
    assert reject_unreliable_verdict({"checklist": "not a list"}) == 0


# ── migration filename + ordering ──────────────────────────────────────────


def test_impossible_date_is_flagged():
    assert check_migration_filename("2026_00_00_000000_create_units_of_measure_table.php")


def test_slice_id_prefix_is_flagged():
    """Probe 27 wrote '1-2026_05_01_...' — a micro-slice index leaking into a path."""
    assert check_migration_filename("1-2026_05_01_000001_create_asset_types_table.php")


def test_wellformed_name_passes():
    assert check_migration_filename("2026_05_01_000001_create_asset_types_table.php") is None


def test_laravel_framework_sentinel_is_whitelisted():
    """0001_01_01_* is Laravel's own deliberate first-sorting prefix."""
    assert check_migration_filename("0001_01_01_000000_create_users_table.php") is None


def test_stale_but_valid_timestamp_is_flagged_by_ordering():
    """The killed run emitted 2024_01_01_* — a valid date that sorts first."""
    siblings = [
        "2026_04_21_091238_create_farms_table.php",
        "2024_01_01_000001_create_asset_types_table.php",
    ]
    assert check_migration_ordering("2024_01_01_000001_create_asset_types_table.php", siblings)


def test_newest_migration_passes_ordering():
    siblings = [
        "2026_04_21_091238_create_farms_table.php",
        "2026_09_01_000002_create_contacts_table.php",
    ]
    assert check_migration_ordering("2026_09_01_000002_create_contacts_table.php", siblings) is None


def test_ordering_ignores_the_framework_sentinel():
    siblings = ["0001_01_01_000000_create_users_table.php", "2026_04_21_091238_x.php"]
    assert check_migration_ordering("0001_01_01_000000_create_users_table.php", siblings) is None


# ── foreignId / uuid PK repair ─────────────────────────────────────────────


def _migrations(tmp_path: Path) -> Path:
    d = tmp_path / "database" / "migrations"
    d.mkdir(parents=True)
    (d / "2026_04_21_091238_create_farms_table.php").write_text(
        "<?php\nSchema::create('farms', function (Blueprint $table) {\n"
        "    $table->uuid('id')->primary();\n});\n",
        encoding="utf-8",
    )
    (d / "2026_04_21_091239_create_jobs_table.php").write_text(
        "<?php\nSchema::create('jobs', function (Blueprint $table) {\n"
        "    $table->id();\n});\n",
        encoding="utf-8",
    )
    return d


def test_pk_types_resolved_from_sibling_migrations(tmp_path):
    pk = migration_pk_types(_migrations(tmp_path))
    assert pk["farms"] == "uuid"
    assert pk["jobs"] == "bigint"


def test_foreign_id_against_uuid_pk_is_repaired(tmp_path):
    pk = migration_pk_types(_migrations(tmp_path))
    src = "$table->foreignId('farm_id')->constrained('farms')->onDelete('cascade');"
    out, changes = repair_foreign_id(src, pk)
    assert "foreignUuid('farm_id')" in out
    assert len(changes) == 1


def test_foreign_id_against_bigint_pk_is_left_alone(tmp_path):
    """A genuine bigint FK must never be rewritten."""
    pk = migration_pk_types(_migrations(tmp_path))
    src = "$table->foreignId('job_id')->constrained('jobs');"
    out, changes = repair_foreign_id(src, pk)
    assert out == src
    assert changes == []


def test_foreign_id_without_constrained_is_left_alone(tmp_path):
    """Nothing proves the target table, so nothing can be concluded."""
    pk = migration_pk_types(_migrations(tmp_path))
    src = "$table->foreignId('farm_id');"
    out, changes = repair_foreign_id(src, pk)
    assert out == src
    assert changes == []


def test_unknown_target_table_is_left_alone(tmp_path):
    pk = migration_pk_types(_migrations(tmp_path))
    src = "$table->foreignId('thing_id')->constrained('things');"
    out, changes = repair_foreign_id(src, pk)
    assert out == src
    assert changes == []
