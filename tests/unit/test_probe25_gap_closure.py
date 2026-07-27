"""Regressions for the six defects probe 25 exposed (agripath UnitOfMeasure).

Probe 25 landed (commit dca6ebc) but shipped a test asserting
``not->toBeEmpty()`` where the criterion demanded a value match, plus a
boilerplate ``test('that true is true')``. Reconstructing that run showed the
defects were independent, and none of them was a model-quality problem:

1. research never read the file the task named (`Farm.php`) — 9 topics,
   17 files, and the exemplar's `use FarmScoped;` reached no phase;
2. a whole re-plan (9m22s of 25m34s) burned on a `create` provides collision;
3. the no-tool editor never received that exemplar either;
4. verify PASSED an equality criterion citing a green check run as evidence;
5. the gap plan's enumerated fixes were never applied and nothing noticed;
6. research cited `.planning/**` prose as if it were the codebase.
"""

from __future__ import annotations

import json

import pytest


# ── 1 + 3. Task-named path pre-read ──────────────────────────────────────────


class TestTaskNamedPaths:
    def test_extracts_paths_from_prose(self) -> None:
        from spine.workflow.task_paths import extract_task_paths

        found = extract_task_paths(
            "Add a model to app/Domain/Farm/Models: follow the conventions of "
            "the existing `app/Domain/Farm/Models/Farm.php` model and the "
            "migrations in database/migrations."
        )
        assert "app/Domain/Farm/Models/Farm.php" in found
        assert "database/migrations" in found

    def test_strips_trailing_punctuation_and_backticks(self) -> None:
        from spine.workflow.task_paths import extract_task_paths

        assert extract_task_paths("see `spine/agents/x.py`, then stop.") == [
            "spine/agents/x.py"
        ]

    def test_reads_the_named_file_verbatim(self, tmp_path) -> None:
        from spine.workflow.task_paths import read_task_named_sources

        target = tmp_path / "app" / "Models"
        target.mkdir(parents=True)
        (target / "Farm.php").write_text("<?php\nuse FarmScoped;\n")

        got = read_task_named_sources(
            "follow app/Models/Farm.php please", tmp_path
        )
        assert [e["path"] for e in got] == ["app/Models/Farm.php"]
        # The trait is the whole point: a 240-char prose summary of this file
        # is what the old path delivered, and it dropped exactly this line.
        assert "use FarmScoped;" in got[0]["source"]

    def test_directory_token_yields_its_code_children(self, tmp_path) -> None:
        from spine.workflow.task_paths import read_task_named_sources

        d = tmp_path / "database" / "migrations"
        d.mkdir(parents=True)
        (d / "0001_create.php").write_text("<?php // one")
        (d / "notes.md").write_text("# not code")

        got = read_task_named_sources("see database/migrations", tmp_path)
        assert [e["path"] for e in got] == ["database/migrations/0001_create.php"]

    def test_absent_path_is_silent(self, tmp_path) -> None:
        from spine.workflow.task_paths import read_task_named_sources

        assert read_task_named_sources("see app/Nope.php", tmp_path) == []

    def test_never_escapes_the_workspace(self, tmp_path) -> None:
        from spine.workflow.task_paths import read_task_named_sources

        (tmp_path.parent / "secret.py").write_text("token = 1")
        assert read_task_named_sources("read ../secret.py", tmp_path) == []

    def test_planning_prose_is_not_an_exemplar(self, tmp_path) -> None:
        from spine.workflow.task_paths import read_task_named_sources

        d = tmp_path / ".planning"
        d.mkdir()
        (d / "PLAN.py").write_text("# intent, not implementation")
        assert read_task_named_sources("see .planning/PLAN.py", tmp_path) == []

    def test_render_is_empty_without_entries(self) -> None:
        from spine.workflow.task_paths import render_task_named_sources

        assert render_task_named_sources([]) == ""

    def test_render_marks_source_as_ground_truth(self) -> None:
        from spine.workflow.task_paths import render_task_named_sources

        out = render_task_named_sources(
            [{"path": "a/B.php", "source": "<?php", "truncated": False}]
        )
        assert "a/B.php" in out and "ground truth" in out


# ── 2. Generic provides verbs must not cost a re-plan ────────────────────────


class TestProvidesOwnerContext:
    @pytest.fixture()
    def index(self, monkeypatch: pytest.MonkeyPatch):
        def _install(symbols: dict[str, list[str]]) -> None:
            monkeypatch.setattr(
                "spine.workflow.plan_reference_gate._symbol_exists_in_index",
                lambda db, sym: False,
            )
            monkeypatch.setattr(
                "spine.workflow.plan_reference_gate._find_symbol_files",
                lambda db, name: [
                    fp
                    for fp, syms in symbols.items()
                    if any(c == name or c.endswith("." + name) for c in syms)
                ],
            )
        return _install

    def test_bare_verb_in_provides_is_not_a_collision(self, index) -> None:
        """Probe 25's 9m22s re-plan. `create` exists in every codebase."""
        index({"database/migrations/old.php": ["create"]})
        plan = {
            "feature_slices": [
                {
                    "id": "create-units-of-measure-migration",
                    "target_files": ["database/migrations/new.php"],
                    "provides": ["create"],
                }
            ]
        }
        from spine.workflow.plan_reference_gate import check_reference_symbols

        assert check_reference_symbols(plan, None, None, db_path="db") is None

    def test_path_qualified_existing_symbol_still_flags(self, index) -> None:
        """Run 019f20e0 must keep firing — the qualifier makes it checkable."""
        index({"spine/ui_api/api.py": ["UIApi.get_providers"]})
        plan = {
            "feature_slices": [
                {
                    "id": "api-slice",
                    "provides": ["spine/ui_api/api.py:get_providers"],
                }
            ]
        }
        from spine.workflow.plan_reference_gate import check_reference_symbols

        out = check_reference_symbols(plan, None, None, db_path="db")
        assert out is not None
        assert "ALREADY EXISTS" in out["reason"]

    def test_class_qualified_existing_symbol_still_flags(self, index) -> None:
        index({"app/Http/FileController.php": ["FileController.store"]})
        plan = {
            "feature_slices": [
                {"id": "s", "provides": ["FileController::store"]}
            ]
        }
        from spine.workflow.plan_reference_gate import check_reference_symbols

        out = check_reference_symbols(plan, None, None, db_path="db")
        assert out is not None and "ALREADY EXISTS" in out["reason"]


# ── 4. A green check run is not evidence of what was asserted ────────────────

# Verbatim from the two real runs of the same task on the same model.
_SPINE_LANDED_TEST = """<?php
use App\\Domain\\Farm\\Models\\UnitOfMeasure;
use Database\\Factories\\UnitOfMeasureFactory;

test('that true is true', function () {
    expect(true)->toBeTrue();
});

test('unit of measure factory creates valid unit', function () {
    $factory = new UnitOfMeasureFactory();
    $unitOfMeasure = $factory->make();

    expect($unitOfMeasure->name)->not->toBeEmpty();
    expect($unitOfMeasure->abbreviation)->not->toBeEmpty();
});
"""

_STRONG_TEST = """<?php
use App\\Domain\\Farm\\Models\\UnitOfMeasure;

it('round-trips name and abbreviation', function () {
    $unit = UnitOfMeasure::factory()->create();
    expect(UnitOfMeasure::find($unit->id)->name)->toBe($unit->name);
});
"""

_CRITERION = (
    "The test asserts that the `name` of the created UnitOfMeasure matches "
    "the expected factory value."
)


class TestAssertionGate:
    def test_recognises_equality_criteria(self) -> None:
        from spine.workflow.assertion_gate import is_equality_criterion

        assert is_equality_criterion(_CRITERION)
        assert is_equality_criterion("asserts the data round-trips")
        assert not is_equality_criterion("The file exists and is valid Pest.")

    def test_boilerplate_true_is_true_is_not_an_equality_assertion(self) -> None:
        """The landed file contained `expect(true)->toBeTrue()`; admitting
        that as a comparison would have let the real defect through."""
        from spine.workflow.assertion_gate import has_equality_assertion

        assert not has_equality_assertion(_SPINE_LANDED_TEST, ".php")
        assert has_equality_assertion(_STRONG_TEST, ".php")

    def test_unknown_language_never_demotes(self) -> None:
        from spine.workflow.assertion_gate import has_equality_assertion

        assert has_equality_assertion("whatever", ".rb")

    def _slice(self, tmp_path, body: str):
        p = tmp_path / "tests" / "Unit"
        p.mkdir(parents=True)
        (p / "UnitOfMeasureTest.php").write_text(body)
        return ["tests/Unit/UnitOfMeasureTest.php"]

    def test_demotes_the_probe25_verdict(self, tmp_path) -> None:
        from spine.workflow.assertion_gate import (
            demote_unsupported_equality_criteria,
        )

        targets = self._slice(tmp_path, _SPINE_LANDED_TEST)
        # The verdict exactly as probe 25's final verify recorded it.
        result = {
            "verdict": "VERIFIED",
            "checklist": [
                {
                    "criterion": _CRITERION,
                    "passed": True,
                    "detail": (
                        "The automated checks (pest_changed_tests) passed, "
                        "indicating the generated value was non-empty."
                    ),
                }
            ],
            "gaps": [],
        }
        n = demote_unsupported_equality_criteria(result, targets, tmp_path)
        assert n == 1
        assert result["verdict"] == "NOT_VERIFIED"
        assert result["checklist"][0]["passed"] is False
        assert result["gaps"]
        assert result["recommendations"]

    def test_leaves_a_genuinely_strong_test_alone(self, tmp_path) -> None:
        from spine.workflow.assertion_gate import (
            demote_unsupported_equality_criteria,
        )

        targets = self._slice(tmp_path, _STRONG_TEST)
        result = {
            "verdict": "VERIFIED",
            "checklist": [{"criterion": _CRITERION, "passed": True, "detail": "ok"}],
            "gaps": [],
        }
        assert demote_unsupported_equality_criteria(result, targets, tmp_path) == 0
        assert result["verdict"] == "VERIFIED"

    def test_ignores_slices_with_no_test_files(self, tmp_path) -> None:
        from spine.workflow.assertion_gate import (
            demote_unsupported_equality_criteria,
        )

        (tmp_path / "Model.php").write_text("<?php class M {}")
        result = {
            "verdict": "VERIFIED",
            "checklist": [{"criterion": _CRITERION, "passed": True, "detail": ""}],
            "gaps": [],
        }
        assert demote_unsupported_equality_criteria(
            result, ["Model.php"], tmp_path
        ) == 0
        assert result["verdict"] == "VERIFIED"

    def test_already_failed_criteria_are_untouched(self, tmp_path) -> None:
        from spine.workflow.assertion_gate import (
            demote_unsupported_equality_criteria,
        )

        targets = self._slice(tmp_path, _SPINE_LANDED_TEST)
        result = {
            "verdict": "NOT_VERIFIED",
            "checklist": [{"criterion": _CRITERION, "passed": False, "detail": "x"}],
            "gaps": ["already known"],
        }
        assert demote_unsupported_equality_criteria(result, targets, tmp_path) == 0


# ── 5. An un-applied gap plan cannot be a converged cycle ────────────────────


class TestUnappliedGapPlan:
    def _write_gap_plan(self, tmp_path, work_id: str, files: list[str]) -> None:
        from spine.agents.artifacts import artifact_path
        from spine.models.enums import PhaseName

        d = tmp_path / artifact_path(work_id, PhaseName.GAP_PLAN.value)
        d.mkdir(parents=True, exist_ok=True)
        (d / "gap_plan.json").write_text(
            json.dumps(
                {
                    "summary": "fix the test",
                    "remediation_items": [
                        {
                            "slice_id": "test-units-of-measure",
                            "failures": ["uses make() not create()"],
                            "root_cause": "no persistence",
                            "fixes": [
                                {
                                    "file_path": f,
                                    "issue_description": "make() not create()",
                                    "suggested_fix": "use create()",
                                }
                                for f in files
                            ],
                        }
                    ],
                }
            )
        )

    def test_verified_over_an_untouched_gap_plan_is_refused(self, tmp_path) -> None:
        """Probe 25's final cycle: perfect gap plan, zero files written,
        verify flips the criterion to PASSED, run lands with the defect."""
        from spine.workflow.compose import _verify_result_mapper

        self._write_gap_plan(tmp_path, "w1", ["tests/Unit/UnitOfMeasureTest.php"])
        parent = {
            "work_id": "w1",
            "workspace_root": str(tmp_path),
            "verify_attempts": 2,
            "files_written": [],  # the rework wrote nothing
        }
        out = _verify_result_mapper(
            {"phase_status": "success", "verification_findings": []}, parent
        )
        assert out["verification_passed"] is False
        assert out["verify_completed"] is False
        assert any(
            f.get("slice_name") == "gap-plan-application"
            for f in out["verification_findings"]
        )

    def test_applied_gap_plan_passes_through(self, tmp_path) -> None:
        from spine.workflow.compose import _verify_result_mapper

        self._write_gap_plan(tmp_path, "w2", ["tests/Unit/UnitOfMeasureTest.php"])
        parent = {
            "work_id": "w2",
            "workspace_root": str(tmp_path),
            "verify_attempts": 2,
            "files_written": ["tests/Unit/UnitOfMeasureTest.php"],
        }
        out = _verify_result_mapper(
            {"phase_status": "success", "verification_findings": []}, parent
        )
        assert out["verification_passed"] is True

    def test_first_pass_has_no_gap_plan_to_apply(self, tmp_path) -> None:
        """verify_attempts == 0 ⇒ no rework happened ⇒ nothing to check."""
        from spine.workflow.compose import _verify_result_mapper

        parent = {
            "work_id": "w3",
            "workspace_root": str(tmp_path),
            "verify_attempts": 0,
            "files_written": [],
        }
        out = _verify_result_mapper(
            {"phase_status": "success", "verification_findings": []}, parent
        )
        assert out["verification_passed"] is True


# ── 6. Planning prose is not the codebase ────────────────────────────────────


class TestResearchExclusions:
    def _tree(self, tmp_path):
        (tmp_path / ".planning" / "phases").mkdir(parents=True)
        (tmp_path / ".planning" / "phases" / "01-PLAN.md").write_text(
            "we will add a UnitOfMeasure with uuid primary keys"
        )
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "Farm.php").write_text("<?php // UnitOfMeasure real code")
        return tmp_path

    def _search(self, tmp_path, **kw):
        from spine.agents.plan_tools import SearchCodebaseTool

        tool = SearchCodebaseTool(workspace_root=str(tmp_path))
        return tool._run(queries=["UnitOfMeasure"], **kw)

    def test_walk_branch_skips_planning_prose(self, tmp_path) -> None:
        self._tree(tmp_path)
        out = self._search(tmp_path)
        assert "01-PLAN.md" not in out
        assert "Farm.php" in out

    def test_glob_branch_also_skips_planning_prose(self, tmp_path) -> None:
        """The pattern branch previously bypassed the skip list entirely."""
        self._tree(tmp_path)
        out = self._search(tmp_path, file_patterns=["**/*.md", "**/*.php"])
        assert "01-PLAN.md" not in out

    def test_defaults_always_hold(self) -> None:
        from spine.agents import plan_tools

        skips = plan_tools._research_skip_dirs()
        assert ".planning" in skips
        assert "node_modules" in skips
        assert "vendor" in skips

    def test_php_is_searchable(self) -> None:
        """Absent until 2026-07-27: on a Laravel repo the walk branch could
        return no source file at all, leaving prose as the only candidate."""
        from spine.agents.plan_tools import _CODE_EXTENSIONS

        for suffix in (".php", ".rb", ".go", ".rs", ".java", ".cs"):
            assert suffix in _CODE_EXTENSIONS
