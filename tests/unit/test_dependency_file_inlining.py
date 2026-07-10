"""Dependency-created file sources inlined into the editor payload.

Probe 22 (run f9aac445, parked at 1 open gap): the test slice called
generateName()/generateUuid() on the factory a sibling slice created two
waves earlier — the factory is not in the codebase index (scrubbed as a
phantom reference), so nothing showed the editor its real API and the
editor invented one. Waves are dependency-ordered: the dependency's files
exist on disk when the dependent slice runs, so they are inlined live.
"""

import json
from pathlib import Path

from spine.workflow.subgraphs.implement_subgraph import _dependency_files_body

FACTORY_SRC = (
    "<?php\n\nnamespace Database\\Factories;\n\n"
    "class UnitOfMeasureFactory extends BaseFactory\n{\n"
    "    protected $model = UnitOfMeasure::class;\n"
    "    public function definition(): array { return []; }\n}\n"
)


def _workspace(tmp_path: Path) -> dict:
    plan_dir = tmp_path / ".spine" / "artifacts" / "w" / "plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.json").write_text(json.dumps({
        "feature_slices": [
            {"id": "create-factory",
             "target_files": ["database/factories/UnitOfMeasureFactory.php"]},
            {"id": "test-persistence",
             "target_files": ["tests/Unit/UnitOfMeasureTest.php"],
             "dependencies": ["create-factory"]},
        ],
    }), encoding="utf-8")
    fdir = tmp_path / "database" / "factories"
    fdir.mkdir(parents=True)
    (fdir / "UnitOfMeasureFactory.php").write_text(FACTORY_SRC, encoding="utf-8")
    return {
        "workspace_root": str(tmp_path),
        "plan_path": ".spine/artifacts/w/plan",
    }


def test_dependency_file_inlined_live(tmp_path: Path) -> None:
    state = _workspace(tmp_path)
    body = _dependency_files_body(
        state,
        {"id": "test-persistence",
         "target_files": ["tests/Unit/UnitOfMeasureTest.php"],
         "dependencies": ["create-factory"]},
        str(tmp_path),
    )
    assert "UnitOfMeasureFactory extends BaseFactory" in body
    assert "created by dependency slice 'create-factory'" in body
    assert "do NOT invent" in body


def test_no_dependencies_no_block(tmp_path: Path) -> None:
    state = _workspace(tmp_path)
    body = _dependency_files_body(
        state, {"id": "create-factory", "dependencies": []}, str(tmp_path)
    )
    assert body == ""


def test_own_target_files_excluded(tmp_path: Path) -> None:
    """A same-file dependency's file is not duplicated — _target_files_body
    already inlines the slice's own targets."""
    state = _workspace(tmp_path)
    body = _dependency_files_body(
        state,
        {"id": "x",
         "target_files": ["database/factories/UnitOfMeasureFactory.php"],
         "dependencies": ["create-factory"]},
        str(tmp_path),
    )
    assert body == ""


def test_missing_dependency_file_skipped(tmp_path: Path) -> None:
    """A dependency whose file was never written degrades to no block."""
    state = _workspace(tmp_path)
    (tmp_path / "database/factories/UnitOfMeasureFactory.php").unlink()
    body = _dependency_files_body(
        state,
        {"id": "test-persistence", "dependencies": ["create-factory"]},
        str(tmp_path),
    )
    assert body == ""


def test_oversized_dependency_file_truncated(tmp_path: Path) -> None:
    state = _workspace(tmp_path)
    f = tmp_path / "database/factories/UnitOfMeasureFactory.php"
    f.write_text("<?php\n" + "// pad\n" * 2000, encoding="utf-8")
    body = _dependency_files_body(
        state,
        {"id": "test-persistence", "dependencies": ["create-factory"]},
        str(tmp_path),
    )
    assert "… (truncated)" in body
    assert len(body) < 5000


def test_no_plan_json_fails_open(tmp_path: Path) -> None:
    body = _dependency_files_body(
        {"workspace_root": str(tmp_path), "plan_path": "nope"},
        {"id": "x", "dependencies": ["create-factory"]},
        str(tmp_path),
    )
    assert body == ""
