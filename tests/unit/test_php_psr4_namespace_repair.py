"""PSR-4 namespace repair at the write exit (agripath probe 7).

PHP slices invented a namespace per slice — the model in
App\\Domain\\Farm\\Models, the factory generating
App\\Domain\\Project\\Models\\UnitOfMeasure, the test importing
App\\Models\\UnitOfMeasure. PSR-4 makes both directions mechanical: the
file path dictates the declaration, and a dangling import resolves to
wherever the class's file actually lives.
"""

from __future__ import annotations

import json
from pathlib import Path

from spine.agents.tools.read_edit_lint import _fix_php_namespaces


def _ws(tmp_path: Path) -> Path:
    (tmp_path / "composer.json").write_text(json.dumps({
        "autoload": {"psr-4": {
            "App\\": "app/",
            "Database\\Factories\\": "database/factories/",
        }},
        "autoload-dev": {"psr-4": {"Tests\\": "tests/"}},
    }), encoding="utf-8")
    d = tmp_path / "app/Domain/Farm/Models"
    d.mkdir(parents=True)
    (d / "UnitOfMeasure.php").write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\nclass UnitOfMeasure {}\n",
        encoding="utf-8",
    )
    return tmp_path


def test_declaration_and_dangling_import_are_repaired(tmp_path):
    ws = _ws(tmp_path)
    fdir = ws / "database/factories"
    fdir.mkdir(parents=True)
    f = fdir / "UnitOfMeasureFactory.php"
    f.write_text(
        "<?php\n\nnamespace App\\Factories;\n\n"
        "use App\\Domain\\Project\\Models\\UnitOfMeasure;\n\n"
        "class UnitOfMeasureFactory {}\n",
        encoding="utf-8",
    )
    changed = _fix_php_namespaces(f, str(ws))
    text = f.read_text(encoding="utf-8")
    assert "namespace Database\\Factories;" in text
    assert "use App\\Domain\\Farm\\Models\\UnitOfMeasure;" in text
    assert "Project" not in text
    assert changed and len(changed) == 2


def test_correct_file_is_untouched(tmp_path):
    ws = _ws(tmp_path)
    tdir = ws / "tests/Unit"
    tdir.mkdir(parents=True)
    f = tdir / "UnitOfMeasureTest.php"
    original = (
        "<?php\n\nnamespace Tests\\Unit;\n\n"
        "use App\\Domain\\Farm\\Models\\UnitOfMeasure;\n\n"
        "class UnitOfMeasureTest {}\n"
    )
    f.write_text(original, encoding="utf-8")
    assert _fix_php_namespaces(f, str(ws)) is None
    assert f.read_text(encoding="utf-8") == original


def test_vendor_imports_are_left_alone(tmp_path):
    ws = _ws(tmp_path)
    d = ws / "app"
    f = d / "Service.php"
    f.write_text(
        "<?php\n\nnamespace App;\n\n"
        "use Illuminate\\Support\\Facades\\DB;\n\n"
        "class Service {}\n",
        encoding="utf-8",
    )
    # Illuminate\* is not PSR-4-mapped in composer.json — never touched.
    assert _fix_php_namespaces(f, str(ws)) is None


def test_ambiguous_basename_is_not_rewritten(tmp_path):
    ws = _ws(tmp_path)
    other = ws / "app/Domain/Project/Models"
    other.mkdir(parents=True)
    (other / "UnitOfMeasure.php").write_text(
        "<?php\n\nnamespace App\\Domain\\Project\\Models;\n\nclass UnitOfMeasure {}\n",
        encoding="utf-8",
    )
    f = ws / "app" / "Uses.php"
    f.write_text(
        "<?php\n\nnamespace App;\n\nuse App\\Wrong\\UnitOfMeasure;\n\nclass Uses {}\n",
        encoding="utf-8",
    )
    changed = _fix_php_namespaces(f, str(ws)) or []
    # Two candidates exist — ambiguity means no import rewrite.
    assert not any("use " in c for c in changed)
