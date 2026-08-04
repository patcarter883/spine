"""PHP FQCN stutter repair and auto-import (agripath probe 26, run 3a7a3597).

That run parked ``needs_review`` with 3 of 4 slices never implemented. The
model emitted PHP FQCNs into JSON string fields with separators stripped and a
tail segment stuttered::

    App\\Infrastructure\\Models\\Traits\\FarmScoped
        -> "AppInfrastructureModelsTraitsFarmScopedModelsTraitsFarmScoped"

``_scrub_phantom_refs`` dropped those as not-in-index, which cost the editor
the list telling it what to import — so it wrote ``use FarmScoped;`` into a
class body with no import statement, a PHP fatal. The same stutter corrupted a
gap-plan target path (``_00_00_00_000000_`` for ``_00_00_000000_``), which made
the P6 guard report that the rework had written nothing.

The corruption is generation variance — the FIRST plan of that same run emitted
the identical FQCNs intact — so these are deterministic repairs, not prompts.
"""

from __future__ import annotations

import json
from pathlib import Path

from spine.agents.gap_plan_tools import _repair_fix_path
from spine.agents.tools.php_fqcn import (
    repair_mangled_fqcn,
    resolve_class_fqcn,
)
from spine.agents.tools.read_edit_lint import _php_auto_import


def _ws(tmp_path: Path) -> Path:
    """A PHP workspace with PSR-4 autoload, first-party classes and a classmap."""
    (tmp_path / "composer.json").write_text(
        json.dumps(
            {
                "autoload": {
                    "psr-4": {
                        "App\\": "app/",
                        "Database\\Factories\\": "database/factories/",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    traits = tmp_path / "app" / "Infrastructure" / "Models" / "Traits"
    traits.mkdir(parents=True)
    (traits / "FarmScoped.php").write_text(
        "<?php\n\nnamespace App\\Infrastructure\\Models\\Traits;\n\n"
        "trait FarmScoped\n{\n}\n",
        encoding="utf-8",
    )

    models = tmp_path / "app" / "Domain" / "Farm" / "Models"
    models.mkdir(parents=True)
    (models / "Farm.php").write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\nclass Farm\n{\n}\n",
        encoding="utf-8",
    )

    factories = tmp_path / "database" / "factories"
    factories.mkdir(parents=True)
    (factories / "FarmFactory.php").write_text(
        "<?php\n\nnamespace Database\\Factories;\n\nclass FarmFactory\n{\n}\n",
        encoding="utf-8",
    )

    # composer's classmap is PHP source, so separators are written escaped.
    vendor = tmp_path / "vendor" / "composer"
    vendor.mkdir(parents=True)
    (vendor / "autoload_classmap.php").write_text(
        "<?php\nreturn array(\n"
        "    'Illuminate\\\\Database\\\\Eloquent\\\\Concerns\\\\HasUuids' => $v . '/a.php',\n"
        "    'Illuminate\\\\Database\\\\Eloquent\\\\Factories\\\\HasFactory' => $v . '/b.php',\n"
        "    'Illuminate\\\\Support\\\\Facades\\\\Schema' => $v . '/c.php',\n"
        "    'Vendor\\\\One\\\\Duplicated' => $v . '/d.php',\n"
        "    'Vendor\\\\Two\\\\Duplicated' => $v . '/e.php',\n"
        ");\n",
        encoding="utf-8",
    )
    return tmp_path


# ── repair_mangled_fqcn ────────────────────────────────────────────────────


def test_stuttered_first_party_fqcn_is_repaired(tmp_path):
    ws = _ws(tmp_path)
    assert repair_mangled_fqcn(
        "AppInfrastructureModelsTraitsFarmScopedModelsTraitsFarmScoped", str(ws)
    ) == "App\\Infrastructure\\Models\\Traits\\FarmScoped"


def test_stuttered_vendor_fqcn_is_repaired_from_the_classmap(tmp_path):
    ws = _ws(tmp_path)
    # Vendor classes are absent from the codebase index entirely, so the
    # composer classmap is the only ground truth for them.
    assert repair_mangled_fqcn(
        "IlluminateDatabaseEloquentConcernsHasUuidsConcernsHasUuids", str(ws)
    ) == "Illuminate\\Database\\Eloquent\\Concerns\\HasUuids"


def test_classmap_fqcns_are_unescaped(tmp_path):
    """A doubled backslash would emit a broken `use` statement."""
    ws = _ws(tmp_path)
    repaired = repair_mangled_fqcn(
        "IlluminateSupportFacadesSchemaFacadesSchema", str(ws)
    )
    assert repaired == "Illuminate\\Support\\Facades\\Schema"
    assert "\\\\" not in repaired


def test_intact_fqcn_is_left_alone(tmp_path):
    ws = _ws(tmp_path)
    assert repair_mangled_fqcn("App\\Domain\\Farm\\Models\\Farm", str(ws)) is None


def test_paths_and_dotted_names_are_left_alone(tmp_path):
    ws = _ws(tmp_path)
    assert repair_mangled_fqcn("app/Domain/Farm/Models/Farm.php", str(ws)) is None
    assert repair_mangled_fqcn("Farm.newFactory", str(ws)) is None


def test_short_or_lowercase_tokens_are_declined(tmp_path):
    ws = _ws(tmp_path)
    # Too short to anchor a longest-prefix match safely.
    assert repair_mangled_fqcn("Farm", str(ws)) is None
    # Not a class-shaped token.
    assert repair_mangled_fqcn("fakeunqualifiedthing", str(ws)) is None


def test_unmatched_symbol_returns_none(tmp_path):
    ws = _ws(tmp_path)
    assert repair_mangled_fqcn("CompletelyUnrelatedThingHere", str(ws)) is None


def test_longest_prefix_wins_between_sibling_classes(tmp_path):
    ws = _ws(tmp_path)
    (ws / "app" / "Domain" / "Farm" / "Models" / "FarmUser.php").write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\nclass FarmUser\n{\n}\n",
        encoding="utf-8",
    )
    assert repair_mangled_fqcn(
        "AppDomainFarmModelsFarmUserModelsFarmUser", str(ws)
    ) == "App\\Domain\\Farm\\Models\\FarmUser"


# ── resolve_class_fqcn ─────────────────────────────────────────────────────


def test_resolve_unique_basename(tmp_path):
    ws = _ws(tmp_path)
    assert resolve_class_fqcn("FarmScoped", str(ws)) == (
        "App\\Infrastructure\\Models\\Traits\\FarmScoped"
    )
    assert resolve_class_fqcn("HasFactory", str(ws)) == (
        "Illuminate\\Database\\Eloquent\\Factories\\HasFactory"
    )


def test_ambiguous_basename_is_refused(tmp_path):
    """Importing the wrong class is worse than leaving the model's output."""
    ws = _ws(tmp_path)
    assert resolve_class_fqcn("Duplicated", str(ws)) is None


# ── _php_auto_import ───────────────────────────────────────────────────────


PROBE26_MODEL = """<?php

declare(strict_types=1);

namespace App\\Domain\\Farm\\Models;

use Illuminate\\Database\\Eloquent\\Model;

class UnitOfMeasure extends Model
{
    use FarmScoped;
    use HasFactory;

    protected $table = 'units_of_measure';
}
"""


def test_probe26_missing_trait_imports_are_added(tmp_path):
    ws = _ws(tmp_path)
    target = ws / "app" / "Domain" / "Farm" / "Models" / "UnitOfMeasure.php"
    target.write_text(PROBE26_MODEL, encoding="utf-8")

    added = _php_auto_import(target, str(ws))

    assert added == [
        "use App\\Infrastructure\\Models\\Traits\\FarmScoped;",
        "use Illuminate\\Database\\Eloquent\\Factories\\HasFactory;",
    ]
    text = target.read_text(encoding="utf-8")
    assert "use App\\Infrastructure\\Models\\Traits\\FarmScoped;" in text
    # The trait use inside the class body must survive untouched.
    assert "    use FarmScoped;" in text


def test_already_imported_names_are_not_duplicated(tmp_path):
    ws = _ws(tmp_path)
    target = ws / "app" / "Domain" / "Farm" / "Models" / "Thing.php"
    target.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "use App\\Infrastructure\\Models\\Traits\\FarmScoped;\n\n"
        "class Thing\n{\n    use FarmScoped;\n}\n",
        encoding="utf-8",
    )
    assert _php_auto_import(target, str(ws)) is None


def test_same_namespace_class_needs_no_import(tmp_path):
    ws = _ws(tmp_path)
    target = ws / "app" / "Domain" / "Farm" / "Models" / "Barn.php"
    (ws / "app" / "Domain" / "Farm" / "Models" / "Local.php").write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\ntrait Local\n{\n}\n",
        encoding="utf-8",
    )
    target.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "class Barn\n{\n    use Local;\n}\n",
        encoding="utf-8",
    )
    assert _php_auto_import(target, str(ws)) is None


def test_ambiguous_reference_is_not_imported(tmp_path):
    ws = _ws(tmp_path)
    target = ws / "app" / "Domain" / "Farm" / "Models" / "Amb.php"
    target.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "class Amb\n{\n    use Duplicated;\n}\n",
        encoding="utf-8",
    )
    assert _php_auto_import(target, str(ws)) is None


def test_aliased_import_binds_the_alias(tmp_path):
    ws = _ws(tmp_path)
    target = ws / "app" / "Domain" / "Farm" / "Models" / "Aliased.php"
    target.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "use App\\Infrastructure\\Models\\Traits\\FarmScoped as Scoped;\n\n"
        "class Aliased\n{\n    use Scoped;\n}\n",
        encoding="utf-8",
    )
    assert _php_auto_import(target, str(ws)) is None


def test_clean_file_is_untouched(tmp_path):
    ws = _ws(tmp_path)
    target = ws / "app" / "Domain" / "Farm" / "Models" / "Plain.php"
    body = "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\nclass Plain\n{\n}\n"
    target.write_text(body, encoding="utf-8")
    assert _php_auto_import(target, str(ws)) is None
    assert target.read_text(encoding="utf-8") == body


def test_new_and_static_calls_are_deliberately_not_scanned(tmp_path):
    """Scanning them rebound PHP builtins to vendor namesakes (see review)."""
    ws = _ws(tmp_path)
    target = ws / "app" / "Domain" / "Farm" / "Models" / "Uses.php"
    target.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "class Uses\n{\n"
        "    public function go(): void\n    {\n"
        "        $f = new FarmFactory();\n"
        "        HasFactory::make();\n"
        "    }\n}\n",
        encoding="utf-8",
    )
    assert _php_auto_import(target, str(ws)) is None


# ── _repair_fix_path ───────────────────────────────────────────────────────


def _migrations(tmp_path: Path) -> Path:
    d = tmp_path / "database" / "migrations"
    d.mkdir(parents=True)
    (d / "2026_00_00_000000_create_units_of_measure_table.php").write_text(
        "<?php\n", encoding="utf-8"
    )
    # Probe 26's directory really did hold a second migration creating the
    # same table — the reason a similarity-only rule is not safe here.
    (d / "2026_05_20_000000_create_units_of_measure_table.php").write_text(
        "<?php\n", encoding="utf-8"
    )
    return tmp_path


def test_stuttered_path_segment_is_repaired(tmp_path):
    ws = _migrations(tmp_path)
    assert _repair_fix_path(
        "database/migrations/2026_00_00_00_000000_create_units_of_measure_table.php",
        str(ws),
    ) == "database/migrations/2026_00_00_000000_create_units_of_measure_table.php"


def test_existing_path_is_left_alone(tmp_path):
    ws = _migrations(tmp_path)
    assert _repair_fix_path(
        "database/migrations/2026_05_20_000000_create_units_of_measure_table.php",
        str(ws),
    ) is None


def test_legitimately_new_file_is_left_alone(tmp_path):
    """A gap plan may name a file that is about to be created."""
    ws = _migrations(tmp_path)
    assert _repair_fix_path("database/migrations/brand_new_thing.php", str(ws)) is None


def test_unknown_directory_is_left_alone(tmp_path):
    ws = _migrations(tmp_path)
    assert _repair_fix_path("does/not/exist/thing.php", str(ws)) is None


def test_empty_path_is_ignored(tmp_path):
    assert _repair_fix_path("", str(tmp_path)) is None


# ── _scrub_phantom_refs (first direct coverage) ────────────────────────────


def _patch_scrub(monkeypatch, tmp_path, indexed: set[str]) -> None:
    """Point the scrubber at *tmp_path* with *indexed* as the whole index."""
    import spine.agents.tools.codebase_query as cq
    from spine.config import SpineConfig

    import json as _json

    def _find(db, name):
        if name not in indexed:
            return None
        return _json.dumps({"status": "ok", "matches": [{"symbol_name": name}]})

    monkeypatch.setattr(cq, "find_symbol", _find)

    class _Cfg:
        checkpoint_path = str(tmp_path / ".spine" / "spine.db")
        workspace_root = str(tmp_path)

    monkeypatch.setattr(SpineConfig, "load", classmethod(lambda cls, path=None: _Cfg()))


def test_scrub_repairs_a_mangled_reference_symbol(monkeypatch, tmp_path):
    from spine.workflow.subgraphs.implement_subgraph import _scrub_phantom_refs

    ws = _ws(tmp_path)
    _patch_scrub(monkeypatch, ws, indexed={"FarmScoped"})

    out = _scrub_phantom_refs(
        {
            "id": "slice-1",
            "reference_symbols": [
                "AppInfrastructureModelsTraitsFarmScopedModelsTraitsFarmScoped"
            ],
        },
        "wk1",
    )
    # Repaired to the BARE LEAF — the only form find_symbol matches, since
    # symbol_metadata stores no namespaces.
    assert out["reference_symbols"] == ["FarmScoped"]


def test_scrub_still_drops_a_genuine_phantom(monkeypatch, tmp_path):
    from spine.workflow.subgraphs.implement_subgraph import _scrub_phantom_refs

    ws = _ws(tmp_path)
    _patch_scrub(monkeypatch, ws, indexed={"FarmScoped"})

    out = _scrub_phantom_refs(
        {"id": "slice-1", "reference_symbols": ["TotallyInventedSymbolName"]}, "wk1"
    )
    assert out["reference_symbols"] == []


def test_scrub_keeps_an_indexed_symbol_untouched(monkeypatch, tmp_path):
    from spine.workflow.subgraphs.implement_subgraph import _scrub_phantom_refs

    ws = _ws(tmp_path)
    _patch_scrub(monkeypatch, ws, indexed={"FarmScoped"})

    out = _scrub_phantom_refs(
        {"id": "slice-1", "reference_symbols": ["FarmScoped"]}, "wk1"
    )
    assert out["reference_symbols"] == ["FarmScoped"]


def test_scrub_keeps_a_vendor_repair_as_the_full_fqcn(monkeypatch, tmp_path):
    """Vendor classes are not indexed at all, so an index hit cannot be required.

    Requiring one discarded essentially every framework repair (3 of probe 26's
    own 8 symbols). The full FQCN is the genuine import, and
    _is_external_reference already classifies backslash-bearing strings.
    """
    from spine.workflow.subgraphs.implement_subgraph import _scrub_phantom_refs

    ws = _ws(tmp_path)
    _patch_scrub(monkeypatch, ws, indexed=set())

    out = _scrub_phantom_refs(
        {
            "id": "slice-1",
            "reference_symbols": [
                "IlluminateDatabaseEloquentConcernsHasUuidsConcernsHasUuids"
            ],
        },
        "wk1",
    )
    assert out["reference_symbols"] == [
        "Illuminate\\Database\\Eloquent\\Concerns\\HasUuids"
    ]


# ── guards added after adversarial review ──────────────────────────────────


def test_php_builtin_is_never_imported_as_a_vendor_namesake(tmp_path):
    """The critical review finding: `new DateTimeImmutable(...)` acquired
    `use Monolog\\DateTimeImmutable;` and the file fatalled at runtime, while
    `php -l` passed both before and after."""
    ws = _ws(tmp_path)
    (ws / "vendor" / "composer" / "autoload_classmap.php").write_text(
        "<?php\nreturn array(\n"
        "    'Monolog\\\\DateTimeImmutable' => $v . '/x.php',\n"
        ");\n",
        encoding="utf-8",
    )
    assert resolve_class_fqcn("DateTimeImmutable", str(ws)) is None


def test_global_namespace_file_is_skipped(tmp_path):
    """Migrations, config/, routes/ and 116 of 129 agripath tests have no
    namespace; there an unqualified name already resolves globally, so adding
    any import changes resolution."""
    ws = _ws(tmp_path)
    target = ws / "database" / "migrations"
    target.mkdir(parents=True)
    f = target / "2026_01_01_000000_thing.php"
    f.write_text(
        "<?php\n\nuse Illuminate\\Database\\Migrations\\Migration;\n\n"
        "return new class extends Migration\n{\n    use FarmScoped;\n};\n",
        encoding="utf-8",
    )
    assert _php_auto_import(f, str(ws)) is None


def test_names_in_comments_are_not_treated_as_references(tmp_path):
    ws = _ws(tmp_path)
    f = ws / "app" / "Domain" / "Farm" / "Models" / "Commented.php"
    f.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "/**\n * Legacy notes:\n *     use FarmScoped;\n */\n"
        "// also mentions FarmFactory::make()\n"
        "class Commented\n{\n}\n",
        encoding="utf-8",
    )
    assert _php_auto_import(f, str(ws)) is None


def test_use_inside_a_block_comment_is_not_an_anchor(tmp_path):
    ws = _ws(tmp_path)
    f = ws / "app" / "Domain" / "Farm" / "Models" / "Anchored.php"
    f.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "use Illuminate\\Database\\Eloquent\\Model;\n\n"
        "/*\nuse App\\Legacy\\Old;\n*/\n"
        "class Anchored\n{\n    use FarmScoped;\n}\n",
        encoding="utf-8",
    )
    added = _php_auto_import(f, str(ws))
    assert added == ["use App\\Infrastructure\\Models\\Traits\\FarmScoped;"]
    text = f.read_text(encoding="utf-8")
    # Inserted above the comment and above the class body, not inside either.
    assert text.index("use App\\Infrastructure") < text.index("/*")


def test_column_zero_trait_use_is_not_an_insertion_anchor(tmp_path):
    """A class-body trait use written at column 0 is legal PHP; treating it as
    a top-level import put the inserted line inside the class body."""
    ws = _ws(tmp_path)
    f = ws / "app" / "Domain" / "Farm" / "Models" / "Flat.php"
    f.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "use Illuminate\\Database\\Eloquent\\Model;\n\n"
        "class Flat\n{\nuse Model;\n    use FarmScoped;\n}\n",
        encoding="utf-8",
    )
    added = _php_auto_import(f, str(ws))
    if added:
        text = f.read_text(encoding="utf-8")
        assert text.index("use App\\Infrastructure") < text.index("class Flat")


def test_grouped_imports_make_the_file_a_no_op(tmp_path):
    """Grouped/comma imports are real bindings this scanner cannot read;
    re-importing their names would be a duplicate-name fatal."""
    ws = _ws(tmp_path)
    f = ws / "app" / "Domain" / "Farm" / "Models" / "Grouped.php"
    f.write_text(
        "<?php\n\nnamespace App\\Domain\\Farm\\Models;\n\n"
        "use App\\Infrastructure\\Models\\Traits\\{FarmScoped};\n\n"
        "class Grouped\n{\n    use FarmScoped;\n}\n",
        encoding="utf-8",
    )
    assert _php_auto_import(f, str(ws)) is None


def test_forward_reference_is_not_rewritten_to_a_different_class(tmp_path):
    """The other critical finding: without a stutter check, a mangled FQCN for
    a class that does not exist YET prefix-matched a real one and was silently
    rewritten onto it — and not-yet-existing is exactly the population
    _scrub_phantom_refs feeds in."""
    ws = _ws(tmp_path)
    assert repair_mangled_fqcn("AppDomainFarmModelsBrandNewThing", str(ws)) is None


def test_psr4_array_form_is_honoured(tmp_path):
    """laravel/framework itself uses the list form; dropping it emptied the
    whole first-party corpus."""
    (tmp_path / "composer.json").write_text(
        json.dumps({"autoload": {"psr-4": {"App\\": ["app/", "src/"]}}}),
        encoding="utf-8",
    )
    d = tmp_path / "src" / "Widgets"
    d.mkdir(parents=True)
    (d / "Sprocket.php").write_text(
        "<?php\n\nnamespace App\\Widgets;\n\nclass Sprocket\n{\n}\n", encoding="utf-8"
    )
    assert resolve_class_fqcn("Sprocket", str(tmp_path)) == "App\\Widgets\\Sprocket"
