"""Deterministic checks for Laravel migration files at the write exit.

Migrations are the one thing every agripath probe has got wrong, in a different
way each time:

* probe 26 — ``2026_00_00_000000_create_units_of_measure_table.php`` (month 00,
  day 00)
* run cc9c2611 (killed) — ``2024_01_01_000001_*``, a two-year-stale prefix
* probe 27 — ``foreignId('farm_id')->constrained('farms')`` where ``farms.id``
  is a ``uuid``, plus a duplicate whose filename carried a slice-id fragment
  (``1-2026_05_01_000001_create_asset_types_table.php``)

Both classes fail SILENTLY at write time. ``php -l`` passes every one of them,
so the pre-write syntax gate and the ``php_syntax`` landing gate are blind, and
the damage only appears when ``php artisan migrate`` runs — by which point it
presents as an unrelated wall of test failures. Probe 27 spent four gap cycles
without fixing a one-token FK error.

Laravel orders migrations by a PLAIN LEXICOGRAPHIC SORT of the basename
(``Migrator::getMigrationFiles`` globs ``*_*.php`` and sorts strings — there is
no date parsing anywhere), so a malformed prefix does not fail loudly, it
silently sorts a new table BEFORE the tables it references.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Laravel's own generator format: date('Y_m_d_His') . '_' . $name . '.php'.
_MIGRATION_NAME = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{6})_\w+\.php$")

# Laravel ships its framework migrations under this deliberate sentinel prefix
# so they always sort first. Three stock files use it; never flag them.
_FRAMEWORK_PREFIX = "0001_01_01_"

# $table->foreignId('farm_id')->constrained('farms')
_FOREIGN_ID = re.compile(
    r"\$table->foreignId\(\s*['\"](?P<col>\w+)['\"]\s*\)"
    r"(?P<chain>(?:->\w+\([^)]*\))*)",
)
_CONSTRAINED = re.compile(r"->constrained\(\s*['\"](?P<table>\w+)['\"]")

# Schema::create('farms', function (Blueprint $table) {
_SCHEMA_CREATE = re.compile(r"Schema::create\(\s*['\"](?P<table>\w+)['\"]")


def _pk_type(body: str) -> Optional[str]:
    """The PK column type declared in a Schema::create closure, or None."""
    if re.search(r"\$table->uuid\(\s*['\"]id['\"]\s*\)\s*->\s*primary\(", body):
        return "uuid"
    if re.search(r"\$table->id\(\s*\)", body):
        return "bigint"
    if re.search(r"\$table->ulid\(\s*['\"]id['\"]\s*\)\s*->\s*primary\(", body):
        return "ulid"
    if re.search(r"\$table->string\(\s*['\"]id['\"]", body):
        return "string"
    return None


def migration_pk_types(migrations_dir: Path) -> dict[str, str]:
    """table -> PK type, parsed from every migration in *migrations_dir*.

    Scans ALL files, not just ``create_<table>_table.php``: one file can create
    several tables (Laravel's stock users migration creates three) and a later
    ALTER can change a PK type. Tables whose type cannot be determined —
    composite-PK pivots, config-driven names — are simply absent, and callers
    must fail open on a miss rather than guess.
    """
    out: dict[str, str] = {}
    try:
        files = sorted(migrations_dir.glob("*.php"))
    except OSError:
        return out
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:  # noqa: PERF203 — unreadable file, skip
            continue
        for m in _SCHEMA_CREATE.finditer(text):
            # Body = from the match to the next Schema:: call (or EOF); good
            # enough to isolate one closure without a PHP parser.
            nxt = text.find("Schema::", m.end())
            body = text[m.end(): nxt if nxt > 0 else len(text)]
            pk = _pk_type(body)
            if pk:
                out[m.group("table")] = pk
    return out


def check_migration_filename(name: str) -> Optional[str]:
    """A warning when *name* is not a sortable Laravel migration name, else None."""
    if name.startswith(_FRAMEWORK_PREFIX):
        return None
    m = _MIGRATION_NAME.match(name)
    if not m:
        return (
            f"migration filename {name!r} is not Laravel's "
            f"YYYY_MM_DD_HHMMSS_name.php form. Laravel orders migrations by a "
            f"plain string sort of the basename, so this will not fail — it "
            f"will silently run in the wrong order."
        )
    year, month, day, _ = m.groups()
    if not (1 <= int(month) <= 12) or not (1 <= int(day) <= 31):
        return (
            f"migration filename {name!r} has an impossible date "
            f"({year}-{month}-{day}); it sorts before every real migration, so "
            f"its tables are created before the tables they reference."
        )
    return None


def check_migration_ordering(name: str, siblings: list[str]) -> Optional[str]:
    """A warning when *name* would not sort last among *siblings*, else None.

    A well-formed but STALE prefix passes :func:`check_migration_filename` and
    still breaks: run cc9c2611 emitted ``2024_01_01_000001_*``, a perfectly
    valid date that sorts before every real migration in the repo, so its table
    would be created before the tables it references. A migration being written
    now should be the newest one; anything else is almost certainly a
    fabricated timestamp rather than a deliberate backdate.
    """
    if name.startswith(_FRAMEWORK_PREFIX) or not _MIGRATION_NAME.match(name):
        return None  # framework sentinel, or already flagged as malformed
    others = [
        s for s in siblings
        if s != name and not s.startswith(_FRAMEWORK_PREFIX) and _MIGRATION_NAME.match(s)
    ]
    if not others:
        return None
    newest = max(others)
    if name > newest:
        return None
    return (
        f"migration {name!r} sorts BEFORE the newest existing migration "
        f"({newest!r}). Laravel runs migrations in lexicographic filename "
        f"order, so this one runs first and any table it references does not "
        f"exist yet. A newly written migration should carry the current "
        f"timestamp."
    )


def repair_foreign_id(content: str, pk_types: dict[str, str]) -> tuple[str, list[str]]:
    """Rewrite ``foreignId`` to ``foreignUuid`` where the target PK is a uuid.

    ``foreignId`` creates a BIGINT. Constraining it to a uuid primary key is a
    hard DDL error, not a style issue — Postgres refuses it outright with
    SQLSTATE 42804 ("Key columns ... are of incompatible types: bigint and
    uuid"). The repo agrees unanimously: 102 uses of ``foreignUuid`` against
    uuid-PK tables, and every one of the 3 real ``foreignId`` uses is a known
    defect the repo later shipped a migration to fix.

    Only rewrites when the target table's PK is KNOWN to be uuid, so a genuine
    bigint FK is never touched. Returns (new_content, changes).
    """
    changes: list[str] = []
    out = content
    for m in _FOREIGN_ID.finditer(content):
        chain = m.group("chain") or ""
        con = _CONSTRAINED.search(chain)
        if not con:
            continue  # no ->constrained(): nothing proves the target table
        table = con.group("table")
        if pk_types.get(table) != "uuid":
            continue  # unknown or genuinely bigint — leave it alone
        old = f"$table->foreignId('{m.group('col')}')"
        new = f"$table->foreignUuid('{m.group('col')}')"
        if old in out:
            out = out.replace(old, new, 1)
        else:  # double-quoted form
            old = f'$table->foreignId("{m.group("col")}")'
            new = f'$table->foreignUuid("{m.group("col")}")'
            out = out.replace(old, new, 1)
        changes.append(
            f"foreignId('{m.group('col')}') -> foreignUuid "
            f"({table}.id is uuid; bigint FK is a hard DDL error)"
        )
    return out, changes
