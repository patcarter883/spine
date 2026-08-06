"""Cross-slice verify evidence, reasoning opt-out, crash scoring, glob targets.

All four come from the arm A / arm B side-by-side (runs cc9c2611 and 593a5cc4)
of the same task, neither of which converged:

* arm A ground SEVEN gap cycles to 4 VERIFIED / 8. Its ``farm-architecture-tests``
  slice failed 8 criteria on "No evidence that AssetType exists" while
  app/Domain/Farm/Models/AssetType.php had been on disk for 1h36m — the judge's
  diff and source block are pathspec-scoped to its OWN target_files, so the
  statement was true about its evidence and false about the workspace. The one
  mechanism that would have widened it read ``state["execution_waves"]``, which
  a LangGraph ``Send`` never carries, so it was dead code.
* arm B lost two of four slices to ``StructuredOutputValidationError`` — the
  VerificationResult truncated mid-JSON because ``reasoning: false`` was
  implemented only in the LOCAL builder and never reached the OpenRouter wire,
  letting reasoning eat 3130 of a 4096-token budget.
* a crashed verifier scored as exactly ONE failed criterion, hiding a slice's
  real 13 and distorting the best-state ratchet that consumes that count.
* arm B wrote four files to LITERAL GLOB paths (``*create_contacts_table.php``)
  because plans put patterns in target_files and the scope check compares
  strings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import spine.agents.helpers as helpers
from spine.agents.helpers import _build_openrouter_model, suppress_reasoning
from spine.agents.synthesis_implementer import _resolve_glob_target
from spine.workflow.subgraphs.verify_subgraph import _sibling_manifest_block

_OR = "openrouter:deepseek/deepseek-v4-flash-0731"


def _cfg(**over):
    base = {"name": "x", "model": _OR, "streaming": False, "stream_usage": False}
    base.update(over)
    return base


# ── reasoning opt-out on the OpenRouter lane ───────────────────────────────


def test_reasoning_false_reaches_the_wire():
    with patch.object(helpers, "_active_provider_config", lambda **k: _cfg(reasoning=False)):
        m = _build_openrouter_model(_OR, "s")
    assert m.extra_body["reasoning"] == {"enabled": False}


def test_reasoning_unset_sends_nothing():
    with patch.object(helpers, "_active_provider_config", lambda **k: _cfg()):
        m = _build_openrouter_model(_OR, "s")
    assert "reasoning" not in (m.extra_body or {})


def test_suppress_reasoning_disables_rather_than_minimising():
    """`effort: minimal` still burns tokens; only `enabled: false` reclaims budget.

    Measured on deepseek-v4-flash-0731 for "Reply with exactly: ok":
    enabled=false -> 0 reasoning tokens, effort=minimal -> 38,
    max_tokens=0 -> 11, exclude=true -> 24. Suppression that does not reclaim
    the budget does not prevent the truncation it exists to prevent.
    """
    with patch.object(helpers, "_active_provider_config", lambda **k: _cfg()):
        m = _build_openrouter_model(_OR, "s")
    assert suppress_reasoning(m).extra_body["reasoning"] == {"enabled": False}


# ── sibling-slice manifest ─────────────────────────────────────────────────


def _ws(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "AssetType.php").write_text("<?php", encoding="utf-8")
    return tmp_path


def test_manifest_reports_a_sibling_file_that_exists(tmp_path):
    block = _sibling_manifest_block(str(_ws(tmp_path)), ["app/AssetType.php"])
    assert "PRESENT" in block
    assert "app/AssetType.php" in block


def test_manifest_distinguishes_absent_from_present(tmp_path):
    block = _sibling_manifest_block(
        str(_ws(tmp_path)), ["app/AssetType.php", "app/Nope.php"]
    )
    assert "PRESENT  app/AssetType.php" in block
    assert "ABSENT " in block and "app/Nope.php" in block


def test_manifest_is_empty_without_siblings(tmp_path):
    ws = str(_ws(tmp_path))
    assert _sibling_manifest_block(ws, []) == ""
    assert _sibling_manifest_block(ws, None) == ""


def test_manifest_skips_glob_entries(tmp_path):
    """A glob is never a real path; reporting it ABSENT would reintroduce the
    same false negative this block exists to remove."""
    block = _sibling_manifest_block(
        str(_ws(tmp_path)), ["database/migrations/*_create_x_table.php"]
    )
    assert block == ""


def test_manifest_is_bounded(tmp_path):
    ws = _ws(tmp_path)
    many = [f"app/F{i}.php" for i in range(200)]
    block = _sibling_manifest_block(str(ws), many)
    assert "more)" in block
    assert block.count("\n") < 100


# ── glob write targets ─────────────────────────────────────────────────────


def _migrations(tmp_path: Path) -> Path:
    d = tmp_path / "database" / "migrations"
    d.mkdir(parents=True)
    (d / "2026_05_01_000001_create_asset_types_table.php").write_text(
        "<?php", encoding="utf-8"
    )
    return tmp_path


def test_glob_resolves_to_a_unique_existing_file(tmp_path):
    ws = _migrations(tmp_path)
    assert _resolve_glob_target(
        str(ws), "database/migrations/*_create_asset_types_table.php"
    ) == "database/migrations/2026_05_01_000001_create_asset_types_table.php"


def test_glob_with_no_match_is_refused(tmp_path):
    """The concrete name cannot be invented: a migration's timestamp decides
    run order, so guessing it is a correctness decision, not a formatting one."""
    ws = _migrations(tmp_path)
    assert _resolve_glob_target(
        str(ws), "database/migrations/*_create_contacts_table.php"
    ) is None


def test_ambiguous_glob_is_refused(tmp_path):
    ws = _migrations(tmp_path)
    d = ws / "database" / "migrations"
    (d / "2026_05_02_000001_create_other_table.php").write_text("<?php", encoding="utf-8")
    assert _resolve_glob_target(str(ws), "database/migrations/*.php") is None


def test_recursive_glob_is_refused_when_unmatched(tmp_path):
    ws = _migrations(tmp_path)
    assert _resolve_glob_target(str(ws), "tests/Feature/**/*Contact*Test.php") is None
