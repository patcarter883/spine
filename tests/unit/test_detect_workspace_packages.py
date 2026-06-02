"""Unit tests for detect_workspace_packages and monorepo-aware group_symbols_by_module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from spine.agents.tools.ast_extract import Symbol
from spine.work.onboarding.analyzer import (
    _build_pkg_index,
    _module_of_with_packages,
    detect_workspace_packages,
    group_symbols_by_module,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_tree(root: Path, structure: dict) -> None:
    """Recursively create dirs and files from a nested dict.

    Values of ``True`` → create the file. Values of ``dict`` → recurse.
    """
    for name, content in structure.items():
        path = root / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            _make_tree(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def _sym(file_path: str) -> Symbol:
    return Symbol(
        file_path=file_path,
        symbol_name="fn",
        symbol_type="function",
        lang="typescript",
        raw_code="",
        start_byte=0,
        end_byte=0,
    )


# ── detect_workspace_packages ────────────────────────────────────────────────


def test_no_markers_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "app.ts").touch()
        assert detect_workspace_packages(tmp) == []


def test_depth1_apps() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_tree(
            Path(tmp),
            {
                "admin": {"package.json": True, "src": {"app.ts": True}},
                "collect": {"package.json": True, "src": {"main.ts": True}},
            },
        )
        pkgs = detect_workspace_packages(tmp)
        names = {p["dotted_name"] for p in pkgs}
        assert names == {"admin", "collect"}
        kinds = {p["dotted_name"]: p["kind"] for p in pkgs}
        assert kinds["admin"] == "app"
        assert kinds["collect"] == "app"


def test_depth2_libs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_tree(
            Path(tmp),
            {
                "libs": {
                    "auth": {"pyproject.toml": True},
                    "calculators": {"pyproject.toml": True},
                },
            },
        )
        pkgs = detect_workspace_packages(tmp)
        names = {p["dotted_name"] for p in pkgs}
        assert names == {"libs.auth", "libs.calculators"}
        # libs/ itself is NOT returned as a package
        assert "libs" not in names
        for p in pkgs:
            assert p["kind"] == "lib"


def test_mixed_depth1_app_and_depth2_libs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_tree(
            Path(tmp),
            {
                "admin": {"package.json": True},
                "libs": {
                    "auth": {"package.json": True},
                },
            },
        )
        pkgs = detect_workspace_packages(tmp)
        by_name = {p["dotted_name"]: p for p in pkgs}
        assert set(by_name) == {"admin", "libs.auth"}
        assert by_name["admin"]["kind"] == "app"
        assert by_name["libs.auth"]["kind"] == "lib"


def test_depth1_package_subdirs_not_re_scanned() -> None:
    """admin/ has package.json so admin/src/ must NOT produce an admin.src package."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_tree(
            Path(tmp),
            {
                "admin": {
                    "package.json": True,
                    "src": {"package.json": True},  # nested — must be ignored
                },
            },
        )
        pkgs = detect_workspace_packages(tmp)
        names = {p["dotted_name"] for p in pkgs}
        assert "admin" in names
        assert "admin.src" not in names


def test_hidden_dirs_skipped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_tree(
            Path(tmp),
            {
                ".swarm": {"package.json": True},
                "admin": {"package.json": True},
            },
        )
        pkgs = detect_workspace_packages(tmp)
        names = {p["dotted_name"] for p in pkgs}
        assert ".swarm" not in names
        assert "admin" in names


def test_single_package_not_monorepo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_tree(Path(tmp), {"admin": {"package.json": True}})
        pkgs = detect_workspace_packages(tmp)
        assert len(pkgs) == 1
        # is_monorepo logic requires >= 2 packages
        assert not (len(pkgs) >= 2)


def test_pyproject_and_package_json_both_detected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _make_tree(
            Path(tmp),
            {
                "backend": {"pyproject.toml": True},
                "frontend": {"package.json": True},
            },
        )
        pkgs = detect_workspace_packages(tmp)
        names = {p["dotted_name"] for p in pkgs}
        assert names == {"backend", "frontend"}


# ── _module_of_with_packages ─────────────────────────────────────────────────


def test_module_of_with_packages_matches_longest_prefix() -> None:
    pkgs = [
        {"path": "libs/auth", "dotted_name": "libs.auth"},
        {"path": "libs", "dotted_name": "libs"},
    ]
    idx = _build_pkg_index(pkgs)
    assert _module_of_with_packages("libs/auth/src/foo.ts", idx) == "libs.auth"


def test_module_of_with_packages_fallback_two_segment() -> None:
    idx = _build_pkg_index([])
    assert _module_of_with_packages("spine/work/foo.py", idx) == "spine.work"


def test_module_of_with_packages_top_level_file() -> None:
    idx = _build_pkg_index([])
    # Single-segment file: fallback matches original _module_of behaviour
    assert _module_of_with_packages("config.py", idx) == "config.py"


# ── group_symbols_by_module ───────────────────────────────────────────────────


def test_group_symbols_monorepo() -> None:
    symbols = [
        _sym("admin/src/foo.ts"),
        _sym("admin/src/bar.ts"),
        _sym("collect/src/main.ts"),
    ]
    pkgs = [
        {"path": "admin", "dotted_name": "admin"},
        {"path": "collect", "dotted_name": "collect"},
    ]
    grouped = group_symbols_by_module(symbols, pkgs)
    assert set(grouped) == {"admin", "collect"}
    assert len(grouped["admin"][0]) == 2
    assert len(grouped["collect"][0]) == 1


def test_group_symbols_depth2_libs() -> None:
    symbols = [
        _sym("libs/auth/src/guard.ts"),
        _sym("libs/calculators/src/calc.ts"),
    ]
    pkgs = [
        {"path": "libs/auth", "dotted_name": "libs.auth"},
        {"path": "libs/calculators", "dotted_name": "libs.calculators"},
    ]
    grouped = group_symbols_by_module(symbols, pkgs)
    assert set(grouped) == {"libs.auth", "libs.calculators"}


def test_group_symbols_fallback_no_packages() -> None:
    """Empty workspace_packages → original two-segment behaviour."""
    symbols = [
        _sym("spine/work/foo.py"),
        _sym("spine/agents/bar.py"),
    ]
    grouped = group_symbols_by_module(symbols, [])
    assert set(grouped) == {"spine.work", "spine.agents"}


def test_group_symbols_fallback_none_packages() -> None:
    """None workspace_packages → original two-segment behaviour."""
    symbols = [_sym("spine/work/foo.py")]
    grouped = group_symbols_by_module(symbols, None)
    assert set(grouped) == {"spine.work"}


def test_group_symbols_file_outside_any_package_falls_back() -> None:
    """Files outside detected packages use two-segment fallback."""
    symbols = [
        _sym("admin/src/foo.ts"),   # matches admin package
        _sym("tools/scripts/gen.py"),  # no package → fallback
    ]
    pkgs = [{"path": "admin", "dotted_name": "admin"}]
    grouped = group_symbols_by_module(symbols, pkgs)
    assert "admin" in grouped
    assert "tools.scripts" in grouped
