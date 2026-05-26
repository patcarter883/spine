"""Unit tests for :mod:`spine.agents.tools.ast_extract`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spine.agents.tools.ast_extract import Symbol, extract_symbols


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _names(symbols: list[Symbol]) -> set[str]:
    return {s.symbol_name for s in symbols}


def test_python_extracts_functions_and_classes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "spine" / "agents" / "factory.py"
    if not target.exists():
        pytest.skip(f"{target} not present in this checkout")

    symbols = extract_symbols(str(target), "spine/agents/factory.py")

    assert len(symbols) >= 10
    names = _names(symbols)
    assert "build_phase_agent" in names
    assert "SpineProjectMemoryMiddleware" in names

    for s in symbols:
        assert s.lang == "python"
        assert s.raw_code  # non-empty body
        assert s.end_byte > s.start_byte
        # The raw_code is a byte slice, never the whole file.
        full_size = os.path.getsize(target)
        assert len(s.raw_code.encode("utf-8")) <= full_size


def test_python_raw_code_is_function_body_not_file() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "spine" / "agents" / "factory.py"
    if not target.exists():
        pytest.skip(f"{target} not present in this checkout")

    symbols = extract_symbols(str(target), "spine/agents/factory.py")
    by_name = {s.symbol_name: s for s in symbols}

    # build_phase_agent is a top-level function — its slice should be
    # much smaller than the surrounding file.
    bpa = by_name["build_phase_agent"]
    file_bytes = target.read_bytes()
    assert len(bpa.raw_code.encode("utf-8")) < len(file_bytes) // 2


def test_php_extracts_functions_classes_interfaces() -> None:
    sample = FIXTURES / "sample.php"
    symbols = extract_symbols(str(sample), "tests/fixtures/sample.php")

    names = _names(symbols)
    assert {"greet", "Greeter", "say", "Speaker"}.issubset(names)
    for s in symbols:
        assert s.lang == "php"


def test_typescript_extracts_functions_arrow_classes_interfaces() -> None:
    sample = FIXTURES / "sample.ts"
    symbols = extract_symbols(str(sample), "tests/fixtures/sample.ts")

    names = _names(symbols)
    # function_declaration, variable_declarator w/ arrow init, class, interface, method
    assert {"greet", "shout", "Greeter", "Speaker", "say"}.issubset(names)
    for s in symbols:
        assert s.lang == "typescript"


def test_unsupported_extension_returns_empty() -> None:
    assert extract_symbols("/tmp/nonexistent.yaml", "x.yaml") == []


def test_missing_file_returns_empty() -> None:
    assert extract_symbols("/tmp/this-file-does-not-exist-xyz123.py", "x.py") == []
