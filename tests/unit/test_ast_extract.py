"""Unit tests for :mod:`spine.agents.tools.ast_extract`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spine.agents.tools.ast_extract import Symbol, extract_edges, extract_symbols


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


def test_c_extracts_functions_structs_enums_unions() -> None:
    sample = FIXTURES / "sample.c"
    symbols = extract_symbols(str(sample), "tests/fixtures/sample.c")

    names = _names(symbols)
    # plain function, pointer-returning static function, struct, union, enum
    assert {"add", "format_point", "Point", "Value", "Color"}.issubset(names)
    by_name = {s.symbol_name: s for s in symbols}
    assert by_name["add"].symbol_type == "function"
    assert by_name["Point"].symbol_type == "struct"
    assert by_name["Value"].symbol_type == "union"
    assert by_name["Color"].symbol_type == "enum"
    for s in symbols:
        assert s.lang == "c"
        assert s.raw_code


def test_cpp_extracts_classes_methods_and_free_functions() -> None:
    sample = FIXTURES / "sample.cpp"
    symbols = extract_symbols(str(sample), "tests/fixtures/sample.cpp")

    names = _names(symbols)
    # class, struct, enum class, in-class method, out-of-line method,
    # reference-returning and pointer-returning free functions
    assert {
        "Greeter", "Point", "Color", "greet",
        "Greeter::shout", "dot", "pick", "make_counter",
    }.issubset(names)

    by_name_type = {(s.symbol_name, s.symbol_type): s for s in symbols}
    assert ("Greeter", "class") in by_name_type
    # The constructor is a method that shares the class's name.
    assert ("Greeter", "method") in by_name_type
    assert ("Point", "struct") in by_name_type
    assert ("Color", "enum") in by_name_type
    # In-class definition is a method qualified by its enclosing class.
    greet = by_name_type[("greet", "method")]
    assert greet.symbol_type == "method"
    assert greet.parent_class == "Greeter"
    assert greet.qualified_name == "Greeter.greet"
    # Destructor is captured too.
    assert "~Greeter" in names
    for s in symbols:
        assert s.lang == "cpp"
        assert s.raw_code


def test_header_extension_parses_as_cpp(tmp_path: Path) -> None:
    header = tmp_path / "util.h"
    header.write_text("int twice(int v);\nint twice(int v) { return v * 2; }\n")
    symbols = extract_symbols(str(header), "src/util.h")
    assert _names(symbols) == {"twice"}
    assert symbols[0].lang == "cpp"


def test_unsupported_extension_returns_empty() -> None:
    assert extract_symbols("/tmp/nonexistent.yaml", "x.yaml") == []


def test_missing_file_returns_empty() -> None:
    assert extract_symbols("/tmp/this-file-does-not-exist-xyz123.py", "x.py") == []


# ── PHP dependency-edge extraction ────────────────────────────────────────


_EDGE_PHP = """<?php
namespace App;

use App\\Services\\Mailer;
use App\\Services\\Long\\PathThing as PT;

class OrderHandler extends BaseHandler implements HandlerInterface {
    use LoggableTrait;

    public function handle($order) {
        $mailer = new Mailer();
        $pt = new PT();
        Validator::check($order);
        $mailer->send($order);
        format_total($order);
    }
}

function top_level() {
    helper_fn();
}
"""


@pytest.fixture
def php_edges(tmp_path: Path) -> list:
    target = tmp_path / "handler.php"
    target.write_text(_EDGE_PHP)
    return extract_edges(str(target), "src/handler.php")


def _edge_set(edges: list) -> set[tuple[str, str, str]]:
    return {(e.src_symbol, e.edge_kind, e.dst_name) for e in edges}


def test_php_edges_cover_all_kinds(php_edges: list) -> None:
    got = _edge_set(php_edges)
    assert ("", "use_import", "Mailer") in got
    assert ("OrderHandler", "extends", "BaseHandler") in got
    assert ("OrderHandler", "implements", "HandlerInterface") in got
    assert ("OrderHandler", "trait_use", "LoggableTrait") in got
    assert ("OrderHandler.handle", "new", "Mailer") in got
    assert ("OrderHandler.handle", "static_call", "Validator.check") in got
    assert ("OrderHandler.handle", "call", "send") in got
    assert ("OrderHandler.handle", "call", "format_total") in got
    assert ("top_level", "call", "helper_fn") in got
    for e in php_edges:
        assert e.lang == "php"
        assert e.src_file == "src/handler.php"


def test_php_use_alias_resolves_to_target_class(php_edges: list) -> None:
    # `use ...\PathThing as PT; new PT()` must record PathThing, not PT.
    got = _edge_set(php_edges)
    assert ("OrderHandler.handle", "new", "PathThing") in got
    assert ("", "use_import", "PathThing") in got


def test_edges_empty_for_non_php() -> None:
    assert extract_edges("/tmp/whatever.py", "x.py") == []
    assert extract_edges("/tmp/missing-file-xyz.php", "x.php") == []
