"""Language-agnostic AST symbol extraction backed by tree-sitter.

Replaces the previous ``ast.parse``-based extractor with a multi-language
implementation that returns byte-sliced symbol bodies instead of whole
files. Used by the vector indexer (Phase 1) and the
``AstExtractSymbolTool`` drill-down (Phase 2).

Supported languages (inferred from file extension):

- ``.py``        → python
- ``.php``       → php
- ``.ts``/``.tsx`` → typescript

Unknown extensions return an empty list (logged at DEBUG).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".php": "php",
    ".ts": "typescript",
    ".tsx": "typescript",
}


@dataclass(frozen=True)
class Symbol:
    """A single extracted symbol with its source slice and location."""

    file_path: str
    symbol_name: str
    symbol_type: str
    raw_code: str
    start_byte: int
    end_byte: int
    lang: str


# S-expression queries per language. Each query captures the symbol's
# name node as ``@name`` and the containing definition node as ``@def``.
_QUERY_STRINGS: dict[str, str] = {
    "python": """
        (function_definition name: (identifier) @name) @def
        (class_definition name: (identifier) @name) @def
    """,
    "php": """
        (function_definition name: (name) @name) @def
        (method_declaration name: (name) @name) @def
        (class_declaration name: (name) @name) @def
        (interface_declaration name: (name) @name) @def
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @def
        (method_definition name: (property_identifier) @name) @def
        (class_declaration name: (type_identifier) @name) @def
        (interface_declaration name: (type_identifier) @name) @def
        (variable_declarator
            name: (identifier) @name
            value: (arrow_function)) @def
    """,
}

# Map node-type → symbol_type for tagging extracted symbols.
_NODE_TYPE_TO_SYMBOL_TYPE: dict[str, str] = {
    "function_definition": "function",
    "function_declaration": "function",
    "class_definition": "class",
    "class_declaration": "class",
    "method_declaration": "method",
    "method_definition": "method",
    "interface_declaration": "interface",
    "variable_declarator": "function",  # arrow function assigned to const
}


# Cached parsers/queries — built lazily on first use per language.
_PARSERS: dict[str, Any] = {}
_QUERIES: dict[str, Any] = {}


def _get_parser(lang: str) -> Any:
    """Return a cached tree-sitter Parser for the given language."""
    if lang in _PARSERS:
        return _PARSERS[lang]

    import tree_sitter

    if lang == "python":
        import tree_sitter_python as ts_lang_mod

        language = tree_sitter.Language(ts_lang_mod.language())
    elif lang == "php":
        import tree_sitter_php as ts_lang_mod

        # tree-sitter-php exposes both php and php_only; use php_only for
        # extracting pure PHP without inline HTML.
        language = tree_sitter.Language(ts_lang_mod.language_php())
    elif lang == "typescript":
        import tree_sitter_typescript as ts_lang_mod

        language = tree_sitter.Language(ts_lang_mod.language_typescript())
    else:
        raise ValueError(f"Unsupported language: {lang}")

    parser = tree_sitter.Parser(language)
    _PARSERS[lang] = parser
    _QUERIES[lang] = tree_sitter.Query(language, _QUERY_STRINGS[lang])
    return parser


def _get_query(lang: str) -> Any:
    """Return the cached query for the given language (builds parser if needed)."""
    if lang not in _QUERIES:
        _get_parser(lang)
    return _QUERIES[lang]


def _infer_lang(full_path: str) -> str | None:
    """Return the language for a file path, or None for unsupported extensions."""
    _, ext = os.path.splitext(full_path)
    return _EXT_TO_LANG.get(ext.lower())


def extract_symbols(full_path: str, rel_path: str) -> list[Symbol]:
    """Extract function/class/method/interface symbols from a source file.

    Args:
        full_path: Absolute path to the source file (read from disk).
        rel_path: Path to record on each :class:`Symbol` (usually the
            workspace-relative form).

    Returns:
        A list of :class:`Symbol` instances. Empty list for unsupported
        languages, missing files, or parse failures (logged at DEBUG).
    """
    lang = _infer_lang(full_path)
    if lang is None:
        logger.debug("Skipping %s: unsupported extension", rel_path)
        return []

    try:
        with open(full_path, "rb") as f:
            source_bytes = f.read()
    except OSError as exc:
        logger.debug("Could not read %s: %s", rel_path, exc)
        return []

    try:
        import tree_sitter

        parser = _get_parser(lang)
        query = _get_query(lang)
        tree = parser.parse(source_bytes)
    except Exception as exc:  # pragma: no cover — tree-sitter setup errors
        logger.warning("Tree-sitter parse failed for %s: %s", rel_path, exc)
        return []

    symbols: list[Symbol] = []
    # tree-sitter 0.23+: QueryCursor.matches returns
    # ``[(pattern_index, {capture_name: [nodes]})]``.
    cursor = tree_sitter.QueryCursor(query)
    matches = cursor.matches(tree.root_node)
    for _pattern_idx, captures in matches:
        def_nodes = captures.get("def", [])
        name_nodes = captures.get("name", [])
        if not def_nodes or not name_nodes:
            continue
        def_node = def_nodes[0]
        name_node = name_nodes[0]
        node_type = def_node.type
        symbol_type = _NODE_TYPE_TO_SYMBOL_TYPE.get(node_type, "symbol")

        try:
            symbol_name = source_bytes[name_node.start_byte : name_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            raw_code = source_bytes[def_node.start_byte : def_node.end_byte].decode(
                "utf-8", errors="replace"
            )
        except (UnicodeDecodeError, AttributeError) as exc:
            logger.debug("Decode error in %s: %s", rel_path, exc)
            continue

        if not symbol_name or not raw_code:
            continue

        symbols.append(
            Symbol(
                file_path=rel_path,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                raw_code=raw_code,
                start_byte=def_node.start_byte,
                end_byte=def_node.end_byte,
                lang=lang,
            )
        )

    return symbols
