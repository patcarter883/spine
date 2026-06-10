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
class Edge:
    """A dependency edge extracted from a source file.

    ``src_symbol`` is the qualified name (``Class.method``) of the
    enclosing symbol, or ``""`` for file-level statements (e.g. top-level
    ``use`` imports). ``dst_name`` is the alias-resolved target with
    namespace prefix stripped to its last segment so it matches the
    ``symbol_name`` values stored in ``symbol_metadata``.
    """

    src_file: str
    src_symbol: str
    edge_kind: str  # use_import|new|static_call|call|extends|implements|trait_use
    dst_name: str
    lang: str


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
    parent_class: str | None = None

    @property
    def qualified_name(self) -> str:
        """``ClassName.method`` for methods nested in a class, else ``symbol_name``."""
        if self.parent_class and self.symbol_type in {"method"}:
            return f"{self.parent_class}.{self.symbol_name}"
        return self.symbol_name


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
        (enum_declaration name: (name) @name) @def
        (trait_declaration name: (name) @name) @def
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
    "enum_declaration": "enum",
    "trait_declaration": "trait",
    "variable_declarator": "function",  # arrow function assigned to const
}


_CLASS_NODE_TYPES: frozenset[str] = frozenset({
    "class_definition",  # python
    "class_declaration",  # php, typescript
    "enum_declaration",  # php
    "trait_declaration",  # php
})


def _enclosing_class_name(node: Any, source_bytes: bytes) -> str | None:
    """Walk up the tree from ``node`` and return the nearest enclosing class name.

    A method's tree-sitter ``def`` node is nested inside the class's
    body; tree-sitter exposes ``.parent`` to walk up. Returns ``None`` if
    the symbol is not inside a class. Python's ``function_definition``
    inside ``class_definition`` is treated as a method even though the
    grammar uses the same node type — handled by the caller via
    ``symbol_type`` re-tagging.
    """
    cur = getattr(node, "parent", None)
    while cur is not None:
        if cur.type in _CLASS_NODE_TYPES:
            for child in cur.children:
                if child.type in ("identifier", "name", "type_identifier"):
                    try:
                        return source_bytes[child.start_byte : child.end_byte].decode(
                            "utf-8", errors="replace"
                        )
                    except (UnicodeDecodeError, AttributeError):
                        return None
            return None
        cur = getattr(cur, "parent", None)
    return None


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
        parent_class = _enclosing_class_name(def_node, source_bytes)
        # Python's grammar has no distinct ``method_definition`` node — a
        # function nested in a class is still ``function_definition``.
        # Re-tag as ``method`` when we see one inside a class so the
        # qualified name pipeline behaves uniformly across languages.
        if symbol_type == "function" and parent_class:
            symbol_type = "method"

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
                parent_class=parent_class,
            )
        )

    return symbols


# ── Dependency-edge extraction (PHP) ─────────────────────────────────────
#
# mcp-codebase-index has no PHP support, so spine builds its own PHP
# dependency edges at index time. Other languages keep using the MCP
# server's pre-built graph; extract_edges returns [] for them.

_PHP_NAME_NODE_TYPES: frozenset[str] = frozenset({"name", "qualified_name"})

_PHP_SYMBOL_DEF_NODE_TYPES: frozenset[str] = frozenset({
    "function_definition",
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "trait_declaration",
})


def _php_node_name(node: Any, source_bytes: bytes) -> str | None:
    """Text of the first ``name``/``qualified_name`` child, or None."""
    for child in node.children:
        if child.type in _PHP_NAME_NODE_TYPES:
            return source_bytes[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            )
    return None


def _php_enclosing_symbol(node: Any, source_bytes: bytes) -> str:
    """Qualified name of the nearest enclosing PHP symbol, '' at file level."""
    cur = getattr(node, "parent", None)
    while cur is not None:
        if cur.type in _PHP_SYMBOL_DEF_NODE_TYPES:
            name = _php_node_name(cur, source_bytes)
            if name is None:
                return ""
            if cur.type == "method_declaration":
                cls = _enclosing_class_name(cur, source_bytes)
                return f"{cls}.{name}" if cls else name
            return name
        cur = getattr(cur, "parent", None)
    return ""


def extract_edges(full_path: str, rel_path: str) -> list[Edge]:
    """Extract dependency edges from a PHP source file.

    Captures top-level ``use`` imports, ``new ClassName`` instantiation,
    ``Foo::bar()`` static calls, plain/member function calls,
    ``extends`` / ``implements`` clauses, and in-class trait ``use``.
    File-local ``use`` aliases are resolved so ``dst_name`` carries the
    final class/function segment (lossy: dynamic dispatch and string
    class names are invisible to the AST).

    Returns an empty list for non-PHP files, missing files, or parse
    failures (logged at DEBUG).
    """
    lang = _infer_lang(full_path)
    if lang != "php":
        return []

    try:
        with open(full_path, "rb") as f:
            source_bytes = f.read()
    except OSError as exc:
        logger.debug("Could not read %s: %s", rel_path, exc)
        return []

    try:
        parser = _get_parser(lang)
        tree = parser.parse(source_bytes)
    except Exception as exc:  # pragma: no cover — tree-sitter setup errors
        logger.warning("Tree-sitter parse failed for %s: %s", rel_path, exc)
        return []

    def text(node: Any) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    # use-statement alias map, filled in document order (PHP requires use
    # statements before references in practice; lossy if violated).
    aliases: dict[str, str] = {}

    def resolve(raw: str) -> str:
        """Alias-resolve a (possibly namespaced) name to its last segment."""
        raw = raw.lstrip("\\")
        first, _, rest = raw.partition("\\")
        fqn = aliases.get(first)
        full = f"{fqn}\\{rest}" if (fqn and rest) else (fqn or raw)
        return full.rsplit("\\", 1)[-1]

    edges: list[Edge] = []

    def add(node: Any, kind: str, dst: str) -> None:
        dst = dst.strip()
        if dst:
            edges.append(Edge(
                src_file=rel_path,
                src_symbol=_php_enclosing_symbol(node, source_bytes),
                edge_kind=kind,
                dst_name=dst,
                lang=lang,
            ))

    # Iterative pre-order DFS keeps document order (alias map correctness).
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))
        ntype = node.type

        if ntype == "namespace_use_clause":
            # Children: (qualified_name|name) ['as' name] — the alias is a
            # bare `name` following the `as` keyword, not a wrapper node.
            target = _php_node_name(node, source_bytes)
            if not target:
                continue
            fqn = target.lstrip("\\")
            alias = None
            saw_as = False
            for child in node.children:
                if child.type == "as":
                    saw_as = True
                elif saw_as and child.type == "name":
                    alias = text(child)
                    break
            short = fqn.rsplit("\\", 1)[-1]
            aliases[alias or short] = fqn
            add(node, "use_import", short)

        elif ntype == "object_creation_expression":
            target = _php_node_name(node, source_bytes)
            if target:  # skip dynamic `new $class`
                add(node, "new", resolve(target))

        elif ntype == "scoped_call_expression":
            scope = node.child_by_field_name("scope")
            method = node.child_by_field_name("name")
            if scope is not None and scope.type in _PHP_NAME_NODE_TYPES:
                cls = resolve(text(scope))
                dst = f"{cls}.{text(method)}" if method is not None else cls
                add(node, "static_call", dst)

        elif ntype == "function_call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type in _PHP_NAME_NODE_TYPES:
                add(node, "call", resolve(text(fn)))

        elif ntype == "member_call_expression":
            method = node.child_by_field_name("name")
            if method is not None and method.type == "name":
                add(node, "call", text(method))

        elif ntype == "base_clause":
            for child in node.children:
                if child.type in _PHP_NAME_NODE_TYPES:
                    add(node, "extends", resolve(text(child)))

        elif ntype == "class_interface_clause":
            for child in node.children:
                if child.type in _PHP_NAME_NODE_TYPES:
                    add(node, "implements", resolve(text(child)))

        elif ntype == "use_declaration":  # trait use, inside a class body
            for child in node.children:
                if child.type in _PHP_NAME_NODE_TYPES:
                    add(node, "trait_use", resolve(text(child)))

    return edges
