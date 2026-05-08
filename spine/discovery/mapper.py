"""Codebase Mapper — scans directory structure, detects imports, and builds
a complete code map with dependency graphs for brownfields discovery.

This module is the first phase of the Brownfields Discovery & Analysis Engine.
It produces a `CodeMap` which feeds into the Analyzer and ReverseEngineer phases.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.hierarchy import RalphLoopEngine
from ..models.types import NodeStatus, PhaseNode, ProjectNode

# ═══════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CodebaseFileInfo:
    """Metadata about a single file in the scanned codebase."""

    path: Path
    name: str
    extension: str
    language: str
    size_bytes: int

    @property
    def is_test(self) -> bool:
        """Detect if file is in a test directory."""
        parts = self.path.parts
        return any(
            p.lower() in ("tests", "test", "spec", "__tests__", "testing")
            for p in parts
        )

    @property
    def relative_path(self) -> str:
        """Path relative to workspace, or just the filename."""
        return str(self.path)

    @property
    def is_python(self) -> bool:
        return self.language == "python"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "extension": self.extension,
            "language": self.language,
            "size_bytes": self.size_bytes,
            "is_test": self.is_test,
        }


@dataclass
class ImportInfo:
    """Information about a single import statement."""

    source_file: str  # relative path of file containing the import
    imported_name: str  # what is imported (e.g., "FastAPI", "os", "helper")
    module_path: str  # full module path (e.g., "fastapi", ".utils")
    import_type: str  # "stdlib", "third_party", "relative", "local"
    line_number: int = 0
    is_conditional: bool = False

    @property
    def is_relative(self) -> bool:
        """Check if this is a relative import."""
        return self.module_path.startswith(".")

    @property
    def is_local(self) -> bool:
        """Check if this imports from within the project."""
        return self.import_type == "local" or self.import_type == "relative"


# Known stdlib modules (Python 3.11+)
_STDLIB: Set[str] = {
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
    "bisect", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath",
    "cmd", "code", "codecs", "codeop", "collections", "colorsys",
    "compileall", "concurrent", "configparser", "contextlib", "copy",
    "copyreg", "cProfile", "crypt", "csv", "ctypes", "curses",
    "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
    "distutils", "doctest", "email", "encodings", "enum", "errno",
    "faulthandler", "fcntl", "filecmp", "fileinput", "fnmatch",
    "formatter", "fractions", "ftplib", "functools", "gc", "getopt",
    "getpass", "gettext", "glob", "grp", "gzip", "hashlib", "heapq",
    "hmac", "html", "http", "idlelib", "imaplib", "imghdr", "imp",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "lib2to3", "linecache", "locale", "logging", "lzma",
    "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
    "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
    "numbers", "operator", "os", "ossaudiodev", "parser", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile",
    "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc",
    "queue", "quopri", "random", "re", "readline", "reprlib",
    "resource", "rlcompleter", "runpy", "sched", "secrets",
    "select", "selectors", "shelve", "shlex", "shutil", "signal",
    "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
    "spwd", "sqlite3", "ssl", "stat", "statistics", "string",
    "stringprep", "struct", "subprocess", "sunau", "symtable",
    "sys", "sysconfig", "syslog", "tabnanny", "tarfile", "telnetlib",
    "tempfile", "termios", "test", "textwrap", "threading", "time",
    "timeit", "tkinter", "token", "tokenize", "trace", "traceback",
    "tracemalloc", "tty", "turtle", "turtledemo", "types",
    "typing", "unicodedata", "unittest", "urllib", "uu", "uuid",
    "venv", "warnings", "wave", "weakref", "webbrowser", "winreg",
    "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp",
    "zipfile", "zipimport", "zlib",
    # Additional from Python 3.11+
    "tomllib", "contextvars",
}


def _classify_import(module_path: str) -> str:
    """Classify a module path as stdlib, third_party, relative, or local."""
    if module_path.startswith("."):
        return "relative"
    top_level = module_path.split(".")[0]
    if top_level in _STDLIB:
        return "stdlib"
    return "third_party"


# ═══════════════════════════════════════════════════════════════════
# CodeMap
# ═══════════════════════════════════════════════════════════════════


@dataclass
class CodeMap:
    """Complete codebase map produced by the mapper phase.

    Contains everything needed by the Analyzer and ReverseEngineer phases:
    - File tree structure
    - Language breakdown
    - All imports
    - Dependency graph (file → set of dependency files)
    """

    root_path: Path
    total_files: int = 0
    total_dirs: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    files: List[CodebaseFileInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    file_tree: Dict[str, Any] = field(default_factory=dict)

    def _recalc_counts(self) -> None:
        """Recalculate counts from files list."""
        self.total_files = len(self.files)
        self.languages.clear()
        for f in self.files:
            lang = f.language or "unknown"
            self.languages[lang] = self.languages.get(lang, 0) + 1


# ═══════════════════════════════════════════════════════════════════
# Language Detection
# ═══════════════════════════════════════════════════════════════════

_LANG_MAP: Dict[str, str] = {
    ".py": "python",
    ".pyx": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".cpp": "c++",
    ".cc": "c++",
    ".cxx": "c++",
    ".h": "c",
    ".hpp": "c++",
    ".rb": "ruby",
    ".m": "objective-c",
    ".mm": "objective-c++",
    ".swift": "swift",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".pl": "perl",
    ".pm": "perl",
    ".php": "php",
    ".r": "r",
    ".scala": "scala",
    ".cs": "c#",
    ".fs": "f#",
    ".elm": "elm",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".lhs": "haskell",
    ".lua": "lua",
    ".dart": "dart",
    ".groovy": "groovy",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".sql": "sql",
    ".dockerfile": "dockerfile",
    ".makefile": "makefile",
    ".cmake": "cmake",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
}


def _detect_language(path: Path) -> str:
    """Detect language from file extension."""
    suffix = path.suffix.lower()
    if not suffix and path.name.lower() in ("dockerfile", "makefile"):
        return "dockerfile" if path.name.lower() == "dockerfile" else "makefile"
    return _LANG_MAP.get(suffix, "unknown")


# Default directories to always ignore
_DEFAULT_IGNORE: Set[str] = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".eggs", "node_modules",
    ".venv", "venv", "env", ".env", "virtualenv",
    ".idea", ".vscode", ".vs", ".DS_Store",
    "dist", "build", "target", "out",
    ".spine", ".hive", ".opencode",
    "egg-info", ".egg-info",
}

# Default patterns to ignore
_DEFAULT_IGNORE_PATTERNS: List[str] = [
    "**/__pycache__/**",
    "**/.git/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/dist/**",
    "**/build/**",
    "**/*.egg-info/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/__pycache__",
]

# ═══════════════════════════════════════════════════════════════════
# CodebaseMapper
# ═══════════════════════════════════════════════════════════════════


class CodebaseMapper:
    """Scans a codebase directory and builds a complete CodeMap.

    The mapper is the first phase of the Brownfields Discovery pipeline.
    It produces structured output that feeds into:
    - Analyzer: for pattern/architecture detection
    - ReverseEngineer: for spec inference

    Usage:
        mapper = CodebaseMapper(ignore_patterns=["tests/**"])
        code_map = mapper.map_codebase(Path("/path/to/project"))
        # code_map.files, code_map.imports, code_map.dependency_graph

    Output feeds into Ralph Loop hierarchy:
        project.metadata["code_map"] = code_map
    """

    def __init__(
        self,
        ignore_patterns: Optional[List[str]] = None,
        include_extensions: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        max_file_size_mb: int = 10,
    ):
        """Initialize mapper with configuration.

        Args:
            ignore_patterns: Glob patterns to ignore.
            include_extensions: Only include files with these extensions.
            max_depth: Maximum directory depth to scan.
            max_file_size_mb: Skip files larger than this.
        """
        self.ignore_patterns = ignore_patterns or _DEFAULT_IGNORE_PATTERNS
        self.include_extensions = include_extensions
        self.max_depth = max_depth
        self.max_file_size_mb = max_file_size_mb

    # ── Scanning ──────────────────────────────────────────────────

    def scan(self, root: Path) -> CodeMap:
        """Scan directory tree and build file inventory.

        Args:
            root: Root directory to scan.

        Returns:
            CodeMap with file listing and language breakdown.
        """
        root = root.resolve()
        cm = CodeMap(root_path=root)

        dirs_scanned = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # Compute depth relative to root
            rel_path = Path(dirpath).relative_to(root)
            depth = len(rel_path.parts) if str(rel_path) != "." else 0

            # Skip if exceeds max depth
            if self.max_depth is not None and depth > self.max_depth:
                dirnames.clear()
                continue

            # Filter out ignored directories in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in _DEFAULT_IGNORE
                and not d.startswith(".")
            ]

            dirs_scanned += 1

            for fname in filenames:
                fpath = Path(dirpath) / fname
                file_rel = str(fpath.relative_to(root))

                # Check ignore patterns
                if self._should_ignore(file_rel):
                    continue

                # Check include extensions
                suffix = fpath.suffix.lower()
                if self.include_extensions and suffix not in self.include_extensions:
                    continue

                # Check file size
                try:
                    size = fpath.stat().st_size
                    if size > self.max_file_size_mb * 1024 * 1024:
                        continue
                except OSError:
                    continue

                language = _detect_language(fpath)
                cm.files.append(CodebaseFileInfo(
                    path=fpath,
                    name=fname,
                    extension=suffix,
                    language=language,
                    size_bytes=size,
                ))

        cm.total_dirs = dirs_scanned
        cm._recalc_counts()
        return cm

    def file_tree(self, root: Path, max_depth: int = 10) -> Dict[str, Any]:
        """Build a nested dictionary representation of the file tree.

        Args:
            root: Root directory.
            max_depth: Maximum depth for tree.

        Returns:
            Nested dict representing directory structure.
        """
        def _build_tree(path: Path, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {"...": "..."}

            result: Dict[str, Any] = {}
            try:
                for entry in sorted(path.iterdir()):
                    name = entry.name
                    if entry.name in _DEFAULT_IGNORE or entry.name.startswith("."):
                        continue
                    if self._should_ignore(str(entry.relative_to(root))):
                        continue
                    if entry.is_dir():
                        subtree = _build_tree(entry, depth + 1)
                        if subtree:
                            result[name + "/"] = subtree
                    else:
                        result[name] = None
            except (OSError, PermissionError):
                result["<error>"] = "access denied"
            return result

        root = root.resolve()
        return _build_tree(root)

    def _should_ignore(self, rel_path: str) -> bool:
        """Check if a relative path matches any ignore pattern."""
        import fnmatch
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Also check if path matches trailing pattern
            parts = rel_path.split("/")
            for pattern_part in pattern.replace("**/", "").split("/"):
                if pattern_part in parts:
                    return True
        return False

    # ── Import Detection ──────────────────────────────────────────

    def detect_imports(self, root: Path) -> List[ImportInfo]:
        """Detect all imports in Python files within the codebase.

        Parses AST of each .py file to extract import statements.

        Args:
            root: Root directory of the codebase.

        Returns:
            List of ImportInfo objects.
        """
        imports: List[ImportInfo] = []
        root = root.resolve()

        for py_file in root.rglob("*.py"):
            rel = str(py_file.relative_to(root))
            if self._should_ignore(rel):
                continue
            if "__pycache__" in py_file.parts:
                continue

            try:
                source = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            file_imports = self._parse_imports(source, rel)
            imports.extend(file_imports)

        return imports

    def _parse_imports(self, source: str, source_file: str) -> List[ImportInfo]:
        """Parse import statements from Python source using AST.

        Args:
            source: Python source code.
            source_file: Relative path of the source file.

        Returns:
            List of ImportInfo objects.
        """
        imports: List[ImportInfo] = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_path = alias.name
                    imports.append(ImportInfo(
                        source_file=source_file,
                        imported_name=alias.asname or alias.name.split(".")[0],
                        module_path=module_path,
                        import_type=_classify_import(module_path),
                        line_number=node.lineno,
                    ))

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue  # Relative import with no module
                module_path = node.module
                level = node.level or 0
                if level > 0:
                    module_path = "." * level + module_path

                for alias in node.names:
                    imports.append(ImportInfo(
                        source_file=source_file,
                        imported_name=alias.asname or alias.name,
                        module_path=module_path,
                        import_type=_classify_import(module_path),
                        line_number=node.lineno,
                    ))

        return imports

    # ── Dependency Graph ──────────────────────────────────────────

    def build_dependency_graph(
        self,
        code_map: CodeMap,
        imports: Optional[List[ImportInfo]] = None,
    ) -> Dict[str, List[str]]:
        """Build a dependency graph from imports.

        Maps each file to the list of files it depends on.
        For relative imports, resolves to actual project files.

        Args:
            code_map: The code map with file info.
            imports: Import list (uses code_map.imports if None).

        Returns:
            Dict mapping file_path → list of dependency file_paths.
        """
        all_imports = imports or code_map.imports

        # Build index of Python modules by name (relative to root)
        module_index: Dict[str, str] = {}
        root_path = code_map.root_path
        for f in code_map.files:
            if f.extension == ".py":
                # Compute path relative to project root
                try:
                    rel = str(f.path.relative_to(root_path))
                except ValueError:
                    rel = f.name
                # Map module name to relative file
                mod_name = f.name.replace(".py", "")
                module_index[mod_name] = rel
                # Also map dotted paths
                dotted = str(Path(rel).with_suffix("")).replace(os.sep, ".")
                module_index[dotted] = rel

        graph: Dict[str, List[str]] = {}

        for imp in all_imports:
            src = imp.source_file
            if src not in graph:
                graph[src] = []

            # Resolve local/relative imports to actual files
            if imp.is_relative:
                src_dir = os.path.dirname(src)
                # Remove leading dots and resolve
                clean_path = imp.module_path.lstrip(".")
                depth = len(imp.module_path) - len(clean_path)
                for _ in range(depth - 1):
                    src_dir = os.path.dirname(src_dir) if src_dir else src_dir
                resolved = os.path.normpath(os.path.join(
                    src_dir, clean_path.replace(".", os.sep) + ".py"
                ))
                if resolved in module_index.values():
                    graph[src].append(resolved)

            elif imp.import_type == "local":
                resolved = imp.module_path.replace(".", os.sep) + ".py"
                if resolved in module_index.values():
                    graph[src].append(resolved)

            # Third-party and stdlib are external dependencies
            # (recorded but not linking to project files)

        code_map.dependency_graph = graph
        return graph

    # ── Full Pipeline ─────────────────────────────────────────────

    def map_codebase(self, root: Path) -> CodeMap:
        """Run the full mapping pipeline: scan → detect imports → build graph.

        Args:
            root: Root directory of the codebase.

        Returns:
            Complete CodeMap with files, imports, and dependency graph.
        """
        root = root.resolve()

        # Phase 1: Scan file tree
        cm = self.scan(root)

        # Phase 2: Detect imports
        cm.imports = self.detect_imports(root)

        # Phase 3: Build dependency graph
        self.build_dependency_graph(cm)

        # Phase 4: Build file tree
        cm.file_tree = self.file_tree(root)

        return cm

    # ── Ralph Loop Integration ────────────────────────────────────

    def to_hierarchy_phase(
        self,
        code_map: CodeMap,
        engine: Optional[RalphLoopEngine] = None,
        parent_project: Optional[ProjectNode] = None,
    ) -> PhaseNode:
        """Create a Ralph Loop PhaseNode populated with mapping results.

        Args:
            code_map: The completed CodeMap.
            engine: Optional RalphLoopEngine instance.
            parent_project: Optional parent project to attach to.

        Returns:
            PhaseNode with mapping metadata.
        """
        node = PhaseNode(
            id=f"discovery-map-{code_map.root_path.name}",
            name="Codebase Mapping",
            status=NodeStatus.SUCCESS,
            progress=100.0,
            metadata={
                "total_files": code_map.total_files,
                "total_dirs": code_map.total_dirs,
                "languages": code_map.languages,
                "import_count": len(code_map.imports),
                "dependency_count": sum(
                    len(deps) for deps in code_map.dependency_graph.values()
                ),
            },
        )

        if engine and parent_project:
            parent_project.phases.append(node)

        return node


__all__ = [
    "CodebaseFileInfo",
    "ImportInfo",
    "CodeMap",
    "CodebaseMapper",
]