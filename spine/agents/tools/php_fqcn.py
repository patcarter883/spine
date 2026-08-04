"""PHP fully-qualified-class-name resolution and repair.

Motivated by agripath probe 26 (run 3a7a3597), which parked ``needs_review``
with 3 of 4 slices never implemented. PHP FQCNs contain backslashes, which are
invalid JSON escapes, so when the model emits one into a JSON string field it
drops the separators AND stutters a tail segment::

    App\\Infrastructure\\Models\\Traits\\FarmScoped
        -> "AppInfrastructureModelsTraitsFarmScopedModelsTraitsFarmScoped"
    Illuminate\\Database\\Eloquent\\Concerns\\HasUuids
        -> "IlluminateDatabaseEloquentConcernsHasUuidsConcernsHasUuids"

``_scrub_phantom_refs`` then dropped those as not-in-index, which dangled the
cross-slice dependency edges (3 slices blocked) and robbed the editor of the
list that told it what to import — so it wrote ``use FarmScoped;`` into a class
body with no matching import statement, a PHP fatal.

The corruption is generation variance, not a code path: the FIRST plan of that
same run emitted the same FQCNs intact. It is therefore not fixable by prompt,
and this module supplies the deterministic ground truth instead.

Two corpora, because neither alone is sufficient:

* **PSR-4 walk** of the workspace — first-party classes. Walked fresh on every
  call, deliberately: a slice routinely imports a class an earlier slice just
  created, and a cached corpus would not see it.
* **composer's autoload classmap** — vendor/framework classes (``Illuminate\\*``),
  which are absent from the codebase index entirely (0 indexed rows under
  ``vendor/``). Cached on (path, mtime); vendor does not move mid-run.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

# A normalised key shorter than this is too collision-prone to anchor a
# longest-prefix match ("App" is a prefix of every App\\* class). Real
# namespaced FQCNs comfortably exceed it.
_MIN_KEY_LEN = 12

# The matched prefix must account for at least this share of the broken
# symbol. Guards against a short-but-legal key claiming a long unrelated
# string; stutter-corrupted names retain ~60-70% coverage.
_MIN_COVERAGE = 0.5

# Bound the PSR-4 walk so a pathological tree cannot stall the write path.
_MAX_PSR4_FILES = 20000

# PHP's global namespace, which no autoload corpus contains. Sourced from
# `get_declared_classes() + get_declared_interfaces()` on PHP 8.5 with a stock
# extension set, filtered to names with no namespace separator. Needed because
# a builtin whose leaf is unique in vendor/ would otherwise look importable.
PHP_GLOBAL_CLASSES = frozenset({
    "AllowDynamicProperties", "AppendIterator", "ArgumentCountError", "ArithmeticError",
    "ArrayAccess", "ArrayIterator", "ArrayObject", "AssertionError", "Attribute",
    "BackedEnum", "BadFunctionCallException", "BadMethodCallException", "CURLFile",
    "CURLStringFile", "CachingIterator", "CallbackFilterIterator",
    "ClosedGeneratorException", "Closure", "CompileError", "Countable", "CurlHandle",
    "CurlMultiHandle", "CurlShareHandle", "CurlSharePersistentHandle", "DOMAttr",
    "DOMCdataSection", "DOMCharacterData", "DOMChildNode", "DOMComment", "DOMDocument",
    "DOMDocumentFragment", "DOMDocumentType", "DOMElement", "DOMEntity",
    "DOMEntityReference", "DOMException", "DOMImplementation", "DOMNameSpaceNode",
    "DOMNamedNodeMap", "DOMNode", "DOMNodeList", "DOMNotation", "DOMParentNode",
    "DOMProcessingInstruction", "DOMText", "DOMXPath", "DateError", "DateException",
    "DateInterval", "DateInvalidOperationException", "DateInvalidTimeZoneException",
    "DateMalformedIntervalStringException", "DateMalformedPeriodStringException",
    "DateMalformedStringException", "DateObjectError", "DatePeriod", "DateRangeError",
    "DateTime", "DateTimeImmutable", "DateTimeInterface", "DateTimeZone",
    "DeflateContext", "DelayedTargetValidation", "Deprecated", "Directory",
    "DirectoryIterator", "DivisionByZeroError", "DomainException", "EmptyIterator",
    "Error", "ErrorException", "Exception", "Fiber", "FiberError",
    "FilesystemIterator", "FilterIterator", "Generator", "GlobIterator", "HashContext",
    "InfiniteIterator", "InflateContext", "InternalIterator", "InvalidArgumentException",
    "Iterator", "IteratorAggregate", "IteratorIterator", "JsonException",
    "JsonSerializable", "LengthException", "LibXMLError", "LimitIterator",
    "LogicException", "MultipleIterator", "NoDiscard", "NoRewindIterator",
    "OpenSSLAsymmetricKey", "OpenSSLCertificate", "OpenSSLCertificateSigningRequest",
    "OutOfBoundsException", "OutOfRangeException", "OuterIterator", "OverflowException",
    "Override", "PDO", "PDOException", "PDORow", "PDOStatement", "ParentIterator",
    "ParseError", "Phar", "PharData", "PharException", "PharFileInfo", "PhpToken",
    "PropertyHookType", "RangeException", "RecursiveArrayIterator",
    "RecursiveCachingIterator", "RecursiveCallbackFilterIterator",
    "RecursiveDirectoryIterator", "RecursiveFilterIterator", "RecursiveIterator",
    "RecursiveIteratorIterator", "RecursiveRegexIterator", "RecursiveTreeIterator",
    "Reflection", "ReflectionAttribute", "ReflectionClass", "ReflectionClassConstant",
    "ReflectionConstant", "ReflectionEnum", "ReflectionEnumBackedCase",
    "ReflectionEnumUnitCase", "ReflectionException", "ReflectionExtension",
    "ReflectionFiber", "ReflectionFunction", "ReflectionFunctionAbstract",
    "ReflectionGenerator", "ReflectionIntersectionType", "ReflectionMethod",
    "ReflectionNamedType", "ReflectionObject", "ReflectionParameter",
    "ReflectionProperty", "ReflectionReference", "ReflectionType", "ReflectionUnionType",
    "ReflectionZendExtension", "Reflector", "RegexIterator", "RequestParseBodyException",
    "ReturnTypeWillChange", "RoundingMode", "RuntimeException", "SeekableIterator",
    "SensitiveParameter", "SensitiveParameterValue", "Serializable", "SessionHandler",
    "SessionHandlerInterface", "SessionIdInterface",
    "SessionUpdateTimestampHandlerInterface", "SimpleXMLElement", "SimpleXMLIterator",
    "SplDoublyLinkedList", "SplFileInfo", "SplFileObject", "SplFixedArray", "SplHeap",
    "SplMaxHeap", "SplMinHeap", "SplObjectStorage", "SplObserver", "SplPriorityQueue",
    "SplQueue", "SplStack", "SplSubject", "SplTempFileObject", "StreamBucket",
    "Stringable", "Throwable", "Traversable", "TypeError", "UnderflowException",
    "UnexpectedValueException", "UnhandledMatchError", "UnitEnum", "ValueError",
    "WeakMap", "WeakReference", "XMLParser", "XMLReader", "XMLWriter", "ZipArchive",
    "__PHP_Incomplete_Class", "finfo", "php_user_filter", "stdClass",
})


def _psr4_map(workspace_root: str) -> dict[str, str]:
    """PSR-4 prefix->dir mapping from the workspace composer.json, {} on failure.

    One root per prefix, for callers that resolve a single expected location.
    Use :func:`_psr4_roots` when every root matters.
    """
    return {prefix: roots[0] for prefix, roots in _psr4_roots(workspace_root)}


def _psr4_roots(workspace_root: str) -> list[tuple[str, list[str]]]:
    """PSR-4 prefix -> ALL its roots, [] on failure.

    composer allows a LIST of directories per prefix and laravel/framework
    itself uses that form ("Illuminate\\Support\\": [...]); taking only the
    ``str`` form silently emptied the first-party half of the corpus.
    """
    try:
        cfg = json.loads(
            (Path(workspace_root) / "composer.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    out: list[tuple[str, list[str]]] = []
    for section in ("autoload", "autoload-dev"):
        for prefix, rel in ((cfg.get(section) or {}).get("psr-4") or {}).items():
            if not isinstance(prefix, str):
                continue
            roots = [rel] if isinstance(rel, str) else [
                r for r in rel if isinstance(r, str)
            ] if isinstance(rel, list) else []
            if roots:
                out.append((prefix, roots))
    return out


def _psr4_namespace_for(rel_path: str, psr4: dict[str, str]) -> Optional[str]:
    """The namespace PSR-4 dictates for *rel_path*, None when unmapped."""
    posix = rel_path.replace("\\", "/")
    for prefix, base in sorted(psr4.items(), key=lambda kv: -len(kv[1])):
        base = base.rstrip("/") + "/"
        if posix.startswith(base):
            parts = posix[len(base):].split("/")[:-1]  # drop the filename
            ns = prefix.rstrip("\\")
            if parts:
                ns += "\\" + "\\".join(parts)
            return ns
    return None


def normalise_fqcn(name: str) -> str:
    """Lowercase alphanumeric skeleton of *name*.

    Collapses every separator PHP or a mangling model might use — ``\\``,
    ``/``, ``_``, ``.``, ``::`` — so an intact FQCN and its backslash-stripped
    corruption share the same skeleton.
    """
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


@lru_cache(maxsize=8)
def _classmap_fqcns(classmap_path: str, mtime: float) -> tuple[str, ...]:
    """Every FQCN in a composer autoload classmap, () on failure.

    *mtime* is part of the cache key only, so a regenerated classmap is
    re-read rather than served stale.
    """
    try:
        text = Path(classmap_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    # The classmap is PHP source, so every separator is written escaped
    # ('Illuminate\\Database\\...'). Unescape, or the FQCN we hand back
    # becomes a broken `use Illuminate\\Database\\...;` statement.
    return tuple(
        n.replace("\\\\", "\\")
        for n in re.findall(r"'([A-Za-z0-9_\\]+)'\s*=>", text)
        if n
    )


def _classmap_path(workspace_root: str) -> Optional[Path]:
    """The composer classmap in reach of *workspace_root*, else None.

    Resolved workspace-relative, NOT CWD-relative: during IMPLEMENT the
    workspace is a sandbox worktree and CWD is not it.

    ``vendor/`` is gitignored in a Laravel repo, so a fresh sandbox WORKTREE
    has no vendor tree at all and the whole framework half of the corpus would
    silently be empty exactly where the write path runs. Fall back to the
    worktree's main checkout, which is where the landing gate also sources
    vendor from (it bind-mounts the clone's vendor into the test container).
    """
    ws = Path(workspace_root or ".")
    rel = Path("vendor") / "composer" / "autoload_classmap.php"
    direct = ws / rel
    if direct.is_file():
        return direct

    # A linked worktree's .git is a FILE pointing at <main>/.git/worktrees/<n>.
    try:
        dotgit = ws / ".git"
        if dotgit.is_file():
            text = dotgit.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.split(":", 1)[1].strip())
                # .../<main>/.git/worktrees/<name> -> <main>
                main_root = gitdir.parent.parent.parent
                candidate = main_root / rel
                if candidate.is_file():
                    return candidate
    except OSError:
        return None
    return None


def fqcn_corpus(workspace_root: str) -> dict[str, str]:
    """Map normalised-skeleton -> canonical FQCN for every class in reach.

    First-party PSR-4 entries win over vendor on a skeleton collision: a repo
    class shadowing a framework name is the one a slice means.
    """
    corpus: dict[str, str] = {}

    cm = _classmap_path(workspace_root)
    if cm is not None:
        try:
            mtime = cm.stat().st_mtime
        except OSError:
            mtime = 0.0
        for fqcn in _classmap_fqcns(str(cm), mtime):
            corpus.setdefault(normalise_fqcn(fqcn), fqcn)

    ws = Path(workspace_root or ".")
    seen = 0
    for prefix, roots in _psr4_roots(workspace_root):
        for base in roots:
            root = ws / base.rstrip("/")
            if not root.is_dir():
                continue
            for f in root.rglob("*.php"):
                seen += 1
                if seen > _MAX_PSR4_FILES:
                    break
                # Derive the namespace from the root being walked, NOT via
                # _psr4_map: that keeps one root per prefix, so every class
                # under a second root ("App\\": ["app/", "src/"]) resolved to
                # None and vanished from the corpus.
                try:
                    parts = f.relative_to(root).parts[:-1]
                except ValueError:  # noqa: PERF203 — outside the root, skip
                    continue
                ns = prefix.rstrip("\\")
                if parts:
                    ns += "\\" + "\\".join(parts)
                fq = f"{ns}\\{f.stem}"
                # First-party overwrites a vendor entry with the same skeleton.
                corpus[normalise_fqcn(fq)] = fq
            if seen > _MAX_PSR4_FILES:
                break
        if seen > _MAX_PSR4_FILES:
            break

    return corpus


def repair_mangled_fqcn(
    symbol: str, workspace_root: str, corpus: Optional[dict[str, str]] = None
) -> Optional[str]:
    """Recover the real FQCN behind a separator-stripped, tail-stuttered name.

    Takes the LONGEST corpus entry whose skeleton is a prefix of *symbol*'s
    skeleton, which absorbs both corruptions at once: the stripping is undone
    by matching on skeletons, and the stuttered tail is simply the unmatched
    remainder. Longest-wins also disambiguates ``Farm`` from ``FarmUser``.

    Returns None for anything already intact (contains ``\\``, ``/`` or ``.``),
    too short to match safely, or with no confident candidate.

    Pass *corpus* when repairing several symbols against the same workspace:
    building it walks the PSR-4 tree, so rebuilding per symbol is pure waste.
    """
    s = (symbol or "").strip()
    if not s or "\\" in s or "/" in s or "." in s or "::" in s:
        return None  # intact FQCN, path, or dotted form — not our corruption
    if not s[0].isupper():
        return None
    key = normalise_fqcn(s)
    if len(key) < _MIN_KEY_LEN:
        return None

    best: Optional[str] = None
    best_len = 0
    for nk, fqcn in (fqcn_corpus(workspace_root) if corpus is None else corpus).items():
        if len(nk) < _MIN_KEY_LEN or len(nk) <= best_len:
            continue
        if key.startswith(nk) and _is_stutter(key[len(nk):], nk):
            best, best_len = fqcn, len(nk)

    if best is None or best_len / len(key) < _MIN_COVERAGE:
        return None
    return best


def _is_stutter(remainder: str, matched: str) -> bool:
    """True when *remainder* is the duplicated-tail artefact of *matched*.

    Without this check a prefix match alone will happily rewrite a symbol for a
    class that does not exist YET onto a different class that does — and
    not-yet-existing is exactly the population _scrub_phantom_refs feeds in
    (its own log line calls them "new method or planner forward-reference").
    ``AppDomainFarmModelsNewThing`` prefix-matches ``App\\Domain\\Farm\\Models``
    with remainder ``newthing``, which is NOT a repeat of anything in the match
    and so must be refused; the genuine corruption leaves a remainder that
    already appears inside the matched key.
    """
    return not remainder or remainder in matched


def resolve_class_fqcn(
    basename: str, workspace_root: str, corpus: Optional[dict[str, str]] = None
) -> Optional[str]:
    """The unique FQCN whose leaf is *basename*, None when absent or ambiguous.

    Ambiguity is deliberately fatal rather than guessed: importing the wrong
    ``Factory`` is worse than leaving the model's output alone, and verify
    hard-fails a wrong PSR-4 import.

    Pass *corpus* when resolving several names against the same workspace —
    one file's worth of imports rebuilt it ten times over at ~15 ms a walk.
    """
    if not basename or not basename[0].isupper():
        return None
    if basename in PHP_GLOBAL_CLASSES:
        # An unqualified builtin already resolves to the global namespace.
        # 16 of them have a UNIQUE same-leaf vendor namesake in a stock Laravel
        # tree (DateTimeImmutable -> Monolog\DateTimeImmutable, SplFileInfo ->
        # Symfony\Component\Finder\SplFileInfo, Throwable -> PHPUnit\...), so
        # "unique in the corpus" is NOT evidence an import is wanted — adding
        # one rebinds working code to the wrong class, and `php -l` still
        # passes so neither the write gate nor the php_syntax gate sees it.
        return None
    entries = fqcn_corpus(workspace_root) if corpus is None else corpus
    hits = {
        fqcn
        for fqcn in entries.values()
        if fqcn.rsplit("\\", 1)[-1] == basename
    }
    return next(iter(hits)) if len(hits) == 1 else None
