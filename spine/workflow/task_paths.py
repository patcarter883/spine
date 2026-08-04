"""Deterministic pre-read of the files a task names by path.

When a task says "following the conventions of the existing
``app/Domain/Farm/Models/Farm.php``", that path is not a hint to be scored
against a similarity threshold — it is an instruction. This module resolves
such paths off the description text and reads them verbatim, before any
model runs.

Why this exists (probe 25, agripath UnitOfMeasure, 2026-07-10). The task
named ``Farm.php`` explicitly. PLAN then spent 17m50s across 9 research
topics, surfaced 17 files, and never retrieved it — semantic recall went to
``FarmScope``/``RelationshipFarmScope``/``Business`` instead. The trait the
exemplar would have shown (``use FarmScoped;``) appears nowhere in any
artifact of that run, and the landed model omitted it. The same task run
through a plain tool-using agent opened the named file as its first action
and used the trait.

Two properties matter and neither is negotiable through a retrieval layer:

* **Exact.** A path in the task is resolved by the filesystem, not ranked.
* **Verbatim.** ``_research_text`` compacts every recall hit to a 240-char
  summary, so even a successful retrieval would have handed the planner
  prose about the file rather than the file. Source read here bypasses that
  compaction, in the manner of the A2 findings ledger.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# A path-ish token: at least one '/' separating word-ish segments. Trailing
# sentence punctuation is stripped separately so "…/Farm.php," resolves.
_PATH_RE = re.compile(r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+")

# Suffixes worth inlining as an exemplar. Deliberately code-only: a task
# naming a markdown file wants it as instructions (already in the prompt),
# not as a source exemplar.
_READABLE_SUFFIXES = frozenset(
    {
        ".py", ".php", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs",
        ".java", ".rb", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs",
        ".sql", ".sh", ".yaml", ".yml", ".toml",
    }
)

# Directory names that never hold an exemplar worth inlining.
_SKIP_PARTS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", "vendor",
     "dist", "build", ".spine", ".planning", "site-packages"}
)

_MAX_FILES = 4
_MAX_CHARS_PER_FILE = 4000


def extract_task_paths(description: str) -> list[str]:
    """Ordered, de-duplicated path-like tokens appearing in *description*.

    Purely textual — makes no filesystem claim. Callers resolve.
    """
    if not description:
        return []
    seen: list[str] = []
    for raw in _PATH_RE.findall(description):
        tok = raw.strip().strip("`'\"").rstrip(".,;:)")
        if not tok or tok in seen:
            continue
        seen.append(tok)
    return seen


def _iter_candidate_files(root: Path, token: str) -> list[Path]:
    """Files a single task token designates, [] when it designates none.

    A token may name a file directly, or a directory — in which case its
    immediate code children are candidates (a task pointing at
    ``database/migrations`` wants the migrations as exemplars).
    """
    target = (root / token).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:  # escapes the workspace — refuse
        return []
    if target.is_file():
        return [target] if target.suffix in _READABLE_SUFFIXES else []
    if target.is_dir():
        return sorted(
            (
                p
                for p in target.iterdir()
                if p.is_file() and p.suffix in _READABLE_SUFFIXES
            ),
            key=lambda p: p.name,
        )
    return []


def read_task_named_sources(
    description: str,
    workspace_root: str | Path,
    *,
    max_files: int = _MAX_FILES,
    max_chars_per_file: int = _MAX_CHARS_PER_FILE,
) -> list[dict[str, object]]:
    """Read the files *description* names by path.

    Returns ``[{"path": str, "source": str, "truncated": bool}]``, capped at
    *max_files*. Never raises: an unreadable or absent path yields nothing,
    because a pre-read that can fail the phase would be worse than the
    retrieval gap it closes.
    """
    root = Path(workspace_root or ".")
    if not root.is_dir():
        return []

    out: list[dict[str, object]] = []
    picked: set[Path] = set()
    for token in extract_task_paths(description):
        if len(out) >= max_files:
            break
        for fpath in _iter_candidate_files(root, token):
            if len(out) >= max_files:
                break
            if fpath in picked:
                continue
            if any(part in _SKIP_PARTS for part in fpath.parts):
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.debug("task pre-read: cannot read %s — %s", fpath, exc)
                continue
            picked.add(fpath)
            truncated = len(text) > max_chars_per_file
            try:
                rel = str(fpath.relative_to(root.resolve()))
            except ValueError:
                rel = str(fpath)
            out.append(
                {
                    "path": rel,
                    "source": text[:max_chars_per_file],
                    "truncated": truncated,
                }
            )
    return out


def render_task_named_sources(entries: list[dict[str, object]]) -> str:
    """Prompt block for pre-read sources, ``""`` when there are none."""
    if not entries:
        return ""
    parts: list[str] = [
        "The task names these paths explicitly. This is their CURRENT source, "
        "read from disk — it is ground truth, not a retrieval guess. Follow "
        "the conventions it actually shows (traits/base classes/imports it "
        "uses, how it declares things) rather than the conventions you expect "
        "a file of this kind to have."
    ]
    for e in entries:
        suffix = Path(str(e.get("path", ""))).suffix.lstrip(".")
        tail = "\n… [truncated]" if e.get("truncated") else ""
        parts.append(
            f"\n--- {e.get('path')} ---\n"
            f"```{suffix}\n{e.get('source')}{tail}\n```"
        )
    return "\n".join(parts)
