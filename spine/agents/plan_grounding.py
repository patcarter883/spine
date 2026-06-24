"""Deterministic grounding of a plan's writable target files.

A planner can list a file in a slice's ``target_files`` that it never actually
saw — because exploration could not retrieve it (gitignored / ``.spine`` runtime
state) — and then mis-scope it as something to *create/curate*. Trace 019ef1e5:
the GLM plan told the editor to author a root ``config.reference.yaml``; the real
file is the read-only ``.spine/config.reference.yaml`` and the task said it was
reference-only. The editor faithfully created a stray root file.

This pass reclassifies each ``target_files`` entry against the LIVE workspace
tree, with no LLM call:

* a path under ``.spine/`` is runtime state → reference-only (never authored);
* a path that exists as given → writable (grounded, existing);
* a path that does NOT exist but whose basename exists elsewhere in the tree →
  reference-only (the planner mis-pathed an existing file); the real path is
  recorded so the implementer can read it for context;
* a path that exists nowhere → kept writable (a genuine new deliverable, e.g. a
  new test file).

Demoted files move to the slice's ``reference_only_files`` and a clarifying note
is appended to ``execution_requirements`` to counter any "create it" prose the
planner emitted. A slice is never left with empty ``target_files`` — if every
target would demote, the originals are kept writable and a warning logged
(better an over-broad slice than an unactionable one).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Directories not worth indexing when matching a mis-pathed file. ``.spine`` is
# deliberately NOT skipped — that is exactly where reference files like
# config.reference.yaml live, and matching them is the point.
_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist",
}


def _norm(p: str) -> str:
    return p.strip().lstrip("/")


def build_workspace_index(workspace_root: str) -> dict[str, list[str]]:
    """Map basename -> [workspace-relative paths]. Includes ``.spine``."""
    idx: dict[str, list[str]] = {}
    root = Path(workspace_root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            try:
                rel = (Path(dirpath) / fn).relative_to(root).as_posix()
            except ValueError:
                continue
            idx.setdefault(fn, []).append(rel)
    return idx


def classify_target(
    path: str, workspace_root: str, index: dict[str, list[str]]
) -> tuple[str, list[str]]:
    """Classify one target path.

    Returns ``("writable", [])`` or ``("reference", [real_paths])`` where
    ``real_paths`` are where the mis-pathed file actually lives (``.spine`` first).
    """
    norm = _norm(path)
    if norm.startswith(".spine/"):
        return "reference", [norm]
    if (Path(workspace_root) / norm).exists():
        return "writable", []
    name = Path(norm).name
    elsewhere = [p for p in index.get(name, []) if _norm(p) != norm]
    if elsewhere:
        # surface .spine paths first, then shortest
        elsewhere.sort(key=lambda p: (not p.startswith(".spine/"), len(p)))
        return "reference", elsewhere[:3]
    return "writable", []  # exists nowhere → genuine new deliverable


def ground_slice_targets(
    slices: list[dict],
    workspace_root: str,
    index: Optional[dict[str, list[str]]] = None,
) -> list[dict]:
    """Return copies of ``slices`` with ``target_files`` split into writable
    targets and ``reference_only_files``, plus a clarifying execution note."""
    if index is None:
        index = build_workspace_index(workspace_root)
    out: list[dict] = []
    for original in slices:
        s = dict(original)
        targets = [
            t for t in (s.get("target_files") or [])
            if isinstance(t, str) and t.strip()
        ]
        writable: list[str] = []
        refonly: list[str] = [
            r for r in (s.get("reference_only_files") or []) if isinstance(r, str)
        ]
        notes: list[str] = []
        for t in targets:
            kind, real = classify_target(t, workspace_root, index)
            if kind == "writable":
                writable.append(t)
                continue
            if not any(_norm(r) == _norm(t) for r in refonly):
                refonly.append(t)
            if real and _norm(real[0]) != _norm(t):
                notes.append(
                    f"`{t}` is reference-only (the real file is `{real[0]}`); "
                    "read it for context, do NOT create or modify it"
                )
            else:
                notes.append(
                    f"`{t}` is reference-only; read it for context, do NOT "
                    "create or modify it"
                )
        # Never strand a slice with no writable target.
        if targets and not writable:
            logger.warning(
                "plan_grounding: slice %r would have no writable targets after "
                "grounding; keeping originals", s.get("id"),
            )
            out.append(s)
            continue
        if not notes:
            out.append(s)
            continue
        s["target_files"] = writable
        seen: set[str] = set()
        s["reference_only_files"] = [
            r for r in refonly if not (_norm(r) in seen or seen.add(_norm(r)))
        ]
        note = "Reference-only files (do not author): " + "; ".join(notes) + "."
        er = s.get("execution_requirements")
        if isinstance(er, list):
            s["execution_requirements"] = list(er) + [note]
        else:
            s["execution_requirements"] = (str(er or "").rstrip() + "\n\n" + note).strip()
        logger.info(
            "plan_grounding: slice %r demoted %d target(s) to reference-only: %s",
            s.get("id"), len(notes), s["reference_only_files"],
        )
        out.append(s)
    return out
