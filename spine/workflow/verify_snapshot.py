"""Best-state ratchet for the verify→gap_plan→implement loop.

Run 019f2579: one merged slice converged 43→22→11→9 open gaps over five
cycles, then the sixth cycle's wholesale re-synthesis regressed it to 23 —
and the loop stopped with the WORSE code on disk. The editor regenerates its
edits from scratch each cycle, so every rework is a variance draw; without a
ratchet, one bad draw late in the run destroys the accumulated progress.

This module snapshots the implementation files (plus the verify artifacts
that describe them) whenever a cycle achieves a new best (lowest) total gap
count, and restores that snapshot whenever a later cycle scores worse. The
verify result mapper drives it: improvements snapshot, regressions restore —
so the loop's floor is monotone and the flagged/final state is always the
best one achieved.

Snapshots live under the work's verify artifact dir (survives cycles, dies
with the sandbox). All operations fail open — a snapshot failure must never
take verification down.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from spine.agents.artifacts import artifact_path
from spine.models.enums import PhaseName

logger = logging.getLogger(__name__)

_SNAP_DIR = "best_state"
_FINDINGS_FILE = "_best_findings.json"


def _snap_root(workspace_root: str, work_id: str) -> Path:
    return (
        Path(workspace_root)
        / artifact_path(work_id, PhaseName.VERIFY.value)
        / _SNAP_DIR
    )


def snapshot_best(
    workspace_root: str,
    work_id: str,
    files: list[str],
    findings: list[dict],
    total: int,
) -> bool:
    """Copy *files* (workspace-relative) + findings into the best-state dir.

    Replaces any previous snapshot wholesale. Returns False (and logs) on any
    failure — the caller then simply has no ratchet for this round.
    """
    try:
        root = _snap_root(workspace_root, work_id)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        copied = 0
        for rel in files:
            src = Path(workspace_root) / rel
            if not src.is_file():
                continue
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        (root / _FINDINGS_FILE).write_text(
            json.dumps({"total": total, "findings": findings}), encoding="utf-8"
        )
        logger.info(
            "[%s] verify ratchet: snapshotted best state (total=%d, %d file(s))",
            work_id, total, copied,
        )
        return copied > 0
    except Exception as exc:  # noqa: BLE001 — ratchet is best-effort
        logger.warning("[%s] verify ratchet: snapshot failed: %s", work_id, exc)
        return False


def restore_best(workspace_root: str, work_id: str) -> bool:
    """Copy every snapshotted file back into the workspace.

    Returns True when at least one file was restored.
    """
    try:
        root = _snap_root(workspace_root, work_id)
        if not root.is_dir():
            return False
        restored = 0
        for src in root.rglob("*"):
            if not src.is_file() or src.name == _FINDINGS_FILE:
                continue
            rel = src.relative_to(root)
            dst = Path(workspace_root) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1
        if restored:
            logger.warning(
                "[%s] verify ratchet: RESTORED best state (%d file(s)) after a "
                "regressed cycle",
                work_id, restored,
            )
        return restored > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] verify ratchet: restore failed: %s", work_id, exc)
        return False


def load_best_findings(workspace_root: str, work_id: str) -> list[dict] | None:
    """The verification findings recorded with the best snapshot, or None."""
    try:
        raw = (_snap_root(workspace_root, work_id) / _FINDINGS_FILE).read_text(
            encoding="utf-8"
        )
        data = json.loads(raw)
        findings = data.get("findings")
        return findings if isinstance(findings, list) else None
    except Exception:  # noqa: BLE001
        return None
