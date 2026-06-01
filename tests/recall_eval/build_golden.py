"""Derive recall-eval golden pairs from local SPECIFY/PLAN research logs.

Each ``ResearchFinding`` a real researcher emitted carries a ``topic``
(a natural-language query) and a ``file_map`` (the production files that
researcher determined were relevant to the topic). That is exactly the
``(query -> relevant files)`` signal a recall eval needs, produced by the
system in anger rather than hand-guessed.

This tool reads ``.spine/artifacts/*/*/research_log.json`` (gitignored,
local-only), de-duplicates by cleaned topic, keeps concrete production
files that still exist on disk, and writes the result to
``golden_mined.jsonl`` for review.

The committed ``golden.jsonl`` is the source of truth for the eval —
this script is a dev tool for *extending* it, not a runtime dependency.
Workflow:

    python tests/recall_eval/build_golden.py        # writes golden_mined.jsonl
    # eyeball golden_mined.jsonl, fold good rows into golden.jsonl by hand

We intentionally do NOT overwrite golden.jsonl: curated entries (sharp,
single-target queries with known-good answers) live only there and must
not be clobbered by a regen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Run from repo root.
_REPO = Path(__file__).resolve().parents[2]
_ARTIFACTS = _REPO / ".spine" / "artifacts"
_OUT = Path(__file__).resolve().parent / "golden_mined.jsonl"

_INDEXABLE = {".py", ".php", ".ts", ".tsx"}
# Cap expected files per entry — beyond this the topic is too diffuse to
# be a crisp recall target (hit@k stays meaningful with a focused set).
_MAX_FILES = 6


def _clean_topic(topic: str) -> str:
    """Strip the ``ResearchFinding`` enrichment suffix from a topic string.

    ``run_explore_node`` stamps topics with ``" — recall symbols: …"``;
    the bare topic is the actual natural-language query.
    """
    return re.split(r"\s+—\s+recall symbols:", topic)[0].strip()


def _concrete_prod_files(file_map: dict) -> list[str]:
    """Keep workspace-relative production files that exist and are indexable."""
    out: list[str] = []
    for path in file_map or {}:
        if not isinstance(path, str) or "*" in path or "?" in path:
            continue
        if path.startswith("tests/") or "/tests/" in path:
            continue
        p = _REPO / path
        if p.suffix.lower() in _INDEXABLE and p.exists():
            out.append(path)
    return sorted(set(out))


def main() -> None:
    if not _ARTIFACTS.exists():
        raise SystemExit(f"No artifacts dir at {_ARTIFACTS} — nothing to mine.")

    by_topic: dict[str, set[str]] = {}
    n_logs = 0
    for log in _ARTIFACTS.glob("*/*/research_log.json"):
        n_logs += 1
        try:
            data = json.loads(log.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for finding in data.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            topic = _clean_topic(finding.get("topic", "") or "")
            if len(topic) < 12:
                continue
            files = _concrete_prod_files(finding.get("file_map", {}))
            if files:
                by_topic.setdefault(topic, set()).update(files)

    rows = []
    for topic, files in sorted(by_topic.items()):
        rows.append(
            {
                "query": topic,
                "expected_files": sorted(files)[:_MAX_FILES],
                "expected_symbols": [],
                "source": "mined",
                "note": "mined from research_log file_map",
            }
        )

    with _OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"Scanned {n_logs} research logs.")
    print(f"Wrote {len(rows)} mined golden rows to {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
