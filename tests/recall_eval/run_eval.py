"""Recall-quality eval harness for the SPINE vector search.

Runs every golden ``(query -> expected files/symbols)`` pair through the
real :class:`RecallTool` and reports retrieval quality:

* ``hit@k`` (k=5,10,20) — fraction of queries with at least one expected
  item in the top-k. This is the metric that tracks "does the researcher
  get a good anchor."
* ``MRR`` — mean reciprocal rank of the first expected hit within the
  retrieval window (0 if it never appears). Sensitive to ordering, so it
  moves when reranking/embedding changes help *position*, not just
  presence.
* ``miss@N`` — fraction of queries whose expected items never appear in
  the top-N window at all. This is the **recall ceiling**: no reranker
  can recover a result the retriever never surfaced. Watch this number —
  it bounds everything downstream.
* ``set_recall@20`` — average fraction of a query's expected files that
  land in the top-20. Secondary signal for multi-target queries.

This is a live script, not a CI test: it needs a populated vector store
(``spine index``) and a reachable embedding endpoint. Run it before and
after each change and diff the JSON it emits.

Usage::

    python tests/recall_eval/run_eval.py                       # human table
    python tests/recall_eval/run_eval.py --label baseline \\
        --out tests/recall_eval/baseline.json                  # + save JSON
    python tests/recall_eval/run_eval.py --retrieve-k 50 --concurrency 4

Compare two runs::

    python tests/recall_eval/run_eval.py --compare tests/recall_eval/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_GOLDEN = _HERE / "golden.jsonl"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_golden(path: Path) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{i}: invalid JSON — {exc}")
        if not row.get("query"):
            raise SystemExit(f"{path}:{i}: missing 'query'")
        if not row.get("expected_files") and not row.get("expected_symbols"):
            raise SystemExit(f"{path}:{i}: need expected_files or expected_symbols")
        rows.append(row)
    return rows


def _symbol_matches(retrieved: str, expected: str) -> bool:
    """True if a retrieved symbol name matches an expected one.

    The indexer writes qualified names for methods (``Class.method``);
    golden entries may give either the bare or qualified form, so match
    on exact, suffix, or last-segment equality.
    """
    if not retrieved or not expected:
        return False
    if retrieved == expected:
        return True
    if retrieved.endswith("." + expected):
        return True
    return retrieved.split(".")[-1] == expected.split(".")[-1] and "." in expected


def _first_hit_rank(results: list[dict], entry: dict) -> int | None:
    """1-based rank of the first retrieved result matching the entry, else None."""
    expected_files = set(entry.get("expected_files") or [])
    expected_symbols = entry.get("expected_symbols") or []
    for rank, r in enumerate(results, 1):
        if r.get("file_path") in expected_files:
            return rank
        sym = r.get("symbol_name", "")
        if any(_symbol_matches(sym, e) for e in expected_symbols):
            return rank
    return None


def _set_recall_at(results: list[dict], entry: dict, k: int) -> float | None:
    """Fraction of expected_files appearing in top-k (None if no expected_files)."""
    expected_files = set(entry.get("expected_files") or [])
    if not expected_files:
        return None
    top_files = {r.get("file_path") for r in results[:k]}
    return len(expected_files & top_files) / len(expected_files)


async def _run_one(entry: dict, retrieve_k: int, db_path: str, sem: asyncio.Semaphore) -> dict:
    from spine.agents.tools.recall_tool import RecallTool

    async with sem:
        recall = RecallTool(db_path=db_path)
        try:
            raw = await recall._arun(
                query=entry["query"],
                k=retrieve_k,
                max_tokens=0,  # 0/None => no token-budget truncation; rank all hits
                summaries_only=True,
            )
            results = json.loads(raw).get("results", []) or []
            return {"entry": entry, "results": results, "error": None}
        except Exception as exc:  # noqa: BLE001 — surface, don't abort the sweep
            return {"entry": entry, "results": [], "error": str(exc)}


def _aggregate(runs: list[dict], retrieve_k: int) -> dict[str, Any]:
    ks = (5, 10, 20)
    ok = [r for r in runs if r["error"] is None]
    errors = [r for r in runs if r["error"] is not None]

    ranks: list[int | None] = []
    set_recalls: list[float] = []
    per_query = []
    for r in ok:
        rank = _first_hit_rank(r["results"], r["entry"])
        ranks.append(rank)
        sr = _set_recall_at(r["results"], r["entry"], 20)
        if sr is not None:
            set_recalls.append(sr)
        per_query.append(
            {
                "query": r["entry"]["query"],
                "source": r["entry"].get("source", "?"),
                "first_hit_rank": rank,
                "n_results": len(r["results"]),
            }
        )

    n = len(ok)

    def hit_at(k: int) -> float:
        if not n:
            return 0.0
        return sum(1 for rk in ranks if rk is not None and rk <= k) / n

    mrr = (sum(1.0 / rk for rk in ranks if rk is not None) / n) if n else 0.0
    miss = (sum(1 for rk in ranks if rk is None) / n) if n else 0.0
    set_recall_20 = (sum(set_recalls) / len(set_recalls)) if set_recalls else 0.0

    return {
        "n_queries": len(runs),
        "n_scored": n,
        "n_errors": len(errors),
        "retrieve_k": retrieve_k,
        "metrics": {
            "hit@5": round(hit_at(5), 4),
            "hit@10": round(hit_at(10), 4),
            "hit@20": round(hit_at(20), 4),
            "mrr": round(mrr, 4),
            f"miss@{retrieve_k}": round(miss, 4),
            "set_recall@20": round(set_recall_20, 4),
        },
        "errors": [{"query": r["entry"]["query"], "error": r["error"]} for r in errors],
        "per_query": per_query,
    }


def _by_source(runs: list[dict], retrieve_k: int) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["entry"].get("source", "?"), []).append(r)
    return {src: _aggregate(rs, retrieve_k)["metrics"] for src, rs in sorted(groups.items())}


def _print_report(report: dict) -> None:
    m = report["metrics"]
    print()
    print("=" * 64)
    print(f"  RECALL EVAL  —  {report['n_scored']}/{report['n_queries']} scored, "
          f"{report['n_errors']} error(s),  retrieve_k={report['retrieve_k']}")
    print("=" * 64)
    for key in ("hit@5", "hit@10", "hit@20", "mrr", "set_recall@20"):
        print(f"  {key:<16} {m[key]:.4f}")
    miss_key = next(k for k in m if k.startswith("miss@"))
    print(f"  {miss_key:<16} {m[miss_key]:.4f}   (recall ceiling gap — lower is better)")
    if report.get("by_source"):
        print("-" * 64)
        print("  by source:")
        for src, sm in report["by_source"].items():
            print(f"    {src:<10} hit@10={sm['hit@10']:.3f}  mrr={sm['mrr']:.3f}  "
                  f"{miss_key}={sm[miss_key]:.3f}")
    if report["errors"]:
        print("-" * 64)
        print("  ERRORS:")
        for e in report["errors"][:10]:
            print(f"    - {e['query'][:60]!r}: {e['error'][:80]}")
    # Worst offenders: scored queries that missed entirely.
    misses = [q for q in report["per_query"] if q["first_hit_rank"] is None]
    if misses:
        print("-" * 64)
        print(f"  MISSED ENTIRELY ({len(misses)}):")
        for q in misses[:15]:
            print(f"    - [{q['source']}] {q['query'][:64]}")
    print("=" * 64)
    print()


def _print_compare(current: dict, baseline_path: Path) -> None:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    bm, cm = base["metrics"], current["metrics"]
    print()
    print(f"  COMPARE vs {baseline_path.name}"
          f"  (baseline label: {base.get('label', '?')})")
    print("-" * 56)
    keys = ["hit@5", "hit@10", "hit@20", "mrr", "set_recall@20"]
    keys += [k for k in cm if k.startswith("miss@")]
    for key in keys:
        b = bm.get(key)
        c = cm.get(key)
        if b is None or c is None:
            continue
        delta = c - b
        arrow = "→" if abs(delta) < 1e-9 else ("▲" if delta > 0 else "▼")
        print(f"  {key:<16} {b:.4f}  {arrow}  {c:.4f}   ({delta:+.4f})")
    print("-" * 56)
    print()


async def _main_async(args: argparse.Namespace) -> dict:
    golden = _load_golden(Path(args.golden))
    if args.limit:
        golden = golden[: args.limit]
    sem = asyncio.Semaphore(args.concurrency)
    runs = await asyncio.gather(
        *[_run_one(e, args.retrieve_k, args.db, sem) for e in golden]
    )
    report = _aggregate(list(runs), args.retrieve_k)
    report["by_source"] = _by_source(list(runs), args.retrieve_k)
    report["label"] = args.label
    report["generated_at"] = args.now or datetime.now(timezone.utc).isoformat()
    report["db"] = args.db
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(_GOLDEN), help="Path to golden.jsonl")
    parser.add_argument("--db", default=None, help="Vector store path (default: config checkpoint_path)")
    parser.add_argument("--retrieve-k", type=int, default=50, help="Retrieval window (recall ceiling)")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent recall calls")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N golden rows (debug)")
    parser.add_argument("--label", default="", help="Label stored in the saved JSON")
    parser.add_argument("--out", default=None, help="Write the full report JSON here")
    parser.add_argument("--compare", default=None, help="Print a diff against a saved report JSON")
    parser.add_argument("--now", default=None, help="Override timestamp (for reproducible output)")
    args = parser.parse_args(argv)

    if args.db is None:
        from spine.config import SpineConfig

        args.db = SpineConfig.load().checkpoint_path

    report = asyncio.run(_main_async(args))
    _print_report(report)

    if args.compare:
        _print_compare(report, Path(args.compare))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  saved report → {args.out}")

    # Non-zero exit if every query errored (store missing / endpoint down)
    # so a CI wrapper or shell `&&` chain notices.
    return 0 if report["n_scored"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
