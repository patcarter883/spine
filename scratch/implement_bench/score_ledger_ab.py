#!/usr/bin/env python3
"""Groundedness scorer for the findings-ledger A/B (item 2).

Given a produced ``plan.json`` it measures how well the plan's
``target_files`` / ``reference_symbols`` are GROUNDED in reality:

  * a target file is *real* if it exists in the repo at the frozen pin;
  * a target file is *in-ledger* if it appears in the deterministic
    file->role map the exploration researchers produced (the durable map the
    ledger injects). Files that are real-but-not-in-ledger are ones the
    synthesizer reached on its own (vector recall / spec text);
  * a reference symbol is *real* if its leaf name is defined (``def``/``class``)
    somewhere in the repo at the pin.

The ledger's job is to keep the decomposed synthesizer grounded even though it
never sees the raw <findings>. So the headline signal is hallucination rate:
fabricated files and fabricated symbols. Compare ON vs OFF.

Usage:
  score_ledger_ab.py <plan.json> --repo <repo> --pin <sha> \
      --research-log <research_log.json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _ledger_files(research_log: Path) -> dict[str, str]:
    if not research_log.exists():
        return {}
    data = json.loads(research_log.read_text())
    out: dict[str, str] = {}
    for f in data.get("findings") or []:
        fm = f.get("file_map")
        if isinstance(fm, dict):
            for p, r in fm.items():
                if isinstance(p, str) and p.strip():
                    out[p] = str(r)
    return out


def _file_exists(repo: Path, pin: str, path: str) -> bool:
    r = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{pin}:{path}"],
                       capture_output=True)
    return r.returncode == 0


def _symbol_defined(repo: Path, pin: str, symbol: str) -> bool:
    """True if the symbol's leaf name is defined as a def/class at the pin."""
    leaf = symbol.replace("()", "").split(".")[-1].split("(")[0].strip()
    if not leaf:
        return False
    # word-boundary match on `def leaf` / `class leaf` across tracked py files.
    pat = rf"^[[:space:]]*(async def|def|class)[[:space:]]+{leaf}\b"
    r = subprocess.run(
        ["git", "-C", str(repo), "grep", "-lE", pat, pin, "--", "*.py"],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def score(plan_path: Path, repo: Path, pin: str, research_log: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    ledger = _ledger_files(research_log)
    ledger_set = set(ledger)

    slices = plan.get("feature_slices") or plan.get("slices") or []
    files: list[str] = []
    symbols: list[str] = []
    for s in slices:
        files.extend(s.get("target_files") or [])
        symbols.extend(s.get("reference_symbols") or [])

    uniq_files = sorted(set(files))
    uniq_syms = sorted(set(symbols))

    real_files = [f for f in uniq_files if _file_exists(repo, pin, f)]
    halluc_files = [f for f in uniq_files if f not in real_files]
    in_ledger = [f for f in uniq_files if f in ledger_set]
    real_off_ledger = [f for f in real_files if f not in ledger_set]

    real_syms = [s for s in uniq_syms if _symbol_defined(repo, pin, s)]
    halluc_syms = [s for s in uniq_syms if s not in real_syms]

    return {
        "slices": len(slices),
        "files_total": len(uniq_files),
        "files_real": len(real_files),
        "files_hallucinated": len(halluc_files),
        "files_in_ledger": len(in_ledger),
        "files_real_off_ledger": len(real_off_ledger),
        "ledger_coverage_pct": round(100 * len(in_ledger) / len(uniq_files), 1)
        if uniq_files else None,
        "file_hallucination_pct": round(100 * len(halluc_files) / len(uniq_files), 1)
        if uniq_files else None,
        "symbols_total": len(uniq_syms),
        "symbols_real": len(real_syms),
        "symbols_hallucinated": len(halluc_syms),
        "symbol_hallucination_pct": round(100 * len(halluc_syms) / len(uniq_syms), 1)
        if uniq_syms else None,
        "_hallucinated_files": halluc_files,
        "_hallucinated_symbols": halluc_syms,
        "_real_off_ledger_files": real_off_ledger,
        "_ledger_size": len(ledger_set),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--research-log", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    m = score(Path(args.plan), Path(args.repo), args.pin, Path(args.research_log))
    if args.label:
        m = {"label": args.label, **m}
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
