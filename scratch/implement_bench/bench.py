#!/usr/bin/env python3
"""Implement-phase model bench — repeatable suitability test harness.

Freezes the spec+plan produced for work item ``b68c01b3`` (a reviewed_task that
stopped at critic_plan, so its workspace is un-mutated) as a golden baseline,
then runs the REAL implement -> verify pipeline against it once per candidate
model, in full isolation, and scores each run from its LangSmith trace.

What is held CONSTANT across candidates (so the only variable is the model):
  * the frozen spec + plan + execution_waves (copied, never mutated)
  * the workspace, pinned to one git commit (disposable clone per run)
  * the VERIFY judge model (a fixed strong model grades every candidate's work)
  * all other config (mcp, embeddings, token_compaction, ...)

What VARIES: the implement-phase provider — specifically these phase keys:
  implement, implement/decomposer, implement/subagents/slice-implementer.

Subcommands:
  freeze                         snapshot current .spine -> baseline/ (run once)
  run    --label L --provider P  one isolated implement+verify run for model P
  score  --label L               audit that run's LangSmith trace -> metrics.json
  compare                        table across all scored runs

Isolation model: each run is a ``git clone --local`` of the repo at the pinned
commit with a private copy of the frozen ``.spine``. The implement phase makes
its own sandbox worktree off the clone's main and (on a verified patch)
fast-forward-merges into the CLONE's main — which we then throw away. The real
repo is never touched.

Usage:
  python scratch/implement_bench/bench.py freeze
  python scratch/implement_bench/bench.py run --label nex --provider local-nex
  python scratch/implement_bench/bench.py score --label nex
  python scratch/implement_bench/bench.py compare
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ── Fixed bench parameters ────────────────────────────────────────────────────
# Paths are env-overridable so a branch/worktree can drive the bench against its
# OWN code + run dir while still reading the (single, expensive) frozen baseline
# from wherever it lives. Defaults reproduce the original main-tree layout.
#   SPINE_BENCH_REPO     — clone source for the disposable run repo
#   SPINE_BENCH_DIR      — bench root (holds _inner_*.py + runs/); set to a
#                          worktree's scratch/implement_bench to use its drivers
#   SPINE_BENCH_BASELINE — frozen baseline dir (default <bench>/baseline)
REPO = Path(os.environ.get("SPINE_BENCH_REPO", "/home/pat/projects/spine"))
BENCH = Path(os.environ.get("SPINE_BENCH_DIR", str(REPO / "scratch" / "implement_bench")))
BASELINE = Path(os.environ.get("SPINE_BENCH_BASELINE", str(BENCH / "baseline")))
RUNS = BENCH / "runs"
WORK_ID = "b68c01b3"  # the frozen reviewed_task (spec+plan, awaiting_approval)

# The VERIFY judge — held constant so every candidate's patch is graded the same
# way. local-gemma is the strongest local model in this config (it runs the
# critic phases). Override with `run --verify-provider`.
DEFAULT_VERIFY_PROVIDER = "local-qwen"

# Phase keys, by ROLE, so runs can target each independently. Confirmed against
# the code: the actual slice EDITOR (the read_edit_lint caller) resolves its
# model under the bare `implement` key (factory.build_phase_agent uses
# phase.value); the per-slice directive PLANNER resolves under
# `implement/subagents/slice-implementer`; the structural decomposer under
# `implement/decomposer/<mode>` (prefix-resolves to `implement/decomposer`).
ROLE_KEYS = {
    "implementer": "implement",                              # the editor
    "decomposer": "implement/decomposer",
    "planner": "implement/subagents/slice-implementer",      # directive planner
}
IMPLEMENT_KEYS = tuple(ROLE_KEYS.values())
VERIFY_KEYS = ("verify", "verify/subagents/slice-verifier")

# .spine entries the implement+verify run needs (spine.db carries BOTH the
# b68c01b3 checkpoint state AND the codebase symbol index that codebase_query
# reads). .spine is small here (~61MB) so a plain copy per run is cheap.
SPINE_SNAPSHOT = [
    "spine.db",
    "work_entries.db",
    "queue.db",
    "audit.db",
    "config.yaml",
    "config.reference.yaml",
    f"artifacts/{WORK_ID}",
    f"checkpoints/{WORK_ID}",
    "state",
    "project",
    "skills",
    "onboarding",
]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, **kw)


# ── freeze ────────────────────────────────────────────────────────────────────
def cmd_freeze(args: argparse.Namespace) -> None:
    pin = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
               capture_output=True).stdout.strip()
    src = REPO / ".spine"

    # Sanity: the work item must be a reviewed_task awaiting approval, or the
    # approve->implement bridge won't fire.
    import sqlite_utils
    row = sqlite_utils.Database(str(src / "work_entries.db"))["work_entries"].get(WORK_ID)
    assert row["work_type"] in ("reviewed_task", "critical_reviewed_task"), row["work_type"]
    assert row["status"] == "awaiting_approval", (
        f"{WORK_ID} status is {row['status']!r}, expected awaiting_approval — "
        "freeze must capture the pristine pre-implement state."
    )

    if BASELINE.exists():
        shutil.rmtree(BASELINE)
    dst = BASELINE / "spine"
    dst.mkdir(parents=True)
    for rel in SPINE_SNAPSHOT:
        s = src / rel
        if not s.exists():
            continue
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d)
        else:
            # reflink where the FS supports it; harmless fallback to full copy.
            _run(["cp", "--reflink=auto", str(s), str(d)])
    (BASELINE / "manifest.json").write_text(json.dumps({
        "work_id": WORK_ID,
        "pin": pin,
        "description": row["description"],
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "snapshot": SPINE_SNAPSHOT,
    }, indent=2))
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"[freeze] baseline @ {pin[:10]}  ({size/1e6:.1f} MB)  -> {BASELINE}")


def _manifest() -> dict:
    return json.loads((BASELINE / "manifest.json").read_text())


# ── run ───────────────────────────────────────────────────────────────────────
def _write_candidate_config(cfg_path: Path, provider: str, verify_provider: str,
                            roles: dict[str, str]) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    phases = cfg["providers"]["phases"]

    def set_provider(key: str, name: str) -> None:
        node = phases.setdefault(key, {})
        node["provider"] = name

    for key in IMPLEMENT_KEYS:
        set_provider(key, roles.get(key, provider))
    for key in VERIFY_KEYS:
        set_provider(key, verify_provider)
    cfg_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))


def cmd_run(args: argparse.Namespace) -> None:
    if not (BASELINE / "manifest.json").exists():
        sys.exit("[run] no baseline — run `bench.py freeze` first.")
    man = _manifest()
    pin = man["pin"]
    label = args.label
    rundir = RUNS / label
    if rundir.exists():
        if not args.force:
            sys.exit(f"[run] {rundir} exists; pass --force to overwrite.")
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    repo = rundir / "repo"

    print(f"[run:{label}] cloning repo @ {pin[:10]} ...")
    _run(["git", "clone", "--local", "--quiet", str(REPO), str(repo)], check=True)
    _run(["git", "-C", str(repo), "checkout", "--quiet", "main"], check=True)
    _run(["git", "-C", str(repo), "reset", "--hard", "--quiet", pin], check=True)

    # The repo tracks ~10 files under .spine/ (incl. config.yaml). We need to
    # populate .spine with the frozen runtime state + a CANDIDATE config, which
    # would dirty those tracked paths — and the implement sandbox preflight
    # (`git status --porcelain` must be empty) would refuse. So untrack .spine
    # in the disposable clone and git-ignore it; now we can freely overlay it.
    _run(["git", "-C", str(repo), "rm", "-r", "--cached", "--quiet", ".spine"])
    with (repo / ".git" / "info" / "exclude").open("a") as f:
        f.write("\n.spine/\n")
    _run(["git", "-C", str(repo), "commit", "-q", "--no-verify",
          "-m", "bench: untrack .spine for isolated run"], check=True)
    # Diff implement work against THIS tree state (source is identical to the
    # baseline commit; only .spine tracking changed).
    pin = _run(["git", "-C", str(repo), "rev-parse", "HEAD"],
               capture_output=True).stdout.strip()

    # private .spine from the frozen baseline (now untracked+ignored)
    spine_dst = repo / ".spine"
    if spine_dst.exists():
        shutil.rmtree(spine_dst)
    shutil.copytree(BASELINE / "spine", spine_dst)

    # Build the per-key role map. Role flags (--implementer/--decomposer/
    # --planner) are the clear path; --provider is a default for any role left
    # unset; raw --role 'key=model' still works for anything exotic.
    roles: dict[str, str] = {}
    for role, key in ROLE_KEYS.items():
        val = getattr(args, role, None)
        if val:
            roles[key] = val
    for kv in (args.role or []):
        k, v = kv.split("=", 1)
        roles[k] = v
    default = args.provider or args.implementer or args.decomposer
    if not default:
        sys.exit("[run] specify at least one of --provider/--implementer/--decomposer.")
    # The per-slice directive PLANNER should run on the capable model (the
    # decomposer's), not inherit the lightweight implementer default.
    planner_key = ROLE_KEYS["planner"]
    if planner_key not in roles and (args.decomposer or args.provider):
        roles[planner_key] = args.decomposer or args.provider
    _write_candidate_config(spine_dst / "config.yaml", default,
                            args.verify_provider, roles)

    # carry LangSmith creds; isolate this run's trace in its own project
    if (REPO / ".env").exists():
        shutil.copy(REPO / ".env", repo / ".env")
    project = f"spine-bench-{label}-{time.strftime('%m%d-%H%M%S')}"

    env = dict(os.environ)
    for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"'))
    env["LANGSMITH_PROJECT"] = project
    env["LANGSMITH_TRACING"] = "true"
    # Slow local models (esp. Qwen3.6-27B-MTP) buffer and exceed langchain's
    # 120s per-chunk streaming watchdog, which raises StreamChunkTimeoutError →
    # the slice gets marked `blocked` → fallback re-decompose → a runaway
    # re-dispatch cascade (run r1: 150 timeouts, 133 slice invocations, 1M
    # tokens). Raise both the per-chunk and workflow stall timeouts so a slow
    # generation completes instead of cascading.
    env.setdefault("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "300")
    env.setdefault("SPINE_STALL_TIMEOUT", "600")
    # resolve_chat_model's string path (used by the decomposer) builds bare
    # `openai:<model>` via init_chat_model WITHOUT passing base_url/api_key, so
    # any phase that falls to that path would hit real OpenAI and fail with
    # missing credentials. Point the OpenAI default at the local server so the
    # default path stays on-box. Derived from the candidate config's providers.
    cfg = yaml.safe_load((spine_dst / "config.yaml").read_text())
    local = next((p for p in cfg.get("providers", {}).get("llm", [])
                  if p.get("base_url")), {})
    if local.get("base_url"):
        env.setdefault("OPENAI_BASE_URL", local["base_url"])
        env.setdefault("OPENAI_API_KEY", local.get("api_key", "vllm"))

    result_path = rundir / "result.json"
    log_path = rundir / "run.log"
    resolved = {role: roles.get(key, default) for role, key in ROLE_KEYS.items()}
    meta = {"label": label, "default_provider": default,
            "verify_provider": args.verify_provider, "roles": roles,
            "resolved_roles": resolved, "pin": pin, "langsmith_project": project,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (rundir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[run:{label}] decomposer={resolved['decomposer']}  "
          f"implementer={resolved['implementer']}  planner={resolved['planner']}  "
          f"verify={args.verify_provider}")
    print(f"[run:{label}] project={project}")
    print(f"[run:{label}] executing approve->implement->verify (this is slow)...")
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = _run(
            [sys.executable, str(BENCH / "_inner_run.py"), WORK_ID,
             "--pin", pin, "--out", str(result_path)],
            cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0

    if result_path.exists():
        res = json.loads(result_path.read_text())
        print(f"[run:{label}] done in {dt:.0f}s — status={res.get('status')} "
              f"diff={res.get('git', {}).get('diffstat', '').splitlines()[-1:] or '∅'}")
        if res.get("error"):
            print(f"[run:{label}] ERROR: {res['error']}  (see {log_path})")
    else:
        print(f"[run:{label}] FAILED — no result.json (exit {proc.returncode}); "
              f"see {log_path}")


# ── replan ───────────────────────────────────────────────────────────────────

PLAN_PROVIDER_KEYS = (
    "plan",
    "plan/subagents/researcher",
    "plan/subagents/researcher/supervisor",
)


def _write_replan_config(cfg_path: Path, provider: str) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    phases = cfg["providers"]["phases"]
    for key in PLAN_PROVIDER_KEYS:
        phases.setdefault(key, {})["provider"] = provider
    cfg_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))


def cmd_replan(args: argparse.Namespace) -> None:
    """Re-run the PLAN phase with a stronger model and optionally update the baseline.

    Clones the repo at the frozen pin, patches the plan-phase provider to
    *args.provider*, invokes the plan subgraph against the frozen spec, then
    copies the resulting plan.json / plan.md into RUNS/replan-{provider}/.

    Pass --update-baseline to promote the new plan into the frozen baseline so
    future `bench run` invocations use it.
    """
    if not (BASELINE / "manifest.json").exists():
        sys.exit("[replan] no baseline — run `bench.py freeze` first.")
    man = _manifest()
    pin = man["pin"]
    description = man["description"]
    label = f"replan-{args.provider}"
    rundir = RUNS / label
    if rundir.exists():
        if not args.force:
            sys.exit(f"[replan] {rundir} exists; pass --force to overwrite.")
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    repo = rundir / "repo"

    print(f"[replan] cloning repo @ {pin[:10]} ...")
    _run(["git", "clone", "--local", "--quiet", str(REPO), str(repo)], check=True)
    _run(["git", "-C", str(repo), "checkout", "--quiet", "main"], check=True)
    _run(["git", "-C", str(repo), "reset", "--hard", "--quiet", pin], check=True)

    spine_dst = repo / ".spine"
    if spine_dst.exists():
        shutil.rmtree(spine_dst)
    shutil.copytree(BASELINE / "spine", spine_dst)

    # Overwrite plan-phase provider keys; leave everything else as frozen.
    _write_replan_config(spine_dst / "config.yaml", args.provider)

    if (REPO / ".env").exists():
        shutil.copy(REPO / ".env", repo / ".env")

    env = dict(os.environ)
    for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip().strip('"'))
    project = f"spine-bench-replan-{args.provider}-{time.strftime('%m%d-%H%M%S')}"
    env["LANGSMITH_PROJECT"] = project
    env["LANGSMITH_TRACING"] = "true"
    cfg = yaml.safe_load((spine_dst / "config.yaml").read_text())
    local = next((p for p in cfg.get("providers", {}).get("llm", [])
                  if p.get("base_url")), {})
    if local.get("base_url"):
        env.setdefault("OPENAI_BASE_URL", local["base_url"])
        env.setdefault("OPENAI_API_KEY", local.get("api_key", "vllm"))

    result_path = rundir / "result.json"
    log_path = rundir / "run.log"
    print(f"[replan] provider={args.provider}  work_id={WORK_ID}")
    print(f"[replan] running plan subgraph (this may be slow)...")
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = _run(
            [sys.executable, str(BENCH / "_inner_replan.py"), WORK_ID,
             "--description", description,
             "--out", str(result_path)],
            cwd=str(repo), env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0

    if not result_path.exists():
        print(f"[replan] FAILED — no result.json (exit {proc.returncode}); see {log_path}")
        return

    res = json.loads(result_path.read_text())
    if res.get("error"):
        print(f"[replan] ERROR: {res['error']}")
        print(f"[replan] see {log_path}")
        return

    plan_src = repo / ".spine" / "artifacts" / WORK_ID / "plan"
    plan_dst = rundir / "plan"
    if plan_src.exists():
        shutil.copytree(plan_src, plan_dst)
        files = sorted(p.name for p in plan_dst.iterdir())
        print(f"[replan] done in {dt:.0f}s — plan files: {files}")
    else:
        print(f"[replan] WARNING: no plan artifacts produced — see {log_path}")
        return

    if args.update_baseline:
        baseline_plan = BASELINE / "spine" / "artifacts" / WORK_ID / "plan"
        if baseline_plan.exists():
            shutil.rmtree(baseline_plan)
        shutil.copytree(plan_dst, baseline_plan)
        print(f"[replan] baseline plan updated → {baseline_plan}")

        # Also patch the LangGraph checkpoint in the baseline's spine.db so
        # approve_and_spawn reads the new plan_json + execution_waves.
        # Without this the checkpoint holds the old plan and the new artifact
        # is ignored at implement time (dispatcher line 2710 reads checkpoint first).
        baseline_db = BASELINE / "spine" / "spine.db"
        plan_json_path = baseline_plan / "plan.json"
        if baseline_db.exists() and plan_json_path.exists():
            import asyncio as _asyncio
            import sys as _sys
            patch_script = str(BENCH / "_patch_checkpoint.py")
            proc2 = _run(
                [_sys.executable, patch_script, str(baseline_db), WORK_ID,
                 str(plan_json_path)],
                capture_output=True,
            )
            if proc2.returncode == 0:
                print(f"[replan] checkpoint patched — execution_waves updated")
            else:
                print(f"[replan] WARNING: checkpoint patch failed (bench will use old waves):")
                print(f"         {proc2.stderr.strip()}")

        print(f"[replan] next: run `bench.py run --provider <model>` to test against the new plan")
    else:
        print(f"[replan] plan saved to {plan_dst}")
        print(f"[replan] inspect, then pass --update-baseline to promote it")


# ── score (LangSmith trace audit) ─────────────────────────────────────────────
def _ls_client():
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    from langsmith import Client
    return Client(api_url=env["LANGSMITH_ENDPOINT"], api_key=env["LANGSMITH_API_KEY"]), env


def _audit(project: str) -> dict:
    from collections import Counter
    client, _ = _ls_client()
    runs = list(client.list_runs(project_name=project))
    llm = [r for r in runs if r.run_type == "llm"]
    tool = [r for r in runs if r.run_type == "tool"]
    tp = tc = cr = 0
    per_model = Counter()
    for r in llm:
        u = (r.extra or {}).get("metadata", {}).get("usage_metadata") or {}
        tp += u.get("input_tokens", 0) or 0
        tc += u.get("output_tokens", 0) or 0
        cr += (u.get("input_token_details", {}) or {}).get("cache_read", 0) or 0
        m = (r.extra or {}).get("metadata", {}).get("ls_model_name", "?")
        per_model[m] += 1
    # error taxonomy
    err = Counter()
    for r in runs:
        if r.error:
            e = str(r.error)
            kind = ("budget" if "TokenBudgetExceeded" in e or "Token budget" in e
                    else "length" if "Length" in e or "length limit" in e
                    else "tool_schema" if "ToolException" in e or "must " in e
                    else "model_load" if "model_load" in e or "Failed to load" in e
                    else "other")
            err[kind] += 1
    denom = tp + cr
    return {
        "project": project,
        "spans": len(runs), "llm_calls": len(llm), "tool_calls": len(tool),
        "input_tokens": tp, "output_tokens": tc, "cache_read": cr,
        "pc_ratio": round(tp / tc, 2) if tc else None,
        "cache_pct": round(cr / denom * 100, 1) if denom else None,
        "length_errors": err.get("length", 0),
        "tool_schema_errors": err.get("tool_schema", 0),
        "model_load_errors": err.get("model_load", 0),
        "budget_exceeded": err.get("budget", 0),
        "other_errors": err.get("other", 0),
        "models": dict(per_model),
    }


def cmd_score(args: argparse.Namespace) -> None:
    rundir = RUNS / args.label
    res = json.loads((rundir / "result.json").read_text())
    project = res.get("langsmith_project") or json.loads(
        (rundir / "meta.json").read_text())["langsmith_project"]
    m = _audit(project)
    (rundir / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"[score:{args.label}] {project}")
    print(f"  llm={m['llm_calls']} tool={m['tool_calls']}  "
          f"in={m['input_tokens']:,} out={m['output_tokens']:,} "
          f"P:C={m['pc_ratio']} cache={m['cache_pct']}%")
    print(f"  errors: length={m['length_errors']} tool_schema={m['tool_schema_errors']} "
          f"model_load={m['model_load_errors']} other={m['other_errors']}")


# ── compare ───────────────────────────────────────────────────────────────────
def cmd_compare(args: argparse.Namespace) -> None:
    rows = []
    for rundir in sorted(RUNS.glob("*")):
        if not (rundir / "result.json").exists():
            continue
        res = json.loads((rundir / "result.json").read_text())
        met = json.loads((rundir / "metrics.json").read_text()) if (
            rundir / "metrics.json").exists() else {}
        diffstat = (res.get("git", {}).get("diffstat", "") or "").splitlines()
        rows.append({
            "label": rundir.name,
            "status": res.get("status"),
            "verify_artifacts": len(res.get("artifacts", {}).get("verify", [])),
            "elapsed_s": res.get("elapsed_s"),
            "diff": diffstat[-1].strip() if diffstat else "∅",
            "budget": met.get("budget_exceeded"),
            "len_err": met.get("length_errors"),
            "tool_err": met.get("tool_schema_errors"),
            "in_tok": met.get("input_tokens"),
            "out_tok": met.get("output_tokens"),
            "pc": met.get("pc_ratio"),
            "cache": met.get("cache_pct"),
        })
    if not rows:
        print("no scored runs yet.")
        return
    hdr = ["label", "status", "verify_artifacts", "elapsed_s", "budget",
           "len_err", "tool_err", "in_tok", "out_tok", "pc", "cache", "diff"]
    w = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in hdr}
    print("  ".join(h.ljust(w[h]) for h in hdr))
    print("  ".join("-" * w[h] for h in hdr))
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(w[h]) for h in hdr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)

    r = sub.add_parser("run")
    r.add_argument("--label", required=True, help="short id for this run, e.g. 'nex'")
    r.add_argument("--provider", default=None,
                   help="default providers.llm name for any role left unset")
    r.add_argument("--implementer", default=None,
                   help="model for the slice EDITOR (read_edit_lint caller) — `implement` key")
    r.add_argument("--decomposer", default=None,
                   help="model for the structural decomposer — `implement/decomposer`")
    r.add_argument("--planner", default=None,
                   help="model for the per-slice directive planner — "
                        "`implement/subagents/slice-implementer`")
    r.add_argument("--verify-provider", default=DEFAULT_VERIFY_PROVIDER,
                   help=f"fixed judge for verify (default {DEFAULT_VERIFY_PROVIDER})")
    r.add_argument("--role", action="append",
                   help="raw override 'phase/key=model' for anything exotic")
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score")
    s.add_argument("--label", required=True)
    s.set_defaults(func=cmd_score)

    sub.add_parser("compare").set_defaults(func=cmd_compare)

    rp = sub.add_parser("replan",
                        help="re-run the PLAN phase with a stronger model")
    rp.add_argument("--provider", required=True,
                    help="providers.llm name to use for all plan-phase keys")
    rp.add_argument("--update-baseline", action="store_true",
                    help="promote the new plan into the frozen baseline after success")
    rp.add_argument("--force", action="store_true",
                    help="overwrite an existing replan run directory")
    rp.set_defaults(func=cmd_replan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
