# Findings-ledger A/B (A2) — control result

**Question (open item 2):** quantify how much the durable codebase ledger
(`build_findings_ledger`, the `<codebase_ledger>` block injected into the PLAN
synthesizers) improves plan groundedness.

**Method.** Held everything constant and toggled only the ledger:
- Frozen baseline `b68c01b3` @ pin `5f5a5d3`; provider `local-qwen`
  (Qwen3.6-27B-MTP). The exploration manager short-circuits to `done` on round 0,
  so the synthesizer's input findings are **deterministic** (4 findings seeded
  from the baseline's `plan/research_log.json`, a 5-file `file_map`). Only the
  decomposed synthesizer's LLM sampling varies between runs.
- **OFF** arm: `SPINE_DISABLE_FINDINGS_LEDGER=1` collapses the ledger to `""`
  at both injection sites (monolithic `_synthesize_plan` + decomposed
  `plan_synthesis`). **ON** arm: gate unset. Identical harness otherwise
  (worktree code via `PYTHONPATH`, same baseline, same provider).
- Scored with `score_ledger_ab.py`: target files real (exist @ pin) / in-ledger /
  hallucinated; reference symbols real (def/class @ pin) vs not-yet-existing.

**Result — no measurable difference:**

| arm                | files | real | in-ledger | file-halluc% | syms | real | new |
|--------------------|-------|------|-----------|--------------|------|------|-----|
| ON (orig hand-run) | 2     | 2    | 2         | 0.0          | 7    | 3    | 4   |
| ON (clean rerun)   | 2     | 2    | 2         | 0.0          | 9    | 5    | 4   |
| OFF (gate fired)   | 2     | 2    | 2         | 0.0          | 9    | 5    | 4   |

All three arms ground 2/2 target files in the ledger with zero hallucinated
files. The "non-real" symbols are the **new** methods the task adds
(`UIApi.set_embedding_provider`, …) anchored by analogy on the real
`set_phase_provider`/`_save_config` — not hallucinations. Which new symbols get
named varies by sampling, independent of ledger state.

**Interpretation — the ledger is insurance that this scenario never cashes in.**
The ledger only changes the synthesizer's view when:
1. (monolithic) findings exceed the synthesis budget and `_compress_findings`
   trims them — here findings are ~4 KB, far under budget, so nothing is
   trimmed; or
2. (decomposed) vector recall (`retrieved_context`, all the manager/workers see
   without the ledger) **misses** a file that only the deterministic `file_map`
   carries — here the 2-file surface (`spine/ui_api/api.py`,
   `spine/ui/_pages/config_view.py`) is small and squarely in the spec's named
   area, so vector recall already surfaces it.

A 2-file plan with all metrics saturated at 100%/0% has **no headroom** to detect
a ledger effect (ceiling effect). The ledger demonstrably does **no harm**
(ON == OFF, no regression), but its **benefit is unmeasurable in this regime**.

**Conclusion.** Item 2 is coupled to item 1: a positive quantification requires
the item-1 stress scenario — a spec whose relevant files are NOT vector-recall
reachable (so the `file_map` is the only carrier) and/or enough findings to force
`_compress_findings`. Reuse `SPINE_DISABLE_FINDINGS_LEDGER` for that A/B.

**Repro.**
```
SPINE_BENCH_REPO=/home/pat/projects/spine \
SPINE_BENCH_DIR=$WORKTREE/scratch/implement_bench \
SPINE_BENCH_BASELINE=/home/pat/projects/spine/scratch/implement_bench/baseline \
PYTHONPATH=$WORKTREE \
[SPINE_DISABLE_FINDINGS_LEDGER=1] \
.venv/bin/python scratch/implement_bench/bench.py replan --provider local-qwen --force
# then: python score_ledger_ab.py runs/<dir>/plan/plan.json --repo ... --pin 5f5a5d3 --research-log runs/<dir>/plan/research_log.json
```
Preserved run dirs (gitignored): `runs/replan-local-qwen-ledger{ON,ON-rerun,OFF}`.
