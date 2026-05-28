# Trace 019e6e92: critic rework death-spiral + garbled researcher topics

**Trace ID:** `019e6e92-5ed0-7f61-bc4d-a86665b82e4a`
**Work ID:** `3c906ff1`
**Date:** 2026-05-28
**Workflow:** `reviewed_task` — SPECIFY → PLAN → CRITIC_PLAN
**Description:** *"Add a --verbose flag to the CLI entrypoint to toggle debug logging"* (80 chars)

---

## Headline

A trivial CLI-flag task ran for **12 m 53 s** and consumed **3.16 M prompt tokens** before completing `success`. The artifacts on disk are correct (focused spec, 3 small slices) — but the trace exposes three structural failures the existing fixes don't cover:

1. **Critic rework is uncapped.** Iteration 3 of `critic_plan` ran 203 s and died on `LengthFinishReasonError`. The plan was already acceptable at iteration 1.
2. **The local model's garbled tokens survive into researcher topics.** Sample fragments dispatched to `explore_do`: `"singletonsement"`, `"chefsENDELLIUS"`.
3. **Cross-phase paraphrase isn't dedup'd.** PLAN's research_manager re-asked variants of questions SPECIFY had already answered. The fuzzy-dedup work landing on 2026-05-28 is intra-phase only.

---

## Overview

| Field | Value |
|---|---|
| Duration | 773 s |
| Phases run | specify · plan · critic_plan · plan · critic_plan · plan · critic_plan |
| LLM calls | 293 |
| Tool calls | 364 |
| Prompt tokens | 3,142,050 |
| Completion tokens | 15,336 |
| Aggregate ratio | **205 : 1** |
| `cache_read` | **0** across all 293 LLM calls |
| Errors | 28 |
| Final status | success |

### Phase breakdown

| Phase | Duration | Turns | Prompt | Compl | Send children | Result |
|-------|---------:|------:|-------:|------:|--------------:|--------|
| specify        | 273 s | 141 | 1,708,162 | 6,509 | 8 explore_do  | spec written (2 KB) |
| plan #1        | 213 s | …   | combined 1,402,909 | 8,485 | 7 explore_do | needs_revision |
| critic_plan #1 | 18 s  | 1   | small | — | — | needs_revision |
| plan #2        | 27 s  | …   | small | — | 0 (done) | needs_revision |
| critic_plan #2 | 5 s   | 1   | small | — | — | needs_revision (deps order, scope) |
| plan #3        | 34 s  | …   | small | — | 0 (done) | accepted |
| critic_plan #3 | **203 s** | 1 | — | — | — | **`LengthFinishReasonError`** |

### Per-phase token roll-up

| Phase | LLM calls | Prompt | Compl | Ratio |
|-------|----------:|-------:|------:|------:|
| specify     | 141 | 1,708,162 | 6,509 | 262 : 1 |
| plan        | 144 | 1,402,909 | 8,485 | 165 : 1 |
| critic_plan |   8 |    30,979 |   342 |  91 : 1 |

### Tool calls top-5

| Count | Tool |
|------:|------|
| 143 | `codebase_query` |
|  68 | `mcp_codebase-index_get_function_source` |
|  44 | `mcp_codebase-index_find_symbol` |
|  34 | `mcp_codebase-index_search_codebase` |
|  13 | `ast_extract_symbol` |

### Errors

| Count | Name | Sample |
|------:|------|--------|
| 7 | `spine-specify` | `GraphRecursionError('Recursion limit of 50 reached')` |
| 7 | `spine-plan` | `GraphRecursionError('Recursion limit of 50 reached')` |
| 6 | `codebase_query` | `ToolException: 'pattern' is required for action='search'` |
| 1 | `ChatOpenAI` (critic_plan #3) | `LengthFinishReasonError` |

Salvage paths absorbed the GraphRecursionErrors and CodebaseQueryError storms, so the workflow completed — but each event consumed turns of the 50-step cap.

---

## Issues

### 1 — Critic rework death-spiral, terminated by completion-cap (Critical)

The plan that the critic eventually accepted is structurally identical to plan #1: 3 small slices, each touching one file, traceable to spec requirements R1–R3. Critic #1 and #2 raised legitimate-sounding nits ("dependencies defined after slices that use them", "no specification.json reference for requirements traceability"), but these were not load-bearing for a 3-slice plan. Critic #3 ran for 203 s and emitted a `LengthFinishReasonError` mid-judgement — i.e. the critic's own structured output grew until it blew the completion window.

The existing code records `attempt: 2` on the critic's output (visible in iteration 2 of this trace), so the cycle-count variable exists. There is no enforcement that caps rework at 2 cycles and routes the third attempt to `needs_review`.

**Fix:** in the subgraph that loops `plan → critic_plan`, when the critic returns `needs_revision` and `attempt >= 2`, accept the prior plan and exit the loop with `workflow_status = "needs_review"` rather than re-entering plan a third time. This matches the cap documented in the trace-analysis skill and prevents the LengthFinishReasonError class of termination.

### 2 — Garbled tokens leak into researcher topics (Critical)

Two `explore_do` topics dispatched to subagents in this run:

- `"What global state or configuration singletonsement holds the current log level throughout the application?"`
- `"How does the CLI group and subcommands chefsENDELLIUS handle command-line options and parameter passing to subcommands?"`

These are decoder artefacts from the local model. They survive the `ResearchManagerDecision` Pydantic validation (Pydantic only validates structure, not semantic English), enter the `Send("explore_do", {"topic": …})` payload, and get persisted into `research_log.json`. A downstream paraphrase-dedup pass cannot collapse them against the human-readable equivalents because the strings don't share lexical content.

**Fix:** after `with_structured_output(ResearchManagerDecision)` returns, filter the `topics` list with two cheap regexes — non-ASCII mid-word fragments (`r'[A-Za-z]+[^\x00-\x7f]+[A-Za-z]*'` is a good starting point) and pure-alpha runs ≥12 chars with no vowel pattern matching a small English-like model. Drop garbled topics and emit a sentinel into the manager's next-round context (consistent with the sentinel-topic work in the progress log §4 bullet from today).

### 3 — `— recall symbols:` suffix attaches unrelated symbols (Critical)

`explore_do[14]` was dispatched with topic:

```
What is the relationship between environment variable processing and logging configuration in the CLI initialization sequence? — recall symbols: build_plan_agent (spine/agents/plan_agent.py), _build_t…
```

The topic is about CLI logging. The decorated symbols are from `plan_agent.py` — a file unrelated to the topic. The `a71c7c9` fix was to stop the manager *re-proposing* prior topics by stripping this suffix in `_new_topics` before comparison, but the suffix is still being *appended* in the first place from the manager's accumulated symbol pool.

**Fix:** restrict the suffix to symbols whose file path appears in the current topic text or in a finding produced in this research round. If no symbols qualify, omit the suffix entirely.

### 4 — PLAN re-explored what SPECIFY already discovered (Critical)

SPECIFY rounds 1–3 investigated:
- CLI entrypoint structure and subcommand routing
- Debug callback integration with LangChain
- Env-var-driven logging configuration

PLAN round 1 then asked variants of the same questions:
- *"How does the CLI group and subcommands handle command-line options and parameter passing to subcommands?"*
- *"How is logging configured currently across the spine package, particularly in relation to debug levels and handlers?"*
- *"How does the debug_callback module integrate with LangChain's debugging infrastructure?"*

The intra-phase fuzzy-dedup work landing on 2026-05-28 catches paraphrase within a single research_manager loop, but PLAN's research_manager never sees SPECIFY's findings.

**Fix:** seed the PLAN exploration_subgraph's research_manager prompt with `artifacts.specify['specification.md']` and the SPECIFY `research_log.json` findings as a "what's already known" block. This is the recommendation in `references/downstream-research-context-injection.md`.

### 5 — `codebase_query` schema errors recurring (Warning)

6 × `codebase_query` ToolException — `action='search'` called with no `pattern`. The uncommitted 2026-05-28 commit adds a one-line retry example to the error message, which should let the local model self-correct on the next turn. Verify the new error string is being constructed at the tool boundary (not behind a wrapper that drops it) on the next live trace.

### 6 — Pre-research gate did not short-circuit a trivial reviewed_task (Warning)

The SPECIFY pre_research_gate reported `classification_confidence=0.9, task_category="Infrastructure", retrieved_context=[]` — high confidence, but no RAG hits, so it proceeded to the full exploration loop and dispatched 8 explore_do children for an 80-char task. Either:

- allow the gate to short-circuit on high confidence alone for `reviewed_task` work, or
- mirror the `quick` work-type's complexity heuristic (`len(description) < 150` → skip exploration) on the `reviewed_task` path.

### 7 — Zero `cache_read` on 3.1 M prompt tokens (Warning)

Expected for a local model on OpenRouter / vLLM, but worth confirming this run was not intended for a caching-capable provider (`deepseek-v4-pro`, Anthropic). At 3.1 M prompt tokens, the cache-hit delta against a caching provider is roughly 5× the cost.

---

## Improvement progress

| Area | Status | Evidence |
|------|--------|----------|
| Router parallel dispatch | ✅ | 3 SPECIFY + 3 PLAN explore rounds; 15 Send children |
| Rich researcher topics (≥200 chars) | ⚠️ | 78–136 char range; only the polluted topic 14 ≥200 |
| File re-read prevention | ✅ | Read cache holds; 143 codebase_query calls are distinct symbols |
| Phase artifact quality | ✅ | spec 2 KB, plan 4 KB, 3 well-scoped slices |
| Critic gate efficiency | ❌ | 3 rework cycles, final critic dies on completion-cap |
| Context bloat control | ⚠️ | Max single call 36,754 prompt tokens; SPECIFY total 1.7 M |
| Post-write continuation | ✅ | Single write_specification / write_structured_plan |
| Slice synthesis honesty | n/a | No IMPLEMENT phase |

---

## Follow-up

If fixes 1–4 ship, update `.spine/reviews/token-behavior-progress.md`:

- §1 — note the rework-cycle cap and link this trace.
- §4 — add bullets for the garbled-topic sanitiser, the symbol-suffix scope fix, and the cross-phase findings seeding.
