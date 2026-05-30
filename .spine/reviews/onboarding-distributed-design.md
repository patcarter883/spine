# Onboarding Distributed Engine — Reconciled Design

**Status:** Design (recommended), **Revision 2**. No feature code in this document.
**Author:** Lead architect (reconciliation of Proposal 1 "reuse-first subgraph" and Proposal 2 "purpose-built isolated StateGraph"; Rev 2 incorporates token-budget forensics + a stronger manager/worker hierarchy for local inference models).
**Scope:** Distribute the onboarding engine's heavy synthesis stage across a *two-tier manager → section-worker hierarchy* with **bounded per-call context**, so it runs reliably on local inference models with a **60,000-token window**. Keep analysis deterministic. Preserve the existing dispatch contract, queue lifecycle, UI phase labels, and idempotent artifact writes.

---

## 0. What changed in Revision 2 (and why)

Rev 1 proposed "one worker per document, each fed ~¼ of the manifest." Token-budget forensics on the *first onboarding run* (which hit >40k of a 60k window) showed that framing was unsafe at scale:

| Component | ~tokens | Type | Source |
|---|---|---|---|
| `repo_manifest.json` inlined in the synthesis prompt | **~26,700** | content | `synthesis.py:114` (`_build_synthesis_prompt`, `Tag.FINDINGS`) |
| Same manifest **again** via `read_repo_manifest` tool | **+~26,700** | content | `synthesis_tools.py` `ReadRepoManifestTool` |
| Full Deep-Agent overhead (SPINE_BASE_PROMPT + filesystem + MCP guidance + skills frontmatter) | ~1,800 | fixed | `factory.py:409–459` |
| Synthesis role/workflow/constraints + tool schemas | ~900 | fixed | `synthesis.py:53–91`, `synthesis_tools.py` |

**Three corrections this forces:**

1. **The Deep-Agent framework is NOT the problem (~1.8k, 3% of the prompt).** Do not chase framework weight. AGENTS.md is already skipped for `PhaseName.PLAN` (`skills_resolver.py` `_SKIP_AGENTS_MD`), and PLAN loads no skill bodies (frontmatter only).
2. **The manifest is the problem (~27k for *spine alone*, loaded twice).** It scales with repo size — for a large repo even a ¼-slice can exceed 60k. A *per-document* worker therefore does **not bound** leaf context. We must decompose **below the document, to bounded manifest fragments**.
3. **The agentic tool loop is a multiplier.** A `build_phase_agent` worker re-sends the full prompt prefix every turn (the researcher refactor notes the legacy loop averaged 3–5 model calls/turn) *and* the `read_repo_manifest` tool double-loads the manifest. Leaf workers should be **bare structured LLM calls** (one call, no tools, no loop), not Deep Agents.

Net: **lean harder on the manager/worker hierarchy** (your guidance), make every LLM call's context **O(section)** not **O(manifest)**, and drop the agentic loop from leaf work.

---

## 1. Problem & Goal

### 1.1 What onboarding does today

`spine/work/onboarding/engine.py::run_onboarding` is a plain coroutine the dispatcher routes to (NOT a LangGraph workflow), branching on `work_type == "onboarding"` at `spine/work/dispatcher.py:219`, bypassing `build_workflow_graph`/`compose.py`. Three sequential stages:

1. **analyze** — `RepoAnalyzer.analyze()` (`analyzer.py:94`). **Almost entirely deterministic Python, zero LLM calls**: file discovery (MCP `list_files` index, `os.walk` fallback, cap `_MAX_FILES=400`), tree-sitter `extract_symbols` (byte-sliced, no line reads), optional summary enrichment from a *pre-indexed* VectorStore (`_load_summaries`, no embeddings at analysis time), module-boundary grouping (`_build_boundaries`, key_symbols cap 8), import-graph edges (`_build_dependency_edges`), heuristic patterns (`_extract_patterns`, evidence cap 3). Output: one `RepoManifest` (`manifest.py`), persisted as `repo_manifest.json`.
2. **scaffold** — greenfield only; deterministic `scaffold_project()`. No LLM.
3. **synthesize** — `synthesize_artifacts()` (`synthesis.py:137`). **The only LLM stage, and the bottleneck.** A single Deep Agent is handed the **entire manifest twice** and must author all **four** documents in one agent lifetime; if any `.md` is missing afterward, it raises `RuntimeError` (all-or-nothing).

### 1.2 Where the context pressure actually is (precise)

- **Analysis is NOT a context-window problem.** Zero model calls. Distributing it across *LLM* explorers would *create* token cost that does not exist. → Analysis stays deterministic (§2.4); an opt-in LLM-enrichment mode is available behind a flag but is not the point.
- **Synthesis IS the problem, and the driver is the manifest, not the document count.** ~27k tokens for spine, loaded twice, on a 60k window — and unbounded as repos grow.

**Goal:** Make synthesis fit a 60k local model with comfortable headroom **regardless of repo size**, by:
1. Never sending the whole manifest to any LLM. A **compact index** (~1–3k) drives planning; **bounded fragments** (≤ a configured cap, default 6k) drive writing.
2. A **two-tier hierarchy**: a *documentation manager* plans sections from the index; *section workers* write one section each from one fragment.
3. **Bare structured LLM calls** for both tiers (no tools, no agentic loop, no `read_repo_manifest`).
4. Per-section retry/checkpoint (kills the all-or-nothing failure mode).
5. Preserving dispatch contract, queue lifecycle, UI labels, idempotency **bit-for-bit**.

---

## 2. Recommended Architecture

### 2.1 Base & key decisions (unchanged from Rev 1 where still valid)

- **Isolated, in-engine `StateGraph`** the engine builds, compiles with a per-work `AsyncSqliteSaver`, and `ainvoke`s. **Not** registered in `compose.py`/`WORKFLOW_SEQUENCES`/`_SUBGRAPH_BUILDER_REGISTRY` — onboarding needs no critic/rework/human-review/inter-phase merge, and registration would force `PhaseName` enum churn through every exhaustive map (the documented Risk #1). Dispatcher branch at `dispatcher.py:219` stays unchanged.
- **One graph, three logical phases**: analysis map-reduce → **synthesis plan** → **synthesis fan-out** → assemble. Manifest flows in-state; persisted once for idempotency + UI read.
- **No new reducers**: reuse `operator.add`, `_slice_list_reducer`, `_merge_read_cache`. Single-writer channels get no reducer.
- **Deterministic-default analysis** (byte-identical to today's `analyze()`, parity-tested), map-reduce topology exercised; opt-in LLM enrichment behind a flag.

### 2.2 The synthesis hierarchy (the heart of Rev 2)

Three roles, only the middle one is heavy on LLM calls and each call is small:

**(A) Documentation Manager — one bare structured LLM call (`with_structured_output`), no tools.**
- **Input:** a **compact manifest index** built deterministically by `manifest_index(manifest)` — domain names, module names + one-line roles + size, pattern *categories* (names only, no evidence), dependency *edge count* per module, top-N largest modules. **Excludes** `key_symbols`, pattern `evidence`, `raw_code`, full edge lists. Target ≤ ~2–3k tokens **regardless of repo size** (it lists names, not bodies; if a repo has thousands of modules, the index ranks + caps to the top-K modules and groups the tail).
- **Output:** a `SectionPlan` — for each of the 4 docs, an **ordered list of sections**, each `{doc_id, order, title, fragment_keys, instruction}`. `fragment_keys` reference manifest entries by stable key (module name / pattern category / domain id) — *not* content.
- **Determinism floor:** the plan has a deterministic skeleton the manager only *refines* (group tiny modules, pick/order top-K, drop empty sections). If the manager call fails or the model is weak, fall back to the deterministic skeleton (below) — synthesis still works without the LLM manager. Greenfield → fixed minimal skeleton.

  Deterministic skeleton (the fallback and the manager's starting point):
  - `ARCHITECTURE_MAP` → one section per **module group** (or per top-K module; tail grouped) — fragment = that group's boundaries + its dependency edges.
  - `CODING_GUIDELINES` → one section per **pattern category** — fragment = that category's findings (with evidence).
  - `PROJECT_DEFINITION` → one section per **core domain** — fragment = that domain's module roles.
  - `SPINE_ASSISTANCE_REQUIREMENTS` → 1–2 sections — fragment = size/budget signals (symbol/file counts, largest modules, notes).

**(B) Section Workers — one bare structured LLM call each, fanned out via `Send`, no tools, no loop.**
- **Input per worker:** ONLY `resolve_fragment(manifest, section.fragment_keys, token_cap)` (a small dict, sub-sliced/truncated to `onboarding_section_token_cap`, default 6k) + `section.instruction` + a short doc-level "voice/role" string. **No whole manifest, no read tool.**
- **Output:** `SectionResult{doc_id, order, markdown, status}`.
- **Bounded:** worst-case prompt ≈ hand-written system prompt (~0.4k) + fragment (≤6k) + instruction (~0.3k) ≈ **<7k**, plus a short single-doc-section completion. Fits 60k with ~8× headroom and **does not grow with repo size** (only the *number* of sections grows, and those are independent `Send` branches).

**(C) Document Assemblers — deterministic, per doc (no LLM).**
- Sort the doc's `SectionResult`s by `order`, concatenate markdown under a doc title, write `<NAME>.md` via `WriteOnboardingDocTool` (idempotent `overwrite_shorter=True`).
- Then `aggregate_synthesis` verifies all 4 `.md` exist → `RuntimeError` listing any missing (preserved verbatim from `synthesis.py:205–223`). Per-section `status != ok` is surfaced before assembly so a single bad section fails loudly (or is retried via checkpoint) rather than producing a silently-truncated doc.

### 2.3 ASCII node / Send diagram (synthesis tiers)

```
                RepoManifest (in-state, from analysis; never sent whole to any LLM)
                                   │
                    manifest_index(manifest)  ── deterministic, ≤~2-3k tokens
                                   │
                                   ▼
                      ┌───────────────────────────┐
                      │   doc_manager (TIER A)     │   ONE bare LLM call,
                      │  with_structured_output    │   with_structured_output(SectionPlan),
                      │      -> SectionPlan         │   NO tools. Falls back to the
                      └────────────┬──────────────┘   deterministic skeleton on failure.
                                   │  sections[] (each: doc_id, order, title,
                                   │              fragment_keys, instruction)
                 _section_router(state) -> list[Send]   (one Send per section;
                                   │                      payload = resolve_fragment(...) ≤6k)
        ┌──────────────┬──────────┼──────────┬──────────────┬───────────────┐
        ▼              ▼                     ▼              ▼               ▼
 ┌────────────┐ ┌────────────┐       ┌────────────┐ ┌────────────┐  ┌────────────┐
 │section_wkr │ │section_wkr │  ...  │section_wkr │ │section_wkr │  │section_wkr │  TIER B
 │ARCH §mod1  │ │CODING §log │       │PROJ §domX  │ │ARCH §mod2  │  │SPINE §size │  bare LLM,
 │frag≤6k     │ │frag≤6k     │       │frag≤6k     │ │frag≤6k     │  │frag≤6k     │  1 call each,
 │→ section_  │ │→ section_  │       │→ section_  │ │→ section_  │  │→ section_  │  NO tools
 │  results+= │ │  results+= │       │  results+= │ │  results+= │  │  results+= │
 └─────┬──────┘ └─────┬──────┘       └─────┬──────┘ └─────┬──────┘  └─────┬──────┘
       │   (operator.add merges section_results race-free)               │
       └──────────────┴──────────┬──────────┴──────────────┴───────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │  assemble_docs (TIER C)     │  group section_results by doc_id,
                   │   deterministic, per doc    │  sort by order, concat markdown,
                   │   WriteOnboardingDocTool ×4 │  write <NAME>.md (idempotent)
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │     aggregate_synthesis     │  verify all 4 <NAME>.md exist;
                   │  RuntimeError if any absent │  return {doc_name: path}
                   └──────────────┬──────────────┘
                                  ▼  END → engine finalises work_entries + ws_bus
```

Analysis (Phase A) is unchanged from Rev 1 (deterministic map-reduce → manifest); see §2.4.

### 2.4 Analysis (unchanged, summarized)

Single-round deterministic map-reduce: `analysis_manager` groups symbols into module units → `Send` one `analysis_explorer` per unit (pure analyzer math by default; opt-in LLM enrichment behind `onboarding_explorer_llm`) → `aggregate_analysis` resolves **global** dependency edges over the union of `raw_imports`, dedupes/re-caps patterns, assembles + persists the `RepoManifest` once. `onboarding_distributed_analysis=False` falls back to monolithic `analyze()` for tiny repos. Byte-parity test guards equivalence.

### 2.5 Why this is "leaning harder on the manager/worker" — and the trade-off

- The **manager** does real decomposition work (sectioning, grouping, prioritization, ordering) but from a *compact index*, so its own context is bounded and cheap.
- **Workers** are many, tiny, independent, retryable — the canonical supervisor/worker shape, now at *section* granularity instead of document granularity.
- **Trade-off:** more LLM calls (1 manager + ΣN_sections, vs. 1 agent loop). For local inference this is the *correct* trade: many small bounded calls that each fit the window beat one call that doesn't. Section count is capped (`onboarding_max_sections`, default 32; tail modules grouped) so call volume stays predictable. Calls fan out via `Send` (graph concurrency cap applies; tune for single-GPU servers via config).

---

## 3. State Schema + Reducers

New file: `spine/work/onboarding/onboarding_state.py`. Kept OUT of `subgraph_state.py` (isolation). Imports the two proven reducers it needs.

```python
from operator import add as _op_add
from typing import Annotated, Any
from typing_extensions import TypedDict

from spine.models.state import _merge_read_cache              # right-wins (LLM-enrich mode only)
from spine.workflow.subgraph_state import _slice_list_reducer  # {"add":[...], "remove":[id...]}


class OnboardingGraphState(TypedDict, total=False):
    # ── inputs (seeded by the engine before ainvoke) ──
    work_id: str
    workspace_root: str
    mode: str                       # "brownfield" | "greenfield"
    tech_stack: list[str]

    # ── PHASE A: analysis ──
    analysis_units: list[dict]      # set ONCE by manager; single writer → no reducer
    active_unit: dict               # transient, per-Send
    repo_slices: Annotated[list[dict], _op_add]      # each explorer appends [one slice]
    manifest: dict                  # RepoManifest.to_dict(); set ONCE by aggregator
    manifest_path: str              # set ONCE by aggregator

    # ── PHASE B: synthesis plan (TIER A) ──
    manifest_index: dict            # compact index; set ONCE by the plan prelude
    sections: Annotated[list[dict], _slice_list_reducer]  # SectionPlan items; seeded once
    active_section: dict            # transient, per-Send: {doc_id, order, title, fragment, instruction}

    # ── PHASE B: synthesis fan-out (TIER B/C) ──
    section_results: Annotated[list[dict], _op_add]       # each worker appends [one SectionResult]
    written: dict                   # {doc_name: path}; set ONCE by aggregate_synthesis

    # ── shared dedupe (LLM-enriched explorer mode ONLY) ──
    read_cache: Annotated[dict, _merge_read_cache]
```

**Reducer rationale (no new reducers):**
- `operator.add` on `repo_slices` and `section_results` — each parallel branch emits exactly `[one_dict]`; LangGraph applies reducer updates sequentially within a super-step, so N branches compose race-free (the `findings`/`verification_results` pattern at `subgraph_state.py:207,149`).
- `_slice_list_reducer` (`subgraph_state.py:73`) on `sections` — reused verbatim; seeds the section list and keeps the door open for dynamic add/remove (e.g. a manager that re-plans).
- `_merge_read_cache` on `read_cache` — LLM-enrich mode only.
- **No reducer** on `analysis_units`, `manifest`, `manifest_path`, `manifest_index`, `written` — single-writer channels; a reducer here is the documented "aggregate appends/duplicates" footgun. Aggregate/assemble nodes **transform but do not re-emit** the `_op_add` channels.

---

## 4. Concrete Component List

### 4.1 New files

| Path | Purpose |
|---|---|
| `spine/work/onboarding/onboarding_state.py` | `OnboardingGraphState` (§3). |
| `spine/work/onboarding/onboarding_graph.py` | `build_onboarding_graph()` → uncompiled `StateGraph`; route maps; base payload builders. |
| `spine/work/onboarding/analysis_nodes.py` | `_analysis_manager_node`, `_analysis_router`, `_analysis_explorer_node`, `_aggregate_analysis_node`; `_build_boundary_for_unit`, `_extract_patterns_for_unit`. |
| `spine/work/onboarding/manifest_index.py` | `manifest_index(manifest) -> dict` (compact, bounded, ranked+capped) and `resolve_fragment(manifest, fragment_keys, token_cap) -> dict` (the core context fix). Pure functions. |
| `spine/work/onboarding/synthesis_nodes.py` | `_doc_manager_node` (Tier A, bare LLM + deterministic skeleton/fallback), `deterministic_section_plan(index, mode)`, `_section_router`, `_section_worker_node` (Tier B, bare LLM), `_assemble_docs_node` (Tier C), `_aggregate_synthesis_node`; `SectionPlan`/`SectionResult` Pydantic schemas; doc-level voice strings. |
| `tests/unit/test_onboarding_manifest_index.py` | `manifest_index` bound (asserts ≤ cap + excludes key_symbols/evidence even for a large fixture) and `resolve_fragment` projection/sub-slicing per key-set. |
| `tests/unit/test_onboarding_synthesis_plan.py` | `deterministic_section_plan` skeleton per doc; manager-fallback path (LLM mocked to fail → skeleton used); greenfield minimal plan. |
| `tests/unit/test_onboarding_graph.py` | end-to-end graph (manager + workers mocked), per-section fan-out, all-4-docs assembly + `RuntimeError`-on-missing, greenfield short-circuit. |
| `tests/unit/test_onboarding_analysis_parity.py` | distributed deterministic analysis == monolithic `analyze()` (name-sorted) under shuffled slice order. |

### 4.2 Changed files — exactly what changes

| Path | Change |
|---|---|
| `spine/work/onboarding/synthesis.py` | **REPLACE the single-agent driver.** Orchestration moves to `synthesis_nodes.py`. Move the all-4-exist verification + `RuntimeError` (lines 205–223) into `_aggregate_synthesis_node`. **Delete** `_build_synthesis_prompt`'s whole-manifest inlining; per-*section* prompts are built from `resolve_fragment(...)`. Keep `synthesize_artifacts` as a thin **back-compat shim** that builds+invokes the synthesis half of the graph (so existing imports/tests survive migration). |
| `spine/work/onboarding/synthesis_tools.py` | **`WriteOnboardingDocTool` reused** but called by the deterministic assembler (Tier C), not by an LLM — so its `Literal`-of-4 schema stays; no per-branch tightening needed. **`ReadRepoManifestTool` is removed from the synthesis path entirely** (no LLM reads the manifest) — keep the class for the UI/back-compat manifest read only, or drop if unused. `ONBOARDING_DOC_NAMES`/`ONBOARDING_PHASE`/`build_synthesis_tools` reused. |
| `spine/work/onboarding/analyzer.py` | **REFACTOR (split, don't rewrite)** — extract `_build_boundary_for_unit`, make `_extract_patterns` accept a symbol subset, move grouping into the manager, move `_build_dependency_edges` call site to the aggregator (needs global module set). `RepoAnalyzer.analyze` stays as the monolithic fallback + parity baseline. All low-level helpers reused verbatim. |
| `spine/work/onboarding/engine.py` | **REFACTOR `run_onboarding`** — keep `_ensure_work_entry`, `_persist_manifest` (called in the aggregator via config), queue-lifecycle calls, final `work_entries.update`, ws_bus publish, try/except. Replace the inline `analyze()→scaffold→synthesize_artifacts()` body with: greenfield `scaffold_project()` (det., pre-graph) → `build_onboarding_graph()` → compile with per-work `AsyncSqliteSaver` at `.spine/checkpoints/<work_id>/onboarding.db` (mirror `subgraph_wrapper._get_phase_checkpointer`; no-checkpointer fallback on failure) → seed state → `ainvoke`. Thread a progress callback through `RunnableConfig["configurable"]` so `current_phase` strings (`analyze`/`scaffold`/`synthesize`/`completed`) + ws events are preserved **bit-for-bit**. Return dict unchanged. |
| `spine/config.py` | **ADD** flags: `onboarding_distributed_analysis: bool = True`, `onboarding_explorer_llm: bool = False`, `onboarding_explorer_max_cycles: int = 3` (under `ConvergenceConfig`), **`onboarding_section_token_cap: int = 6000`** (per-fragment ceiling), **`onboarding_max_sections: int = 32`** (call-volume cap; tail modules grouped). Per-phase model overrides recognized via existing `providers.phases` convention: **`onboarding/doc-manager`** and **`onboarding/section-worker`** (point section-worker at the cheapest capable local model). Env overrides per convention. |
| `spine/work/dispatcher.py` | **UNCHANGED** (line 219 branch stays). |
| `spine/ui/_pages/onboarding.py` | **UNCHANGED** — phase labels preserved because the engine writes the same `current_phase` strings at node boundaries. |
| `tests/unit/test_onboarding_engine.py`, `tests/unit/test_onboarding_synthesis*.py` | **UPDATE** to drive the graph/shim and assert the section fan-out. |

---

## 5. Dispatch & Progress Integration

Unchanged from Rev 1: `submit_work` → `dispatcher.py:219` → `run_onboarding` (same return dict); `_ensure_work_entry` seeds the row; the engine owns terminal `work_entries.update` + ws_bus `work_completed`/`work_failed`. Progress callback threaded through `RunnableConfig["configurable"]` fires `update_work_phase_started`/`_update_work_progress` at node boundaries with the **same phase strings the UI expects** (`analyze`, `scaffold`, `synthesize`, `completed`). Idempotency preserved (`_persist_manifest` once; `WriteOnboardingDocTool` idempotent). **New benefit:** per-work checkpointer gives selective resume — a re-run can skip already-written sections.

**MEMORY rule (never leak tool-error text into findings/docs):** the LLM-enrich explorer error sentinel carries only `{error: True, module_name}` (no exception text), filtered by the aggregator. Section workers make no tool calls, so they have no tool-error text to leak; a failed section returns `status="error"` with a generic reason, never raw exception text embedded in markdown.

---

## 6. Reuse Map

### 6.1 Reused VERBATIM
- **`analyzer.py`** low-level helpers (`_discover_files*`, `_extract_symbols`, `_load_summaries`, `_symbol_ref`, `_describe_module`, `_module_of`, `_module_names`, `_iter_imports`, `_match_module`, `_infer_tech_stack`); `_build_dependency_edges` (runs in aggregator).
- **`manifest.py`** entire `RepoManifest`/`SymbolRef`/`ModuleBoundary`/`DependencyEdge`/`PatternFinding` tree + `to_dict`/`from_dict` — the cross-node contract, **unchanged**.
- **`scaffold.py`** untouched.
- **`synthesis_tools.py`** `WriteOnboardingDocTool`, `ONBOARDING_DOC_NAMES`, `ONBOARDING_PHASE`, `build_synthesis_tools`.
- **Reducers:** `_slice_list_reducer` (`subgraph_state.py:73`), `operator.add`, `_merge_read_cache`.
- **Engine plumbing:** `_ensure_work_entry`, `_persist_manifest`, `update_work_phase_started`, `_update_work_progress`, `get_bus().publish_sync`.
- **Per-work checkpointer:** mirror `subgraph_wrapper.py:_get_phase_checkpointer` (`AsyncSqliteSaver` + fallback).

### 6.2 Reused as the **bare-LLM-call primitive** (the Rev 2 lever)
Both Tier A (manager) and Tier B (section worker) follow the **`run_research_manager` shape** — *not* `build_phase_agent`:
- **`spine.agents.helpers.resolve_model(config, session_id, phase)`** → `str | BaseChatModel`; `init_chat_model(model)` if `str`. (`helpers.py:18–84`)
- **`model.with_structured_output(Schema)`** then a single `await structured.ainvoke([SystemMessage(...), HumanMessage(...)])`, with the post-call coercion handling instance/`.parsed`/dict shapes (pattern at `exploration_agents.py:605–633`; also `classification.py`, `decomposer.py:134–140`, `summarise_evidence` at `exploration_agents.py:1238–1312`).
- **`spine.agents.prompt_format`** `Tag`/`xml_blocks`/`hostage_layout` for the tiny prompts.
- **Token metering:** `spine.agents._tokens._count_tokens` to enforce `onboarding_section_token_cap` inside `resolve_fragment`.
- **Optional supervisor↔worker loop** (`researcher_supervisor.py` `run_supervisor_node`/`run_worker_node`, `ToolClass`, `TOOL_CLASS_TO_TOOLNAMES`) reused **only** in the opt-in LLM-enriched *analysis* explorer mode — synthesis workers are single-shot and need no tools.

> Note: leaf workers deliberately **do not** call `build_phase_agent`. Its ~1.8k fixed overhead is small, but the agentic tool loop (multi-turn prefix re-send) and `read_repo_manifest` double-load are what we are eliminating; a single bounded structured call is cheaper and more predictable on local models.

### 6.3 NEW symbols
- `OnboardingGraphState`; `build_onboarding_graph` + route maps.
- Analysis: `_analysis_manager_node`, `_analysis_router`, `_analysis_explorer_node`, `_aggregate_analysis_node`, `_build_boundary_for_unit`, `_extract_patterns_for_unit`.
- `manifest_index(manifest)`, `resolve_fragment(manifest, fragment_keys, token_cap)`.
- Synthesis: `_doc_manager_node`, `deterministic_section_plan`, `_section_router`, `_section_worker_node`, `_assemble_docs_node`, `_aggregate_synthesis_node`; `SectionPlan`, `SectionResult`.
- Config: `onboarding_distributed_analysis`, `onboarding_explorer_llm`, `onboarding_explorer_max_cycles`, `onboarding_section_token_cap`, `onboarding_max_sections`; phases `onboarding/doc-manager`, `onboarding/section-worker`.

### 6.4 `manifest_index` / `resolve_fragment` (the core context fix)

```
manifest_index(manifest) -> {                       # ≤ ~2-3k tokens, NO bodies/evidence
  mode, tech_stack, core_domains,
  modules: [{name, path, role, symbol_count}]        # ranked desc; top-K kept, tail grouped
            (capped to onboarding_max_sections-ish),
  pattern_categories: [name, ...],                   # names only
  edge_counts: {module: out_degree},
  totals: {symbol_count, file_count}, notes }

resolve_fragment(manifest, fragment_keys, token_cap) -> small dict:
  ARCHITECTURE_MAP/<module-group>  → boundaries(full key_symbols) + edges FOR those modules
  CODING_GUIDELINES/<category>     → that pattern category's findings (with evidence)
  PROJECT_DEFINITION/<domain>      → module roles for that domain
  SPINE_ASSISTANCE_REQUIREMENTS    → size/budget signals only
  → then truncate/sub-slice to token_cap (default 6k); if a single module exceeds the cap,
    drop key_symbols to names-only, then truncate — fragment NEVER exceeds the cap.
```

---

## 7. Phased Implementation Plan

Build **synthesis-first** (the real win) and **bottom-up** (pure helpers before the graph).

- **PR-1 — Index + fragment helpers (pure functions, no graph).** `manifest_index`, `resolve_fragment`, `deterministic_section_plan`; `test_onboarding_manifest_index.py` + `test_onboarding_synthesis_plan.py` (skeleton). Asserts the index/fragment bounds hold on a large synthetic manifest. *No engine change.*
- **PR-2 — Synthesis hierarchy graph (manifest in memory).** Tier A (`_doc_manager_node` with deterministic skeleton + LLM refine/fallback), Tier B (`_section_worker_node`, bare LLM), Tier C (`_assemble_docs_node`) + `_aggregate_synthesis_node` (preserve `RuntimeError`). Wire `synthesize_artifacts` as a shim invoking this. `test_onboarding_graph.py` (synthesis half). **This alone fixes the 60k bottleneck** and is independent of analysis changes.
- **PR-3 — Analysis map-reduce (deterministic default).** Refactor `analyzer.py`; build Phase A; global edges in aggregator; guard with `onboarding_distributed_analysis`. `test_onboarding_analysis_parity.py`.
- **PR-4 — One graph + engine integration + checkpointer.** Compose A→B; refactor `engine.run_onboarding` (build/compile/ainvoke + progress callback). Verify dispatcher + UI unchanged.
- **PR-5 (optional) — LLM-enriched analysis explorer** behind `onboarding_explorer_llm` (reuses `run_supervisor_node`/`run_worker_node`).

### 7.1 Mapping onto a future multi-agent IMPLEMENT workflow
The **section fan-out** is the supervisor/worker template at sub-task granularity with `operator.add` accumulation and bounded per-worker context via `resolve_fragment` — directly the discipline a future IMPLEMENT needs (give each slice-implementer only its plan slice). The **analysis map-reduce** demonstrates "map is fully enumerable → single round, no convergence loop" (vs. exploration's open-ended 3-round loop), clarifying *when* heavy exploration machinery is warranted.

---

## 8. Risks & Open Questions

**Risks (with mitigations):**
1. **More LLM calls (1 manager + ΣN_sections).** *Mitigation:* sections capped by `onboarding_max_sections` (tail modules grouped); workers fan out via `Send`; point `onboarding/section-worker` at the cheapest capable model. Each call is bounded and short.
2. **Fragment must never exceed the window.** *Mitigation:* `resolve_fragment` hard-truncates to `onboarding_section_token_cap` (degrade key_symbols→names→truncate); test asserts no fragment exceeds the cap even for a pathologically large module.
3. **`manifest_index` must stay bounded as repos grow.** *Mitigation:* index lists names only and ranks+caps to top-K modules (tail grouped), so it is O(K), not O(repo). Test on a synthetic 5,000-module manifest.
4. **Manager produces an incoherent/empty plan (weak local model).** *Mitigation:* deterministic skeleton is the floor; manager only *refines* it; on parse failure or empty plan, use the skeleton. Greenfield uses a fixed minimal plan.
5. **Section assembly coherence (docs written in fragments).** *Mitigation:* doc-level voice/role string shared across a doc's workers; deterministic ordering by `section.order`; optional future "doc polish" pass (deferred — only if reviews show seams).
6. **Global dependency edges in a fan-out / per-unit pattern dedup (analysis).** *Mitigation:* explorers emit `raw_imports` only; aggregator resolves edges over the union and dedupes/re-caps patterns; parity test asserts equality to the monolith under shuffled order.
7. **Completion-order non-determinism** of `repo_slices`/`section_results`. *Mitigation:* aggregators look up by key (`module_name`/`doc_id`+`order`), never by index; outputs re-sorted deterministically.
8. **Per-work checkpointer is new surface.** *Mitigation:* mirror `subgraph_wrapper.py:251–271` fallback (no checkpointer on failure) — strictly ≥ today's non-resumable behavior.
9. **Local-server concurrency** (single GPU may serialize `Send` branches). *Mitigation:* respect the graph concurrency cap; expose it via config; small calls keep per-branch latency low.

**Resolved decisions (locked):**
- **Manager intelligence — DECIDED: deterministic skeleton + light LLM refinement.** `deterministic_section_plan(index, mode)` is the floor; `_doc_manager_node` makes one bare structured LLM call to refine grouping/ordering/prioritization, and falls back to the skeleton on parse failure or empty/incoherent output. (Not fully-deterministic, not manager-driven.)
- **Worker model — DECIDED: one model for manager + workers.** No separate `onboarding/section-worker` tier required; `onboarding/doc-manager` and `onboarding/section-worker` phase keys may both resolve to the same model by default (the split exists only so it *can* be overridden later, not because it must be).

**Open questions:**
1. **Default `onboarding_section_token_cap`** (6k) and **`onboarding_max_sections`** (32) — confirm against your target local model's real usable window after completion-token reservation.
2. **Keep `synthesize_artifacts` shim** during migration (recommended yes; remove in cleanup PR)?
3. **Single graph vs. two composed graphs** (manifest in-state vs. redundant disk read) — recommended one graph.
4. **Manifest versioning** (pre-existing: no version field) — follow-up to stamp analyzer version so distributed vs. monolithic runs are distinguishable.
