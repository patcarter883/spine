# Plan: Migrate codebase intelligence to codebase-memory-mcp

**Status:** proposed (not started)
**Author:** Claude (session 2026-07-10), reviewed by Pat
**Target:** replace `mcp-codebase-index` + the local PHP fallback with
[DeusData/codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp)
as the structural-intelligence backend behind `CodebaseQueryTool`.

---

## Why

1. **PHP (and every other language) natively.** The entire
   `codebase_query_local.py` fallback — `_looks_empty()` sniffing, FTS5
   structural queries, `symbol_edges` — exists only because
   `mcp-codebase-index` has no PHP analyzer. codebase-memory-mcp ships
   tree-sitter coverage for 158 languages with enhanced semantic resolution
   for 12 (Python, TS/JS, PHP, C#, Go, C, C++, Java, Kotlin, Rust, Perl).
2. **Kills the per-language extension burden.** Adding a language to spine's
   structural coverage today means a new tree-sitter grammar dependency, an
   `_EXT_TO_LANG` entry, and a hand-written query string in
   `spine/agents/tools/ast_extract.py`. With the new backend, structural
   coverage for a new language is already there.
3. **New capabilities.** `trace_path` (call-chain traversal),
   `detect_changes` (git-diff → impacted call paths), `get_architecture`
   (repo architecture summary) map directly onto spine phase needs:
   researcher call-chain questions, verify/gap-plan diff-impact analysis,
   and the onboarding analyzer respectively.
4. **Token efficiency.** The consolidated-tool docstring in
   `codebase_query.py` records researcher branches exhausting the 50-step
   recursion cap on iterative exploration; a graph backend answers
   relationship questions in one call.

## What it is (for the executing agent)

codebase-memory-mcp is a static C/C++ binary (zero runtime deps, SQLite
storage in `~/.cache/codebase-memory-mcp/`, LZ4-compressed graphs) exposing
~14 MCP tools over stdio, plus a CLI mode where every MCP tool is callable
from the command line. Key tools: `index_repository`, `search_graph`,
`trace_path`, `detect_changes`, `query_graph` (read-only openCypher subset),
`get_code_snippet`, `get_architecture`, `index_status`, `list_projects`,
`delete_project`. Release binaries carry SLSA L3 provenance and Sigstore
cosign signatures. Fully local; read-only analysis; no embedded LLM.

## Current state map (what touches what)

| Component | File | Role today |
|---|---|---|
| `CodebaseQueryTool` | `spine/agents/tools/codebase_query.py` | The ONLY sanctioned agent entry point. 5 actions (`find_symbol`, `get_source`, `get_dependencies`, `get_dependents`, `search`) dispatched via `_ACTION_TO_MCP` to `mcp_codebase-index_*` tools. Arg-validation guards (markup tokens, nullish words, whitespace) exist because small local models emit malformed args — **backend-independent, must survive the migration**. Phase-aware search caps (`search_cap_for_subagent`). |
| Local fallback | `spine/agents/tools/codebase_query_local.py` | Serves the same 5 actions from `.spine/spine.db` (`symbol_metadata` + `symbol_fts` + `symbol_edges`) when the MCP result `_looks_empty()` — i.e. for PHP. |
| AST extraction | `spine/agents/tools/ast_extract.py` | Hand-wired tree-sitter for exactly 5 languages (py, php, ts, c, cpp). Two consumers: symbol **chunking** for the embedding pipeline, and `extract_edges` for the fallback's dependency queries. |
| Vector indexer | `spine/workflow/workers/vector_indexer.py` | Chunks via `ast_extract`, LLM-summarizes, embeds (nomic, vec0), writes `symbol_metadata`/`symbol_fts`/`symbol_edges`. |
| Vector store | `spine/persistence/vector_store.py` | Owns `.spine/spine.db` schema incl. `symbol_edges` (written only by vector_indexer, read only by codebase_query_local). |
| Programmatic facades | `codebase_query.list_files`, `.list_file_symbols`, `.find_symbol` | Non-agent callers: onboarding analyzer, decomposer enrichment, implement-phase anchor scrubbing. These reach `codebase_query_local` through the facade. |
| MCP config | `.spine/config.reference.yaml` `mcp_servers:` block | stdio-only; currently one entry: `codebase-index` → `mcp-codebase-index`. |
| Prompts | `spine/agents/subagents.py` (~line 145–171) | Researcher prompt documents the 5-action table. Also `plan_do.py`, `plan_tools.py`, onboarding templates mention `codebase_query`. |
| Tests | `tests/unit/test_codebase_query_tool.py` + friends (`test_symbol_cache.py`, `test_mcp_path_exclusions.py`, …) | Contract tests for the wrapper. |

**Consumers that must keep working:** researcher supervisor, implement
subgraph, plan_reference_gate, decomposer, symbol_cache, context_editing,
onboarding analyzer.

---

## Phase 0 — Evaluation spike (gate: go/no-go)

No spine code changes. Goal: verify the marketing claims on our actual
workloads before committing.

1. Download a release binary; **verify the Sigstore signature and SLSA
   provenance** (`cosign verify-blob` per their README). Record versions.
2. Index a real PHP target repo and the spine repo itself. Record wall time
   and db size.
3. Quality comparison, PHP: pick ~10 symbols we know from past work
   (methods with cross-file callers, dynamic dispatch sites). For each,
   compare `get_dependents` / `get_dependencies` answers from
   codebase-memory-mcp vs the current local index (`codebase_query_local`).
   PHP dynamic dispatch is exactly where structural graphs go thin —
   **this is the go/no-go check**.
4. Quality comparison, Python: same for ~10 spine symbols vs
   `mcp-codebase-index` answers.
5. Capture the **actual MCP tool schemas** (`tools/list` over stdio, or CLI
   `--help`). We need exact names/args for Phase 1 mapping — the README's
   tool list may not match the wire names.
6. Check indexing freshness semantics: does it watch files, or does it need
   explicit re-index? How stale are results after an edit mid-implement-phase?
   (The current vector_indexer re-indexes; the implementer edits files and
   then queries.)

**Exit criteria:** PHP edge quality ≥ the local index on the sample set;
re-index path understood; schemas captured into this doc or a sibling note.

## Phase 1 — Backend swap behind a config flag

1. Add an `mcp_servers` entry `codebase-memory` (stdio, the new binary) to
   `.spine/config.reference.yaml`, alongside the existing `codebase-index`.
2. Add a config key (e.g. `codebase_query.backend: codebase-index |
   codebase-memory`, default `codebase-index`) surfaced through the same
   config plumbing as other keys (see `spine/ui/_pages/config_view.py` for
   the UI surface).
3. In `codebase_query.py`, introduce a second action→tool map for the new
   backend and a thin **response adapter** normalizing its output shapes
   into what `_parse_tool_result` / consumers expect. Do NOT touch the
   arg-validation guards, nullish handling, or phase-aware search caps —
   they are model-side protections, not backend-specific.
4. Map the five actions (exact mapping to be pinned from Phase 0 schemas):
   - `find_symbol` → `search_graph`
   - `get_source` → `get_code_snippet`
   - `get_dependencies` / `get_dependents` → `query_graph` or `search_graph`
     over `CALLS`/`IMPORTS` edges
   - `search` → `search_graph` (structural) — keep result-cap behaviour.
5. Ensure the repo is indexed before first query: hook `index_repository` /
   `index_status` into wherever the MCP session is established for a
   workspace (mirror how vector_indexer decides freshness), NOT per-call.
6. Tests: parametrize `test_codebase_query_tool.py` over both backends
   (mock MCP layer); add adapter-shape tests from real captured responses.

**Exit criteria:** full unit suite green with flag off AND on; one live
research branch (SPECIFY/PLAN on a PHP target) run with the flag on,
producing findings without local-fallback hits.

## Phase 2 — New actions

Extend the `CodebaseQueryAction` enum (only when the flag selects the new
backend — the tool should omit unsupported actions from its schema, not
error at runtime):

1. `trace_path` — researcher + slice-verifier. Answers "how does A reach B"
   in one call.
2. `impact` (backed by `detect_changes`) — wire into the verify/gap-plan
   phase so the verifier gets "call paths impacted by this slice's diff"
   instead of reconstructing it by reading files.
3. `architecture` (backed by `get_architecture`) — onboarding analyzer
   (`spine/work/onboarding/analyzer.py`) as a structured supplement to its
   current AST + summary enrichment.

Update the researcher prompt's action table in `subagents.py` (and the
prompt-consistency tests) in the same change — prompts must only name
actions the tool actually exposes.

**Exit criteria:** prompts and tool schema agree; one live trace showing the
researcher using `trace_path` productively (fewer steps than the equivalent
`get_dependents` chain).

## Phase 3 — Flip default; delete the fallback

Precondition: the flag has been default-on locally for enough runs to trust
it (suggest: all phases on a PHP work item + a Python work item, no
fallback regressions).

1. Default `codebase_query.backend: codebase-memory`; keep `codebase-index`
   config supported for one release, then remove.
2. Delete `codebase_query_local.py`, `_looks_empty()`, and the fallback
   branch in `CodebaseQueryTool`.
3. **Re-point the programmatic facades** before deleting: `list_files`
   (pure filesystem walk — can stay as-is), `list_file_symbols` and
   `find_symbol` (used by decomposer enrichment + implement anchor
   scrubbing) move to the new backend's CLI or MCP calls, or to
   `symbol_metadata` (which survives — see Phase 4 note).
4. Drop `symbol_edges`: remove `extract_edges` calls from
   `vector_indexer.py`, `replace_file_edges` + table/indexes from
   `vector_store.py` (alembic migration), and `extract_edges` from
   `ast_extract.py`.
5. Sweep prompts/docs for stale references (`mcp_codebase-index`,
   local-index fallback wording in `codebase_query.py` docstring,
   onboarding templates).

**Exit criteria:** grep for `codebase_query_local|symbol_edges|_looks_empty`
returns nothing outside git history; full suite green; live PHP + Python
work items pass end-to-end.

## Phase 4 (optional, separate decision) — chunker rework

`ast_extract.extract_symbols` survives Phases 0–3 as the embedding
pipeline's chunker, still limited to 5 languages. Option: rework
`vector_indexer` to enumerate symbol boundaries via the new server
(CLI/`get_code_snippet`/`search_graph`) so embedding coverage tracks 158
languages and `ast_extract.py` is deleted entirely. Decide after Phase 3
based on whether embedding coverage beyond the 5 languages is actually
needed. The LLM-summary + embedding pipeline itself (`enriched_summary`,
`search_similar`, vec0) is **out of scope** — the new backend has no
embedded LLM and does not replace semantic-similarity search.

## Risks / open questions

- **PHP edge quality** (Phase 0 gate). Dynamic dispatch, magic methods,
  container-resolved services may produce thin `CALLS` edges.
- **Staleness during implement phase**: the implementer edits then queries;
  if re-indexing is manual and slow on large repos, mid-slice queries can
  return pre-edit answers. Phase 0 item 6 must answer this; may need a
  re-index hook after slice apply.
- **Project maturity**: young project, opaque binaries. Mitigations: pin an
  exact release, verify signatures in the install step, keep the old
  backend flag-restorable through Phase 3.
- **Index location**: it stores graphs in `~/.cache/codebase-memory-mcp/`,
  not in-workspace. Check multi-workspace / sandbox behaviour (spine-sandbox
  runs) and whether the cache path is configurable.
- **Schema drift**: their MCP tool schemas are not a stable contract yet;
  the response adapter in Phase 1 is the isolation layer — keep all shape
  knowledge there.
- **Arg-shape regressions on small local models**: new tool args may invite
  new malformed-arg patterns. Keep validator-level guards; extend them from
  live-trace evidence, as before (see docstring history in
  `codebase_query.py`).
