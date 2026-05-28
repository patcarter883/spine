# Token-usage & model-behaviour progress

**Window:** `2026-05-14 → 2026-05-28` (~80 commits). All hashes are real and resolvable in this repo.

This is a thematic, not chronological, view. Each section sketches the problem, the fixes that landed, and pointers to the commits that did the work. Use it as the entry point for the next round of trace audits.

---

## TL;DR

- **Token-budget enforcement is back, this time wired to real usage data.** A prior attempt was removed in `65683fe` because it triggered phase-wide retries on overrun; `cb4f023` reintroduces it as a per-work_id cumulative tracker that routes overruns to `needs_review`, and fixes the underlying reason the first version was unreliable: `stream_usage=True` is now default on both OpenRouter and local providers.
- **Caching is now correct under parallel dispatch and rework.** `bd152a6` promotes the read-cache to a checkpointed state field (the original cache in `f51d448` was effectively a no-op — 343 redundant tool calls in one trace). `e7e6529` adds Anthropic prefix caching. `4e284ac` shares MCP symbol-fetch results across parallel `Send()` branches and caps concurrency at 4.
- **The RAG/symbol-ingestion overhaul replaces whole-file embedding.** Two-Pass Hybrid RAG (`48fe309` and the cluster around it), tree-sitter symbol slicing (`34a0dbc`), subagent-bound `read_edit_lint`/`ast_extract_symbol` (`f7cf960`), and a SPECIFY pre-research short-circuit (`6277bdf`) collectively stop the researcher from pulling whole files into context.
- **The researcher is materially more robust.** Self-correcting tool loops (`f711064`), raised recursion cap with partial-finding salvage (`80c9a66`, `ad90798`), tool-error text filtered out of findings at every extraction point (`6578e4d`, `90cd8a0`), dedup against enriched topic suffixes (`a71c7c9`), seeded rework state (`4c45c46`), and a consolidated single-tool MCP surface plus a convergence middleware (`6faf0f2`).
- **Eval interpreter and the `tools.task` PTC dispatcher are gone.** `02d2bd2` removes them; all parallelism now runs through LangGraph's Send API via fail-closed routers (`9163957`), matching the established exploration-subgraph pattern.
- **Structured I/O end-to-end, with critic alignment.** SPECIFY/PLAN JSON contract checks (`eab6229`), research-manager structured decisions (`ec0530f`), IMPLEMENT/VERIFY JSON reports + agentic GC (`24eb99c`), OpenRouter native `json_schema` for subagents (`f8f3108`), and the critic verdict now flows through a dedicated `last_critic_review` field rather than tail-scanning `feedback` (`d1a5672`).
- **Cross-phase leak fixes.** Original work description no longer bleeds into downstream phases (`9527eb2`); failed result mappers actually surface `status=failed` (`15ecae7`); reviewed workflows now stop at `critic_plan` and don't silently run IMPLEMENT/VERIFY before the human gate (`15b085e`).
- **`spine.max_completion_tokens` is wired.** The top-level config key was previously parsed but never applied; finite-window local providers ran without an output cap and 400'd at the provider once the prompt approached the model window (trace `019e6e53`). Now both model builders fall back to it when no per-provider value is set.

---

## 1. Token accounting & enforcement

The budget enforcer has been through two cycles. The current incarnation is wired to actual usage_metadata and routes to `needs_review` instead of triggering phase-level retries.

- `c69ef0e` — original token-budget system: per-work-type budgets (quick=200K, spec=500K, critical=1M), `SPINE_TOKEN_BUDGET` env override, `MaxTokenBudgetExceeded` exception, plus enabling `ToolOutputTrimmer` and `add_summarization` for TASKS.
- `65683fe` — **removed** the v1 enforcer because the exception was caught as retryable, causing entire phases to re-execute. Worth re-reading before touching the new version.
- `cb4f023` — v2 enforcer: per-`work_id` cumulative tracker in `retry.py`, subgraph wrapper routes `MaxTokenBudgetExceeded` to `needs_review`, dispatcher resets the counter at workflow launch. Crucially, **defaults `stream_usage=True`** on both OpenRouter and local model builders — without this the v1 enforcer was flying blind because no `usage_metadata` chunks were arriving.
- *(uncommitted, 2026-05-28)* — wires `spine.max_completion_tokens` through to both model builders as a global fallback when `providers.llm[].max_completion_tokens` is unset. Previously the top-level key in `config.yaml` was parsed but never used; finite-window local providers got no output cap, so cumulative prompt + remaining-window budget would 400 at the provider once the prompt approached the model window (trace `019e6e53`: 5× `BadRequestError("80000 token context exceeded, 0 output tokens requested")`). Adds `SPINE_MAX_COMPLETION_TOKENS` env override.
- `f3e0974` — pre-overhaul "RLM optimization" pass (~40–50% token savings on simple tasks). Most of this is superseded by the eval-interpreter removal, but the conditional-prompts and artifact-truncation pieces still apply.

## 2. Caching & deduplication

Three caches now compose: prefix cache for static system prompts, dedupe cache for read tools, and a shared symbol cache for MCP fan-outs.

- `f51d448` — initial dedupe cache, stored on per-invocation `SpineContext`. Superseded; the researcher path never received a context, making this a no-op.
- `bd152a6` — promotes `read_cache` to a checkpointed state field with a merge reducer; survives parallel `Send()`, phase transitions, and critic-driven rework. `trace_audit.py` gains universal duplicate detection across MCP/SPINE/filesystem tool families.
- `e7e6529` — adds `StaticPrefixCacheMiddleware` for Anthropic models; updates MCP allowlist.
- `4e284ac` — symbol-fetch cache shared across parallel explore branches; caps concurrent explores at 4; deferred topics get re-proposed by DA's compaction line.

## 3. RAG / context discipline upgrade

The arc that replaces whole-file `raw_code` embedding with symbol-level slicing and on-demand summaries. This is the single largest token-reduction lever in the window.

- **Two-Pass Hybrid RAG pipeline** for SPECIFY: `48fe309` → `7cfc5fa` (`spine index` CLI) → `70a5258` (vector-store config) → `097c6d2` (provider-based embeddings) → `b14df95`, `1ca4188`, `62297a7`, `9d723ce`, `93dd411` (MCP plumbing fixes) → `c0c2d01` (local AST parsing replaces per-file MCP calls) → `13f0328` (shared embedding client + vLLM overload retry) → `e9f6769` (embed LLM summary, not raw code) → `719681e`, `a0fb4a1` (Qwen3-Embedding-8B at dim 4096).
- `34a0dbc` — phase 1 of the context-discipline upgrade: language-agnostic tree-sitter AST extractor (Python, PHP, TypeScript) replaces whole-file raw_code in vector indexing.
- `f7cf960` — phase 2: `read_edit_lint` + `ast_extract_symbol` bound to subagents.
- `6277bdf` — phase 3: SPECIFY `pre_research_gate` short-circuits high-confidence runs; PLAN explicitly opts out (always runs the loop).
- `124662f`, `451fb3f` — vector-store filter/export and config follow-ups.

## 4. Researcher robustness

Most of the late-window work clusters here. Each fix was driven by a specific trace.

- `f711064` — self-correcting tool loops: hardened `ToolSchemaValidator`, `SymbolCache` improvements. Source trace `019e6833…`.
- `80c9a66` — raise recursion cap and salvage partial findings rather than discarding on cap.
- `ad90798` — salvage from `ToolMessage` contents (capped at 2 KB) rather than the last `AIMessage`, which is often empty when the cap fires.
- `6578e4d`, `90cd8a0` — filter tool-error text from findings at *every* message-extraction point (the three reverse-walks all had the same bug). See `[[feedback_no_error_text_in_research_results]]`.
- `8404924` — concurrency / error-handling improvements in exploration agents; new findings-filter tests; `http_clients.py` introduced.
- `a71c7c9` — stop the research-manager rephrasing prior topics across rounds: `_new_topics` now strips the enriched `— recall symbols: …` suffix before comparing.
- `4c45c46` — seed rework with prior `research_log.json` + critic feedback. Without this, rework restarted from `findings=[]`/`topics=[]` and produced an identical exploration plan — a no-op rework loop. `pre_research_gate` is disabled on rework.
- `89da191` — declare `specification_json` on `ExplorationSubgraphState` so it actually crosses the channel.
- `6faf0f2` — consolidates the researcher's five MCP codebase-index tools into one `CodebaseQueryTool` with an action enum (eliminates malformed-arg failures that exhausted recursion budget); re-fixes `specification_json` drop; routes salvage diagnostics into structured `error_class`/`error_topic` fields; filters dot-folders from MCP results; adds `ResearcherConvergenceMiddleware` (soft nudge / hard stop). Source traces `019e6bde`, `019e6c22`, `019e6c94`, `019e6cc4`, `019e6d27`.
- *(uncommitted, 2026-05-28)* — generalises explore_do salvage to non-`GraphRecursionError` exceptions. `_ainvoke_explore_collecting` now attaches the streamed `partial_state` to any terminal exception that has it (BadRequestError on 80K context overflow, future provider-specific failures); `run_explore_do_node` drops the `isinstance(e, GraphRecursionError)` guard. Source trace `019e6e53` — 2 of 9 explores lost 80K of in-flight investigation when the local provider rejected the prompt. Test: `tests/unit/test_explore_summarise_split.py::test_ainvoke_collecting_attaches_partial_state_on_badrequest_error`.
- *(uncommitted, 2026-05-28)* — `codebase_query` schema error messages now embed a one-line retry example (`pattern='<regex>'` / `name='<identifier>'`) so the local model can self-correct after a malformed call instead of looping on the same wrong action. Source trace `019e6e53` — repeated `action='search'` calls with no `pattern`. The structural fix lives in [[codebase_query.py]] (`_normalise_pattern`/`_normalise_name`).
- *(uncommitted, 2026-05-28)* — three-part hardening so the research_manager stops re-proposing prior topics under different wording:
  1. **Fuzzy near-duplicate dedup** in `_new_topics` / `_topics_near_duplicate` (content-word overlap coefficient ≥ 0.6, stop-word filtered). Catches the local-model paraphrase pattern the exact-string `_normalise_topic` check missed (e.g. "How does the CLI entrypoint parse arguments?" ↔ "How does the command-line interface handle flags?", overlap = 4/6 = 0.67).
  2. **Sentinel topics surfaced to the manager.** `_summarize_findings` previously dropped `error=True` entries entirely, so a round whose explores all sentinelled left the manager reading "(no findings yet)" and re-proposing the same questions. The sentinel now renders as `(attempted; no usable findings — do NOT re-propose this topic)` — neutral marker, no error text (preserves the [[feedback_no_error_text_in_research_results]] rule).
  3. **Per-topic outcome roll-up** replaces the bare `"## Topics Already Explored\n{json.dumps([...])}"` block. Each prior topic now renders inline as `- <topic> — investigated; N file(s) examined` / `attempted; no usable findings` / `proposed; no result recorded`, paired against findings via `_normalise_topic` (matches enriched-suffix topics back to bare wording). The manager sees coverage and outcome in one place instead of having to cross-reference two disconnected sections.
  User-reported symptom: identical topic emitted across rounds 1 and 2 ("How does the CLI parse and handle flags?") on a live run.

## 5. Phase tool surface tightening

Curated, purpose-built tools replace the generic filesystem surface across orchestrator phases. Parallelism moves from interpreter `eval` to graph-layer `Send`.

- `ec1e127` — replace `FilesystemMiddleware` on orchestrators with narrow tools (`ReadSliceFilesTool`, `WriteImplementationReportTool`, `ReadWorkContextTool`, `WriteSpecificationTool`, `ReadPriorArtifactsTool`, `SearchCodebaseTool`, `WritePlanTool`, `WriteTasksArtifactsTool`). Drives behaviour at the tool level, not via prompt instructions the model can ignore. Source traces `019e4447` (6M-token implement + GeneratorExit) and `019e4483` (87 `read_file` calls / 80 min).
- `c7b5349` — parallel subagent dispatch + `SpineInterpreterMiddleware` (now also gone) + stop-running/pending controls.
- `80ea8cf` — MCP tool injection refactor for subagents + trace auditing.
- `034087b` — `SummarizationMiddleware` → `ToolOutputTrimmer` (state-preserving trim, no LLM summarisation cost).
- `9163957` — Send-API dispatch for IMPLEMENT & VERIFY subgraphs with fail-closed routers (raise `CriticalContractFailure` on missing/malformed `execution_waves`; `_research_router` also fail-closed).
- `9299242` — SmallCode IMPLEMENT refactor: restricted-tool dispatch and decompose-on-failure, designed to keep Qwen3-class local models from looping on MCP codebase-index tools.
- `02d2bd2` — **remove** `CodeInterpreterMiddleware` and the `task` PTC dispatcher from phase agents. `deepagents[quickjs]` and `langchain-quickjs` dropped. The eval/`tools.task` escape hatches were bypassing curated tool surfaces and producing brittle code-in-prompt patterns.

## 6. Structured I/O & critic alignment

JSON contracts replace markdown scraping for inter-phase communication, and the critic now sees the right signal.

- `eab6229` — critical contract checks for SPECIFY/PLAN JSON outputs.
- `ec0530f` — structured output for `research_manager` decisions.
- `24eb99c` — structured I/O for IMPLEMENT/VERIFY: `implementation.json` / `verification.json` become authoritative; legacy `slice-*.md` fallback removed; `execution_waves` fail-closed in the artifact gate. Bundles agentic GC: `commit_findings_and_clear_search` tool, boundary-preserving eviction, scratchpad accumulator on `ExplorationSubgraphState`, amnesia warning in the researcher prompt.
- `f8f3108` — OpenRouter native `json_schema` (strict) bound at the model level via `response_format`, replacing DA's tool-call extraction strategy. Fixes HTTP 400 on thinking models (Qwen3, QwQ, DeepSeek-R1) that crashed under forced `tool_choice=any`. Also writes `extra_body.guided_json` for older local vLLM/SGLang engines.
- `d1a5672` — route critic verdict via a dedicated `last_critic_review` field instead of scanning `feedback[-1]`. Stops stale `passed` entries from another tier shadowing a current `NEEDS_REVISION`.
- `cb4f023` (critic portion) — phase-specific traceability/proportionality/scope-creep instructions; original user description inlined into the critic prompt so the spec can be compared against intent.

## 7. Workflow-level guards

- `15b085e` — truncate `WORKFLOW_SEQUENCES` for `reviewed_task` / `critical_reviewed_task` to end at `critic_plan`. Previously both types ran all the way through IMPLEMENT and VERIFY and the dispatcher only relabeled the result `awaiting_approval`. The graph reaching END *is* now the gate. `prereq_gate_*` nodes made conditional on the phase actually being in the sequence.
- `9527eb2` — stop leaking the raw description into PLAN/TASKS-spec/IMPLEMENT/VERIFY/CRITIC; each phase must work from artifacts on disk plus review feedback only.
- `15ecae7` — result mappers set `status=failed` on error so retries see a real signal.
- `0432092` — six fixes from trace `019e6974` (6.67M tokens for "Add --verbose flag"): `PROJECT_ROOT` override stops MCP-index drift to a 1-file stub directory (~360 wasted MCP calls/run); IMPLEMENT synthesizer demotes phase_status to `needs_review` when every slice reports empty `files_modified`/`files_created`; topic_lookup drops test-file hits unless the topic is about tests; research_manager force-decides `done` on critic-rework re-entry when the critic's reason doesn't mention research keywords; `SearchCodebaseTool` resolves `workspace_root` to absolute and runs blocking `_run` on a worker thread.

## 8. What's still open

Lifted from the commit text — these are intentional skips or empirical knobs, not regressions:

- The `critical_reviewed_task` complexity heuristic is **intentionally skipped** (per `0432092`) — that tier is deliberately stress-testing SPECIFY/PLAN/critic gates.
- `ResearcherConvergenceMiddleware` thresholds (soft nudge / hard stop) in `6faf0f2` are still empirical; they'll need trace data to tune.
- `cb4f023` lands a `Round-1-with-findings → done` bias in the manager prompt and dedup based on enriched topic suffixes; both are heuristics worth revisiting after a few traces.
- The salvage path in `ad90798`/`6578e4d`/`90cd8a0` keeps growing message-scanning helpers. Worth auditing for a single canonical extractor next time it's touched.

---

## How to extend this document

When trace audits land new fixes, append rows to the relevant section using the same `hash short-message — one-line impact` shape. Keep the TL;DR honest: if a section grows by more than two entries, bump a bullet up top so the reader sees it on first scroll.
