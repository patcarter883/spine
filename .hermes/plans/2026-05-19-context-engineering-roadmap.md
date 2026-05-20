# SPINE Context-Engineering Improvement Roadmap

**Status:** Active, post-trace-743e5acb
**Last updated:** 2026-05-19
**Owner:** pat
**Trace baselines:** `79bc301c` (quick, plan rework), `743e5acb` (quick, phase-start tracking)

---

## Context

Two LangSmith traces of SPINE `quick` workflows on `z-ai/glm-5.1-20260406` showed runaway context growth in the IMPLEMENT phase:

| Metric | 79bc301c | 743e5acb |
|--------|:------:|:------:|
| LLM calls | 94 | 229 |
| Total prompt tokens | 2.73M | 11.31M |
| Avg prompt/call | ~29K | 49,387 |
| Max prompt/call | ~37K | 84,630 |
| P:C ratio | 51:1 | 110:1 |
| Cache hit rate | 43% | 57.4% |
| Files re-read 3+ times | (not measured) | 6 |
| `compose.py` reads (worst case) | — | 29 |

Root cause: GLM-5.1 ignores eviction metadata produced by `ToolOutputTrimmer` and re-reads source files instead of trusting `[read: path (N lines) — def X, ...]` placeholders. Context middleware is correctly implemented; the model behaviour is the constraint.

Full assessment with section-by-section system prompt breakdown is in `2026-05-19-trace-743e5acb-improvements.md` (legacy plan file) — superseded by this roadmap.

---

## Strategy

Three layers, all kept simple:

1. **Architectural** — split orchestration from implementation so the model that ignores eviction metadata can't get into a 200-turn loop in the first place.
2. **Configuration** — tool restrictions, slice-count injection, model-specific harness profiles.
3. **Evaluation** — automated pass^k regression dataset so we stop flying blind on prompt changes.

Each item below is independently mergeable.

---

## Done — PR #1 (current, in branch)

Orchestrator + subagent split for IMPLEMENT and VERIFY phases.

**Files:**
- `spine/agents/factory.py` — `allowed_tools` filter + `_filter_filesystem_tools` helper
- `spine/agents/artifacts.py` — `list_slice_files()` discovery helper
- `spine/agents/implement_agent.py` — rewritten as dispatch-only orchestrator
- `spine/agents/verify_agent.py` — same
- `spine/agents/tasks_agent.py` — trimmed prompt, emphasises codebase-map.md as downstream contract
- `tests/unit/test_context_integration.py` — three assertions updated for refactored prompt helpers

**Behavioural changes:**
- Implement orchestrator tools = `[ls, read_file, glob, grep, write_file]`
- Verify orchestrator tools = `[ls, read_file, glob, grep, write_file]`
- Both orchestrators MUST dispatch one subagent per slice (even 1-slice case for consistency)
- Slice inventory injected at agent build time
- Phase-specific prompts shrunk to ~3.2 KB (orchestrator) and ~2.5 KB (tasks)
- "Tool errors are by design" clarification added — pre-empts GLM's "tool not found → recovery spiral" failure mode (Cursor sandboxing insight)

**Explicitly out of scope (per direction):**
- No `max_iterations` cap — iteration count is a metric, not a guardrail
- No AGENTS.md memory injection change — separate broader question

---

## Next up — PR #2: model-specific harness profile for GLM / OpenRouter default

Source: LangChain "Tuning Deep Agents to Work Well with Different Models" (April 2026).

**Why:** SPINE already registers `HarnessProfile` for `openrouter`, `openai`, `anthropic` — but all three share the same `SPINE_BASE_PROMPT`. GLM-5.1 routes through the OpenRouter profile and gets generic instructions designed for nothing in particular.

**Changes:**
1. Add a GLM-aware (or OpenRouter-default) profile with a `<tool_usage>` block specifically addressing:
   - Eviction metadata trust — *"When you see `[read: X.py — def foo, ...]`, do NOT call read_file on that path again."*
   - Parallel tool calling — *"Before any tool call, decide ALL files and patterns you need; emit them in one response, not one at a time."* (Codex profile pattern from blog post)
   - Tool result reflection — *"After receiving tool results, reflect on their content before the next step."* (Opus profile pattern)
2. Move profile definitions to YAML under `.spine/profiles/` matching the blog post's pattern and config-driven preference.
3. Wire `register_harness_profile()` from YAML at SPINE startup.

**Risk:** Low — `HarnessProfile` is already the plumbing. We're just authoring better content per model.

---

## Next up — PR #3: pass^k regression dataset + automated trace comparison

Source: `awesome-harness-engineering` — *"Backtesting AI Agents"* (drdroid) and *"AgentAssay"* (arxiv 2603.02601).

**Why:** Currently, the only way we know a prompt change worked is to run a work item, manually fetch the LangSmith trace, and inspect 4000+ runs. That doesn't scale beyond 1-2 iterations.

**Changes:**
1. Build a small regression dataset (5-10 work items with known-good outcomes) under `tests/integration/regression_workitems/`.
2. Run new SPINE configs against the dataset with `pass^k` semantics (require all N trials to succeed, not just one).
3. Capture trace fingerprints — LLM call count, total tokens, P:C ratio, duplicate file reads, tool error counts — and diff vs. baseline.
4. CI gate: regression in any of those metrics blocks merge.

**Risk:** Medium — need real LLM calls in CI (cost). Mitigations: run on a private branch only, use cached model with deterministic seed if available.

---

## Backlog — PR #4: symbol-index navigation tool

Source: `awesome-harness-engineering` — *"Token Savior"* (77% active-token reduction) and *"semble"* / *"codebase-memory-mcp"*.

**Why:** Even with the orchestrator restriction, slice-implementer subagents still tend to `read_file` whole modules to find a function. A `find_definition` / `grep_symbol` tool that returns `path:line + 5-line snippet` would let them navigate by pointer.

**Changes:**
1. Add a `find_definition(name)` tool backed by tree-sitter (deps already present transitively).
2. Add `grep_symbol(name, kind)` that returns `path:line + ±2 lines context`.
3. Add to `slice-implementer` and `slice-verifier` tool allowlists.
4. Update tasks-phase prompt to suggest these tools to subagents in codebase-map.md "How to navigate" section.

**Risk:** Low-medium — additive, easy to roll back. Tree-sitter parser availability varies by language; start with Python only.

---

## Backlog — PR #5: structured trace analysis tooling

Source: `awesome-harness-engineering` — *"claude-devtools"*, *"AgentStepper"*, *"Polly + langsmith-fetch"* (LangSmith blog).

**Why:** We're currently doing manual `mcp_langsmith_fetch_runs` → giant JSON → Python script analysis. Once PR #3 produces a regular stream of traces to compare, we need tooling.

**Changes:**
1. Build a `spine trace analyze <work_id>` CLI that pulls a LangSmith trace and produces a structured report:
   - Per-phase LLM call count, tokens, P:C ratio
   - Tool call distribution
   - Duplicate file read detection
   - Stall / loop detection
   - Diff against a named baseline trace
2. Save reports as Markdown under `.spine/trace_reports/{work_id}.md` for cross-session sharing.
3. Reuse the analysis logic in PR #3's CI gate.

**Risk:** Low — pure tooling, doesn't touch agent runtime.

---

## Not doing — rejected from `awesome-harness-engineering` survey

For future reference, items considered and explicitly rejected:

| Item | Why rejected |
|------|-------------|
| AutoHarness / Meta-Harness / AutoAgent | Premature — failure modes are deterministic, don't need RL/search |
| Active Context Compression (Focus Agent) | Conflicts with our LangGraph workflow design; existing SummarizationMiddleware suffices |
| AnthropicPromptCachingMiddleware patterns | Irrelevant to GLM-5.1; OpenRouter cache already gives us 57% hit rate |
| OpenViking / Mirage (filesystem-as-context) | Invasive replacement of `LocalShellBackend`; not worth disruption |
| AG-UI / A2A multi-agent protocols | SPINE is single-org, single-process |
| NeMo Guardrails / OPA/Rego | Heavyweight; `allowed_tools` + LangGraph edges cover same need with 10× less code |

---

## How to measure success

After each PR, run a representative `quick` work item and compare against the trace 743e5acb baseline:

| Metric | Baseline | Target after PR #1 | Target after PR #2 | Target after PR #4 |
|--------|:------:|:------:|:------:|:------:|
| Implement LLM calls | 160 | ≤ 20 | ≤ 15 | ≤ 15 |
| Implement avg prompt | 57K | ≤ 25K | ≤ 18K | ≤ 15K |
| Implement max prompt | 84K | ≤ 30K | ≤ 25K | ≤ 22K |
| P:C ratio (whole work) | 110:1 | ≤ 50:1 | ≤ 35:1 | ≤ 25:1 |
| Duplicate reads (≥3×) | 6 files | 0 at orchestrator | 0 anywhere | 0 anywhere |
| Total prompt tokens | 11.3M | ≤ 3M | ≤ 2M | ≤ 1.5M |

Numbers are estimates, not hard contracts — the real signal is whether each PR moves these all in the right direction without regressing solution quality.

---

## Cross-references

- Trace 79bc301c improvements plan: `.hermes/plans/2026-05-19-trace-79bc301c-improvements.md`
- LangChain blog: https://www.langchain.com/blog/tuning-deep-agents-different-models
- Harness engineering survey: https://github.com/ai-boost/awesome-harness-engineering
- SPINE agent architecture overview: `spine/agents/AGENTS.md` (if present) or `AGENTS.md` workspace root
