# How SPINE Learned to Run on Small Models — A Development Story

> *A narrative companion to the [README](README.md). Where the README states the goals
> and the [engineering log](.spine/reviews/token-behavior-progress.md) lists the commits,
> this document tells the story between them: how a two-week run of trace audits
> (`2026-05-14 → 2026-05-29`, ~86 commits) turned two ambitions into something that
> actually runs.*

---

## The two promises in the README

SPINE opens with two claims that are easy to write and hard to earn:

1. **Self-development feasibility** — that LLMs can meaningfully build and modify the
   very harness they run inside.
2. **Local-first inference** — that a serious agent workflow can run on ~30B-class
   models served locally, not just frontier APIs.

These two goals are not independent. They are the same problem seen from two angles.
A harness that can run on a modest local model is a harness whose every prompt is cheap
enough, bounded enough, and legible enough that a 30B model doesn't lose the thread —
and a harness legible enough for a small model to operate is, not coincidentally, a
harness an AI agent can reason about and rebuild. The work in the engineering log is the
record of paying for both promises at once, one trace at a time.

What follows is how that happened.

---

## Where it started: a 226-to-1 problem

Early in the window, a single SPECIFY run could push **13.9 million prompt tokens**
through the model to produce **62 thousand** tokens of actual output. That is a
prompt-to-completion ratio of **226 : 1** (audit #1, trace `019e7164`). The model spent
almost all of its effort *re-reading* and almost none *deciding*.

For a frontier API this is merely expensive. For the local-first goal it is fatal: a
30B-class model served on local hardware has a finite context window and finite patience.
Feed it a 13.9M-token workload and it doesn't get slow — it falls over. One trace
(`019e6e53`) showed exactly this failure five times in a row:
`BadRequestError("80000 token context exceeded, 0 output tokens requested")`. The local
provider wasn't refusing to think; the prompt had simply grown past the window before a
single token of output was requested.

So the central engineering story of this window is a single number being driven down:
**226 : 1 → 15.3 : 1**, prompt volume **21× smaller**, with one SPECIFY run dropping from
13.9M prompt tokens to roughly 668K. The rest of this document is how, and why each step
mattered to the two promises.

---

## Act I — Finding the leak (the researcher rebuild)

The first instinct was wrong, and the log is honest about it. Audit #2 (trace `019e71b4`)
blamed per-cycle *context duplication* — the idea that the same data was being re-injected
into each prompt. The fix for that (a flag to stop re-injecting the MCP tool catalog)
helped, but the ratio barely moved: 228 : 1. The leak was somewhere else.

The breakthrough came from a question, not a metric. The user asked, in effect, *"is the
worker actually one-and-done?"* — and the answer, found by reading trace `019e71b4`
carefully, was no. The researcher's worker was built on a standard agent loop
(`agent.ainvoke()`), and that loop was quietly auto-cycling model → tool → model → tool,
three to five rounds per invocation. Each round re-sent the full conversation *plus* the
accumulating tool output, so a worker that should have made one tool call instead grew its
own prompt from 4K to 24K tokens before finishing. The prompt that said "make ONE tool
call" was a polite suggestion the local model felt free to ignore.

The fix (`7296969`) was structural rather than persuasive — and that distinction becomes
the recurring theme of this whole story. Instead of *asking* the worker to be
one-and-done, the worker was rebuilt to *be incapable of anything else*: bypass the agent
loop entirely, call `model.bind_tools().ainvoke()` exactly once, execute the first tool
call by hand, return the result. No loop, no middleware stack, no accumulation.

The effect was immediate and enormous. Per-worker input collapsed from ~22K to ~10K, and
the total trace prompt fell from **15.6M to 668K — a 23× reduction** in one commit (audit
#3, trace `019e721d`, P:C 20.7 : 1).

Around the same single insight, the researcher was reorganised into a **supervisor↔worker
micro-loop** (`3f1741d`): a no-tool supervisor LLM decides, each cycle, whether the
investigation has converged and which *class* of tool the next step needs
(SEARCH / FIND_SYMBOL / READ_SOURCE / TRACE_DEPS); the worker executes exactly that one
thing. Convergence stopped being a soft middleware nudge and became a hard structural
signal the supervisor owns.

**Why this served both goals.** A 23× smaller prompt is the difference between a workload
a local 30B model can hold and one it 400s on. And a researcher decomposed into a clean
"decide / do one thing" rhythm is a researcher whose behaviour an *AI* developer can
predict and modify — which is what made the later self-directed work on this very module
possible.

---

## Act II — Driving behaviour at the tool layer, not the prompt

Once the worker rebuild proved the principle, it spread. The recurring lesson —
**you cannot prompt a small model out of a behaviour the tooling permits** — reshaped phase
after phase.

- The generic filesystem surface was replaced with **curated, purpose-built tools**
  (`ec1e127`): `ReadSliceFilesTool`, `WriteSpecificationTool`, `WritePlanTool`, and the
  rest. This was driven by traces where a model, handed a general `read_file`, made *87
  calls over 80 minutes* (`019e4483`) or burned 6M tokens on an IMPLEMENT phase
  (`019e4447`). When the tool can only do the right thing, the model does the right thing.

- The **eval interpreter and the `tools.task` dispatcher were removed entirely**
  (`02d2bd2`). These were escape hatches that let the model write code-in-prompt to
  orchestrate parallelism, bypassing every curated surface. All parallelism moved onto
  LangGraph's `Send` API behind fail-closed routers (`9163957`) — deterministic topology
  the graph enforces, not behaviour the model improvises.

- The **MCP tool catalog stopped being re-injected** into every orchestrator and worker
  (`7a26454`, `390df77`). Each injection cost 3–6KB of schema per call and, worse, gave
  the model too many similar-looking tools to choose between — the PLAN synthesizer was
  caught hallucinating a `mcp_codebase-index_get_function_source` call with a missing
  required argument simply because the catalog was in front of it. Remove the catalog and
  the hallucination cannot happen.

- The slice-implementer's tool surface was **trimmed to exactly what a slice needs**
  (`read_file`, `read_edit_lint`, `execute`) after a trace showed it making 83 LLM calls
  and 1.39M prompt tokens to implement a *one-line flag* — it had used its broad search
  tools to "research half the codebase." The fix reframed the slice as *already specified*
  and removed the tools that enabled the over-research.

**Why this served both goals.** Every one of these is a token reduction *and* a
legibility win. A narrow tool surface is what keeps a 30B model on-rails — and a narrow,
deterministic, fail-closed surface is exactly the kind of system an AI agent can author
new phases against without the whole thing becoming non-deterministic mush.

---

## Act III — Making prompts legible to small models

If Act II was about constraining *actions*, Act III was about clarifying *inputs*.

The single-commit prompt-format overhaul (`9300a28`) rebuilt every LLM-facing prompt in
the codebase on two observations about smaller and MoE models (Qwen / DeepSeek / Llama-3
class):

1. They lose precision when raw data is spliced into instruction prose without structural
   bounds.
2. They attend most strongly to the *start* and *end* of a prompt (the U-shaped attention
   curve), so the layout should be **data first, instruction last**.

Out of this came `spine/agents/prompt_format.py` — a canonical `Tag` enum and
`xml_block` / `xml_blocks` / `hostage_layout` helpers. Every dynamic block (`<objective>`,
`<findings>`, `<latest_finding>`, `<directive>`, `<critic_feedback>`, `<retrieved_code>`,
`<scratchpad>`, …) became explicitly bounded, and the plain-text directive — what the
model must actually *do* — was moved to the absolute tail of every prompt. The log calls
this the "hostage layout": the high-attention end of the prompt is reserved for the
instruction, not the data.

This was deliberately a *structural* change, not a behavioural one — the prose of the role
prompts was preserved verbatim; only the data-shaped chunks gained bounds. But the
structural payoff was large: data became machine-parseable, the directive's position
became invariant, and a fragile test suite full of substring assertions could be replaced
with checks on tag *structure*.

**Why this served both goals.** Legible, bounded, U-curve-respecting prompts are how you
get reliable tool calls and valid structured output out of a small model — the make-or-break
capability for local-first. And a prompt system that is machine-parseable and
position-invariant is one an AI agent can safely generate and rearrange, because the
structure carries the meaning instead of brittle wording.

---

## Act IV — Teaching the harness to recover instead of loop

A small local model makes more mistakes than a frontier one: malformed tool arguments,
empty required fields, the same wrong call repeated verbatim. A serious local-first harness
cannot treat each of these as fatal — it has to absorb them. A large share of the
late-window work was building that resilience, each fix traced to a specific failure.

- **Self-correcting tool errors.** When a tool call failed on a teachable mistake
  (`codebase_query` / `search_codebase` with a bad or missing pattern), the error message
  was rewritten to *embed a one-line worked example* (`pattern='<regex>'` /
  `name='<identifier>'`), and the worker was given a single bounded retry: feed the
  teaching message back as a `ToolMessage(status="error")` and re-invoke once so the model
  fixes its own arguments. Bounded to one extra call so the one-and-done token economy
  holds. This came straight from traces where a local model re-emitted the identical
  malformed `action='search'` call four times in a row.

- **Salvaging partial work.** When an investigation hit the recursion cap or the provider
  rejected an over-long prompt mid-flight, the in-progress findings used to be thrown away.
  Now the streamed partial state is attached to *any* terminal exception that carries it
  (`80c9a66`, `ad90798`, and the generalisation in the uncommitted `019e6e53` fix), so a
  worker that 80K-overflows still surrenders what it learned instead of losing 80K of
  in-flight investigation.

- **Never leaking error text into findings.** A subtle, repeated bug: tool-error strings
  were bleeding into the research evidence at *every* message-extraction point (three
  separate reverse-walks all had it — `6578e4d`, `90cd8a0`). Filtered out everywhere,
  because a small model fed its own error text as "evidence" compounds the mistake.

- **Breaking the topic-rephrasing loop.** Local models love to re-propose a question they
  already answered, just worded differently. Exact-string dedup missed it; the fix added
  *fuzzy* near-duplicate detection (content-word overlap ≥ 0.6) plus a per-topic outcome
  roll-up so the research manager sees, inline, that a topic was already
  "investigated — N files examined" or "attempted; no usable findings — do NOT re-propose."

**Why this served both goals.** This is the unglamorous machinery that lets a fallible
30B model finish a job a frontier model would breeze through — turning its mistakes into
self-corrections instead of infinite loops. And every recovery path is itself a piece of
the harness the AI had to design, debug from traces, and harden — self-development in
miniature, repeated dozens of times.

---

## Act V — Closing the loop on usage and cost

The earliest attempt at a token budget enforcer (`c69ef0e`) had to be *removed*
(`65683fe`) because its overrun exception was caught as retryable, so hitting the budget
re-ran entire phases — the opposite of saving tokens. The honest record of that failure is
in the log on purpose.

The v2 enforcer (`cb4f023`) fixed the real reason v1 was unreliable: it had been flying
blind. No `usage_metadata` was actually arriving, so the budget never saw the spend. The
v2 commit made **`stream_usage=True` the default** on both the OpenRouter and the local
model builders, switched to a per-`work_id` cumulative tracker, and routed overruns to
`needs_review` instead of triggering retries.

Two more pieces closed the loop:

- **`spine.max_completion_tokens` was finally wired through** to both model builders as a
  global fallback. The top-level config key had been *parsed but never applied*, so
  finite-window local providers ran with no output cap and 400'd at the provider the moment
  the prompt approached the model window (the `019e6e53` failure again). Now there's always
  a cap.

- **Prefix caching went from invisible to measured.** For most of the window, cache-read
  sat at a flat 0% on the local provider — confirmed across four audit traces — because the
  backing server simply wasn't reporting `prompt_token_details.cached_tokens`. By trace
  `019e72f5` the backend exposed it, and the same calls suddenly reported **~65% cache-read
  (223,712 of 342,672 prompt tokens)**. The caching machinery (`bd152a6` checkpointed
  read-cache, `e7e6529` Anthropic prefix caching, `4e284ac` shared symbol cache across
  parallel branches) had been working the whole time; only the *measurement* had been
  missing.

**Why this served both goals.** A budget wired to real usage and a hard completion cap are
what keep a local model inside its window instead of 400-ing — the literal precondition for
local-first. And a system that *measures its own* token economy, surfaces overruns as
review signals, and reports its cache hit rate is a system that gives an AI developer the
feedback loop needed to keep improving it.

---

## The arc, in one table

The numbers tell the story compactly. Three commits, four audits, over two weeks:

| Audit | Trace | What changed | Prompt | P:C | Status |
|---|---|---|---:|---:|---|
| #1 baseline | `019e7164` | (before) | 13.9 M | 226 : 1 | stalled |
| #2 | `019e71b4` | skip MCP re-injection | 15.6 M | 228 : 1 | misdiagnosed |
| #3 | `019e721d` | **worker direct-bind** | 668 K | 20.7 : 1 | success |
| #4 | `019e723c` | orchestrator MCP skip | 667 K | 15.3 : 1 | **healthy** |

The dominant lever was a single structural insight — make the worker one-and-done — found
by reading a trace because a human asked the right question. Everything else cleaned up the
rest.

---

## What the story actually proves

By the end of the window the picture had inverted. A `task`-tier run that once stalled at
226 : 1 was reaching IMPLEMENT (trace `019e784c`), running on a local 30B-class model,
inside its context window, with ~65% prefix-cache reuse and a budget enforcer watching real
usage. The critic stopped death-spiralling on hallucinated scope creep; autonomous runs
stopped stranding on human interrupts they could never resume.

That is the local-first promise, earned: a serious, multi-phase agent workflow that runs
on modest local hardware — not because the model got smarter, but because the harness
around it got legible, bounded, and recoverable enough that a small model could succeed
inside it.

And the way it was earned is the self-development promise, demonstrated: nearly every fix
in this story was diagnosed from a LangSmith trace, framed as a structural change rather
than a prompt plea, and increasingly authored *through SPINE itself*. The repository is
both the product and the proof — the engineering log behind this story is the audit trail
of an AI harness being taught, trace by trace, to run its own kind of work on a model small
enough to live on your own machine.

---

*Sources: [`README.md`](README.md) for the stated goals; the full commit-level record,
trace IDs, and open issues live in
[`.spine/reviews/token-behavior-progress.md`](.spine/reviews/token-behavior-progress.md).*
