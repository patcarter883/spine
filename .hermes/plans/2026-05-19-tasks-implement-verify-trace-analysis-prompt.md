# Prompt: Analyse a SPINE tasks → implement → verify workflow LangSmith trace

Copy this entire block (everything below the line) into a new Hermes session
and replace `<WORK_ID>` with the 8-character work ID from `.spine/`.

This is the workflow used by the `quick` / `critical_quick` work types. The
output structure parallels the SPECIFY → PLAN analysis prompt; the diffs are
the phase identifiers, the artifact paths, and the explicit comparison to
trace 743e5acb (which used this exact workflow).

The analysis you'll get back follows the same structure used for trace 743e5acb
(see `.hermes/plans/2026-05-19-context-engineering-roadmap.md` for context).

---

I need you to perform a deep analysis of the SPINE LangSmith trace for work
`<WORK_ID>`, a workflow whose work_type is `quick` or `critical_quick` — i.e.
exercises the TASKS → IMPLEMENT → VERIFY sequence (with optional CRITIC_TASKS
gate after tasks for `critical_quick`). There is NO artifact gate between
implement and verify; verify always runs after implement.

## What to do

Use the LangSmith MCP and SPINE's repo to produce a written assessment covering:

### 1. Trace acquisition and structure
- Find the root LangGraph run in project `spine` for `metadata.work_id` =
  `<WORK_ID>` (try both `spine` and `spine-or` LangSmith projects).
- Capture the full trace via `mcp_langsmith_fetch_runs` with the
  `trace_id` filter. Trace dumps are typically 20-100 MB — write a Python
  script to stream-parse them; do not try to read the whole thing inline.
- Note the trace status (running vs complete), how many child runs exist,
  and how many LangGraph subgraph runs (one per phase).

### 2. Token economics — overall and per phase
For each of these phases, extract from `outputs.generations[].message.kwargs.usage_metadata`:
- `tasks`
- `implement`
- `verify`
- `critic_tasks` (only present for `critical_quick`)
- Any researcher / slice-implementer / slice-verifier subagents

Report:
- Total LLM calls, total prompt tokens, total completion tokens
- Per-phase averages, min/max, median prompt size
- Cache hit rate (`input_token_details.cache_read` / `input_tokens`)
- Prompt-to-completion ratio per phase
- Prompt growth curve within each phase (first call → mid call → last call)

Compare to these reference points from prior traces of this exact workflow:

| Metric | 79bc301c (baseline) | 743e5acb (worse) | Healthy target |
|--------|:------:|:------:|:------:|
| Total LLM calls | 94 | 229 | <80 |
| Avg implement prompt/call | ~29K | 57K | <20K |
| Max implement prompt/call | ~37K | 84K | <30K |
| Whole-work P:C ratio | 51:1 | 110:1 | <30:1 |
| Cache hit | 43% | 57.4% | >60% |
| Total prompt tokens | 2.73M | 11.3M | <3M |

### 3. Tool usage assessment
Count every tool call by name. Pay particular attention to:
- `read_file` calls — are any files read more than 3 times? (Use full
  file paths; absolute vs relative paths count as duplicates.)
  - In trace 743e5acb, `compose.py` was read 29 times. Check whether the
    new orchestrator design eliminated this pattern.
- `task` (subagent dispatch) — how many `slice-implementer` and
  `slice-verifier` were spawned? Were they dispatched in parallel via
  `Promise.allSettled` inside one `eval` call, or sequentially via
  separate conversation turns?
- `eval` (RLM interpreter) — was it used at all? Did it work? Any
  syntax errors (e.g. `redeclaration of`)?
- `write_file` — were the expected artifacts written?
  - TASKS expects: `tasks.md`, `codebase-map.md`, and one `slice-*.md`
    per feature under `.spine/artifacts/<WORK_ID>/tasks/`
  - IMPLEMENT expects: `implementation.md` under `.spine/artifacts/<WORK_ID>/implement/`
  - VERIFY expects: `verification.md` under `.spine/artifacts/<WORK_ID>/verify/`
    (first line must be `VERIFIED` / `PASSED` / `FAILED`)
- `edit_file` / `execute` — these MUST NOT be called from the
  IMPLEMENT or VERIFY orchestrators (their tools are filtered). Any
  attempts indicate the orchestrator restriction failed or the model
  ignored the "tool errors by design" clarification.

### 4. Orchestrator vs subagent split assessment (NEW — specific to this workflow)
The IMPLEMENT and VERIFY phases were redesigned as dispatch-only orchestrators
in PR #1 of the context-engineering roadmap. Check whether the design held:
- How many LLM calls did the IMPLEMENT orchestrator itself make (i.e.
  child runs of the `implement` chain that are NOT inside a `task`
  subagent)? Target: ≤ 10.
- How many `slice-implementer` subagents were dispatched? Should equal
  the slice count produced by the tasks phase.
- For each subagent, how many LLM calls did it make? How many tokens?
  Did it complete successfully or return `blocked`/`partial`?
- Was every subagent dispatched in ONE `eval` call via `Promise.allSettled`,
  or were they spread across multiple turns?
- Same questions for VERIFY → `slice-verifier`.

### 5. RLM (eval) effectiveness
- How many `eval` calls per phase? How many succeeded vs errored?
- Were variables redeclared across calls (QuickJS scope bug pattern)?
- Did the agent use `Promise.all` / `Promise.allSettled` for parallel
  subagent dispatch, or sequential conversation tool calls?
- In trace 743e5acb the agent made only 6 eval calls total, all in tasks.
  Has eval usage spread to implement and verify as the redesign intended?

### 6. System prompts delivered
Extract the system prompts actually sent to the model for the TASKS,
IMPLEMENT, and VERIFY phases by reading the first LLM call's
`inputs.messages` for each. Save them to:
- `/tmp/spine_prompts/tasks_system_prompt.txt`
- `/tmp/spine_prompts/implement_system_prompt.txt`
- `/tmp/spine_prompts/verify_system_prompt.txt`

Then break down each prompt by section (Workflow / SPINE_BASE_PROMPT /
write_todos / Skills / Filesystem / Interpreter / task / AGENTS.md memory /
memory_guidelines). Quote sizes in chars + estimated tokens. Compare
to the prior baselines from trace 743e5acb:

| Phase | Trace 743e5acb size | Target after PR #1 |
|-------|:------:|:------:|
| TASKS | 18,944 chars / ~4,750 tokens | ~2,500 chars phase-specific + same shared sections |
| IMPLEMENT | 46,367 chars / ~11,590 tokens | ~3,200 chars phase-specific + same shared sections |
| VERIFY | 46,506 chars / ~11,630 tokens | ~3,200 chars phase-specific + same shared sections |

Confirm whether the "Expected tool errors (BY DESIGN)" section is present
in implement and verify prompts (it should be, after the clarification was
added).

### 7. Researcher subagent quality (TASKS phase)
In trace 743e5acb, all 3 researchers returned near-empty results — one
literally returned `"I'll search broadly"` as its summary. Check whether
SPECIFY's researchers in this trace did better:
- Did each researcher read ≥2 files before producing output?
- Was the `file_map` populated with ≥1 entry?
- Did the parent agent re-dispatch any researcher due to empty results?
- Did the researcher's structured response (`ResearchFindings`) parse?

### 8. Slice-implementer / slice-verifier subagent quality (NEW)
For each `slice-implementer` dispatched by IMPLEMENT:
- Was the task description fully self-contained (slice text + codebase-map
  excerpts + file list)? Or did it just pass the slice name and force the
  subagent to re-explore?
- Did the subagent run tests? Did they pass?
- Did the subagent return the expected structured `SliceResult` shape?

Same for `slice-verifier` subagents from VERIFY:
- Did each one read the actual modified source (not just implementation.md)?
- Did it run tests / linters?
- Did it produce a structured `VerificationResult`?

### 9. CRITIC behaviour (if present)
If `critic_tasks` ran (only for `critical_quick`):
- What verdict did it return (passed / needs_revision / needs_review)?
- How many critic LLM calls?
- Did the critic re-read the tasks artifacts unnecessarily?
- If `needs_revision`, did the rework loop converge?

### 10. Prompt sufficiency vs failure modes
For each prompt section, ask: did the agent actually follow it?
- "Dispatch one slice-implementer per slice" — was it respected, or did
  the orchestrator try to implement inline?
- "Batch reads ≥3 files per turn" — was it respected?
- "Use eval for orchestration" — was eval actually used?
- "Dispatch subagents via Promise.allSettled" — were they parallel or
  sequential?
- "Tool errors are by design" — did the model treat `edit_file` /
  `execute` errors correctly, or did it spiral on recovery attempts?

Identify the top 3 places where the prompt told the agent to do X and
the agent did Y instead.

### 11. Comparison with trace 743e5acb (the failure baseline)
Map your findings against the implement-phase failure modes from the
743e5acb baseline:
- Has the "model ignores eviction metadata" pattern been mitigated by
  the orchestrator/subagent split? (The orchestrator's context should
  stay small because it's not doing the work.)
- Is the test-edit spiral pattern present in any subagent? (In 743e5acb,
  the agent created a test file then iterated 20+ times trying to make
  it pass, eventually `rm`-ing it.)
- Did the agent dispatch a non-existent subagent type (e.g.
  `general-purpose`)? In 743e5acb this happened once; the new prompts
  explicitly list the valid type.

### 12. Recommendations
Same shape as the trace-743e5acb assessment:
- 3-5 highest-impact prompt rewrites
- 1-2 architectural changes if warranted
- Anything specific to TASKS/IMPLEMENT/VERIFY that wouldn't apply to SPECIFY/PLAN
- Explicitly call out which problems are model-behaviour (need profile changes
  → PR #2 in the roadmap) vs prompt-design (need rewrite) vs architecture
  (need new orchestrator pattern)

## Working notes
- The LangSmith trace dumps are large. Write Python via `execute_code` or
  `write_file` + `terminal` to stream-parse them. Do not paste raw JSON
  into the conversation.
- The token usage extraction pattern that works is:
  `outputs.generations[].message.kwargs.usage_metadata.{input_tokens, output_tokens, input_token_details.cache_read}`.
  Anything else (response_metadata.token_usage, generation_info, etc.)
  is often unpopulated for OpenRouter responses.
- Identify phase from system prompt content:
  - TASKS: `"task decomposition specialist"`
  - IMPLEMENT orchestrator (new): `"IMPLEMENT phase orchestrator"`
  - IMPLEMENT (pre-PR-#1): `"implementation engineer"`
  - VERIFY orchestrator (new): `"VERIFY phase orchestrator"`
  - VERIFY (pre-PR-#1): `"verification engineer"`
  - Researcher: `"codebase researcher"`
  - Slice-implementer: `"code implementer"` / `"YOU MUST USE TOOLS. Do not describe changes"`
  - Slice-verifier: `"verification engineer"` / `"YOU MUST USE TOOLS. Do not verify from memory"`

  If the strings don't match, grep `spine/agents/{tasks,implement,verify}_agent.py`
  and `spine/agents/subagents.py` for the exact role identifier.
- If the workflow type is `critical_quick`, you will additionally have a
  `critic_tasks` run. Plain `quick` skips it.
- Note that subagent runs appear as children of the parent phase chain
  with their own LLM calls — when counting "orchestrator LLM calls" be
  careful not to include subagent calls in the count.
- Save your written assessment to
  `.hermes/plans/2026-05-19-trace-<WORK_ID>-tasks-implement-verify-assessment.md`
  so it can feed into the SPINE context-engineering roadmap.

## Done condition
You're done when:
1. The full trace has been streamed and analysed.
2. System prompts for TASKS, IMPLEMENT, and VERIFY are saved to `/tmp/spine_prompts/`.
3. The written assessment exists at the path above.
4. Recommendations are categorised by intervention type (profile / prompt / architecture).
5. The assessment is delivered in the chat with a tight TL;DR up front
   that explicitly says whether PR #1 (orchestrator/subagent split) achieved
   its stated metrics from the roadmap.
