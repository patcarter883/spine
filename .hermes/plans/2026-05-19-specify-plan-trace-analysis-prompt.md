# Prompt: Analyse a SPINE specify → plan workflow LangSmith trace

Copy this entire block (everything below the line) into a new Hermes session
and replace `<WORK_ID>` with the 8-character work ID from `.spine/`.

The analysis you'll get back follows the same structure used for trace 743e5acb
(see `.hermes/plans/2026-05-19-context-engineering-roadmap.md` for context).

---

I need you to perform a deep analysis of the SPINE LangSmith trace for work
`<WORK_ID>`, a workflow whose work_type is `spec`, `plan_spec`, or a similar
variant that exercises the SPECIFY → PLAN sequence (with optional CRITIC_PLAN
gate in between).

## What to do

Use the LangSmith MCP and SPINE's repo to produce a written assessment covering:

### 1. Trace acquisition and structure
- Find the root LangGraph run in project `spine` for `metadata.work_id` =
  `<WORK_ID>` (try both `spine` and `spine-or` LangSmith projects).
- Capture the full trace via `mcp_langsmith_fetch_runs` with the
  `trace_id` filter. Trace dumps are typically 20-100 MB — write a Python
  script to stream-parse them; do not try to read the whole thing inline.
- Note the trace status (running vs complete) and how many child runs exist.

### 2. Token economics — overall and per phase
For each of these phases, extract from `outputs.generations[].message.kwargs.usage_metadata`:
- `specify`
- `plan`
- `critic_specify` (if `critical_spec` work_type)
- `critic_plan` (if present)
- Any researcher subagents dispatched by `specify` or `plan`

Report:
- Total LLM calls, total prompt tokens, total completion tokens
- Per-phase averages, min/max, median prompt size
- Cache hit rate (`input_token_details.cache_read` / `input_tokens`)
- Prompt-to-completion ratio per phase
- Prompt growth curve within each phase (first call → mid call → last call)

Compare to these reference points (from the IMPLEMENT failure trace 743e5acb):

| Metric | 743e5acb implement | Healthy target |
|--------|:------:|:------:|
| Avg prompt/call | 57K | <20K |
| Max prompt/call | 84K | <30K |
| P:C ratio | 116:1 | <30:1 |
| Cache hit | 65.9% | >60% |

### 3. Tool usage assessment
Count every tool call by name. Pay particular attention to:
- `read_file` calls — are any files read more than 3 times? (Use full
  file paths; absolute vs relative paths count as duplicates.)
- `task` (subagent dispatch) — how many researchers were spawned?
  did any return empty / minimal results?
- `eval` (RLM interpreter) — was it used at all? Did it work? Any
  syntax errors (e.g. `redeclaration of`)?
- `write_file` — were the expected artifacts written?
  - SPECIFY phase expects: `specification.md` under `.spine/artifacts/<WORK_ID>/specify/`
  - PLAN phase expects: `plan.md` under `.spine/artifacts/<WORK_ID>/plan/`
- `write_todos` — productive use, or DA boilerplate noise?

### 4. RLM (eval) effectiveness
Same questions as the trace-743e5acb analysis:
- How many `eval` calls? How many succeeded vs errored?
- Were variables redeclared across calls (QuickJS scope bug pattern)?
- Did the agent use `Promise.all`/`Promise.allSettled` for parallel
  subagent dispatch, or sequential conversation tool calls?

### 5. System prompts delivered
Extract the system prompts actually sent to the model for the SPECIFY and
PLAN phases by reading the first LLM call's `inputs.messages`. Save them
to `/tmp/spine_prompts/specify_system_prompt.txt` and
`/tmp/spine_prompts/plan_system_prompt.txt` for reference.

Then break down each prompt by section (Workflow / SPINE_BASE_PROMPT /
write_todos / Skills / Filesystem / Interpreter / task / AGENTS.md memory /
memory_guidelines). Quote sizes in chars + estimated tokens. Identify
sections that are wasteful for this phase (e.g. AGENTS.md memory injection,
unused tool API dumps, DA boilerplate).

### 6. Researcher subagent quality
The `tasks` phase researcher was the worst performer in trace 743e5acb
(one researcher's `summary` was literally "I'll search broadly"). Check
whether SPECIFY's researchers behaved better:
- Did each researcher read ≥2 files before producing output?
- Was the `file_map` populated with ≥1 entry?
- Did the parent agent re-dispatch any researcher due to empty results?
- Did the researcher's structured response (`ResearchFindings`) parse?

### 7. CRITIC behaviour (if present)
If `critic_specify` or `critic_plan` nodes ran:
- What verdict did they return (passed / needs_revision / needs_review)?
- How many critic LLM calls per gate?
- Did the critic re-read the artifact under review unnecessarily?
- If `needs_revision`, did the rework loop converge?

### 8. Prompt sufficiency vs failure modes
For each prompt section, ask: did the agent actually follow it?
- "Spend at most N turns" — was it respected?
- "Batch reads ≥3 files per turn" — was it respected?
- "Use eval for orchestration" — was eval actually used?
- "Dispatch researchers in parallel" — were they parallel or sequential?

Identify the top 3 places where the prompt told the agent to do X and
the agent did Y instead.

### 9. Comparison with trace 743e5acb (the implement-phase failure trace)
Map your findings against the implement-phase failure modes:
- Is the same "model ignores eviction metadata" pattern present in SPECIFY/PLAN?
- Is unbounded conversation growth happening here too, or is it confined
  to IMPLEMENT?
- Are researchers in SPECIFY/PLAN higher quality than in TASKS?

### 10. Recommendations
Same shape as the trace-743e5acb assessment:
- 3-5 highest-impact prompt rewrites
- 1-2 architectural changes if warranted
- Anything specific to SPECIFY/PLAN that wouldn't apply to IMPLEMENT/VERIFY
- Explicitly call out which problems are model-behaviour (need profile changes)
  vs prompt-design (need rewrite) vs architecture (need new orchestrator pattern)

## Working notes
- The LangSmith trace dumps are large. Write Python via `execute_code` or
  `write_file` + `terminal` to stream-parse them. Do not paste raw JSON
  into the conversation.
- The token usage extraction pattern that works is:
  `outputs.generations[].message.kwargs.usage_metadata.{input_tokens, output_tokens, input_token_details.cache_read}`.
  Anything else (response_metadata.token_usage, generation_info, etc.)
  is often unpopulated for OpenRouter responses.
- Identify phase from system prompt content: strings like
  `"specification writer"` or `"requirements analyst"` for SPECIFY,
  `"planning specialist"` or `"plan engineer"` for PLAN. If those don't
  match, grep `spine/agents/specify_agent.py` and `spine/agents/plan_agent.py`
  for the exact role identifier.
- If the workflow type is `critical_spec`, you will additionally have
  `critic_specify` AND `critic_plan` runs. Plain `spec` has only `critic_plan`.
- Save your written assessment to
  `.hermes/plans/2026-05-19-trace-<WORK_ID>-specify-plan-assessment.md`
  so it can feed into the SPINE context-engineering roadmap.

## Done condition
You're done when:
1. The full trace has been streamed and analysed.
2. System prompts for SPECIFY and PLAN are saved to `/tmp/spine_prompts/`.
3. The written assessment exists at the path above.
4. Recommendations are categorised by intervention type (profile / prompt / architecture).
5. The assessment is delivered in the chat with a tight TL;DR up front.
