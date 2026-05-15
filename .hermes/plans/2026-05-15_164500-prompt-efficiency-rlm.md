# Prompt Efficiency & RLM Agent Optimization — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce the 84:1 prompt-to-completion ratio and 116 LLM calls per "quick" work item by optimizing prompts, enforcing RLM patterns, fixing subagent delegation, and adding context management middleware.

**Architecture:** Five-pronged approach: (1) Compress system prompts and eliminate redundancy between profile/skill/preamble, (2) Make subagents actually autonomous so the main agent loop doesn't do all the work, (3) Add aggressive context editing middleware with custom state-preserving summarization that evicts stale tool outputs, (4) Add explicit "batch-first" prompting to reduce agent loop iterations from 53 to ~10-15, (5) Treat context as L1 cache with offloaded history as swap — agents can page back via read_file when summarization strips granular detail.

**Tech Stack:** Python 3.12+, deepagents≥0.5.9, langchain-quickjs

---

## Problem Analysis (from trace 4d78e159)

| Metric | Current | Target |
|--------|---------|--------|
| Prompt:Completion ratio | 84:1 | <20:1 |
| Total LLM calls (quick) | 116 | ~30-40 |
| Agent loop iterations (implement) | 53 | ~10-15 |
| File re-reads | 55 (31 redundant) | <20 |
| Subagent tool calls | 0 (all single-turn) | Multiple per subagent |
| Avg prompt/call | 34K tokens | <15K tokens |
| Wall time (quick) | 21 min | <10 min |

## Root Causes

1. **Prompt bloat**: SPINE_BASE_PROMPT (~1.5K tokens) + _RLM_PREAMBLE (~500 tokens) + rlm-pattern skill frontmatter + filesystem middleware prompt + summarization middleware prompt + skills listing + todo middleware = massive system prompt on every call.

2. **Subagents are trivial**: Every `task` call returns a single LLM response with zero tool use. The subagent gets tools but the model never calls them — the system prompt doesn't enforce autonomous tool use, and the `response_format` forces structured output which the model produces in one shot without exploring.

3. **No context editing**: The DA SummarizationMiddleware only kicks in at 85% context. Between 0-85%, every tool result stays in full. Old `read_file` results from iteration 1 are still there at iteration 50, bloating every subsequent LLM call.

4. **One-at-a-time tool use**: The agent reads one file, reasons, reads another, reasons — instead of batching reads. No explicit prompting to parallelize.

5. **RLM preamble is ignored**: The _RLM_PREAMBLE and rlm-pattern skill tell the agent to use eval, but the model doesn't actually do it. The trace shows 0 eval calls in the implement phase despite the preamble and skill being loaded.

---

## Plan

### Task 1: Compress SPINE_BASE_PROMPT — remove redundancy with DA middleware prompts

**Objective:** Cut ~600 tokens from the always-injected system prompt by removing content that DA middleware already injects (tool descriptions, filesystem instructions, subagent descriptions).

**Files:**
- Modify: `spine/agents/profile.py:39-104`

**Step 1: Audit what DA middleware already injects**

The following content is duplicated between SPINE_BASE_PROMPT and DA middleware:
- "Filesystem: read_file, write_file, edit_file, ls, glob, grep" → FilesystemMiddleware already injects a full section
- "Execute: run shell commands" → FilesystemMiddleware already injects this when backend supports it
- "Task: delegate to subagents" → SubAgentMiddleware already injects a full `task` tool section
- "Eval (when enabled): a QuickJS interpreter" → CodeInterpreterMiddleware already injects this

**Step 2: Rewrite SPINE_BASE_PROMPT**

Replace the current `## Tools` section (lines 55-74) with a brief cross-reference that doesn't duplicate the middleware injections:

```python
SPINE_BASE_PROMPT = """\
You are a phase executor inside SPINE, a deterministic AI agent harness. You are \
NOT a conversational assistant — there is no user in the loop during \
phase execution. You receive phase-specific context and must produce a \
structured artifact for the next phase.

## Core Behaviour

- Act, don't narrate. Never say "I'll now do X" — just do it.
- Work until the phase objective is fully met. Do not yield early with a \
summary of what you would do.
- If something fails repeatedly, stop and analyze *why* before retrying. \
Don't pound the same broken approach.
- Your first attempt is rarely correct — iterate.
- Be concise in reasoning. Reserve verbosity for the final artifact.
- **Batch independent operations.** When you need to read ≥2 files or run \
≥2 searches, make all calls in one response instead of sequentially. Every \
round-trip costs tokens — minimize them.
- **Use the interpreter (eval) for orchestration.** When processing ≥3 files \
or dispatching ≥2 subagents, write a JS program in eval that reads files, \
dispatches work, and returns only the synthesis. Intermediate data stays \
out of context.

## Tools

Tool descriptions are provided by the runtime. Key principles:

- **Read before write.** Never speculate about file contents.
- **Test after write.** Never assume your code works.
- **Use task subagents** for parallel work (research, implementation, \
verification slices). Don't re-do their work in the main loop.
- **Use eval** to orchestrate subagents and process data without bloating \
conversation context.

## Workflow Context

- You are running inside a phase of a larger workflow (SPECIFY → PLAN → \
TASKS → IMPLEMENT → VERIFY, with a CRITIC gate between phases).
- Your output will be reviewed by the critic and may be sent back for \
revision, or forwarded to the next phase.
- Do NOT ask follow-up questions — work with the context you are given.
- Do NOT seek user approval — execute autonomously within your phase scope.

## Output

- Produce the artifact your phase requires (specification, plan, slice \
definitions, implementation, verification report).
- Structure your output clearly with headers so downstream phases can \
parse it.
- End with a clear status indicator when the phase artifact is complete.
"""
```

**Step 3: Remove _RLM_PREAMBLE from factory.py**

The RLM guidance is now integrated into SPINE_BASE_PROMPT (the batch and eval principles). The detailed "5 core rules" preamble was ~500 tokens of content that the model ignored anyway (trace shows 0 eval calls). Move the detailed guidance to the rlm-pattern skill only — it's already there and uses progressive disclosure.

Remove `_RLM_PREAMBLE` constant and its append in `build_phase_agent()`:
```python
# Delete these lines from factory.py:
_RLM_PREAMBLE = (...)  # lines 52-77

# Delete from build_phase_agent():
if has_interpreter:
    system_prompt += _RLM_PREAMBLE  # lines 173-174
```

**Step 4: Run existing tests**

```bash
pytest tests/unit/ -k "profile or factory" -v
```

Expected: All pass. The prompt change is cosmetic from a test perspective.

**Step 5: Commit**

```bash
git add spine/agents/profile.py spine/agents/factory.py
git commit -m "refactor: compress SPINE_BASE_PROMPT, remove _RLM_PREAMBLE

Removes ~1100 tokens of system prompt that duplicated DA middleware
injections (tool descriptions, filesystem, eval). Core behavioural
guidance (batch ops, use eval) retained concisely. Detailed RLM
instructions remain in the rlm-pattern skill for progressive disclosure."
```

---

### Task 2: Make subagents autonomous — remove response_format, enforce tool use

**Objective:** Fix the #1 waste: 11 subagent calls that are trivial single-turn responses with zero tool use. The `response_format` forces the model to produce structured output immediately without exploring. Remove it and let subagents work autonomously, then extract the structured result from their final message.

**Files:**
- Modify: `spine/agents/subagents.py:60-95, 295-302`
- Modify: `spine/agents/subagents.py:121-165` (system prompts)

**Step 1: Remove response_format from subagent specs**

In `build_subagent_spec()`, stop passing `response_format` to the spec dict. The structured output format was causing the model to produce a JSON response immediately instead of exploring the codebase first.

```python
# Remove this line from build_subagent_spec():
"response_format": SUBAGENT_RESPONSE_MODELS[name],
```

Instead, move the response schema into the system prompt so the model knows what format to produce *after* it has done its work:

```python
# Add to each subagent prompt:
# researcher:
"6. End with a structured report in this format:\n"
"```json\n"
'{"summary": "...", "patterns": [...], "file_map": {...}, "dependencies": [...]}\n'
"```\n"

# slice-implementer:
"8. End with a structured result:\n"
"```json\n"
'{"status": "implemented|partial|blocked", "files_modified": [...], '
'"files_created": [...], "test_results": "...", "issues": [...]}\n'
"```\n"

# slice-verifier:
"6. End with a structured verification result:\n"
"```json\n"
'{"verdict": "VERIFIED|NOT_VERIFIED", "checklist": [...], '
'"gaps": [...], "recommendations": [...]}\n'
"```\n"
```

**Step 2: Strengthen subagent system prompts to enforce tool use**

The current prompts say "Read key files" but don't mandate tool calls. Make them explicit:

```python
SUBAGENT_PROMPTS = {
    "researcher": (
        "You are a codebase researcher. Your job is to investigate the area "
        "of the codebase described in the task and report back with structured "
        "findings.\n\n"
        "YOU MUST USE TOOLS. Do not produce a report from memory or speculation.\n"
        "Minimum: call `ls` to list directories, `read_file` to inspect key files, "
        "and `grep` to search for patterns before writing any output.\n\n"
        "Guidelines:\n"
        "1. Start by listing the relevant directories with `ls`.\n"
        "2. Use `read_file` to read 3-5 key files in a SINGLE response (batch).\n"
        "3. Use `grep` to find patterns, imports, and references.\n"
        "4. Focus on what is relevant to the task — do not explore broadly.\n"
        "5. Report conventions (naming, imports, patterns) you discover.\n"
        "6. Map important file paths with brief descriptions.\n"
        "7. Note any dependencies or external services.\n\n"
        "IMPORTANT: You are read-only. Do not modify any files.\n"
        "Be concise — your output will be consumed by the specification writer.\n\n"
        "End with a structured report:\n"
        "```json\n"
        '{"summary": "...", "patterns": [...], "file_map": {...}, "dependencies": [...]}\n'
        "```\n"
    ),
    "slice-implementer": (
        "You are a code implementer. Your job is to implement the single "
        "feature slice described in the task.\n\n"
        "YOU MUST USE TOOLS. Do not describe changes — make them with write_file "
        "and edit_file, then verify with execute.\n\n"
        "Workflow:\n"
        "1. Read the slice definition and any referenced prior artifacts (batch reads).\n"
        "2. Read the target files you will modify (batch reads).\n"
        "3. Write or edit code using write_file/edit_file.\n"
        "4. Run tests and linters with execute.\n"
        "5. Fix any errors found.\n"
        "6. Repeat steps 3-5 until tests pass or you are blocked.\n\n"
        "Guidelines:\n"
        "- Follow the project's existing coding conventions.\n"
        "- Include type annotations and docstrings.\n"
        "- Handle errors gracefully.\n"
        "- Focus on this slice only — do not modify files outside its scope.\n"
        "- If blocked by a missing dependency, report it rather than implementing it.\n\n"
        "End with a structured result:\n"
        "```json\n"
        '{"status": "implemented|partial|blocked", "files_modified": [...], '
        '"files_created": [...], "test_results": "...", "issues": [...]}\n'
        "```\n"
    ),
    "slice-verifier": (
        "You are a verification engineer. Your job is to verify the single "
        "feature slice described in the task against its acceptance criteria.\n\n"
        "YOU MUST USE TOOLS. Do not verify from memory — inspect actual files "
        "and run actual tests.\n\n"
        "Workflow:\n"
        "1. Read the slice definition and its acceptance criteria.\n"
        "2. Inspect the implemented files with `read_file` (batch reads).\n"
        "3. Run relevant tests with `execute`.\n"
        "4. Check each acceptance criterion individually.\n"
        "5. Produce a verification report.\n\n"
        "IMPORTANT: You are report-only. Do not fix issues you find.\n"
        "If a test fails or a criterion is not met, record it in the "
        "checklist and gaps.\n\n"
        "End with a structured verification result:\n"
        "```json\n"
        '{"verdict": "VERIFIED|NOT_VERIFIED", "checklist": '
        '[{"criterion": "...", "passed": true, "detail": "..."}], '
        '"gaps": [...], "recommendations": [...]}\n'
        "```\n"
    ),
}
```

**Step 3: Update the parent agent's subagent result parsing**

Since we removed `response_format`, the parent agent receives the subagent's final message as plain text instead of structured JSON. The parent agent (main loop) already handles text results fine — the `task` tool returns whatever the subagent produces. No code change needed on the parent side; the subagent's structured JSON at the end of its response is still parseable by the parent's LLM.

However, we should add a helper to extract the JSON from the subagent's text response for the `build_artifact_prompt` and `feedback` paths that currently expect structured data. For now, the LLM in the parent agent will parse the JSON naturally — this is a non-breaking change.

**Step 4: Run tests**

```bash
pytest tests/unit/test_subagents.py -v
pytest tests/unit/ -k "subagent" -v
```

**Step 5: Commit**

```bash
git add spine/agents/subagents.py
git commit -m "fix: make subagents autonomous — remove response_format, enforce tool use

Subagents were producing single-turn responses with zero tool calls
because response_format forced immediate structured output. Now they
explore the codebase first (mandatory tool calls), then produce
structured JSON at the end of their response. Expected: subagents
go from 0 tool calls to 5-10 per invocation, main agent loop
shrinks from 53 to ~10 iterations."
```

---

### Task 3: Add "gather-then-execute" prompting to phase system prompts

**Objective:** Reduce agent loop iterations from 53→~15 by explicitly structuring the agent's workflow into a "gather context" phase followed by an "execute" phase, rather than the current exploratory one-thing-at-a-time pattern.

**Files:**
- Modify: `spine/agents/implement_agent.py:49-78`
- Modify: `spine/agents/verify_agent.py:50-83`
- Modify: `spine/agents/tasks_agent.py:66-136`

**Step 1: Rewrite implement_agent.py system prompt**

Replace the current system prompt with a structured gather→plan→execute→verify workflow:

```python
system_prompt = (
    "You are an implementation engineer. Given feature slices, "
    "generate production-quality code to implement each one.\n\n"
    f"Your workspace root is: {workspace_root}\n\n"
    "## Workflow (follow this order)\n\n"
    "### Phase 1: Gather (1-2 turns)\n"
    "Batch-read ALL relevant files in ONE response:\n"
    "- Read the tasks artifact and all slice files\n"
    "- Read every target source file you will modify\n"
    "- Use grep/glob to find related files\n"
    "Do NOT start writing until you have gathered context.\n\n"
    "### Phase 2: Plan (1 turn, use eval)\n"
    "Use `eval` to:\n"
    "- Parse slice dependencies and sort into waves\n"
    "- Determine which slices can be implemented in parallel\n"
    "- Build an execution plan with file-level changes\n\n"
    "### Phase 3: Execute (2-4 turns)\n"
    "For each wave:\n"
    "- If ≥2 independent slices: dispatch slice-implementer subagents "
    "via `Promise.all(tools.task(...))` from eval\n"
    "- If 1 slice or dependent work: implement directly with "
    "write_file/edit_file (batch related edits in one response)\n"
    "- After each wave, run tests with execute\n\n"
    "### Phase 4: Verify (1-2 turns)\n"
    "- Run the full test suite\n"
    "- Fix any failures\n"
    "- Write implementation.md summary to disk\n\n"
    "## Rules\n"
    "- Batch reads: read ≥3 files per turn, not one at a time\n"
    "- Use eval for orchestration, not conversation\n"
    "- Never re-read a file you already have in context\n"
    "- After 2 failed attempts at the same fix, stop and re-analyze\n\n"
    + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.IMPLEMENT.value, work_id=work_id
    )
)
```

**Step 2: Rewrite verify_agent.py system prompt**

```python
system_prompt = (
    "You are a verification engineer. Review the implementation "
    "against the specification, plan, and feature slices.\n\n"
    f"Your workspace root is: {workspace_root}\n\n"
    "## Workflow (follow this order)\n\n"
    "### Phase 1: Gather (1-2 turns)\n"
    "Batch-read ALL relevant artifacts and source files in ONE response:\n"
    "- Read tasks.md and all slice files\n"
    "- Read the implementation summary\n"
    "- Read the actual source files that were modified\n\n"
    "### Phase 2: Verify (1-2 turns)\n"
    "For ≥2 slices: dispatch slice-verifier subagents via "
    "`Promise.allSettled(tools.task(...))` from eval — one per slice.\n"
    "For 1 slice: verify directly using read_file and execute.\n\n"
    "### Phase 3: Report (1 turn)\n"
    "Synthesize findings into a verification report:\n"
    "- VERIFIED or NOT VERIFIED status\n"
    "- Checklist of each feature slice and its status\n"
    "- Any gaps or issues found\n"
    "- Write verification.md to disk\n\n"
    "## Rules\n"
    "- Batch reads: never read one file at a time\n"
    "- Use eval for parallel subagent dispatch\n"
    "- Inspect actual code, not just the implementation summary\n"
    "- Run tests — do not assume they pass\n\n"
    + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.VERIFY.value, work_id=work_id
    )
)
```

**Step 3: Rewrite tasks_agent.py system prompt**

The tasks agent already has a structured prompt but it's verbose and the RLM strategy section is conditional. Simplify:

```python
system_prompt = (
    "You are a task decomposition specialist. Given a work description, "
    "break it into smaller, executable feature slices.\n\n"
    f"Your workspace root is: {workspace_root}\n\n"
    "## Workflow (follow this order)\n\n"
    "### Phase 1: Explore (1-2 turns)\n"
)
if is_quick:
    system_prompt += (
    "This is a quick workflow — no prior spec or plan exists.\n"
    "Use `eval` + researcher subagents for parallel exploration:\n"
    "- Dispatch 2-3 researcher subagents via `Promise.all(tools.task(...))`\n"
    "- Each researcher investigates one relevant module\n"
    "- Synthesize results in eval code\n\n"
    )
else:
    system_prompt += (
    "Read prior artifacts (spec, plan) from disk — batch read them.\n\n"
    )

system_prompt += (
    "### Phase 2: Decompose (1-2 turns)\n"
    "Write feature slices to disk:\n"
    f"- Write `slice-<name>.md` files to `{tasks_artifact_dir}/`\n"
    f"- Write `tasks.md` summary to `{tasks_artifact_dir}/`\n"
    "- Group by dependency waves (DAG structure)\n\n"
    "## Rules\n"
    "- You MUST call write_file — conversation-only output is lost\n"
    "- Batch reads — read ≥3 files per turn\n"
    "- Spend at most 2-3 turns exploring, then start writing\n"
    "- Each slice: name, description, files to modify, dependencies, "
    "acceptance criteria, complexity\n\n"
    + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.TASKS.value, work_id=work_id
    )
)
```

**Step 4: Run tests**

```bash
pytest tests/unit/ -k "implement_agent or verify_agent or tasks_agent" -v
```

**Step 5: Commit**

```bash
git add spine/agents/implement_agent.py spine/agents/verify_agent.py spine/agents/tasks_agent.py
git commit -m "refactor: structured gather→execute workflow in phase prompts

Replaces open-ended agent prompts with explicit 2-4 phase workflows
that enforce batch reads upfront and structured execution. Key changes:
- Implement: gather(1-2) → plan(1) → execute(2-4) → verify(1-2)
- Verify: gather(1-2) → verify(1-2) → report(1)
- Tasks: explore(1-2) → decompose(1-2)

Target: reduce implement iterations from 53 to ~10-15 by eliminating
exploratory one-at-a-time tool calls."
```

---

### Task 4: Context as L1 Cache — ToolOutputTrimmer, state-preserving summarization, and filesystem paging

**Objective:** Treat the context window as a fast L1 cache with the offloaded conversation history as swap space. Reduce the growing context window that causes 34K average prompt tokens per call. Three mechanisms: (1) ToolOutputTrimmer evicts stale tool results, (2) SummarizationMiddleware with aggressive token-based trigger and custom state-extraction summary prompt, (3) Agent knows it can page back from offloaded history.

**Files:**
- Modify: `spine/agents/factory.py:138-149, 200-236`
- Create: `spine/agents/context_editing.py` (new)
- Modify: `spine/agents/profile.py` (SPINE_BASE_PROMPT — filesystem paging hint)

**Step 1: Check if DA has context editing middleware**

The installed DA v0.5.9 does NOT include `ContextEditingMiddleware` or `ClearToolUsesEdit`. These are mentioned in LangChain docs but not yet in deepagents. We'll implement a lightweight version as a custom middleware.

**Step 2: Create spine/agents/context_editing.py**

This middleware trims old tool outputs from the conversation, keeping only the last N tool results in full. Older results are replaced with a compact summary line. It runs BEFORE the summarization middleware, keeping the active context lean so summarization triggers less often and runs with less overhead.

```python
"""SPINE context editing middleware — trims old tool outputs.

DA's built-in SummarizationMiddleware triggers at a configurable token
threshold (default 80K for SPINE). Between triggers, tool results
accumulate in full. This middleware trims old tool results earlier,
keeping the conversation lean and reducing peak KV cache pressure.

Strategy: When tool result count exceeds `max_full_tool_results`, replace
old tool call results with a compact placeholder. This preserves the
conversation structure (the agent knows it read a file) but removes
the potentially large file content from context.

The offloaded conversation history (written by DA SummarizationMiddleware
to /conversation_history/{thread_id}.md) serves as swap space — the
agent can page back by reading that file if the placeholder strips
out a crucial detail.

This is a DA AgentMiddleware — add it to the middleware list in
build_phase_agent().
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolOutputTrimmer:
    """Trims old tool outputs from the conversation to keep context lean.

    Replaces old tool result content with a compact placeholder when
    the tool result count exceeds the threshold. Only trims tool results
    (ToolMessage), not human or AI messages.

    Design: treats context as L1 cache. Evicted content lives in the
    offloaded conversation history (swap) and can be paged back via
    read_file if needed.
    """

    def __init__(
        self,
        max_full_tool_results: int = 20,
        placeholder: str = "[evicted — use read_file or check conversation history]",
    ) -> None:
        self.max_full_tool_results = max_full_tool_results
        self.placeholder = placeholder

    # DA middleware interface: wrap_model_call
    async def awrap_model_call(self, request, handler):
        """Trim old tool results before each model call."""
        messages = request.messages

        # Count tool results in the message list
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_result_indices.append(i)

        # If within budget, pass through unchanged
        if len(tool_result_indices) <= self.max_full_tool_results:
            return await handler(request)

        # Trim old results — keep the last N in full
        trim_count = len(tool_result_indices) - self.max_full_tool_results
        trimmed_messages = list(messages)
        for idx in tool_result_indices[:trim_count]:
            msg = trimmed_messages[idx]
            # Replace content with placeholder, preserving metadata
            content = self.placeholder
            if hasattr(msg, "content") and isinstance(msg.content, str):
                # Keep first 100 chars as a hint of what was evicted
                hint = msg.content[:100].split("\n")[0]
                if hint and len(hint) > 10:
                    content = f"[evicted: {hint}... — re-read only if essential]"
            try:
                trimmed_messages[idx] = msg.__class__(
                    content=content,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
            except Exception:
                # If reconstruction fails, skip this message
                pass

        return await handler(request.override(messages=trimmed_messages))
```

**Step 3: Add ToolOutputTrimmer to build_phase_agent**

Modify `build_phase_agent()` to include the trimmer for IMPLEMENT and VERIFY phases:

```python
# In build_phase_agent(), after the middleware list is built:
if phase in (PhaseName.IMPLEMENT, PhaseName.VERIFY) and not is_subagent:
    from spine.agents.context_editing import ToolOutputTrimmer
    middleware.append(ToolOutputTrimmer(max_full_tool_results=20))
```

**Step 4: Rewrite _add_summarization_middleware with token-based trigger, state-preserving summary prompt, and wider keep window**

This is the critical change informed by expert advice. Three key improvements over the original plan:

1. **Token-based trigger instead of fraction** — `("tokens", 80000)` instead of `("fraction", 0.65)`. A fraction trigger is model-dependent (0.65 × 128K = 83K on some models). An absolute 80K token trigger with a ~48K buffer is predictable and prevents OOM during summarization (the summarization pass needs to append its summary_prompt + generate the summary within the same context).

2. **Custom state-extraction summary prompt** — DA's default summarization prompt is tuned for chatbots ("Summarize the user's chat history"). For SPINE's phase executors, the summary must preserve: active file paths, unresolved errors, feature slice objectives, and the offloaded history file path. Otherwise the agent wakes up after compaction and doesn't know what it was doing.

3. **Keep window of 20 messages** — `keep=("messages", 20)` instead of 10. An agent mid-edit needs ~6-7 message pairs intact (write_file → execute → error → read_file → edit_file → execute → pass). 10 messages (~5 pairs) cuts it too close; 20 messages (~7 pairs) covers a full edit-test-fix cycle.

```python
# ── Custom summarization prompt for SPINE phase executors ──────────
# The default DA summarization prompt is chatbot-oriented and strips
# technical state. This prompt acts as a strict state-extraction parser
# that preserves the information a phase executor needs to continue.

_SPINE_SUMMARY_PROMPT = """\
You are summarizing the conversation history of an autonomous code agent \
(inside a SPINE workflow phase). The agent is NOT a chatbot — it is a \
phase executor that reads files, writes code, runs tests, and dispatches \
subagents. Your summary MUST preserve the agent's working state so it \
can continue seamlessly after compaction.

PRESERVE these in your summary (in this order):

1. **Active objective**: What is the agent currently working on? Include \
the exact work description and which phase (tasks/implement/verify).

2. **Files currently being modified**: List every absolute file path the \
agent has read or written. Mark which ones have UNCOMMITTED changes.

3. **Unresolved errors**: Any compiler errors, linter failures, or test \
failures the agent has NOT yet fixed. Include the exact error messages.

4. **Feature slice status**: For each slice being implemented/verified, \
note: slice name, status (not started / in progress / done), and any \
blockers.

5. **Subagent results**: Brief summary of any subagent (researcher, \
slice-implementer, slice-verifier) results received.

6. **Offloaded history path**: If the conversation was previously \
compacted, the offloaded history file path is referenced in the summary \
message. Preserve that path so the agent can page back if needed.

STRIP: Narration ("I will now..."), planning chatter, and repeated \
file contents that are available on disk. The agent can re-read files \
from disk — do not include full file contents in the summary.
"""

def _add_summarization_middleware(
    middleware: list[Any],
    model: Any,
    backend: Any,
) -> None:
    """Add DA summarization middleware with SPINE-specific configuration.

    Three key design decisions:

    1. Token-based trigger (80K) instead of fraction-based. Fraction
       triggers are model-dependent and can leave insufficient buffer
       for the summarization pass itself, causing OOM on 128K-context
       models. 80K tokens leaves a ~48K safety buffer.

    2. Custom state-extraction summary prompt. The default DA prompt
       is chatbot-oriented and strips technical state (file paths,
       errors, slice objectives). Our prompt preserves these.

    3. Keep window of 20 messages. The agent needs ~7 message pairs
       intact to continue an edit-test-fix cycle without losing
       immediate context.
    """
    try:
        from deepagents.middleware.summarization import (
            create_summarization_middleware,
            create_summarization_tool_middleware,
        )

        try:
            # Auto-summarization with aggressive token trigger
            auto_mw = create_summarization_middleware(
                model,
                backend,
                trigger=("tokens", 80000),  # absolute, not fraction
                keep=("messages", 20),      # preserve edit-test-fix cycle
                summary_prompt=_SPINE_SUMMARY_PROMPT,
            )
            middleware.append(auto_mw)

            # Manual compact_conversation tool for on-demand use
            tool_mw = create_summarization_tool_middleware(model, backend)
            middleware.append(tool_mw)

            logger.debug(
                "Added summarization middleware "
                "(trigger=80K tokens, keep=20 msgs, custom prompt)"
            )
        except Exception as exc:
            logger.debug(
                "Summarization middleware could not be initialized "
                "(skipping): %s", exc
            )
    except ImportError:
        logger.debug(
            "Summarization middleware not available "
            "(requires deepagents >= 0.5.0)"
        )
```

**Step 5: Add filesystem paging hint to SPINE_BASE_PROMPT**

Add to the `## Core Behaviour` section (this combines with the Task 6 "never re-read" instruction):

```python
"- **Context is L1 cache; conversation history is swap.** If a "
"compaction summary references an offloaded history file at "
"`/conversation_history/{thread_id}.md`, you can read_file that "
"path to page back specific details. Do NOT re-read source files "
"just because they were evicted — cache them in eval instead.\n"
```

**Step 6: Run tests**

```bash
pytest tests/unit/ -k "factory or summarization or context" -v
```

**Step 7: Commit**

```bash
git add spine/agents/context_editing.py spine/agents/factory.py spine/agents/profile.py
git commit -m "feat: context as L1 cache — ToolOutputTrimmer + state-preserving summarization + paging

Three-layer context management:

1. ToolOutputTrimmer: evicts tool results >20 back to compact
   placeholders with content hints. Runs before summarization
   to keep peak KV cache pressure low.

2. SummarizationMiddleware: aggressive 80K token trigger (not
   fraction-based — prevents OOM during summarization pass on
   128K-context models). Custom state-extraction summary prompt
   preserves file paths, errors, slice status, and offloaded
   history path. Keep window of 20 messages (covers full
   edit-test-fix cycle).

3. Filesystem paging: agents know they can read_file the
   offloaded /conversation_history/{thread_id}.md to page back
   specific details stripped by compaction. Prompt instruction
   to cache in eval instead of re-reading source files.

Target: reduce average prompt/call from 34K to <15K tokens by
preventing stale file contents from accumulating in context."
```

---

### Task 5: Update rlm-pattern skill — enforce eval-first strategy with concrete examples

**Objective:** The current rlm-pattern skill is too abstract — the agent reads it but doesn't act on it (0 eval calls in the trace). Rewrite it with concrete, copy-paste-ready eval programs and enforce "eval before manual tool calls."

**Files:**
- Modify: `spine/skills/rlm-pattern/SKILL.md`

**Step 1: Rewrite the skill**

```markdown
---
name: rlm-pattern
description: RLM pattern — use eval to orchestrate work, batch operations, and keep context lean. MUST use eval before making ≥3 manual tool calls.
phase: specify, tasks, implement, verify
---

# RLM Pattern — Eval-First Strategy

**Rule: Before making ≥3 manual tool calls, ask yourself: "Can I write one eval program that does this?"**

The eval tool is a persistent QuickJS interpreter. Variables survive between
turns. Use it to keep intermediate data OUT of the model context.

## When to use eval

| Situation | Use eval? | Why |
|-----------|-----------|-----|
| Reading ≥3 files | YES | Batch reads in one program, return synthesis |
| Dispatching ≥2 subagents | YES | Promise.all for parallel, keep results in JS |
| Sorting/filtering data | YES | Deterministic — no token cost for logic |
| Single file read | NO | Just use read_file directly |
| Writing a file | NO | Use write_file tool directly |

## Pattern 1: Batch file inspection

Instead of reading files one at a time (5 turns × 34K tokens = 170K tokens),
do this:

```js
// Read tasks artifact and extract slice names
const tasks = await tools.read_file({path: '.spine/artifacts/4d78e159/tasks/tasks.md'});
const sliceMatches = tasks.match(/slice-\w+/g);
const uniqueSlices = [...new Set(sliceMatches)];
console.log('Slices:', uniqueSlices.join(', '));
```

Then read only the slice files you need — in one eval call.

## Pattern 2: Parallel subagent dispatch (IMPLEMENT/VERIFY)

```js
const subagent = runtime.context?.active_subagent || 'slice-implementer';
const slices = ['slice-queue-pending-reorder', 'slice-work-detail-reorder', 'slice-tests'];

// Wave sort: all slices have no dependencies → one wave
const results = await Promise.allSettled(
  slices.map(name => tools.task({
    description: `Implement the slice defined in .spine/artifacts/${runtime.context?.work_id}/tasks/${name}.md. Read the slice file first, then implement the changes described.`,
    subagent_type: subagent,
  }))
);

const succeeded = results.filter(r => r.status === 'fulfilled').map(r => r.value);
const failed = results.filter(r => r.status === 'rejected').map(r => r.reason);
console.log(`Done: ${succeeded.length}/${slices.length}, Failed: ${failed.length}`);
JSON.stringify({succeeded, failed}, null, 2);
```

## Pattern 3: Codebase exploration (TASKS/SPECIFY)

```js
const subagent = 'researcher';
const modules = ['spine/work/ralph_worker.py', 'spine/ui_api/api.py', 'spine/ui/_pages/queue.py'];

const reports = await Promise.all(
  modules.map(path => tools.task({
    description: `Research the module at ${path}. Report: 1) Key classes and functions, 2) Imports and dependencies, 3) Patterns and conventions used.`,
    subagent_type: subagent,
  }))
);

// Process reports in eval — don't dump into conversation
const summaries = reports.map(r => r.substring(0, 200));
console.log(summaries.join('\\n\\n'));
```

## Critical rules

1. **Never dump raw data into conversation.** Process in eval, return synthesis.
2. **Use runtime.context.** Access `runtime.context.work_id`, `runtime.context.active_subagent` etc.
3. **Subagent descriptions must be self-contained.** Include file paths, not "read the slice I mentioned earlier."
4. **Keep eval output under 4000 chars.** The runtime truncates at max_result_chars.
5. **Variables persist across turns.** Store intermediate results in `window.results = ...`.
```

**Step 2: Commit**

```bash
git add spine/skills/rlm-pattern/SKILL.md
git commit -m "refactor: rewrite rlm-pattern skill with concrete eval programs

Previous skill was too abstract — agent ignored it (0 eval calls in
traces). New version: copy-paste-ready eval programs for the three
key patterns (batch reads, parallel dispatch, exploration), explicit
'eval before ≥3 manual calls' rule, and concrete subagent description
templates."
```

---

### Task 6: Add "never re-read" heuristic and eval-caching instruction

**Objective:** Eliminate the 31 redundant file re-reads. When the ToolOutputTrimmer evicts old tool results, the agent re-reads the same files. Add prompt-level instruction to use eval as a cache and make eviction placeholders actionable.

**Files:**
- Modify: `spine/agents/profile.py` (SPINE_BASE_PROMPT — already modified in Task 1 and Task 4)
- Modify: `spine/agents/context_editing.py` (created in Task 4)

**Note:** The filesystem paging hint was already added to SPINE_BASE_PROMPT in Task 4 Step 5. This task adds the eval-caching strategy and improves the eviction placeholder format.

**Step 1: Add eval-caching instruction to SPINE_BASE_PROMPT**

Add to the `## Core Behaviour` section (after the batch/eval bullets from Task 1):

```python
"- **Never re-read a file in the same phase.** If context editing evicts "
"a prior read result, recover from eval: "
"`window.files = window.files || {}; window.files['path'] = content;`. "
"Retrieve from eval instead of calling read_file again.\n"
```

**Step 2: Improve ToolOutputTrimmer placeholders with path extraction**

In `context_editing.py`, improve the placeholder to try extracting the file path from the corresponding tool call, so the agent knows exactly which file was evicted:

```python
# In ToolOutputTrimmer.awrap_model_call, improve the hint logic:
if hasattr(msg, "content") and isinstance(msg.content, str):
    hint = msg.content[:100].split("\n")[0]
    if hint and len(hint) > 10:
        content = f"[evicted: {hint}... — recover from eval or re-read only if essential]"
```

**Step 3: Run tests**

```bash
pytest tests/unit/ -k "profile or context_editing" -v
```

**Step 4: Commit**

```bash
git add spine/agents/profile.py spine/agents/context_editing.py
git commit -m "feat: never re-read heuristic + eval caching instruction

Adds prompt instruction to cache file contents in eval variables
instead of re-reading. Improves ToolOutputTrimmer placeholders to
show a hint of the evicted content so the agent knows what was
there without wasting a read_file call."
```

---

### Task 7: Add integration test — verify reduced iterations on a mock quick workflow

**Objective:** Create a test that verifies the prompt changes produce the expected structural improvements (batched reads, eval usage, subagent tool calls) without requiring a live LLM.

**Files:**
- Create: `tests/unit/test_prompt_efficiency.py`

**Step 1: Write the test**

```python
"""Tests for prompt efficiency improvements.

These tests verify the structural properties of the agent configuration
that lead to reduced token usage, not actual LLM behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.profile import SPINE_BASE_PROMPT


class TestPromptEfficiency:
    """Verify prompt changes that reduce token usage."""

    def test_base_prompt_no_tool_duplicates(self):
        """SPINE_BASE_PROMPT should not duplicate DA middleware injections.

        The FilesystemMiddleware, SubAgentMiddleware, and
        CodeInterpreterMiddleware all inject their own tool descriptions.
        Our base prompt should cross-reference, not duplicate.
        """
        # These strings appear in DA middleware prompts — we should NOT
        # have full descriptions of them in our base prompt
        duplicated_phrases = [
            "read_file, write_file, edit_file, ls, glob, grep",
            "run shell commands",
            "QuickJS interpreter",
            "task tool to launch short-lived subagents",
        ]
        for phrase in duplicated_phrases:
            assert phrase not in SPINE_BASE_PROMPT, (
                f"SPINE_BASE_PROMPT duplicates DA middleware content: {phrase!r}"
            )

    def test_base_prompt_has_batch_instruction(self):
        """Base prompt must instruct agents to batch independent operations."""
        assert "batch" in SPINE_BASE_PROMPT.lower() or "Batch" in SPINE_BASE_PROMPT

    def test_base_prompt_has_eval_instruction(self):
        """Base prompt must reference eval/interpreter for orchestration."""
        assert "eval" in SPINE_BASE_PROMPT.lower()

    def test_base_prompt_has_no_re_read_instruction(self):
        """Base prompt must tell agents not to re-read files."""
        assert "re-read" in SPINE_BASE_PROMPT.lower() or "never" in SPINE_BASE_PROMPT.lower()

    def test_base_prompt_under_token_budget(self):
        """Base prompt should be under 800 tokens (rough char estimate)."""
        # At ~4 chars/token, 800 tokens = ~3200 chars
        # Our compressed prompt should be well under this
        assert len(SPINE_BASE_PROMPT) < 3200, (
            f"SPINE_BASE_PROMPT is {len(SPINE_BASE_PROMPT)} chars — "
            f"should be under 3200 chars (~800 tokens)"
        )

    def test_rlm_preamble_removed(self):
        """_RLM_PREAMBLE should no longer exist in factory.py."""
        from spine.agents import factory
        assert not hasattr(factory, "_RLM_PREAMBLE"), (
            "_RLM_PREAMBLE still exists in factory.py — should be removed"
        )


class TestSubagentAutonomy:
    """Verify subagents are configured for autonomous tool use."""

    def test_subagent_no_response_format(self):
        """Subagent specs should NOT include response_format."""
        from spine.agents.subagents import build_subagent_spec

        state: WorkflowState = {
            "work_id": "test",
            "work_type": "quick",
            "description": "test",
            "workspace_root": "/tmp",
            "artifacts": {},
            "critic_reviewing": "",
            "current_phase": "",
            "feedback": [],
            "max_retries": 3,
            "phase_index": 0,
            "prompt_request": None,
            "retry_count": {},
            "status": "running",
        }

        for name in ["researcher", "slice-implementer", "slice-verifier"]:
            spec = build_subagent_spec(
                name=name,
                phase=PhaseName.IMPLEMENT,
                state=state,
            )
            assert "response_format" not in spec, (
                f"Subagent {name!r} still has response_format — "
                f"should be removed for autonomous tool use"
            )

    def test_subagent_prompt_enforces_tools(self):
        """Subagent prompts must contain 'MUST USE TOOLS' instruction."""
        from spine.agents.subagents import SUBAGENT_PROMPTS

        for name, prompt in SUBAGENT_PROMPTS.items():
            assert "MUST USE TOOLS" in prompt or "must use tools" in prompt.lower(), (
                f"Subagent {name!r} prompt doesn't enforce tool use"
            )


class TestContextEditing:
    """Verify context editing middleware is configured."""

    def test_trimmer_class_exists(self):
        """ToolOutputTrimmer should be importable."""
        from spine.agents.context_editing import ToolOutputTrimmer
        trimmer = ToolOutputTrimmer(max_full_tool_results=20)
        assert trimmer.max_full_tool_results == 20

    def test_trimmer_preserves_recent_results(self):
        """Trimmer should not trim results within the budget."""
        from spine.agents.context_editing import ToolOutputTrimmer
        trimmer = ToolOutputTrimmer(max_full_tool_results=5)
        assert trimmer.max_full_tool_results == 5


class TestSummarizationConfig:
    """Verify summarization middleware uses SPINE-specific configuration."""

    def test_custom_summary_prompt_exists(self):
        """_SPINE_SUMMARY_PROMPT should exist and preserve technical state."""
        from spine.agents.factory import _SPINE_SUMMARY_PROMPT
        assert _SPINE_SUMMARY_PROMPT is not None
        # Must preserve file paths
        assert "file" in _SPINE_SUMMARY_PROMPT.lower()
        # Must preserve errors
        assert "error" in _SPINE_SUMMARY_PROMPT.lower()
        # Must preserve slice status
        assert "slice" in _SPINE_SUMMARY_PROMPT.lower()
        # Must reference offloaded history
        assert "offloaded" in _SPINE_SUMMARY_PROMPT.lower() or "history" in _SPINE_SUMMARY_PROMPT.lower()

    def test_summary_prompt_not_chatbot_oriented(self):
        """Summary prompt should NOT use chatbot framing."""
        from spine.agents.factory import _SPINE_SUMMARY_PROMPT
        assert "chat history" not in _SPINE_SUMMARY_PROMPT.lower()
        assert "user's chat" not in _SPINE_SUMMARY_PROMPT.lower()
```

**Step 2: Run the tests**

```bash
pytest tests/unit/test_prompt_efficiency.py -v
```

Note: Some tests will FAIL initially (e.g., `test_subagent_no_response_format` will pass because the change hasn't been made yet, but `test_rlm_preamble_removed` will fail until Task 1 is done). That's expected — run them as you implement each task.

**Step 3: Commit**

```bash
git add tests/unit/test_prompt_efficiency.py
git commit -m "test: add prompt efficiency integration tests

Tests verify structural properties that reduce token usage:
- No duplicated tool descriptions in base prompt
- Batch/eval/no-re-read instructions present
- Token budget compliance (<800 tokens for base prompt)
- _RLM_PREAMBLE removed
- Subagents have no response_format and enforce tool use
- ToolOutputTrimmer is importable and configurable"
```

---

## Summary of Expected Impact

| Change | Token Savings | Iteration Reduction | Mechanism |
|--------|---------------|---------------------|-----------|
| Compressed SPINE_BASE_PROMPT | ~800 tokens/call | — | Removed tool description duplicates |
| Removed _RLM_PREAMBLE | ~500 tokens/call | — | Moved to skill (progressive disclosure) |
| Subagent autonomy | — | 53→~15 (implement) | Subagents do real work instead of trivial responses |
| Gather-then-execute prompting | — | 53→~15 | Structured workflow prevents one-at-a-time exploration |
| ToolOutputTrimmer (L1 cache) | ~15-20K tokens/call | — | Evicts old tool results, keeps recent N |
| Token-based summarization (80K) | ~10-15K tokens/call | — | Predictable trigger, prevents OOM during compaction |
| State-preserving summary prompt | Prevents agent amnesia | — | Preserves file paths, errors, slice status after compaction |
| Filesystem paging (swap) | Enables safe eviction | — | Agent pages back from offloaded history when needed |
| Keep window 20 messages | Preserves edit-test-fix | — | Agent doesn't lose immediate context mid-cycle |
| Never re-read heuristic | ~5K tokens/re-read | 31→~5 re-reads | Prompt + eval caching |

**Combined estimated impact on the traced workload:**

| Metric | Before | After (est.) | Improvement |
|--------|--------|-------------|-------------|
| Total tokens | 4,043,483 | ~800,000-1,200,000 | 70-80% reduction |
| Prompt:Completion ratio | 84:1 | ~15-20:1 | 4-5x improvement |
| LLM calls | 116 | ~30-40 | 65-70% reduction |
| Wall time | 21 min | ~8-12 min | ~50% reduction |
| Subagent tool calls | 0 | 30-50 | Actual work delegation |
| Peak KV cache usage | ~128K tokens (OOM risk) | ~80K tokens (48K buffer) | Prevents OOM |

## Key Design Principles (from expert advice)

1. **Token-based trigger, not fraction-based.** `("tokens", 80000)` is model-independent and leaves a predictable 48K buffer for the summarization pass. Fraction triggers vary by model max_input_tokens and can leave insufficient buffer.

2. **Custom state-extraction summary prompt.** The default DA summarization prompt is chatbot-oriented and strips technical state. SPINE's prompt preserves: active file paths, unresolved errors, feature slice objectives, and the offloaded history path.

3. **Filesystem paging as swap.** DA's SummarizationMiddleware offloads raw evicted messages to `/conversation_history/{thread_id}.md`. The agent can `read_file` this path to page back specific details — treating context as L1 cache and disk as swap.

4. **Keep window of 20 messages.** An agent mid-edit needs ~6-7 message pairs intact (write → test → error → read → fix → test → pass). 10 messages cuts it too close; 20 covers a full cycle.
