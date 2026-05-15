# SPINE Prompt Efficiency & RLM Optimization — Implementation Prompt

You are implementing 8 tasks to dramatically reduce token waste in the SPINE agent system. The baseline trace (4d78e159) showed: 4M tokens, 84:1 prompt:completion ratio, 116 LLM calls, 21 min wall time for a single "quick" work item. Your changes should bring this to ~1M tokens, ~15-20:1 ratio, ~30-40 LLM calls, ~8-12 min.

**Working directory:** `/home/pat/projects/spine`
**Test command:** `pytest tests/unit/ -v` (run after each task)
**Lint command:** `ruff check spine/ tests/ && ruff format spine/ tests/`

Implement tasks IN ORDER. Each task builds on the previous. Commit after each.

---

## Task 1: Compress SPINE_BASE_PROMPT — remove DA middleware duplicates

**File:** `spine/agents/profile.py`

The current `SPINE_BASE_PROMPT` (lines 39-104) duplicates tool descriptions that DA middleware already injects:
- "Filesystem: read_file, write_file, edit_file, ls, glob, grep" → FilesystemMiddleware injects this
- "Execute: run shell commands" → FilesystemMiddleware injects this
- "Task: delegate to subagents..." → SubAgentMiddleware injects this
- "Eval (when enabled): a QuickJS interpreter" → CodeInterpreterMiddleware injects this

Replace the `## Tools` section (lines 55-104) with a concise cross-reference. The new prompt should:

1. Keep `## Core Behaviour` but ADD these two bullets:
   - "**Batch independent operations.** When you need to read ≥2 files or run ≥2 searches, make all calls in one response instead of sequentially."
   - "**Use the interpreter (eval) for orchestration.** When processing ≥3 files or dispatching ≥2 subagents, write a JS program in eval that reads files, dispatches work, and returns only the synthesis."

2. Replace `## Tools` with a brief section that says "Tool descriptions are provided by the runtime" and lists only principles (read before write, test after write, use task subagents for parallel work, use eval to orchestrate).

3. Keep `## Workflow Context` and `## Output` sections but shorten them.

Target: under 3200 chars (~800 tokens).

**Also:** Remove `_RLM_PREAMBLE` from `spine/agents/factory.py`:
- Delete the `_RLM_PREAMBLE` constant (lines 52-77)
- Delete the `if has_interpreter: system_prompt += _RLM_PREAMBLE` block (lines 173-175)
- The RLM guidance is now in SPINE_BASE_PROMPT (batch + eval bullets) and the rlm-pattern skill (progressive disclosure)

**Commit message:** `refactor: compress SPINE_BASE_PROMPT, remove _RLM_PREAMBLE`

---

## Task 2: Make subagents autonomous — remove response_format, enforce tool use

**File:** `spine/agents/subagents.py`

The current subagent specs pass `response_format` (a Pydantic model) which forces the LLM to produce structured JSON in one shot without exploring. This is why all 11 subagent calls in the trace had 0 tool calls.

Changes:

1. **Remove `response_format` from the spec dict** in `build_subagent_spec()` (line 301). Delete:
   ```python
   "response_format": SUBAGENT_RESPONSE_MODELS[name],
   ```

2. **Add the JSON schema as a prompt instruction** to each `SUBAGENT_PROMPTS` entry so the model knows the expected output format AFTER it does its work:

   - **researcher**: Add `"YOU MUST USE TOOLS. Do not produce a report from memory or speculation.\n"` at the start of guidelines. Add at end: `"End with a structured report:\n```json\n{\"summary\": \"...\", \"patterns\": [...], \"file_map\": {...}, \"dependencies\": [...]}\n```\n"`

   - **slice-implementer**: Add `"YOU MUST USE TOOLS. Do not describe changes — make them with write_file and edit_file, then verify with execute.\n"` at the start. Add at end: `"End with a structured result:\n```json\n{\"status\": \"implemented|partial|blocked\", \"files_modified\": [...], \"files_created\": [...], \"test_results\": \"...\", \"issues\": [...]}\n```\n"`

   - **slice-verifier**: Add `"YOU MUST USE TOOLS. Do not verify from memory — inspect actual files and run actual tests.\n"` at the start. Add at end: `"End with a structured verification result:\n```json\n{\"verdict\": \"VERIFIED|NOT_VERIFIED\", \"checklist\": [{\"criterion\": \"...\", \"passed\": true, \"detail\": \"...\"}], \"gaps\": [...], \"recommendations\": [...]}\n```\n"`

3. **Keep the Pydantic model classes** (ResearchFindings, SliceResult, VerificationResult) in the file — they may be useful for validation later. Just stop passing them as `response_format`.

4. **Update subagent tool prompts to emphasize batch reads**: In each prompt, add `"Batch reads: read 3-5 files per turn, not one at a time.\n"` to the guidelines.

**Commit message:** `fix: make subagents autonomous — remove response_format, enforce tool use`

---

## Task 3: Add gather-then-execute workflow to phase system prompts

**Files:** 
- `spine/agents/implement_agent.py`
- `spine/agents/verify_agent.py`
- `spine/agents/tasks_agent.py`

Replace the open-ended system prompts with structured multi-phase workflows. Each prompt must explicitly sequence the agent's turns into gather→plan→execute→verify phases with turn budgets.

**implement_agent.py** — Replace the system prompt with:
```
You are an implementation engineer. Given feature slices, generate production-quality code to implement each one.

Your workspace root is: {workspace_root}

## Workflow (follow this order)

### Phase 1: Gather (1-2 turns)
Batch-read ALL relevant files in ONE response:
- Read the tasks artifact and all slice files
- Read every target source file you will modify
- Read the codebase map (if available): `{tasks_path}/codebase-map.md`
- Use grep/glob to find related files
Do NOT start writing until you have gathered context.

### Phase 2: Plan (1 turn, use eval)
Use `eval` to:
- Parse slice dependencies and sort into waves
- Determine which slices can be implemented in parallel
- Build an execution plan with file-level changes

### Phase 3: Execute (2-4 turns)
For each wave:
- If ≥2 independent slices: dispatch slice-implementer subagents via `Promise.all(tools.task(...))` from eval
- If 1 slice or dependent work: implement directly with write_file/edit_file (batch related edits in one response)
- After each wave, run tests with execute

### Phase 4: Verify (1-2 turns)
- Run the full test suite
- Fix any failures
- Write implementation.md summary to disk

## Rules
- Batch reads: read ≥3 files per turn, not one at a time
- Use eval for orchestration, not conversation
- Never re-read a file you already have in context
- After 2 failed attempts at the same fix, stop and re-analyze
```
Then append `build_artifact_prompt(...)` as before.

**verify_agent.py** — Replace with:
```
You are a verification engineer. Review the implementation against the specification, plan, and feature slices.

Your workspace root is: {workspace_root}

## Workflow (follow this order)

### Phase 1: Gather (1-2 turns)
Batch-read ALL relevant artifacts and source files in ONE response:
- Read tasks.md and all slice files
- Read the codebase map (if available): `{tasks_path}/codebase-map.md`
- Read the implementation summary
- Read the actual source files that were modified

### Phase 2: Verify (1-2 turns)
For ≥2 slices: dispatch slice-verifier subagents via `Promise.allSettled(tools.task(...))` from eval — one per slice.
For 1 slice: verify directly using read_file and execute.

### Phase 3: Report (1 turn)
Synthesize findings into a verification report:
- VERIFIED or NOT VERIFIED status
- Checklist of each feature slice and its status
- Any gaps or issues found
- Write verification.md to disk

## Rules
- Batch reads: never read one file at a time
- Use eval for parallel subagent dispatch
- Inspect actual code, not just the implementation summary
- Run tests — do not assume they pass
```
Then append `build_artifact_prompt(...)` as before.

**tasks_agent.py** — Replace with:
```
You are a task decomposition specialist. Given a work description, break it into smaller, executable feature slices.

Your workspace root is: {workspace_root}

## Workflow (follow this order)

### Phase 1: Explore (1-2 turns)
[if is_quick:]
This is a quick workflow — no prior spec or plan exists.
Use `eval` + researcher subagents for parallel exploration:
- Dispatch 2-3 researcher subagents via `Promise.all(tools.task(...))`
- Each researcher investigates one relevant module
- Synthesize results in eval code

[else:]
Read prior artifacts (spec, plan) from disk — batch read them.

### Phase 2: Decompose (1-2 turns)
Write feature slices to disk:
- Write `slice-<name>.md` files to `{tasks_artifact_dir}/`
- Write `tasks.md` summary to `{tasks_artifact_dir}/`
- Write `codebase-map.md` to `{tasks_artifact_dir}/` (see Task 8)
- Group by dependency waves (DAG structure)

## Rules
- You MUST call write_file — conversation-only output is lost
- Batch reads — read ≥3 files per turn
- Spend at most 2-3 turns exploring, then start writing
- Each slice: name, description, files to modify, dependencies, acceptance criteria, complexity
```
Then append `build_artifact_prompt(...)` as before.

**IMPORTANT:** Read each existing file fully before rewriting. Preserve all imports, helper function calls, and the `build_artifact_prompt()` appendage. Only change the `system_prompt` string construction.

**Commit message:** `refactor: structured gather→execute workflow in phase prompts`

---

## Task 4: Context as L1 Cache — ToolOutputTrimmer + state-preserving summarization + filesystem paging

**Files:**
- Create: `spine/agents/context_editing.py`
- Modify: `spine/agents/factory.py`
- Modify: `spine/agents/profile.py` (add paging hint)

### 4a. Create `spine/agents/context_editing.py`

This is a DA AgentMiddleware that trims old tool outputs from the conversation. When tool result count exceeds `max_full_tool_results` (default 20), old ToolMessage content is replaced with a compact placeholder that includes a hint of the evicted content.

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
        placeholder: str = "[evicted — recover from eval or re-read only if essential]",
    ) -> None:
        self.max_full_tool_results = max_full_tool_results
        self.placeholder = placeholder

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
            content = self.placeholder
            if hasattr(msg, "content") and isinstance(msg.content, str):
                hint = msg.content[:100].split("\n")[0]
                if hint and len(hint) > 10:
                    content = f"[evicted: {hint}... — recover from eval or re-read only if essential]"
            try:
                trimmed_messages[idx] = msg.__class__(
                    content=content,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
            except Exception:
                pass

        return await handler(request.override(messages=trimmed_messages))
```

### 4b. Add ToolOutputTrimmer to `build_phase_agent()` in factory.py

After the middleware list is built and before the agent is constructed, add:

```python
# Context editing: trim old tool results for long-running phases
if phase in (PhaseName.IMPLEMENT, PhaseName.VERIFY) and not is_subagent:
    from spine.agents.context_editing import ToolOutputTrimmer
    middleware.append(ToolOutputTrimmer(max_full_tool_results=20))
```

Insert this AFTER the summarization middleware block (line 149) and BEFORE the memory block (line 152).

### 4c. Rewrite `_add_summarization_middleware()` in factory.py

Replace the current function (lines 201-236) with a version that:
1. Adds BOTH auto-summarization AND tool-based summarization
2. Uses `trigger=("tokens", 80000)` — absolute token count, not fraction
3. Uses `keep=("messages", 20)` — preserves edit-test-fix cycles
4. Uses a custom `_SPINE_SUMMARY_PROMPT` that preserves technical state

Add `_SPINE_SUMMARY_PROMPT` as a module-level constant in factory.py (before `_add_summarization_middleware`):

```python
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
```

Then rewrite `_add_summarization_middleware` to try importing BOTH `create_summarization_middleware` and `create_summarization_tool_middleware`, and configure the auto one with the SPINE-specific params:

```python
def _add_summarization_middleware(
    middleware: list[Any],
    model: Any,
    backend: Any,
) -> None:
    """Add DA summarization middleware with SPINE-specific configuration.

    Three key design decisions:
    1. Token-based trigger (80K) — model-independent, leaves 48K buffer.
    2. Custom state-extraction summary prompt — preserves file paths,
       errors, slice objectives, and offloaded history path.
    3. Keep window of 20 messages — covers full edit-test-fix cycle.
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
                trigger=("tokens", 80000),
                keep=("messages", 20),
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
            # Fallback: try just the tool middleware
            logger.debug(
                "Auto-summarization middleware failed, trying tool-only: %s", exc
            )
            try:
                tool_mw = create_summarization_tool_middleware(model, backend)
                middleware.append(tool_mw)
                logger.debug("Added summarization tool middleware (fallback)")
            except Exception as exc2:
                logger.debug(
                    "Summarization middleware could not be initialized "
                    "(skipping): %s", exc2
                )
    except ImportError:
        logger.debug(
            "Summarization middleware not available "
            "(requires deepagents >= 0.5.0)"
        )
```

### 4d. Add filesystem paging hint to SPINE_BASE_PROMPT

In `spine/agents/profile.py`, add to the `## Core Behaviour` section:

```python
"- **Context is L1 cache; conversation history is swap.** If a "
"compaction summary references an offloaded history file at "
"`/conversation_history/{thread_id}.md`, you can read_file that "
"path to page back specific details. Do NOT re-read source files "
"just because they were evicted — cache them in eval instead.\n"
```

**Commit message:** `feat: context as L1 cache — ToolOutputTrimmer + state-preserving summarization + paging`

---

## Task 5: Rewrite rlm-pattern skill with concrete eval programs

**File:** `spine/skills/rlm-pattern/SKILL.md`

The current skill is too abstract — the agent ignored it (0 eval calls in the trace). Rewrite it with copy-paste-ready eval programs.

Read the existing file first. Then replace its content with:

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
const tasks = await tools.read_file({path: '.spine/artifacts/WORK_ID/tasks/tasks.md'});
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
    description: `Implement the slice defined in .spine/artifacts/${runtime.context?.work_id}/tasks/${name}.md. Read the slice file and codebase-map.md first, then implement the changes described.`,
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
3. **Subagent descriptions must be self-contained.** Include file paths and reference codebase-map.md, not "read the slice I mentioned earlier."
4. **Keep eval output under 4000 chars.** The runtime truncates at max_result_chars.
5. **Variables persist across turns.** Store intermediate results in `window.results = ...`.
```

**Commit message:** `refactor: rewrite rlm-pattern skill with concrete eval programs`

---

## Task 6: Add "never re-read" heuristic and eval-caching instruction

**Files:**
- `spine/agents/profile.py` (SPINE_BASE_PROMPT — already modified in Tasks 1 & 4)
- `spine/agents/context_editing.py` (created in Task 4)

### 6a. Add eval-caching instruction to SPINE_BASE_PROMPT

In the `## Core Behaviour` section, add:

```python
"- **Never re-read a file in the same phase.** If context editing evicts "
"a prior read result, recover from eval: "
"`window.files = window.files || {}; window.files['path'] = content;`. "
"Retrieve from eval instead of calling read_file again.\n"
```

### 6b. Improve ToolOutputTrimmer placeholder in context_editing.py

The current placeholder (from Task 4) already has `"[evicted: {hint}... — recover from eval or re-read only if essential]"`. This is fine — no additional change needed here. The eval-caching prompt instruction is the main value.

**Commit message:** `feat: never re-read heuristic + eval caching instruction`

---

## Task 7: Add integration tests

**File:** Create `tests/unit/test_prompt_efficiency.py`

Write tests that verify the structural properties of the changes (not LLM behavior):

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
        """SPINE_BASE_PROMPT should not duplicate DA middleware injections."""
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
        """Base prompt should be under 800 tokens (~3200 chars)."""
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
        assert "file" in _SPINE_SUMMARY_PROMPT.lower()
        assert "error" in _SPINE_SUMMARY_PROMPT.lower()
        assert "slice" in _SPINE_SUMMARY_PROMPT.lower()
        assert "offloaded" in _SPINE_SUMMARY_PROMPT.lower() or "history" in _SPINE_SUMMARY_PROMPT.lower()

    def test_summary_prompt_not_chatbot_oriented(self):
        """Summary prompt should NOT use chatbot framing."""
        from spine.agents.factory import _SPINE_SUMMARY_PROMPT
        assert "chat history" not in _SPINE_SUMMARY_PROMPT.lower()
        assert "user's chat" not in _SPINE_SUMMARY_PROMPT.lower()


class TestCodebaseMap:
    """Verify codebase map artifact support."""

    def test_codebase_map_in_tasks_prompt(self):
        """Tasks agent system prompt should reference codebase-map.md."""
        from spine.agents.tasks_agent import build_tasks_agent
        # We check the prompt constant, not the full agent build
        # (which requires mocks). Just verify the module is importable
        # and the prompt references codebase-map.
        import spine.agents.tasks_agent as mod
        source = open(mod.__file__).read()
        assert "codebase-map" in source, (
            "tasks_agent.py must reference codebase-map.md in its prompt"
        )

    def test_codebase_map_in_implement_prompt(self):
        """Implement agent system prompt should reference codebase-map.md."""
        import spine.agents.implement_agent as mod
        source = open(mod.__file__).read()
        assert "codebase-map" in source, (
            "implement_agent.py must reference codebase-map.md in its prompt"
        )

    def test_codebase_map_in_verify_prompt(self):
        """Verify agent system prompt should reference codebase-map.md."""
        import spine.agents.verify_agent as mod
        source = open(mod.__file__).read()
        assert "codebase-map" in source, (
            "verify_agent.py must reference codebase-map.md in its prompt"
        )
```

Run the tests. Some will fail until their corresponding tasks are done — that's expected.

**Commit message:** `test: add prompt efficiency integration tests`

---

## Task 8: Cross-phase knowledge transfer — codebase map artifact

**Problem:** The implement phase re-reads 6 of 11 source files that the tasks phase already discovered. The verify phase re-reads 3 of those same files. Each phase starts with a blank context and re-explores the codebase from scratch. In the trace, this caused ~15-20 redundant tool calls and ~3-4 minutes of wasted wall time.

**Solution:** Make the tasks phase write a `codebase-map.md` artifact alongside its slices. This artifact captures the exploration findings (file paths, key functions, imports, patterns) so subsequent phases can read the map instead of re-discovering everything.

**Files:**
- Modify: `spine/agents/tasks_agent.py` (add codebase-map.md to prompt)
- Modify: `spine/phases/tasks.py` (add codebase-map.md to instructions)
- Modify: `spine/agents/implement_agent.py` (reference codebase-map.md in gather phase)
- Modify: `spine/agents/verify_agent.py` (reference codebase-map.md in gather phase)
- Modify: `spine/agents/subagents.py` (add codebase-map.md path to subagent descriptions)

### 8a. Add codebase-map.md to tasks_agent.py prompt

In the tasks agent system prompt's `### Phase 2: Decompose` section, add:

```
- Write `codebase-map.md` to `{tasks_artifact_dir}/` — a structured summary of your exploration findings:
  - File paths with brief descriptions (what each file does)
  - Key classes and functions (names, signatures, line ranges)
  - Import chains and dependencies between the relevant modules
  - Conventions discovered (naming, patterns, error handling)
  This map eliminates re-exploration by subsequent phases.
```

### 8b. Add codebase-map.md to tasks.py prompt instructions

In `spine/phases/tasks.py`, add to the `prompt_lines` instructions (after the slice writing instructions):

```python
prompt_lines.extend([
    f"5. Write a `codebase-map.md` to `{tasks_artifact_dir}/codebase-map.md` that captures your exploration findings:\n"
    f"   - File paths with descriptions (what each file does)\n"
    f"   - Key classes and functions (names, signatures)\n"
    f"   - Import chains between relevant modules\n"
    f"   - Conventions discovered (naming, patterns, error handling)\n"
    f"   This map will be read by the implement and verify phases — it saves them from re-exploring the codebase.\n",
])
```

### 8c. Reference codebase-map.md in implement_agent.py

Already added in Task 3's gather phase: `"Read the codebase map (if available): {tasks_path}/codebase-map.md"`. Verify this is present.

### 8d. Reference codebase-map.md in verify_agent.py

Already added in Task 3's gather phase: `"Read the codebase map (if available): {tasks_path}/codebase-map.md"`. Verify this is present.

### 8e. Add codebase-map.md path to subagent descriptions

In `spine/agents/subagents.py`, update the `slice-implementer` and `slice-verifier` prompts to include a reference to the codebase map:

For **slice-implementer**, add to the workflow step 1:
```
"1. Read the slice definition, codebase-map.md, and any referenced prior artifacts (batch reads).\n"
```

For **slice-verifier**, add to the workflow step 1:
```
"1. Read the slice definition, codebase-map.md, and its acceptance criteria.\n"
```

### 8f. Add codebase-map.md to the implement phase prompt in implement.py

In `spine/phases/implement.py`, add to the `prompt_lines` (in the else branch for quick workflows, after the slice reference):

```python
f"- Codebase map: `{tasks_path}/codebase-map.md`",
```

And in the instructions section, add:
```python
"Read the codebase map FIRST — it contains file paths, key functions, and conventions "
"discovered during the tasks phase. Use it instead of re-exploring the codebase.\n",
```

### 8g. Add codebase-map.md to the verify phase prompt in verify.py

Similarly, in `spine/phases/verify.py`, add references to `codebase-map.md` in the prior artifacts list and instructions.

**Commit message:** `feat: cross-phase knowledge transfer via codebase-map.md artifact`

---

## Execution Order

1. Task 1 → compress SPINE_BASE_PROMPT, remove _RLM_PREAMBLE
2. Task 2 → subagent autonomy
3. Task 3 → gather→execute workflow prompts
4. Task 4 → context editing + summarization + paging
5. Task 5 → rlm-pattern skill rewrite
6. Task 6 → never re-read heuristic
7. Task 7 → integration tests
8. Task 8 → codebase map artifact

Run `pytest tests/unit/ -v` after EACH task. Fix any failures before moving on.
Run `ruff check spine/ tests/ && ruff format spine/ tests/` after EACH task.
