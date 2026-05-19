# SPINE Trace 79bc301c — Improvement Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reduce the 51:1 prompt-to-completion token ratio and address the six quality issues identified in the trace audit of work 79bc301c.

**Architecture:** Three layers of context management working together — (1) smarter eviction that preserves actionable metadata, (2) AI-message argument trimming to remove redundant write-content from history, and (3) earlier summarization with richer summary output. Plus targeted fixes for subagent quality, PTC readFile, duplicate verification, prompt compliance, and tool validation.

**Tech Stack:** Python 3.12+, Deep Agents middleware (AgentMiddleware), LangSmith SDK for validation, pytest for tests

---

## Context Window Management — Deep Analysis

### The Problem (from trace 79bc301c)

| Metric | Value | Healthy Range |
|--------|-------|---------------|
| Prompt:Completion ratio | 51:1 | <20:1 |
| Total prompt tokens | 2,731,451 | — |
| Total completion tokens | 53,241 | — |
| Cache read (prompt tokens saved) | 1,161,056 (43%) | >50% |
| LLM calls | 94 | — |
| Avg prompt tokens/call | ~29,000 | <20,000 |
| Peak prompt tokens (last calls) | ~37,000 | <30,000 |

### Root Cause: Unbounded Conversation Growth

Each agent turn adds 2-3 messages to the conversation. By the implement phase's final calls, the conversation contained ~60-70 messages accumulated over 34+ LLM turns. The three existing defenses:

1. **ToolOutputTrimmer** (max_full_tool_results=20) — Replaces tool results beyond the last 20 with `[evicted: <first 100 chars>...]`. But:
   - The hint is too vague — `[evicted:      1\tSystem reminder...]` tells the agent nothing useful
   - It only trims ToolMessage content, not the AIMessage that contains the tool_call arguments
   - A `write_file` with a 3KB `content` argument leaves that 3KB in the AI message even after the tool result is evicted
   - The agent still knows it *called* a tool but not what the *result* was

2. **SummarizationMiddleware** (trigger=80K tokens) — Compresses old messages into a structured summary. But:
   - By 80K tokens, the KV cache is already 62.5% full on a 128K model
   - The summarization pass itself needs 10-15K tokens of working room inside the active context
   - On the observed trace, summarization triggered 2-3 times during the implement phase, each time burning ~5K completion tokens on the summary
   - Earlier trigger = smaller summaries = cheaper summarization calls

3. **Prompt instructions** ("cache files in eval, never re-read") — The trace shows the agent ignored this. The implementer re-read dispatcher.py 5+ times. Prompt instructions alone aren't enough — structural mechanisms must enforce the behavior.

### Token Budget Per LLM Call (Observed)

```
System prompt (base + phase + DA middleware injections):  ~5-8K tokens
Conversation history (messages):                          ~15-30K tokens
New tool results (last 2-3 turns):                       ~3-8K tokens
─────────────────────────────────────────────────────────────────────
Total prompt per call:                                    ~29-37K tokens
```

The conversation history is 50-80% of every prompt. Reducing it is the highest-ROI change.

### Proposed Three-Layer Defense

```
Layer 1: Smart Eviction (ToolOutputTrimmer v2)
  - Extract structured metadata from tool results before evicting
  - For read_file: path + line count + key symbol names
  - For execute: command + exit code + result summary
  - For grep: pattern + path + match count
  - For write_file/edit_file: path + success status
  - Keeps the agent oriented without full content

Layer 2: AI Message Argument Trimming (new middleware)
  - After tool results are evicted, truncate corresponding
    AI message tool_call arguments
  - write_file content → "[content written to <path>]"
  - edit_file old_string/new_string → "[edited <path>]"
  - read_file arguments are small, leave intact
  - Reduces persistent token cost of old write operations

Layer 3: Earlier Summarization (lower trigger threshold)
  - Reduce trigger from 80K → 60K tokens
  - Keeps KV cache under 50% on 128K models
  - More frequent but cheaper summarization passes
  - Custom summary prompt already preserves working state
```

### Expected Impact

| Change | Tokens Saved Per Call (late phase) | Cumulative |
|--------|-----------------------------------|------------|
| Smart eviction metadata vs. vague hints | ~2-4K (agent doesn't re-read to recover context) | ~30-60K |
| AI argument trimming (write/edit content) | ~1-3K (write args are 1-3KB each) | ~15-45K |
| Earlier summarization (60K vs 80K trigger) | ~5-10K (earlier compression = smaller history) | ~50-100K |
| **Total estimated reduction** | | **~100-200K prompt tokens** |

On the observed trace (2.7M prompt tokens), this is a 4-7% reduction in total, but the per-call improvement is larger in the late phase where costs are highest — from ~37K/call to ~25-28K/call.

---

## Task 1: Smart Eviction — Upgrade ToolOutputTrimmer with Structured Metadata

**Objective:** Replace vague eviction hints with actionable metadata so agents know what was evicted without re-reading.

**Files:**
- Modify: `spine/agents/context_editing.py` (full rewrite of eviction logic)
- Test: `tests/unit/test_context_editing.py`

**Step 1: Write failing tests for smart eviction**

```python
# tests/unit/test_context_editing.py
import pytest
from spine.agents.context_editing import ToolOutputTrimmer


class TestSmartEviction:
    """Test that ToolOutputTrimmer extracts structured metadata from tool results."""

    def test_read_file_metadata(self):
        """read_file results should extract path and line count."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=1)
        # Simulate a read_file tool result with file path in name field
        # and multi-line content
        messages = [
            _make_tool_msg("read_file", '     1\tdef hello():\n     2\t    return "world"\n     3\t', name="read_file"),
            _make_tool_msg("read_file", '     1\tclass Foo:\n     2\t    pass\n', name="read_file"),
        ]
        # After trimming, the evicted message should contain path info
        # and line count
        result = trimmer._evict(messages[0], tool_name="read_file", args={"file_path": "src/main.py"})
        assert "src/main.py" in result
        assert "3 lines" in result

    def test_execute_metadata(self):
        """execute results should extract command, exit code, and summary."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=1)
        result = trimmer._evict(
            _make_tool_msg("execute", "5 passed, 1 failed\nexit code: 1"),
            tool_name="execute",
            args={"command": "pytest tests/"},
        )
        assert "pytest tests/" in result
        assert "exit=1" in result

    def test_grep_metadata(self):
        """grep results should extract pattern, path, and match count."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=1)
        content = "/src/main.py:\n  42: def hello():\n  89: def goodbye():\n"
        result = trimmer._evict(
            _make_tool_msg("grep", content),
            tool_name="grep",
            args={"pattern": "def ", "path": "src/"},
        )
        assert "def " in result
        assert "2 matches" in result

    def test_write_file_metadata(self):
        """write_file results should extract path and success status."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=1)
        result = trimmer._evict(
            _make_tool_msg("write_file", "Updated file /src/main.py"),
            tool_name="write_file",
            args={"file_path": "src/main.py"},
        )
        assert "src/main.py" in result
        assert "written" in result.lower() or "updated" in result.lower()


def _make_tool_msg(name, content, tool_call_id="tc_1"):
    """Create a mock ToolMessage."""
    from langchain_core.messages import ToolMessage
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
```

**Step 2: Run tests to verify failure**

Run: `cd /home/pat/projects/spine && uv run pytest tests/unit/test_context_editing.py -v --tb=short 2>&1 | tail -30`
Expected: FAIL — `_evict` method doesn't exist yet

**Step 3: Implement smart eviction in ToolOutputTrimmer**

Rewrite `context_editing.py` with structured metadata extraction:

```python
class ToolOutputTrimmer(AgentMiddleware):
    """Trims old tool outputs from the conversation with structured metadata.

    Instead of replacing evicted tool results with a vague placeholder,
    extracts actionable metadata (file path, line count, key symbols,
    command status, match counts) so the agent knows what was evicted
    without re-reading the full content.
    """

    def __init__(
        self,
        max_full_tool_results: int = 20,
    ) -> None:
        self.max_full_tool_results = max_full_tool_results

    def _extract_metadata(
        self,
        content: str,
        tool_name: str,
        tool_args: dict | None = None,
    ) -> str:
        """Extract structured metadata from a tool result for eviction placeholder."""
        tool_args = tool_args or {}

        if tool_name == "read_file":
            path = tool_args.get("file_path", "?")
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            # Extract function/class names from line-numbered content
            symbols = []
            for line in content.split("\n"):
                stripped = line.strip()
                # Remove line number prefix (e.g., "    42\t")
                if "\t" in stripped[:10]:
                    stripped = stripped.split("\t", 1)[-1].strip()
                if stripped.startswith(("def ", "class ", "async def ")):
                    name = stripped.split("(")[0].split(":")[0].strip()
                    if len(name) < 80:
                        symbols.append(name)
            sym_str = ", ".join(symbols[:5]) if symbols else ""
            hint = f"[read: {path} ({line_count} lines)"
            if sym_str:
                hint += f" — {sym_str}"
            hint += "]"
            return hint

        elif tool_name == "execute":
            cmd = tool_args.get("command", "?")
            if len(cmd) > 60:
                cmd = cmd[:57] + "..."
            # Extract exit code from content or args
            exit_code = tool_args.get("timeout", None)
            # Parse common result patterns
            last_lines = content.strip().split("\n")[-3:]
            summary = last_lines[-1][:80] if last_lines else ""
            hint = f"[exec: {cmd}"
            if "exit code" in content.lower():
                for line in last_lines:
                    if "exit code" in line.lower():
                        hint += f" — {line.strip()}"
                        break
            elif summary:
                hint += f" — {summary}"
            hint += "]"
            return hint

        elif tool_name == "grep":
            pattern = tool_args.get("pattern", "?")
            path = tool_args.get("path", tool_args.get("glob", "?"))
            matches = content.count("\n") + 1 if content.strip() else 0
            # Grep output has one line per match (with filename prefix)
            # Count unique file matches
            files = set()
            for line in content.split("\n"):
                if line.startswith("/"):
                    files.add(line.split(":")[0])
            hint = f"[grep: '{pattern}' in {path}"
            if files:
                hint += f" — {len(files)} files, {matches} lines"
            else:
                hint += f" — {matches} lines"
            hint += "]"
            return hint

        elif tool_name in ("write_file", "edit_file"):
            path = tool_args.get("file_path", "?")
            action = "written" if tool_name == "write_file" else "edited"
            hint = f"[{action}: {path}"
            # For edit_file, show what changed
            if tool_name == "edit_file" and tool_args.get("new_string"):
                new = tool_args["new_string"][:60]
                hint += f" → {new}..."
            hint += "]"
            return hint

        elif tool_name == "glob":
            count = content.count("\n") + 1 if content.strip() else 0
            pattern = tool_args.get("pattern", "?")
            return f"[glob: '{pattern}' — {count} files]"

        elif tool_name == "ls":
            count = content.count("\n") + 1 if content.strip() else 0
            path = tool_args.get("path", "?")
            return f"[ls: {path} — {count} entries]"

        else:
            # Generic eviction with content hint
            hint = content[:100].split("\n")[0]
            if hint and len(hint) > 10:
                return f"[evicted({tool_name}): {hint}...]"
            return f"[evicted: {tool_name} result]"

    async def awrap_model_call(self, request, handler):
        """Trim old tool results before each model call with structured metadata."""
        messages = request.messages

        # Collect tool result indices and their corresponding AI message tool_calls
        tool_result_info = []  # (index, tool_name, tool_args)
        ai_tool_call_map = {}  # tool_call_id → (tool_name, args)

        # First pass: build map of tool_call_id → (name, args) from AI messages
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    ai_tool_call_map[tc.get("id", "")] = (
                        tc.get("name", ""),
                        tc.get("args", {}),
                    )

        # Second pass: collect tool result indices
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_result_indices.append(i)

        if len(tool_result_indices) <= self.max_full_tool_results:
            return await handler(request)

        # Trim old results with structured metadata
        trim_count = len(tool_result_indices) - self.max_full_tool_results
        trimmed_messages = list(messages)

        for idx in tool_result_indices[:trim_count]:
            msg = trimmed_messages[idx]
            # Look up the tool name and args from the AI message that triggered it
            tc_id = getattr(msg, "tool_call_id", "")
            tool_name = getattr(msg, "name", "")
            tool_args = {}
            if tc_id in ai_tool_call_map:
                tool_name, tool_args = ai_tool_call_map[tc_id]
            elif not tool_name:
                tool_name = ai_tool_call_map.get(tc_id, ("", {}))[0]

            metadata = self._extract_metadata(
                getattr(msg, "content", ""),
                tool_name,
                tool_args,
            )
            try:
                trimmed_messages[idx] = msg.__class__(
                    content=metadata,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
            except Exception:
                pass

        return await handler(request.override(messages=trimmed_messages))

    # ── Pass-through tool call wrapping ──
    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)
```

**Step 4: Run tests to verify pass**

Run: `cd /home/pat/projects/spine && uv run pytest tests/unit/test_context_editing.py -v --tb=short`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add spine/agents/context_editing.py tests/unit/test_context_editing.py
git commit -m "feat: smart eviction metadata in ToolOutputTrimmer

Replace vague eviction hints with structured metadata extracted from
tool results. For read_file: path + line count + key symbols.
For execute: command + exit code + summary. For grep: pattern +
match count. For write/edit: path + action. Agents can now
understand what was evicted without re-reading full content.

Trace 79bc301c showed 51:1 prompt-to-completion ratio — the
vague hints caused agents to re-read files to recover context."
```

---

## Task 2: AI Message Argument Trimming — Remove Redundant Write Content from History

**Objective:** After tool results are evicted, truncate the corresponding AI message's tool_call arguments for write_file and edit_file. The content is already on disk — keeping the full argument in history is pure waste.

**Files:**
- Modify: `spine/agents/context_editing.py` (add AI message trimming to awrap_model_call)
- Test: `tests/unit/test_context_editing.py`

**Step 1: Write failing tests**

```python
class TestAIArgumentTrimming:
    """Test that AI message tool_call arguments are trimmed after eviction."""

    def test_write_file_args_trimmed(self):
        """write_file content arg should be replaced with path reference after eviction."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=0)
        ai_msg = _make_ai_msg_with_tool_calls([
            {"id": "tc_1", "name": "write_file", "args": {
                "file_path": "src/main.py",
                "content": "def hello():\n    return 'world'\n" * 50,  # ~800 chars
            }},
        ])
        tool_msg = _make_tool_msg("write_file", "Updated file /src/main.py", tool_call_id="tc_1")
        messages = [ai_msg, tool_msg]
        # After trimming, the AI message's write_file args should have truncated content
        result = trimmer._trim_ai_args(messages, evicted_ids={"tc_1"})
        ai_result = result[0]
        tc = ai_result.tool_calls[0]
        assert len(tc["args"].get("content", "")) < 50  # Truncated to path reference

    def test_read_file_args_not_trimmed(self):
        """read_file args (small) should NOT be trimmed."""
        trimmer = ToolOutputTrimmer(max_full_tool_results=0)
        ai_msg = _make_ai_msg_with_tool_calls([
            {"id": "tc_1", "name": "read_file", "args": {"file_path": "src/main.py"}},
        ])
        tool_msg = _make_tool_msg("read_file", "...", tool_call_id="tc_1")
        messages = [ai_msg, tool_msg]
        result = trimmer._trim_ai_args(messages, evicted_ids={"tc_1"})
        tc = result[0].tool_calls[0]
        assert tc["args"]["file_path"] == "src/main.py"  # Unchanged


def _make_ai_msg_with_tool_calls(tool_calls):
    from langchain_core.messages import AIMessage
    return AIMessage(content="", tool_calls=tool_calls)
```

**Step 2: Run tests to verify failure**

Run: `cd /home/pat/projects/spine && uv run pytest tests/unit/test_context_editing.py::TestAIArgumentTrimming -v --tb=short`
Expected: FAIL — `_trim_ai_args` method doesn't exist

**Step 3: Implement AI argument trimming**

Add to `ToolOutputTrimmer` in `context_editing.py`:

```python
# Maximum character length for tool_call args before trimming
_MAX_ARG_LEN = 100

# Tool calls whose args should be trimmed after eviction
_TRIMMABLE_TOOLS = {"write_file", "edit_file"}

def _trim_ai_args(self, messages: list, evicted_ids: set[str]) -> list:
    """Trim tool_call arguments in AI messages for evicted tool results.

    After tool results are evicted, the AI message that triggered them
    still contains the full tool_call arguments. For write_file and
    edit_file, the 'content', 'old_string', and 'new_string' args
    can be 1-10KB each. Since the content is already on disk, these
    are pure waste in the conversation history.

    Replace large args with compact path references.
    """
    if not evicted_ids:
        return messages

    trimmed = list(messages)
    for i, msg in enumerate(trimmed):
        if not (hasattr(msg, "type") and msg.type == "ai"):
            continue
        if not hasattr(msg, "tool_calls") or not msg.tool_calls:
            continue

        needs_update = False
        new_tool_calls = []
        for tc in msg.tool_calls:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            tc_args = dict(tc.get("args", {}))

            if tc_id in evicted_ids and tc_name in self._TRIMMABLE_TOOLS:
                # Replace large args with compact references
                if tc_name == "write_file" and "content" in tc_args:
                    path = tc_args.get("file_path", "?")
                    tc_args["content"] = f"[{len(tc_args['content'])} chars written to {path}]"
                    needs_update = True
                elif tc_name == "edit_file":
                    path = tc_args.get("file_path", "?")
                    if "old_string" in tc_args and len(tc_args["old_string"]) > self._MAX_ARG_LEN:
                        tc_args["old_string"] = f"[{len(tc_args['old_string'])} chars from {path}]"
                        needs_update = True
                    if "new_string" in tc_args and len(tc_args["new_string"]) > self._MAX_ARG_LEN:
                        tc_args["new_string"] = f"[{len(tc_args['new_string'])} chars → {path}]"
                        needs_update = True

            new_tool_calls.append({**tc, "args": tc_args})

        if needs_update:
            try:
                trimmed[i] = msg.__class__(
                    content=msg.content,
                    tool_calls=new_tool_calls,
                )
            except Exception:
                pass

    return trimmed
```

Then update `awrap_model_call` to also trim AI args after evicting tool results. Collect the evicted tool_call_ids and pass them to `_trim_ai_args`.

**Step 4: Run tests**

Run: `cd /home/pat/projects/spine && uv run pytest tests/unit/test_context_editing.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add spine/agents/context_editing.py tests/unit/test_context_editing.py
git commit -m "feat: trim AI message tool_call args after eviction

write_file content and edit_file old_string/new_string can be 1-10KB
each. After the tool result is evicted, these args are pure waste —
the content is already on disk. Replace with compact path references.

Reduces persistent token cost of write-heavy phases (implement, verify)
by ~1-3KB per evicted write operation."
```

---

## Task 3: Earlier Summarization Trigger — Reduce from 80K to 60K Tokens

**Objective:** Reduce the summarization trigger threshold from 80K to 60K tokens to compress conversation history earlier, keeping KV cache under 50% on 128K models.

**Files:**
- Modify: `spine/agents/factory.py:453-456` (trigger parameter)
- Test: `tests/unit/test_factory.py`

**Step 1: Update the trigger threshold**

In `spine/agents/factory.py`, function `_add_summarization_middleware`:

Change:
```python
auto_mw = create_summarization_middleware(
    model,
    backend,
    trigger=("tokens", 80000),
    keep=("messages", 20),
    summary_prompt=_SPINE_SUMMARY_PROMPT,
)
```

To:
```python
auto_mw = create_summarization_middleware(
    model,
    backend,
    trigger=("tokens", 60000),
    keep=("messages", 20),
    summary_prompt=_SPINE_SUMMARY_PROMPT,
)
```

**Step 2: Update the docstring**

In `_add_summarization_middleware` docstring, change:
```
1. Token-based trigger (80K) — model-independent, leaves 48K buffer.
```
To:
```
1. Token-based trigger (60K) — model-independent, keeps KV cache under
   50% on 128K models. Earlier compression means smaller summaries and
   less re-reading by the agent.
```

**Step 3: Add a test verifying the trigger value**

```python
# In tests/unit/test_factory.py
class TestSummarizationConfig:
    def test_summarization_trigger_60k(self):
        """Summarization trigger should be 60K tokens, not 80K."""
        import inspect
        from spine.agents.factory import _add_summarization_middleware
        source = inspect.getsource(_add_summarization_middleware)
        assert 'trigger=("tokens", 60000)' in source or "trigger=('tokens', 60000)" in source
```

**Step 4: Run tests**

Run: `cd /home/pat/projects/spine && uv run pytest tests/unit/test_factory.py -v --tb=short`
Expected: PASS

**Step 5: Commit**

```bash
git add spine/agents/factory.py tests/unit/test_factory.py
git commit -m "perf: reduce summarization trigger from 80K to 60K tokens

Earlier compression keeps KV cache under 50% on 128K models.
The 80K trigger left the cache at 62.5% before summarization,
making the summarization pass itself expensive and leaving the
agent with a large context that encourages re-reading.

60K trigger = more frequent but cheaper summarization calls,
with the agent spending less time in the high-context-cost zone."
```

---

## Task 4: Subagent Quality Gate — Re-dispatch Empty Researchers

**Objective:** When a researcher subagent returns empty results (no file_map, no patterns), re-dispatch with a more specific prompt instead of accepting the empty result silently.

**Files:**
- Modify: `spine/agents/subagents.py` (add `_RE_RESEARCH_PROMPT` and quality check in researcher prompt)
- Test: `tests/unit/test_subagents.py`

**Step 1: Strengthen the researcher prompt with minimum output enforcement**

In `SUBAGENT_PROMPTS["researcher"]`, append:

```
"\n\nMINIMUM OUTPUT REQUIREMENTS:\n"
"- You MUST read at least 2 files before producing your summary.\n"
"- If you cannot read files (tool errors, permission issues), report that\n"
"  as your summary with the error details — do NOT return empty results.\n"
"- Your file_map MUST contain at least 1 entry.\n"
"- Your summary MUST be at least 2 sentences.\n"
"- If you produce empty results, you WILL be re-dispatched, wasting time\n"
"  and tokens. Do the work correctly the first time.\n"
```

**Step 2: Add re-research prompt for re-dispatch**

In `subagents.py`:

```python
_RE_RESEARCH_PROMPT_SUFFIX = (
    "\n\n⚠ RE-DISPATCH: A previous researcher returned empty results for this "
    "task. This is your second chance. You MUST:\n"
    "1. Read at least 3 files relevant to the task description.\n"
    "2. Produce a file_map with at least 2 entries.\n"
    "3. If files cannot be found, explain what you searched and what went wrong.\n"
    "Do NOT return empty results again."
)
```

**Step 3: Add quality check logic to the researcher prompt assembly**

This is a prompt-level fix — the agent calling `tools.task()` should check the result and re-dispatch if empty. Add a section to the tasks and specify phase prompts:

In `spine/agents/tasks_agent.py` and `spine/agents/specify_agent.py`, after the subagent dispatch instructions, add:

```
"After researcher subagents return, check their results. If a researcher "
"returns empty file_map and empty patterns, re-dispatch it with a more "
"specific description and the re-research instruction appended. Do not "
"accept empty researcher results silently."
```

**Step 4: Write test**

```python
class TestResearcherQualityGate:
    def test_researcher_prompt_has_minimum_output(self):
        from spine.agents.subagents import SUBAGENT_PROMPTS
        prompt = SUBAGENT_PROMPTS["researcher"]
        assert "at least 2 files" in prompt
        assert "empty results" in prompt.lower()

    def test_re_research_prompt_exists(self):
        from spine.agents.subagents import _RE_RESEARCH_PROMPT_SUFFIX
        assert "RE-DISPATCH" in _RE_RESEARCH_PROMPT_SUFFIX
```

**Step 5: Run tests and commit**

```bash
git add spine/agents/subagents.py tests/unit/test_subagents.py
git commit -m "fix: enforce minimum output quality on researcher subagents

Trace 79bc301c: 1 of 3 researcher subagents returned empty results
(empty file_map, patterns, dependencies). This wasted a subagent
turn and tokens. Strengthened the prompt with minimum output
requirements and added a re-research suffix for re-dispatch."
```

---

## Task 5: Fix PTC readFile Null Returns for Source Files

**Objective:** Debug and fix why `tools.readFile()` returns `null` for `.py` source files inside eval context. This negates the key benefit of the RLM pattern — batch file reads inside eval.

**Files:**
- Investigate: `spine/agents/interpreter.py` (PTC allowlist configuration)
- Investigate: Deep Agents QuickJS PTC layer (`langchain_quickjs`)
- Modify: `spine/agents/context_editing.py` or `spine/agents/interpreter.py` (depending on root cause)
- Test: `tests/unit/test_interpreter.py`

**Step 1: Reproduce the bug**

Write a minimal test that simulates the eval context and calls `tools.readFile`:

```python
class TestPTCReadFile:
    @pytest.mark.asyncio
    async def test_readfile_returns_content_for_py_files(self):
        """tools.readFile inside eval should return content, not null."""
        # This test requires a running interpreter with PTC enabled
        # and a virtual filesystem with a known .py file
        # If this test fails, it confirms the bug from trace 79bc301c
        ...
```

**Step 2: Root cause investigation**

The trace showed that `tools.readFile({ file_path: 'spine/work/dispatcher.py' })` returned `null` while `tools.ls` and `tools.glob` worked. The most likely cause:

- PTC's `readFile` maps to DA's `read_file` tool
- Under `virtual_mode=True`, the path `spine/work/dispatcher.py` might need a leading `/` → `/spine/work/dispatcher.py`
- Or the path resolution in PTC doesn't go through `_NormalizingLocalShellBackend`

**Investigation commands:**
```bash
# Check the QuickJS PTC bridge code
find .venv -path "*/langchain_quickjs/*" -name "*.py" | head -5
# Check how readFile maps to the DA tool
grep -r "readFile\|read_file" .venv/lib/python3.13/site-packages/langchain_quickjs/ --include="*.py" -l
```

**Step 3: Fix based on findings**

If the issue is path resolution:
- Add path normalization in the PTC bridge or the eval context
- Update the `SPINE_FILESYSTEM_PROMPT` PTC section to clarify path conventions

If the issue is a DA bug:
- Add a workaround in the eval seed code that tests `readFile` and falls back gracefully
- Document the limitation in the prompt

**Step 4: Update the SPINE_BASE_PROMPT PTC guidance**

In `spine/agents/profile.py`, the PTC section currently says:
```
PTC tool names are camelCase (`tools.readFile`), arguments are snake_case
(`{file_path: '...'}`), and return values are native JS types —
`readFile` returns a string, not an object.
```

Add:
```
For `.py` source files, `readFile` may return `null` due to virtual_mode
path resolution. If this happens, use the native `read_file` tool instead.
Prefer eval for discovery (ls, glob, grep) and subagent dispatch; use
native read_file for source code content.
```

Wait — this is already documented in the spine skill. Let me check if it's in the actual prompt...

Actually, looking at the trace data, the agent DID fall back to `read_file` after the null returns. The issue is the wasted turn. The fix should either:
(a) Make PTC readFile work, or
(b) Remove it from the PTC allowlist for phases that need source file reading, and direct agents to use `eval` for orchestration only (ls, glob, grep, task dispatch)

Option (b) is simpler and more reliable:

```python
_PTC_ALLOWLISTS = {
    "specify":  ["task", "grep", "glob", "ls", "write_file", "edit_file"],
    "tasks":    ["task", "grep", "glob", "ls", "write_file", "edit_file"],
    "implement": ["task", "grep", "glob", "ls", "write_file", "edit_file"],
    "verify":   ["task", "grep", "glob", "ls", "write_file", "edit_file"],
}
```

Remove `read_file` from the PTC allowlist. Agents will use the native `read_file` tool for source code reading and use eval for discovery/orchestration only. This eliminates the null-return problem entirely.

**Step 5: Commit**

```bash
git add spine/agents/interpreter.py tests/unit/test_interpreter.py
git commit -m "fix: remove read_file from PTC allowlist (returns null for .py files)

tools.readFile inside eval returns null for project source files under
virtual_mode=True. Removing read_file from the PTC allowlist forces
agents to use the native read_file tool (which works correctly) and
reserve eval for orchestration: ls, glob, grep, subagent dispatch.

The wasted turn from null returns (trace 79bc301c) is eliminated.
Agents get the same functionality via native read_file with batch
reads (≥3 files per turn)."
```

---

## Task 6: Remove Implement-Phase Verifier Subagent (Deduplicate Verification)

**Objective:** The implement phase dispatches a verifier subagent AND the workflow runs a dedicated verify phase afterward. Both produce verification reports. Remove the implement-phase verifier to eliminate redundant token spend.

**Files:**
- Modify: `spine/agents/implement_agent.py` (remove verifier subagent from phase prompt)
- Modify: `spine/agents/subagents.py` (remove slice-verifier from IMPLEMENT phase subagents)
- Test: `tests/unit/test_implement_agent.py`

**Step 1: Remove slice-verifier from IMPLEMENT phase subagents**

In `spine/agents/subagents.py`, `PHASE_SUBAGENTS`:

```python
PHASE_SUBAGENTS: dict[str, list[str]] = {
    PhaseName.SPECIFY.value: ["researcher"],
    PhaseName.TASKS.value: ["researcher"],
    PhaseName.IMPLEMENT.value: ["slice-implementer"],  # Remove slice-verifier
    PhaseName.VERIFY.value: ["slice-verifier"],
}
```

**Step 2: Remove verifier dispatch instructions from implement prompt**

In `spine/agents/implement_agent.py`, remove the "Phase 4: Verify" section from the implement system prompt. Replace with:

```python
"### Phase 4: Report (1 turn)\n"
"Write implementation.md summary to disk with:\n"
"- Files changed and what was modified\n"
"- Test results summary\n"
"- Any remaining issues or warnings\n"
```

**Step 3: Update the implement prompt's subagent section**

Remove any references to dispatching slice-verifier subagents from eval. The implement agent now only dispatches slice-implementer subagents.

**Step 4: Write test**

```python
class TestImplementNoVerifier:
    def test_implement_subagents_only_slice_implementer(self):
        from spine.agents.subagents import PHASE_SUBAGENTS
        from spine.models.enums import PhaseName
        assert PHASE_SUBAGENTS[PhaseName.IMPLEMENT.value] == ["slice-implementer"]

    def test_implement_prompt_no_verifier_dispatch(self):
        """Implement prompt should not mention slice-verifier dispatch."""
        # Build a minimal state and check the prompt
        from spine.agents.implement_agent import build_implement_agent
        # Can't fully build without config, so check the prompt construction
        # by inspecting the source
        import inspect
        from spine.agents import implement_agent
        source = inspect.getsource(implement_agent)
        assert "slice-verifier" not in source
```

**Step 5: Run tests and commit**

```bash
git add spine/agents/implement_agent.py spine/agents/subagents.py tests/unit/test_implement_agent.py
git commit -m "refactor: remove implement-phase verifier subagent (deduplicate)

The implement phase dispatched a slice-verifier subagent AND the
workflow runs a dedicated verify phase afterward. Both produced
verification reports, doubling verification token cost.

The dedicated verify phase is the authoritative one. Removing the
implement-phase verifier simplifies the implement prompt and saves
the tokens spent on duplicate verification.

Trace 79bc301c: implement produced its own verification AND verify
produced verification.md — both were thorough."
```

---

## Task 7: Enforce Verify-Phase Subagent Dispatch for ≥2 Slices

**Objective:** The verify phase prompt says "for ≥2 slices: dispatch slice-verifier subagents" but the agent verified all 3 slices inline. Strengthen the prompt to make this mandatory.

**Files:**
- Modify: `spine/agents/verify_agent.py` (strengthen dispatch instruction)

**Step 1: Update the verify prompt**

In `spine/agents/verify_agent.py`, change the "Phase 2: Verify" section from:

```python
"### Phase 2: Verify (1-2 turns)\n"
"For ≥2 slices: dispatch slice-verifier subagents via "
"\"Promise.allSettled(tools.task(...))\" from eval — one per slice.\n"
"For 1 slice: verify directly using read_file and execute.\n"
```

To:

```python
"### Phase 2: Verify (1-2 turns) — MANDATORY PARALLEL\n"
"When there are ≥2 slices, you MUST dispatch one slice-verifier "
"subagent per slice via \"Promise.allSettled(tools.task(...))\" from "
"eval. Do NOT verify all slices inline — parallel verification is "
"required to keep context lean. Each subagent gets a fresh, small "
"context instead of bloating your conversation.\n"
"When there is exactly 1 slice: verify directly using read_file and "
"execute.\n"
```

**Step 2: Add enforcement note**

```python
"## Subagent Dispatch — MANDATORY FOR ≥2 SLICES\n"
"Failure to dispatch slice-verifier subagents when there are ≥2 slices "
"violates the workflow contract. The slice-verifier subagents have "
"isolated contexts and return compact results. Verifying all slices "
"inline bloats your conversation and wastes tokens.\n"
```

**Step 3: Commit**

```bash
git add spine/agents/verify_agent.py
git commit -m "fix: enforce parallel slice-verifier dispatch for ≥2 slices

Trace 79bc301c: the verify agent verified all 3 slices inline
instead of dispatching slice-verifier subagents as the prompt
instructed. This bloats the verify phase's context unnecessarily.

Strengthened the prompt to make subagent dispatch mandatory for
≥2 slices, with explicit enforcement language."
```

---

## Task 8: edit_file Empty old_string Validation

**Objective:** Reject `edit_file` calls with empty `old_string` with a helpful error message instead of the confusing "appears 2308 times" error.

**Files:**
- This requires either a DA middleware patch or a SPINE ToolSchemaValidator enhancement
- Simplest: add validation in `spine/agents/tool_schema_validator.py`

**Step 1: Add edit_file validation to ToolSchemaValidator**

In `spine/agents/tool_schema_validator.py`, add a pre-check for edit_file:

```python
# In the validation logic, add:
if tool_name == "edit_file":
    old_string = args.get("old_string", "")
    if not old_string:
        return {
            "error": "edit_file: old_string cannot be empty — "
                     "it matches every location in the file. "
                     "Use write_file instead if you want to "
                     "replace the entire file content.",
        }
```

**Step 2: Write test**

```python
class TestEditFileValidation:
    def test_empty_old_string_rejected(self):
        from spine.agents.tool_schema_validator import ToolSchemaValidator
        validator = ToolSchemaValidator()
        result = validator._validate_tool_call("edit_file", {
            "file_path": "test.py",
            "old_string": "",
            "new_string": "content",
        })
        assert result is not None
        assert "empty" in result["error"].lower()
```

**Step 3: Run tests and commit**

```bash
git add spine/agents/tool_schema_validator.py tests/unit/test_tool_schema_validator.py
git commit -m "fix: reject edit_file with empty old_string

The agent tried edit_file with old_string='' which matches
every location (2308 occurrences). This produced a confusing
error message. Now returns a clear message directing to
write_file for full-file replacement."
```

---

## Task 9: Enhance codebase-map.md — Include Function Signatures and Key Snippets

**Objective:** The current codebase-map.md contains file paths and brief descriptions but not the actual function signatures or line ranges. The implement phase still needs to re-read files to find the exact functions. Enrich the map with concrete code context.

**Files:**
- Modify: `spine/agents/tasks_agent.py` (enhance codebase-map instructions in prompt)

**Step 1: Update tasks_agent.py codebase-map instructions**

In the tasks system prompt, the codebase-map section currently says:

```
## codebase-map.md
Write `codebase-map.md` to `.spine/artifacts/79bc301c/tasks/` — a structured
summary of your exploration findings:
- File paths with brief descriptions (what each file does)
- Key classes and functions (names, signatures, line ranges)
- Import chains and dependencies between the relevant modules
- Conventions discovered (naming, patterns, error handling)
```

Replace with:

```python
"## codebase-map.md\n"
"Write `codebase-map.md` to `{tasks_path}/codebase-map.md` — a structured\n"
"summary of your exploration findings. This map is the PRIMARY context\n"
"for the implement and verify phases — they read it first instead of\n"
"re-exploring. Include enough detail that implementers can locate the\n"
"exact code to modify without re-reading entire files.\n\n"
"Required sections:\n"
"1. **Files** — path, description, line count\n"
"2. **Key Functions** — name, signature, line range (from read_file output),\n"
"   and a 1-line description. For example:\n"
"   `submit_work(description, work_type, config) → dict  [L367-420]  — creates work entry and starts graph`\n"
"3. **Import Chains** — which modules import which\n"
"4. **Conventions** — naming, patterns, error handling\n"
"5. **Modification Targets** — for each file that will be modified,\n"
"   include the 3-5 line code snippet around the change site. This\n"
"   eliminates the need for implementers to re-read those sections.\n"
"   Mark each snippet with its line range.\n\n"
"Example modification target:\n"
"```python\n"
"# spine/work/dispatcher.py [L407-420]\n"
"        if any(\n"
"            isinstance(f, dict) and f.get(\"status\") == \"needs_review\"\n"
"            for f in feedback\n"
"        ):\n"
"            final_status = \"needs_review\"\n"
"```\n"
"This shows implementers the exact insertion point for the\n"
"awaiting_approval logic.\n"
```

**Step 2: Commit**

```bash
git add spine/agents/tasks_agent.py
git commit -m "feat: enrich codebase-map with function signatures and code snippets

Trace 79bc301c: implement phase re-read 55% of the files that tasks
already explored, plus 16 redundant globs. The codebase-map had only
descriptions, not concrete code context. Adding function signatures,
line ranges, and modification-target snippets lets implementers locate
the exact code to modify without re-reading entire files."
```

---

## Task 10: Prompt Efficiency — Reduce SPINE_BASE_PROMPT Size

**Objective:** Audit the SPINE_BASE_PROMPT for redundancy with DA middleware injections. Remove any content that DA middleware already provides (tool descriptions, filesystem instructions, etc.).

**Files:**
- Modify: `spine/agents/profile.py` (SPINE_BASE_PROMPT)
- Test: `tests/unit/test_profile.py`

**Step 1: Audit current SPINE_BASE_PROMPT against DA middleware**

The current prompt has these sections:
1. "You are a phase executor" — needed, replaces DA BASE_AGENT_PROMPT
2. "Core Behaviour" — needed, behavioral guidance
3. "Tools" — partially redundant with FilesystemMiddleware, SkillsMiddleware, etc.
4. "Workflow Context" — needed
5. "Output" — needed

The "Tools" section says:
```
- Read before write — inspect existing code before modifying it.
- Test after write — run tests immediately after making changes.
- Use `task` subagents for parallel work on independent slices.
- Use `eval` to orchestrate multi-step workflows in code, not conversation.
- Context is L1 cache; conversation history is swap...
- Never re-read a file in the same phase...
```

Lines 1-4 are behavioral principles (keep). Lines 5-6 are context management instructions (keep but strengthen). The PTC naming convention (`tools.readFile`, camelCase, snake_case args) is ALSO in the eval middleware's injection — check for duplication.

**Step 2: Measure and reduce**

Current SPINE_BASE_PROMPT is ~1,200 chars (~300 tokens). This is already quite lean. The main duplication risk is with the CodeInterpreterMiddleware's PTC instructions. Check if the eval middleware injects PTC naming conventions separately — if so, remove the duplicate from SPINE_BASE_PROMPT.

**Step 3: Add a size budget test**

```python
class TestProfileEfficiency:
    def test_base_prompt_under_token_budget(self):
        from spine.agents.profile import SPINE_BASE_PROMPT
        # At ~4 chars/token, 2000 chars = ~500 tokens
        # We want the base prompt under 500 tokens
        assert len(SPINE_BASE_PROMPT) < 2500
```

**Step 4: Commit**

```bash
git add spine/agents/profile.py tests/unit/test_profile.py
git commit -m "perf: audit SPINE_BASE_PROMPT for DA middleware duplication

Ensured no duplication between SPINE_BASE_PROMPT and DA middleware
injections (FilesystemMiddleware, CodeInterpreterMiddleware, etc.).
Added size budget test (<500 tokens). Current prompt is already lean
at ~300 tokens — this commit formalizes the budget."
```

---

## Task 11: Integration Test — Run a Quick Work End-to-End and Validate Token Metrics

**Objective:** After all changes, run a quick work item end-to-end and validate that the prompt-to-completion ratio improves from the baseline 51:1.

**Files:**
- Create: `tests/integration/test_context_efficiency.py`

**Step 1: Write integration test skeleton**

```python
import pytest

@pytest.mark.integration
class TestContextEfficiency:
    """Validate token efficiency improvements from context management changes.

    This test requires:
    - LANGSMITH_API_KEY set
    - A running SPINE instance (or in-process workflow)
    - A small work item to execute

    After execution, validate:
    - Prompt:Completion ratio < 30:1 (down from 51:1)
    - ToolOutputTrimmer produced structured metadata (no vague hints)
    - Summarization triggered at ~60K tokens
    - No AI message with full write_file content after eviction
    """
    pass  # Implementation depends on test infrastructure
```

**Step 2: Commit**

```bash
git add tests/integration/test_context_efficiency.py
git commit -m "test: add context efficiency integration test skeleton

Validates the improvements from Tasks 1-3 (smart eviction,
AI argument trimming, earlier summarization). Full implementation
requires SPINE infrastructure — this is the test contract."
```

---

## Summary — Expected Impact

| Task | Issue | Impact | Priority |
|------|-------|--------|----------|
| 1 | Smart eviction metadata | Agents don't re-read evicted content | 🔴 High |
| 2 | AI argument trimming | ~1-3KB saved per evicted write | 🟡 Medium |
| 3 | Earlier summarization (60K) | KV cache stays under 50% | 🔴 High |
| 4 | Researcher quality gate | No more empty subagent results | 🟡 Medium |
| 5 | Remove readFile from PTC | Eliminates null-return waste | 🟡 Medium |
| 6 | Remove implement verifier | Deduplicate verification | 🟢 Low |
| 7 | Enforce verify subagent dispatch | Parallel verification | 🟢 Low |
| 8 | edit_file empty old_string | Clearer error message | 🟢 Low |
| 9 | Enrich codebase-map | Less cross-phase re-reading | 🔴 High |
| 10 | Prompt efficiency audit | Formalize token budget | 🟢 Low |
| 11 | Integration test | Validate improvements | 🟡 Medium |

**Estimated combined impact on trace 79bc301c:**
- Prompt:Completion ratio: 51:1 → ~30-35:1
- Total prompt tokens: 2.7M → ~2.0-2.2M
- File re-reads: ~15 redundant → ~5-8
- Wasted subagent turns: 1 → 0
