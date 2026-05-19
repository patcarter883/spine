# Trace Fixes: Absolute Path Nesting & Token Explosion

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fix two recurring problems surfaced by the `aecf6210` trace — 1.8M tokens for a single SPECIFY phase, and artifact files written to double-nested absolute paths under virtual_mode. Then leverage the interpreter (RLM pattern) and DA memory system to prevent the problems structurally rather than just adding guardrails.

**Architecture:** Three layers — (1) eliminate the root causes of absolute path nesting, (2) extend context management to all phases, (3) shift the agent operating model from "read everything into context" to "explore in the interpreter, write synthesis to context."

**Tech Stack:** Python 3.12+, LangChain agents, Deep Agents CodeInterpreterMiddleware, MemoryMiddleware, StoreBackend, pytest

---

## Part A: Fix Absolute Path Nesting

### Root Cause Analysis

Under `virtual_mode=True`, `LocalShellBackend._resolve_path` treats any path starting with `/` as a virtual path relative to `cwd`. So `/home/pat/projects/spine/.spine/artifacts/...` resolves to `cwd + /home/pat/projects/spine/.spine/artifacts/...` — a double-nested path.

The agent constructs absolute paths because:
1. Phase system prompts say `Your workspace root is: /home/pat/projects/spine` — the model combines this with the relative artifact path to build absolute paths
2. `tasks.py` line 90 explicitly shows `Full path: /home/pat/projects/spine/.spine/artifacts/.../` — trains the model to use absolute paths
3. Read calls with absolute paths like `/home/pat/projects/spine/spine/models/enums.py` silently work (the double-nested path resolves within the actual filesystem), reinforcing the behavior

---

### Task A1: Remove workspace root from all phase system prompts

**Objective:** Stop telling the model the absolute workspace path so it can't construct absolute paths from it.

**Files:**
- Modify: `spine/agents/specify_agent.py:54`
- Modify: `spine/agents/plan_agent.py:41`
- Modify: `spine/agents/implement_agent.py:52`
- Modify: `spine/agents/tasks_agent.py:86`
- Modify: `spine/agents/verify_agent.py:52`

**Step 1: Replace workspace_root lines in all 5 agent builders**

In each file, replace:
```python
f"Your workspace root is: {workspace_root}\n\n"
```
with:
```python
"Your filesystem is rooted at the project workspace. Use relative paths (e.g. `src/main.py`, `.spine/artifacts/...`).\n\n"
```

Remove the `workspace_root = state.get("workspace_root", ".")` line from each agent builder if no other reference to it remains (it's still used by `build_phase_agent` via `state` directly, not via the local variable).

**Verification:** `grep -r "workspace_root is:" spine/agents/` returns nothing.

---

### Task A2: Remove "Full path:" line from tasks.py prompt

**Objective:** Stop showing the model the absolute path to the artifact directory.

**Files:**
- Modify: `spine/phases/tasks.py:88-90`

**Step 1: Remove the Full path line**

Replace:
```python
f"Write ALL artifact files (slice files AND tasks.md) to: `{tasks_artifact_dir}/`\n"
f"This is relative to your workspace root (`{workspace_root}`).\n"
f"Full path: `{workspace_root}/{tasks_artifact_dir}/`\n",
```
with:
```python
f"Write ALL artifact files (slice files AND tasks.md) to: `{tasks_artifact_dir}/`\n"
f"Use this relative path with `write_file` — do NOT construct absolute paths.\n",
```

---

### Task A3: Add path normalization to the backend wrapper

**Objective:** Defense-in-depth — even if a model constructs an absolute path that matches `workspace_root`, normalize it before passing to `_resolve_path`. This catches the case without modifying DA internals.

**Files:**
- Modify: `spine/agents/backend.py`

**Step 1: Add a normalizing LocalShellBackend subclass**

Override `_resolve_path` via a subclass. Since `LocalShellBackend._resolve_path` is the single chokepoint for all path resolution, we only need to patch one method.

```python
class _NormalizingLocalShellBackend(LocalShellBackend):
    """LocalShellBackend that strips accidental absolute paths before resolution."""

    def _resolve_path(self, key: str) -> Path:
        # If the key starts with our own root_dir (as an absolute path),
        # strip it to get the relative portion before delegating.
        # This prevents double-nesting under virtual_mode.
        if self.virtual_mode and key.startswith(str(self.cwd)):
            key = key[len(str(self.cwd)):].lstrip("/")
        return super()._resolve_path(key)
```

Then in `build_backend`, use `_NormalizingLocalShellBackend` instead of `LocalShellBackend`.

**Step 2: Write test for path normalization**

**File:** `tests/unit/test_backend_normalization.py`

```python
import pytest
from pathlib import Path
from spine.agents.backend import _NormalizingLocalShellBackend


class TestPathNormalization:
    def test_relative_path_unchanged(self, tmp_path):
        backend = _NormalizingLocalShellBackend(
            root_dir=str(tmp_path), virtual_mode=True
        )
        result = backend._resolve_path(".spine/artifacts/test/spec.md")
        assert result == tmp_path / ".spine" / "artifacts" / "test" / "spec.md"

    def test_workspace_root_prefix_stripped(self, tmp_path):
        backend = _NormalizingLocalShellBackend(
            root_dir=str(tmp_path), virtual_mode=True
        )
        absolute = f"{tmp_path}/.spine/artifacts/test/spec.md"
        result = backend._resolve_path(absolute)
        assert result == tmp_path / ".spine" / "artifacts" / "test" / "spec.md"

    def test_non_root_absolute_path(self, tmp_path):
        backend = _NormalizingLocalShellBackend(
            root_dir=str(tmp_path), virtual_mode=True
        )
        # /etc/passwd doesn't start with tmp_path, so it's treated as
        # a virtual path under tmp_path — which is the existing DA behavior
        # (not a security issue since virtual_mode restricts to root)
        result = backend._resolve_path("/etc/passwd")
        assert result == tmp_path / "etc" / "passwd"
```

**Step 3: Run tests**

```bash
pytest tests/unit/test_backend_normalization.py -v
```

---

### Task A4: Strengthen SPINE_FILESYSTEM_PROMPT anti-absolute-path guidance

**Objective:** Make the "never use absolute paths" instruction more prominent and add a concrete example of the failure mode.

**Files:**
- Modify: `spine/agents/factory.py:85-109` (`SPINE_FILESYSTEM_PROMPT`)

**Step 1: Update the path conventions section**

Replace the current path conventions block in `SPINE_FILESYSTEM_PROMPT`:
```python
"**Path conventions (critical):**\n"
"- Use **relative paths** from the workspace root: `.spine/artifacts/file.md`, `src/main.py`.\n"
"- A leading `/` is treated as workspace-relative (e.g. `/src/main.py` → `{root}/src/main.py`).\n"
"- **Never** use full absolute paths like `/home/user/project/...` — they will be double-nested and break.\n"
"- Path traversal (`..`, `~`) is blocked by the virtual filesystem.\n"
"- Use pagination (offset/limit) when reading large files.\n"
```

with:
```python
"**Path conventions (CRITICAL — violations break artifact storage):**\n"
"- Use **relative paths** from the workspace root: `.spine/artifacts/file.md`, `src/main.py`.\n"
"- A leading `/` is treated as workspace-relative (e.g. `/src/main.py` resolves correctly).\n"
"- **NEVER use absolute paths** like `/home/user/project/src/main.py` — the filesystem\n"
"  treats them as virtual paths relative to the workspace root, so they get double-nested\n"
"  and your files end up at the wrong location. Always use `src/main.py` or `/src/main.py`.\n"
"- Path traversal (`..`, `~`) is blocked by the virtual filesystem.\n"
"- Use pagination (offset/limit) when reading large files.\n"
```

Also update `SPINE_FILESYSTEM_EXEC_PROMPT` similarly since it inherits the base.

---

### Task A5: Clean up existing double-nested artifact directories

**Objective:** Remove any stale double-nested directories from past runs.

**Files:** None (filesystem cleanup)

**Step 1: Scan and remove**

```bash
find /home/pat/projects/spine/.spine -type d -path "*/home/pat*" -exec rm -rf {} + 2>/dev/null
```

---

## Part B: Fix Token Explosion

### Root Cause Analysis

The SPECIFY phase for `aecf6210` used 1.8M tokens (1.79M prompt, 51K completion) across 118 tool calls in a single agent run. Key contributors:

1. **No ToolOutputTrimmer for SPECIFY/PLAN** — `factory.py:415` only adds it for TASKS/IMPLEMENT/VERIFY. 79 `read_file` results accumulate in full.
2. **No summarization for SPECIFY** — `specify_agent.py` passes `add_summarization=False`. No context compression happens at all.
3. **Unlimited recursion** — `recursion_limit: 9_999` in the agent config means no effective cap on turns.
4. **Re-reading** — `dispatcher.py` was read 14×, `compose.py` 7×, the agent's own output 7×. The SPINE_BASE_PROMPT says "Never re-read a file in the same phase" but there's no enforcement.

---

### Task B1: Enable ToolOutputTrimmer for all phases

**Objective:** Trim old tool results in SPECIFY and PLAN phases, not just TASKS/IMPLEMENT/VERIFY.

**Files:**
- Modify: `spine/agents/factory.py:415`

**Step 1: Change the phase check**

Replace:
```python
if phase in (PhaseName.TASKS, PhaseName.IMPLEMENT, PhaseName.VERIFY):
```
with:
```python
# All phases benefit from trimming — SPECIFY/PLAN can accumulate 80+ read_file results
if not is_subagent:
```

Remove the outer `if not is_subagent:` check on line 412 since we're now always adding spine middleware for non-subagents. Actually, the whole `_add_spine_middleware` function is already gated on `not is_subagent` (line 372), so we can simplify the inner check.

**Step 2: Run existing tests**

```bash
pytest tests/unit/ -v
```

---

### Task B2: Enable summarization for SPECIFY and PLAN

**Objective:** Add context compression to phases that currently run without any token ceiling.

**Files:**
- Modify: `spine/agents/specify_agent.py:78` (add `add_summarization=True`)
- Modify: `spine/agents/plan_agent.py` (add `add_summarization=True`)

**Step 1: Update specify_agent.py**

In `build_specify_agent`, change:
```python
agent = build_phase_agent(
    state=state,
    config=config,
    phase=PhaseName.SPECIFY,
    system_prompt=system_prompt,
    subagents=_build_subagents(PhaseName.SPECIFY, state, config),
)
```
to:
```python
agent = build_phase_agent(
    state=state,
    config=config,
    phase=PhaseName.SPECIFY,
    system_prompt=system_prompt,
    subagents=_build_subagents(PhaseName.SPECIFY, state, config),
    add_summarization=True,  # SPECIFY can accumulate 80+ tool results
)
```

**Step 2: Update plan_agent.py similarly**

Add `add_summarization=True` to the `build_phase_agent` call in `build_plan_agent`.

---

### Task B3: Reduce recursion_limit from 9,999 to per-phase defaults

**Objective:** Cap agent turns so runaway loops don't burn unlimited tokens.

**Files:**
- Modify: `spine/agents/factory.py:284` (the `.with_config` call)

**Step 1: Add phase-based recursion limits**

Add a mapping near the top of the factory:
```python
# Phase-specific recursion limits.
# SPECIFY/PLAN: exploration + writing, ~30-40 turns max
# TASKS: exploration + decomposition, ~50-60 turns
# IMPLEMENT/VERIFY: multi-slice work, ~80-100 turns
# CRITIC: single review pass, ~20 turns
PHASE_RECURSION_LIMITS: dict[str, int] = {
    PhaseName.SPECIFY.value: 50,
    PhaseName.PLAN.value: 40,
    PhaseName.TASKS.value: 60,
    PhaseName.IMPLEMENT.value: 100,
    PhaseName.VERIFY.value: 80,
    PhaseName.CRITIC.value: 20,
}
```

Then replace the hardcoded `9999`:
```python
"recursion_limit": PHASE_RECURSION_LIMITS.get(phase.value, 9999),
```

**Step 2: Test**

```bash
pytest tests/unit/ -v
```

---

## Part C: Leverage Interpreter (RLM) to Prevent Token Bloat Structurally

### Problem

Parts A and B add guardrails (trimming, summarization, recursion caps) that limit damage. But the fundamental pattern is still "the model reads files into context, one tool call at a time, and every result bloats the conversation." The 79 `read_file` calls in `aecf6210` prove this — even with trimming and summarization, the agent is still doing too many individual reads.

The DA interpreter (RLM pattern) offers a structural fix: **the model writes a small program that reads files, and only the program's compact synthesis returns to the model context.** But SPINE's current interpreter setup doesn't let the agent do filesystem exploration from eval — the PTC allowlist only includes `task` (subagent delegation), not `read_file`/`grep`/`glob`/`ls`.

---

### Task C1: Add filesystem tools to the PTC allowlist for all phases

**Objective:** Allow the agent to explore the codebase from the interpreter, keeping intermediate read results in JS variables instead of the model context.

**Files:**
- Modify: `spine/agents/interpreter.py` (`_PTC_ALLOWLISTS`)

**Step 1: Update PTC allowlists**

The DA docs confirm: "Filesystem access: No — Add the built-in filesystem tools via the PTC allowlist." The tools available are `read_file`, `grep`, `glob`, `ls` (read-only) and `write_file`, `edit_file` (write). For SPINE phases, read-only filesystem PTC makes exploration token-efficient; write operations should still go through the normal tool path so the FilesystemMiddleware prompt validation and the artifact scanning both work.

Replace:
```python
_PTC_ALLOWLISTS: dict[str, list[str | Any]] = {
    PhaseName.SPECIFY.value: ["task"],
    PhaseName.TASKS.value: ["task"],
    PhaseName.IMPLEMENT.value: ["task"],
    PhaseName.VERIFY.value: ["task"],
}
```
with:
```python
# Read-only filesystem tools — safe for PTC because they don't mutate state.
# The agent can batch reads in eval, store results in JS variables, and return
# only the synthesis to the model context. This is the key mechanism for
# keeping context lean during codebase exploration.
_FS_READ_TOOLS: list[str] = ["read_file", "grep", "glob", "ls"]

_PTC_ALLOWLISTS: dict[str, list[str | Any]] = {
    PhaseName.SPECIFY.value: ["task", *_FS_READ_TOOLS],
    PhaseName.TASKS.value: ["task", *_FS_READ_TOOLS],
    PhaseName.IMPLEMENT.value: ["task", *_FS_READ_TOOLS],
    PhaseName.VERIFY.value: ["task", *_FS_READ_TOOLS],
    PhaseName.PLAN.value: _FS_READ_TOOLS,  # PLAN has no subagents but does read files
    # CRITIC — no PTC needed, it reviews artifacts
}
```

**Step 2: Update the interpreter module docstring**

Update the `PTC allowlists per phase` section in the module docstring to document the new tools.

**Step 3: Update the rlm-pattern SKILL.md**

Add a "Pattern 4: Filesystem exploration from eval" section to `spine/skills/rlm-pattern/SKILL.md` showing the agent how to use `tools.readFile`, `tools.grep`, `tools.glob` from eval code. Example:

```js
// Explore the codebase from eval — results stay in JS, not model context
const files = await tools.glob({pattern: 'spine/work/*.py'});
const contents = await Promise.all(
  files.slice(0, 5).map(f => tools.readFile({path: f}))
);
// Build a compact summary
const summary = contents.map((c, i) => `${files[i]}: ${c.substring(0, 100)}...`).join('\n');
summary;
```

---

### Task C2: Seed interpreter state with known context at phase start

**Objective:** Pre-populate the interpreter's `window` object with information the agent will need — artifact paths, work metadata, key file paths — so it doesn't have to discover them through individual tool calls or rely on them being in the system prompt.

**Files:**
- Modify: `spine/agents/interpreter.py` (add initial eval injection)
- Modify: `spine/agents/factory.py` (call the seed function after building middleware)

**Background:** The `CodeInterpreterMiddleware` doesn't have a built-in `initial_state` parameter. But since the interpreter snapshots persist across turns and the first `eval` call can set up `window.*` variables, we can inject a "setup eval" that runs before the agent's first model call. The mechanism: a custom middleware that wraps the first `awrap_model_call` and injects a setup eval.

**Design:**

Create an `InterpreterSeeder` middleware that:
1. On the first `awrap_model_call` of a phase run, executes a setup eval program that seeds `window.context` with known values
2. Sets a flag so it only runs once per agent invocation
3. The seeded data includes:
   - `window.context.workId` — the current work ID
   - `window.context.phase` — the current phase name
   - `window.context.artifactDir` — the base artifact path (e.g. `.spine/artifacts/aecf6210/`)
   - `window.context.artifactPaths` — a map of phase → artifact directory (e.g. `{specify: ".spine/artifacts/aecf6210/specify/"}`)
   - `window.context.workspaceRoot` — workspace root (available in JS but NOT exposed to the model in the prompt)

The model can then access `window.context.artifactPaths.specify` in eval instead of constructing paths. The workspace root is available in the interpreter for programmatic path construction but not in the system prompt where it could train the model to build absolute paths.

**Step 1: Create `spine/agents/interpreter_seeder.py`**

```python
"""Seeds the QuickJS interpreter with phase context before the first model call.

The CodeInterpreterMiddleware has no built-in initial_state parameter. This
middleware detects the first model call of a phase run and executes a setup
eval that populates window.context with known values the agent will need —
artifact paths, work metadata, workspace root.

This keeps the data accessible from eval (for programmatic use) without
leaking it into the system prompt where it could train the model to
construct absolute paths.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)


class InterpreterSeeder(AgentMiddleware):
    """Seeds the interpreter with phase context on the first model call."""

    def __init__(self, context: dict[str, Any]) -> None:
        self._context = context
        self._seeded = False

    async def awrap_model_call(self, request, handler):
        if not self._seeded:
            self._seeded = True
            # The eval tool is available on the agent — we inject a setup
            # message that seeds window.context before the first model call.
            # This is done by appending a tool result that looks like the
            # eval tool was called, which triggers the interpreter setup.
            # Actually, we can't call eval from here directly. Instead, we
            # add a system hint to the first model request.
            #
            # Simpler approach: prepend a HumanMessage with the seed program
            # that the model will see as prior context. But this adds tokens.
            #
            # Best approach: use the _registry directly to eval the setup
            # code. The CodeInterpreterMiddleware stores its registry at
            # self._registry on the middleware stack. We access it through
            # the agent's middleware list.
            pass  # See Step 2 for the actual mechanism

        return await handler(request)
```

**Step 2: Choose the seeding mechanism**

There are three approaches, in order of preference:

**Option A — Seed via the first eval call's result context.** The `CodeInterpreterMiddleware._registry` manages per-thread REPL instances. We can get the REPL for the current thread and call `eval_sync` directly with the seed code. This is clean but requires accessing the registry (private API).

**Option B — Inject a seed instruction in the system prompt.** Add a section to the system prompt that says "On your first turn, run this eval to set up your workspace:" followed by the seed code. This costs ~200 prompt tokens but is simple and doesn't depend on private APIs.

**Option C — Use the `awrap_model_call` hook to modify the first request.** Inject a fake prior message pair (tool_call + tool_result) that simulates the agent having already run the seed eval. This is hacky and fragile.

**Recommendation: Option B for now.** It's the simplest, doesn't rely on private APIs, and 200 tokens is negligible. If we later need to hide the workspace root from the prompt entirely, we can migrate to Option A.

**Step 3: Implement Option B — seed instruction in the system prompt**

In `build_phase_agent`, after assembling the system prompt, append a section:

```python
# Interpreter seed — pre-populate window.context
if has_interpreter:
    context_json = json.dumps({
        "workId": work_id,
        "phase": phase.value,
        "artifactDir": f".spine/artifacts/{work_id}/",
        "artifactPaths": {
            p.value: f".spine/artifacts/{work_id}/{p.value}/"
            for p in PhaseName
        },
    })
    seed_block = (
        "\n## Interpreter Setup\n"
        "On your FIRST turn, run this eval to set up your workspace context:\n"
        "```js\n"
        f"window.context = {context_json};\n"
        "console.log('Context seeded:', Object.keys(window.context));\n"
        "```\n"
        "Then use `window.context.artifactPaths.specify` (etc.) in subsequent "
        "eval calls instead of constructing paths manually.\n"
    )
    final_system_prompt += seed_block
```

This gives the agent programmatic access to all artifact paths and the work ID without leaking the absolute workspace root into the prompt. The model runs the seed eval once, then references `window.context.*` in all subsequent eval calls.

**Step 4: Update the rlm-pattern SKILL.md**

Add guidance about using `window.context`:

```markdown
## Pattern: Use window.context for paths

At the start of each phase, the interpreter is seeded with `window.context` containing:
- `workId` — the current work item ID
- `phase` — the current phase name
- `artifactDir` — base artifact directory (relative path)
- `artifactPaths` — map of phase → artifact directory

Use these instead of constructing paths manually:

\```js
const specPath = window.context.artifactPaths.specify + 'specification.md';
const content = await tools.readFile({path: specPath});
\```
```

---

### Task C3: Direct subagents to return structured summaries, not raw data

**Objective:** Change researcher subagent prompts to return focused findings (summary, patterns, file map) rather than raw file contents, and change parent-phase prompts to instruct the agent to have subagents process data rather than the parent re-reading.

**Files:**
- Modify: `spine/agents/subagents.py` (researcher prompt + response model)
- Modify: `spine/skills/rlm-pattern/SKILL.md` (add subagent direction guidance)

**Step 1: Strengthen the researcher subagent prompt**

The current researcher prompt already says "Be concise — your output will be consumed by the specification writer." But it doesn't explicitly tell the researcher to synthesize rather than quote. Update `SUBAGENT_PROMPTS["researcher"]`:

Add after "IMPORTANT: You are read-only. Do not modify any files.":
```
"SYNTHESIZE, don't quote. Return your analysis and conclusions — the parent agent\n"
"does NOT need raw file contents. If the parent needs to see a specific section,\n"
"it can read the file itself. Your job is to save the parent from reading.\n\n"
```

**Step 2: Update the rlm-pattern SKILL.md subagent dispatch section**

In Pattern 3 (Codebase exploration), add guidance about what to ask subagents for:

```markdown
## Subagent direction — ask for synthesis, not data

When dispatching researcher subagents, ask them to return:
- **Summary**: What does this module do? What patterns does it follow?
- **Relevant interfaces**: Key classes, functions, and their signatures
- **Dependencies**: What it imports and what imports it
- **Conventions**: Naming, error handling, testing patterns

Do NOT ask subagents to "return the full contents of..." — they should analyze
and summarize. If you need specific details, read just that file yourself.
```

**Step 3: Enforce structured output from the researcher**

The researcher already has a `ResearchFindings` response model with `summary`, `patterns`, `file_map`, `dependencies`. Verify that this is actually being applied as `response_format` on the subagent spec. Check `build_subagent_spec` in `subagents.py` — the spec dict should include `"response_format": ResearchFindings` for the researcher.

Currently the spec does NOT include response_format. Add it:

```python
# In build_subagent_spec, after building the spec dict:
response_model = SUBAGENT_RESPONSE_MODELS.get(name)
if response_model:
    spec["response_format"] = response_model
```

---

### Task C4: Pre-seed the DA memory store with project knowledge

**Objective:** Use the DA `MemoryMiddleware` + `StoreBackend` to persist project knowledge across work items, so agents don't have to re-discover the same codebase conventions on every run.

**Files:**
- Modify: `spine/agents/backend.py` (add project knowledge seeding)
- Modify: `spine/agents/skills_resolver.py` (add `/memories/` to memory paths)

**Background:**

Currently, the cross-work memory store (`InMemoryStore`) is empty — no agent has ever written to `/memories/`. The `resolve_memory` function loads `AGENTS.md` from the filesystem, but that's loaded into the system prompt every turn (costing ~5K tokens). The DA memory system supports a more efficient pattern: store project knowledge in the `StoreBackend` once, then agents read it on demand via `edit_file` / `read_file` to `/memories/` rather than having it injected into every turn.

**What to pre-seed:**

The key insight from the memory docs is: "Populate the store with initial memories, then invoke the agent." We should seed:

1. **`/memories/project-structure.md`** — A compact map of the project's directory layout, key modules, and their roles. This is the "codebase map" that the tasks phase already produces as an artifact — but it should be available from the start of every run, not just after tasks completes.

2. **`/memories/conventions.md`** — Project coding conventions extracted from `AGENTS.md`. This is the ~22K char file that's currently injected every turn. By putting it in the store, agents can read it on demand instead of carrying it in the system prompt.

**Step 1: Create a memory seeding function**

In `spine/agents/backend.py`, add:

```python
def seed_project_memory(workspace_root: str) -> None:
    """Seed the cross-work memory store with project knowledge.

    Called once at worker startup (or on first work submission) so that
    all subsequent agent runs can read from /memories/ instead of
    having large context injected every turn.
    """
    from deepagents.backends.utils import create_file_data

    store = _get_store()
    root = Path(workspace_root)
    namespace = ("spine-project",)

    # Seed project structure from AGENTS.md if available
    agents_md = root / "AGENTS.md"
    if agents_md.exists():
        content = agents_md.read_text(encoding="utf-8")
        # Extract just the structure/conventions sections (not the full 22K)
        # For now, store the whole thing — agents can read selectively
        store.put(
            namespace,
            "/memories/conventions.md",
            create_file_data(content),
        )
        logger.info("Seeded project conventions to memory store (%d chars)", len(content))

    # Seed project structure
    structure = _build_project_structure(root)
    if structure:
        store.put(
            namespace,
            "/memories/project-structure.md",
            create_file_data(structure),
        )
        logger.info("Seeded project structure to memory store")


def _build_project_structure(root: Path) -> str:
    """Build a compact project structure map."""
    lines = ["# Project Structure\n"]
    # Walk top-level and one level deep
    for item in sorted(root.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir():
            lines.append(f"\n## {item.name}/")
            for child in sorted(item.iterdir())[:20]:
                if child.name.startswith(".") or child.name == "__pycache__":
                    continue
                prefix = "  " if child.is_file() else "  📁"
                lines.append(f"{prefix} {child.name}")
        else:
            lines.append(f"- {item.name}")
    return "\n".join(lines)
```

**Step 2: Call seeding at the right time**

Call `seed_project_memory(workspace_root)` from `submit_work()` in `spine/work/dispatcher.py` before the first graph invocation for a given workspace. Add a guard so it only seeds once per process:

```python
_seeded_workspaces: set[str] = set()

def _ensure_memory_seeded(workspace_root: str) -> None:
    if workspace_root not in _seeded_workspaces:
        seed_project_memory(workspace_root)
        _seeded_workspaces.add(workspace_root)
```

**Step 3: Add `/memories/` paths to the memory resolver**

In `spine/agents/skills_resolver.py`, update `resolve_memory` to include `/memories/conventions.md` and `/memories/project-structure.md` in the memory list. Since the MemoryMiddleware loads these as on-demand references (not always-injected), they cost zero tokens until the agent reads them.

Actually, per the DA docs, `memory=` paths are "always injected" into the system prompt, while `skills=` paths are "progressive disclosure" (loaded on demand). So putting `/memories/` paths in `memory=` would still inject them every turn. Instead:

**Better approach:** Don't add `/memories/` to the `memory=` list. Instead, let agents discover and read from `/memories/` naturally via the filesystem tools — just like they read from `.spine/artifacts/`. The store is already routed through the `CompositeBackend` at `/memories/`, so `read_file("/memories/conventions.md")` works. The key change is:

1. Remove `AGENTS.md` from the `memory=` list (saving ~5K tokens/turn)
2. Seed its content into `/memories/conventions.md` in the store
3. Add a line to the phase system prompt: "Project conventions are at `/memories/conventions.md` — read if needed."

This way, conventions are available on demand but don't cost tokens unless the agent actually needs them.

**Step 4: Update `resolve_memory` to skip AGENTS.md for all phases**

Change `resolve_memory` to no longer load the project root `AGENTS.md` into the always-injected memory list. Instead, it's seeded in the store:

```python
def resolve_memory(
    workspace_root: str | None = None,
    phase: str | None = None,
) -> list[str]:
    if not workspace_root:
        return []

    memory_paths: list[str] = []
    root = Path(workspace_root)

    # Project root AGENTS.md — NO LONGER injected every turn.
    # Instead, it's seeded into /memories/conventions.md by
    # seed_project_memory() and available on demand via read_file.
    # This saves ~5K tokens per turn across all phases.

    # .spine/AGENTS.md — SPINE-specific conventions for this project.
    # Typically small (<1K) so still injected.
    spine_agents = root / ".spine" / "AGENTS.md"
    if spine_agents.exists():
        memory_paths.append(str(spine_agents))

    return memory_paths
```

**Step 5: Add memory path reference to phase system prompts**

In each phase agent builder, add a line to the system prompt:

```python
"Project conventions are available at `/memories/conventions.md` — "
"read with `read_file` or `eval` only if you need them.\n\n"
```

---

## Implementation Order

1. **A1–A5** — Fix absolute path nesting (immediate, high confidence)
2. **B1–B3** — Extend context management to all phases (immediate, high confidence)
3. **C1** — Add filesystem tools to PTC allowlist (enables the RLM pattern properly)
4. **C2** — Seed interpreter state (builds on C1)
5. **C3** — Direct subagents to return summaries (prompt-level change)
6. **C4** — Pre-seed memory store (largest change, do last after C1–C3 are proven)

Each part is independently shippable — Parts A and B don't depend on C, and each C task can ship on its own.
