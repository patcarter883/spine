"""SPINE subagent definitions and factory.

Defines the three custom subagents SPINE uses for parallel delegation:

- **researcher** (SPECIFY): read-only codebase investigation
- **slice-implementer** (IMPLEMENT): single feature slice implementation
- **slice-verifier** (VERIFY): single feature slice verification

Each subagent is built as a dictionary matching the Deep Agents ``SubAgent``
spec — with custom system prompts, tool restrictions, model overrides, and
structured response formats.  The factory functions here resolve model,
memory, and skill configuration from the same sources as the parent phase
agents (``SpineConfig`` + ``RunnableConfig``), so there is no duplication.

Subagents are intentionally built as dict specs (not ``CompiledSubAgent``)
because they are leaf agents with no internal multi-step workflow.  If a
subagent later needs its own state machine (e.g. a multi-step
verify-analyse-retry loop), it can be upgraded to ``CompiledSubAgent``
wrapping a compiled LangGraph graph — but that complexity is deferred until
real usage justifies it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

from pydantic import BaseModel, Field

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)


# ── Response format models ─────────────────────────────────────────────
# These Pydantic models are passed as ``response_format`` on the SubAgent
# dict.  DA (≥0.5.3) will parse the subagent's final message into the
# schema and return JSON to the parent — no free-text parsing needed.


class ResearchFindings(BaseModel):
    """Structured output from the researcher subagent."""

    summary: str = Field(description="Concise summary of findings (2-3 paragraphs)")
    patterns: list[str] = Field(description="Notable patterns, conventions, or idioms discovered")
    file_map: dict[str, str] = Field(
        description="Mapping of important file paths to brief descriptions"
    )
    dependencies: list[str] = Field(
        description="Key dependencies, imports, or external services found"
    )


class SliceResult(BaseModel):
    """Structured output from the slice-implementer subagent."""

    status: str = Field(description="One of: implemented, partial, blocked")
    files_modified: list[str] = Field(description="Files that were modified")
    files_created: list[str] = Field(description="Files that were created")
    test_results: str = Field(description="Summary of test/lint outcomes for this slice")
    issues: list[str] = Field(description="Unresolved issues or blockers (empty if none)")


class CheckItem(BaseModel):
    """A single verification checklist item."""

    criterion: str = Field(description="What was checked")
    passed: bool = Field(description="Whether it passed")
    detail: str = Field(description="Brief evidence or explanation")


class VerificationResult(BaseModel):
    """Structured output from the slice-verifier subagent."""

    verdict: str = Field(description="One of: VERIFIED, NOT_VERIFIED")
    checklist: list[CheckItem] = Field(description="Verification checklist results")
    gaps: list[str] = Field(description="Gaps or missing items found (empty if none)")
    recommendations: list[str] = Field(description="Suggested improvements (empty if none)")


# ── Subagent descriptions ──────────────────────────────────────────────

SUBAGENT_DESCRIPTIONS: dict[str, str] = {
    "researcher": (
        "Investigates specific areas of the codebase in read-only mode. "
        "Use to research dependencies, patterns, conventions, and file "
        "structure before writing specifications. Delegates parallel "
        "research tasks to reduce context pressure on the main agent."
    ),
    "slice-implementer": (
        "Implements a single feature slice: writes code, runs tests and "
        "linters, fixes errors. Use for each slice in the implementation "
        "wave. Returns a structured result with files changed and issues."
    ),
    "slice-verifier": (
        "Verifies a single feature slice against its acceptance criteria: "
        "inspects implementation, runs tests and linters, reports verdict. "
        "Use to verify each slice independently. Returns a structured "
        "verification result — does not fix issues."
    ),
}

# ── Subagent system prompts ────────────────────────────────────────────

SUBAGENT_PROMPTS: dict[str, str] = {
    "researcher": (
        "YOU MUST USE TOOLS. Do not produce a report from memory or speculation.\n"
        "You are a codebase researcher. Your job is to investigate the area "
        "of the codebase described in the task and report back with structured "
        "findings.\n\n"
        "## Codebase navigation — USE MCP TOOLS FIRST\n"
        "You have MCP codebase index tools for efficient structural navigation. "
        "These answer symbol-level questions in sub-millisecond time with "
        "minimal token usage — ALWAYS use them before reading files.\n\n"
        "### Batching and Parallel Tool Calling (CRITICAL)\n"
        "- **NEVER query symbols, function sources, or files sequentially turn-by-turn.** "
        "If you need to look up 3 function sources or search multiple directories, generate "
        "all those tool calls in parallel in a SINGLE response. This is highly efficient and avoids wasting turns.\n"
        "- **Plan your queries:** In your first turn, identify 3-5 files or functions of interest, and "
        "issue ALL lookup/search calls in parallel, instead of requesting them one-by-one sequentially.\n\n"
        "| Question | Tool to use |\n"
        "|----------|-------------|\n"
        "| Where is X defined? | `mcp_codebase-index_find_symbol` |\n"
        "| What does function X call? | `mcp_codebase-index_get_dependencies` |\n"
        "| Who calls function X? | `mcp_codebase-index_get_dependents` |\n"
        "| What breaks if I change X? | `mcp_codebase-index_get_change_impact` |\n"
        "| How does X connect to Y? | `mcp_codebase-index_get_call_chain` |\n"
        "| Show me function X's code | `mcp_codebase-index_get_function_source` |\n"
        "| What files match pattern X? | `mcp_codebase-index_list_files` |\n"
        "| Search for pattern X everywhere | `mcp_codebase-index_search_codebase` |\n"
        "| High-level project overview | `mcp_codebase-index_get_project_summary` |\n\n"
        "## Tool surface\n"
        "### Primary (use these FIRST)\n"
        "- `mcp_codebase-index_find_symbol` — locate symbol definition (file, line, type). "
        'Call with `{"name": "symbol_name"}`.\n'
        "- `mcp_codebase-index_get_function_source` — get full function source. "
        'Call with `{"name": "func_name"}`.\n'
        "- `mcp_codebase-index_get_dependencies` — what a symbol calls/uses. "
        'Call with `{"name": "symbol_name"}`.\n'
        "- `mcp_codebase-index_get_dependents` — what calls/uses a symbol. "
        'Call with `{"name": "symbol_name"}`.\n'
        "- `mcp_codebase-index_get_change_impact` — direct + transitive dependents. "
        'Call with `{"name": "symbol_name"}`.\n'
        "- `mcp_codebase-index_get_call_chain` — BFS path between two symbols. "
        'Call with `{"from_name": "A", "to_name": "B"}`.\n'
        "- `mcp_codebase-index_search_codebase` — regex search across all files. "
        'Call with `{"pattern": "regex", "max_results": 20}`.\n'
        "- `mcp_codebase-index_list_files` — list files matching a glob pattern. "
        'Call with `{"pattern": "*.py"}`.\n'
        "- `mcp_codebase-index_get_project_summary` — file count, packages, top symbols. "
        "No arguments needed.\n"
        "- `mcp_codebase-index_get_classes` — list classes with methods and bases. "
        'Call with `{"file_path": "path/to/file.py"}` or no args for all.\n'
        "- `mcp_codebase-index_get_functions` — list functions with params. "
        'Call with `{"file_path": "path/to/file.py"}` or no args for all.\n'
        "- `mcp_codebase-index_get_imports` — list imports for a file. "
        'Call with `{"file_path": "path/to/file.py"}` or no args for all.\n\n'
        "### Fallback (only when MCP tools don't have what you need)\n"
        "- `search_codebase` — multi-query keyword file search with content previews\n"
        "- `read_file` — read specific files (use offset/limit for large files)\n"
        "- `ls`, `glob`, `grep` — traditional filesystem tools for targeted lookups\n\n"
        "## Path conventions (CRITICAL)\n"
        "All paths MUST be relative from the project workspace root:\n"
        "- `.spine/artifacts/file.md`, `spine/ui/pages.py`\n"
        "- A leading `/` is workspace-relative.\n"
        "- **NEVER** use absolute paths like `/home/user/project/...` — they "
        "double-nest under the virtual filesystem root and resolve to non-existent files.\n\n"
        "## Research workflow (3-5 turns)\n"
        "1. **MCP structural search (1 turn):** Call `mcp_codebase-index_get_project_summary` "
        "for orientation, then `mcp_codebase-index_find_symbol` to locate key symbols "
        "mentioned in the task. Use `mcp_codebase-index_get_dependencies` to trace "
        "relationships. This replaces 5-10 sequential read_file calls and saves tokens.\n"
        "2. **Targeted MCP follow-up (1 turn):** Call `mcp_codebase-index_get_function_source` "
        "to see the actual code of key symbols. Use `mcp_codebase-index_get_dependents` if "
        "you need to know what calls a particular function.\n"
        "3. **Fallback search only if needed (0-1 turns):** If you need content-level "
        "pattern matching the MCP tools don't provide, use `search_codebase` or `grep`. "
        "This should be RARE — MCP tools cover 90% of research needs.\n"
        "4. **Synthesize (1 turn):** Report findings. Do NOT include raw file contents — "
        "summarize key facts, signatures, conventions, and patterns.\n\n"
        "## Hard limits\n"
        "- You MUST call at least 2 MCP tools before falling back to read_file.\n"
        "- Your file_map MUST contain at least 1 entry.\n"
        "- Your summary MUST be at least 2 sentences.\n"
        "- Total turns: 3-5. More than 5 calls without producing "
        "output means you're over-exploring — report what you have.\n"
        "- If you cannot read files (tool errors, permission issues), report that "
        "with the error details — do NOT return empty results.\n"
        "- If you produce empty results, you WILL be re-dispatched, wasting time "
        "and tokens. Do the work correctly the first time.\n\n"
        "## Output format\n"
        "Your output MUST follow the ResearchFindings schema:\n"
        "- summary: concise paragraph summarizing findings\n"
        "- patterns: notable patterns, conventions, or idioms discovered\n"
        "- file_map: mapping of important file paths to brief descriptions\n"
        "- dependencies: key dependencies, imports, or external services\n"
    ),
    "slice-implementer": (
        "YOU MUST USE TOOLS. Do not describe changes — make them with "
        "write_file and edit_file, then verify with execute.\n"
        "You are a code implementer for a single feature slice. "
        "Your task description contains the full slice definition, "
        "codebase context, modification targets with exact line ranges, "
        "and files to modify — read all of it carefully before starting.\n\n"
        "## Path conventions (CRITICAL)\n"
        "All file paths MUST be relative from the project workspace root:\n"
        "- **Correct:** `spine/ui/pages.py`, `.spine/artifacts/doc.md`\n"
        "- **Correct:** `/spine/ui/pages.py` (leading `/` = workspace-relative)\n"
        "- **WRONG:** `/home/pat/Projects/spine/spine/ui/pages.py` — absolute paths "
        "double-nest under the virtual filesystem and create files in the wrong location.\n"
        "- **WRONG:** `../other/file.py` — traversal blocked by virtual filesystem.\n"
        "- Use `search_codebase` to verify a file exists before writing to its "
        "parent directory. Do not invent paths.\n\n"
        "## Tool surface\n"
        "- `search_codebase` — multi-query file search with content previews. "
        "Use this FIRST to understand existing code structure before making changes.\n"
        "- `read_file` — read files (use offset/limit for pagination).\n"
        "- `write_file` — create or overwrite a file.\n"
        "- `edit_file` — find-and-replace within a file (use replace_all for all occurrences).\n"
        "- `ls`, `glob`, `grep` — directory listing and text search.\n"
        "- `execute` — run shell commands (tests, linters, builds).\n\n"
        "## Implementation workflow (4-6 turns)\n"
        "1. **Read existing code (1 turn, batch):** Read ≥3 files in a single "
        "turn — the files listed in your task description, plus any imports or "
        "dependencies they reference. Check the modification targets in your "
        "task description for exact change sites.\n"
        "2. **Search if needed (0-1 turns):** If you need to understand code "
        "not covered by the task description, call `search_codebase` with "
        "specific queries. Do NOT explore broadly.\n"
        "3. **Make changes (1-2 turns, batch):** Apply all edits. Read-before-write. "
        "Write/edit ≥2 files in a single turn where possible. Focus only on "
        "files listed in the slice — do NOT modify or create files outside its scope.\n"
        "4. **Test (1 turn):** Run the tests listed in your task description's "
        "acceptance criteria. Run linters (ruff) on the files you changed.\n"
        "5. **Fix if needed (0-1 turns):** If tests fail, fix and re-test.\n"
        "6. **Report (final turn):** Return the SliceResult with exactly what "
        "you changed and any remaining issues.\n\n"
        "## Hard limits\n"
        "- **Batch reads:** Always read ≥3 files per turn or use `search_codebase` "
        "instead. Never read one file at a time — it wastes turns and bloat context.\n"
        "- **Exploration budget:** Maximum 3 turns of read/search before your first "
        "write/edit. If you haven't changed code by turn 4, you're over-exploring — "
        "make changes with what you know.\n"
        "- **Scope:** Modify ONLY files listed in the slice. Do not touch files "
        "outside its scope even if you think they need changes — report them as issues.\n"
        "- **Stuck?** If you are blocked by a missing dependency from another slice, "
        "set status='blocked' with the dependency name in issues. Do not try to "
        "implement the dependency yourself.\n"
        "- **Silence is failure:** If you make zero file changes and set "
        "status='implemented', the orchestrator will not know the slice was skipped. "
        "Always report exactly what files you changed.\n\n"
        "## Output\n"
        "End with this exact JSON structure (no markdown wrapping, no backticks):\n"
        '{"status": "implemented|partial|blocked", '
        '"files_modified": ["path1", "path2"], '
        '"files_created": ["path3"], '
        '"test_results": "summary of test/lint outcomes", '
        '"issues": ["any remaining issues or empty list"]}\n'
    ),
    "slice-verifier": (
        "YOU MUST USE TOOLS. Do not verify from memory — inspect actual files "
        "and run actual tests.\n"
        "You are a verification engineer. Your task description contains "
        "the full slice definition, acceptance criteria, and files to verify — "
        "read it carefully before starting.\n\n"
        "Guidelines:\n"
        "1. Review the slice definition and acceptance criteria from your "
        "task description.\n"
        "2. Inspect the implemented files — use read_file and ls "
        "(batch reads — ≥3 files per turn).\n"
        "3. Run relevant tests and linters.\n"
        "4. Check each acceptance criterion individually.\n"
        "5. Produce a structured verification report.\n\n"
        "IMPORTANT: You are report-only. Do not fix issues you find.\n"
        "If a test fails or a criterion is not met, record it in the "
        "checklist and gaps. Your job is to provide evidence, not to repair.\n\n"
        "End with a structured verification result:\n"
        "```json\n"
        '{"verdict": "VERIFIED|NOT_VERIFIED", "checklist": '
        '[{"criterion": "...", "passed": true, "detail": "..."}], '
        '"gaps": [...], "recommendations": [...]}\n'
        "```\n"
    ),
}

# ── Response format mapping ───────────────────────────────────────────

SUBAGENT_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "researcher": ResearchFindings,
    "slice-implementer": SliceResult,
    "slice-verifier": VerificationResult,
}

# ── Tool restrictions ──────────────────────────────────────────────────
# These are imported from the Deep Agents FilesystemMiddleware — they are
# the tool *names*, not callables.  When ``tools`` is specified on a
# SubAgent dict, it entirely overrides the parent's tool set, so we must
# list every tool the subagent should have access to.

_READ_ONLY_TOOLS: list[str] = [
    "ls",
    "read_file",
    "glob",
    "grep",
    "search_codebase",
]

_FULL_TOOLS: list[str] = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "search_codebase",
    "execute",
]

_READ_AND_EXECUTE_TOOLS: list[str] = [
    "ls",
    "read_file",
    "glob",
    "grep",
    "search_codebase",
    "execute",
]

SUBAGENT_TOOLS: dict[str, list[str]] = {
    "researcher": _READ_ONLY_TOOLS,
    "slice-implementer": _FULL_TOOLS,
    "slice-verifier": _READ_AND_EXECUTE_TOOLS,
}

# ── Re-research prompt for empty researcher results ────────────────────
# When a researcher subagent returns empty results (no file_map, no patterns),
# re-dispatch with this suffix appended to the task description.

_RE_RESEARCH_PROMPT_SUFFIX = (
    "\n\n⚠ RE-DISPATCH: A previous researcher returned empty results for this "
    "task. This is your second chance. You MUST:\n"
    "1. Read at least 3 files relevant to the task description.\n"
    "2. Produce a file_map with at least 2 entries.\n"
    "3. If files cannot be found, explain what you searched and what went wrong.\n"
    "Do NOT return empty results again."
)

# ── Which phases use which subagents ────────────────────────────────────

PHASE_SUBAGENTS: dict[str, list[str]] = {
    PhaseName.SPECIFY.value: ["researcher"],
    PhaseName.PLAN.value: ["researcher"],
    PhaseName.TASKS.value: ["researcher"],
    PhaseName.IMPLEMENT.value: [
        "slice-implementer"
    ],  # No verifier — dedicated verify phase is authoritative
    PhaseName.VERIFY.value: ["slice-verifier"],
    # CRITIC — single-agent, no subagents
}


# ── Factory functions ──────────────────────────────────────────────────


def _inject_mcp_tools(
    tools: list, workspace_root: str, *, subagent_name: str = "researcher"
) -> None:
    """Inject MCP codebase-index tools into a subagent's tool list.

    Loads MCP tools from the SpineConfig, wrapping each as a LangChain
    ``BaseTool``.  Errors during MCP tool loading are logged and swallowed
    — subagents fall back to filesystem tools if MCP is unavailable.

    The cache_key now includes the subagent name (researcher vs slice-implementer)
    so the two subagents do not collide in the tool cache and both receive
    the full MCP index surface — critical for preventing the 11× redundant
    `dispatcher.py` reads seen in trace 019e486e.

    Args:
        tools: The subagent's tool list (mutated in place).
        workspace_root: Project root for PROJECT_ROOT env injection.
        subagent_name: Calling subagent name (used for cache key + debug logs).
    """
    try:
        from spine.config import SpineConfig
        from spine.mcp.client import get_mcp_tools

        config = SpineConfig.load()
        cache_key = f"subagent-{subagent_name}-{workspace_root}"
        mcp_tools = get_mcp_tools(
            config.mcp_servers,
            cache_key=cache_key,
            workspace_root=workspace_root,
        )
        tools.extend(mcp_tools)
        logger.debug("Injected %d MCP tools into %s subagent", len(mcp_tools), subagent_name)
    except Exception:
        logger.debug("MCP tool injection skipped for %s subagent", subagent_name, exc_info=True)


# ── Model capability guards ────────────────────────────────────────────

# Model name patterns for models that reject tool_choice="any"/"required"
# when their thinking/reasoning mode is active.  These models crash with
# HTTP 400 when create_agent forces tool_choice for structured output.
_THINKING_MODEL_PATTERNS: tuple[str, ...] = (
    "qwen3",  # Qwen 3.x series (thinking mode enabled by default)
    "qwq",  # QwQ reasoning model
    "deepseek-r",  # DeepSeek-R1 reasoning
)


def _extract_model_name(model: Any) -> str:
    """Extract a lowercase model name string from a model spec or instance.

    Handles:
    - String specs: ``"openrouter:qwen/qwen3-235b-a22b:free"`` → ``"qwen/qwen3-235b-a22b"``
    - String specs without org: ``"openai:gpt-4o"`` → ``"gpt-4o"``
    - Pre-built instances with ``.model`` attr (ChatOpenRouter, ChatOpenAI)
    - Pre-built instances with ``.model_name`` attr (ChatAnthropic)

    Returns:
        Lowercase model name with provider prefix and quality suffix stripped.
    """
    raw: str = ""
    if isinstance(model, str):
        raw = model
    elif hasattr(model, "model_name"):
        raw = str(model.model_name)
    elif hasattr(model, "model"):
        raw = str(model.model)

    # Strip provider prefix (e.g. "openrouter:" or "openai:")
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    # Strip trailing quality suffix (:free, :beta, etc.) — these appear
    # after the model name and contain only short alpha-only strings.
    # Model names like "qwen3-235b-a22b" contain hyphens/digits, so a
    # trailing segment with only letters + digits/periods is a quality tag.
    while ":" in raw:
        last_part = raw.rsplit(":", 1)[1]
        # If the last segment contains no "/" and looks like a quality tag
        # (short, no hyphens suggesting model version), strip it.
        if "/" not in last_part and not any(c == "-" for c in last_part):
            raw = raw.rsplit(":", 1)[0]
        else:
            break

    return raw.lower()


def _supports_forced_tool_choice(model: Any) -> bool:
    """Check whether the model supports ``tool_choice="any"`` / ``"required"``.

    DA's structured output (``ToolStrategy``) forces ``tool_choice="any"``
    to guarantee the model calls the extraction tool.  Some models —
    notably Qwen 3.x and QwQ in thinking/reasoning mode — reject this
    parameter with a 400 error, crashing every researcher subagent.

    When this returns ``False``, the caller should skip ``response_format``
    and rely on prompt-based structured output instead.

    Args:
        model: A model string (``"openrouter:qwen/qwen3-235b-a22b"``) or
            a pre-built ``BaseChatModel`` instance.

    Returns:
        ``True`` if the model supports forced tool choice, ``False`` if
        the model is known to reject it.
    """
    name = _extract_model_name(model)
    return not any(pattern in name for pattern in _THINKING_MODEL_PATTERNS)


def build_subagent_spec(
    name: str,
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None = None,
    *,
    extra_skills: list[str] | None = None,
) -> dict[str, Any]:
    """Build a dict SubAgent spec for a SPINE subagent.

    Resolves model, memory, and skills from the same sources as the parent
    phase agent so there is no duplication.  The returned dict is ready to
    pass directly into ``create_deep_agent(subagents=[...])``.

    Args:
        name: Subagent name (e.g. ``"researcher"``).
        phase: The parent phase (used for model resolution fallback).
        state: The current workflow state.
        config: LangGraph runtime config.
        extra_skills: Additional skill directories to load for this subagent.
            These are merged on top of any default skills the subagent type
            would normally receive.

    Returns:
        A dictionary matching the DA ``SubAgent`` spec.

    Raises:
        ValueError: If the subagent name is not recognised.
    """
    if name not in SUBAGENT_DESCRIPTIONS:
        raise ValueError(f"Unknown subagent {name!r}. Available: {sorted(SUBAGENT_DESCRIPTIONS)}")

    from spine.agents.helpers import resolve_model
    from spine.agents.skills_resolver import resolve_memory

    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "")

    # ── Model: phase/subagents/<name> → phase → default ──────────
    # Pass session_id so OpenRouter models get a pre-built ChatOpenRouter
    # instance with request_timeout set. Without this, the model string
    # goes through init_chat_model() which creates a ChatOpenRouter
    # WITHOUT request_timeout — hung API connections then block the
    # workflow indefinitely (no 5-minute timeout).
    model_path = f"{phase.value}/subagents/{name}"
    model = resolve_model(config, session_id=work_id, phase=model_path)

    # ── Memory: include project AGENTS.md (skip for TASKS/CRITIC phases) ──
    memory = resolve_memory(workspace_root, phase=phase.value)

    # ── Skills: default per subagent type + any extra ─────────────
    skills = _resolve_subagent_skills(name, phase, workspace_root, extra_skills)

    # ── Tools: resolve string names to actual BaseTool instances ──
    # The SubAgent spec requires actual tool instances (BaseTool | Callable | dict),
    # not string names. We use the shared build_backend() for consistency,
    # then create FilesystemMiddleware to get the filesystem tools.
    # IMPORTANT: Use SPINE_FILESYSTEM_PROMPT to avoid the "All file paths must
    # start with a /" default that conflicts with virtual_mode=True.
    from deepagents.middleware.filesystem import FilesystemMiddleware

    from spine.agents.backend import build_backend
    from spine.agents.factory import SPINE_FILESYSTEM_PROMPT, SPINE_FILESYSTEM_EXEC_PROMPT
    from spine.agents.plan_tools import SearchCodebaseTool

    backend = build_backend(workspace_root)
    # Choose prompt based on whether this subagent has execute tool access
    subagent_tool_names = SUBAGENT_TOOLS.get(name, [])
    has_execute = "execute" in subagent_tool_names
    fs_prompt = SPINE_FILESYSTEM_EXEC_PROMPT if has_execute else SPINE_FILESYSTEM_PROMPT
    fs_mw = FilesystemMiddleware(backend=backend, system_prompt=fs_prompt)
    allowed_tool_names = SUBAGENT_TOOLS[name]
    tools = [t for t in fs_mw.tools if t.name in allowed_tool_names]

    # ── Inject SearchCodebaseTool for subagents that have it listed ──
    # search_codebase is a standalone BaseTool from plan_tools, not part of
    # FilesystemMiddleware. It must be added separately so the subagent can
    # use multi-query codebase search instead of sequential ls/glob/grep/read_file.
    if "search_codebase" in allowed_tool_names:
        tools.append(SearchCodebaseTool(workspace_root=workspace_root))

    # ── Inject MCP codebase-index tools for researcher + slice-implementer ─
    # Both subagents benefit enormously from structural queries instead of
    # full-file reads (see trace 019e486e: 11× redundant dispatcher.py reads
    # during implement).  MCP gives "where is X / who calls X" in milliseconds.
    # slice-verifier is read+execute only and rarely needs deep navigation.
    if name in ("researcher", "slice-implementer"):
        _inject_mcp_tools(tools, workspace_root, subagent_name=name)

    # ── Build spec ───────────────────────────────────────────────
    # ToolOutputTrimmer REMOVED from the entire SPINE stack (2026-05 directive).
    # It was aggressively discarding full file-read & execute outputs that
    # leaf sub-agents (slice-implementer / slice-verifier) critically needed,
    # directly causing the 40+:1 prompt:completion ratios seen in traces
    # 019e486e… and 019e488f….
    # Context control now comes ONLY from tight system prompts + read cache.
    subagent_middleware: list[Any] = []

    spec: dict[str, Any] = {
        "name": name,
        "description": SUBAGENT_DESCRIPTIONS[name],
        "system_prompt": SUBAGENT_PROMPTS[name],
        "model": model,
        "tools": tools,
        "middleware": subagent_middleware,
    }
    # Only researcher gets response_format — structured summaries prevent
    # raw file contents from bloating the parent agent's context.
    # slice-implementer and slice-verifier need free-form tool use.
    #
    # Guard: DA's create_agent resolves AutoStrategy → ToolStrategy which
    # forces tool_choice="any" (wire: "required").  Models with thinking/
    # reasoning mode (Qwen3, QwQ, etc.) reject this parameter, crashing
    # every researcher subagent with a 400 error.  Skip response_format
    # for those models — the prompt already instructs JSON output.
    if name == "researcher" and _supports_forced_tool_choice(model):
        spec["response_format"] = SUBAGENT_RESPONSE_MODELS[name]
    if memory:
        spec["memory"] = memory
    if skills:
        spec["skills"] = skills

    logger.debug(
        "Built subagent spec %r for phase %s (model=%s, skills=%s)",
        name,
        phase.value,
        model if isinstance(model, str) else type(model).__name__,
        skills,
    )

    return spec


def build_phase_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None = None,
    *,
    extra_skills: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]] | None:
    """Build all subagent specs for a given phase.

    Returns ``None`` if the phase has no subagents (PLAN, TASKS, CRITIC).

    Args:
        phase: The parent phase.
        state: The current workflow state.
        config: LangGraph runtime config.
        extra_skills: Mapping of subagent name → extra skill directories.
            Allows callers to inject additional skills for specific subagents
            (e.g. from task complexity analysis).

    Returns:
        A list of SubAgent dicts, or ``None`` if the phase uses no subagents.
    """
    names = PHASE_SUBAGENTS.get(phase.value)
    if not names:
        return None

    extra_skills = extra_skills or {}
    specs: list[dict[str, Any]] = []

    for name in names:
        spec = build_subagent_spec(
            name=name,
            phase=phase,
            state=state,
            config=config,
            extra_skills=extra_skills.get(name),
        )
        specs.append(spec)

    return specs


def _resolve_subagent_skills(
    name: str,
    phase: PhaseName,
    workspace_root: str,
    extra_skills: list[str] | None = None,
) -> list[str]:
    """Resolve the skill list for a subagent.

    Default skills are defined per subagent type.  Extra skills (from task
    complexity analysis or explicit config) are appended.

    Args:
        name: Subagent name.
        phase: Parent phase (for resolving phase-specific skills).
        workspace_root: Project root for finding skill directories.
        extra_skills: Additional skill directories to include.

    Returns:
        List of absolute skill directory paths.
    """
    from spine.agents.skills_resolver import resolve_skills

    # Phase-specific skills are resolved for the subagent's parent phase,
    # but with include_rlm=False (subagents don't use the interpreter).
    # Most subagents get no phase skills — they are focused workers.
    default_skills: list[str] = []

    if name == "slice-verifier":
        # The verifier benefits from code-review skills (VERIFY phase has this)
        phase_skills = resolve_skills(
            phase=phase.value,
            workspace_root=workspace_root,
            include_rlm=False,
        )
        default_skills = phase_skills

    # Merge extra skills (deduplicated, preserving order)
    seen: set[str] = set()
    merged: list[str] = []
    for s in default_skills + list(extra_skills or []):
        if s not in seen:
            seen.add(s)
            merged.append(s)

    return merged
