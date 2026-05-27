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

    status: str = Field(
        pattern="^(implemented|partial|blocked)$",
        description="One of: implemented, partial, blocked",
    )
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
        "You are an Architectural Scout. Your objective is to map the boundaries "
        "of the requested feature, not to implement it. You may only extract "
        "structural information. For every file you investigate, your final "
        "report back to the SPECIFY orchestrator must be strictly formatted as:\n"
        "1. Dependencies: What does this file import?\n"
        "2. Interfaces: What are the public function signatures?\n"
        "3. State: Does this file manage global state or configuration?\n"
        "Do not return implementation logic.\n\n"
        "## Your task topic\n"
        "Your research topic names specific symbols discovered by semantic "
        "(vector) search. For example: \"Investigate CLI verbosity — symbols: "
        "cli/__init__.py::index, spine/config.py::SpineConfig\"\n\n"
        "Use MCP tools to look up these specific symbols — get their source, "
        "dependencies, and dependents. Never guess their interfaces.\n\n"
        "## Codebase navigation — USE MCP TOOLS FIRST\n"
        "You have MCP codebase index tools for efficient structural navigation. "
        "These answer symbol-level questions in sub-millisecond time with "
        "minimal token usage — ALWAYS use them before reading files.\n\n"
        "### Batching and Parallel Tool Calling (CRITICAL)\n"
        "- **NEVER query symbols, function sources, or files sequentially turn-by-turn.** "
        "If you need to look up 3 function sources or search multiple directories, generate "
        "all those tool calls in parallel in a SINGLE response. This is highly efficient and avoids wasting turns.\n"
        "- **Plan your queries:** In your first turn, identify all symbols named in your "
        "topic, and issue ALL lookup calls in parallel, instead of requesting them one-by-one sequentially.\n\n"
        "| Question | Tool to use |\n"
        "|----------|-------------|\n"
        "| Where is X defined? | `mcp_codebase-index_find_symbol` |\n"
        "| What does function X call? | `mcp_codebase-index_get_dependencies` |\n"
        "| Who calls function X? | `mcp_codebase-index_get_dependents` |\n"
        "| Show me function X's code | `mcp_codebase-index_get_function_source` |\n"
        "| Search for pattern X everywhere | `mcp_codebase-index_search_codebase` |\n\n"
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
        "- `mcp_codebase-index_search_codebase` — regex search across all files. "
        'Call with `{"pattern": "regex", "max_results": 20}`. Output is capped to ~8 KB / 50 hits; '
        "refine the regex (anchors, file globs) rather than retrying naively.\n\n"
        "### Fallback (only when MCP tools don't have what you need)\n"
        "- `ast_extract_symbol` — fetch a single named symbol's body from the "
        "vector index (filesystem fallback when the symbol isn't indexed yet). "
        "Use when you know the symbol name and want the body straight from disk.\n"
        "- `search_codebase` — multi-query keyword file search with content previews\n\n"
        "## Path conventions (CRITICAL)\n"
        "All paths MUST be relative from the project workspace root:\n"
        "- `.spine/artifacts/file.md`, `spine/ui/pages.py`\n"
        "- A leading `/` is workspace-relative.\n"
        "- **NEVER** use absolute paths like `/home/user/project/...` — they "
        "double-nest under the virtual filesystem root and resolve to non-existent files.\n\n"
        "## Research workflow (3-5 turns)\n"
        "1. **Look up all symbols from your topic (1 turn):** Call "
        "`mcp_codebase-index_get_function_source` for EVERY symbol named "
        "in your research topic. Call `mcp_codebase-index_get_dependencies` "
        "to trace relationships. Do ALL lookups in parallel in a single turn.\n"
        "2. **Targeted follow-up (1 turn):** If the dependencies reveal more "
        "relevant symbols, look those up too. Call `mcp_codebase-index_get_dependents` "
        "to understand what calls a key function.\n"
        "3. **Fallback search only if needed (0-1 turns):** If you need content-level "
        "pattern matching the MCP tools don't provide, use `search_codebase`. "
        "This should be RARE — MCP tools cover 90% of research needs.\n"
        "4. **Synthesize (1 turn):** Report findings. Do NOT include raw file contents — "
        "summarize key facts, signatures, conventions, and patterns.\n\n"
        "## Hard limits\n"
        "- You MUST look up every symbol named in your topic before falling back to search_codebase.\n"
        "- Your file_map MUST contain at least 1 entry.\n"
        "- Your summary MUST be at least 2 sentences.\n"
        "- Total turns: 3-5. More than 5 calls without producing "
        "output means you're over-exploring — report what you have.\n"
        "- If your tools fail (tool errors, permission issues), report that "
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
    "researcher-plan": (
        "YOU MUST USE TOOLS. Do not produce a report from memory or speculation.\n"
        "You are a Blueprint Scout. Your objective is to map the specification "
        "requirements to the codebase surface area that will need to change. "
        "You are preparing the terrain for the PLAN orchestrator to decompose "
        "the work into executable slices. For every file you investigate, your "
        "final report back to the PLAN orchestrator must be strictly formatted as:\n"
        "1. Touch Points: Which functions/classes in this file will need modification?\n"
        "2. Dependencies: What imports or callers will be affected by changes here?\n"
        "3. Risk: Are there complex data flows, global state, or tight coupling to flag?\n"
        "Do not propose solutions or implementation details.\n\n"
        "## Your task topic\n"
        "Your research topic names specific symbols discovered by semantic "
        "(vector) search. For example: \"Investigate CLI verbosity — symbols: "
        "cli/__init__.py::index, spine/config.py::SpineConfig\"\n\n"
        "Use MCP tools to look up these specific symbols — get their source, "
        "dependencies, and dependents. Never guess their interfaces.\n\n"
        "## Codebase navigation — USE MCP TOOLS FIRST\n"
        "You have MCP codebase index tools for efficient structural navigation. "
        "These answer symbol-level questions in sub-millisecond time with "
        "minimal token usage — ALWAYS use them before reading files.\n\n"
        "### Batching and Parallel Tool Calling (CRITICAL)\n"
        "- **NEVER query symbols, function sources, or files sequentially turn-by-turn.** "
        "If you need to look up 3 function sources or search multiple directories, generate "
        "all those tool calls in parallel in a SINGLE response. This is highly efficient and avoids wasting turns.\n"
        "- **Plan your queries:** In your first turn, identify all symbols named in your "
        "topic, and issue ALL lookup calls in parallel, instead of requesting them one-by-one sequentially.\n\n"
        "| Question | Tool to use |\n"
        "|----------|-------------|\n"
        "| Where is X defined? | `mcp_codebase-index_find_symbol` |\n"
        "| What does function X call? | `mcp_codebase-index_get_dependencies` |\n"
        "| Who calls function X? | `mcp_codebase-index_get_dependents` |\n"
        "| Show me function X's code | `mcp_codebase-index_get_function_source` |\n"
        "| Search for pattern X everywhere | `mcp_codebase-index_search_codebase` |\n\n"
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
        "- `mcp_codebase-index_search_codebase` — regex search across all files. "
        'Call with `{"pattern": "regex", "max_results": 20}`. Output is capped to ~8 KB / 50 hits; '
        "refine the regex (anchors, file globs) rather than retrying naively.\n\n"
        "### Fallback (only when MCP tools don't have what you need)\n"
        "- `ast_extract_symbol` — fetch a single named symbol's body from the "
        "vector index (filesystem fallback when the symbol isn't indexed yet). "
        "Use when you know the symbol name and want the body straight from disk.\n"
        "- `search_codebase` — multi-query keyword file search with content previews\n\n"
        "## Path conventions (CRITICAL)\n"
        "All paths MUST be relative from the project workspace root:\n"
        "- `.spine/artifacts/file.md`, `spine/ui/pages.py`\n"
        "- A leading `/` is workspace-relative.\n"
        "- **NEVER** use absolute paths like `/home/user/project/...` — they "
        "double-nest under the virtual filesystem root and resolve to non-existent files.\n\n"
        "## Research workflow (3-5 turns)\n"
        "1. **Look up all symbols from your topic (1 turn):** Call "
        "`mcp_codebase-index_get_function_source` for EVERY symbol named "
        "in your research topic. Call `mcp_codebase-index_get_dependencies` "
        "to trace relationships. Do ALL lookups in parallel in a single turn.\n"
        "2. **Targeted follow-up (1 turn):** If the dependencies reveal more "
        "relevant symbols, look those up too. Call `mcp_codebase-index_get_dependents` "
        "to understand what calls a key function.\n"
        "3. **Fallback search only if needed (0-1 turns):** If you need content-level "
        "pattern matching the MCP tools don't provide, use `search_codebase`. "
        "This should be RARE — MCP tools cover 90% of research needs.\n"
        "4. **Synthesize (1 turn):** Report findings. Do NOT include raw file contents — "
        "summarize key facts, signatures, conventions, and patterns.\n\n"
        "## Hard limits\n"
        "- You MUST look up every symbol named in your topic before falling back to search_codebase.\n"
        "- Your file_map MUST contain at least 1 entry.\n"
        "- Your summary MUST be at least 2 sentences.\n"
        "- Total turns: 3-5. More than 5 calls without producing "
        "output means you're over-exploring — report what you have.\n"
        "- If your tools fail (tool errors, permission issues), report that "
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
        "read_edit_lint, then verify with execute.\n"
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
        "- `ast_extract_symbol` — fetch a single named symbol's source straight "
        "from the index. Use when you know the symbol name and want its body without "
        "reading the whole file.\n"
        "- `read_edit_lint` — the ONLY write tool. Pass either `old_str`+`new_str` "
        "(exact, single-occurrence find-and-replace) OR `full_replace` (whole-file "
        "content). The tool runs a syntax check before writing — on a syntax error "
        "it returns `{\"status\":\"syntax_error\",...}` and the file is left "
        "untouched. Fix the snippet and call again.\n"
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
        "3. **Make changes (1-2 turns, batch):** Apply all edits with "
        "`read_edit_lint`. Read-before-write. Issue ≥2 edits in a single turn "
        "where possible. Focus only on files listed in the slice — do NOT modify "
        "or create files outside its scope. If `read_edit_lint` returns "
        "`status=\"syntax_error\"`, the file was NOT written — correct the "
        "snippet and call again.\n"
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
    "search_codebase",
    "ast_extract_symbol",
]

_FULL_TOOLS: list[str] = [
    "ls",
    "read_file",
    "read_edit_lint",
    "glob",
    "grep",
    "search_codebase",
    "ast_extract_symbol",
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
    "1. Look up at least 3 symbols relevant to the task description.\n"
    "2. Produce a file_map with at least 2 entries.\n"
    "3. If symbols cannot be found, explain what you searched and what went wrong.\n"
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


# MCP tools the researcher (Explore) loop is actually allowed to call.
# The upstream codebase-index server exposes 21+ tools, but the researcher
# prompts (Blueprint Scout / Architectural Scout) only describe these five.
# Loading the others is dead schema in the prefix — each tool description
# costs ~150-400 prefix tokens per researcher turn. Trim aggressively.
_RESEARCHER_MCP_ALLOWLIST: frozenset[str] = frozenset({
    "mcp_codebase-index_find_symbol",
    "mcp_codebase-index_get_function_source",
    "mcp_codebase-index_get_dependencies",
    "mcp_codebase-index_get_dependents",
    "mcp_codebase-index_search_codebase",
})


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
        if subagent_name == "researcher":
            mcp_tools = [t for t in mcp_tools if t.name in _RESEARCHER_MCP_ALLOWLIST]
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


def _strict_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Build a strict-mode JSON schema dict from a Pydantic model.

    OpenAI / OpenRouter strict json_schema mode requires
    ``additionalProperties: false`` and every property listed in
    ``required`` — for the root object *and* every nested ``$defs`` entry.
    Pydantic's default schema sets neither, so normalize before sending.
    """

    def enforce(obj: dict[str, Any]) -> None:
        if obj.get("type") == "object" and "properties" in obj:
            obj["additionalProperties"] = False
            obj["required"] = list(obj["properties"].keys())

    schema = model_cls.model_json_schema()
    enforce(schema)
    for sub in schema.get("$defs", {}).values():
        if isinstance(sub, dict):
            enforce(sub)
    return schema


def _bind_response_format(
    model: Any,
    schema_model: type[BaseModel],
    name: str,
) -> bool:
    """Bind an OpenAI/OpenRouter json_schema response_format on the model.

    Mutates ``model.model_kwargs`` so every API call from this model
    sends the strict json_schema response_format.  Both OpenRouter
    (ChatOpenRouter) and local OpenAI-compatible servers (ChatOpenAI)
    use the same OpenAI-style ``response_format`` — no separate
    ``guided_json`` fallback is needed for modern vLLM versions.

    Returns ``True`` when binding succeeded — callers should fall back
    to Deep Agents' spec-level ``response_format`` when this returns
    ``False`` (e.g. for ChatAnthropic which doesn't expose
    ``model_kwargs``).
    """
    model_kwargs = getattr(model, "model_kwargs", None)
    if not isinstance(model_kwargs, dict):
        return False
    schema = _strict_json_schema(schema_model)
    model_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }
    return True


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


# ── Prompt resolution ──────────────────────────────────────────────────


def _resolve_subagent_prompt(name: str, phase: PhaseName) -> str:
    """Resolve the system prompt for a subagent, with phase-specific variants.

    The researcher subagent uses different prompts depending on whether it is
    scouting for SPECIFY (Architectural Scout — boundaries/contracts) or PLAN
    (Blueprint Scout — change surface/risk assessment).
    """
    if name == "researcher":
        plan_key = f"{name}-{phase.value}"
        if plan_key in SUBAGENT_PROMPTS:
            return SUBAGENT_PROMPTS[plan_key]
    return SUBAGENT_PROMPTS[name]


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

    from spine.agents.helpers import (
        resolve_model,
    )
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

    # ── Inject compound tools (read_edit_lint, ast_extract_symbol) ──
    # These are standalone BaseTools that replace generic write_file/edit_file
    # (slice-implementer) and supplement MCP get_function_source (researcher).
    if "read_edit_lint" in allowed_tool_names:
        from spine.agents.tools.read_edit_lint import ReadEditLintTool

        tools.append(ReadEditLintTool(workspace_root=workspace_root))
    if "ast_extract_symbol" in allowed_tool_names:
        from spine.agents.tools.ast_extract_symbol import AstExtractSymbolTool
        from spine.config import SpineConfig

        tools.append(
            AstExtractSymbolTool(
                workspace_root=workspace_root,
                db_path=SpineConfig.load().checkpoint_path,
            )
        )

    # ── Inject MCP codebase-index tools for researcher + slice-implementer ─
    # Both subagents benefit enormously from structural queries instead of
    # full-file reads (see trace 019e486e: 11× redundant dispatcher.py reads
    # during implement).  MCP gives "where is X / who calls X" in milliseconds.
    # slice-verifier is read+execute only and rarely needs deep navigation.
    if name in ("researcher", "slice-implementer"):
        _inject_mcp_tools(tools, workspace_root, subagent_name=name)

    # ── Build spec ───────────────────────────────────────────────
    # ToolOutputTrimmer is intentionally absent from leaf code-producing
    # subagents (slice-implementer / slice-verifier) — they need full
    # file-read & execute outputs to do their job. The researcher subagent
    # is different: it is a survey loop that synthesises findings, so old
    # tool results past the recent window are safe to evict to a metadata
    # placeholder. This caps Explore transcripts at ~30 K tokens given the
    # 8 KB cap on `mcp_codebase-index_search_codebase` output.
    subagent_middleware: list[Any] = []
    if name == "researcher":
        from spine.agents.context_editing import ToolOutputTrimmer
        subagent_middleware.append(ToolOutputTrimmer(max_full_tool_results=8))

    spec: dict[str, Any] = {
        "name": name,
        "description": SUBAGENT_DESCRIPTIONS[name],
        "system_prompt": _resolve_subagent_prompt(name, phase),
        "model": model,
        "tools": tools,
        "middleware": subagent_middleware,
    }
    # Structured outputs are NOT bound for subagents that need to use tools
    # before reporting results.  Both ``model_kwargs["response_format"]``
    # (native API json_schema) and LangChain's ToolStrategy/ProviderStrategy
    # cause the model to satisfy the schema on its very first reply, skipping
    # tool calls and emitting plausible-but-hallucinated content (observed on
    # deepseek-v4-pro and -flash, glm, local vLLM, etc.).
    #
    # - ``researcher``: no schema — free-form exploration, finalized via
    #   ``run_explore_node`` → ``_finalize_research_findings()``.
    # - ``slice-implementer``: no schema — model uses write_file/edit_file/
    #   execute tools, then reports SliceResult in its final message.
    #   ``_extract_slice_result()`` parses it from the last assistant content.
    # - ``slice-verifier``: no schema — model uses read_file/execute tools,
    #   then reports VerificationResult.  The verify subgraph extracts it
    #   from the last assistant content.
    #
    # Subagents NOT in this exclusion list receive ProviderStrategy schema
    # binding by default (suitable for report-only agents with no tools).
    if name in SUBAGENT_RESPONSE_MODELS and name not in (
        "researcher",
        "slice-implementer",
        "slice-verifier",
    ):
        schema_model = SUBAGENT_RESPONSE_MODELS[name]
        if _supports_forced_tool_choice(model):
            from langchain.agents.structured_output import ProviderStrategy
            spec["response_format"] = ProviderStrategy(schema=schema_model)
        else:
            _bind_response_format(
                model,
                schema_model,
                name=f"{name}_response",
            )
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
