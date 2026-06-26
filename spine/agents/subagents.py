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

from spine.agents.prompt_format import Tag, xml_block
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


def _build_researcher_prompt(*, scout_kind: str, role_blurb: str, report_format: str) -> str:
    """Build a researcher (Scout) system prompt in the XML-tagged hostage form.

    The two scout variants (Architectural / Blueprint) share ~85 % of the
    content — only the role wording and the per-file report format differ.
    Factoring this out keeps both in sync.
    """
    role_body = (
        f"YOU MUST USE TOOLS. Do not produce a report from memory or speculation.\n"
        f"You are a {scout_kind}. {role_blurb} For every file you "
        f"investigate, your final report must be strictly formatted as:\n"
        f"{report_format}\n"
        "Do not return implementation logic.\n\n"
        "Your research topic names specific symbols discovered by semantic "
        "(vector) search. For example: \"Investigate CLI verbosity — symbols: "
        "cli/__init__.py::index, spine/config.py::SpineConfig\"\n\n"
        "Use `codebase_query` to look up these specific symbols — get their "
        "source, dependencies, and dependents. Never guess their interfaces."
    )

    tools_body = (
        "USE `codebase_query` FIRST. It answers symbol-level questions in "
        "sub-millisecond time — ALWAYS use it before reading files.\n\n"
        "Batching and parallel tool calling (CRITICAL):\n"
        "- NEVER query symbols, function sources, or files sequentially "
        "turn-by-turn. If you need to look up 3 function sources or search "
        "multiple directories, generate all those tool calls in parallel in "
        "a SINGLE response. This is highly efficient and avoids wasting turns.\n"
        "- Plan your queries: In your first turn, identify all symbols named "
        "in your topic, and issue ALL lookup calls in parallel.\n\n"
        "| Question | Call |\n"
        "|----------|------|\n"
        '| Where is X defined? | `codebase_query(action="find_symbol", name="X")` |\n'
        '| Show me X\'s source | `codebase_query(action="get_source", name="X")` |\n'
        '| What does X call? | `codebase_query(action="get_dependencies", name="X")` |\n'
        '| Who calls X? | `codebase_query(action="get_dependents", name="X")` |\n'
        '| Regex P across code | `codebase_query(action="search", pattern="P")` |\n\n'
        "Primary tool (use FIRST):\n"
        "- `codebase_query` — single tool, five actions. Pick `action`, fill "
        "the one argument it needs:\n"
        "  - `find_symbol`, `get_source`, `get_dependencies`, "
        "`get_dependents` all take `name` (clean identifier — no whitespace, "
        "no module prefix, no parentheses).\n"
        "  - `search` takes `pattern` (regex). Output is capped to ~8 KB / "
        "50 hits — refine with anchors / file globs rather than retrying naively.\n"
        "  - `name` and `pattern` are mutually exclusive. Do NOT pass both.\n\n"
        "Fallback (only when `codebase_query` doesn't have what you need):\n"
        "- `ast_extract_symbol` — fetch a single named symbol's body from "
        "the vector index (filesystem fallback when the symbol isn't indexed "
        "yet). Use when you know the symbol name and want the body straight "
        "from disk.\n"
        "- `search_codebase` — multi-query keyword file search with content "
        "previews."
    )

    workflow_body = (
        "Research workflow (3-5 turns):\n"
        "1. Look up all symbols from your topic (1 turn): For EVERY symbol "
        "named in your research topic, issue "
        "`codebase_query(action=\"get_source\", name=...)` and "
        "`codebase_query(action=\"get_dependencies\", name=...)` in parallel "
        "in a single turn.\n"
        "2. Targeted follow-up (1 turn): If the dependencies reveal more "
        "relevant symbols, look those up too. Use "
        "`codebase_query(action=\"get_dependents\", name=...)` to understand "
        "what calls a key function.\n"
        "3. Fallback search only if needed (0-1 turns): If you need content-"
        "level pattern matching, use "
        "`codebase_query(action=\"search\", pattern=...)`. This should be "
        "RARE — the symbol actions cover 90% of research needs.\n"
        "4. Synthesize (1 turn): Report findings. Do NOT include raw file "
        "contents — summarize key facts, signatures, conventions, and patterns."
    )

    constraints_body = (
        "Path conventions: all paths MUST be relative from the project "
        "workspace root.\n"
        "- `.spine/artifacts/file.md`, `spine/ui/pages.py`\n"
        "- A leading `/` is workspace-relative.\n"
        "- NEVER use absolute paths like `/home/user/project/...` — they "
        "double-nest under the virtual filesystem root and resolve to "
        "non-existent files.\n\n"
        "Hard limits:\n"
        "- You MUST look up every symbol named in your topic before falling "
        "back to search.\n"
        "- Your file_map MUST contain at least 1 entry.\n"
        "- Your summary MUST be at least 2 sentences.\n"
        "- Total turns: 3-5. More than 5 calls without producing output "
        "means you're over-exploring — report what you have.\n"
        "- If your tools fail (tool errors, permission issues), report that "
        "with the error details — do NOT return empty results.\n"
        "- If you produce empty results, you WILL be re-dispatched, wasting "
        "time and tokens. Do the work correctly the first time."
    )

    output_body = (
        "Your output MUST follow the ResearchFindings schema:\n"
        "- summary: concise paragraph summarizing findings\n"
        "- patterns: notable patterns, conventions, or idioms discovered\n"
        "- file_map: mapping of important file paths to brief descriptions\n"
        "- dependencies: key dependencies, imports, or external services"
    )

    return (
        xml_block(Tag.ROLE, role_body)
        + "\n\n"
        + xml_block(Tag.TOOLS, tools_body)
        + "\n\n"
        + xml_block(Tag.WORKFLOW, workflow_body)
        + "\n\n"
        + xml_block(Tag.CONSTRAINTS, constraints_body)
        + "\n\n"
        + xml_block(Tag.OUTPUT_SCHEMA, output_body)
    )


_ARCHITECTURAL_SCOUT_REPORT = (
    "1. Dependencies: What does this file import?\n"
    "2. Interfaces: What are the public function signatures?\n"
    "3. State: Does this file manage global state or configuration?"
)

_BLUEPRINT_SCOUT_REPORT = (
    "1. Touch Points: Which functions/classes in this file will need modification?\n"
    "2. Dependencies: What imports or callers will be affected by changes here?\n"
    "3. Risk: Are there complex data flows, global state, or tight coupling to flag?"
)


SUBAGENT_PROMPTS: dict[str, str] = {
    "researcher": _build_researcher_prompt(
        scout_kind="Architectural Scout",
        role_blurb=(
            "Your objective is to map the boundaries of the requested feature, "
            "not to implement it. You may only extract structural information."
        ),
        report_format=_ARCHITECTURAL_SCOUT_REPORT,
    ),
    "researcher-plan": _build_researcher_prompt(
        scout_kind="Blueprint Scout",
        role_blurb=(
            "Your objective is to map the specification requirements to the "
            "codebase surface area that will need to change. You are preparing "
            "the terrain for the PLAN orchestrator to decompose the work into "
            "executable slices."
        ),
        report_format=_BLUEPRINT_SCOUT_REPORT,
    ),
    "slice-implementer": (
        xml_block(
            Tag.ROLE,
            "YOU MUST USE TOOLS. Do not describe changes — make them with "
            "read_edit_lint.\n"
            "You are a code implementer for a single feature slice. Your "
            "task description contains the full slice definition, codebase "
            "context, modification targets with exact line ranges, and "
            "files to modify — read all of it carefully before starting.",
        )
        + "\n\n"
        + xml_block(
            Tag.TOOLS,
            "When a specific symbol you must call/extend is NOT already in "
            "your task context, `codebase_query` answers symbol-level "
            "questions in sub-millisecond time:\n\n"
            "| Question | Call |\n"
            "|----------|------|\n"
            '| Where is X defined? | `codebase_query(action="find_symbol", name="X")` |\n'
            '| Show me X\'s source | `codebase_query(action="get_source", name="X")` |\n'
            '| What does X call? | `codebase_query(action="get_dependencies", name="X")` |\n'
            '| Who calls X? | `codebase_query(action="get_dependents", name="X")` |\n'
            '| Regex P across code | `codebase_query(action="search", pattern="P")` |\n\n'
            "Your slice is ALREADY specified — target files, acceptance "
            "criteria, and an implementation directive are in your task. "
            "Codebase research happened upstream in PLAN. Do NOT survey the "
            "codebase. Read the named files, edit, run tests. Reach for "
            "`codebase_query` ONLY when a specific unknown symbol blocks an "
            "edit.\n\n"
            "Your tools:\n"
            "- `codebase_query` — targeted symbol lookup (use ONLY when a "
            "specific symbol you must call/extend isn't in your task context). "
            "Pick `action`, fill the one argument it needs:\n"
            "  - `find_symbol`, `get_source`, `get_dependencies`, "
            "`get_dependents` all take `name` (clean identifier — no "
            "whitespace, no module prefix, no parentheses).\n"
            "  - `search` takes `pattern` (regex). Output is capped to ~8 KB / "
            "50 hits.\n"
            "  - `name` and `pattern` are mutually exclusive. Do NOT pass "
            "both.\n"
            "- `read_edit_lint` — your ONLY filesystem tool: read AND write.\n"
            "  READ is ANCHORED only (no whole-file or arbitrary line-range "
            "reads — you already know the target from the plan): "
            "`read_symbol=\"ClassName.method\"` returns a definition's current "
            "source; `read_around=\"exact snippet\"` returns the region around a "
            "snippet. Read at most the ONE symbol you are about to edit. NEVER "
            "re-read a file you just edited: a status=\"ok\" result means the "
            "edit is in the file exactly as you sent it.\n"
            "  EDIT: pass exactly ONE edit mode. Prefer the robust modes:\n"
            "    • `ast_edit` — BEST for changing or adding a whole "
            "function/method/class: name the symbol (e.g. "
            "`{\"symbol\":\"ClassName.method\",\"action\":\"replace\","
            "\"code\":\"...\"}`) or `insert_before`/`insert_after`. No line "
            "numbers, no exact-byte matching — immune to indentation drift.\n"
            "    • `patch` — a batch of WHITESPACE-TOLERANT search/replace ops "
            "(`[{\"search\":...,\"replace\":...}]`); use when you are not "
            "certain of exact indentation. Re-indents to the match.\n"
            "    • `old_str`+`new_str` / `edits` — EXACT find-and-replace "
            "(single / atomic batch). Use only when you have the bytes exactly.\n"
            "    • `full_replace` (whole-file content) for new or small files; "
            "`start_line`+`end_line`+`replacement` for a line range (pass "
            "`expected` to guard stale line numbers).\n"
            "  The tool runs a syntax check before writing — on a syntax error "
            "or failed match it returns `{\"status\":\"syntax_error\",...}` (or "
            "`no_match`/`ambiguous_match`/`stale`) and the file is left "
            "untouched. Fix and call again. `already_applied` means the change "
            "is ALREADY in the file — move on, do not retry.\n"
            "  LINT: successful Python writes include a `ruff` field with "
            "bounded diagnostics. There is no shell — do not try to run "
            "tests or linters; the verify phase does that.",
        )
        + "\n\n"
        + xml_block(
            Tag.WORKFLOW,
            "Implementation workflow (3-5 turns):\n"
            "1. Read existing code (1 turn, batch): issue parallel "
            "`read_edit_lint` READ calls for the files listed in your task "
            "description — ranged reads around the modification targets, "
            "not whole files. Check the targets in your task description "
            "for exact change sites.\n"
            "2. Navigate if needed (0-1 turns): ONLY if a specific symbol you "
            "must call/extend isn't in your task context, use `codebase_query` "
            "to locate it. Do NOT explore broadly — the slice is pre-specified.\n"
            "3. Make changes (1-2 turns, batch): Apply all edits with "
            "`read_edit_lint`. Read-before-write. Issue ≥2 edits in a "
            "single turn where possible. Focus only on files listed in the "
            "slice — do NOT modify or create files outside its scope. If "
            "`read_edit_lint` returns `status=\"syntax_error\"`, the file "
            "was NOT written — correct the snippet and call again. Address "
            "any `ruff` diagnostics reported on your own edits.\n"
            "4. Report (final turn): Return the SliceResult with exactly "
            "what you changed and any remaining issues. Do NOT attempt to "
            "run tests — the verify phase executes them after all slices "
            "land.",
        )
        + "\n\n"
        + xml_block(
            Tag.CONSTRAINTS,
            "Path conventions: all file paths MUST be relative from the "
            "project workspace root.\n"
            "- Correct: `spine/ui/pages.py`, `.spine/artifacts/doc.md`\n"
            "- Correct: `/spine/ui/pages.py` (leading `/` = workspace-relative)\n"
            "- WRONG: `/home/pat/Projects/spine/spine/ui/pages.py` — "
            "absolute paths double-nest under the virtual filesystem and "
            "create files in the wrong location.\n"
            "- WRONG: `../other/file.py` — traversal blocked by virtual "
            "filesystem.\n"
            "- Use `codebase_query` to verify a symbol exists before calling "
            "it. Do not invent paths.\n\n"
            "Hard limits:\n"
            "- Batch reads: issue the READ calls for the files named in your "
            "task in a single turn (parallel tool calls). Never read one "
            "file per turn, and never re-read a file after a status=\"ok\" "
            "edit — the result already tells you the edit landed.\n"
            "- Exploration budget: maximum 2 turns of read/lookup before "
            "your first write/edit. If you haven't changed code by turn 3, "
            "you're over-exploring — make changes with what you know.\n"
            "- Scope: modify ONLY files listed in the slice. Do not touch "
            "files outside its scope even if you think they need changes — "
            "report them as issues.\n"
            "- Stuck? If you are blocked by a missing dependency from "
            "another slice, set status='blocked' with the dependency name "
            "in issues. Do not try to implement the dependency yourself.\n"
            "- Silence is failure: if you make zero file changes and set "
            "status='implemented', the orchestrator will not know the "
            "slice was skipped. Always report exactly what files you changed.",
        )
        + "\n\n"
        + xml_block(
            Tag.OUTPUT_SCHEMA,
            "End with this exact JSON structure (no markdown wrapping, no "
            "backticks):\n"
            '{"status": "implemented|partial|blocked", '
            '"files_modified": ["path1", "path2"], '
            '"files_created": ["path3"], '
            '"test_results": "summary of test/lint outcomes", '
            '"issues": ["any remaining issues or empty list"]}',
        )
    ),
    "slice-verifier": (
        xml_block(
            Tag.ROLE,
            "YOU MUST USE TOOLS. Do not verify from memory — inspect actual "
            "files and run actual tests.\n"
            "You are a verification engineer. Your task description contains "
            "the full slice definition, acceptance criteria, and files to "
            "verify — read it carefully before starting.",
        )
        + "\n\n"
        + xml_block(
            Tag.WORKFLOW,
            "1. Review the slice definition and acceptance criteria from "
            "your task description — it names the target_files to verify.\n"
            "2. Inspect the implemented files with read_file (ranged, "
            "read-only — there is no shell cat/grep/ls); batch your reads "
            "(≥2 files per turn).\n"
            "3. Run relevant tests and linters with run_checks (e.g. "
            "'pytest …', 'ruff check …'). It runs checks only; it will not "
            "grep/find/list files for you — read_file does that.\n"
            "4. Check each acceptance criterion individually.\n"
            "5. Produce a structured verification report.",
        )
        + "\n\n"
        + xml_block(
            Tag.CONSTRAINTS,
            "You are report-only. Do not fix issues you find. If a test "
            "fails or a criterion is not met, record it in the checklist "
            "and gaps. Your job is to provide evidence, not to repair.",
        )
        + "\n\n"
        + xml_block(
            Tag.OUTPUT_SCHEMA,
            "End with a structured verification result:\n"
            "```json\n"
            '{"verdict": "VERIFIED|NOT_VERIFIED", "checklist": '
            '[{"criterion": "...", "passed": true, "detail": "..."}], '
            '"gaps": [...], "recommendations": [...]}\n'
            "```",
        )
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

# Slice-implementer surface. A slice arrives fully specified — target_files,
# acceptance criteria, and an upstream planning directive — and codebase
# research already happened in PLAN. So the broad keyword search
# (``search_codebase``, returns multi-file content previews) and the
# redundant ``ast_extract_symbol`` are intentionally OMITTED: they let the
# implementer "research half the codebase" (trace 019e784c: 83 LLM calls /
# 1.39M prompt tokens on a one-line flag). Targeted symbol lookups still go
# through the injected ``codebase_query`` wrapper when a specific unknown
# symbol blocks an edit.
#
# ``read_file`` and ``execute`` were removed after trace 019eb502 (1.76M
# prompt tokens): implementers read the same two files 35× via read_file
# (~50K tokens of raw output riding every later turn) and used execute for
# ad-hoc ast.parse/ruff/pytest runs plus interpreter spelunking — linting
# that read_edit_lint performs itself. read_edit_lint now carries a
# line-numbered READ mode (ranged, output-capped) and reports bounded ruff
# diagnostics on successful Python writes; test execution belongs to the
# slice-verifier, not the implementer.
_IMPLEMENT_TOOLS: list[str] = [
    "read_edit_lint",
]

# Slice-verifier surface. Like the slice-implementer, the verifier was a leaf
# code agent still carrying the raw filesystem+shell surface (ls/read_file/
# glob/grep/execute/search_codebase). Trace 019f0212 showed it doing the exact
# survey spiral the implementer lockdown (read_edit_lint) was built to kill:
# ~290 execute calls in one verify pass, >90% shelling grep/find/ls/cat, many
# returning no output. It now gets two purpose-built tools (built by
# spine.agents.verify_subagent_tools): a ranged read-only ``read_file`` and a
# constrained ``run_checks`` runner that rejects pure-exploration shell
# commands. The verifier legitimately RUNS tests, so — unlike the implementer —
# it keeps an execute path, but only through the policed wrapper.
_VERIFY_TOOLS: list[str] = [
    "read_file",
    "run_checks",
]

SUBAGENT_TOOLS: dict[str, list[str]] = {
    "researcher": _READ_ONLY_TOOLS,
    "slice-implementer": _IMPLEMENT_TOOLS,
    "slice-verifier": _VERIFY_TOOLS,
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

# ── Factory functions ──────────────────────────────────────────────────


def _inject_mcp_tools(
    tools: list, workspace_root: str, *, subagent_name: str = "researcher"
) -> None:
    """Inject the consolidated ``codebase_query`` tool into a subagent's tool list.

    Every subagent receives a single ``CodebaseQueryTool`` wrapper (tool name
    ``codebase_query``) that collapses the raw MCP index surface — no raw
    ``mcp_``-prefixed tool is ever appended to a subagent's tool list. Errors
    are logged and swallowed — subagents fall back to filesystem tools when
    the index is unavailable.

    Args:
        tools: The subagent's tool list (mutated in place).
        workspace_root: Project root for PROJECT_ROOT env injection.
        subagent_name: Calling subagent name (used for cache key + debug logs).
    """
    try:
        from spine.config import SpineConfig

        config = SpineConfig.load()
        # Every subagent gets ONE consolidated codebase_query tool — never
        # the individual raw MCP wrappers. Collapsing the surface area
        # eliminates the wrong-key / invented-arg failure class observed
        # in trace 019e6cc4 (23/23 research branches failed with malformed
        # args) and ensures no raw mcp_-prefixed tool reaches any agent.
        # The wrapper lazy-loads its MCP backend on first use.
        from spine.agents.tools.codebase_query import (
            CodebaseQueryTool,
            search_cap_for_subagent,
        )

        mcp_tools: list = [
            CodebaseQueryTool(
                workspace_root=workspace_root,
                mcp_servers=config.mcp_servers,
                db_path=config.checkpoint_path,
                search_result_char_cap=search_cap_for_subagent(subagent_name),
            )
        ]
        tools.extend(mcp_tools)
        logger.info(
            "Injected %d MCP-related tools into %s subagent: %s",
            len(mcp_tools), subagent_name, [t.name for t in mcp_tools],
        )
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
) -> dict[str, Any]:
    """Build a dict SubAgent spec for a SPINE subagent.

    Resolves model and memory from the same sources as the parent phase
    agent so there is no duplication.  The returned dict is ready to pass
    directly into ``create_deep_agent(subagents=[...])``.

    Args:
        name: Subagent name (e.g. ``"researcher"``).
        phase: The parent phase (used for model resolution fallback).
        state: The current workflow state.
        config: LangGraph runtime config.

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

    # ── slice-verifier: bounded, purpose-built surface (no raw fs/shell) ──
    # Replaces ls/read_file/glob/grep/execute/search_codebase with a ranged
    # read-only read_file + a constrained run_checks runner (trace 019f0212).
    # run_checks wraps the FilesystemMiddleware ``execute`` tool so it still
    # runs real tests, but rejects pure-exploration commands. The generic-tool
    # injection blocks below are gated on names not in _VERIFY_TOOLS, so they
    # naturally skip; downstream spec assembly (ToolOutputTrimmer, schema
    # exclusion) still applies via the existing slice-verifier branches.
    if name == "slice-verifier":
        from spine.agents.verify_subagent_tools import build_verify_subagent_tools

        fs_execute = next((t for t in fs_mw.tools if t.name == "execute"), None)
        if fs_execute is None:
            logger.warning(
                "slice-verifier: no execute tool from FilesystemMiddleware; "
                "run_checks will report an error if invoked."
            )
        tools = build_verify_subagent_tools(
            workspace_root=workspace_root, execute_tool=fs_execute
        )
    else:
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

        # Pass the active slice's authoritative target_files so the tool can
        # ground "did you mean" path suggestions on the plan's pinned paths
        # (the editor otherwise invents variants — trace 019ef1e5). Absent for
        # non-slice callers (e.g. researcher), which leaves it an empty list.
        _active_slice = state.get("active_slice") or {}
        tools.append(
            ReadEditLintTool(
                workspace_root=workspace_root,
                target_files=list(_active_slice.get("target_files") or []),
                reference_only_files=list(
                    _active_slice.get("reference_only_files") or []
                ),
            )
        )
    if "ast_extract_symbol" in allowed_tool_names:
        from spine.agents.tools.ast_extract_symbol import AstExtractSymbolTool
        from spine.config import SpineConfig

        tools.append(
            AstExtractSymbolTool(
                workspace_root=workspace_root,
                db_path=SpineConfig.load().checkpoint_path,
            )
        )

    # ── Inject codebase_query wrapper (researcher; implementer ONLY when the
    #    planner gave it no edit_plan) ──
    # The researcher's whole job is structural survey, so it always gets the
    # wrapper. The slice-implementer is different: handing it codebase_query
    # lets it re-research the codebase instead of editing — across every model
    # tested it ran 20-26 codebase_query calls and made ZERO edits, burning
    # 1.4M+ prompt tokens on a fully specified slice. When the planner supplied
    # an ``edit_plan`` OR ``reference_symbols`` (symbols already resolved by the
    # planner, read via read_edit_lint's read_symbol and applied via ast_edit),
    # the implementer needs no research surface at all: its only move should be
    # to read the named symbols and apply the edit. So withhold codebase_query
    # whenever the active slice carries targeting; legacy slices without any
    # keep the lookup as a fallback. (trace 019ede24: the decomposed plan
    # provides reference_symbols, not edit_plan — gating only on edit_plan left
    # codebase_query available and the implementer ran 23 codebase_query calls
    # with 2 read_edit_lint calls.)
    active_slice = (state or {}).get("active_slice") or {}
    implementer_has_plan = name == "slice-implementer" and bool(
        active_slice.get("edit_plan") or active_slice.get("reference_symbols")
    )
    if name == "researcher" or (name == "slice-implementer" and not implementer_has_plan):
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
        # ToolOutputTrimmer caps any single worker turn's tool-result blob
        # so a large search hit doesn't bloat the next supervisor cycle's
        # prompt. ResearcherConvergenceMiddleware was removed when the
        # supervisor↔worker loop in spine.agents.exploration_agents took
        # over convergence — the supervisor's explicit ``is_complete``
        # flag is now the termination signal, governed by the per-phase
        # cycle cap (ConvergenceConfig.researcher_supervisor_max_cycles_*).
        from spine.agents.context_editing import ToolOutputTrimmer

        subagent_middleware.append(ToolOutputTrimmer(max_full_tool_results=8))
    elif name in ("slice-implementer", "slice-verifier"):
        # The leaf code agents run a genuine multi-step read→edit→execute loop
        # (not the researcher's survey pattern), so they keep the agent.ainvoke
        # tool loop. But with no trimming the message history grows
        # monotonically: trace 019e87dd showed a single slice climbing
        # 6K→34K prompt tokens over ~6-21 turns (whole-file reads + execute
        # output never evicted), then crashing when prompt + the requested
        # completion exceeded the model window. Trimming the OLD tail to a
        # metadata placeholder while keeping the recent window flattens the
        # climb without starving the agent of its working context — the larger
        # window (12 vs the researcher's 8) preserves more recent file/test
        # output that the code agents legitimately need.
        from spine.agents.context_editing import ToolOutputTrimmer

        subagent_middleware.append(ToolOutputTrimmer(max_full_tool_results=12))

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
    # - ``slice-verifier``: no schema — model uses read_file/run_checks tools,
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

    logger.debug(
        "Built subagent spec %r for phase %s (model=%s)",
        name,
        phase.value,
        model if isinstance(model, str) else type(model).__name__,
    )

    return spec
