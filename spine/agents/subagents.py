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
        "Guidelines:\n"
        "1. Start by listing the relevant directories and files.\n"
        "2. Read key files to understand structure, patterns, and dependencies.\n"
        "3. Focus on what is relevant to the task — do not explore broadly.\n"
        "4. Report conventions (naming, imports, patterns) you discover.\n"
        "5. Map important file paths with brief descriptions.\n"
        "6. Note any dependencies or external services.\n"
        "7. Batch reads: read 3-5 files per turn, not one at a time.\n\n"
        "IMPORTANT: You are read-only. Do not modify any files.\n"
        "Be concise — your output will be consumed by the specification writer.\n\n"
        "CRITICAL: Return a concise structured summary. Do NOT return raw file "
        "contents — summarize findings, key facts, and relevant code signatures "
        "only. The parent agent needs your analysis, not the raw source.\n\n"
        "MINIMUM OUTPUT REQUIREMENTS:\n"
        "- You MUST read at least 2 files before producing your summary.\n"
        "- If you cannot read files (tool errors, permission issues), report that "
        "as your summary with the error details — do NOT return empty results.\n"
        "- Your file_map MUST contain at least 1 entry.\n"
        "- Your summary MUST be at least 2 sentences.\n"
        "- If you produce empty results, you WILL be re-dispatched, wasting time "
        "and tokens. Do the work correctly the first time.\n\n"
        "Your output MUST follow the ResearchFindings schema:\n"
        "- summary: concise paragraph summarizing findings\n"
        "- patterns: notable patterns, conventions, or idioms discovered\n"
        "- file_map: mapping of important file paths to brief descriptions\n"
        "- dependencies: key dependencies, imports, or external services\n"
    ),
    "slice-implementer": (
        "YOU MUST USE TOOLS. Do not describe changes — make them with "
        "write_file and edit_file, then verify with execute.\n"
        "You are a code implementer. Your task description contains "
        "the full slice definition, relevant codebase context, and "
        "files to modify — read it carefully before starting.\n\n"
        "Guidelines:\n"
        "1. Review the slice definition, acceptance criteria, and "
        "codebase context provided in your task description.\n"
        "2. Read any additional source files you need to inspect "
        "(batch reads — ≥3 files per turn).\n"
        "3. Write clean, well-documented code following project conventions.\n"
        "4. Include appropriate type annotations and docstrings.\n"
        "5. Handle errors gracefully.\n"
        "6. Run tests and linters for your slice.\n"
        "7. Fix any errors found.\n"
        "8. Report exactly what you changed and any remaining issues.\n\n"
        "Focus on this slice only — do not modify files outside its scope.\n"
        "If you are blocked by a missing dependency from another slice, "
        "report it as an issue rather than trying to implement that "
        "dependency.\n\n"
        "End with a structured result:\n"
        "```json\n"
        '{"status": "implemented|partial|blocked", "files_modified": [...], '
        '"files_created": [...], "test_results": "...", "issues": [...]}\n'
        "```\n"
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
]

_FULL_TOOLS: list[str] = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
]

_READ_AND_EXECUTE_TOOLS: list[str] = [
    "ls",
    "read_file",
    "glob",
    "grep",
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
    PhaseName.TASKS.value: ["researcher"],
    PhaseName.IMPLEMENT.value: ["slice-implementer"],  # No verifier — dedicated verify phase is authoritative
    PhaseName.VERIFY.value: ["slice-verifier"],
    # PLAN, CRITIC — single-agent, no subagents
}


# ── Factory functions ──────────────────────────────────────────────────


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
    # then create FilesystemMiddleware to get the tools.
    # IMPORTANT: Use SPINE_FILESYSTEM_PROMPT to avoid the "All file paths must
    # start with a /" default that conflicts with virtual_mode=True.
    from deepagents.middleware.filesystem import FilesystemMiddleware

    from spine.agents.backend import build_backend
    from spine.agents.factory import SPINE_FILESYSTEM_PROMPT, SPINE_FILESYSTEM_EXEC_PROMPT

    backend = build_backend(workspace_root)
    # Choose prompt based on whether this subagent has execute tool access
    subagent_tools = SUBAGENT_TOOLS.get(name, [])
    has_execute = "execute" in subagent_tools
    fs_prompt = SPINE_FILESYSTEM_EXEC_PROMPT if has_execute else SPINE_FILESYSTEM_PROMPT
    fs_mw = FilesystemMiddleware(backend=backend, system_prompt=fs_prompt)
    allowed_tool_names = SUBAGENT_TOOLS[name]
    tools = [t for t in fs_mw.tools if t.name in allowed_tool_names]

    # ── Build spec ───────────────────────────────────────────────
    spec: dict[str, Any] = {
        "name": name,
        "description": SUBAGENT_DESCRIPTIONS[name],
        "system_prompt": SUBAGENT_PROMPTS[name],
        "model": model,
        "tools": tools,
    }
    # Only researcher gets response_format — structured summaries prevent
    # raw file contents from bloating the parent agent's context.
    # slice-implementer and slice-verifier need free-form tool use.
    if name == "researcher":
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
