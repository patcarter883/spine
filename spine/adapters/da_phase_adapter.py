"""Deep Agents phase adapter for SPINE.

Constructs create_deep_agent() instances with phase-specific configuration,
SubAgent specs from FeatureSlices, and middleware stacks for each SPINE phase.

This module is the bridge between SPINE's state machine (which handles
routing, critic gates, and persistence) and Deep Agents' agent loop (which
handles tool execution, context compaction, and subagent delegation).

Key design decisions (from HYBRID_ARCHITECTURE.md):
  1. State machine stays at the top level — DA is NOT the outer loop.
  2. One create_deep_agent() per phase — preserves phase isolation.
  3. FeatureSlices map to SubAgent specs during EXECUTION.
  4. Providers through config, not state — eliminates serialization bugs.
  5. Local models bypass OpenCode entirely via init_chat_model().
"""

from __future__ import annotations

import os
import logging
from typing import Any

from deepagents import create_deep_agent, SubAgent
from deepagents.backends import LocalShellBackend

from ..middleware.critic_gate import CriticGateMiddleware
from ..middleware.step_limit import StepLimitMiddleware
from ..middleware.message_queue import MessageQueueMiddleware
from ..models.types import FeatureSlice
from ..core.constants import PhaseName

logger = logging.getLogger(__name__)


# ── Prompt templates ────────────────────────────────────────────────────

PLANNING_SYSTEM_PROMPT = """You are a software planning agent in the SPINE workflow.

Your job is to analyze the given requirement and produce a structured execution plan.

You have access to subagents for specialized analysis:
- explorer: Analyzes requirements and existing codebase (read-only)
- sme: Researches technology stack and dependencies (read-only + web search)
- analyst: Identifies risks, constraints, and edge cases (read-only)

WORKFLOW:
1. Delegate analysis to your subagents using the task() tool
2. Synthesize their findings into a coherent plan
3. Decompose the work into FeatureSlices (independent, delegatable units)
4. Output your plan with PLAN_COMPLETE marker when done

FeatureSlice format (output as JSON array):
{
  "id": "short-kebab-id",
  "description": "What to build at feature granularity",
  "scope": ["module/dir1/", "module/dir2/"],
  "depends_on": ["other-slice-id"],
  "agent_role": "coder|test_engineer|reviewer",
  "acceptance": ["Criterion 1", "Criterion 2"]
}

Rules:
- Slices must be independently implementable (one agent, one session)
- depends_on captures real architectural dependencies (DAG edges)
- Prefer fewer, richer slices over many micro-tasks
- Each slice should be feature-granularity, NOT file-level

End your plan with: PLAN_COMPLETE
"""

EXPLORER_SYSTEM_PROMPT = """You are a requirements explorer. Analyze the given requirement
and the existing codebase to understand what needs to be built. Focus on:
- What the requirement asks for
- What already exists in the codebase
- What patterns and conventions the project follows
- What interfaces the new feature must conform to

You have read-only tools (ls, glob, grep, read_file). Use them to explore the codebase.
Return a structured analysis summary.
"""

SME_SYSTEM_PROMPT = """You are a subject matter expert. Research the technology stack
and dependencies relevant to the requirement. Focus on:
- Which libraries, frameworks, and APIs are relevant
- What the project's existing tech stack provides
- What third-party dependencies are needed
- Best practices and patterns for this type of feature

You have read-only tools plus web search. Research thoroughly.
Return a technology research summary with specific recommendations.
"""

ANALYST_SYSTEM_PROMPT = """You are a risk analyst. Identify risks, constraints,
and edge cases for the requirement. Focus on:
- Technical risks (compatibility, performance, security)
- Edge cases and failure scenarios
- Project-specific constraints (architecture, conventions)
- What could go wrong during implementation

You have read-only tools. Analyze the codebase for potential issues.
Return a risk assessment with categorized findings.
"""

EXECUTION_SYSTEM_PROMPT = """You are a software execution orchestrator in the SPINE workflow.

Your job is to implement the planned feature slices. You have access to subagents
for each slice — delegate implementation work to them using the task() tool.

WORKFLOW:
1. Review the plan and feature slices
2. Start with slices that have no dependencies
3. Delegate each slice to the appropriate subagent
4. After each slice completes, move to slices whose dependencies are satisfied
5. Verify all acceptance criteria are met

For each slice, the subagent has:
- Isolated context window (no context pollution from other slices)
- Full file + shell tools for implementation
- The slice's description, scope, and acceptance criteria

Coordinate the execution order based on slice dependencies (DAG).
"""

VERIFICATION_SYSTEM_PROMPT = """You are a verification agent in the SPINE workflow.

Your job is to verify that the implementation meets all acceptance criteria.

You have access to subagents:
- reviewer: Reviews code changes for correctness and quality
- test_engineer: Runs tests and validates acceptance criteria

WORKFLOW:
1. Delegate code review to the reviewer subagent
2. Delegate test execution to the test_engineer subagent
3. Compile verification results
4. If any criteria fail, report which ones and why
5. If all criteria pass, report VERIFICATION_PASSED

Output format:
- VERIFICATION_PASSED if all criteria met
- VERIFICATION_FAILED with list of failed criteria if any fail
"""

REVIEWER_SYSTEM_PROMPT = """You are a code reviewer. Review the implementation
for correctness, quality, and adherence to best practices. Focus on:
- Logic correctness
- Edge case handling
- Code style and conventions
- Security considerations
- Performance implications

Use read-only and shell tools to inspect code and run linters.
Return a review summary with findings categorized by severity.
"""

TEST_ENGINEER_SYSTEM_PROMPT = """You are a test engineer. Validate that the
implementation meets all acceptance criteria. Focus on:
- Running existing tests to check for regressions
- Writing new tests for the implemented feature
- Verifying each acceptance criterion specifically
- Checking for common failure patterns

Use shell and file tools to run tests and inspect results.
Return a test results summary with pass/fail for each criterion.
"""


# ── Backend selection ────────────────────────────────────────────────────

def get_backend(
    phase: str = PhaseName.EXECUTION,
    root_dir: str | None = None,
) -> Any:
    """Select a DA backend based on execution context.

    Args:
        phase: Current SPINE phase — determines backend type.
        root_dir: Root directory for filesystem backends (defaults to cwd).

    Returns:
        A BackendProtocol instance suitable for the phase.
    """
    root_dir = root_dir or os.getcwd()

    if phase == PhaseName.PLANNING:
        # Planning needs read access to the real codebase so explorer/sme/analyst
        # subagents can ls, glob, grep, and read files.  StateBackend operates on
        # an in-memory files channel that starts empty, so the subagents find
        # nothing.  LocalShellBackend gives real filesystem access; the planning
        # prompts already instruct agents to use read-only tools.
        return LocalShellBackend(root_dir=root_dir, virtual_mode=False)

    # Execution and verification need real filesystem access
    return LocalShellBackend(root_dir=root_dir, virtual_mode=False)


# ── Helper: resolve providers from config ────────────────────────────────

def _get_providers_from_config(config: Any) -> dict[str, Any]:
    """Resolve providers exclusively from config — never from state.

    Args:
        config: LangGraph RunnableConfig dict.

    Returns:
        Dict mapping provider category to provider instance.
    """
    if config:
        return config.get("configurable", {}).get("providers", {})
    return {}


def _get_chat_model(providers: dict[str, Any]) -> Any:
    """Extract a BaseChatModel from the providers dict.

    Handles both DeepAgentsModelProvider (which exposes chat_model)
    and legacy LLMProvider (which we'll phase out).
    """
    llm_provider = providers.get("llm")
    if llm_provider is None:
        return None

    # DeepAgentsModelProvider — direct chat model
    if hasattr(llm_provider, "chat_model"):
        return llm_provider.chat_model

    # Legacy path: try to create via init_chat_model from provider config
    # This will be removed in Phase 4 cleanup
    if hasattr(llm_provider, "_config"):
        from langchain.chat_models import init_chat_model
        model_str = llm_provider._config.get("model", "openai:gpt-4")
        kwargs = {}
        for key in ("base_url", "api_key", "temperature", "max_tokens"):
            if key in llm_provider._config:
                kwargs[key] = llm_provider._config[key]
        return init_chat_model(model_str, **kwargs)

    return None


# ── Planning agent ───────────────────────────────────────────────────────

def create_planning_agent(
    requirement: str,
    providers: dict[str, Any],
    backend: Any | None = None,
    max_steps: int = 50,
) -> Any:
    """Construct a DA agent for the PLANNING phase.

    The planning agent delegates analysis to explorer, SME, and analyst
    subagents (all read-only), then synthesizes a plan with FeatureSlices.

    Args:
        requirement: The user's requirement text.
        providers: Provider dict from config (must include "llm").
        backend: DA backend (defaults to LocalShellBackend for planning).
        max_steps: Maximum model call steps before wrap-up notification.

    Returns:
        A compiled LangGraph StateGraph (DA agent) ready to invoke.
    """
    chat_model = _get_chat_model(providers)
    if chat_model is None:
        raise ValueError("No LLM provider configured — cannot create planning agent")

    backend = backend or get_backend(phase=PhaseName.PLANNING)

    subagents = [
        SubAgent(
            name="explorer",
            description="Analyzes requirements and existing codebase. "
                        "Use when you need to understand what exists and what's needed.",
            system_prompt=EXPLORER_SYSTEM_PROMPT,
        ),
        SubAgent(
            name="sme",
            description="Researches technology stack and dependencies. "
                        "Use when you need technology recommendations and stack analysis.",
            system_prompt=SME_SYSTEM_PROMPT,
        ),
        SubAgent(
            name="analyst",
            description="Identifies risks, constraints, and edge cases. "
                        "Use when you need risk assessment and constraint analysis.",
            system_prompt=ANALYST_SYSTEM_PROMPT,
        ),
    ]

    middleware = [
        CriticGateMiddleware(llm_provider=providers.get("llm")),
        StepLimitMiddleware(max_steps=max_steps),
        MessageQueueMiddleware(),
    ]

    # Planning is read-only: deny writes anywhere, allow reads under project root.
    # First-match-wins, so the deny-write rule must come before any broader allows.
    from deepagents import FilesystemPermission
    permissions = [
        FilesystemPermission(
            operations=["write"],
            paths=["/**"],
            mode="deny",
        ),
    ]

    agent = create_deep_agent(
        model=chat_model,
        system_prompt=PLANNING_SYSTEM_PROMPT,
        subagents=subagents,
        middleware=middleware,
        backend=backend,
        permissions=permissions,
        name="spine-planning",
    )

    return agent


# ── Execution agent ──────────────────────────────────────────────────────

def create_execution_agent(
    requirement: str,
    providers: dict[str, Any],
    feature_slices: list[FeatureSlice],
    planning_context: dict[str, Any],
    spec_content: str = "",
    backend: Any | None = None,
    max_steps: int = 100,
) -> Any:
    """Construct a DA agent for the EXECUTION phase.

    Each FeatureSlice becomes a SubAgent spec. The main execution agent
    orchestrates slice execution order based on dependencies.

    Args:
        requirement: The user's requirement text.
        providers: Provider dict from config.
        feature_slices: List of FeatureSlice objects from planning.
        planning_context: Structured planning context dict.
        spec_content: Content of the spec file (from disk).
        backend: DA backend (defaults to LocalShellBackend).
        max_steps: Maximum model call steps.

    Returns:
        A compiled DA agent ready to invoke.
    """
    chat_model = _get_chat_model(providers)
    if chat_model is None:
        raise ValueError("No LLM provider configured — cannot create execution agent")

    backend = backend or get_backend(phase=PhaseName.EXECUTION)

    # Build SubAgent specs from FeatureSlices
    subagents = []
    for slice_obj in feature_slices:
        acceptance_text = (
            chr(10).join(f"- {c}" for c in slice_obj.acceptance)
            if slice_obj.acceptance
            else "Feature works as described"
        )
        context_text = (
            "PLANNING CONTEXT:" + chr(10) + _format_planning_context(planning_context)
            if planning_context
            else ""
        )
        spec_text = "SPEC:" + chr(10) + spec_content if spec_content else ""
        slice_prompt = f"""You are implementing feature slice: {slice_obj.id}

DESCRIPTION: {slice_obj.description}

SCOPE: {', '.join(slice_obj.scope) if slice_obj.scope else 'project root'}

ACCEPTANCE CRITERIA:
{acceptance_text}

{context_text}

{spec_text}

Implement this feature slice end-to-end. Write code, run tests, and verify
the acceptance criteria are met. Use file and shell tools as needed.
"""
        subagents.append(SubAgent(
            name=slice_obj.id,
            description=slice_obj.description,
            system_prompt=slice_prompt,
        ))

    middleware = [
        StepLimitMiddleware(max_steps=max_steps),
        MessageQueueMiddleware(),
    ]

    agent = create_deep_agent(
        model=chat_model,
        system_prompt=EXECUTION_SYSTEM_PROMPT,
        subagents=subagents,
        middleware=middleware,
        backend=backend,
        name="spine-execution",
    )

    return agent


# ── Verification agent ───────────────────────────────────────────────────

def create_verification_agent(
    requirement: str,
    providers: dict[str, Any],
    backend: Any | None = None,
    max_steps: int = 50,
) -> Any:
    """Construct a DA agent for the VERIFICATION phase.

    Has reviewer and test_engineer subagents for code review and test execution.

    Args:
        requirement: The user's requirement text.
        providers: Provider dict from config.
        backend: DA backend (defaults to LocalShellBackend).
        max_steps: Maximum model call steps.

    Returns:
        A compiled DA agent ready to invoke.
    """
    chat_model = _get_chat_model(providers)
    if chat_model is None:
        raise ValueError("No LLM provider configured — cannot create verification agent")

    backend = backend or get_backend(phase=PhaseName.VERIFICATION)

    subagents = [
        SubAgent(
            name="reviewer",
            description="Reviews code changes for correctness and quality. "
                        "Use when you need code review with findings by severity.",
            system_prompt=REVIEWER_SYSTEM_PROMPT,
        ),
        SubAgent(
            name="test_engineer",
            description="Runs tests and validates acceptance criteria. "
                        "Use when you need test execution and validation.",
            system_prompt=TEST_ENGINEER_SYSTEM_PROMPT,
        ),
    ]

    middleware = [
        StepLimitMiddleware(max_steps=max_steps),
        MessageQueueMiddleware(),
    ]

    agent = create_deep_agent(
        model=chat_model,
        system_prompt=VERIFICATION_SYSTEM_PROMPT,
        subagents=subagents,
        middleware=middleware,
        backend=backend,
        name="spine-verification",
    )

    return agent


# ── Utility ──────────────────────────────────────────────────────────────

def _format_planning_context(context: dict[str, Any]) -> str:
    """Format planning context dict into a readable string for agent prompts."""
    if not context:
        return ""

    sections = []
    for key, value in context.items():
        if isinstance(value, dict):
            # Try common sub-keys
            text = value.get("output", value.get("task_outputs", str(value)))
            if isinstance(text, list):
                text = "\n".join(str(t) for t in text)
        else:
            text = str(value)
        sections.append(f"## {key.upper()}\n{text}")

    return "\n\n".join(sections)


def _extract_feature_slices_from_da_result(result: dict[str, Any]) -> list[FeatureSlice]:
    """Extract FeatureSlice objects from a DA agent's final output.

    Scans the message history for JSON array output containing slice specs.
    Handles markdown code fences and prose surrounding the JSON block.
    """
    import json
    import re

    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = ""
        if hasattr(msg, "content"):
            content = msg.content or ""
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        if not content:
            continue

        # Strategy 1: Extract JSON from markdown code blocks (```json ... ```)
        code_block_re = re.compile(r"```(?:json)?\s*\n([\s\S]*?)\n```", re.DOTALL)
        for match in code_block_re.finditer(content):
            block_text = match.group(1).strip()
            try:
                items = json.loads(block_text)
                if isinstance(items, list):
                    slices = []
                    for item in items:
                        if isinstance(item, dict) and "id" in item:
                            slices.append(FeatureSlice.from_dict(item))
                    if slices:
                        return slices
            except json.JSONDecodeError:
                continue

        # Strategy 2: Try the whole content as JSON
        try:
            items = json.loads(content.strip())
            if isinstance(items, list):
                slices = []
                for item in items:
                    if isinstance(item, dict) and "id" in item:
                        slices.append(FeatureSlice.from_dict(item))
                if slices:
                    return slices
        except json.JSONDecodeError:
            pass

        # Strategy 3: Greedy regex for the largest JSON array
        json_pattern = re.compile(r"\[[\s\S]*\]", re.DOTALL)
        matches = json_pattern.findall(content)
        # Try longest match first (greedy will give us the outermost array)
        for match in reversed(matches):
            try:
                items = json.loads(match)
                if isinstance(items, list):
                    slices = []
                    for item in items:
                        if isinstance(item, dict) and "id" in item:
                            slices.append(FeatureSlice.from_dict(item))
                    if slices:
                        return slices
            except json.JSONDecodeError:
                continue

    return []


def _extract_planning_context_from_da_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract planning context from DA agent message history.

    Looks for subagent task results and synthesizes them into the
    planning_context dict format that SPINE expects.
    """
    context: dict[str, Any] = {}
    messages = result.get("messages", [])

    # Scan for ToolMessage results from subagents
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "tool":
            name = getattr(msg, "name", "")
            content = msg.content if hasattr(msg, "content") else str(msg)

            if name == "explorer":
                context["analysis"] = content
            elif name == "sme":
                context["tech_research"] = content
            elif name == "analyst":
                context["risk_assessment"] = content
            elif name in ("task", "general-purpose"):
                # General-purpose subagent results — classify heuristically
                has_analysis = "requirement" in content.lower() or "analyze" in content.lower()
                if "analysis" not in context and has_analysis:
                    context["analysis"] = content
                has_tech = "stack" in content.lower() or "technology" in content.lower()
                if "tech_research" not in context and has_tech:
                    context["tech_research"] = content

    return context


def _extract_verification_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract verification results from DA agent output.

    Scans message history for VERIFICATION_PASSED/VERIFICATION_FAILED markers.
    """
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = ""
        if hasattr(msg, "content"):
            content = msg.content or ""
        elif isinstance(msg, dict):
            content = msg.get("content", "")

        if "VERIFICATION_PASSED" in content:
            return {"passed": True, "details": content}
        if "VERIFICATION_FAILED" in content:
            return {"passed": False, "details": content, "failed_criteria": _extract_failed_criteria(content)}

    # Fallback: no explicit marker found
    return {"passed": None, "details": ""}


def _extract_failed_criteria(content: str) -> list[str]:
    """Extract failed acceptance criteria from verification output."""
    import re
    criteria = []
    # Look for bullet-pointed failures
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("-") or line.startswith("*") or re.match(r"^\d+\.", line):
            if "fail" in line.lower() or "not met" in line.lower() or "missing" in line.lower():
                criteria.append(line.lstrip("-*0123456789. ").strip())
    return criteria


__all__ = [
    "create_planning_agent",
    "create_execution_agent",
    "create_verification_agent",
    "get_backend",
    "_get_providers_from_config",
    "_get_chat_model",
    "_extract_feature_slices_from_da_result",
    "_extract_planning_context_from_da_result",
    "_extract_verification_result",
]