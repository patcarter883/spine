"""SPINE FeatureSlice synthesis module.

Provides heuristic and agent-based decomposition of requirements into
FeatureSlice objects with dependency edges.  The legacy SwarmDAGExecutor
class was removed in Phase 4 — Deep Agents is now the sole execution path.
"""

import os
import re
from typing import Optional, Any

from .types import FeatureSlice


def _extract_components(requirement: str) -> list[str]:
    """Extract component hints from a requirement string.

    Splits on common connectors and heuristics to identify distinct
    components or features mentioned in the requirement.
    """
    # Normalize and split on common connectors
    text = requirement.lower()
    separators = [" and ", " with ", " including ", " plus ", ", "]
    parts = [text.strip()]
    for sep in separators:
        new_parts = []
        for p in parts:
            if sep in p:
                new_parts.extend(s.strip() for s in p.split(sep) if s.strip())
            else:
                new_parts.append(p)
        parts = new_parts

    # Filter out very short fragments and deduplicate
    seen: set[str] = set()
    components: list[str] = []
    for p in parts:
        if len(p) > 5 and p not in seen:
            seen.add(p)
            components.append(p)

    return components if components else [requirement[:120]]


def _extract_requirements(requirement: str) -> list[str]:
    """Extract key requirement items from a requirement string."""
    # Heuristic: split on semicolons, newlines, or bullet-like patterns
    text = requirement.strip()
    items = re.split(r"[;\n]|\b(?:•|-\s+)\s*", text)
    items = [i.strip() for i in items if i.strip() and len(i.strip()) > 5]
    return items if items else [requirement[:200]]


def _estimate_complexity(requirement: str) -> str:
    """Estimate project complexity from the requirement text.

    Returns one of: 'low', 'medium', 'high'.
    """
    text = requirement.lower()
    high_keywords = [
        "microservice", "distributed", "scalable", "high-throughput",
        "real-time", "production-grade", "enterprise", "multi-tenant",
        "kubernetes", "cluster", "load-balanc", "failover", "disaster",
    ]
    medium_keywords = [
        "api", "web", "full-stack", "database", "auth", "authentication",
        "rest", "graphql", "frontend", "backend", "integration",
    ]

    high_count = sum(1 for kw in high_keywords if kw in text)
    medium_count = sum(1 for kw in medium_keywords if kw in text)
    word_count = len(text.split())

    if high_count >= 2 or word_count > 60:
        return "high"
    if high_count >= 1 or medium_count >= 2 or word_count > 30:
        return "medium"
    return "low"


def synthesize_slices(
    requirement: str,
    context: dict[str, Any],
    agent_provider: Optional[Any] = None,
) -> list[FeatureSlice]:
    """Produce FeatureSlice objects from requirement + context.

    Uses the agent provider when available for intelligent decomposition.
    Falls back to a heuristic slicer that produces 2-6 slices.

    Args:
        requirement: The original requirement text.
        context: Execution context (analysis results, tech research, etc.).
        agent_provider: Optional agent provider for decomposition.

    Returns:
        List of FeatureSlice objects with dependency edges.
    """
    # ── Agent path ─────────────────────────────────────────────────
    if agent_provider and not isinstance(agent_provider, dict) and agent_provider.enabled:
        try:
            prompt = _build_slice_synthesis_prompt(requirement, context)
            result = agent_provider.execute(prompt, workdir=os.getcwd(), timeout=120)
            if result.success and result.output:
                return _parse_llm_slices(result.output)
        except Exception:
            pass  # fall through to heuristic

    # ── Heuristic path ────────────────────────────────────────────
    return _heuristic_slices(requirement, context)


def _build_slice_synthesis_prompt(requirement: str, context: dict[str, Any]) -> str:
    """Build a prompt asking the LLM to decompose into FeatureSlices."""
    analysis = context.get("requirement", requirement)
    tech = context.get("tech_research", "Not available")
    risk = context.get("risk_assessment", "Not available")

    return f"""You are a software architect decomposing a project into feature slices.

REQUIREMENT:
{requirement}

ANALYSIS:
{analysis}

TECH RESEARCH:
{tech}

RISK ASSESSMENT:
{risk}

Decompose this into 2-6 independent feature slices. Each slice should be a
cohesive unit of work that a single developer could implement in a focused
session without coordinating with others working in parallel.

Output JSON array. Each element:
{{
  "id": "short-kebab-id",
  "description": "What to build at feature granularity (NOT file-level)",
  "scope": ["module/dir1/", "module/dir2/"],
  "depends_on": ["other-slice-id"],
  "agent_role": "coder|test_engineer|reviewer",
  "acceptance": ["Criterion 1", "Criterion 2"]
}}

Rules:
- Slices must be independently implementable (one developer, one session)
- If you need to read source files to decompose, the slice is too small
- depends_on captures the DAG edges — the real architectural dependencies
- Prefer fewer, richer slices over many micro-tasks

JSON:"""


def _parse_llm_slices(raw: str) -> list[FeatureSlice]:
    """Parse LLM response into FeatureSlice objects."""
    import json

    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        items = json.loads(text)
        if not isinstance(items, list):
            items = [items]
        return [
            FeatureSlice.from_dict(item)
            for item in items
            if isinstance(item, dict) and "id" in item and "description" in item
        ]
    except json.JSONDecodeError:
        return _heuristic_slices("parsed from LLM", {})


def _heuristic_slices(requirement: str, context: dict[str, Any]) -> list[FeatureSlice]:
    """Fallback heuristic slicer — produces slices from requirement keywords."""
    complexity = _estimate_complexity(requirement)
    components = _extract_components(requirement)

    if complexity == "high":
        # 4-6 slices: core, each major component, integration, tests
        slices = [
            FeatureSlice(
                id="core-foundation",
                description="Implement core data models, configuration, and shared utilities",
                scope=["core/", "models/", "config/"],
                depends_on=[],
                agent_role="coder",
                acceptance=["Core models compile and import correctly", "Config loads without errors"],
            ),
        ]
        for i, comp in enumerate(components[:4]):
            slug = comp.replace(" ", "-")[:30]
            slices.append(FeatureSlice(
                id=f"feature-{slug}",
                description=f"Implement {comp} module with full business logic",
                scope=[f"{comp.split()[0]}/"],
                depends_on=["core-foundation"],
                agent_role="coder",
                acceptance=[f"{comp} module works end-to-end", "No lint errors"],
            ))
        slices.append(FeatureSlice(
            id="integration-wiring",
            description="Wire all feature modules together, add API layer and cross-cutting concerns",
            scope=["api/", "routes/", "middleware/"],
            depends_on=[s.id for s in slices if s.id != "core-foundation"],
            agent_role="coder",
            acceptance=["All modules importable from main entrypoint", "API routes respond"],
        ))
        slices.append(FeatureSlice(
            id="test-coverage",
            description="Write unit and integration tests for all modules",
            scope=["tests/"],
            depends_on=["integration-wiring"],
            agent_role="test_engineer",
            acceptance=["All tests pass", "No regressions"],
        ))

    elif complexity == "medium":
        slices = [
            FeatureSlice(
                id="core-impl",
                description="Implement core models and primary business logic",
                scope=["models/", "services/"],
                depends_on=[],
                agent_role="coder",
                acceptance=["Models serialize/deserialize correctly", "Core logic passes basic validation"],
            ),
            FeatureSlice(
                id="feature-modules",
                description="Implement feature modules and API layer",
                scope=["routes/", "api/"],
                depends_on=["core-impl"],
                agent_role="coder",
                acceptance=["API endpoints respond", "Feature modules integrate with core"],
            ),
            FeatureSlice(
                id="tests",
                description="Write unit and integration tests",
                scope=["tests/"],
                depends_on=["feature-modules"],
                agent_role="test_engineer",
                acceptance=["All tests pass", "No regressions"],
            ),
        ]

    else:  # low
        slices = [
            FeatureSlice(
                id="implementation",
                description="Implement the required feature end-to-end",
                scope=["."],
                depends_on=[],
                agent_role="coder",
                acceptance=["Feature works as described in requirement"],
            ),
            FeatureSlice(
                id="verification",
                description="Write tests and validate the implementation",
                scope=["tests/"],
                depends_on=["implementation"],
                agent_role="test_engineer",
                acceptance=["Tests pass", "Implementation matches requirement"],
            ),
        ]

    return slices


__all__ = [
    "synthesize_slices",
    "_extract_components",
    "_extract_requirements",
    "_estimate_complexity",
    "_build_slice_synthesis_prompt",
    "_parse_llm_slices",
    "_heuristic_slices",
]
