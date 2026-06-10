"""Pure planning schemas + the deterministic section skeleton for synthesis.

This module holds the *graph-free* half of the distributed synthesis hierarchy
(design Revision 2, §2.2): the Pydantic plan/result schemas and
:func:`deterministic_section_plan`, the deterministic floor the documentation
manager only *refines* and falls back to whenever the LLM fails, returns an
empty/incoherent plan, or is unavailable.

The skeleton turns a compact :func:`spine.work.onboarding.manifest_index` into
an ordered list of sections — one per natural unit per document:

- ``ARCHITECTURE_MAP`` — one section per module (top-K already ranked + tail
  grouped by the index); fragment = that module's boundary + its edges.
- ``CODING_GUIDELINES`` — one section per pattern category; fragment = that
  category's findings (with evidence).
- ``PROJECT_DEFINITION`` — one section per core domain (domains below
  :data:`_MIN_DOMAIN_SYMBOLS` symbols are merged into a single "Supporting
  Domains" section so no worker is asked to write prose about a near-empty
  domain); fragment = that domain's module roles.
- ``SPINE_ASSISTANCE_REQUIREMENTS`` — 1-2 sections; fragment = size/budget
  signals only.
- greenfield — a fixed minimal plan (one section per doc, no manifest content).

Every produced section carries non-empty ``fragment_keys`` and ``instruction``
so a section worker always has something concrete to resolve and write.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from spine.work.onboarding.synthesis_tools import ONBOARDING_DOC_NAMES


class SectionPlan(BaseModel):
    """One planned section of one onboarding document.

    A :class:`SectionPlan` references manifest entries by **stable key**
    (module names / pattern categories / domain ids) via ``fragment_keys`` — it
    NEVER carries manifest content. The section worker calls
    :func:`spine.work.onboarding.manifest_index.resolve_fragment` with these
    keys to obtain its bounded fragment.
    """

    doc_id: str = Field(
        description=(
            "Which onboarding document this section belongs to. One of: "
            + ", ".join(ONBOARDING_DOC_NAMES)
        )
    )
    order: int = Field(
        description="0-based position of this section within its document."
    )
    title: str = Field(description="Short human-readable section heading.")
    fragment_keys: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Stable manifest selectors for this section's fragment, e.g. "
            '{"doc_id": "ARCHITECTURE_MAP", "modules": ["spine.work"]}. '
            "Resolved by resolve_fragment(); never contains manifest content."
        ),
    )
    instruction: str = Field(
        description="Concrete writing instruction for the section worker."
    )


class SectionEntry(BaseModel):
    """One concrete item within a section (a module, convention, domain, …)."""

    name: str = Field(
        min_length=1,
        description="The item's name (module, convention, domain, or component).",
    )
    path: str = Field(
        default="",
        description="Repository path for the item, if it has one (else empty).",
    )
    description: str = Field(
        min_length=1,
        description=(
            "What this item is/does, grounded in the supplied fragment. Inline "
            "markdown (backticks, bold) is allowed."
        ),
    )


class SectionContent(BaseModel):
    """The section worker's structured output — content fields ONLY.

    This is the schema bound via ``with_structured_output`` for each Tier-B
    worker call. It deliberately carries NO ``doc_id``/``order``/``status`` —
    the worker already knows those from its ``active_section`` input, and
    asking the model to echo them only invites hallucination (trace
    019eaf55: workers returned envelope-only JSON with invented orders).
    ``overview`` is required and non-empty so json_schema enforcement makes a
    metadata-only response an invalid generation. The markdown document shape
    is rendered deterministically by :func:`render_section_markdown` — the
    model never authors document structure.
    """

    overview: str = Field(
        min_length=1,
        description=(
            "1-3 paragraphs of prose for this section, grounded ONLY in the "
            "supplied fragment. Inline markdown (backticks, bold) is allowed; "
            "no headings."
        ),
    )
    entries: list[SectionEntry] = Field(
        default_factory=list,
        description=(
            "One entry per concrete item the section covers (module, "
            "convention, domain, component). Empty if the section is pure prose."
        ),
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Optional caveats or pointers worth calling out (short strings).",
    )


def render_section_markdown(title: str, content: SectionContent) -> str:
    """Render one section's markdown from its structured content.

    Deterministic — the renderer owns the markdown shape (mirrors
    ``_render_spec_markdown`` in :mod:`spine.agents.specify_tools`): ``##``
    section heading from the PLAN title (never the model), overview prose,
    one ``###`` subsection per entry, then a trailing notes list. Sections are
    concatenated under the assembler's ``# <Document Title>``, so headings
    start at ``##``.
    """
    parts: list[str] = []
    if title.strip():
        parts.append(f"## {title.strip()}")
    if content.overview.strip():
        parts.append(content.overview.strip())
    for entry in content.entries:
        name = entry.name.strip()
        description = entry.description.strip()
        if not (name and description):
            continue
        path = entry.path.strip()
        heading = f"### {name}" + (f" (`{path}`)" if path else "")
        parts.append(f"{heading}\n\n{description}")
    cleaned_notes = [n.strip() for n in content.notes if n and n.strip()]
    if cleaned_notes:
        parts.append("**Notes:**\n" + "\n".join(f"- {n}" for n in cleaned_notes))
    return "\n\n".join(parts)


class SectionResult(BaseModel):
    """One section's RESULT carried on the graph state channel.

    NOT an LLM output schema — the worker authors a :class:`SectionContent`,
    renders it via :func:`render_section_markdown`, and emits this shape (as a
    plain dict) with its own known ``doc_id``/``order``.
    """

    doc_id: str = Field(description="Which onboarding document this section is for.")
    order: int = Field(description="Position within the document (for assembly).")
    markdown: str = Field(
        default="",
        description="The rendered markdown for this section.",
    )
    status: str = Field(
        default="ok",
        description='"ok" or "error" (generic reason; never raw exception text).',
    )


class SectionPlanSet(BaseModel):
    """A whole plan: the ordered sections for all four documents.

    This is the structured-output schema the documentation manager fills in one
    bare LLM call. Tier-A coercion validates against this; on failure the caller
    falls back to :func:`deterministic_section_plan`.
    """

    sections: list[SectionPlan] = Field(
        default_factory=list,
        description="Ordered sections across all four onboarding documents.",
    )


# ── Deterministic skeleton ──────────────────────────────────────────────────

# A core domain whose backing module carries fewer symbols than this is merged
# into one shared "Supporting Domains" section instead of getting a dedicated
# section. A worker given a 1-4 symbol domain has nothing to write — trace
# 019eaf55 showed such sections deterministically coming back content-free.
_MIN_DOMAIN_SYMBOLS = 5


def _greenfield_plan() -> list[dict[str, Any]]:
    """The fixed minimal plan for greenfield projects (no manifest content).

    One section per document; fragments resolve to size/best-practice signals
    only because a greenfield manifest has no boundaries/patterns/domains.
    """
    plan: list[dict[str, Any]] = []
    for doc_id in ONBOARDING_DOC_NAMES:
        plan.append(
            {
                "doc_id": doc_id,
                "order": 0,
                "title": doc_id.replace("_", " ").title(),
                "fragment_keys": {"doc_id": doc_id, "greenfield": True},
                "instruction": (
                    f"This is a greenfield project. Author the {doc_id} document "
                    "from the declared tech stack and best-practice defaults; "
                    "there is no existing codebase to describe."
                ),
            }
        )
    return plan


def _architecture_sections(index: dict[str, Any]) -> list[dict[str, Any]]:
    """One ARCHITECTURE_MAP section per module surfaced in the index."""
    sections: list[dict[str, Any]] = []
    modules = list(index.get("modules", []) or [])
    if not modules:
        return [
            {
                "doc_id": "ARCHITECTURE_MAP",
                "order": 0,
                "title": "System Overview",
                "fragment_keys": {"doc_id": "ARCHITECTURE_MAP", "modules": []},
                "instruction": (
                    "Describe the overall architecture from the available module "
                    "boundaries and dependency edges."
                ),
            }
        ]
    for order, mod in enumerate(modules):
        name = mod.get("name", "")
        role = mod.get("role", "")
        sections.append(
            {
                "doc_id": "ARCHITECTURE_MAP",
                "order": order,
                "title": f"Module: {name}",
                "fragment_keys": {
                    "doc_id": "ARCHITECTURE_MAP",
                    "modules": [name],
                },
                "instruction": (
                    f"Document the '{name}' module ({role or 'role: see fragment'}): "
                    "its responsibility, key symbols, and how it depends on / is "
                    "depended on by other modules. Use only the supplied fragment."
                ),
            }
        )
    return sections


def _coding_sections(index: dict[str, Any]) -> list[dict[str, Any]]:
    """One CODING_GUIDELINES section per pattern category in the index."""
    categories = list(index.get("pattern_categories", []) or [])
    if not categories:
        return [
            {
                "doc_id": "CODING_GUIDELINES",
                "order": 0,
                "title": "Conventions",
                "fragment_keys": {"doc_id": "CODING_GUIDELINES", "categories": []},
                "instruction": (
                    "Summarise the project's coding conventions from the available "
                    "pattern findings; if none, state recommended defaults for the "
                    "tech stack."
                ),
            }
        ]
    sections: list[dict[str, Any]] = []
    for order, cat in enumerate(categories):
        sections.append(
            {
                "doc_id": "CODING_GUIDELINES",
                "order": order,
                "title": f"{cat.replace('_', ' ').title()} Conventions",
                "fragment_keys": {
                    "doc_id": "CODING_GUIDELINES",
                    "categories": [cat],
                },
                "instruction": (
                    f"Document the project's '{cat}' convention from the supplied "
                    "findings and their evidence. Show the established pattern as a "
                    "rule contributors should follow."
                ),
            }
        )
    return sections


def _project_sections(index: dict[str, Any]) -> list[dict[str, Any]]:
    """One PROJECT_DEFINITION section per core domain in the index."""
    domains = list(index.get("core_domains", []) or [])
    if not domains:
        return [
            {
                "doc_id": "PROJECT_DEFINITION",
                "order": 0,
                "title": "Project Overview",
                "fragment_keys": {"doc_id": "PROJECT_DEFINITION", "domains": []},
                "instruction": (
                    "Define what this project is and does, from the module roles "
                    "and tech stack in the supplied fragment."
                ),
            }
        ]
    # Domains map to modules of the same name (see resolve_fragment). Domains
    # whose module carries too few symbols (or fell into the index's collapsed
    # tail, so we have no count) are merged into ONE shared section.
    symbol_counts = {
        m.get("name", ""): int(m.get("symbol_count", 0) or 0)
        for m in (index.get("modules", []) or [])
    }
    primary = [d for d in domains if symbol_counts.get(d, 0) >= _MIN_DOMAIN_SYMBOLS]
    minor = [d for d in domains if d not in primary]

    sections: list[dict[str, Any]] = []
    for order, domain in enumerate(primary):
        sections.append(
            {
                "doc_id": "PROJECT_DEFINITION",
                "order": order,
                "title": f"Domain: {domain}",
                "fragment_keys": {
                    "doc_id": "PROJECT_DEFINITION",
                    "domains": [domain],
                },
                "instruction": (
                    f"Define the '{domain}' domain: its purpose and the modules "
                    "that implement it, using only the supplied module roles."
                ),
            }
        )
    if minor:
        sections.append(
            {
                "doc_id": "PROJECT_DEFINITION",
                "order": len(primary),
                "title": "Supporting Domains",
                "fragment_keys": {
                    "doc_id": "PROJECT_DEFINITION",
                    "domains": list(minor),
                },
                "instruction": (
                    "Briefly define each of these smaller supporting domains — "
                    + ", ".join(f"'{d}'" for d in minor)
                    + " — one short entry apiece, using only the supplied "
                    "module roles."
                ),
            }
        )
    return sections


def _spine_sections(index: dict[str, Any]) -> list[dict[str, Any]]:
    """1-2 SPINE_ASSISTANCE_REQUIREMENTS sections from size/budget signals.

    Always emits an overview section; emits a second "hot spots" section when the
    repo is large enough that budget guidance is worthwhile.
    """
    totals = dict(index.get("totals", {}) or {})
    symbol_count = int(totals.get("symbol_count", 0) or 0)
    module_count = int(totals.get("module_count", 0) or 0)

    sections: list[dict[str, Any]] = [
        {
            "doc_id": "SPINE_ASSISTANCE_REQUIREMENTS",
            "order": 0,
            "title": "Assistance Overview",
            "fragment_keys": {"doc_id": "SPINE_ASSISTANCE_REQUIREMENTS"},
            "instruction": (
                "Describe how an AI assistant should approach this repository: "
                "its size, the largest modules to be careful around, and where "
                "context budget should be spent. Use only the supplied signals."
            ),
        }
    ]
    if symbol_count > 500 or module_count > 8:
        sections.append(
            {
                "doc_id": "SPINE_ASSISTANCE_REQUIREMENTS",
                "order": 1,
                "title": "Hot Spots & Budget Guidance",
                "fragment_keys": {"doc_id": "SPINE_ASSISTANCE_REQUIREMENTS"},
                "instruction": (
                    "Call out the largest modules and any analysis notes so an "
                    "assistant avoids loading oversized context. Use only the "
                    "supplied size/budget signals."
                ),
            }
        )
    return sections


def deterministic_section_plan(index: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """Produce the deterministic section skeleton from a compact *index*.

    This is the floor the documentation manager refines and the guaranteed
    fallback when the manager LLM fails. For greenfield projects it returns the
    fixed minimal plan (one section per doc). For brownfield it produces one
    section per natural unit per document (module / pattern category / domain)
    plus 1-2 assistance sections.

    Args:
        index: Output of :func:`spine.work.onboarding.manifest_index.manifest_index`.
        mode: ``"greenfield"`` or ``"brownfield"`` (anything else is treated as
            brownfield).

    Returns:
        A list of section dicts, each with the shape of :class:`SectionPlan`
        (``doc_id``, ``order``, ``title``, ``fragment_keys``, ``instruction``).
        Every section has non-empty ``fragment_keys`` and ``instruction``.
    """
    if mode == "greenfield":
        return _greenfield_plan()

    plan: list[dict[str, Any]] = []
    plan.extend(_architecture_sections(index))
    plan.extend(_coding_sections(index))
    plan.extend(_project_sections(index))
    plan.extend(_spine_sections(index))
    return plan
