"""Typed onboarding contract — the ``RepoManifest`` dataclass tree.

The :class:`RepoManifest` is the SOLE typed contract between the analysis
backend (slice 1, producer) and the artifact synthesiser (slice 3, consumer).
It mirrors the codebase convention of frozen dataclasses (see
:class:`spine.agents.tools.ast_extract.Symbol`) — the project does not use
pydantic for internal data models, only for tool ``ArgsSchema`` definitions.

The manifest is persisted as ``repo_manifest.json`` via the ``ArtifactStore``
under phase ``"onboarding"`` so re-runs overwrite idempotently and slice 3 can
load it without re-analysing. :meth:`RepoManifest.to_dict` /
:meth:`RepoManifest.from_dict` provide the JSON round-trip used for that
persistence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SymbolRef:
    """A lightweight reference to a single extracted symbol.

    Carries only the metadata needed by the synthesiser — never raw source
    text — so the manifest stays small and the context window is protected.
    """

    file_path: str  # repo-relative, e.g. "spine/work/dispatcher.py"
    symbol_name: str
    symbol_type: str  # "function" | "class" | "method" | "interface"
    lang: str  # "python" | "php" | "typescript"
    summary: str = ""  # from VectorStore enriched_summary when available, else ""


@dataclass(frozen=True)
class ModuleBoundary:
    """A logical module/package boundary within the repository."""

    name: str  # logical module/package, e.g. "spine.work"
    path: str  # repo-relative dir, e.g. "spine/work"
    role: str  # short prose: what this module owns
    key_symbols: list[SymbolRef] = field(default_factory=list)


@dataclass(frozen=True)
class DependencyEdge:
    """A directed dependency between two symbols or modules."""

    src: str  # symbol or module name
    dst: str  # symbol or module name it calls/uses
    kind: str  # "calls" | "imports" | "depends_on"


@dataclass(frozen=True)
class PatternFinding:
    """An extracted coding-convention finding backed by representative symbols."""

    category: str  # "logging" | "config" | "data_model" | "error_handling" | "testing" | "naming"
    description: str  # extracted convention, e.g. "module-level logging.getLogger(__name__)"
    evidence: list[SymbolRef] = field(default_factory=list)  # representative symbols (NOT raw file text)


@dataclass(frozen=True)
class RepoManifest:
    """The compiled analysis of a repository — slice 1's typed output."""

    workspace_root: str  # absolute path analysed
    mode: str  # "brownfield" | "greenfield"
    tech_stack: list[str]  # e.g. ["python", "langgraph", "streamlit"]
    core_domains: list[str]  # high-level business/domain areas
    module_boundaries: list[ModuleBoundary]
    dependency_chains: list[DependencyEdge]
    patterns: list[PatternFinding]
    symbol_count: int
    file_count: int
    generated_at: str  # ISO timestamp
    notes: str = ""  # analysis caveats (e.g. "index unavailable, AST-only")

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-round-trippable dict.

        Uses :func:`dataclasses.asdict`, which recurses into the nested
        frozen dataclasses (boundaries/edges/patterns and their ``SymbolRef``
        evidence/key-symbol lists).
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RepoManifest":
        """Reconstruct a :class:`RepoManifest` from :meth:`to_dict` output.

        Rebuilds the nested frozen dataclasses by hand because
        :func:`dataclasses.asdict` flattens them to plain dicts.
        """
        module_boundaries = [
            ModuleBoundary(
                name=mb["name"],
                path=mb["path"],
                role=mb["role"],
                key_symbols=[_symbol_ref_from_dict(s) for s in mb.get("key_symbols", [])],
            )
            for mb in data.get("module_boundaries", [])
        ]
        dependency_chains = [
            DependencyEdge(src=e["src"], dst=e["dst"], kind=e["kind"])
            for e in data.get("dependency_chains", [])
        ]
        patterns = [
            PatternFinding(
                category=p["category"],
                description=p["description"],
                evidence=[_symbol_ref_from_dict(s) for s in p.get("evidence", [])],
            )
            for p in data.get("patterns", [])
        ]
        return cls(
            workspace_root=data["workspace_root"],
            mode=data["mode"],
            tech_stack=list(data.get("tech_stack", [])),
            core_domains=list(data.get("core_domains", [])),
            module_boundaries=module_boundaries,
            dependency_chains=dependency_chains,
            patterns=patterns,
            symbol_count=int(data.get("symbol_count", 0)),
            file_count=int(data.get("file_count", 0)),
            generated_at=data.get("generated_at", ""),
            notes=data.get("notes", ""),
        )


def _symbol_ref_from_dict(data: dict[str, Any]) -> SymbolRef:
    """Reconstruct a :class:`SymbolRef` from its serialised dict form."""
    return SymbolRef(
        file_path=data["file_path"],
        symbol_name=data["symbol_name"],
        symbol_type=data["symbol_type"],
        lang=data["lang"],
        summary=data.get("summary", ""),
    )
