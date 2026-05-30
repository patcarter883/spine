"""Spine onboarding engine.

Analyses a target repository (brownfield) or seeds a greenfield project and
synthesises the four Spine onboarding artifacts. This package is the
producer/consumer boundary for the onboarding workflow:

- :mod:`spine.work.onboarding.manifest` — the typed ``RepoManifest`` contract
  (frozen dataclass tree, JSON round-trippable).
- :mod:`spine.work.onboarding.analyzer` — the analysis backend that compiles a
  ``RepoManifest`` from AST symbol extraction + the codebase index (slice 1).

Synthesis (slice 3, :mod:`spine.work.onboarding.synthesis`) and the dispatch
entrypoint (:func:`spine.work.onboarding.engine.run_onboarding`) complete the
package.
"""

from __future__ import annotations

from spine.work.onboarding.manifest import (
    DependencyEdge,
    ModuleBoundary,
    PatternFinding,
    RepoManifest,
    SymbolRef,
)

__all__ = [
    "DependencyEdge",
    "ModuleBoundary",
    "PatternFinding",
    "RepoManifest",
    "SymbolRef",
    "run_onboarding",
]


def __getattr__(name: str):  # noqa: ANN202 — module-level lazy attribute
    """Lazily expose ``run_onboarding`` without importing the engine eagerly.

    The engine pulls in the synthesis stack (Deep Agent factory, etc.) and the
    dispatcher; importing it at package import time would create an import cycle
    (dispatcher imports this package's engine). Resolving it on first access
    keeps ``import spine.work.onboarding`` cheap and cycle-free.
    """
    if name == "run_onboarding":
        from spine.work.onboarding.engine import run_onboarding

        return run_onboarding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
