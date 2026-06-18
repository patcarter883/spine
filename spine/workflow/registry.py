"""SPINE phase registry — maps phase names to their node functions and agent builders.

Each workflow phase registers itself with a name, a call function (the LangGraph
node), and an agent builder that creates the Deep Agent for that phase.

Phase node functions may be sync or async.  LangGraph handles both
transparently — sync nodes run in a thread pool, async nodes run directly
on the event loop.  All SPINE phase nodes are async to avoid event-loop
binding errors with the checkpointer's ``asyncio.Lock``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Awaitable


@dataclass
class PhaseDefinition:
    """A registered workflow phase.

    Attributes:
        name: The phase identifier (must match a PhaseName enum value).
        call_fn: The LangGraph node function. Signature: ``(state, config) -> dict``
            (sync) or ``async (state, config) -> dict`` (async).
            **DEPRECATED** — use ``subgraph_node_fn`` instead for new phases.
        build_agent_fn: Factory that creates a Deep Agent for this phase.
            Signature: ``(state, config) -> CompiledGraph``.
        subgraph_node_fn: New-style subgraph wrapper node function.
            When set, this takes precedence over ``call_fn``.
        description: Human-readable description of the phase.
    """

    name: str
    call_fn: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]] | None = None
    build_agent_fn: Callable[..., Any] | None = None
    subgraph_node_fn: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]] | None = None
    description: str = ""


class PhaseRegistry:
    """Registry of workflow phase definitions.

    Phases register themselves at import time via ``register_phase()``.
    The composer looks up phases by name when building a workflow graph.
    """

    def __init__(self) -> None:
        self._phases: dict[str, PhaseDefinition] = {}

    def register(
        self,
        name: str,
        call_fn: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]] | None = None,
        build_agent_fn: Callable[..., Any] | None = None,
        subgraph_node_fn: Callable[..., dict[str, Any] | Awaitable[dict[str, Any]]] | None = None,
        description: str = "",
    ) -> None:
        """Register a workflow phase.

        Args:
            name: Phase name (must match a PhaseName enum value).
            call_fn: LangGraph node function for this phase (legacy).
            build_agent_fn: Factory that creates a Deep Agent for this phase.
            subgraph_node_fn: New-style subgraph wrapper node function.
            description: Human-readable description.
        """
        self._phases[name] = PhaseDefinition(
            name=name,
            call_fn=call_fn,
            build_agent_fn=build_agent_fn,
            subgraph_node_fn=subgraph_node_fn,
            description=description,
        )

    def get(self, name: str) -> PhaseDefinition | None:
        """Look up a phase definition by name.

        Args:
            name: The phase name.

        Returns:
            The PhaseDefinition, or None if not registered.
        """
        return self._phases.get(name)

    def all_phases(self) -> dict[str, PhaseDefinition]:
        """Return all registered phases."""
        return dict(self._phases)

    def require(self, name: str) -> PhaseDefinition:
        """Look up a phase, raising if not found.

        Args:
            name: The phase name.

        Returns:
            The PhaseDefinition.

        Raises:
            KeyError: If the phase is not registered.
        """
        phase = self._phases.get(name)
        if phase is None:
            raise KeyError(f"Phase '{name}' is not registered")
        return phase


# ── Module-level singleton ──

_registry: PhaseRegistry | None = None


def get_registry() -> PhaseRegistry:
    """Return the global phase registry singleton.

    On first call, auto-imports all phase modules so they register themselves.
    """
    global _registry
    if _registry is None:
        _registry = PhaseRegistry()
        # Auto-import phase modules so they call register()
        _import_phase_modules()
    return _registry


def _import_phase_modules() -> None:
    """Import all phase modules to trigger their registration side effects."""
    import spine.phases.specify  # noqa: F401
    import spine.phases.plan  # noqa: F401
    import spine.phases.implement  # noqa: F401
    import spine.phases.verify  # noqa: F401
    import spine.phases.critic  # noqa: F401
    import spine.phases.adversarial  # noqa: F401
    import spine.phases.gap_plan  # noqa: F401
