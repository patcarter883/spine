"""SPINE adapters for Deep Agents integration.

Provides helper functions to construct create_deep_agent() instances
with phase-specific configuration, SubAgent specs, and middleware stacks.
"""

from .da_phase_adapter import (
    create_planning_agent,
    create_execution_agent,
    create_verification_agent,
    get_backend,
)

__all__ = [
    "create_planning_agent",
    "create_execution_agent",
    "create_verification_agent",
    "get_backend",
]