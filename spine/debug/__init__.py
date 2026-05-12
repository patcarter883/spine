"""SPINE debug utilities.

Provides model I/O logging and other debugging tools for Deep Agents
integration.  Enable with ``SPINE_DEBUG_MODEL_IO=1``.
"""

from .model_io import (
    ModelIOLogger,
    is_debug_enabled,
    set_debug_phase,
    get_debug_phase,
)

__all__ = [
    "ModelIOLogger",
    "is_debug_enabled",
    "set_debug_phase",
    "get_debug_phase",
]
