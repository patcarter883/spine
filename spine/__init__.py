"""SPINE - Deterministic AI agent harness."""

from .core import (
    SpineState, SpineStateMachine, Phase, Task, SubPhase,
    PhaseResult, SubPhaseResult, create_spine_workflow
)
from .models import PhaseName, StateStatus, SubPhaseStatus
from . import providers
from . import swarm
from . import hive

__version__ = "0.1.0"

__all__ = [
    "SpineState",
    "SpineStateMachine",
    "Phase",
    "Task",
    "SubPhase",
    "PhaseResult",
    "SubPhaseResult",
    "create_spine_workflow",
    "PhaseName",
    "StateStatus",
    "SubPhaseStatus",
    "providers",
    "swarm",
    "hive",
]
