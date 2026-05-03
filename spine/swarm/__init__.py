"""SPINE Swarm module - Parallel agent patterns and coordination."""

from .supervisor import Supervisor, create_supervisor, AgentRole
from .agents import SwarmAgent, MessageTypes
from .gates import SwarmGate, CriticGate
from .mail import SwarmMail
from .learning import LearningIntegration, create_learning_integration, sync_patterns_to_hivemind

__all__ = [
    "Supervisor",
    "create_supervisor",
    "SwarmAgent",
    "MessageTypes",
    "AgentRole",
    "SwarmGate",
    "CriticGate",
    "SwarmMail",
    "LearningIntegration",
    "create_learning_integration",
    "sync_patterns_to_hivemind",
]