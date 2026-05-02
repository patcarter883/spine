"""SPINE Swarm module - Parallel agent patterns and coordination."""

from .supervisor import Supervisor, create_supervisor, AgentRole
from .agents import SwarmAgent
from .gates import SwarmGate, CriticGate
from .mail import SwarmMail

__all__ = [
    "Supervisor",
    "create_supervisor",
    "SwarmAgent",
    "AgentRole",
    "SwarmGate",
    "CriticGate",
    "SwarmMail",
]