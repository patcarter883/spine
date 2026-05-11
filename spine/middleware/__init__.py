"""SPINE middleware for Deep Agents integration.

Middleware classes that run inside create_deep_agent() instances,
providing SPINE-specific behaviour within the DA agent loop.
"""

from .critic_gate import CriticGateMiddleware
from .step_limit import StepLimitMiddleware
from .message_queue import MessageQueueMiddleware

__all__ = [
    "CriticGateMiddleware",
    "StepLimitMiddleware",
    "MessageQueueMiddleware",
]