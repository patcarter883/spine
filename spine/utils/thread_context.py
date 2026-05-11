"""Thread context propagation using context variables.

Provides a ContextVar-based mechanism to propagate a consistent
thread ID through all work-entry operations without requiring
explicit parameter passing through every function call.

Usage:
    from spine.utils.thread_context import (
        get_current_thread_id,
        set_current_thread_id,
        ensure_thread_id,
        reset_thread_id,
    )

    # At a work entry point:
    set_current_thread_id(str(uuid.uuid4()))

    # Anywhere downstream:
    tid = get_current_thread_id()
"""

import uuid
from contextvars import ContextVar, Token
from typing import Optional

_thread_id_var: ContextVar[Optional[str]] = ContextVar("thread_id", default=None)


def generate_thread_id() -> str:
    """Generate a unique thread ID using UUID4."""
    return str(uuid.uuid4())


def get_current_thread_id() -> str:
    """Get the current thread ID, generating one if not set.

    Returns:
        The current thread ID string (UUID4). A new ID is generated
        and stored the first time this is called in a given context.
    """
    tid = _thread_id_var.get()
    if tid is None:
        tid = generate_thread_id()
        _thread_id_var.set(tid)
    return tid


def set_current_thread_id(thread_id: str) -> Token:
    """Set the current thread ID and return a reset token.

    Args:
        thread_id: The thread ID to set.

    Returns:
        A Token that can be used with reset_thread_id() to restore
        the previous value.
    """
    return _thread_id_var.set(thread_id)


def reset_thread_id(token: Token) -> None:
    """Reset the thread ID to its previous value using a token.

    Args:
        token: A token returned by set_current_thread_id().
    """
    _thread_id_var.reset(token)


def ensure_thread_id(thread_id: Optional[str] = None) -> str:
    """Ensure a thread ID is set, using the provided one or the current context.

    If *thread_id* is provided, it is set as the current context value.
    Otherwise the existing context value is returned (or a new ID is
    generated and stored if none is set).

    Args:
        thread_id: Optional explicit thread ID to adopt.

    Returns:
        The active thread ID.
    """
    if thread_id is not None:
        set_current_thread_id(thread_id)
        return thread_id
    return get_current_thread_id()


__all__ = [
    "generate_thread_id",
    "get_current_thread_id",
    "set_current_thread_id",
    "reset_thread_id",
    "ensure_thread_id",
]
