"""Backend routes package."""

from .work import router as work_router
from .status import router as status_router
from .audit import router as audit_router

__all__ = ["work_router", "status_router", "audit_router"]
