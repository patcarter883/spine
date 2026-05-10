"""Backend schemas package."""

from .work import (
    WorkSubmitRequest,
    WorkSubmitResponse,
    JobStatusResponse,
    ErrorResponse,
)

__all__ = [
    "WorkSubmitRequest",
    "WorkSubmitResponse",
    "JobStatusResponse",
    "ErrorResponse",
]
