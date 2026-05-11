"""Pydantic schemas for the /audit endpoint."""

from typing import Optional

from pydantic import BaseModel, Field


class AuditEntryResponse(BaseModel):
    entry_id: str
    thread_id: str
    action: str
    details: dict = {}
    timestamp: str
    created_at: str


class AuditQueryResponse(BaseModel):
    entries: list[AuditEntryResponse]
    total: int
    limit: int
    offset: int
