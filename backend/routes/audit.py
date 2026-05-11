"""GET /audit endpoint for querying work history entries."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from starlette.status import HTTP_400_BAD_REQUEST

from backend.schemas.audit import AuditEntryResponse, AuditQueryResponse
from spine.models.work_entry import WorkEntryStore

router = APIRouter(prefix="/audit", tags=["audit"])

store = WorkEntryStore()


@router.get("", response_model=AuditQueryResponse)
async def query_audit(
    thread_id: Optional[str] = Query(None, description="Filter by thread ID"),
    action: Optional[str] = Query(None, description="Filter by action name"),
    timestamp_from: Optional[str] = Query(None, alias="timestamp-from", description="Filter by timestamp >= (ISO format)"),
    timestamp_to: Optional[str] = Query(None, alias="timestamp-to", description="Filter by timestamp <= (ISO format)"),
    limit: int = Query(50, ge=1, le=1000, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> AuditQueryResponse:
    entries, total = store.query_entries(
        thread_id=thread_id,
        action=action,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        limit=limit,
        offset=offset,
    )

    return AuditQueryResponse(
        entries=[AuditEntryResponse(
            entry_id=e.entry_id,
            thread_id=e.thread_id,
            action=e.action,
            details=e.details,
            timestamp=e.timestamp,
            created_at=e.created_at,
        ) for e in entries],
        total=total,
        limit=limit,
        offset=offset,
    )
