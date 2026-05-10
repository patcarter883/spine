"""GET /jobs/:id endpoint to retrieve job status."""

from fastapi import APIRouter, HTTPException
from starlette.status import HTTP_404_NOT_FOUND

from ..models.job import JobStore
from ..schemas.work import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])

job_store = JobStore()


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Retrieve the current status and metadata for a job.

    Args:
        job_id: The UUID of the job to query.

    Returns:
        JobStatusResponse with current status and metadata.

    Raises:
        HTTPException 404 if job not found.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        requirement=job.requirement,
        method=job.method,
        project_type=job.project_type,
        llm_provider=job.llm_provider,
        parallel_agents=job.parallel_agents,
        thread_id=job.thread_id,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )
