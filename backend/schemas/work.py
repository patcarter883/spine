"""Pydantic schemas for work submission and job status."""

import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


SANITIZE_PATTERN = re.compile(r"[<>&\"';()]")
ALLOWED_METHODS = {"Quick Work", "Full Spec Work", "Full Spec Project"}
ALLOWED_PROJECT_TYPES = {"Greenfield", "Brownfield"}


def sanitize_input(value: str) -> str:
    """Strip characters commonly used in injection attacks.

    Removes HTML/XML special chars, semicolons, and parentheses.
    """
    return SANITIZE_PATTERN.sub("", value)


class WorkSubmitRequest(BaseModel):
    requirement: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The requirement description for the work item",
    )
    method: str = Field(
        default="Quick Work",
        description="Automation level",
    )
    project_type: str = Field(
        default="Greenfield",
        description="Environment type",
    )
    llm_provider: str = Field(
        default="ollama",
        max_length=500,
        description="LLM provider name",
    )
    parallel_agents: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum parallel agents",
    )

    @field_validator("requirement")
    @classmethod
    def sanitize_requirement(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("requirement must not be empty")
        sanitized = sanitize_input(stripped)
        if len(sanitized) < 1:
            raise ValueError("requirement contains only invalid characters")
        return sanitized

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ALLOWED_METHODS:
            raise ValueError(f"method must be one of {', '.join(sorted(ALLOWED_METHODS))}")
        return v

    @field_validator("project_type")
    @classmethod
    def validate_project_type(cls, v: str) -> str:
        if v not in ALLOWED_PROJECT_TYPES:
            raise ValueError(f"project_type must be one of {', '.join(sorted(ALLOWED_PROJECT_TYPES))}")
        return v


class WorkSubmitResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str = "Work submitted successfully"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    requirement: str
    method: str
    project_type: str
    llm_provider: str
    parallel_agents: int
    thread_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
