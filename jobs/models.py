from typing import List, Optional
from pydantic import BaseModel

from models import TranscriptionResponse


class JobCreateResponse(BaseModel):
    job_id: str
    status: str = "queued"
    created_at: float


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    created_at: float
    updated_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    original_filename: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None


class JobListResponse(BaseModel):
    jobs: List[JobStatus]
    total: int


class JobResult(BaseModel):
    job_id: str
    status: str
    result: TranscriptionResponse
