"""FastAPI router for async job queue endpoints."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import get_config
from models import TranscriptionResponse
from jobs.models import JobCreateResponse, JobStatus, JobListResponse, JobResult

logger = logging.getLogger(__name__)
config = get_config()

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

# These are set by api.py at startup via set_dependencies()
_db = None
_worker = None


def set_dependencies(db, worker):
    """Called by api.py to inject the DB and worker instances."""
    global _db, _worker
    _db = db
    _worker = worker


@router.post("", response_model=JobCreateResponse)
async def create_job(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(True),
    analyze: bool = Form(False),
    include_diarization_in_text: Optional[bool] = Form(None),
    response_format: str = Form("json"),
    webhook_url: Optional[str] = Form(None),
):
    """Submit an audio file for async transcription. Returns a job ID immediately."""
    if not _db:
        raise HTTPException(status_code=503, detail="Job queue not enabled.")

    # Save audio to persistent location (not cleaned up until retention expires)
    jobs_dir = os.path.join(config.temp_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    temp_id = os.urandom(8).hex()
    audio_path = os.path.join(jobs_dir, f"{temp_id}_audio{suffix}")

    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        job_id = await _db.create_job(
            audio_file_path=audio_path,
            original_filename=file.filename,
            language=language,
            diarize=diarize,
            analyze=analyze,
            include_diarization_in_text=include_diarization_in_text,
            response_format=response_format,
            webhook_url=webhook_url,
        )
    except Exception:
        # Clean up the audio file if DB insert fails
        try:
            os.unlink(audio_path)
        except OSError:
            pass
        raise

    # Wake the worker so it picks up the job immediately
    if _worker:
        _worker.wake()

    job = await _db.get_job(job_id)
    logger.info(f"Job {job_id} created for {file.filename}")
    return JobCreateResponse(job_id=job_id, status="queued", created_at=job["created_at"])


@router.get("", response_model=JobListResponse)
async def list_jobs(limit: int = 50, offset: int = 0):
    """List recent jobs, newest first."""
    if not _db:
        raise HTTPException(status_code=503, detail="Job queue not enabled.")
    jobs, total = await _db.list_jobs(limit=limit, offset=offset)
    return JobListResponse(
        jobs=[_job_to_status(j) for j in jobs],
        total=total,
    )


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a specific job."""
    if not _db:
        raise HTTPException(status_code=503, detail="Job queue not enabled.")
    job = await _db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_to_status(job)


@router.get("/{job_id}/result", response_model=JobResult)
async def get_job_result(job_id: str):
    """Get the full transcription result for a completed job."""
    if not _db:
        raise HTTPException(status_code=503, detail="Job queue not enabled.")
    job = await _db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job['status']}. Results are only available for completed jobs."
        )
    result_path = job.get("result_file_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file missing.")

    try:
        with open(result_path, "r") as f:
            result_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=500, detail="Result file is corrupt.")

    return JobResult(
        job_id=job_id,
        status="completed",
        result=TranscriptionResponse(**result_data),
    )


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete a job. Cannot delete jobs that are currently processing."""
    if not _db:
        raise HTTPException(status_code=503, detail="Job queue not enabled.")
    job = await _db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete a job that is currently processing.")

    # Clean up files
    for path_key in ("audio_file_path", "result_file_path"):
        path = job.get(path_key)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass

    await _db.delete_job(job_id)
    return {"detail": "Job deleted.", "job_id": job_id}


def _job_to_status(job: dict) -> JobStatus:
    """Convert a DB row dict to a JobStatus model."""
    duration = None
    if job.get("started_at") and job.get("completed_at"):
        duration = round(job["completed_at"] - job["started_at"], 2)
    return JobStatus(
        job_id=job["id"],
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        original_filename=job.get("original_filename"),
        error_message=job.get("error_message"),
        duration_seconds=duration,
    )
