"""Background worker that processes the async job queue.

Runs as a single asyncio.Task, polling SQLite for queued jobs and
processing them through the shared transcription pipeline.
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

import httpx

from config import get_config
from jobs.db import JobDB
from pipeline import run_transcription_pipeline

logger = logging.getLogger(__name__)
config = get_config()


class JobWorker:
    """Processes queued transcription jobs in the background."""

    def __init__(
        self,
        db: JobDB,
        transcribe_semaphore: asyncio.Semaphore,
        diarize_semaphore: asyncio.Semaphore,
        model,
        diarizer,
        llm_client,
    ):
        self._db = db
        self._transcribe_semaphore = transcribe_semaphore
        self._diarize_semaphore = diarize_semaphore
        self._model = model
        self._diarizer = diarizer
        self._llm_client = llm_client
        self._task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._wake_event = asyncio.Event()
        self._running = False

    async def start(self):
        """Start the worker and cleanup loops."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Job worker started")

    async def stop(self):
        """Stop the worker gracefully."""
        self._running = False
        self._wake_event.set()
        for task in (self._task, self._cleanup_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._cleanup_task = None
        logger.info("Job worker stopped")

    def wake(self):
        """Signal the worker that a new job was enqueued."""
        self._wake_event.set()

    async def _run_loop(self):
        """Main loop: pick up queued jobs and process them."""
        while self._running:
            job = await self._db.get_next_queued()
            if job is None:
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                self._wake_event.clear()
                continue

            await self._process_job(job)

    async def _process_job(self, job: dict):
        """Run the full transcription pipeline for a single job."""
        job_id = job["id"]

        if self._model is None:
            logger.warning(f"[job {job_id}] Model not loaded yet — skipping, will retry")
            return

        logger.info(f"[job {job_id}] Processing: {job.get('original_filename', 'unknown')}")

        await self._db.update_status(job_id, "processing", started_at=time.time())

        try:
            response = await run_transcription_pipeline(
                audio_file_path=job["audio_file_path"],
                model=self._model,
                diarizer=self._diarizer,
                llm_client=self._llm_client,
                transcribe_semaphore=self._transcribe_semaphore,
                diarize_semaphore=self._diarize_semaphore,
                language=job.get("language"),
                diarize=bool(job.get("diarize", 1)),
                analyze=bool(job.get("analyze", 0)),
                include_diarization_in_text=(
                    True if job.get("include_diarization_in_text") == 1
                    else (False if job.get("include_diarization_in_text") == 0 else None)
                ),
                response_format=job.get("response_format", "json"),
                timestamps=True,
            )

            # Save result to disk
            jobs_dir = os.path.join(config.temp_dir, "jobs")
            os.makedirs(jobs_dir, exist_ok=True)
            result_path = os.path.join(jobs_dir, f"{job_id}_result.json")
            with open(result_path, "w") as f:
                json.dump(response.dict(), f)

            completed_at = time.time()
            retention_expires = completed_at + config.job_retention_hours * 3600

            await self._db.update_status(
                job_id, "completed",
                completed_at=completed_at,
                result_file_path=result_path,
                retention_expires_at=retention_expires,
            )
            logger.info(f"[job {job_id}] Completed successfully")

            # Fire webhook if configured
            if job.get("webhook_url"):
                await self._fire_webhook(job, response)

        except Exception as e:
            logger.error(f"[job {job_id}] Failed: {e}")
            await self._db.update_status(
                job_id, "failed",
                completed_at=time.time(),
                error_message=str(e),
                retention_expires_at=time.time() + config.job_retention_hours * 3600,
            )

    async def _fire_webhook(self, job: dict, response):
        """POST job result to the webhook URL. Fire-and-forget, no retries."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    job["webhook_url"],
                    json={
                        "job_id": job["id"],
                        "status": "completed",
                        "result": response.dict(),
                    },
                )
            logger.info(f"[job {job['id']}] Webhook delivered to {job['webhook_url']}")
        except Exception as e:
            logger.warning(f"[job {job['id']}] Webhook failed: {e}")

    async def _cleanup_loop(self):
        """Periodically clean up expired jobs and their files."""
        while self._running:
            try:
                await asyncio.sleep(config.job_cleanup_interval)
                await self._db.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Job cleanup error: {e}")
