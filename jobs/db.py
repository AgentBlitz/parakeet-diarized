"""SQLite-backed job persistence layer.

Stores job metadata in SQLite. Results and audio files are stored on disk
to avoid bloating the database with large JSON blobs.
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    original_filename TEXT,
    audio_file_path TEXT NOT NULL,
    language TEXT,
    diarize INTEGER NOT NULL DEFAULT 1,
    analyze INTEGER NOT NULL DEFAULT 0,
    include_diarization_in_text INTEGER,
    response_format TEXT NOT NULL DEFAULT 'json',
    result_file_path TEXT,
    error_message TEXT,
    webhook_url TEXT,
    retention_expires_at REAL
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_jobs_retention ON jobs(retention_expires_at);",
]


class JobDB:
    """Async SQLite wrapper for job queue persistence."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Create the database and tables if they don't exist."""
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute(CREATE_TABLE_SQL)
        for sql in CREATE_INDEX_SQL:
            await self._db.execute(sql)
        await self._db.commit()

        # Recovery: reset any jobs stuck in 'processing' (server crashed mid-job)
        cursor = await self._db.execute(
            "UPDATE jobs SET status='queued', updated_at=? WHERE status='processing'",
            (time.time(),)
        )
        if cursor.rowcount > 0:
            await self._db.commit()
            logger.warning(f"Recovered {cursor.rowcount} interrupted job(s) → queued")

        logger.info(f"Job database initialized at {self._db_path}")

    async def create_job(
        self,
        audio_file_path: str,
        original_filename: Optional[str] = None,
        language: Optional[str] = None,
        diarize: bool = True,
        analyze: bool = False,
        include_diarization_in_text: Optional[bool] = None,
        response_format: str = "json",
        webhook_url: Optional[str] = None,
    ) -> str:
        """Insert a new job and return its ID."""
        job_id = uuid.uuid4().hex
        now = time.time()
        await self._db.execute(
            """INSERT INTO jobs
               (id, status, created_at, updated_at, original_filename, audio_file_path,
                language, diarize, analyze, include_diarization_in_text,
                response_format, webhook_url)
               VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, now, now, original_filename, audio_file_path,
                language,
                1 if diarize else 0,
                1 if analyze else 0,
                1 if include_diarization_in_text is True else (0 if include_diarization_in_text is False else None),
                response_format, webhook_url,
            ),
        )
        await self._db.commit()
        return job_id

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return a single job as a dict, or None if not found."""
        cursor = await self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> tuple[List[Dict[str, Any]], int]:
        """Return (jobs, total_count) ordered by newest first."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM jobs")
        total = (await cursor.fetchone())[0]

        cursor = await self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows], total

    async def get_next_queued(self) -> Optional[Dict[str, Any]]:
        """Return the oldest queued job (FIFO), or None."""
        cursor = await self._db.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # Columns that callers are allowed to set via update_status(**kwargs)
    _UPDATABLE_COLUMNS = frozenset({
        "started_at", "completed_at", "result_file_path",
        "error_message", "retention_expires_at",
    })

    async def update_status(self, job_id: str, status: str, **kwargs):
        """Update job status and any additional fields."""
        for k in kwargs:
            if k not in self._UPDATABLE_COLUMNS:
                raise ValueError(f"update_status: disallowed column '{k}'")
        now = time.time()
        sets = ["status=?", "updated_at=?"]
        vals = [status, now]
        for k, v in kwargs.items():
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(job_id)
        await self._db.execute(
            f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", vals
        )
        await self._db.commit()

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and return True if it existed."""
        cursor = await self._db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def cleanup_expired(self) -> int:
        """Delete expired jobs and their files. Returns count deleted."""
        now = time.time()
        cursor = await self._db.execute(
            "SELECT id, audio_file_path, result_file_path FROM jobs "
            "WHERE retention_expires_at IS NOT NULL AND retention_expires_at < ?",
            (now,),
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            row = dict(row)
            for path_key in ("audio_file_path", "result_file_path"):
                path = row.get(path_key)
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            await self._db.execute("DELETE FROM jobs WHERE id=?", (row["id"],))
            count += 1
        if count:
            await self._db.commit()
            logger.info(f"Cleaned up {count} expired job(s)")
        return count

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
