# Bug Review Task

You are reviewing a set of changes to a FastAPI-based audio transcription server. Your job is to find bugs, race conditions, logic errors, and security issues. Read every file listed below carefully and report what you find.

Do NOT suggest style improvements, refactors, or nice-to-haves. Only report things that are actually broken or will break at runtime.

---

## What this server does

A FastAPI app that transcribes audio using an NVIDIA NeMo ASR model on GPU, with optional speaker diarization (pyannote) and LLM meeting analysis (Ollama). It runs in WSL with a single RTX 4090 GPU. The NeMo model is NOT thread-safe — a single `asyncio.Semaphore` serializes all GPU transcription calls.

## What changed

Two features were added in a single batch of changes:

### 1. `/transcribe` endpoint hardening

The real-time transcription endpoint (used by a Chrome extension sending raw Float32 audio chunks) was hardened for 10-20 concurrent users:

- **Backpressure**: Rejects with 503 when queue depth exceeds `MAX_TRANSCRIBE_QUEUE` (default 20)
- **Request IDs**: Each request gets an 8-char hex ID for log correlation
- **Queue depth counters**: `_rt_queue_depth`, `_rt_total_served`, `_rt_total_rejected` — plain ints accessed via `nonlocal` from a nested function inside `create_app()`
- **Dedicated timeout**: Uses `TRANSCRIBE_TIMEOUT` (30s) instead of the general 300s `REQUEST_TIMEOUT`
- **Health endpoint**: Now reports `transcribe_queue: {depth, max, total_served, total_rejected}`

### 2. Async job queue

Users can now submit audio files for background processing and poll for results:

- **Pipeline extraction**: The 230-line 4-phase transcription pipeline was moved from `api.py` into `pipeline.py` as `run_transcription_pipeline()`. Both the sync `/v1/audio/transcriptions` endpoint and the async job worker call this shared function.
- **Job queue**: SQLite-backed (via `aiosqlite`), with a background `asyncio.Task` worker that processes jobs sequentially through the same GPU semaphores as the sync endpoints.
- **API**: `POST /v1/jobs` (submit), `GET /v1/jobs` (list), `GET /v1/jobs/{id}` (status), `GET /v1/jobs/{id}/result` (result), `DELETE /v1/jobs/{id}` (delete)
- **Lifecycle**: Enabled via `ENABLE_JOB_QUEUE=true`. Worker starts at app startup, stops at shutdown. Crash recovery resets `processing` → `queued` on init. Expired jobs are cleaned up periodically.

---

## Files to review

Read ALL of the following files. They are the complete set of changed/new files.

### `config.py` (modified)

Key additions:
- `max_transcribe_queue` (int, default 20) — backpressure limit for `/transcribe`
- `transcribe_timeout` (float, default 30, `None` if ≤0) — per-request timeout for `/transcribe`
- `temp_dir` was moved ABOVE `job_db_path` to fix an ordering dependency
- `enable_job_queue`, `job_db_path`, `job_retention_hours`, `job_cleanup_interval`

Things to verify:
- Is `temp_dir` set before `job_db_path` references it? (Yes — was fixed)
- Does `transcribe_timeout = None` work with `asyncio.wait_for()`? (Should — None means no timeout)

### `api.py` (modified)

Key changes:
- Removed `batching_engine` import and variable (was `BatchingEngine`)
- Added `pipeline.py` import for `run_transcription_pipeline`
- `transcribe_audio` endpoint: replaced 230 lines of inline pipeline with a call to `run_transcription_pipeline()`, then formats the response
- `/transcribe` endpoint: added `nonlocal` counters, backpressure, request IDs, enhanced logging
- `/health`: added `transcribe_queue` stats
- Startup: conditionally initializes `JobDB`, `JobWorker`, calls `set_dependencies()`, starts worker
- Shutdown: stops worker → closes DB → closes LLM → cleans temp dir → releases GPU
- Router: `app.include_router(jobs_router)` when `enable_job_queue` is true

Things to verify:
- The `_rt_queue_depth` counter is incremented AFTER the backpressure check, and decremented in `finally`. If the 503 fires, the counter was never incremented, so `finally` runs but the counter goes negative? No — the 503 `raise` happens BEFORE `_rt_queue_depth += 1`, and the `try/finally` block starts AFTER the increment. Verify this is actually correct by reading the code flow carefully.
- The `nonlocal` keyword is used for `_rt_queue_depth` etc. These variables are defined in the `create_app()` function scope (not module scope). The endpoint function is nested inside `create_app()`. Is `nonlocal` the correct mechanism here?
- The shutdown temp cleanup (`for f in temp_dir.iterdir(): f.unlink()`) — does this delete the `jobs/` subdirectory or files inside it? (`iterdir()` lists direct children; `unlink()` fails on directories and is caught by `except`)
- `batching_engine` was removed from `api.py` but `pipeline.py` doesn't support the batching path. If someone sets `ENABLE_BATCH_QUEUE=true`, does anything break? (The batching engine was also removed from imports — check if this causes an ImportError at startup)

### `pipeline.py` (new)

The extracted 4-phase pipeline. Called by both `api.py` (sync endpoint) and `jobs/worker.py` (async worker).

Things to verify:
- Does the pipeline clean up WAV + chunk files in its `finally` block? Does the caller (`api.py` endpoint, `jobs/worker.py`) clean up the original audio file?
- The pipeline raises `RuntimeError` on transcription timeout (not `HTTPException`). In `api.py`, this is caught by `except Exception as e: raise HTTPException(500, str(e))`. In `worker.py`, it's caught by `except Exception as e: update_status("failed", error_message=str(e))`. Is this correct?
- If `convert_audio_to_wav` fails (Phase 1), `wav_file` is still `None`. The `finally` block checks `if wav_file and os.path.exists(wav_file)` — is this safe?
- If `split_audio_into_chunks_async` fails, `audio_chunks` is still `[]`. The `finally` loop over `audio_chunks` is safe (empty list = no-op). Correct?
- The `diarize_task` is created with `asyncio.create_task()`. If the pipeline raises during transcription (Phase 2), the `finally` block cancels the diarize_task. But the diarize_task holds the `diarize_semaphore`. Does cancelling it release the semaphore? (`async with` should handle this via `__aexit__`)

### `jobs/db.py` (new)

SQLite persistence layer.

Things to verify:
- `update_status` builds SQL dynamically from `**kwargs`. Column names are validated against `_UPDATABLE_COLUMNS` frozenset. Are all columns used by `worker.py` in the whitelist? (Check: `started_at`, `completed_at`, `result_file_path`, `error_message`, `retention_expires_at`)
- `list_jobs` uses `tuple[List[...], int]` type hint — this requires Python 3.9+. The server runs Python 3.10 in WSL, so this is fine, but verify.
- `cleanup_expired` deletes files then DB rows. If the server crashes between file deletion and DB deletion, the DB row remains but files are gone. On next cleanup, `os.path.exists()` returns False, row is deleted anyway. Is this safe? (Yes)
- The crash recovery (`UPDATE ... WHERE status='processing'`) runs during `initialize()`. If two server instances somehow start simultaneously, both could reset the same job. Is this a concern? (No — single server by design)

### `jobs/worker.py` (new)

Background asyncio.Task that processes jobs.

Things to verify:
- The worker skips jobs if `self._model is None` (line 91-93) but does NOT update the job status. The job stays `queued` and will be picked up again on the next poll (5s later). This creates an infinite retry loop if the model never loads. Is this acceptable?
- After processing a job, the loop goes back to `get_next_queued()` immediately (no sleep). This is correct for throughput — but verify there's no tight loop if `get_next_queued()` keeps returning the same failed-to-process job.
- The `_cleanup_loop` sleeps FIRST (`await asyncio.sleep(interval)`), so first cleanup happens after `job_cleanup_interval` seconds (default 1 hour). Jobs submitted and completed in the first hour won't be cleaned up until hour 2. Is this intentional?
- `_fire_webhook` creates a new `httpx.AsyncClient` per call. The `async with` context manager properly closes it after use. No connection leak.
- When `stop()` is called, it cancels both tasks. If the worker is mid-`_process_job` (which holds the `transcribe_semaphore`), does the cancellation properly release the semaphore? The pipeline's `finally` block should run on cancellation, cleaning up files and releasing the diarize task. But `async with transcribe_semaphore` — if the `await asyncio.wait_for(...)` inside it is cancelled, does the `async with` `__aexit__` still release the semaphore? (It should — `CancelledError` is a `BaseException` and `async with` handles it)

### `jobs/router.py` (new)

FastAPI router for `/v1/jobs` endpoints.

Things to verify:
- `set_dependencies()` sets module-level globals `_db` and `_worker`. This is called once during startup. All endpoint functions read these globals. Since FastAPI is single-process async, there's no thread-safety issue. Correct?
- `create_job`: saves audio file, then tries DB insert. If insert fails, audio file is cleaned up in `except`. If the `await file.read()` or `open()` fails, no DB row exists, but the audio file might be partially written. The `except` block only wraps the DB call, not the file write. Is this a problem? (Partially written files would be orphaned)
- `delete_job`: tries to `unlink` files that may have already been deleted by the retention cleanup. `os.path.exists()` check + `except OSError: pass` handles this. Correct?
- Route ordering: `GET /v1/jobs` (list) and `GET /v1/jobs/{job_id}` (status) — FastAPI matches routes in definition order. Since the empty-string list route is defined first, there's no conflict. Correct?

### `jobs/models.py` (new)

Pydantic models. Straightforward — just check that `JobResult.result` uses `TranscriptionResponse` correctly and that all field types match what the DB/worker actually produces.

### `jobs/__init__.py` (new)

Re-exports `JobDB`, `JobWorker`, `jobs_router`. Check for circular import risks: `__init__.py` imports from `db.py`, `worker.py`, `router.py`. `worker.py` imports from `pipeline.py`. `router.py` imports from `models.py` (the project root one) and `jobs/models.py`. `api.py` imports from `jobs/__init__.py`. Is there a circular chain?

---

## Output format

For each bug found, report:
1. **File and line**: e.g., `worker.py:91`
2. **Severity**: Critical (will crash/corrupt), Medium (incorrect behavior under specific conditions), Low (edge case, cosmetic)
3. **Description**: What's wrong and why
4. **Fix**: Specific code change needed

If you find no bugs, say so explicitly. Do not invent issues that don't exist.
