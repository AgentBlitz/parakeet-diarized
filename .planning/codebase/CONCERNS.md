# Technical Concerns

**Analysis Date:** 2026-04-03

## Critical Issues

**No API authentication:**
- The API has zero authentication. Any client on the network can submit audio files for transcription.
- Files: `api.py` (lines 46-52) — CORS is `allow_origins=["*"]` with all methods/headers allowed.
- No API key, bearer token, or any auth middleware exists anywhere in the codebase.
- Impact: In any deployment beyond localhost, this is a wide-open GPU compute endpoint. An attacker could exhaust GPU resources or exfiltrate transcription results.
- Fix approach: Add optional `API_KEY` env var and a FastAPI dependency that checks `Authorization: Bearer <key>` header. Gate behind a config flag so local dev remains frictionless.

**No upload file size limit:**
- The endpoint reads the entire uploaded file into memory with `await file.read()` before writing to disk.
- Files: `api.py` (line 171)
- Impact: A single malicious request with a multi-GB file will exhaust server RAM. There is no `max_upload_size` configured on FastAPI or uvicorn.
- Fix approach: Add a streaming write with a size cap (e.g., 500MB). Use `file.read(chunk_size)` in a loop with a running total, raising 413 when exceeded.

**No input file type validation:**
- The endpoint accepts any file and passes it to ffmpeg. There is no validation of MIME type, file extension, or magic bytes.
- Files: `api.py` (line 168) — `Path(file.filename).suffix` is used only for temp file naming, never for validation.
- Impact: Arbitrary files (executables, zip bombs, etc.) get written to `/tmp/parakeet` and piped to ffmpeg. While ffmpeg will likely fail gracefully, this is an unnecessary attack surface.
- Fix approach: Validate file extension against an allowlist (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.mp4`, `.aac`) before saving. Optionally check Content-Type header.

**Startup error is silently swallowed:**
- If model loading fails during startup, the exception is caught and logged but the server continues running.
- Files: `api.py` (lines 97-99) — `except Exception as e: logger.error(...)` with no re-raise.
- Impact: Server appears healthy at the HTTP level but returns 503 on every transcription request. The `/health` endpoint does report `model_loaded: false`, but automated clients may not check this.
- Fix approach: Consider re-raising fatal startup errors, or at minimum returning a non-200 status from `/health` when model loading failed.

## Technical Debt

| Area | Debt | Impact | Effort |
|------|------|--------|--------|
| Deprecated event hooks | `@app.on_event("startup")` and `@app.on_event("shutdown")` in `api.py` (lines 54, 101) are deprecated in modern FastAPI. | Will break on future FastAPI upgrade. | Low — migrate to `lifespan` context manager. |
| Unused `pydantic.types.T` import | `config.py` line 7: `from pydantic.types import T` is imported but never used. | Lint warning, confusing for readers. | Trivial — delete the line. |
| Dead code: `split_audio_into_chunks` (sync) | `audio.py` lines 13-78: The synchronous `split_audio_into_chunks()` is never called by the API (which uses `split_audio_into_chunks_async`). Only referenced by `test_chunking.py` and the re-export in `main.py`. | Dead code with its own test that tests only the dead path. | Low — remove or mark as legacy. |
| Dead code: `transcribe_audio_chunk` | `transcription.py` lines 276-303: Single-chunk wrapper is never called anywhere in the codebase. | Dead code. | Trivial — remove. |
| `main.py` re-exports | `main.py` lines 52-54 re-export `split_audio_into_chunks`, `convert_audio_to_wav`, `load_model`, `format_srt`, `format_vtt`, `WhisperSegment`, `TranscriptionResponse` "for backwards compatibility". | Pollutes module namespace, creates hidden coupling. | Low — remove once no external consumers depend on it. |
| Duration estimate is fake | `api.py` line 336: `duration=sum(len(segment.text.split()) for segment in all_segments) / 150` estimates duration from word count at 150 WPM instead of using actual audio duration. | API returns inaccurate `duration` field — misleading for clients. | Low — compute from `len(audio_chunks) * chunk_duration` or from ffprobe. |
| Global mutable state | `api.py` lines 27-31: `asr_model`, `diarizer_instance`, semaphores are module-level globals mutated via `global` keyword inside nested functions. | Makes testing difficult, prevents running multiple app instances. | Medium — refactor into app state (`app.state.model`). |
| Hardcoded model in `/v1/models` | `api.py` lines 429-442: Returns a hardcoded `whisper-1` model with a fake `created` timestamp (1677649963). | Misleading to clients expecting real model metadata. | Trivial — use actual model ID and current timestamp. |

## Security Concerns

**Unrestricted CORS:**
- `api.py` lines 46-52: `allow_origins=["*"]` allows any website to make requests to the API from a browser.
- Risk: If the server is exposed beyond localhost, any webpage can trigger transcriptions.
- Fix approach: Restrict origins to known frontends (e.g., `http://localhost:8001` for Gradio).

**Temp file predictability:**
- `api.py` line 168: Temp files use `os.urandom(8).hex()` which is fine for uniqueness, but the temp directory `/tmp/parakeet` is world-readable by default.
- Risk: Other processes on the same machine can read uploaded audio files.
- Fix approach: Use `tempfile.mkdtemp()` with restricted permissions per request, or set directory permissions on `/tmp/parakeet`.

**Health endpoint leaks full config:**
- `api.py` line 422: `/health` returns `config.as_dict()` which includes all configuration values.
- Risk: Exposes internal configuration (batch sizes, timeouts, feature flags) to unauthenticated callers.
- Fix approach: Return only essential health info (status, model_loaded, cuda_available). Move detailed config to a separate admin endpoint.

**HuggingFace token in Diarizer fallback:**
- `diarization/__init__.py` lines 41-44: If no token is passed to `Diarizer.__init__`, it reads directly from env vars as a fallback. This creates a second code path for token resolution beyond `config.py`.
- Risk: Token handling inconsistency; harder to audit secret access patterns.
- Fix approach: Always pass token from `config.py`, remove the env var fallback from `Diarizer`.

**No rate limiting:**
- No rate limiting middleware exists. Combined with no authentication, the API is vulnerable to resource exhaustion.
- Fix approach: Add `slowapi` or a simple token bucket middleware.

## Performance Concerns

**Entire file read into memory before writing:**
- `api.py` line 171: `content = await file.read()` loads the full upload into RAM before writing to disk.
- Impact: A 1GB audio file requires 1GB of RAM just for the upload buffer, on top of the file copy on disk.
- Fix approach: Stream the upload to disk in chunks using `file.read(8192)` in a loop.

**Sequential batch processing in Gradio:**
- `app.py` line 213: `transcribe_batch` processes files sequentially via `_call_api()` in a loop.
- Impact: Batch of 10 files takes 10x single-file time. No parallelism even when GPU semaphore would allow it.
- Fix approach: Use `asyncio.gather` or `concurrent.futures` for parallel API calls (respecting server concurrency limits).

**Chunk temp directory leak on async split failure:**
- `audio.py` line 151: `temp_dir = tempfile.mkdtemp()` creates a directory that is never cleaned up if `split_audio_into_chunks_async` raises an exception after partial chunk extraction.
- Impact: Accumulated temp directories on long-running servers consume disk space.
- Fix approach: Add cleanup in the exception handler or use a context manager.

**No GPU memory management between requests:**
- There is no `torch.cuda.empty_cache()` call between requests (only at shutdown).
- Impact: GPU memory fragmentation over time, especially with varying batch sizes.
- Fix approach: Consider periodic cache clearing, or after OOM errors.

## Reliability Concerns

**Diarization failure returns empty result, not error:**
- `diarization/__init__.py` line 204: If diarization throws any exception, it returns `DiarizationResult(segments=[], num_speakers=0)` instead of propagating the error.
- Files: `diarization/__init__.py` lines 202-204
- Impact: Callers silently get undiarized output with no indication that diarization failed. The log shows the error but the API response gives no signal.
- Fix approach: Add a `diarization_error` field to the response, or return a warning header.

**Chunk splitting error falls back to original file:**
- `audio.py` lines 75-78 (sync) and lines 181-183 (async): If splitting fails, the original full-length WAV is returned as a single "chunk".
- Impact: If the audio exceeds NeMo's ~40s max duration limit, the transcription will silently produce garbage or empty results.
- Fix approach: Raise the error instead of silently degrading. Log a clear warning if fallback is used.

**No request ID or correlation:**
- No request ID is generated or propagated through the request lifecycle (except in the batch queue path where `os.urandom(4).hex()` is used).
- Files: `api.py` — no middleware for request tracing.
- Impact: Difficult to correlate logs across phases for debugging production issues.
- Fix approach: Add middleware that generates a UUID per request and includes it in all log messages.

**`asyncio.get_event_loop()` deprecation:**
- `api.py` line 174, `batching.py` line 101: Use `asyncio.get_event_loop()` which is deprecated in Python 3.10+ in favor of `asyncio.get_running_loop()`.
- Impact: Will emit DeprecationWarning and may break in future Python versions.
- Fix approach: Replace with `asyncio.get_running_loop()`.

**Error responses leak internal details:**
- `api.py` line 373: `raise HTTPException(status_code=500, detail=str(e))` exposes raw exception messages to clients.
- Impact: Stack traces, file paths, and internal state could leak to external callers.
- Fix approach: Return generic error messages in production; log full details server-side.

## Missing Features / Gaps

**No local LLM integration (planned but not implemented):**
- `docs/LLM2.md` describes a planned integration with Granite 3.3 8B Instruct for meeting summarization and structured extraction from transcripts.
- The document specifies system prompts, user prompts with `<think>`/`<response>` tags, and generation parameters for vLLM/Ollama/LM Studio.
- No code implements this. There is no LLM client, no summarization endpoint, and no reference to Ollama/vLLM in any Python file.
- Impact: The summarization feature described in documentation does not exist in the codebase. This is a planned enhancement, not a bug.

**No WebSocket or streaming support:**
- The API only supports synchronous request-response. Long audio files block until completion.
- Impact: No progress indication for API clients (Gradio works around this with generator streaming on its side).
- Fix approach: Add SSE or WebSocket endpoint for progress updates.

**No language auto-detection feedback:**
- The `language` parameter is accepted but ignored by NeMo TDT (which is English-only).
- Files: `transcription.py` line 213 comment: "not used by NeMo TDT, kept for API compat"
- Impact: Clients may send non-English audio expecting language support and get back garbage English transcription with no error.
- Fix approach: Validate language parameter; return 400 if non-English language is explicitly requested.

**No automated tests for core paths:**
- `tests/test_api.py` is a manual CLI script (not a pytest/unittest suite) that requires a running server.
- `tests/test_chunking.py` is a proper unittest but only tests the dead synchronous `split_audio_into_chunks` function.
- No tests exist for: `transcribe_audio_batch`, `_parse_hypothesis`, diarization merging, batching engine, response formatting, or any API endpoint.
- Impact: No regression safety net. Changes to core transcription or diarization logic are untestable without manual verification.
- Fix approach: Add pytest fixtures with mock NeMo model, test the async API endpoints with `httpx.AsyncClient`.

## Test Coverage Gaps

**Untested: Transcription pipeline:**
- What's not tested: `transcribe_audio_batch()`, `_parse_hypothesis()`, NeMo result unwrapping
- Files: `transcription.py`
- Risk: NeMo version upgrades could silently change return types (as documented in Known Bugs)
- Priority: High

**Untested: Diarization merge logic:**
- What's not tested: `Diarizer.merge_with_transcription()`, speaker label formatting in `api.py` lines 292-323
- Files: `diarization/__init__.py`, `api.py`
- Risk: Speaker assignment algorithm uses overlap duration heuristic that could produce wrong labels
- Priority: High

**Untested: Batching engine:**
- What's not tested: `BatchingEngine` flush loop, cross-request merging, timeout handling, shutdown draining
- Files: `batching.py`
- Risk: Race conditions in async flush loop, futures never resolving on error
- Priority: Medium

**Untested: Response format outputs:**
- What's not tested: SRT/VTT formatting, speaker label inclusion in subtitle formats
- Files: `transcription.py` (`format_srt`, `format_vtt`)
- Risk: Malformed subtitle output
- Priority: Low

## Dependencies at Risk

**NeMo version sensitivity:**
- The codebase contains multiple workarounds for NeMo API changes (documented in CLAUDE.md "Known Bugs" section).
- `transcription.py` lines 263-268: Handles NeMo 2.x tuple return format as a special case.
- `transcription.py` lines 44-48: Disables CUDA graph decoder due to cuda-python API mismatch.
- Risk: Any NeMo upgrade requires careful testing of the transcription pipeline. The workarounds are fragile.
- Impact: Model loading or transcription could silently break.

**pyannote version sensitivity:**
- `diarization/__init__.py` lines 158-176: Contains four different code paths to handle varying return types across pyannote 3.1, 3.3+, and other builds.
- Risk: New pyannote versions may introduce yet another return type.

**Pydantic v1/v2 ambiguity:**
- `models.py` line 30: Overrides `.dict()` method (Pydantic v1 API). Pydantic v2 uses `.model_dump()`.
- `config.py` line 7: Imports `from pydantic.types import T` which is a v1 artifact.
- Risk: Pydantic v2 migration will break model serialization.

---

*Concerns audit: 2026-04-03*
