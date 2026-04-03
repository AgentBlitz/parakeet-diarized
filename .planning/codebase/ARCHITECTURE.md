# Architecture

**Analysis Date:** 2026-04-03

## System Overview

Parakeet-Diarized is a FastAPI server that wraps NVIDIA's `parakeet-tdt-0.6b-v2` NeMo RNNT ASR model with pyannote speaker diarization. It exposes an OpenAI Whisper-compatible REST API for audio transcription with optional speaker identification. A Gradio frontend provides a browser UI for single-file and batch transcription workflows.

## Pattern Overview

**Overall:** Async request-lifecycle pipeline with GPU semaphore gating

**Key Characteristics:**
- 3-phase request lifecycle: I/O preparation, GPU inference, result assembly
- Singleton pattern for expensive ML models (ASR model, diarizer pipeline)
- asyncio semaphores to serialize GPU access (NeMo is not thread-safe)
- Concurrent diarization + transcription via independent semaphores on separate GPU memory regions
- Optional cross-request chunk batching engine for throughput optimization
- All blocking operations (ffmpeg, model.transcribe, diarize) run in thread executors to avoid blocking the event loop

## Component Diagram

```
                    ┌─────────────────────────────────┐
                    │         Gradio Frontend          │
                    │          `app.py` :8001          │
                    └──────────────┬───────────────────┘
                                   │ HTTP POST (requests lib)
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Server :8000                       │
│                     `api.py` → `main.py`                     │
│                                                              │
│  ┌────────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │   audio.py  │   │transcription │   │  diarization/     │  │
│  │  (ffmpeg)   │   │    .py       │   │  __init__.py      │  │
│  │             │   │  (NeMo ASR)  │   │  (pyannote)       │  │
│  └────────────┘   └──────────────┘   └───────────────────┘  │
│                                                              │
│  ┌────────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │ models.py   │   │ config.py    │   │  batching.py      │  │
│  │ (Pydantic)  │   │ (Singleton)  │   │  (opt-in queue)   │  │
│  └────────────┘   └──────────────┘   └───────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                         │                    │
                         ▼                    ▼
                   ┌──────────┐        ┌──────────────┐
                   │ CUDA GPU │        │   ffmpeg      │
                   │ (RTX4090)│        │  (subprocess) │
                   └──────────┘        └──────────────┘
```

## Data Flow

### Request Lifecycle (POST /v1/audio/transcriptions)

**Phase 1 - File I/O (no GPU semaphore, fully concurrent):**

1. Client uploads audio file (any format) via multipart form
2. `api.py` saves upload to temp file in `config.temp_dir` (`/tmp/parakeet`)
3. `audio.convert_audio_to_wav()` runs ffmpeg in a thread executor -> 16kHz mono WAV
4. `audio.split_audio_into_chunks_async()` splits WAV into N-second chunks (default 30s) using parallel async ffmpeg subprocesses (capped at 8 concurrent via semaphore)
5. If audio <= chunk_duration, no splitting occurs (original WAV path returned)

**Phase 2 - GPU Inference (semaphore-gated, concurrent diarization + transcription):**

6. Diarization starts as background `asyncio.Task` (if requested):
   - Acquires `diarize_semaphore` (default capacity 1)
   - Runs `diarizer.diarize(wav_file)` in thread executor
   - pyannote pipeline: segmentation -> embedding extraction -> discrete diarization
   - Returns `DiarizationResult` with `SpeakerSegment` list

7. Transcription runs (two paths):
   - **Direct path** (default): Acquires `transcribe_semaphore`, calls `transcribe_audio_batch()` in executor
   - **Batch queue path** (`ENABLE_BATCH_QUEUE=true`): Submits chunks to `BatchingEngine`, which merges chunks from multiple concurrent requests into one `model.transcribe()` call

8. `transcribe_audio_batch()` in `transcription.py`:
   - Filters out tiny chunks (< 1000 bytes)
   - Calls `model.transcribe(paths, batch_size=N, return_hypotheses=True)` under `torch.no_grad()`
   - Parses NeMo `Hypothesis` objects via `_parse_hypothesis()` -> `(text, List[WhisperSegment])`
   - Timestamp conversion: `offset * model._secs_per_offset` (window_stride * subsampling_factor = 0.08s)

9. Awaits diarization task completion

**Phase 3 - Assembly (no GPU, no semaphore):**

10. Applies chunk time offsets to segment timestamps (chunk_index * chunk_duration)
11. If diarization available: `diarizer.merge_with_transcription()` assigns speakers to segments by maximum temporal overlap
12. Optionally prepends speaker labels to segment text (controlled by `include_diarization_in_text`)
13. Constructs `TranscriptionResponse` and returns in requested format

**Cleanup (finally block):**
- Cancels/awaits diarize_task if still running
- Deletes temp files (upload, WAV, chunks)

### State Management

- **ASR Model**: Global singleton loaded at startup in `api.startup_event()`, stored as `asr_model`
- **Diarizer**: Global singleton `diarizer_instance`, initialized once with HuggingFace token
- **BatchingEngine**: Optional global, started/stopped with app lifecycle
- **Semaphores**: `transcribe_semaphore` and `diarize_semaphore` created at startup with configurable capacity
- **Config**: Singleton via `Config.__new__()` in `config.py`, reads `.env` + environment variables once

## API Surface

### POST /v1/audio/transcriptions
OpenAI Whisper-compatible transcription endpoint.

**Input (multipart/form-data):**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | UploadFile | required | Audio file (any ffmpeg-supported format) |
| `model` | str | `"whisper-1"` | Model name (ignored, kept for API compat) |
| `language` | str | None | Language hint (not used by NeMo TDT) |
| `response_format` | str | `"json"` | One of: `json`, `verbose_json`, `text`, `srt`, `vtt` |
| `temperature` | float | 0.0 | Unused (kept for API compat) |
| `timestamps` | bool | False | Include segment timestamps in response |
| `diarize` | bool | True | Enable speaker diarization |
| `include_diarization_in_text` | bool | None | Prepend speaker labels to text (None = use config default) |
| `prompt` | str | None | Unused (kept for API compat) |
| `vad_filter` | bool | False | Unused |
| `word_timestamps` | bool | False | Unused |
| `timestamp_granularities` | List[str] | None | Unused |

**Output (json/verbose_json):**
```json
{
  "text": "Full transcription text, optionally with speaker labels",
  "task": "transcribe",
  "language": "en",
  "duration": 120.5,
  "model": "parakeet-tdt-0.6b-v2",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.56,
      "text": "Hello everyone",
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": 0.0,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.1,
      "speaker": "speaker_SPEAKER_00"
    }
  ]
}
```

- `json` format: Returns `text` field only (segments omitted unless `timestamps=true`)
- `verbose_json` format: Always includes `segments` array
- `text` format: Plain text response (just the transcript string)
- `srt` format: SubRip subtitle format with optional `[speaker]` prefix
- `vtt` format: WebVTT subtitle format with optional `<v speaker>` tag

**Error responses:**
- 400: Unsupported response format
- 503: Model not loaded yet
- 504: Request timeout (configurable via `REQUEST_TIMEOUT`)
- 500: Internal error

### GET /health
Returns server status, model state, GPU stats, and full config.

**Output:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "diarizer_loaded": true,
  "model_id": "nvidia/parakeet-tdt-0.6b-v2",
  "cuda_available": true,
  "gpu_info": "NVIDIA GeForce RTX 4090",
  "torch_compile_enabled": false,
  "gpu_memory_allocated_mb": 1234.5,
  "gpu_memory_reserved_mb": 2048.0,
  "gpu_max_memory_mb": 1500.3,
  "config": { ... }
}
```

### GET /v1/models
OpenAI-compatible model listing. Returns a single hardcoded `whisper-1` model entry.

## Key Abstractions

**WhisperSegment** (`models.py`):
- Purpose: Represents one timed transcription segment, compatible with OpenAI Whisper segment schema
- Fields: `id`, `start`, `end`, `text`, `speaker` (optional), plus Whisper-compat fields (`seek`, `tokens`, etc.)
- Created by `_parse_hypothesis()` in `transcription.py` from NeMo Hypothesis timestamp data

**TranscriptionResponse** (`models.py`):
- Purpose: Top-level API response matching OpenAI Whisper response schema
- Custom `dict()` method strips `segments` when None

**DiarizationResult** (`diarization/__init__.py`):
- Purpose: Encapsulates pyannote diarization output as typed data
- Contains: `List[SpeakerSegment]` (start, end, speaker label) + `num_speakers`

**Diarizer** (`diarization/__init__.py`):
- Purpose: Singleton wrapper around pyannote `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")`
- Key methods: `diarize(audio_path)` -> `DiarizationResult`, `merge_with_transcription(diarization, segments)` -> annotated segments
- Handles multiple pyannote versions (3.1 Annotation, 3.3+ DiarizeOutput)

**BatchingEngine** (`batching.py`):
- Purpose: Cross-request chunk merging for GPU throughput optimization
- Pattern: Producer-consumer with asyncio Queue + background flush loop
- Callers submit chunks and receive Futures; background worker flushes when batch_size reached or max_wait elapsed

**Config** (`config.py`):
- Purpose: Singleton configuration, reads all env vars at startup
- Pattern: `__new__` singleton with `_initialize()` method
- All env vars are `.strip()`ed to handle Windows CRLF in `.env` files

## Entry Points

**API Server:**
- Location: `main.py`
- Triggers: `python main.py` or `uvicorn main:app`
- Responsibilities: Configures logging, suppresses noisy loggers, creates FastAPI app via `api.create_app()`

**Gradio Frontend:**
- Location: `app.py`
- Triggers: `python app.py`
- Responsibilities: Browser UI on port 8001, calls API via HTTP `requests` library

**Benchmark:**
- Location: `benchmark.py`
- Triggers: `python benchmark.py --file <audio>`
- Responsibilities: Performance testing with sequential/concurrent modes, uses raw `urllib` (no dependencies)

## Error Handling

**Strategy:** Exception propagation with HTTP error mapping

**Patterns:**
- GPU/model errors caught in `api.py` try/except, mapped to HTTP 500
- Model not loaded -> HTTP 503 (checked before any work begins)
- Timeout via `asyncio.wait_for()` -> HTTP 504
- `finally` block always cleans up temp files and cancels pending diarize tasks
- `transcribe_audio_batch()` has fallback: if `return_hypotheses=True` fails, retries without it
- Diarizer returns empty `DiarizationResult` on any failure (graceful degradation)
- Empty/tiny audio chunks (< 1000 bytes) silently skipped with `("", [])` result

## Cross-Cutting Concerns

**Logging:** Python `logging` module, INFO level by default, DEBUG via `DEBUG=1` env var. Per-request timing logs with phase breakdown and RTF (real-time factor). Noisy third-party loggers (megatron, NeMo, lhotse) suppressed to ERROR.

**Validation:** Pydantic models for request/response schemas. Form parameter validation via FastAPI's `Form()` with defaults.

**Authentication:** None on API itself. HuggingFace access token required only for pyannote model download (read from env var at startup).

**CORS:** Wide open (`allow_origins=["*"]`) for local development use.

**Temp File Management:** All temp files created in configurable `TEMP_DIR` (default `/tmp/parakeet`). Cleanup in `finally` blocks per-request and on shutdown.

---

*Architecture analysis: 2026-04-03*
