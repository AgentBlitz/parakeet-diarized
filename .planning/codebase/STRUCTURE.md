# Project Structure

**Analysis Date:** 2026-04-03

## Directory Layout

```
parakeet-diarized/
├── main.py                  # Uvicorn entrypoint, logging config, app creation
├── api.py                   # FastAPI app factory, endpoints, request lifecycle
├── transcription.py         # NeMo model loading, batch transcription, SRT/VTT formatting
├── audio.py                 # ffmpeg WAV conversion + sync/async chunk splitting
├── models.py                # Pydantic models (WhisperSegment, TranscriptionResponse, etc.)
├── config.py                # Singleton Config class, reads .env + env vars
├── batching.py              # Cross-request chunk batching engine (opt-in)
├── app.py                   # Gradio frontend (single-file + batch tabs)
├── benchmark.py             # CLI benchmark tool for API performance testing
├── diarization/
│   └── __init__.py          # Diarizer class (pyannote pipeline wrapper)
├── tests/
│   ├── test_api.py          # API endpoint tests
│   └── test_chunking.py     # Audio chunking tests
├── experiments/             # Audio test files for benchmarking
├── docs/                    # Documentation assets
├── requirements.txt         # Python dependencies (no version pins)
├── Dockerfile               # Multi-stage Docker build (CUDA 12.1)
├── docker-compose.yml       # Docker Compose config
├── docker-entrypoint.sh     # Container startup script
├── start.ps1                # PowerShell: start API server via WSL
├── start_ui.ps1             # PowerShell: start Gradio frontend via WSL
├── stop.ps1                 # PowerShell: stop server
├── run.sh                   # Bash startup script
├── .env                     # Environment configuration (not committed)
├── .gitignore               # Git ignore rules
├── CLAUDE.md                # AI assistant instructions
├── README.md                # Project documentation
├── before.md                # Notes (pre-optimization state)
├── after.md                 # Notes (post-optimization state)
└── followup.md              # Notes (follow-up improvements)
```

## Directory Purposes

**Root (`.`):**
- Purpose: All Python source files live at the root level (flat module structure)
- Contains: Core server modules, frontend, config, benchmark
- Key files: `api.py`, `transcription.py`, `audio.py`, `models.py`, `config.py`

**`diarization/`:**
- Purpose: Speaker diarization module wrapping pyannote.audio
- Contains: Single `__init__.py` with `Diarizer` class, `SpeakerSegment`, `DiarizationResult`
- Key files: `diarization/__init__.py`

**`tests/`:**
- Purpose: Test files
- Contains: `test_api.py`, `test_chunking.py`

**`experiments/`:**
- Purpose: Audio test files for manual testing and benchmarking
- Contains: Sample audio files (.m4a, .mp3, etc.)
- Generated: No (manually added)
- Committed: Partial (check .gitignore)

**`docs/`:**
- Purpose: Documentation assets
- Contains: Supplementary docs

## Key File Locations

**Entry Points:**
- `main.py`: Uvicorn server entrypoint, creates FastAPI app, configures logging
- `app.py`: Gradio frontend entrypoint (port 8001)
- `benchmark.py`: CLI benchmark tool

**Configuration:**
- `config.py`: Singleton `Config` class with all settings
- `.env`: Environment variables (not committed, see CLAUDE.md for full list)
- `Dockerfile`: Multi-stage CUDA 12.1 build
- `docker-compose.yml`: Container orchestration

**Core Logic:**
- `api.py`: FastAPI app factory with `create_app()`, all endpoints, 3-phase request lifecycle
- `transcription.py`: `load_model()`, `transcribe_audio_batch()`, `_parse_hypothesis()`, `format_srt()`, `format_vtt()`
- `audio.py`: `convert_audio_to_wav()`, `split_audio_into_chunks_async()`, `split_audio_into_chunks()` (sync legacy)
- `diarization/__init__.py`: `Diarizer` class with `diarize()` and `merge_with_transcription()`
- `batching.py`: `BatchingEngine` class with `submit_chunks()`, background `_flush_loop()`

**Data Models:**
- `models.py`: `WhisperSegment`, `TranscriptionResponse`, `ModelInfo`, `ModelList`

**Testing:**
- `tests/test_api.py`: API endpoint tests
- `tests/test_chunking.py`: Audio chunking tests

## Naming Conventions

**Files:**
- snake_case for all Python modules: `transcription.py`, `audio.py`
- Module-per-concern: each file handles one domain (audio processing, transcription, diarization, config)

**Directories:**
- lowercase: `diarization/`, `tests/`, `experiments/`, `docs/`
- Package directories have `__init__.py`

**Functions:**
- snake_case: `transcribe_audio_batch()`, `convert_audio_to_wav()`, `split_audio_into_chunks_async()`
- Private helpers prefixed with underscore: `_parse_hypothesis()`, `_format_timestamp()`, `_run_diarize()`

**Classes:**
- PascalCase: `Diarizer`, `BatchingEngine`, `Config`, `WhisperSegment`
- Pydantic BaseModel for data classes

## Module Dependencies

```
main.py
├── api.create_app()
│   ├── models.py (WhisperSegment, TranscriptionResponse, ModelInfo, ModelList)
│   ├── audio.py (convert_audio_to_wav, split_audio_into_chunks_async)
│   ├── transcription.py (load_model, format_srt, format_vtt, transcribe_audio_batch)
│   ├── diarization/__init__.py (Diarizer)
│   ├── batching.py (BatchingEngine)
│   └── config.py (get_config)
└── config.py (get_config)

transcription.py
├── models.py (WhisperSegment, TranscriptionResponse)
└── config.py (get_config) — for torch_compile settings

diarization/__init__.py
└── config.py (get_config) — for batch sizes and segmentation step

batching.py
├── models.py (WhisperSegment)
└── transcription.py (transcribe_audio_batch) — lazy import in _flush_batch()

audio.py
└── (no internal imports, uses subprocess/asyncio for ffmpeg)

app.py
└── (standalone, calls API via HTTP requests library)

benchmark.py
└── (standalone, calls API via urllib)
```

**Dependency flow:** `main.py` -> `api.py` -> [`transcription.py`, `audio.py`, `diarization/`, `batching.py`] -> `models.py`, `config.py`

**No circular dependencies.** `batching.py` uses a lazy import of `transcription.transcribe_audio_batch` inside `_flush_batch()` to avoid circular import with potential future changes.

## Where to Add New Code

**New API Endpoint:**
- Add to `api.py` inside `create_app()` function, after existing endpoints
- Follow pattern: use `@app.get()` or `@app.post()` decorator
- Add Pydantic response models to `models.py`

**New Audio Processing Feature:**
- Add to `audio.py`
- Follow pattern: blocking functions for executor use, async wrappers for direct await

**New ML Model or Pipeline:**
- Load in `api.py` `startup_event()` as a global singleton
- Add semaphore if GPU-bound
- Wrap blocking inference in `loop.run_in_executor(None, ...)`

**New Configuration Option:**
- Add env var reading in `config.py` `Config._initialize()` with `.strip()` for CRLF safety
- Add to `Config.as_dict()` for health endpoint exposure
- Document in CLAUDE.md `.env` section

**New Response Format:**
- Add format handler in `api.py` at the response format switch (line ~356)
- Add formatting function in `transcription.py` following `format_srt()`/`format_vtt()` pattern

**New Pydantic Model:**
- Add to `models.py`

**New Test:**
- Add to `tests/` directory
- Follow existing naming: `test_*.py`

## Special Directories

**`/tmp/parakeet` (runtime):**
- Purpose: Temporary audio files during processing (uploads, WAV conversions, chunks)
- Generated: Yes, at runtime
- Committed: No
- Cleaned up per-request in `finally` block and on shutdown

**`experiments/`:**
- Purpose: Sample audio files for testing/benchmarking
- Generated: No (manually placed)
- Committed: Partial

**`.planning/`:**
- Purpose: GSD planning and codebase analysis documents
- Generated: By tooling
- Committed: Yes

---

*Structure analysis: 2026-04-03*
