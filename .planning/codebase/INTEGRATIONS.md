# External Integrations

**Analysis Date:** 2026-04-03

## APIs & Services

**HuggingFace Hub:**
- Purpose: Download pretrained ASR model (`nvidia/parakeet-tdt-0.6b-v2`) and diarization pipeline (`pyannote/speaker-diarization-3.1`)
- SDK/Client: `nemo.collections.asr.models.EncDecRNNTBPEModel.from_pretrained()` and `pyannote.audio.Pipeline.from_pretrained()`
- Auth: `HUGGINGFACE_ACCESS_TOKEN` or `HF_TOKEN` env var (required for pyannote, model is public)
- Called once at startup, models cached locally

**No other external APIs are called.** This is a fully self-contained local inference server.

## Data Storage

**Databases:**
- None. No database is used. All state is in-memory.

**File Storage:**
- Local filesystem only
- Temp directory: `TEMP_DIR` env var (default `/tmp/parakeet`)
- Upload files saved to temp, converted to WAV, chunked, then cleaned up in `finally` block
- Docker volumes for model cache persistence:
  - `huggingface-cache` mounted at `/root/.cache/huggingface`
  - `torch-cache` mounted at `/root/.cache/torch`

**Caching:**
- HuggingFace model cache (disk) - models downloaded once, reused across restarts
- In-memory model singletons: ASR model (`asr_model`) and diarizer (`diarizer_instance`) loaded once at startup
- No application-level caching of transcription results

## Authentication & Identity

**Auth Provider:**
- None. The API has no authentication. CORS is fully open (`allow_origins=["*"]`).
- HuggingFace token is server-side only, used for model downloads

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Datadog, etc.)

**Logging:**
- Python `logging` module, configured in `main.py`
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Per-request timing logs: `timing: phase1=Xs phase2=Xs phase3=Xs total=Xs rtf=X`
- Per-diarization timing: `Diarization completed in Xs -- segmentation=Xs embeddings=Xs discrete_diarization=Xs`
- GPU memory logged before/after transcription and diarization
- Noisy third-party loggers suppressed to ERROR level in `main.py`

**Health Check:**
- `GET /health` returns model status, GPU memory stats, full config
- Docker healthcheck polls `/health` every 30s with 5-min startup grace period

**Benchmarking:**
- `benchmark.py` - CLI tool for measuring API throughput
- Reports per-file timing, RTF (real-time factor), segment counts
- Supports sequential and concurrent (`--concurrent N`) modes

## External System Dependencies

**ffmpeg (system binary):**
- Purpose: Audio format conversion (any format to 16kHz mono WAV) and chunk splitting
- Called via `subprocess.run()` (sync) and `asyncio.create_subprocess_exec()` (async parallel)
- Required at runtime - no Python fallback
- Location: System PATH (installed via apt in Docker)
- Used in: `audio.py` (`convert_audio_to_wav()`, `split_audio_into_chunks_async()`)

**NVIDIA GPU + CUDA:**
- Purpose: ML inference acceleration
- Detection: `torch.cuda.is_available()` at startup
- Falls back to CPU if unavailable (functional but very slow)
- Memory monitoring via `torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()`
- `nvidia-ml-py` package available for additional GPU monitoring

## CI/CD & Deployment

**Hosting:**
- Self-hosted (local machine or Docker)
- No cloud deployment configuration

**CI Pipeline:**
- None detected. No GitHub Actions, no CI config files.

## Environment Configuration

**Required env vars:**
- `HUGGINGFACE_ACCESS_TOKEN` - Required for pyannote diarization (gated model). Without it, diarization is disabled but transcription works.

**Optional env vars (with defaults):**
| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_ID` | `nvidia/parakeet-tdt-0.6b-v2` | ASR model to load |
| `BATCH_SIZE` | `16` | Chunks per GPU batch |
| `MAX_CONCURRENT_REQUESTS` | `1` | Transcription GPU semaphore |
| `MAX_CONCURRENT_DIARIZE` | `1` | Diarization GPU semaphore |
| `CHUNK_DURATION` | `30` | Seconds per audio chunk |
| `TORCH_COMPILE` | `false` | Enable torch.compile on encoder |
| `TORCH_COMPILE_MODE` | `default` | torch.compile mode |
| `ENABLE_BATCH_QUEUE` | `false` | Cross-request chunk batching |
| `BATCH_QUEUE_MAX_WAIT` | `0.5` | Max wait before flushing batch |
| `REQUEST_TIMEOUT` | `300` | Request timeout in seconds |
| `ENABLE_DIARIZATION` | `true` | Global diarization toggle |
| `INCLUDE_DIARIZATION_IN_TEXT` | `true` | Speaker labels in text output |
| `DIARIZE_SEGMENTATION_BATCH_SIZE` | `8` | Pyannote segmentation batch |
| `DIARIZE_EMBEDDING_BATCH_SIZE` | `8` | Pyannote embedding batch |
| `DIARIZE_SEGMENTATION_STEP` | `0.3` | Sliding window step ratio |
| `TEMP_DIR` | `/tmp/parakeet` | Temp file directory |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `DEBUG` | `0` | Debug logging |
| `API_URL` | `http://localhost:8000/v1/audio/transcriptions` | Gradio -> API URL |

**Secrets location:**
- `.env` file in project root (not committed to git)
- Docker: `env_file: .env` in `docker-compose.yml`

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Internal Service Communication

**Gradio Frontend -> FastAPI API:**
- `app.py` calls `http://localhost:8000/v1/audio/transcriptions` via `requests.post()`
- Configurable via `API_URL` env var
- Timeout: 600 seconds per request
- In Docker, both services run in the same container (started by `docker-entrypoint.sh`)

## LLM Post-Processing (Planned, Not Implemented)

**Granite 3.3 8B Instruct:**
- Documented in `docs/LLM2.md` as a planned integration
- Purpose: Extract structured meeting summaries from diarized transcripts
- Target output: JSON with `summary`, `action_items`, `unresolved_questions`
- Inference engine: vLLM, Ollama, or LM Studio (not yet chosen)
- No code exists for this integration yet

---

*Integration audit: 2026-04-03*
