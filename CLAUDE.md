# Parakeet Diarized

FastAPI server wrapping `nvidia/parakeet-tdt-0.6b-v2` (NeMo RNNT ASR) + pyannote speaker diarization + LLM-powered meeting intelligence (IBM Granite 3.3 8B via Ollama). Exposes OpenAI-compatible `/v1/audio/transcriptions`, `/v1/meeting/analyze`, and a lightweight `/transcribe` endpoint for real-time browser extension streaming. Gradio frontend in `app.py`.

---

## Environment

- **Runs in WSL** (Ubuntu), not Windows — all runtime paths are `/mnt/c/_Dev/parakeet-diarized`
- Python venv: `./venv` — activate with `source venv/bin/activate`
- GPU: NVIDIA RTX 4090 (24GB VRAM), CUDA available in WSL
- **Ollama** installed natively in WSL (not Docker) — serves LLM on `http://localhost:11434`
- Server: `http://localhost:8000` | Frontend: `http://localhost:8001`
- **Note:** Dockerfile/docker-compose.yml exist but are not used day-to-day. The primary workflow is `start.ps1` → WSL → uvicorn directly.

---

## How to Run

### 1. Start Ollama (LLM server)
```bash
# In WSL — runs in background, serves Granite 3.3 8B on port 11434
ollama serve &>/dev/null &
```
First-time setup: `sudo apt-get install -y zstd && curl -fsSL https://ollama.com/install.sh | sh && ollama pull granite3.3:8b`

### 2. Start API server
```powershell
.\start.ps1
```
Kills any process on port 8000, activates venv, starts uvicorn with hot-reload.

### 3. Start Gradio frontend (separate terminal)
```bash
wsl bash -c "cd /mnt/c/_Dev/parakeet-diarized && source venv/bin/activate && python app.py"
```

### Health check (poll until model_loaded=true — takes 2-3 min)
```bash
curl http://localhost:8000/health
# llm_available: true confirms Ollama connection
```

### Test transcription
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@/path/to/audio.m4a \
  -F model=whisper-1 \
  -F timestamps=true \
  -F diarize=true
```

### Test real-time transcription (raw binary, for TransMeet extension)
```bash
# Generate a test float32 audio file from a WAV, then POST it:
python -c "
import numpy as np, scipy.io.wavfile as wav
sr, data = wav.read('/path/to/test.wav')
audio = data.astype(np.float32) / 32768.0
audio.tofile('/tmp/test.raw')
"
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/tmp/test.raw
# Returns: {"text": "transcribed text"}
```

### Test meeting analysis
```bash
curl -X POST http://localhost:8000/v1/meeting/analyze \
  -H "Content-Type: application/json" \
  -d '{"transcript":"Speaker 1: Lets review the migration. Speaker 2: I will handle backups by Friday."}'
```

---

## Key Files

| File | Purpose |
|------|---------|
| `api.py` | FastAPI app, 4-phase async request lifecycle, GPU semaphore, per-phase timing |
| `transcription.py` | NeMo model load + `transcribe_audio_batch()` (bulk GPU) + `transcribe_audio_numpy()` (real-time numpy) + optional `torch.compile()` |
| `audio.py` | ffmpeg WAV conversion + async parallel chunk extraction |
| `diarization/__init__.py` | pyannote.audio speaker diarization (singleton, loaded once at startup) |
| `llm.py` | LLM client — async Ollama wrapper for meeting intelligence extraction |
| `prompts.py` | System + user prompt templates for Granite 3.3 structured reasoning |
| `batching.py` | Cross-request chunk batching engine (opt-in via `ENABLE_BATCH_QUEUE`) |
| `config.py` | Singleton `Config`, reads all env vars |
| `app.py` | Gradio frontend — single file + batch upload UI + meeting analysis |
| `models.py` | Pydantic models (`WhisperSegment`, `TranscriptionResponse`, `MeetingIntelligence`) |
| `benchmark.py` | API benchmark script — sequential or concurrent file testing |
| `main.py` | uvicorn entrypoint |

---

## Configuration (`.env`)

```env
# GPU throughput
BATCH_SIZE=16                   # chunks per GPU batch — try 16-32 on RTX 4090
MAX_CONCURRENT_REQUESTS=1       # transcription GPU semaphore (keep 1; >1 needs multiple model instances)
MAX_CONCURRENT_DIARIZE=1        # diarization GPU semaphore (runs concurrently with transcription)
CHUNK_DURATION=30               # seconds per audio chunk (NeMo hard max ~40s)

# Authentication
HUGGINGFACE_ACCESS_TOKEN=...    # required for pyannote/speaker-diarization-3.1

# GPU optimization
TORCH_COMPILE=false             # opt-in: torch.compile() on encoder for kernel fusion (~10-40% speedup)
TORCH_COMPILE_MODE=default          # default=inductor (no nvcc needed), reduce-overhead (needs nvcc), max-autotune

# Diarization tuning (pyannote defaults: batch=1, step=0.1 — very slow)
DIARIZE_SEGMENTATION_BATCH_SIZE=8   # segments per GPU batch in pyannote segmentation model
DIARIZE_EMBEDDING_BATCH_SIZE=8      # segments per GPU batch in pyannote embedding model
DIARIZE_SEGMENTATION_STEP=0.3       # sliding window step ratio (0.1=90% overlap, 0.3=70%, 0.5=50%). Higher=faster, less precise boundaries. 0.3 tested identical to 0.1 on 23min meeting.

# Cross-request batching (for concurrent workloads)
ENABLE_BATCH_QUEUE=false        # merge chunks from multiple requests into one GPU call
BATCH_QUEUE_MAX_WAIT=0.5        # max seconds to wait before flushing an incomplete batch

# Reliability
REQUEST_TIMEOUT=300             # seconds before 504 timeout (0 = no timeout)

# LLM meeting intelligence (Ollama sidecar)
LLM_ENABLED=true                # enable/disable LLM analysis feature
LLM_BASE_URL=http://localhost:11434  # Ollama API (use http://ollama:11434 in Docker)
LLM_MODEL=granite3.3:8b         # IBM Granite 3.3 8B Instruct (Q4_K_M, ~5GB VRAM)
LLM_TIMEOUT=120                 # seconds before LLM request timeout
LLM_MAX_TOKENS=4096             # max generation tokens for meeting analysis

# Optional
MODEL_ID=nvidia/parakeet-tdt-0.6b-v2
TEMP_DIR=/tmp/parakeet
ENABLE_DIARIZATION=true
INCLUDE_DIARIZATION_IN_TEXT=true
```

---

## Architecture: Request Lifecycle

```
HTTP POST /v1/audio/transcriptions
│
├── Phase 1 (no semaphore — concurrent with other requests)
│   ├── Save upload to temp file
│   ├── convert_audio_to_wav()  →  run_in_executor (blocking ffmpeg)
│   └── split_audio_into_chunks_async()  →  asyncio.gather (parallel ffmpeg)
│
├── Phase 2 (semaphores — GPU ops)
│   ├── diarize_semaphore: diarizer.diarize(wav_file)  →  run_in_executor
│   │       (runs concurrently with transcription via asyncio.create_task)
│   └── transcribe_semaphore: transcribe_audio_batch(all_chunks, batch_size=N)
│               model.transcribe([chunk1, chunk2, ...], batch_size=N)
│
├── Phase 3 (no semaphore)
│   ├── Apply chunk time offsets to segments
│   ├── diarizer.merge_with_transcription()
│   └── Prepend speaker labels to text (if enabled)
│
└── Phase 4 (optional, when analyze=true — Ollama manages its own GPU)
    ├── POST diarized transcript to Ollama /v1/chat/completions
    ├── Granite 3.3 8B: <think> reasoning → <response> JSON extraction
    ├── Parse MeetingIntelligence (summary, actions, decisions, questions...)
    └── Return JSON with text + segments + meeting_intelligence
```

### Real-time streaming endpoint (TransMeet browser extension)
```
HTTP POST /transcribe
  Content-Type: application/octet-stream
  Body: raw Float32Array (16kHz mono) — 2-30s chunks
│
├── np.frombuffer(body, dtype=float32)
├── transcribe_semaphore
├── model.transcribe(np_array)  →  run_in_executor (no ffmpeg, no temp files)
└── Return {"text": "..."}
```
No diarization (speaker detection is done via DOM in the browser extension).
No chunking (chunks are already ≤30s). No file I/O — numpy straight to GPU.

### Standalone analysis endpoint
```
POST /v1/meeting/analyze
  Body: {"transcript": "Speaker 1: ..."}
  Returns: MeetingIntelligence JSON
```

---

## Model Details

- **Model**: `nvidia/parakeet-tdt-0.6b-v2` — Token-and-Duration Transducer, ~600M params
- **Class**: `EncDecRNNTBPEModel` (NOT `EncDecCTCModelBPE`)
- **Precision**: fp16 (`model.half()`) — ~2x throughput vs fp32
- **Timestamp math**: `offset × window_stride(0.01s) × subsampling_factor(8) = 0.08s per unit`
  - Stored at runtime as `model._secs_per_offset`
- **Load time**: ~2-3 minutes on first start

---

## Known Bugs & Hard-Won Fixes

| Bug | Fix |
|-----|-----|
| CUDA graph API mismatch: `cuda-python` returns 5 values, NeMo TDT expects 6 → all transcriptions empty | `decoding_cfg.greedy.use_cuda_graph_decoder = False` in `load_model()` |
| `timestamps=True` kwarg throws internal NeMo TDT unpack error | Use `return_hypotheses=True` instead; access `.timestamp` on Hypothesis object |
| NeMo 2.x `model.transcribe()` returns `(List[str], List[Hypothesis])` tuple | Check for 2-element tuple of lists; use `transcriptions[1]` (hypotheses) |
| pyannote 3.3+ `DiarizeOutput` has no `.segments` — use `.speaker_diarization` | See `diarization/__init__.py` |
| `if candidate` on empty tensor raises `RuntimeError` | Use `if candidate is not None` |
| `model.transcribe()` is blocking — called from async handler blocks event loop | Always wrap in `loop.run_in_executor(None, ...)` |
| Wrong model class (`EncDecCTCModelBPE`) produces no output | Must use `EncDecRNNTBPEModel` |
| `num_workers > 0` in `model.transcribe()` hangs at `0it` in WSL | WSL fork-based multiprocessing deadlocks with CUDA. Keep `num_workers` at default (omit the arg). |
| Gradio generator streaming not updating UI live | Requires `demo.queue()` before `demo.launch()`. Also: use `gr.update()` (not `None`) for State in intermediate yields — `None` resets state and fires `.change` events mid-run. |
| `torch.compile(mode="reduce-overhead")` fails in WSL with `PermissionError: nvcc` | WSL can't access `nvcc` for CUDA graph compilation. Use `mode="default"` (inductor backend) instead. |
| Windows CRLF in `.env` breaks boolean env var parsing | `os.environ.get("VAR").lower() == "true"` fails because value has trailing `\r`. Always `.strip()` before comparing. |
| `finally` cleanup deletes WAV while diarize_task is still reading it | Must cancel/await `diarize_task` before unlinking temp files. Initialize `diarize_task = None` before try block. |

---

## GPU Utilization Notes

The RTX 4090 (24GB VRAM) is capable of much higher throughput than default settings show. Key levers:

1. **`BATCH_SIZE`** — most impactful. Sends all audio chunks to GPU in one call. Default 16; try 32 for long recordings.
2. **`num_workers` is NOT safe in WSL** — NeMo DataLoader uses fork-based multiprocessing which deadlocks under WSL. Leave at default (0). This was tried and reverted.
3. **Concurrent diarization** — diarize and transcribe run in parallel (separate asyncio tasks + semaphores).
4. **Batch upload** — Gradio batch tab sends multiple files to server sequentially; each queues through the GPU semaphore.
5. **`TORCH_COMPILE=true`** — applies `torch.compile()` to encoder only (Conformer stack). Expected 10-40% speedup. First request is slow (compilation overhead). RNNT decoder is NOT compiled (dynamic control flow).
6. **`ENABLE_BATCH_QUEUE=true`** — cross-request chunk batching. Merges chunks from concurrent requests into one `model.transcribe()` call. Adds up to `BATCH_QUEUE_MAX_WAIT` latency per request. Best for batch/concurrent workloads.
7. **Diarizer singleton** — pyannote pipeline loaded once at startup, reused across requests (saves ~5-10s per diarized request).
8. **Per-phase timing** — every request logs a `timing:` line with phase1/phase2/phase3 breakdown and RTF (real-time factor).
9. **Diarization per-stage timing** — every diarize call logs `Diarization completed in Xs — segmentation=Xs embeddings=Xs discrete_diarization=Xs`. Embeddings is the bottleneck (~80% of diarization time).
10. **`DIARIZE_SEGMENTATION_STEP=0.3`** — most impactful diarization speedup. Reduces sliding window overlap from 90% to 70%, cutting embedding count by ~3x. Tested identical output quality on 23-min meeting. Reduced diarization from 35s to 17s.
11. **`DIARIZE_EMBEDDING_BATCH_SIZE=8`** and **`DIARIZE_SEGMENTATION_BATCH_SIZE=8`** — batch GPU work in pyannote (defaults were 1). Modest speedup. Avoid >8 for embedding batch as it causes GPU contention with concurrent NeMo transcription.

Use `benchmark.py` to measure: `python benchmark.py --file experiments/meeting.m4a`

GPU memory drop during diarization is normal — PyTorch's caching allocator frees intermediate tensors, and pyannote moves embeddings to CPU after extraction. Models stay on GPU.

---

## Gradio Frontend (`app.py`)

Runs on port 8001. Two tabs:

- **Single File** — record or upload one file, view transcript with speaker rename + export
- **Batch** — drop multiple audio files, process all sequentially, export ZIP of `.md` transcripts

Speaker labels are auto-detected; users can rename them in the table before exporting.

---

## WSL Gotchas

- Server must run inside WSL — NeMo/CUDA do not work from Windows Python
- `.\start.ps1` handles the WSL invocation from PowerShell
- Harmless noise in logs: `UtilTranslatePathList Z:\bin` (WSL path warning), NvOneLogger/telemetry warnings
- Empty chunks (silence at start/end of recording) are normal — produce `("", [])` and are skipped
