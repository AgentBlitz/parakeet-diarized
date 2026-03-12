# Parakeet Diarized

FastAPI server wrapping `nvidia/parakeet-tdt-0.6b-v2` (NeMo RNNT ASR) + pyannote speaker diarization. Exposes an OpenAI-compatible `/v1/audio/transcriptions` endpoint. Gradio frontend in `app.py`.

---

## Environment

- **Runs in WSL** (Ubuntu), not Windows — all runtime paths are `/mnt/c/_Dev/parakeet-diarized`
- Python venv: `./venv` — activate with `source venv/bin/activate`
- GPU: NVIDIA RTX 4090 (24GB VRAM), CUDA available in WSL
- Server: `http://localhost:8000` | Frontend: `http://localhost:7860`

---

## How to Run

### Start API server
```powershell
.\start.ps1
```
Kills any process on port 8000, activates venv, starts uvicorn with hot-reload.

### Start Gradio frontend (separate terminal)
```bash
wsl bash -c "cd /mnt/c/_Dev/parakeet-diarized && source venv/bin/activate && python app.py"
```

### Health check (poll until model_loaded=true — takes 2-3 min)
```bash
curl http://localhost:8000/health
```

### Test transcription
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@/path/to/audio.m4a \
  -F model=whisper-1 \
  -F timestamps=true \
  -F diarize=true
```

---

## Key Files

| File | Purpose |
|------|---------|
| `api.py` | FastAPI app, 3-phase async request lifecycle, GPU semaphore |
| `transcription.py` | NeMo model load + `transcribe_audio_batch()` (bulk GPU processing) |
| `audio.py` | ffmpeg WAV conversion + async parallel chunk extraction |
| `diarization/__init__.py` | pyannote.audio speaker diarization |
| `config.py` | Singleton `Config`, reads all env vars |
| `app.py` | Gradio frontend — single file + batch upload UI |
| `models.py` | Pydantic models (`WhisperSegment`, `TranscriptionResponse`) |
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
│               model.transcribe([chunk1, chunk2, ...], batch_size=N, num_workers=4)
│
└── Phase 3 (no semaphore)
    ├── Apply chunk time offsets to segments
    ├── diarizer.merge_with_transcription()
    ├── Prepend speaker labels to text (if enabled)
    └── Return JSON / text / SRT / VTT
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

---

## GPU Utilization Notes

The RTX 4090 (24GB VRAM) is capable of much higher throughput than default settings show. Key levers:

1. **`BATCH_SIZE`** — most impactful. Sends all audio chunks to GPU in one call. Default 16; try 32 for long recordings.
2. **`num_workers=4`** in `model.transcribe()` — overlaps CPU audio preprocessing with GPU inference.
3. **Concurrent diarization** — diarize and transcribe run in parallel (separate asyncio tasks + semaphores).
4. **Batch upload** — Gradio batch tab sends multiple files to server sequentially; each queues through the GPU semaphore.

If GPU still appears low: check `nvidia-smi` during a long file (>5 min). Short files spend most time in Phase 1 (ffmpeg) which is CPU-only.

---

## Gradio Frontend (`app.py`)

Runs on port 7860. Two tabs:

- **Single File** — record or upload one file, view transcript with speaker rename + export
- **Batch** — drop multiple audio files, process all sequentially, export ZIP of `.md` transcripts

Speaker labels are auto-detected; users can rename them in the table before exporting.

---

## WSL Gotchas

- Server must run inside WSL — NeMo/CUDA do not work from Windows Python
- `.\start.ps1` handles the WSL invocation from PowerShell
- Harmless noise in logs: `UtilTranslatePathList Z:\bin` (WSL path warning), NvOneLogger/telemetry warnings
- Empty chunks (silence at start/end of recording) are normal — produce `("", [])` and are skipped
