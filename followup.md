Parakeet Diarized — session notes (2026-03-13)

## What was done this session

### Code changes
1. **Per-phase timing instrumentation** (`api.py`) — every request logs `timing:` line with phase1/phase2/phase3 breakdown, RTF
2. **GPU memory logging** (`transcription.py`) — logs allocated/peak MB before/after model.transcribe()
3. **Enhanced /health endpoint** — returns gpu_memory_allocated_mb, gpu_memory_reserved_mb, gpu_max_memory_mb, torch_compile_enabled, diarizer_loaded
4. **Diarizer singleton** (`api.py`) — pyannote pipeline loaded once at startup, reused across requests (was re-created every request, ~5-10s overhead)
5. **torch.compile() support** (`transcription.py`, `config.py`) — compiles encoder only (Conformer stack), opt-in via TORCH_COMPILE=true
6. **Reliability** — try/finally for temp file cleanup, graceful shutdown handler (torch.cuda.empty_cache), request timeout (REQUEST_TIMEOUT=300), cancel diarize_task before file cleanup
7. **Cross-request batch queue** (`batching.py`) — BatchingEngine with asyncio.Queue, merges chunks from concurrent requests into single GPU call. Opt-in via ENABLE_BATCH_QUEUE=true
8. **Benchmark script** (`benchmark.py`) — uses urllib (no external deps), sequential or concurrent mode, prints RTF table
9. **Config .strip() fix** — Windows CRLF in .env caused boolean env vars to fail string comparison

### Bugs found & fixed
1. **Diarization file deletion race**: The `finally` cleanup block deleted WAV files while diarize_task (asyncio background task) was still reading them → "File does not exist" errors. Fixed by initializing `diarize_task = None` before try, and cancelling/awaiting it before cleanup in finally.
2. **CRLF .env parsing**: Windows line endings in `.env` caused `"true\r".lower() == "true"` to fail. Added `.strip()` to all boolean env var reads in config.py.

### What DOES NOT work
- **torch.compile() — BROKEN in this WSL environment**. Both `reduce-overhead` AND `default` (inductor) modes fail with `PermissionError: Permission denied: 'nvcc'`. The inductor backend still invokes nvcc/triton for kernel compilation. This contaminates the model state — even the fallback retry path (`return_hypotheses=False`) then also fails with nvcc errors, causing 500s. **TORCH_COMPILE must stay false.** To make it work, `nvcc` would need to be installed and in PATH inside WSL (`apt install nvidia-cuda-toolkit` or add CUDA toolkit bin to PATH).
- **Hot-reload does NOT re-initialize Config singleton** — changing `.env` and waiting for uvicorn reload does not pick up new boolean config values. A full server restart (`.\start.ps1`) is always needed after `.env` changes.

### GPU memory observation
User noticed: GPU memory spikes during the first request then drops. This is normal — NeMo allocates temp CUDA buffers for the first batch, PyTorch's caching allocator reserves them, then they get reused on subsequent calls. The "peak" shown in /health reflects the high-water mark. The "allocated" is current usage. Nothing is being unloaded.

### Benchmark results (BATCH_SIZE=32, TORCH_COMPILE=false)

With diarization (this is the real-world scenario):
| File | Size | Audio duration | Elapsed | RTF |
|------|------|---------------|---------|-----|
| meeting.m4a | 8.5M | ~23min | 43.89s | 1.86 |
| test.mp3 | 1.2M | ~54s | 6.27s | 7.29 |
| test.wav | 4.6M | ~30s | 4.49s | 8.75 |

Without diarization (transcription only — shows GPU speed):
| File | Elapsed | RTF |
|------|---------|-----|
| meeting.m4a | 4.72s | 0.20 |
| test.mp3 | 3.18s | 3.76 |
| test.wav | 2.42s | 4.97 |

Concurrent (3 files at once, with diarization):
| Wall time | vs sequential |
|-----------|--------------|
| 39.04s | 29% faster than 54.65s sequential |

**Key finding**: Diarization (pyannote) is 88% of wall time for long files. GPU transcription is already very fast (RTF 0.20 = 5x faster than real-time for the 23min meeting). The biggest optimization target is diarization speed, not transcription.

BATCH_SIZE=32 vs 16: negligible difference (~7% on meeting.m4a, within noise).

### Current .env
```
HUGGINGFACE_ACCESS_TOKEN=hf_...
TORCH_COMPILE=false
BATCH_SIZE=32
```

### New files added
- `benchmark.py` — API benchmark script (uses urllib, no deps)
- `batching.py` — Cross-request chunk batching engine (not yet tested)

---

## Prompt for new chat session

```
Parakeet Diarized — continuing optimization work

I have a FastAPI audio transcription server (c:\_Dev\parakeet-diarized) running nvidia/parakeet-tdt-0.6b-v2 (NeMo RNNT) + pyannote speaker diarization on RTX 4090 in WSL. Gradio frontend in app.py.

Read CLAUDE.md first (architecture, known bugs, gotchas), then followup.md (detailed session notes with benchmarks).

Previous session summary:
- Added profiling/timing instrumentation, GPU memory logging, enhanced /health, benchmark.py
- Diarizer is now a singleton (loaded once at startup, not per-request)
- Added cross-request batch queue (batching.py, ENABLE_BATCH_QUEUE=true) — NOT YET TESTED
- Added reliability: try/finally cleanup, shutdown handler, request timeout, diarize_task race fix
- torch.compile() DOES NOT WORK in WSL — both reduce-overhead and default modes fail (nvcc not accessible). Keep TORCH_COMPILE=false.
- Fixed: CRLF .env parsing, diarize file deletion race

KEY BENCHMARK FINDING: Pyannote diarization is 88% of wall time. Transcription alone: RTF 0.20 (23min audio in 4.7s). With diarization: RTF 1.86 (43.9s). The GPU is fast — diarization is the bottleneck.

Diarization is required for all requests — we always need speaker labels.

Goals for this session:
1. Restart server clean (.\start.ps1), run benchmark.py to confirm everything works after fixes
2. Test cross-request batching: ENABLE_BATCH_QUEUE=true, benchmark --concurrent 3
3. MAIN GOAL: Optimize diarization speed — this is 88% of processing time. Ideas:
   - Chunk-level diarization (diarize 30s chunks instead of full audio — pyannote scales poorly with duration)
   - Move diarization to CPU (free GPU for transcription) — pyannote currently uses GPU
   - Faster pyannote pipeline config or alternative
   - Skip full pipeline if single-speaker quickly detected
4. Pin dependency versions (pip freeze)

Read CLAUDE.md and followup.md before touching anything.
```
