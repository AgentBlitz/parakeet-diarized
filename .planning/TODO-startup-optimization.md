# Startup Time Optimization Plan

## Context
The server takes 50-120 seconds to start (until `/health` returns `model_loaded=true`). The bottlenecks are:
- **NeMo ASR model load**: 45-90s (`EncDecRNNTBPEModel.from_pretrained()` + `.cuda()` + `.half()` + decoding config)
- **Pyannote diarizer load**: 5-15s (`Pipeline.from_pretrained()` + `.to(cuda)` + config)
- These run **sequentially** in `api.py:57-103`

The first real request also pays an additional 2-8s "cold start" penalty (cuDNN autotuning, CUDA allocator warmup).

## Changes

### 1. CUDA Context Pre-initialization (`api.py`)
Add `torch.cuda.init()` + tiny allocation at the start of `startup_event()`, before model loading. Currently the CUDA context only fully initializes on the first `.cuda()` call inside `load_model()`.

**Saves: ~0.5-2s** (and prevents thread contention in step 2)

### 2. Parallel Model Loading (`api.py`)
Load NeMo ASR model and pyannote diarizer **concurrently** using `ThreadPoolExecutor`. Both are independent — different models, different HF cache entries, different GPU memory regions. Python threads release the GIL during disk I/O and CUDA operations, so they genuinely overlap.

```
Before: load_model (45-90s) → Diarizer (5-15s) = 50-105s
After:  load_model (45-90s) ─┐
        Diarizer  (5-15s)  ──┘ = 45-90s (diarizer is "free")
```

**Saves: 5-15s** (the entire diarizer load time)

### 3. GPU Warmup with Dummy Inference (`transcription.py`)
After models are loaded, run a 1-second silent audio through `model.transcribe()` to trigger:
- cuDNN kernel autotuning (`cudnn.benchmark = True` was set during load)
- CUDA memory allocator pool sizing
- Any lazy PyTorch kernel initialization

This shifts the "cold first request" penalty from the user's first real request to startup.

**Saves: 2-8s on first request** (adds ~2-3s to startup, but net UX improvement)

### 4. Startup Timing Instrumentation (`api.py`)
Add `time.perf_counter()` measurements around each phase so we can see exactly where time goes and validate improvements. Log a summary line like:
```
Startup complete: cuda_init=0.5s, model_load=52.3s, diarizer_load=0.0s (parallel), warmup=2.1s, total=54.9s
```

## Files to Modify

| File | Change |
|------|--------|
| `api.py` | Add `ThreadPoolExecutor` import; rewrite `startup_event()` with CUDA pre-init, parallel loading, warmup call, timing |
| `transcription.py` | Add `warmup_model(model)` function — generates 1s silent numpy array, runs `model.transcribe()`, cleans up |

## Expected Results
- **Startup time**: ~6-17 seconds faster (parallel loading + CUDA pre-init)
- **First request**: ~2-8 seconds faster (warmup absorbs cold-start penalty)
- **No config changes needed** — no new env vars required

## Verification
1. Start server with `.\start.ps1`
2. Watch logs for the new timing summary line
3. Compare startup time against baseline (~50-120s currently)
4. `curl http://localhost:8000/health` — should still show `model_loaded: true`
5. Send a test transcription — first request should be fast (no cold-start penalty)

## Future Investigation (not in this PR)
- Profile `from_pretrained()` breakdown to see if a local `.nemo` checkpoint cache would help further (could save 20-60s but adds complexity + staleness risk)
- ONNX/TensorRT export of models for faster load + inference
