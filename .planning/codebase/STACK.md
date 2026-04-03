# Technology Stack

**Analysis Date:** 2026-04-03

## Languages

**Primary:**
- Python 3.10+ - All server code, frontend, benchmarking

**Secondary:**
- Bash - `docker-entrypoint.sh`, `run.sh`
- PowerShell - `start.ps1`, `stop.ps1`, `start_ui.ps1` (Windows/WSL launcher scripts)

## Runtime

**Environment:**
- Python 3.10 (pinned in Dockerfile)
- WSL (Ubuntu) on Windows - required for CUDA/NeMo
- CUDA 12.1 (Dockerfile base: `nvidia/cuda:12.1.1-devel-ubuntu22.04`)

**Package Manager:**
- pip (no poetry/pipenv)
- Lockfile: **missing** - only `requirements.txt` with unpinned versions

## Frameworks

**Core:**
- FastAPI - REST API server (`api.py`)
- Uvicorn - ASGI server (`main.py`)
- Pydantic - Request/response models (`models.py`, `config.py`)
- Gradio >= 4.0 - Web frontend (`app.py`, port 8001)

**ML/AI:**
- NeMo Toolkit (`nemo_toolkit`) - ASR model inference, `EncDecRNNTBPEModel`
- PyTorch (`torch`) - GPU compute, fp16 inference, optional `torch.compile()`
- pyannote-audio - Speaker diarization pipeline (`pyannote/speaker-diarization-3.1`)
- Transformers (HuggingFace) - Model downloading/caching

**Build/Dev:**
- Docker (multi-stage build) - `Dockerfile`, `docker-compose.yml`
- ffmpeg (system binary) - Audio format conversion and chunking

## Key Dependencies

**Critical (ML pipeline):**
| Package | Purpose |
|---------|---------|
| `nemo_toolkit` | NVIDIA NeMo ASR framework - loads and runs `parakeet-tdt-0.6b-v2` |
| `torch` | PyTorch GPU inference, fp16, cudnn.benchmark, optional torch.compile |
| `pyannote-audio` | Speaker diarization pipeline (segmentation + embedding + clustering) |
| `cuda-python` | CUDA bindings (note: API mismatch with NeMo TDT graph decoder, disabled) |
| `numpy` | Tensor/array operations |

**Infrastructure:**
| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP API framework |
| `uvicorn` | ASGI server |
| `pydantic` | Data validation and response models |
| `python-multipart` | File upload parsing for FastAPI |
| `gradio` | Web UI frontend |
| `python-dotenv` | `.env` file loading |
| `requests` | HTTP client (Gradio frontend calls API) |
| `pandas` | DataFrame for speaker rename table in Gradio UI |

**Audio processing:**
| Package | Purpose |
|---------|---------|
| `ffmpeg-python` | Python bindings for ffmpeg |
| `soundfile` | Audio file I/O |
| `librosa` | Audio analysis |
| `pydub` | Audio manipulation |
| `lhotse` | NeMo's audio data loading (CutSet/DataLoader) |

**NeMo ecosystem:**
| Package | Purpose |
|---------|---------|
| `hydra-core` | NeMo config management (OmegaConf) |
| `lightning` | PyTorch Lightning (NeMo dependency) |
| `fiddle` | NeMo configuration |
| `omegaconf` | Config dict used in `transcription.py` (`open_dict`) |
| `einops` | Tensor operations for NeMo/Conformer |
| `sentencepiece` | BPE tokenizer for NeMo |
| `webdataset` | NeMo data loading |
| `datasets` | HuggingFace datasets |

**Utility:**
| Package | Purpose |
|---------|---------|
| `nvidia-ml-py` | GPU monitoring (nvidia-smi Python bindings) |
| `jiwer` | Word error rate metrics |
| `editdistance` | Edit distance computation |
| `IPython` | Interactive debugging |

## ASR Model

**Model:** `nvidia/parakeet-tdt-0.6b-v2`
- Class: `EncDecRNNTBPEModel` (Token-and-Duration Transducer, ~600M params)
- Loaded from HuggingFace Hub via `EncDecRNNTBPEModel.from_pretrained()`
- Precision: fp16 (`model.half()`) for ~2x throughput
- Timestamp math: `offset * window_stride(0.01s) * subsampling_factor(8) = 0.08s per unit`
- Configurable in `.env` via `MODEL_ID`

**Diarization Model:** `pyannote/speaker-diarization-3.1`
- Requires HuggingFace access token (`HUGGINGFACE_ACCESS_TOKEN`)
- Singleton loaded once at startup, reused across requests
- Configurable batch sizes for segmentation and embedding stages

## API Surface

**Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/audio/transcriptions` | OpenAI Whisper-compatible transcription endpoint |
| `GET` | `/health` | Health check with GPU stats, model status, config |
| `GET` | `/v1/models` | List available models (OpenAI API compat) |

**`POST /v1/audio/transcriptions` - Form Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | UploadFile | required | Audio file (any ffmpeg-supported format) |
| `model` | str | `"whisper-1"` | Model name (ignored, always uses parakeet) |
| `language` | str | None | Language hint (not used by NeMo TDT) |
| `prompt` | str | None | Unused, kept for API compat |
| `response_format` | str | `"json"` | `json`, `verbose_json`, `text`, `srt`, `vtt` |
| `temperature` | float | 0.0 | Unused, kept for API compat |
| `timestamps` | bool | False | Include segment timestamps in response |
| `timestamp_granularities` | List[str] | None | Unused, kept for API compat |
| `vad_filter` | bool | False | Unused, kept for API compat |
| `word_timestamps` | bool | False | Word-level timestamps |
| `diarize` | bool | True | Enable speaker diarization |
| `include_diarization_in_text` | bool | None | Prepend speaker labels to text |

**Response Formats:**
- `json` / `verbose_json`: Returns `TranscriptionResponse` JSON
- `text`: Plain text transcription
- `srt`: SubRip subtitle format
- `vtt`: WebVTT subtitle format

## Structured Output (JSON Schemas)

**`TranscriptionResponse`** (`models.py`):
```python
{
    "text": str,                    # Full transcription text
    "segments": [WhisperSegment],   # Optional, included when timestamps=true
    "language": str | None,
    "task": "transcribe",
    "duration": float | None,
    "model": str | None
}
```

**`WhisperSegment`** (`models.py`):
```python
{
    "id": int,
    "seek": 0,
    "start": float,                 # Seconds
    "end": float,                   # Seconds
    "text": str,
    "tokens": [],
    "temperature": 0.0,
    "avg_logprob": 0.0,
    "compression_ratio": 1.0,
    "no_speech_prob": 0.1,
    "speaker": str | None           # e.g. "speaker_SPEAKER_00"
}
```

**`ModelInfo`** (`models.py`):
```python
{
    "id": str,
    "object": "model",
    "created": int,
    "owned_by": str,
    "permission": [dict],
    "root": str,
    "parent": str | None
}
```

**Health Response** (inline in `api.py`):
```python
{
    "status": "ok",
    "version": "1.0.0",
    "model_loaded": bool,
    "diarizer_loaded": bool,
    "model_id": str,
    "cuda_available": bool,
    "gpu_info": str | None,
    "torch_compile_enabled": bool,
    "gpu_memory_allocated_mb": float,
    "gpu_memory_reserved_mb": float,
    "gpu_max_memory_mb": float,
    "config": dict                   # Full config via Config.as_dict()
}
```

## Configuration

**Environment:**
- All config via environment variables, loaded through `python-dotenv`
- Singleton `Config` class in `config.py` reads all env vars at import time
- `.env` file present (not committed) - contains `HUGGINGFACE_ACCESS_TOKEN` and tuning params
- Windows CRLF safety: all env var reads use `.strip()` before comparison

**Key Config Categories:**
- GPU throughput: `BATCH_SIZE`, `MAX_CONCURRENT_REQUESTS`, `MAX_CONCURRENT_DIARIZE`, `CHUNK_DURATION`
- Diarization tuning: `DIARIZE_SEGMENTATION_BATCH_SIZE`, `DIARIZE_EMBEDDING_BATCH_SIZE`, `DIARIZE_SEGMENTATION_STEP`
- GPU optimization: `TORCH_COMPILE`, `TORCH_COMPILE_MODE`
- Cross-request batching: `ENABLE_BATCH_QUEUE`, `BATCH_QUEUE_MAX_WAIT`
- Reliability: `REQUEST_TIMEOUT`

## LLM Integration (Planned/Documented)

**Current role:** No LLM is invoked in the codebase. The docs folder contains research notes on using a local LLM for post-processing:
- `docs/LLM2.md` - Prompt engineering guide for **Granite 3.3 8B Instruct** to extract structured meeting summaries (action items, unresolved questions) from diarized transcripts
- Uses `<think>` / `<response>` tags for chain-of-thought reasoning
- Targets greedy decoding (temperature=0) with structured JSON output
- No code integration exists yet - this is a planned feature

## Build & Deploy

**Docker:**
- Multi-stage Dockerfile: builder (`nvidia/cuda:12.1.1-devel`) + runtime (`nvidia/cuda:12.1.1-runtime`)
- `docker-compose.yml` with GPU passthrough, health checks, volume caching
- Exposes ports 8000 (API) and 8001 (Gradio)
- 5-minute health check start period for model loading
- Volume mounts for HuggingFace and PyTorch model caches

**Local (WSL):**
- `start.ps1` - PowerShell launcher (kills port 8000, activates venv, starts uvicorn)
- `docker-entrypoint.sh` - Starts API server in background, waits for health, then launches Gradio

**System Dependencies:**
- ffmpeg (audio conversion/chunking)
- libsndfile1 (audio I/O)
- NVIDIA GPU driver + CUDA toolkit

## Platform Requirements

**Development:**
- Windows with WSL (Ubuntu) - NeMo/CUDA require Linux
- NVIDIA GPU with CUDA support (tested on RTX 4090, 24GB VRAM)
- Python 3.10
- ffmpeg installed system-wide
- HuggingFace access token for pyannote diarization model

**Production:**
- Docker with NVIDIA Container Toolkit
- Single GPU with >= 8GB VRAM (fp16 model ~1.5GB + diarization ~2-4GB + batch working memory)

---

*Stack analysis: 2026-04-03*
