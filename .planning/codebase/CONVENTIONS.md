# Coding Conventions

**Analysis Date:** 2026-04-03

## Naming Patterns

**Files:**
- Snake_case for all Python modules: `transcription.py`, `audio.py`, `batching.py`
- Single-word or short descriptive names preferred
- Tests follow `test_{module}.py` pattern in `tests/` directory
- Package directories are lowercase: `diarization/`

**Functions:**
- Snake_case throughout: `load_model()`, `transcribe_audio_batch()`, `split_audio_into_chunks_async()`
- Private/internal functions prefixed with underscore: `_parse_hypothesis()`, `_format_timestamp()`, `_flush_batch()`, `_build_name_map()`
- Async functions use same naming but are `async def`

**Variables:**
- Snake_case for locals and module-level globals: `asr_model`, `diarizer_instance`, `wav_file`
- UPPER_SNAKE_CASE for module-level constants: `DEFAULT_HOST`, `DEFAULT_PORT`, `DEFAULT_MODEL_ID`, `AUDIO_EXTENSIONS`
- Private class attributes prefixed with underscore: `self._model`, `self._batch_size`, `self._queue`

**Types/Classes:**
- PascalCase: `WhisperSegment`, `TranscriptionResponse`, `DiarizationResult`, `BatchingEngine`
- Pydantic BaseModel subclasses for all data structures

## Code Style

**Formatting:**
- No formatter configuration detected (no `.prettierrc`, `pyproject.toml` style section, or `.editorconfig`)
- 4-space indentation (Python standard)
- f-strings used universally for string formatting
- Line length varies, no enforced maximum

**Linting:**
- No linter configuration detected (no `ruff.toml`, `.flake8`, `pylintrc`, or similar)
- No pre-commit hooks

**Type Hints:**
- Used consistently in function signatures: `def transcribe_audio_batch(model, audio_paths: List[str], batch_size: int = 8, language: Optional[str] = None) -> List[Tuple[str, List[WhisperSegment]]]:`
- Imports from `typing`: `List`, `Optional`, `Dict`, `Any`, `Union`, `Tuple`
- Not used for local variables

## Import Organization

**Order:**
1. Standard library (`os`, `logging`, `time`, `asyncio`, `tempfile`, `subprocess`, `math`, `wave`)
2. Third-party (`torch`, `numpy`, `fastapi`, `pydantic`, `gradio`, `requests`)
3. Local modules (`from models import WhisperSegment`, `from config import get_config`)

**Path Aliases:**
- None. All imports are direct module references from the project root.
- `sys.path.append()` used in `tests/test_chunking.py` to find parent modules (no proper package setup)

## Patterns

### Singleton Pattern
- `config.py`: `Config` class uses `__new__` for singleton: only one instance exists globally
- `diarization/__init__.py`: `Diarizer` initialized once at startup, stored in `api.py` global `diarizer_instance`
- Access via `get_config()` function in `config.py`

### Factory Pattern
- `api.py`: `create_app() -> FastAPI` builds and returns the configured application with all routes registered as closures

### GPU Semaphore Pattern
- Two independent `asyncio.Semaphore` instances in `api.py`:
  - `transcribe_semaphore` (default concurrency 1) guards `model.transcribe()` calls
  - `diarize_semaphore` (default concurrency 1) guards `diarizer.diarize()` calls
- Allows transcription and diarization to run concurrently on the same GPU

### Blocking-to-Async Bridge
- All GPU and subprocess work is wrapped in `loop.run_in_executor(None, ...)` to avoid blocking the asyncio event loop
- `functools.partial` used to pass arguments: `partial(diarizer.diarize, wav_file)`
- Async subprocess variant in `audio.py`: `split_audio_into_chunks_async()` uses `asyncio.create_subprocess_exec`

### Three-Phase Request Lifecycle
- Phase 1: File I/O (no semaphore) -- save upload, convert to WAV, split into chunks
- Phase 2: GPU work (semaphored) -- transcription + diarization run concurrently via `asyncio.create_task`
- Phase 3: Assembly (no semaphore) -- merge results, format response
- Per-phase timing logged with RTF (real-time factor)

### Environment Configuration
- All config from env vars with sensible defaults in `config.py`
- `.strip()` applied to all env var reads to handle Windows CRLF from `.env` files
- Boolean parsing: `os.environ.get("VAR", "false").strip().lower() == "true"`
- `python-dotenv` used to load `.env` file at import time

## Models & Schemas

### Pydantic Response Models

**`WhisperSegment`** (`models.py`):
```python
class WhisperSegment(BaseModel):
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: List[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.1
    speaker: Optional[str] = None  # Added for diarization
```
- Mirrors OpenAI Whisper API segment format
- `speaker` field is the only non-Whisper extension
- Default values for compatibility fields (`tokens`, `avg_logprob`, etc.) that Parakeet does not produce

**`TranscriptionResponse`** (`models.py`):
```python
class TranscriptionResponse(BaseModel):
    text: str
    segments: Optional[List[WhisperSegment]] = None
    language: Optional[str] = None
    task: str = "transcribe"
    duration: Optional[float] = None
    model: Optional[str] = None
```
- Custom `.dict()` method strips `segments` key when None (cleaner JSON output)
- `json_schema_extra` provides example for OpenAPI docs
- `segments` included only when `timestamps=true` or `response_format=verbose_json`

**`SpeakerSegment`** and **`DiarizationResult`** (`diarization/__init__.py`):
```python
class SpeakerSegment(BaseModel):
    start: float
    end: float
    speaker: str

class DiarizationResult(BaseModel):
    segments: List[SpeakerSegment]
    num_speakers: int
```
- Internal-only models, not exposed in API responses
- Speaker labels follow format `speaker_SPEAKER_00`, `speaker_SPEAKER_01`, etc.

**`ModelInfo`** and **`ModelList`** (`models.py`):
- OpenAI-compatible `/v1/models` endpoint response models
- Hardcoded `whisper-1` model ID for compatibility

**`BatchItem`** (`batching.py`):
```python
@dataclass
class BatchItem:
    chunk_path: str
    future: asyncio.Future
    request_id: str
    submitted_at: float = field(default_factory=time.monotonic)
```
- Uses `@dataclass` (not Pydantic) for internal queue items -- appropriate since no serialization needed

### Output Format Pipeline

The API supports five response formats controlled by `response_format` form parameter:

| Format | Content-Type | Handler |
|--------|-------------|---------|
| `json` | application/json | `TranscriptionResponse.dict()` |
| `verbose_json` | application/json | `TranscriptionResponse.dict()` (includes segments) |
| `text` | text/plain | Raw `full_text` string |
| `srt` | text/plain | `format_srt()` in `transcription.py` |
| `vtt` | text/plain | `format_vtt()` in `transcription.py` |

Speaker labels in SRT use `[speaker_SPEAKER_00]` prefix; VTT uses `<v speaker_SPEAKER_00>` voice tags.

### Diarization-Transcription Merge
- `Diarizer.merge_with_transcription()` assigns speakers to `WhisperSegment` objects by maximum temporal overlap
- Speaker labels optionally prepended to segment text: `Speaker 1: text` (first appearance) or `1: text` (subsequent)
- Controlled by `include_diarization_in_text` parameter (request-level) or `INCLUDE_DIARIZATION_IN_TEXT` env var (default)

## Error Handling

**Patterns:**
- `HTTPException` raised for client errors (400, 503) and timeouts (504)
- Generic `Exception` caught at top level of endpoint, re-raised as `HTTPException(500)`
- `HTTPException` re-raised without wrapping: `except HTTPException: raise`
- GPU/model failures return graceful defaults: `DiarizationResult(segments=[], num_speakers=0)`, `("", [])`
- `finally` blocks handle temp file cleanup, including canceling in-flight diarization tasks

**Startup resilience:**
- Model load failure is caught but does NOT crash the server -- health endpoint still works
- Diarizer init failure logged as error, diarization silently disabled

## Logging

**Framework:** Python `logging` module (stdlib)

**Setup:** Configured in `main.py`:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
```

**Patterns:**
- One logger per module: `logger = logging.getLogger(__name__)`
- Info-level for request lifecycle events, timing summaries, GPU memory
- Warning-level for degraded behavior (no CUDA, empty transcriptions, missing tokens)
- Error-level for failures that affect output
- Debug-level for detailed internal state (enabled via `DEBUG=1` env var)
- Noisy third-party loggers suppressed to ERROR in `main.py`: `megatron.core`, `nv_one_logger`, `nemo.*`, `lhotse`

**Structured timing logs:**
```
timing: phase1=0.45s(wav=0.30s chunks=0.15s) phase2=3.20s(transcribe=2.80s diarize_wait=0.00s) phase3=0.01s total=3.66s chunks=2 audio~60s rtf=0.0610
```

## Comments

**When to Comment:**
- Inline comments explain WHY, not WHAT: e.g., `# fp16: ~2x GPU throughput; revert to remove this line if accuracy degrades`
- Bug workarounds documented inline with context: `# cuda-python API changed (returns 5 values, NeMo expects 6)`
- Module-level docstrings explain purpose and design decisions (see `batching.py`)

**Docstrings:**
- Google-style docstrings on public functions: `Args:`, `Returns:` sections
- Not present on all internal functions
- Class-level docstrings are single-line descriptive

## Function Design

**Parameters:**
- Required params first, optional params with defaults after
- `Optional[str] = None` pattern for nullable parameters
- Config values read from singleton, not passed through call chains

**Return Values:**
- Tuples for multi-value returns: `Tuple[str, List[WhisperSegment]]`
- Pydantic models for API responses
- Empty defaults on failure: `("", [])`, `DiarizationResult(segments=[], num_speakers=0)`

## Module Design

**Exports:**
- No `__all__` declarations
- `main.py` re-exports key symbols for backward compatibility:
  ```python
  from audio import split_audio_into_chunks, convert_audio_to_wav
  from transcription import load_model, format_srt, format_vtt
  from models import WhisperSegment, TranscriptionResponse
  ```

**Barrel Files:**
- `diarization/__init__.py` serves as the package's single public interface

---

*Convention analysis: 2026-04-03*
