# Testing Patterns

**Analysis Date:** 2026-04-03

## Test Infrastructure

**Runner:**
- Python `unittest` (stdlib)
- No dedicated test framework installed (no pytest, nose, or similar in `requirements.txt`)
- No test configuration files (`pytest.ini`, `setup.cfg [tool:pytest]`, `pyproject.toml`)

**Assertion Library:**
- `unittest.TestCase` assertions: `self.assertEqual`, `self.assertIn`

**Mocking:**
- `unittest.mock`: `patch`, `MagicMock`
- Used to mock `subprocess.run` and `wave.open` in `tests/test_chunking.py`

**Run Commands:**
```bash
# Run all tests (from project root)
python -m unittest discover tests/

# Run specific test file
python -m unittest tests/test_chunking.py
python tests/test_chunking.py
```

## Test File Organization

**Location:**
- Separate `tests/` directory at project root
- NOT co-located with source files

**Naming:**
- `test_{module}.py` pattern

**Current test files:**
```
tests/
├── test_api.py        # Manual CLI integration test (NOT a unittest)
├── test_chunking.py   # Unit tests for audio chunking
```

**Important distinction:**
- `tests/test_api.py` is a **manual CLI script** (argparse-based), NOT an automated test suite. It requires a running server and an audio file argument. It exercises the API endpoints interactively.
- `tests/test_chunking.py` is the **only automated test file** with proper `unittest.TestCase` structure.

## Test Structure

**Suite Organization (`tests/test_chunking.py`):**
```python
class TestAudioChunking(unittest.TestCase):
    def create_test_wav(self, duration_seconds=10, sample_rate=16000):
        """Helper: generates a sine wave WAV file"""
        # Uses numpy + wave module to create real audio files
        ...

    def test_no_chunking_for_short_audio(self):
        """Verifies short files return as-is without splitting"""
        ...

    @patch('subprocess.run')
    def test_chunking_for_long_audio(self, mock_subprocess_run):
        """Verifies long files are split into correct number of chunks"""
        ...
```

**Patterns:**
- Helper method `create_test_wav()` generates real WAV files with numpy sine waves
- Manual cleanup in `try/finally` blocks (no `setUp`/`tearDown`)
- `sys.path.append()` hack to import from parent directory (no package install)

## Mocking

**Patterns:**
```python
@patch('subprocess.run')
def test_chunking_for_long_audio(self, mock_subprocess_run):
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stderr = b''
    mock_subprocess_run.return_value = mock_process

    # Also patches wave.open with a wrapper to fake duration
    with patch('wave.open', mock_wave_open):
        result = split_audio_into_chunks(test_file, chunk_duration=300)
```

**What is mocked:**
- `subprocess.run` (ffmpeg calls) -- avoids needing ffmpeg installed
- `wave.open` -- partially mocked to override duration while keeping real file I/O

**What is NOT mocked:**
- File system operations (real temp files created and cleaned up)
- numpy audio generation (real WAV files)

## Fixtures and Factories

**Test Data:**
```python
def create_test_wav(self, duration_seconds=10, sample_rate=16000):
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), False)
    data = np.sin(2 * np.pi * 440 * t) * 32767
    data = data.astype(np.int16)
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    return temp_file.name
```

**Location:**
- Inline in test class (no shared fixtures directory)
- `experiments/test.mp3` and `experiments/test.wav` exist as manual test audio files

## Benchmark Tool

**`benchmark.py`** is a dedicated performance testing script (not a unit test):
```bash
python benchmark.py --file experiments/meeting.m4a
python benchmark.py --dir experiments/ --concurrent 3
python benchmark.py --file experiments/test.mp3 --no-diarize --repeat 5
```
- Measures wall time, RTF (real-time factor), segments count
- Supports sequential and concurrent modes via `ThreadPoolExecutor`
- Uses raw `urllib.request` (no `requests` dependency) with manual multipart body construction
- Performs health check before benchmarking
- Requires running server

## Coverage

**Requirements:** None enforced. No coverage tool configured.

**No coverage reporting:** No `coverage`, `pytest-cov`, or similar in dependencies.

## Test Types

**Unit Tests:**
- `tests/test_chunking.py`: 2 tests covering `split_audio_into_chunks()` from `audio.py`
  - Short audio (no-split path)
  - Long audio (multi-chunk split path with mocked ffmpeg)

**Integration Tests:**
- `tests/test_api.py`: Manual CLI script that hits running server endpoints
  - `/v1/audio/transcriptions` with various parameters
  - `/health` endpoint
  - `/v1/models` endpoint
- NOT automated -- requires manual invocation with `--file` argument

**E2E Tests:**
- None

**Load/Performance Tests:**
- `benchmark.py` serves this role informally

## Test Coverage

| Area | Coverage | Notes |
|------|----------|-------|
| `audio.py` - `split_audio_into_chunks()` | Partial | 2 unit tests (short/long paths). Async variant untested. |
| `audio.py` - `convert_audio_to_wav()` | None | No tests |
| `audio.py` - `split_audio_into_chunks_async()` | None | No tests |
| `transcription.py` - `load_model()` | None | Would require GPU/NeMo mocking |
| `transcription.py` - `transcribe_audio_batch()` | None | No tests |
| `transcription.py` - `_parse_hypothesis()` | None | Complex parsing logic, no tests |
| `transcription.py` - `format_srt()` / `format_vtt()` | None | Pure functions, easy to test |
| `models.py` - Pydantic models | None | No schema validation tests |
| `models.py` - `TranscriptionResponse.dict()` | None | Custom dict behavior untested |
| `config.py` - `Config` singleton | None | Env var parsing untested |
| `diarization/__init__.py` - `Diarizer` | None | Would require pyannote mocking |
| `diarization/__init__.py` - `merge_with_transcription()` | None | Pure logic, easy to test |
| `batching.py` - `BatchingEngine` | None | Async queue logic untested |
| `api.py` - endpoints | Manual only | `test_api.py` CLI script, not automated |
| `app.py` - Gradio UI | None | No UI tests |

## Test Gaps

**Critical gaps (pure logic, easy to test):**
- `_parse_hypothesis()` in `transcription.py` -- complex timestamp parsing from NeMo hypotheses with multiple fallback paths. Most likely source of subtle bugs.
- `merge_with_transcription()` in `diarization/__init__.py` -- temporal overlap matching logic, assigns speakers to segments.
- `format_srt()` and `format_vtt()` in `transcription.py` -- subtitle formatting with speaker labels.
- `TranscriptionResponse.dict()` in `models.py` -- custom serialization that strips null segments.
- `Config` env var parsing in `config.py` -- boolean coercion, `.strip()` for CRLF, type conversions.

**Important gaps (require mocking but high value):**
- API endpoint integration tests using FastAPI `TestClient` -- currently no automated HTTP tests at all.
- `split_audio_into_chunks_async()` -- the async variant is used in production but only the sync variant has tests.
- `BatchingEngine` -- complex async queue/flush logic with futures, timeouts, and cross-request batching.

**Lower priority:**
- `load_model()` -- heavy GPU dependency, better tested via integration tests.
- Gradio UI flows -- would require Gradio test client or Playwright.

## Recommendations for New Tests

**Add tests here:** `tests/` directory, following `test_{module}.py` pattern.

**Immediate wins (no GPU needed):**
1. `tests/test_transcription.py` -- test `_parse_hypothesis()` with mock Hypothesis objects, `format_srt()`, `format_vtt()`
2. `tests/test_models.py` -- test `WhisperSegment` validation, `TranscriptionResponse.dict()` behavior
3. `tests/test_config.py` -- test env var parsing with `@patch.dict(os.environ, ...)`
4. `tests/test_diarization.py` -- test `merge_with_transcription()` with synthetic segment data

**Framework upgrade path:**
- Consider adding `pytest` to `requirements.txt` -- enables fixtures, parametrize, better assertion messages
- Add `conftest.py` with shared fixtures (test WAV generator, mock segments, mock hypotheses)
- Add `pytest-asyncio` for testing async functions like `split_audio_into_chunks_async()`

---

*Testing analysis: 2026-04-03*
