# Fix Report: Empty Transcriptions from NeMo Parakeet-TDT

## Status: FIXED (parakeet server) + OPEN (TransMeet extension audio capture)

---

## Problem
NeMo's `model.transcribe()` returned near-empty text from valid audio. 66+ seconds of clear speech (WAV peak=20538, rms=838.7) produced only 18-73 characters. The RNNT decoder was emitting mostly blank tokens, completing in ~2s for 90s of audio (should take 5-10s).

## Root Cause: NeMo Lhotse Dataloader
NeMo 2.x defaults to `use_lhotse=True` in `model.transcribe()`. The Lhotse `DynamicBucketingSampler` was silently corrupting/truncating audio data when processing small batches (3-4 chunks). The legacy NeMo dataloader reads WAV files directly and works correctly.

Traced through NeMo venv source:
- `rnnt_models.py:244` -- `use_lhotse=True` default
- `rnnt_bpe_models.py:508` -- routes to `get_lhotse_dataloader_from_config()`
- `lhotse/dataloader.py:582` -- `DurationFilter`, `DynamicBucketingSampler` applied to cuts
- Manifest entries use `duration: 100000` placeholder (line 737) which Lhotse may misinterpret

## Fix Applied (2026-04-07)

### `transcription.py` -- `use_lhotse=False`
```python
model.transcribe(
    list(valid_paths),
    batch_size=batch_size,
    return_hypotheses=True,
    use_lhotse=False,  # bypass Lhotse DynamicBucketingSampler
)
```

### Additional changes in `transcription.py`:
- Hypothesis quality logging: `score`, `y_sequence.shape` for every chunk
- Auto-save sparse-output chunks to `/tmp/parakeet/debug/` for offline analysis
- Chunk-level WAV diagnostics in `audio.py` (peak, rms, duration per chunk)

### New file: `debug_transcribe.py`
Standalone script that tests `use_lhotse=True` vs `use_lhotse=False` side-by-side on any audio file.

## Test Results (2026-04-07)

| File | Before (Lhotse) | After (Legacy) |
|------|-----------------|----------------|
| `experiments/test.wav` | ~18-73 chars, 1-2 segments | **417 chars, 5 segments** -- full accurate transcription |
| `experiments/test.mp3` | not tested before | **674 chars, 8 segments** -- full accurate transcription |

Both tests produced correct speaker diarization and coherent multi-speaker text.

---

## OPEN ISSUE: TransMeet Sending Silent Audio

**This is a TransMeet extension problem, not a parakeet server problem.**

### Evidence from server logs (2026-04-07):
```
Job 5a81c75f: meeting.webm size=9107 bytes
  -> WAV: duration=35.94s peak=0 rms=0.0 -- COMPLETELY SILENT
  -> FAILED: "Audio conversion produced a completely silent WAV"

Job abfd9eef: meeting.webm size=3349 bytes
  -> WAV: duration=12.84s peak=0 rms=0.0 -- COMPLETELY SILENT
  -> FAILED: "Audio conversion produced a completely silent WAV"
```

### What the server sees:
- WebM/Opus files arrive with valid container metadata (48kHz, stereo, Opus codec)
- ffmpeg decodes them successfully to WAV (no errors)
- The resulting WAV has correct duration (12-36 seconds) but **zero audio content** (peak=0, rms=0.0)
- File sizes are suspiciously small: 3-9 KB for 12-36 seconds of audio (should be ~50-200 KB for Opus)

### What this means for TransMeet:
The Chrome extension's `MediaRecorder` is capturing a valid WebM container with correct timestamps and duration, but the actual audio samples are all zeros. The Opus encoder faithfully encodes silence.

### Likely cause in TransMeet:
The **AudioContext mixing path** (used when mic is available) creates a `MediaStreamSource` from tab capture audio, routes it through an AudioContext graph for tab+mic mixing, then records the mixed output. The AudioContext graph is not delivering audio samples -- only silence flows through.

Previous fixes (commits e1f831b, d081ca7) addressed AudioContext silence for the **offscreen document path** and **tab-only path**. But when a mic is connected, the code still uses the AudioContext mixing path, which has the same underlying issue: the `MediaStreamSource` from `chrome.tabCapture` doesn't start delivering audio through the AudioContext immediately (or at all in some cases).

### What parakeet server does correctly:
1. Accepts the WebM upload
2. Converts to WAV via ffmpeg (correct)
3. Detects the silent output via WAV diagnostics (peak=0)
4. Rejects with a clear error message
5. When valid audio IS provided (test.wav, test.mp3, meeting.m4a), transcription works perfectly

### What TransMeet needs to fix:
1. **Verify audio is non-silent before sending** -- check `AnalyserNode.getByteFrequencyData()` or compute RMS on the `AudioContext` output before starting `MediaRecorder`
2. **Debug the AudioContext mixing graph** -- the `MediaStreamSource` from tab capture may need explicit `.connect()` verification or the AudioContext may need user gesture activation
3. **Consider bypassing AudioContext for tab-only audio** -- if no mic mixing is needed, pass the tab capture stream directly to `MediaRecorder` (as the tab-only fix already does)
4. **Add a minimum file size check** -- 3-9 KB for 12-36 seconds is clearly wrong; abort and retry if the WebM is under ~20 KB for >5 seconds of recording
