import os
import logging
import tempfile
from typing import List, Optional, Dict, Any, Union, Tuple

import torch
import numpy as np

from models import WhisperSegment, TranscriptionResponse

logger = logging.getLogger(__name__)

def load_model(model_id: str = "nvidia/parakeet-tdt-0.6b-v2"):
    """
    Load the ASR model (Parakeet-TDT)

    Args:
        model_id: The HuggingFace model ID to load

    Returns:
        The loaded model
    """
    try:
        from nemo.collections.asr.models import EncDecRNNTBPEModel
        from omegaconf import open_dict

        logger.info(f"Loading model {model_id}")
        # parakeet-tdt is a TDT/RNNT model — use the correct class
        model = EncDecRNNTBPEModel.from_pretrained(model_id)

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            model = model.half()  # fp16: ~2x GPU throughput; revert to remove this line if accuracy degrades
            torch.backends.cudnn.benchmark = True  # auto-tune fastest convolution kernels for this GPU
            logger.info(f"Model loaded on GPU (fp16): {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, running on CPU (will be slow)")

        # Disable CUDA graph decoder: cuda-python API changed (returns 5 values, NeMo
        # expects 6), causing ValueError in TDT _full_graph_compile. The non-graph path
        # is fully functional and produces correct transcriptions.
        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.greedy.use_cuda_graph_decoder = False
            decoding_cfg.compute_timestamps = True
        model.change_decoding_strategy(decoding_cfg)
        logger.info("CUDA graph decoder disabled; compute_timestamps enabled")

        # Pre-compute seconds-per-offset for timestamp conversion:
        # offset units = encoder subsampling_factor × preprocessor window_stride
        window_stride = model.cfg.preprocessor.window_stride  # e.g. 0.01s
        subsampling = model.cfg.encoder.subsampling_factor     # e.g. 8
        model._secs_per_offset = window_stride * subsampling   # e.g. 0.08s

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def _format_timestamp(seconds: float, always_include_hours: bool = False,
                     decimal_marker: str = '.') -> str:
    """
    Format a timestamp as a string (HH:MM:SS.mmm)

    Args:
        seconds: Time in seconds
        always_include_hours: Always include hours in the output
        decimal_marker: Marker to use for decimal point

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds / 3600)
    seconds = seconds % 3600
    minutes = int(seconds / 60)
    seconds = seconds % 60

    hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""

    # Handle different format requirements (SRT vs VTT)
    if decimal_marker == ',':  # SRT format
        return f"{hours_marker}{minutes:02d}:{seconds:06.3f}".replace('.', decimal_marker)
    else:  # VTT format
        return f"{hours_marker}{minutes:02d}:{seconds:06.3f}"

def format_srt(segments: List[WhisperSegment]) -> str:
    """
    Format segments as SRT subtitle format

    Args:
        segments: List of transcription segments

    Returns:
        SRT formatted string
    """
    srt_content = ""
    for i, segment in enumerate(segments):
        segment_id = i + 1
        start = _format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')
        end = _format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')
        text = segment.text.strip().replace('-->', '->')

        # Format for SRT (with speaker if available)
        speaker_prefix = f"[{segment.speaker}] " if hasattr(segment, "speaker") and segment.speaker else ""
        srt_content += f"{segment_id}\n{start} --> {end}\n{speaker_prefix}{text}\n\n"

    return srt_content.strip()

def format_vtt(segments: List[WhisperSegment]) -> str:
    """
    Format segments as WebVTT subtitle format

    Args:
        segments: List of transcription segments

    Returns:
        WebVTT formatted string
    """
    vtt_content = "WEBVTT\n\n"
    for i, segment in enumerate(segments):
        start = _format_timestamp(segment.start, always_include_hours=True)
        end = _format_timestamp(segment.end, always_include_hours=True)
        text = segment.text.strip()

        # Format for VTT (with speaker if available)
        speaker_prefix = f"<v {segment.speaker}>" if hasattr(segment, "speaker") and segment.speaker else ""
        vtt_content += f"{start} --> {end}\n{speaker_prefix}{text}\n\n"

    return vtt_content.strip()

def _parse_hypothesis(result, model) -> Tuple[str, List[WhisperSegment]]:
    """
    Parse one NeMo Hypothesis (or plain string fallback) into (text, segments).
    Timestamps are relative to the chunk start (0-based); callers apply offsets.
    """
    # NeMo 2.x may return ([text,...], [hyp,...]) tuple as first element — unwrap
    if isinstance(result, (list, tuple)):
        logger.debug(f"result is {type(result).__name__}, unwrapping")
        result = result[0] if len(result) > 0 else ""

    text = result.text if hasattr(result, 'text') else str(result)
    logger.info(f"Parsed hypothesis type={type(result).__name__}, text length={len(text)}, preview={repr(text[:80])}")
    if not text:
        ts_attrs = {a: type(getattr(result, a, None)).__name__ for a in ('timestamp', 'timestep', 'score', 'y_sequence')}
        logger.warning(f"Empty transcription. Hypothesis attrs: {ts_attrs}")

    segments = []

    # NeMo uses result.timestamp (older) or result.timestep (newer) keyed by 'segment'
    ts_data = None
    for attr in ('timestamp', 'timestep'):
        candidate = getattr(result, attr, None)
        # Use `is not None` to avoid `bool(tensor)` which raises for empty tensors
        if candidate is not None and isinstance(candidate, dict) and 'segment' in candidate:
            ts_data = candidate
            break

    if ts_data is not None:
        secs_per_offset = getattr(model, '_secs_per_offset', 0.08)
        for i, stamp in enumerate(ts_data['segment']):
            start_off = stamp.get('start_offset', stamp.get('start', 0))
            end_off = stamp.get('end_offset', stamp.get('end', 0))
            segments.append(WhisperSegment(
                id=i,
                start=round(start_off * secs_per_offset, 3),
                end=round(end_off * secs_per_offset, 3),
                text=stamp['segment']
            ))
    else:
        segments.append(WhisperSegment(id=0, start=0.0, end=0.0, text=text))

    return text, segments


def transcribe_audio_batch(
    model,
    audio_paths: List[str],
    batch_size: int = 8,
    language: Optional[str] = None,
    word_timestamps: bool = False
) -> List[Tuple[str, List[WhisperSegment]]]:
    """
    Transcribe all audio chunks in a single model.transcribe() call.

    NeMo handles internal batching so the GPU processes `batch_size` chunks at once,
    maximising throughput. Returns results in the same order as audio_paths.
    Silent/tiny chunks (< 1000 bytes) are skipped and return ("", []).

    Args:
        model: Loaded ASR model
        audio_paths: Ordered list of chunk file paths
        batch_size: How many chunks the model processes per GPU batch
        language: Optional language hint (not used by NeMo TDT, kept for API compat)
        word_timestamps: Whether to include word-level timestamps

    Returns:
        List of (text, segments) tuples in the same order as audio_paths
    """
    results: List[Tuple[str, List[WhisperSegment]]] = [("", []) for _ in audio_paths]

    # Filter out tiny/silent chunks, preserving original indices
    valid = [(i, p) for i, p in enumerate(audio_paths)
             if os.path.getsize(p) >= 1000]
    if not valid:
        logger.warning("All chunks are too small — nothing to transcribe")
        return results

    valid_indices, valid_paths = zip(*valid)
    logger.info(f"Batch transcribing {len(valid_paths)} chunk(s) with batch_size={batch_size}")

    with torch.no_grad():
        try:
            transcriptions = model.transcribe(
                list(valid_paths),
                batch_size=batch_size,
                return_hypotheses=True
            )
        except Exception as e:
            logger.warning(f"Batch transcription with return_hypotheses failed ({e}), retrying plain")
            transcriptions = model.transcribe(list(valid_paths), batch_size=batch_size)

    if not transcriptions or len(transcriptions) == 0:
        logger.warning("model.transcribe() returned empty result")
        return results

    # NeMo 2.x may return (List[str], List[Hypothesis]) — use the hypotheses list
    if (isinstance(transcriptions, (list, tuple))
            and len(transcriptions) == 2
            and isinstance(transcriptions[0], list)
            and isinstance(transcriptions[1], list)):
        logger.debug("Unwrapping NeMo 2.x (texts, hypotheses) tuple")
        transcriptions = transcriptions[1]

    for orig_i, hyp in zip(valid_indices, transcriptions):
        results[orig_i] = _parse_hypothesis(hyp, model)

    return results


def transcribe_audio_chunk(model, audio_path: str, language: Optional[str] = None,
                           word_timestamps: bool = False) -> Tuple[str, List[WhisperSegment]]:
    """
    Transcribe a single audio chunk. Delegates to transcribe_audio_batch for consistency.

    Args:
        model: The loaded ASR model
        audio_path: Path to the audio file
        language: Optional language code
        word_timestamps: Whether to generate word-level timestamps

    Returns:
        Tuple of (transcription text, list of WhisperSegment objects)
    """
    file_size = os.path.getsize(audio_path)
    logger.info(f"Transcribing chunk: {audio_path}, size={file_size} bytes")
    if file_size < 1000:
        logger.warning(f"Chunk file suspiciously small ({file_size} bytes), skipping")
        return "", []

    try:
        return transcribe_audio_batch(
            model, [audio_path], batch_size=1,
            language=language, word_timestamps=word_timestamps
        )[0]
    except Exception as e:
        logger.error(f"Error transcribing audio chunk: {str(e)}")
        return "", []
