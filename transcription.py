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

        logger.info(f"Loading model {model_id}")
        # parakeet-tdt is a TDT/RNNT model — use the correct class
        model = EncDecRNNTBPEModel.from_pretrained(model_id)

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, running on CPU (will be slow)")

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

def transcribe_audio_chunk(model, audio_path: str, language: Optional[str] = None,
                          word_timestamps: bool = False) -> Tuple[str, List[WhisperSegment]]:
    """
    Transcribe a single audio chunk using the Parakeet-TDT model

    Args:
        model: The loaded ASR model
        audio_path: Path to the audio file
        language: Optional language code
        word_timestamps: Whether to generate word-level timestamps

    Returns:
        Tuple of (transcription text, list of WhisperSegment objects)
    """
    try:
        # Validate chunk before sending to model
        file_size = os.path.getsize(audio_path)
        logger.info(f"Transcribing chunk: {audio_path}, size={file_size} bytes")
        if file_size < 1000:
            logger.warning(f"Chunk file suspiciously small ({file_size} bytes), skipping")
            return "", []

        # Use the NeMo model to transcribe audio
        # return_hypotheses=True is the NeMo 2.x API for getting Hypothesis objects
        # with timestep data; timestamps=True triggers an internal unpack error in
        # newer NeMo TDT decoders so we avoid it.
        with torch.no_grad():
            try:
                transcription = model.transcribe(
                    [audio_path],
                    return_hypotheses=True
                )
            except Exception as ts_err:
                logger.warning(f"Transcription with return_hypotheses failed ({ts_err}), retrying plain")
                transcription = model.transcribe([audio_path])

        # Extract the text from the result
        if not transcription or len(transcription) == 0:
            logger.warning(f"No transcription generated for {audio_path}")
            return "", []

        # NeMo 2.0+ may return (List[str], List[Hypothesis]) tuple when return_hypotheses=True
        # is triggered internally. Without it, it returns List[str] or List[Hypothesis].
        result = transcription[0]
        if isinstance(result, (list, tuple)):
            # Unwrap one more level (e.g. NeMo 2.x returns ([text,...], [hyp,...]))
            logger.debug(f"transcription[0] is {type(result).__name__}, unwrapping")
            result = result[0] if len(result) > 0 else ""

        text = result.text if hasattr(result, 'text') else str(result)
        logger.info(f"Transcription result type={type(result).__name__}, text length={len(text)}, preview={repr(text[:80])}")
        if not text:
            ts_attrs = {a: type(getattr(result, a, None)).__name__ for a in ('timestamp', 'timestep', 'score', 'y_sequence')}
            logger.warning(f"Empty transcription for {audio_path}. Hypothesis attrs: {ts_attrs}")

        # Create segments from the timestamp information if available
        segments = []

        # Check if we have timestamp information
        # NeMo uses result.timestamp (older) or result.timestep (newer) keyed by 'segment'
        ts_data = None
        for attr in ('timestamp', 'timestep'):
            candidate = getattr(result, attr, None)
            # Use `is not None` to avoid `bool(tensor)` which raises for empty tensors
            if candidate is not None and isinstance(candidate, dict) and 'segment' in candidate:
                ts_data = candidate
                break

        if ts_data is not None:
            for i, stamp in enumerate(ts_data['segment']):
                segments.append(WhisperSegment(
                    id=i,
                    start=stamp['start'],
                    end=stamp['end'],
                    text=stamp['segment']
                ))
        else:
            # If no segments available, create a single segment for the entire chunk
            segments.append(WhisperSegment(
                id=0,
                start=0.0,
                end=len(text.split()) / 2.0,  # Rough estimate based on word count
                text=text
            ))

        return text, segments

    except Exception as e:
        logger.error(f"Error transcribing audio chunk: {str(e)}")
        return "", []
