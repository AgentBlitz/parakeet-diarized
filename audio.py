import asyncio
import json
import os
import struct
import tempfile
import logging
import subprocess
import math
import wave
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def _probe_input_file(audio_path: str) -> None:
    """Log codec/format info of the input file via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                audio_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.warning(f"ffprobe failed: {result.stderr[:200]}")
            return

        info = json.loads(result.stdout)
        fmt = info.get("format", {})
        logger.info(
            f"Input probe: format={fmt.get('format_name')} duration={fmt.get('duration')}s "
            f"size={fmt.get('size')} bitrate={fmt.get('bit_rate')}"
        )
        for s in info.get("streams", []):
            if s.get("codec_type") == "audio":
                logger.info(
                    f"  audio stream: codec={s.get('codec_name')} "
                    f"sample_rate={s.get('sample_rate')} channels={s.get('channels')} "
                    f"channel_layout={s.get('channel_layout')} "
                    f"bits_per_sample={s.get('bits_per_sample')}"
                )
    except Exception as e:
        logger.warning(f"ffprobe error: {e}")


def _log_wav_properties(wav_path: str) -> int:
    """Read the converted WAV and log sample rate, duration, peak & RMS amplitude.
    Returns peak amplitude (0 if file is empty or unreadable)."""
    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / sample_rate if sample_rate else 0
            raw = wf.readframes(n_frames)

        # Decode int16 samples
        n_samples = len(raw) // 2
        if n_samples == 0:
            logger.warning(f"WAV diagnostic: {wav_path} has 0 samples (empty file)")
            return 0

        samples = struct.unpack(f"<{n_samples}h", raw)
        peak = max(abs(s) for s in samples)
        rms = math.sqrt(sum(s * s for s in samples) / n_samples)

        logger.info(
            f"WAV diagnostic: rate={sample_rate} ch={n_channels} frames={n_frames} "
            f"duration={duration:.2f}s peak={peak} rms={rms:.1f} "
            f"file_size={os.path.getsize(wav_path)}"
        )
        if peak < 100:
            logger.warning(
                f"WAV appears SILENT (peak={peak} < 100 on int16 scale) — "
                f"ffmpeg may have failed to decode the input audio"
            )
        return peak
    except Exception as e:
        logger.warning(f"WAV diagnostic error: {e}")
        return 0

def split_audio_into_chunks(audio_path: str, chunk_duration: int = 300) -> List[str]:
    """
    Split a long audio file into smaller chunks for processing.
    
    Args:
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds (default: 5 minutes)
        
    Returns:
        List of paths to the chunked audio files
    """
    try:
        # Check audio duration using wave module
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / rate
            
        logger.info(f"Audio duration: {duration:.2f} seconds")
        
        # If duration is less than chunk_duration, no need to split
        if duration <= chunk_duration:
            logger.info("Audio is shorter than chunk duration, no splitting needed")
            return [audio_path]
            
        # Calculate number of chunks
        num_chunks = math.ceil(duration / chunk_duration)
        logger.info(f"Splitting audio into {num_chunks} chunks")
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp()
        chunk_paths = []
        
        # Process each chunk
        for i in range(num_chunks):
            start_time = i * chunk_duration
            output_path = os.path.join(temp_dir, f"chunk_{i}.wav")
            
            # Use ffmpeg to extract chunk
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output files
                "-ss", str(start_time),  # Start time
                "-i", audio_path,  # Input file
                "-t", str(chunk_duration),  # Duration to extract
                "-c:a", "pcm_s16le",  # Audio codec
                "-ar", "16000",  # Sample rate
                "-ac", "1",  # Mono audio
                output_path
            ]
            
            logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error splitting audio chunk {i}: {result.stderr}")
                raise Exception(f"Failed to split audio: {result.stderr}")
                
            chunk_paths.append(output_path)
            
        return chunk_paths
        
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        # If there's an error, return the original file
        return [audio_path]

def convert_audio_to_wav(audio_path: str) -> str:
    """
    Convert any audio format to WAV format (16kHz, mono, 16-bit PCM)
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Path to the converted WAV file
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    output_path = temp_file.name
    
    try:
        # Probe input file for codec/format diagnostics
        _probe_input_file(audio_path)

        # Use ffmpeg to convert audio
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files
            "-i", audio_path,  # Input file
            "-c:a", "pcm_s16le",  # Audio codec (16-bit PCM)
            "-ar", "16000",  # Sample rate (16kHz)
            "-ac", "1",  # Mono audio
            output_path
        ]

        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Error converting audio: {result.stderr}")
            raise Exception(f"Failed to convert audio: {result.stderr}")

        # Log stderr even on success — catches codec warnings
        if result.stderr:
            logger.debug(f"ffmpeg stderr (success): {result.stderr[:500]}")

        # Diagnostic: log WAV properties to catch silent/corrupt output
        peak = _log_wav_properties(output_path)
        if peak == 0:
            raise Exception(
                "Audio conversion produced a completely silent WAV (peak=0). "
                "The input file likely contains no audio data. "
                f"Input size: {os.path.getsize(audio_path)} bytes"
            )

        return output_path
        
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        # Clean up temporary file
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        raise e


async def split_audio_into_chunks_async(audio_path: str, chunk_duration: int = 300) -> List[str]:
    """
    Split a long audio file into smaller chunks in parallel using asyncio subprocesses.

    Args:
        audio_path: Path to the WAV audio file
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of paths to the chunked audio files (in order)
    """
    try:
        with wave.open(audio_path, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()

        logger.info(f"Audio duration: {duration:.2f} seconds")

        if duration <= chunk_duration:
            logger.info("Audio is shorter than chunk duration, no splitting needed")
            return [audio_path]

        num_chunks = math.ceil(duration / chunk_duration)
        logger.info(f"Splitting audio into {num_chunks} chunks (parallel)")

        temp_dir = tempfile.mkdtemp()
        chunk_paths = [os.path.join(temp_dir, f"chunk_{i}.wav") for i in range(num_chunks)]

        # Cap concurrent ffmpeg processes to avoid I/O thrash
        ffmpeg_sem = asyncio.Semaphore(8)

        async def extract_chunk(i: int) -> None:
            async with ffmpeg_sem:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(i * chunk_duration),
                    "-i", audio_path,
                    "-t", str(chunk_duration),
                    "-c:a", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    chunk_paths[i]
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"ffmpeg chunk {i} failed: {stderr.decode()}")

        await asyncio.gather(*[extract_chunk(i) for i in range(num_chunks)])

        # Diagnostic: log WAV properties of each chunk to catch silent/corrupt splits
        for i, cp in enumerate(chunk_paths):
            try:
                with wave.open(cp, "rb") as wf:
                    ch_frames = wf.getnframes()
                    ch_rate = wf.getframerate()
                    ch_dur = ch_frames / ch_rate if ch_rate else 0
                    raw = wf.readframes(ch_frames)
                n_samples = len(raw) // 2
                if n_samples > 0:
                    samples = struct.unpack(f"<{n_samples}h", raw)
                    peak = max(abs(s) for s in samples)
                    rms = math.sqrt(sum(s * s for s in samples) / n_samples)
                else:
                    peak, rms = 0, 0.0
                logger.info(
                    f"Chunk {i} diagnostic: duration={ch_dur:.2f}s frames={ch_frames} "
                    f"peak={peak} rms={rms:.1f} size={os.path.getsize(cp)}"
                )
                if peak < 100:
                    logger.warning(f"Chunk {i} appears SILENT (peak={peak})")
            except Exception as e:
                logger.warning(f"Chunk {i} diagnostic error: {e}")

        return chunk_paths

    except Exception as e:
        logger.error(f"Error splitting audio (async): {str(e)}")
        return [audio_path]