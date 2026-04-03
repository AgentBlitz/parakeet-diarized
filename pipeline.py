"""Reusable 4-phase transcription pipeline.

Extracted from api.py so both the sync /v1/audio/transcriptions endpoint
and the async job worker can share the same logic.

Phases:
  1. Audio conversion + chunking (no GPU semaphore)
  2. GPU transcription + diarization (semaphore-guarded)
  3. Result assembly + speaker labels
  4. Optional LLM meeting analysis
"""

import asyncio
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import List, Optional

from models import WhisperSegment, TranscriptionResponse
from audio import convert_audio_to_wav, split_audio_into_chunks_async
from transcription import transcribe_audio_batch
from config import get_config

logger = logging.getLogger(__name__)
config = get_config()


async def run_transcription_pipeline(
    audio_file_path: str,
    model,
    diarizer,
    llm_client,
    transcribe_semaphore: asyncio.Semaphore,
    diarize_semaphore: asyncio.Semaphore,
    *,
    language: Optional[str] = None,
    diarize: bool = True,
    analyze: bool = False,
    include_diarization_in_text: Optional[bool] = None,
    response_format: str = "json",
    word_timestamps: bool = False,
    timestamps: bool = False,
) -> TranscriptionResponse:
    """Execute the full 4-phase transcription pipeline.

    Args:
        audio_file_path: Path to the uploaded audio file (any format ffmpeg supports).
        model: Loaded NeMo ASR model.
        diarizer: Diarizer instance (or None to skip diarization).
        llm_client: LLMClient instance (or None to skip analysis).
        transcribe_semaphore: asyncio.Semaphore for GPU transcription.
        diarize_semaphore: asyncio.Semaphore for GPU diarization.
        language: Optional language hint.
        diarize: Whether to run speaker diarization.
        analyze: Whether to run LLM meeting analysis.
        include_diarization_in_text: Override config for speaker labels in text.
        response_format: Output format (json, verbose_json, text, srt, vtt).
        word_timestamps: Whether to include word-level timestamps.
        timestamps: Whether to include segment timestamps.

    Returns:
        TranscriptionResponse with text, segments, and optional meeting_intelligence.

    Note:
        This function does NOT handle temp file cleanup. The caller is responsible
        for cleaning up audio_file_path and any intermediate files.
        However, it DOES clean up the WAV file and chunk files it creates internally.
    """
    chunk_duration = config.chunk_duration
    loop = asyncio.get_event_loop()
    t_request = time.perf_counter()

    wav_file = None
    audio_chunks = []
    diarize_task = None

    try:
        # --- Phase 1: File I/O (no GPU semaphore) ---
        t_phase1 = time.perf_counter()

        t_wav = time.perf_counter()
        wav_file = await loop.run_in_executor(None, convert_audio_to_wav, audio_file_path)
        t_wav_done = time.perf_counter()

        audio_chunks = await split_audio_into_chunks_async(wav_file, chunk_duration=chunk_duration)
        t_phase1_done = time.perf_counter()

        # Use diarizer if requested and available
        active_diarizer = diarizer if diarize else None
        if diarize and not active_diarizer:
            logger.warning("Diarization requested but diarizer not initialized (no HuggingFace token)")

        # --- Phase 2: GPU work (diarization and transcription run concurrently) ---
        t_phase2 = time.perf_counter()
        diarization_result = None
        batch_results = []

        async def _run_diarize():
            async with diarize_semaphore:
                logger.info("Performing speaker diarization")
                result = await loop.run_in_executor(
                    None, partial(active_diarizer.diarize, wav_file)
                )
                logger.info(f"Diarization found {result.num_speakers} speakers")
                return result

        diarize_task = asyncio.create_task(_run_diarize()) if active_diarizer else None

        logger.info(f"Batch transcribing {len(audio_chunks)} chunk(s) with batch_size={config.batch_size}")
        t_transcribe = time.perf_counter()

        async with transcribe_semaphore:
            try:
                batch_results = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(
                            transcribe_audio_batch,
                            model,
                            audio_chunks,
                            config.batch_size,
                            language,
                            word_timestamps
                        )
                    ),
                    timeout=config.request_timeout
                )
            except asyncio.TimeoutError:
                raise RuntimeError(f"Transcription timed out after {config.request_timeout}s")
        t_transcribe_done = time.perf_counter()

        # Wait for diarization to finish
        t_diarize_elapsed = 0.0
        if diarize_task:
            t_diarize_wait = time.perf_counter()
            diarization_result = await diarize_task
            t_diarize_elapsed = time.perf_counter() - t_diarize_wait
        t_phase2_done = time.perf_counter()

        # --- Phase 3: Assemble results (no GPU) ---
        t_phase3 = time.perf_counter()
        all_text = []
        all_segments = []

        for i, (chunk_text, chunk_segments) in enumerate(batch_results):
            offset = i * chunk_duration
            if offset > 0:
                for segment in chunk_segments:
                    segment.start += offset
                    segment.end += offset
            all_text.append(chunk_text)
            all_segments.extend(chunk_segments)

        full_text = " ".join(all_text)

        # Apply diarization if available
        if active_diarizer and diarization_result and diarization_result.segments:
            logger.info(f"Found {diarization_result.num_speakers} speakers")
            all_segments = active_diarizer.merge_with_transcription(diarization_result, all_segments)

            use_diarization_in_text = (
                include_diarization_in_text
                if include_diarization_in_text is not None
                else config.include_diarization_in_text
            )

            if use_diarization_in_text:
                logger.info("Including speaker labels in transcript text")
                previous_speaker = None
                seen_speakers = set()

                for segment in all_segments:
                    if hasattr(segment, 'speaker') and segment.speaker:
                        speaker_label = segment.speaker
                        if speaker_label.startswith("speaker_"):
                            try:
                                parts = speaker_label.split("_")
                                speaker_num = int(parts[-1]) + 1

                                if speaker_label != previous_speaker:
                                    if speaker_label not in seen_speakers:
                                        prefix = f"Speaker {speaker_num}: "
                                        seen_speakers.add(speaker_label)
                                    else:
                                        prefix = f"{speaker_num}: "
                                    segment.text = f"{prefix}{segment.text}"

                                previous_speaker = speaker_label
                            except (ValueError, IndexError):
                                if "Speaker" != previous_speaker:
                                    segment.text = f"Speaker: {segment.text}"
                                    previous_speaker = "Speaker"

                full_text = " ".join(segment.text for segment in all_segments)
                logger.info(f"Speaker diarization applied to {len(all_segments)} segments and included in text")
            else:
                logger.info("Speaker diarization applied to segments but not included in text")
        else:
            logger.warning("Diarization not applied or returned no speakers")

        response = TranscriptionResponse(
            text=full_text,
            segments=all_segments if timestamps or response_format == "verbose_json" else None,
            language=language,
            duration=sum(len(segment.text.split()) for segment in all_segments) / 150 if all_segments else 0,
            model="parakeet-tdt-0.6b-v2"
        )
        t_phase3_done = time.perf_counter()

        # --- Phase 4: LLM meeting analysis (optional) ---
        t_phase4 = time.perf_counter()
        if analyze and llm_client:
            try:
                if await llm_client.is_available():
                    intelligence = await llm_client.analyze_transcript(full_text)
                    response.meeting_intelligence = intelligence
                    logger.info(f"Phase 4: LLM analysis attached ({intelligence.generation_time_seconds}s)")
                else:
                    logger.warning("LLM requested but service unavailable — skipping analysis")
            except Exception as llm_err:
                logger.error(f"LLM analysis failed: {llm_err}")
        elif analyze and not llm_client:
            logger.warning("analyze=true but LLM client not initialized")
        t_phase4_done = time.perf_counter()

        # Log timing summary
        audio_dur = len(audio_chunks) * chunk_duration if audio_chunks else 0
        total_time = t_phase4_done - t_request
        rtf = total_time / audio_dur if audio_dur > 0 else 0
        phase4_str = f" phase4={t_phase4_done - t_phase4:.2f}s" if analyze else ""
        logger.info(
            f"timing: phase1={t_phase1_done - t_phase1:.2f}s"
            f"(wav={t_wav_done - t_wav:.2f}s chunks={t_phase1_done - t_wav_done:.2f}s) "
            f"phase2={t_phase2_done - t_phase2:.2f}s"
            f"(transcribe={t_transcribe_done - t_transcribe:.2f}s diarize_wait={t_diarize_elapsed:.2f}s) "
            f"phase3={t_phase3_done - t_phase3:.2f}s"
            f"{phase4_str} "
            f"total={total_time:.2f}s chunks={len(audio_chunks)} "
            f"audio~{audio_dur}s rtf={rtf:.4f}"
        )

        return response

    finally:
        # Cancel/await diarize_task before cleaning up files it may be reading
        if diarize_task and not diarize_task.done():
            try:
                diarize_task.cancel()
                await diarize_task
            except (asyncio.CancelledError, Exception):
                pass
        # Clean up WAV and chunk files (NOT the original audio_file_path — caller owns that)
        if wav_file and os.path.exists(wav_file):
            try:
                os.unlink(wav_file)
            except OSError:
                pass
        for chunk in audio_chunks:
            if chunk and chunk != str(wav_file) and os.path.exists(chunk):
                try:
                    os.unlink(chunk)
                except OSError:
                    pass
