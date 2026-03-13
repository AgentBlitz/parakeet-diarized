import asyncio
import os
import logging
import time
from functools import partial
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import torch

from models import WhisperSegment, TranscriptionResponse, ModelInfo, ModelList
from audio import convert_audio_to_wav, split_audio_into_chunks_async
from transcription import load_model, format_srt, format_vtt, transcribe_audio_batch
from diarization import Diarizer
from batching import BatchingEngine
from config import get_config

# Initialize logging
logger = logging.getLogger(__name__)

# Global model, diarizer singleton, and per-operation GPU semaphores
# transcribe_semaphore: serializes model.transcribe() calls (NeMo not thread-safe)
# diarize_semaphore: serializes pyannote diarization (runs concurrently with transcription)
asr_model = None
diarizer_instance: Optional[Diarizer] = None
batching_engine: Optional[BatchingEngine] = None
transcribe_semaphore: asyncio.Semaphore = None
diarize_semaphore: asyncio.Semaphore = None

# Get configuration
config = get_config()

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(title="Parakeet Whisper-Compatible API")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        """Initialize resources during startup"""
        global asr_model, diarizer_instance, batching_engine, transcribe_semaphore, diarize_semaphore

        transcribe_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        diarize_semaphore = asyncio.Semaphore(config.max_concurrent_diarize)
        logger.info(
            f"Semaphores initialized — transcribe={config.max_concurrent_requests}, "
            f"diarize={config.max_concurrent_diarize}"
        )

        try:
            # Check CUDA availability
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, using CPU (this will be slow)")

            # Load the ASR model
            model_id = config.model_id
            asr_model = load_model(model_id)
            logger.info(f"Model {model_id} loaded successfully")

            # Initialize diarizer singleton if token is available
            hf_token = config.get_hf_token()
            if hf_token:
                logger.info("Initializing diarizer singleton (pyannote pipeline)...")
                diarizer_instance = Diarizer(access_token=hf_token)
                logger.info("Diarizer singleton initialized — pipeline loaded once, reused across requests")
            else:
                logger.info("No HuggingFace access token, speaker diarization will be disabled")

            # Initialize cross-request batching engine if enabled
            if config.enable_batch_queue and asr_model:
                batching_engine = BatchingEngine(
                    model=asr_model,
                    batch_size=config.batch_size,
                    max_wait=config.batch_queue_max_wait,
                )
                await batching_engine.start()
                logger.info("Cross-request batch queue enabled")

        except Exception as e:
            logger.error(f"Error during startup: {str(e)}")
            # We don't want to fail startup completely, as the health endpoint should still work

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources during shutdown"""
        global batching_engine
        logger.info("Shutting down — cleaning up resources")
        # Stop batching engine
        if batching_engine:
            await batching_engine.stop()
            batching_engine = None
        # Clean up temp directory
        temp_dir = Path(config.temp_dir)
        if temp_dir.exists():
            for f in temp_dir.iterdir():
                try:
                    f.unlink()
                except Exception:
                    pass
        # Release GPU memory
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            logger.info(f"GPU memory at shutdown: {gpu_mb:.1f} MB")
            torch.cuda.empty_cache()
        logger.info("Shutdown complete")

    @app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        timestamps: bool = Form(False),
        timestamp_granularities: Optional[List[str]] = Form(None),
        vad_filter: bool = Form(False),
        word_timestamps: bool = Form(False),
        diarize: bool = Form(True),
        include_diarization_in_text: Optional[bool] = Form(None)
    ):
        """
        Transcribe audio file using the Parakeet-TDT model

        This endpoint is compatible with the OpenAI Whisper API
        """

        global asr_model, diarizer_instance, transcribe_semaphore, diarize_semaphore

        if not asr_model:
            raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a few moments.")

        # Process parameters
        logger.info(f"Transcription requested: {file.filename}, format: {response_format}")

        t_request = time.perf_counter()
        temp_file = None
        wav_file = None
        audio_chunks = []
        diarize_task = None

        try:
            # --- Phase 1: File I/O (no GPU semaphore — runs concurrently with other requests) ---
            t_phase1 = time.perf_counter()

            # Save uploaded file to temp location
            temp_dir = Path(config.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            temp_file = temp_dir / f"upload_{os.urandom(8).hex()}{Path(file.filename).suffix}"
            with open(temp_file, "wb") as f:
                content = await file.read()
                f.write(content)

            chunk_duration = config.chunk_duration
            loop = asyncio.get_event_loop()

            # Convert to WAV in a thread (blocking subprocess)
            t_wav = time.perf_counter()
            wav_file = await loop.run_in_executor(None, convert_audio_to_wav, str(temp_file))
            t_wav_done = time.perf_counter()

            # Split into chunks in parallel (async subprocesses)
            audio_chunks = await split_audio_into_chunks_async(wav_file, chunk_duration=chunk_duration)
            t_phase1_done = time.perf_counter()

            # Use diarizer singleton (initialized once at startup)
            diarizer = diarizer_instance if diarize else None
            if diarize and not diarizer:
                logger.warning("Diarization requested but diarizer not initialized (no HuggingFace token)")

            # --- Phase 2: GPU work (diarization and transcription run concurrently) ---
            # diarize_semaphore and transcribe_semaphore are independent — both can run at
            # the same time on the RTX 4090 (pyannote ~2-4GB, NeMo fp16 ~1.5GB + batch).
            t_phase2 = time.perf_counter()
            diarization_result = None
            batch_results = []

            async def _run_diarize():
                async with diarize_semaphore:
                    logger.info("Performing speaker diarization")
                    result = await loop.run_in_executor(
                        None, partial(diarizer.diarize, wav_file)
                    )
                    logger.info(f"Diarization found {result.num_speakers} speakers")
                    return result

            # Start diarization as a background task (if requested)
            diarize_task = asyncio.create_task(_run_diarize()) if diarizer else None

            # Run transcription — either via batch queue (cross-request) or direct semaphore
            logger.info(f"Batch transcribing {len(audio_chunks)} chunk(s) with batch_size={config.batch_size}")
            t_transcribe = time.perf_counter()

            if batching_engine:
                # Cross-request batching: chunks are queued and merged with other requests
                request_id = os.urandom(4).hex()
                try:
                    batch_results = await asyncio.wait_for(
                        batching_engine.submit_chunks(
                            audio_chunks, request_id,
                            batch_size=config.batch_size,
                            language=language,
                            word_timestamps=word_timestamps,
                        ),
                        timeout=config.request_timeout
                    )
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=504, detail=f"Transcription timed out after {config.request_timeout}s")
            else:
                # Direct path: semaphore-guarded single-request transcription
                async with transcribe_semaphore:
                    try:
                        batch_results = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                partial(
                                    transcribe_audio_batch,
                                    asr_model,
                                    audio_chunks,
                                    config.batch_size,
                                    language,
                                    word_timestamps
                                )
                            ),
                            timeout=config.request_timeout
                        )
                    except asyncio.TimeoutError:
                        raise HTTPException(status_code=504, detail=f"Transcription timed out after {config.request_timeout}s")
            t_transcribe_done = time.perf_counter()

            # Wait for diarization to finish (usually already done by now)
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

            # Combine results
            full_text = " ".join(all_text)

            # Apply diarization if available
            if diarizer and diarization_result and diarization_result.segments:
                logger.info(f"Found {diarization_result.num_speakers} speakers")
                all_segments = diarizer.merge_with_transcription(diarization_result, all_segments)

                # Determine whether to include diarization in text
                # Use the request parameter if provided, otherwise use the config setting
                use_diarization_in_text = include_diarization_in_text if include_diarization_in_text is not None else config.include_diarization_in_text

                if use_diarization_in_text:
                    logger.info("Including speaker labels in transcript text")
                    # Keep track of the previous speaker
                    previous_speaker = None
                    # Track which speakers we've seen before
                    seen_speakers = set()

                    # Process segments to include speaker info in the text field
                    for segment in all_segments:
                        if hasattr(segment, 'speaker') and segment.speaker:
                            # Extract speaker number (e.g., 'speaker_SPEAKER_00' -> '1')
                            speaker_label = segment.speaker
                            if speaker_label.startswith("speaker_"):
                                try:
                                    # Extract speaker number from the label
                                    parts = speaker_label.split("_")
                                    speaker_num = int(parts[-1]) + 1  # Add 1 to make it 1-indexed

                                    # Only add speaker prefix if this is a different speaker than the previous one
                                    if speaker_label != previous_speaker:
                                        # First time seeing this speaker
                                        if speaker_label not in seen_speakers:
                                            prefix = f"Speaker {speaker_num}: "
                                            seen_speakers.add(speaker_label)
                                        else:
                                            # We've seen this speaker before
                                            prefix = f"{speaker_num}: "

                                        segment.text = f"{prefix}{segment.text}"

                                    previous_speaker = speaker_label

                                except (ValueError, IndexError):
                                    # If parsing fails, use a generic label
                                    if "Speaker" != previous_speaker:
                                        segment.text = f"Speaker: {segment.text}"
                                        previous_speaker = "Speaker"

                    # Reconstruct full text with speaker labels
                    full_text = " ".join(segment.text for segment in all_segments)
                    logger.info(f"Speaker diarization applied to {len(all_segments)} segments and included in text")
                else:
                    logger.info("Speaker diarization applied to segments but not included in text")
            else:
                logger.warning("Diarization not applied or returned no speakers")


            # Create response
            response = TranscriptionResponse(
                text=full_text,
                segments=all_segments if timestamps or response_format == "verbose_json" else None,
                language=language,
                duration=sum(len(segment.text.split()) for segment in all_segments) / 150 if all_segments else 0,
                model="parakeet-tdt-0.6b-v2"
            )
            t_phase3_done = time.perf_counter()

            # Log timing summary
            audio_dur = len(audio_chunks) * chunk_duration if audio_chunks else 0
            total_time = t_phase3_done - t_request
            rtf = total_time / audio_dur if audio_dur > 0 else 0
            logger.info(
                f"timing: phase1={t_phase1_done - t_phase1:.2f}s"
                f"(wav={t_wav_done - t_wav:.2f}s chunks={t_phase1_done - t_wav_done:.2f}s) "
                f"phase2={t_phase2_done - t_phase2:.2f}s"
                f"(transcribe={t_transcribe_done - t_transcribe:.2f}s diarize_wait={t_diarize_elapsed:.2f}s) "
                f"phase3={t_phase3_done - t_phase3:.2f}s "
                f"total={total_time:.2f}s chunks={len(audio_chunks)} "
                f"audio~{audio_dur}s rtf={rtf:.4f}"
            )

            # Return in requested format
            if response_format == "json":
                return response.dict()
            elif response_format == "text":
                return PlainTextResponse(full_text)
            elif response_format == "srt":
                return PlainTextResponse(format_srt(all_segments))
            elif response_format == "vtt":
                return PlainTextResponse(format_vtt(all_segments))
            elif response_format == "verbose_json":
                return response.dict()
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Wait for diarization to finish before deleting files it may be reading
            if diarize_task and not diarize_task.done():
                try:
                    diarize_task.cancel()
                    await diarize_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Clean up temporary files (runs even on error)
            for path in [temp_file, wav_file]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            for chunk in audio_chunks:
                if chunk and chunk != str(wav_file) and os.path.exists(chunk):
                    try:
                        os.unlink(chunk)
                    except OSError:
                        pass

    @app.get("/health")
    async def health_check():
        """
        Check the health of the API and the loaded model
        """
        global asr_model, diarizer_instance

        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats = {
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
                "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
                "gpu_max_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            }

        return {
            "status": "ok",
            "version": "1.0.0",
            "model_loaded": asr_model is not None,
            "diarizer_loaded": diarizer_instance is not None,
            "model_id": config.model_id,
            "cuda_available": torch.cuda.is_available(),
            "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_compile_enabled": config.torch_compile,
            **gpu_stats,
            "config": config.as_dict()
        }

    @app.get("/v1/models")
    async def list_models():
        """
        List available models (compatibility with OpenAI API)
        """
        models = [
            ModelInfo(
                id="whisper-1",
                created=1677649963,
                owned_by="parakeet",
                root="whisper-1",
                permission=[{"id": "modelperm-1", "object": "model_permission", "created": 1677649963,
                           "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": True,
                           "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False,
                           "organization": "*", "group": None, "is_blocking": False}]
            )
        ]

        return ModelList(data=models)

    return app
