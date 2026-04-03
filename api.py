import asyncio
import os
import logging
import time
from functools import partial
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import numpy as np
import torch

from models import (
    WhisperSegment, TranscriptionResponse, ModelInfo, ModelList,
    MeetingIntelligence, AnalyzeRequest,
)
from transcription import load_model, format_srt, format_vtt, transcribe_audio_numpy
from pipeline import run_transcription_pipeline
from diarization import Diarizer
from llm import LLMClient
from config import get_config

# Initialize logging
logger = logging.getLogger(__name__)

# Global model, diarizer singleton, and per-operation GPU semaphores
# transcribe_semaphore: serializes model.transcribe() calls (NeMo not thread-safe)
# diarize_semaphore: serializes pyannote diarization (runs concurrently with transcription)
asr_model = None
diarizer_instance: Optional[Diarizer] = None
llm_client: Optional[LLMClient] = None
transcribe_semaphore: asyncio.Semaphore = None
diarize_semaphore: asyncio.Semaphore = None
job_db = None
job_worker = None

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
        global asr_model, diarizer_instance, llm_client, transcribe_semaphore, diarize_semaphore, job_db, job_worker

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

            # Initialize LLM client (Ollama sidecar) — non-fatal if unavailable
            if config.llm_enabled:
                try:
                    llm_client = LLMClient()
                    if await llm_client.is_available():
                        await llm_client.ensure_model_pulled()
                        logger.info(f"LLM client initialized — model={config.llm_model}")
                    else:
                        logger.warning("LLM service not reachable — meeting analysis will be unavailable")
                except Exception as llm_err:
                    logger.warning(f"LLM initialization failed ({llm_err}) — meeting analysis will be unavailable")

            # Initialize async job queue if enabled
            if config.enable_job_queue:
                try:
                    from jobs import JobDB, JobWorker, jobs_router
                    from jobs.router import set_dependencies

                    job_db = JobDB(config.job_db_path)
                    await job_db.initialize()

                    job_worker = JobWorker(
                        db=job_db,
                        transcribe_semaphore=transcribe_semaphore,
                        diarize_semaphore=diarize_semaphore,
                        model=asr_model,
                        diarizer=diarizer_instance,
                        llm_client=llm_client,
                    )
                    set_dependencies(job_db, job_worker)
                    await job_worker.start()
                    logger.info("Async job queue enabled")
                except Exception as job_err:
                    logger.error(f"Job queue initialization failed: {job_err}")

        except Exception as e:
            logger.error(f"Error during startup: {str(e)}")
            # We don't want to fail startup completely, as the health endpoint should still work

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources during shutdown"""
        global llm_client, job_db, job_worker
        logger.info("Shutting down — cleaning up resources")
        # Stop job worker
        if job_worker:
            await job_worker.stop()
            job_worker = None
        # Close job database
        if job_db:
            await job_db.close()
            job_db = None
        # Close LLM client
        if llm_client:
            await llm_client.close()
            llm_client = None
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
        include_diarization_in_text: Optional[bool] = Form(None),
        analyze: bool = Form(False),
    ):
        """
        Transcribe audio file using the Parakeet-TDT model

        This endpoint is compatible with the OpenAI Whisper API
        """

        global asr_model, diarizer_instance, llm_client, transcribe_semaphore, diarize_semaphore

        if not asr_model:
            raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again in a few moments.")

        logger.info(f"Transcription requested: {file.filename}, format: {response_format}")

        # Save uploaded file to temp location
        temp_dir = Path(config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"upload_{os.urandom(8).hex()}{Path(file.filename).suffix}"

        try:
            with open(temp_file, "wb") as f:
                content = await file.read()
                f.write(content)

            # Run the shared pipeline (handles phases 1-4, cleans up WAV/chunks)
            response = await run_transcription_pipeline(
                audio_file_path=str(temp_file),
                model=asr_model,
                diarizer=diarizer_instance,
                llm_client=llm_client,
                transcribe_semaphore=transcribe_semaphore,
                diarize_semaphore=diarize_semaphore,
                language=language,
                diarize=diarize,
                analyze=analyze,
                include_diarization_in_text=include_diarization_in_text,
                response_format=response_format,
                word_timestamps=word_timestamps,
                timestamps=timestamps,
            )

            # Return in requested format
            if response_format == "json":
                return response.dict()
            elif response_format == "text":
                return PlainTextResponse(response.text)
            elif response_format == "srt":
                return PlainTextResponse(format_srt(response.segments or []))
            elif response_format == "vtt":
                return PlainTextResponse(format_vtt(response.segments or []))
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
            # Clean up the uploaded temp file (pipeline cleans up WAV/chunks)
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    # Queue depth counters for /transcribe backpressure
    # Safe to use plain int — all increments/decrements happen in async coroutines
    # on the same event loop thread (never from run_in_executor threads).
    _rt_queue_depth = 0
    _rt_total_served = 0
    _rt_total_rejected = 0

    @app.post("/transcribe")
    async def transcribe_raw(request: Request):
        """
        Lightweight transcription endpoint for real-time audio streams.

        Accepts raw Float32Array binary (16kHz mono) as application/octet-stream
        and returns {"text": "..."}.  Designed for browser extensions that capture
        audio at 16kHz mono and send short chunks (2-30s) in real time.

        No diarization, no ffmpeg, no temp files — numpy straight to GPU.
        """
        nonlocal _rt_queue_depth, _rt_total_served, _rt_total_rejected
        global asr_model, transcribe_semaphore

        if not asr_model:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")

        raw = await request.body()
        # 3200 bytes = 800 float32 samples = 0.05s at 16kHz — too short to transcribe
        if len(raw) < 3200:
            return JSONResponse({"text": ""})

        audio = np.frombuffer(raw, dtype=np.float32)
        duration = len(audio) / 16000
        if duration > 45:
            raise HTTPException(status_code=400, detail=f"Audio too long ({duration:.1f}s). Max ~40s per chunk.")

        req_id = os.urandom(4).hex()

        # Backpressure: reject when queue is full
        if _rt_queue_depth >= config.max_transcribe_queue:
            _rt_total_rejected += 1
            logger.warning(
                f"/transcribe [{req_id}]: REJECTED — queue full "
                f"({_rt_queue_depth}/{config.max_transcribe_queue}), "
                f"total_rejected={_rt_total_rejected}"
            )
            raise HTTPException(
                status_code=503,
                detail=f"Server busy — {_rt_queue_depth} requests queued. Retry shortly."
            )

        _rt_queue_depth += 1
        t_start = time.perf_counter()
        loop = asyncio.get_event_loop()

        logger.info(
            f"/transcribe [{req_id}]: enqueued — "
            f"queue_depth={_rt_queue_depth}/{config.max_transcribe_queue}, "
            f"audio={duration:.1f}s"
        )

        try:
            async with transcribe_semaphore:
                t_wait = time.perf_counter() - t_start
                logger.info(f"/transcribe [{req_id}]: acquired GPU — waited {t_wait:.3f}s")
                try:
                    text, segments = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            partial(transcribe_audio_numpy, asr_model, audio)
                        ),
                        timeout=config.transcribe_timeout
                    )
                except asyncio.TimeoutError:
                    raise HTTPException(status_code=504, detail="Transcription timed out.")

            t_elapsed = time.perf_counter() - t_start
            logger.info(
                f"/transcribe [{req_id}]: {duration:.1f}s audio → {len(text)} chars "
                f"in {t_elapsed:.2f}s (wait={t_wait:.3f}s, gpu={t_elapsed - t_wait:.3f}s) "
                f"queue_after={_rt_queue_depth - 1}"
            )

            return JSONResponse({"text": text})
        finally:
            _rt_queue_depth -= 1
            _rt_total_served += 1

    @app.post("/v1/meeting/analyze")
    async def analyze_meeting(request: AnalyzeRequest):
        """
        Analyze a diarized transcript and extract meeting intelligence.

        Accepts a transcript string (with speaker labels) and returns structured
        meeting intelligence: summary, action items, decisions, questions, etc.
        """
        if not llm_client:
            raise HTTPException(
                status_code=503,
                detail="LLM service not configured. Set LLM_ENABLED=true and ensure Ollama is running."
            )

        if not await llm_client.is_available():
            raise HTTPException(
                status_code=503,
                detail="LLM service is not reachable. Ensure Ollama is running."
            )

        if not request.transcript.strip():
            raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

        try:
            intelligence = await llm_client.analyze_transcript(
                request.transcript, model=request.model
            )
            return intelligence.dict()
        except Exception as e:
            logger.error(f"Meeting analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM analysis failed: {e}")

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
            "llm_available": llm_client is not None,
            "llm_model": config.llm_model if llm_client else None,
            "model_id": config.model_id,
            "cuda_available": torch.cuda.is_available(),
            "gpu_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_compile_enabled": config.torch_compile,
            **gpu_stats,
            "transcribe_queue": {
                "depth": _rt_queue_depth,
                "max": config.max_transcribe_queue,
                "total_served": _rt_total_served,
                "total_rejected": _rt_total_rejected,
            },
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

    # Include job queue router (endpoints are gated by 503 if not enabled)
    if config.enable_job_queue:
        from jobs import jobs_router
        app.include_router(jobs_router)
        logger.info("Job queue API endpoints registered at /v1/jobs")

    return app
