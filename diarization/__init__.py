# Speaker diarization module for Parakeet
# This module integrates pyannote.audio for speaker identification

from typing import Dict, List, Optional, Tuple, Union
import os
import logging
import tempfile
import time
import numpy as np
import torch
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SpeakerSegment(BaseModel):
    """A segment of speech from a specific speaker"""
    start: float
    end: float
    speaker: str

class DiarizationResult(BaseModel):
    """Result of speaker diarization"""
    segments: List[SpeakerSegment]
    num_speakers: int

class Diarizer:
    """Speaker diarization using pyannote.audio"""

    def __init__(self, access_token: Optional[str] = None):
        self.pipeline = None
        self.access_token = access_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize()

    def _initialize(self):
        """Initialize the diarization pipeline"""
        try:
            from pyannote.audio import Pipeline

            if not self.access_token:
                logger.warning("No access token provided. Using HUGGINGFACE_ACCESS_TOKEN environment variable.")
                self.access_token = (
                    os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN", "")
                ).strip() or None

            if not self.access_token:
                logger.error("No access token available. Diarization will not work.")
                return

            # Initialize the pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.access_token
            )

            # Move to GPU if available
            self.pipeline.to(torch.device(self.device))
            logger.info(f"Diarization pipeline initialized on {self.device}")

            # Apply GPU batch sizes — pyannote defaults both to 1, which severely
            # underutilizes the GPU. These are property setters on SpeakerDiarization.
            from config import get_config
            cfg = get_config()
            self.pipeline.segmentation_batch_size = cfg.diarize_segmentation_batch_size
            self.pipeline.embedding_batch_size = cfg.diarize_embedding_batch_size

            # Override segmentation step if configured (default 0.1 = 90% overlap).
            # Higher step = fewer windows = fewer embeddings = faster diarization.
            # Must update both pipeline attribute AND the Inference object's step.
            if cfg.diarize_segmentation_step != 0.1:
                seg_duration = self.pipeline._segmentation.duration
                self.pipeline.segmentation_step = cfg.diarize_segmentation_step
                self.pipeline._segmentation.step = cfg.diarize_segmentation_step * seg_duration
                logger.info(f"Segmentation step overridden: {cfg.diarize_segmentation_step} "
                            f"(window={seg_duration:.1f}s, step={self.pipeline._segmentation.step:.1f}s)")

            logger.info(
                f"Diarization config: seg_batch={cfg.diarize_segmentation_batch_size}, "
                f"emb_batch={cfg.diarize_embedding_batch_size}, "
                f"seg_step={cfg.diarize_segmentation_step}"
            )

        except ImportError:
            logger.error("Failed to import pyannote.audio. Please install it with 'pip install pyannote.audio'")
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {str(e)}")

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file

        Args:
            audio_path: Path to the audio file
            num_speakers: Optional number of speakers (if known)

        Returns:
            DiarizationResult with speaker segments
        """
        if self.pipeline is None:
            logger.error("Diarization pipeline not initialized")
            return DiarizationResult(segments=[], num_speakers=0)

        try:
            # Timing hook — logs elapsed time for each diarization stage.
            # pyannote calls the hook multiple times per stage (progress callbacks),
            # so we only record time when the stage NAME changes.
            stage_times = {}
            current_stage = {"name": None, "start": None}

            def _timing_hook(step_name, step_artefact, **kwargs):
                now = time.perf_counter()
                if current_stage["name"] != step_name:
                    # Stage changed — record elapsed time for previous stage
                    if current_stage["name"] is not None:
                        stage_times[current_stage["name"]] = now - current_stage["start"]
                    current_stage["name"] = step_name
                    current_stage["start"] = now
                completed = kwargs.get("completed")
                total = kwargs.get("total")
                if completed is not None and total is not None:
                    if completed == 1 or completed == total or completed % 50 == 0:
                        logger.debug(f"Diarization [{step_name}]: {completed}/{total}")

            # Log GPU memory before diarization
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024 / 1024
                logger.info(f"GPU memory before diarize: {mem_before:.1f} MB")

            # Run the diarization pipeline with timing hook
            t_start = time.perf_counter()
            diarization = self.pipeline(
                audio_path,
                num_speakers=num_speakers,
                hook=_timing_hook
            )
            t_total = time.perf_counter() - t_start

            # Close out the last stage
            if current_stage["name"] is not None:
                stage_times[current_stage["name"]] = t_total - sum(stage_times.values())

            # Log per-stage breakdown
            stage_str = " ".join(f"{k}={v:.2f}s" for k, v in stage_times.items())
            logger.info(f"Diarization completed in {t_total:.2f}s — {stage_str}")

            # Log GPU memory after diarization
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024 / 1024
                mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                logger.info(f"GPU memory after diarize: {mem_after:.1f} MB (peak {mem_peak:.1f} MB)")

            # Convert to our format
            segments = []
            speakers = set()

            # Handle different return types across pyannote.audio versions:
            # - 3.1: returns Annotation directly
            # - 3.3+: returns SpeakerDiarizationOutput(diarization=Annotation, embeddings=...)
            # - Some builds: returns a namedtuple where the Annotation is the first field
            annotation = diarization
            if not hasattr(diarization, 'itertracks'):
                if hasattr(diarization, 'diarization'):
                    annotation = diarization.diarization
                elif hasattr(diarization, 'speaker_diarization'):
                    # pyannote.audio 3.3+ DiarizeOutput namedtuple
                    annotation = diarization.speaker_diarization
                elif isinstance(diarization, tuple) and len(diarization) > 0:
                    annotation = diarization[0]
                else:
                    logger.error(
                        f"Unsupported diarization result type: {type(diarization).__name__}, "
                        f"attrs: {[a for a in dir(diarization) if not a.startswith('_')]}"
                    )
                    raise AttributeError(
                        f"Diarization result type '{type(diarization).__name__}' has no itertracks method"
                    )

            # Process the diarization result
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                # Convert speaker label to consistent format
                # This handles different formats from pyannote.audio versions
                if isinstance(speaker, str) and not speaker.startswith("SPEAKER_"):
                    speaker_id = f"SPEAKER_{speaker}"
                else:
                    speaker_id = speaker

                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=f"speaker_{speaker_id}"
                ))
                speakers.add(speaker_id)

            # Sort segments by start time
            segments.sort(key=lambda x: x.start)

            return DiarizationResult(
                segments=segments,
                num_speakers=len(speakers)
            )

        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            return DiarizationResult(segments=[], num_speakers=0)

    def merge_with_transcription(self,
                                diarization: DiarizationResult,
                                transcription_segments: list) -> list:
        """
        Merge diarization results with transcription segments

        Args:
            diarization: Speaker diarization result
            transcription_segments: List of transcription segments with start/end times

        Returns:
            Merged list of segments with speaker information
        """
        # If no diarization results, return original transcription
        if not diarization.segments:
            return transcription_segments

        # For each transcription segment, find the dominant speaker
        for segment in transcription_segments:
            # Get segment time bounds
            start = segment.start
            end = segment.end

            # Find overlapping diarization segments
            overlapping = []
            for spk_segment in diarization.segments:
                # Calculate overlap
                overlap_start = max(start, spk_segment.start)
                overlap_end = min(end, spk_segment.end)

                if overlap_end > overlap_start:
                    # There is an overlap
                    duration = overlap_end - overlap_start
                    overlapping.append((spk_segment.speaker, duration))

            # Assign the speaker with most overlap
            if overlapping:
                # Sort by duration (descending)
                overlapping.sort(key=lambda x: x[1], reverse=True)
                # Assign the dominant speaker
                setattr(segment, "speaker", overlapping[0][0])
            else:
                # No overlap found, assign unknown
                setattr(segment, "speaker", "unknown")

        return transcription_segments
