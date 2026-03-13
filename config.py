# Configuration settings for Parakeet
import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

from pydantic.types import T

# Set up logging
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if it exists
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("dotenv package not installed. Environment variables will only be loaded from system.")

# API settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

# Model settings
DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CHUNK_DURATION = 30  # 30 seconds — NeMo's Lhotse dataloader drops samples > max_duration (40s)

# Hugging Face configuration — strip \r in case the env var was set with Windows line endings
HF_TOKEN = (os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN", "")).strip() or None

# Diarization settings
DEFAULT_DIARIZE = True
DEFAULT_NUM_SPEAKERS = None  # None means auto-detection
DEFAULT_INCLUDE_DIARIZATION_IN_TEXT = True  # Whether to include speaker labels in the text


class Config:
    """Global configuration for Parakeet"""

    # Singleton instance
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration with default values"""
        # API settings
        self.host = os.environ.get("HOST", DEFAULT_HOST)
        self.port = int(os.environ.get("PORT", DEFAULT_PORT))
        self.debug = DEBUG_MODE

        # Model settings
        self.model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        self.temperature = float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE))
        self.chunk_duration = int(os.environ.get("CHUNK_DURATION", DEFAULT_CHUNK_DURATION))

        # Diarization settings
        self.hf_token = HF_TOKEN
        self.enable_diarization = os.environ.get("ENABLE_DIARIZATION", str(DEFAULT_DIARIZE)).lower() == "true"
        self.include_diarization_in_text = os.environ.get("INCLUDE_DIARIZATION_IN_TEXT", str(DEFAULT_INCLUDE_DIARIZATION_IN_TEXT)).lower() == "true"
        self.default_num_speakers = DEFAULT_NUM_SPEAKERS


        # Concurrency / throughput settings
        self.batch_size = int(os.environ.get("BATCH_SIZE", 16))
        self.max_concurrent_requests = int(os.environ.get("MAX_CONCURRENT_REQUESTS", 1))
        self.max_concurrent_diarize = int(os.environ.get("MAX_CONCURRENT_DIARIZE", 1))

        # Diarization GPU batch sizes — pyannote defaults both to 1, severely underutilizing GPU
        self.diarize_segmentation_batch_size = int(os.environ.get("DIARIZE_SEGMENTATION_BATCH_SIZE", "8").strip())
        self.diarize_embedding_batch_size = int(os.environ.get("DIARIZE_EMBEDDING_BATCH_SIZE", "8").strip())

        # Segmentation step — ratio of window duration. Default 0.1 = 90% overlap between
        # consecutive windows. Higher = fewer windows = fewer embeddings = faster.
        # 0.1 = ~1400 windows for 23min audio. 0.5 = ~280 windows (5x fewer embeddings).
        # Trade-off: higher step = less precise speaker boundaries.
        self.diarize_segmentation_step = float(os.environ.get("DIARIZE_SEGMENTATION_STEP", "0.3").strip())

        # torch.compile() — opt-in encoder compilation for GPU speedup
        # Default mode is "default" (inductor backend). "reduce-overhead" requires nvcc which
        # may not be accessible in WSL. "max-autotune" is slowest to compile but fastest to run.
        self.torch_compile = os.environ.get("TORCH_COMPILE", "false").strip().lower() == "true"
        self.torch_compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default").strip()

        # Request timeout (seconds) — 0 means no timeout
        self.request_timeout = float(os.environ.get("REQUEST_TIMEOUT", "300").strip())
        if self.request_timeout <= 0:
            self.request_timeout = None  # asyncio.wait_for treats None as no timeout

        # Cross-request batch queue
        self.enable_batch_queue = os.environ.get("ENABLE_BATCH_QUEUE", "false").strip().lower() == "true"
        self.batch_queue_max_wait = float(os.environ.get("BATCH_QUEUE_MAX_WAIT", "0.5").strip())

        # File paths
        self.temp_dir = os.environ.get("TEMP_DIR", "/tmp/parakeet")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        logger.debug(f"Initialized configuration: debug={self.debug}, model={self.model_id}")

    def update_hf_token(self, token: str) -> None:
        """Update the HuggingFace token"""
        self.hf_token = token
        logger.info("Updated HuggingFace token")

    def get_hf_token(self) -> Optional[str]:
        """Get the HuggingFace token"""
        return self.hf_token

    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary (for API responses)"""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "chunk_duration": self.chunk_duration,
            "enable_diarization": self.enable_diarization,
            "include_diarization_in_text": self.include_diarization_in_text,
            "has_hf_token": self.hf_token is not None,
            "batch_size": self.batch_size,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_concurrent_diarize": self.max_concurrent_diarize,
            "diarize_segmentation_batch_size": self.diarize_segmentation_batch_size,
            "diarize_embedding_batch_size": self.diarize_embedding_batch_size,
            "diarize_segmentation_step": self.diarize_segmentation_step,
            "torch_compile": self.torch_compile,
            "torch_compile_mode": self.torch_compile_mode,
            "request_timeout": self.request_timeout,
            "enable_batch_queue": self.enable_batch_queue,
        }


# Create a global instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config
