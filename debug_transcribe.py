#!/usr/bin/env python3
"""
Standalone debug script to test NeMo transcription outside the API pipeline.
Compares use_lhotse=True vs use_lhotse=False to isolate Lhotse dataloader issues.

Usage (in WSL):
    source venv/bin/activate
    python debug_transcribe.py experiments/test.wav
    python debug_transcribe.py experiments/test.mp3
"""
import sys
import time
import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import open_dict

    model_id = "nvidia/parakeet-tdt-0.6b-v2"
    logger.info(f"Loading {model_id}...")
    model = EncDecRNNTBPEModel.from_pretrained(model_id)

    if torch.cuda.is_available():
        model = model.cuda()
        model = model.half()
        logger.info(f"Model on GPU (fp16): {torch.cuda.get_device_name(0)}")

    # Same decoder config as api.py
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        decoding_cfg.compute_timestamps = True
    model.change_decoding_strategy(decoding_cfg)

    return model


def transcribe_test(model, audio_path, use_lhotse, label):
    logger.info(f"\n{'='*60}")
    logger.info(f"Test: {label} | use_lhotse={use_lhotse} | file={audio_path}")
    logger.info(f"{'='*60}")

    t0 = time.perf_counter()
    with torch.no_grad():
        result = model.transcribe(
            [audio_path],
            batch_size=1,
            return_hypotheses=True,
            use_lhotse=use_lhotse,
        )
    elapsed = time.perf_counter() - t0

    # Unwrap NeMo 2.x tuple
    if (isinstance(result, (list, tuple))
            and len(result) == 2
            and isinstance(result[0], list)
            and isinstance(result[1], list)):
        texts = result[0]
        hyps = result[1]
    else:
        texts = []
        hyps = result if isinstance(result, list) else [result]

    for i, hyp in enumerate(hyps):
        text = hyp.text if hasattr(hyp, 'text') else str(hyp)
        score = getattr(hyp, 'score', None)
        y_seq = getattr(hyp, 'y_sequence', None)
        y_shape = y_seq.shape if hasattr(y_seq, 'shape') else None

        logger.info(f"  Result[{i}]: {len(text)} chars, score={score}, y_seq_shape={y_shape}")
        logger.info(f"  Time: {elapsed:.2f}s")
        logger.info(f"  Text: {repr(text[:200])}")
        if texts:
            logger.info(f"  Plain text[{i}]: {repr(texts[i][:200])}")

    return hyps


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_transcribe.py <audio_file> [audio_file2 ...]")
        print("Example: python debug_transcribe.py experiments/test.wav experiments/test.mp3")
        sys.exit(1)

    audio_files = sys.argv[1:]
    model = load_model()

    for audio_path in audio_files:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Testing: {audio_path}")
        logger.info(f"{'#'*60}")

        # Test 1: use_lhotse=False (legacy NeMo dataloader)
        transcribe_test(model, audio_path, use_lhotse=False, label="LEGACY (no Lhotse)")

        # Test 2: use_lhotse=True (Lhotse dataloader — suspected broken)
        transcribe_test(model, audio_path, use_lhotse=True, label="LHOTSE")

    if torch.cuda.is_available():
        logger.info(f"\nGPU peak memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
