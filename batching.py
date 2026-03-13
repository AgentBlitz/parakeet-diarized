"""
Cross-request chunk batching engine.

When ENABLE_BATCH_QUEUE=true, chunks from multiple concurrent requests are
merged into a single model.transcribe() call for maximum GPU utilization.

Design:
- Callers submit chunks via submit_chunks() and get back Futures
- A background worker collects chunks and flushes them to the GPU when:
  (a) batch_size chunks have accumulated, OR
  (b) max_wait seconds have elapsed since the first chunk in the batch
- Results are dispatched back to callers via their Futures

Trade-off: single-request latency increases by up to max_wait (default 0.5s),
but throughput for concurrent requests is much higher.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

from models import WhisperSegment

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """One chunk submitted to the batching queue."""
    chunk_path: str
    future: asyncio.Future
    request_id: str
    submitted_at: float = field(default_factory=time.monotonic)


class BatchingEngine:
    """Collects transcription chunks across requests and flushes them in batches."""

    def __init__(self, model, batch_size: int, max_wait: float = 0.5):
        """
        Args:
            model: Loaded NeMo ASR model
            batch_size: Max chunks per GPU call
            max_wait: Max seconds to wait before flushing an incomplete batch
        """
        self._model = model
        self._batch_size = batch_size
        self._max_wait = max_wait
        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._batches_flushed = 0
        self._total_chunks_processed = 0

    async def start(self):
        """Start the background flush worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._flush_loop())
        logger.info(
            f"BatchingEngine started — batch_size={self._batch_size}, "
            f"max_wait={self._max_wait}s"
        )

    async def stop(self):
        """Stop the worker and flush remaining items."""
        self._running = False
        if self._worker_task:
            # Put a sentinel to unblock the queue.get()
            await self._queue.put(None)
            await self._worker_task
            self._worker_task = None
        logger.info(
            f"BatchingEngine stopped — {self._batches_flushed} batches flushed, "
            f"{self._total_chunks_processed} chunks processed"
        )

    async def submit_chunks(
        self,
        chunk_paths: List[str],
        request_id: str,
        batch_size: int = 0,
        language: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> List[Tuple[str, List[WhisperSegment]]]:
        """
        Submit chunks for transcription and wait for results.

        Args:
            chunk_paths: Ordered list of audio chunk file paths
            request_id: Identifier for logging
            batch_size: Unused (batch_size is controlled by the engine)
            language: Unused (kept for API compatibility)
            word_timestamps: Unused (kept for API compatibility)

        Returns:
            List of (text, segments) tuples in same order as chunk_paths
        """
        loop = asyncio.get_event_loop()
        items = []
        for path in chunk_paths:
            future = loop.create_future()
            item = BatchItem(chunk_path=path, future=future, request_id=request_id)
            items.append(item)
            await self._queue.put(item)

        logger.debug(
            f"[{request_id}] Submitted {len(items)} chunks to batch queue "
            f"(queue size ~{self._queue.qsize()})"
        )

        # Wait for all futures to resolve
        results = await asyncio.gather(*(item.future for item in items))
        return list(results)

    async def _flush_loop(self):
        """Background loop: collect items and flush to GPU."""
        while self._running or not self._queue.empty():
            batch: List[BatchItem] = []
            deadline = None

            # Collect up to batch_size items
            while len(batch) < self._batch_size:
                try:
                    timeout = None
                    if deadline is not None:
                        timeout = max(0.01, deadline - time.monotonic())
                    elif batch:
                        # We have items but no deadline yet — shouldn't happen
                        timeout = self._max_wait

                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=timeout
                    )

                    # Sentinel check (shutdown signal)
                    if item is None:
                        if not self._running:
                            break
                        continue

                    batch.append(item)

                    # Set deadline on first item
                    if len(batch) == 1:
                        deadline = time.monotonic() + self._max_wait

                except asyncio.TimeoutError:
                    # max_wait elapsed — flush what we have
                    break

            if not batch:
                if not self._running:
                    break
                await asyncio.sleep(0.05)
                continue

            # Process the batch on GPU
            await self._flush_batch(batch)

    async def _flush_batch(self, batch: List[BatchItem]):
        """Send a batch of chunks to model.transcribe() and resolve futures."""
        from transcription import transcribe_audio_batch

        paths = [item.chunk_path for item in batch]
        request_ids = set(item.request_id for item in batch)
        logger.info(
            f"Flushing batch: {len(paths)} chunks from {len(request_ids)} request(s)"
        )

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                partial(
                    transcribe_audio_batch,
                    self._model,
                    paths,
                    self._batch_size,
                )
            )
            # Dispatch results to futures
            for item, result in zip(batch, results):
                if not item.future.done():
                    item.future.set_result(result)

            self._batches_flushed += 1
            self._total_chunks_processed += len(batch)

        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
