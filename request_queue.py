"""Request Queue Manager - Handles queuing when resources are unavailable."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class QueueItemState(str, Enum):
    """State of a queued request."""
    WAITING = "waiting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class QueueItem:
    """A queued request."""
    id: str
    model_name: str
    gpu_count: int
    priority: str
    created_at: datetime
    state: QueueItemState = QueueItemState.WAITING
    event: asyncio.Event = field(default_factory=asyncio.Event)
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "gpu_count": self.gpu_count,
            "priority": self.priority,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "wait_time_seconds": (datetime.now() - self.created_at).total_seconds(),
        }


class RequestQueue:
    """Manages queued requests when resources are unavailable."""

    PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}

    def __init__(self, max_size: int, timeout: int):
        """Initialize the request queue.

        Args:
            max_size: Maximum number of items in queue.
            timeout: Default timeout in seconds for queued requests.
        """
        self._max_size = max_size
        self._timeout = timeout
        self._queue: list[QueueItem] = []
        self._lock = asyncio.Lock()
        self._stats = {
            "total_queued": 0,
            "total_completed": 0,
            "total_timeout": 0,
            "total_rejected": 0,
        }

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return len(self._queue) >= self._max_size

    async def enqueue(
        self,
        model_name: str,
        gpu_count: int,
        priority: str = "normal",
    ) -> Optional[QueueItem]:
        """Add a request to the queue.

        Args:
            model_name: Name of the model requested.
            gpu_count: Number of GPUs needed.
            priority: Priority level (high, normal, low).

        Returns:
            QueueItem if enqueued, None if queue is full.
        """
        async with self._lock:
            if self.is_full:
                self._stats["total_rejected"] += 1
                logger.warning(f"Queue full, rejecting request for {model_name}")
                return None

            item = QueueItem(
                id=str(uuid4())[:8],
                model_name=model_name,
                gpu_count=gpu_count,
                priority=priority,
                created_at=datetime.now(),
            )

            # Insert based on priority
            insert_idx = len(self._queue)
            for i, existing in enumerate(self._queue):
                if self.PRIORITY_ORDER.get(priority, 1) < self.PRIORITY_ORDER.get(
                    existing.priority, 1
                ):
                    insert_idx = i
                    break

            self._queue.insert(insert_idx, item)
            self._stats["total_queued"] += 1

            logger.info(
                f"Enqueued request {item.id} for {model_name} "
                f"(priority: {priority}, position: {insert_idx + 1}/{len(self._queue)})"
            )

            return item

    async def wait_for_turn(
        self,
        item: QueueItem,
        timeout: Optional[int] = None,
    ) -> bool:
        """Wait for a queued item to be processed.

        Args:
            item: Queue item to wait for.
            timeout: Timeout in seconds (uses default if None).

        Returns:
            True if item was processed, False if timed out.
        """
        timeout = timeout or self._timeout

        try:
            await asyncio.wait_for(item.event.wait(), timeout=timeout)
            return item.state == QueueItemState.COMPLETED
        except asyncio.TimeoutError:
            async with self._lock:
                item.state = QueueItemState.TIMEOUT
                item.error = f"Queue timeout after {timeout}s"
                self._stats["total_timeout"] += 1
                if item in self._queue:
                    self._queue.remove(item)
            logger.warning(f"Queue item {item.id} timed out after {timeout}s")
            return False

    async def get_next(self) -> Optional[QueueItem]:
        """Get the next item from the queue.

        Returns:
            Next QueueItem or None if queue is empty.
        """
        async with self._lock:
            for item in self._queue:
                if item.state == QueueItemState.WAITING:
                    return item
            return None

    async def mark_processing(self, item: QueueItem):
        """Mark an item as being processed."""
        async with self._lock:
            item.state = QueueItemState.PROCESSING

    async def mark_completed(self, item: QueueItem, result: Any = None):
        """Mark an item as completed and notify waiter.

        Args:
            item: Queue item to complete.
            result: Optional result to pass to waiter.
        """
        async with self._lock:
            item.state = QueueItemState.COMPLETED
            item.result = result
            self._stats["total_completed"] += 1
            if item in self._queue:
                self._queue.remove(item)

        item.event.set()
        logger.info(f"Queue item {item.id} completed")

    async def mark_failed(self, item: QueueItem, error: str):
        """Mark an item as failed and notify waiter.

        Args:
            item: Queue item that failed.
            error: Error message.
        """
        async with self._lock:
            item.state = QueueItemState.CANCELLED
            item.error = error
            if item in self._queue:
                self._queue.remove(item)

        item.event.set()
        logger.warning(f"Queue item {item.id} failed: {error}")

    async def cancel(self, item_id: str) -> bool:
        """Cancel a queued item.

        Args:
            item_id: ID of item to cancel.

        Returns:
            True if item was found and cancelled.
        """
        async with self._lock:
            for item in self._queue:
                if item.id == item_id:
                    item.state = QueueItemState.CANCELLED
                    item.error = "Cancelled by user"
                    self._queue.remove(item)
                    item.event.set()
                    logger.info(f"Cancelled queue item {item_id}")
                    return True
            return False

    async def clear(self) -> int:
        """Clear all items from the queue.

        Returns:
            Number of items cleared.
        """
        async with self._lock:
            count = len(self._queue)
            for item in self._queue:
                item.state = QueueItemState.CANCELLED
                item.error = "Queue cleared"
                item.event.set()
            self._queue.clear()
            logger.info(f"Cleared {count} items from queue")
            return count

    def get_waiting_items(self) -> list[QueueItem]:
        """Get all waiting items."""
        return [item for item in self._queue if item.state == QueueItemState.WAITING]

    def get_position(self, item_id: str) -> Optional[int]:
        """Get position of an item in the queue.

        Args:
            item_id: ID of item to find.

        Returns:
            Position (1-indexed) or None if not found.
        """
        for i, item in enumerate(self._queue):
            if item.id == item_id:
                return i + 1
        return None

    def get_status(self) -> dict:
        """Get queue status for API response."""
        return {
            "size": len(self._queue),
            "max_size": self._max_size,
            "timeout_seconds": self._timeout,
            "items": [item.to_dict() for item in self._queue],
            "stats": dict(self._stats),
        }

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return dict(self._stats)
