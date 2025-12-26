"""GPU Pool Manager - Handles GPU allocation and tracking."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GPUState(str, Enum):
    """State of a GPU."""
    FREE = "free"
    ALLOCATED = "allocated"


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    id: int
    state: GPUState = GPUState.FREE
    allocated_to_instance: Optional[str] = None  # Instance ID
    model_loaded: Optional[str] = None
    allocated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "state": self.state.value,
            "allocated_to_instance": self.allocated_to_instance,
            "model_loaded": self.model_loaded,
            "allocated_at": self.allocated_at.isoformat() if self.allocated_at else None,
        }


@dataclass
class AllocationResult:
    """Result of a GPU allocation attempt."""
    success: bool
    gpu_ids: list[int] = field(default_factory=list)
    instance_id: Optional[str] = None
    error: Optional[str] = None
    reused_existing: bool = False


class GPUPool:
    """Manages a pool of GPUs for Ollama instances."""

    def __init__(self, gpu_ids: list[int]):
        """Initialize the GPU pool.

        Args:
            gpu_ids: List of GPU IDs to manage.
        """
        self._gpus: dict[int, GPUInfo] = {
            gpu_id: GPUInfo(id=gpu_id) for gpu_id in gpu_ids
        }
        self._lock = asyncio.Lock()
        self._instances: dict[str, set[int]] = {}  # instance_id -> set of GPU IDs
        self._model_instances: dict[str, str] = {}  # model_name -> instance_id

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs in pool."""
        return len(self._gpus)

    @property
    def free_gpus(self) -> list[int]:
        """List of free GPU IDs."""
        return [gpu.id for gpu in self._gpus.values() if gpu.state == GPUState.FREE]

    @property
    def free_count(self) -> int:
        """Number of free GPUs."""
        return len(self.free_gpus)

    @property
    def allocated_count(self) -> int:
        """Number of allocated GPUs."""
        return self.total_gpus - self.free_count

    def get_gpu_info(self, gpu_id: int) -> Optional[GPUInfo]:
        """Get information about a specific GPU."""
        return self._gpus.get(gpu_id)

    def get_all_gpus(self) -> list[GPUInfo]:
        """Get information about all GPUs."""
        return list(self._gpus.values())

    def get_instance_for_model(self, model_name: str) -> Optional[str]:
        """Get instance ID if model is already loaded.

        Args:
            model_name: Name of the model to check.

        Returns:
            Instance ID if model is loaded, None otherwise.
        """
        return self._model_instances.get(model_name)

    def get_instance_gpus(self, instance_id: str) -> set[int]:
        """Get GPU IDs allocated to an instance."""
        return self._instances.get(instance_id, set())

    async def try_allocate(
        self,
        gpu_count: int,
        instance_id: str,
        model_name: str,
    ) -> AllocationResult:
        """Try to allocate GPUs for a new instance.

        This is atomic - either all GPUs are allocated or none.

        Args:
            gpu_count: Number of GPUs needed.
            instance_id: Unique identifier for the instance.
            model_name: Name of the model to be loaded.

        Returns:
            AllocationResult with success status and allocated GPU IDs.
        """
        async with self._lock:
            # Check if model is already loaded
            existing_instance = self._model_instances.get(model_name)
            if existing_instance and existing_instance in self._instances:
                logger.info(
                    f"Model {model_name} already loaded on instance {existing_instance}"
                )
                return AllocationResult(
                    success=True,
                    gpu_ids=list(self._instances[existing_instance]),
                    instance_id=existing_instance,
                    reused_existing=True,
                )

            # Check if enough GPUs are free
            free_gpus = self.free_gpus
            if len(free_gpus) < gpu_count:
                logger.warning(
                    f"Not enough GPUs: need {gpu_count}, have {len(free_gpus)} free"
                )
                return AllocationResult(
                    success=False,
                    error=f"Insufficient GPUs: need {gpu_count}, available {len(free_gpus)}",
                )

            # Allocate the first N free GPUs
            allocated_gpus = free_gpus[:gpu_count]
            now = datetime.now()

            for gpu_id in allocated_gpus:
                self._gpus[gpu_id].state = GPUState.ALLOCATED
                self._gpus[gpu_id].allocated_to_instance = instance_id
                self._gpus[gpu_id].model_loaded = model_name
                self._gpus[gpu_id].allocated_at = now

            self._instances[instance_id] = set(allocated_gpus)
            self._model_instances[model_name] = instance_id

            logger.info(
                f"Allocated GPUs {allocated_gpus} for instance {instance_id} "
                f"(model: {model_name})"
            )

            return AllocationResult(
                success=True,
                gpu_ids=allocated_gpus,
                instance_id=instance_id,
                reused_existing=False,
            )

    async def release(self, instance_id: str) -> bool:
        """Release GPUs allocated to an instance.

        Args:
            instance_id: Instance to release GPUs for.

        Returns:
            True if GPUs were released, False if instance not found.
        """
        async with self._lock:
            if instance_id not in self._instances:
                logger.warning(f"Instance {instance_id} not found in pool")
                return False

            gpu_ids = self._instances.pop(instance_id)

            # Find and remove model mapping
            model_to_remove = None
            for model, inst_id in self._model_instances.items():
                if inst_id == instance_id:
                    model_to_remove = model
                    break
            if model_to_remove:
                del self._model_instances[model_to_remove]

            # Free the GPUs
            for gpu_id in gpu_ids:
                if gpu_id in self._gpus:
                    self._gpus[gpu_id].state = GPUState.FREE
                    self._gpus[gpu_id].allocated_to_instance = None
                    self._gpus[gpu_id].model_loaded = None
                    self._gpus[gpu_id].allocated_at = None

            logger.info(f"Released GPUs {list(gpu_ids)} from instance {instance_id}")
            return True

    async def release_all(self) -> int:
        """Release all allocated GPUs.

        Returns:
            Number of instances released.
        """
        async with self._lock:
            count = len(self._instances)

            for gpu_id in self._gpus:
                self._gpus[gpu_id].state = GPUState.FREE
                self._gpus[gpu_id].allocated_to_instance = None
                self._gpus[gpu_id].model_loaded = None
                self._gpus[gpu_id].allocated_at = None

            self._instances.clear()
            self._model_instances.clear()

            logger.info(f"Released all GPUs, cleared {count} instances")
            return count

    def get_status(self) -> dict:
        """Get current pool status for API response."""
        return {
            "total_gpus": self.total_gpus,
            "free_gpus": self.free_count,
            "allocated_gpus": self.allocated_count,
            "gpus": [gpu.to_dict() for gpu in self._gpus.values()],
            "instances": {
                inst_id: list(gpus) for inst_id, gpus in self._instances.items()
            },
            "loaded_models": dict(self._model_instances),
        }

    async def can_allocate(self, gpu_count: int) -> bool:
        """Check if allocation is possible without actually allocating.

        Args:
            gpu_count: Number of GPUs needed.

        Returns:
            True if allocation would succeed.
        """
        async with self._lock:
            return len(self.free_gpus) >= gpu_count
