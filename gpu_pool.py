"""GPU Pool Manager - Handles GPU allocation and tracking."""

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


def _get_gpu_hardware_info() -> dict[int, dict]:
    """Get GPU name and VRAM from nvidia-smi.

    Returns:
        Dict mapping GPU ID to {"name": str, "vram_mb": int}
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed, GPU info unavailable")
            return {}

        gpu_info = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_id = int(parts[0])
                gpu_info[gpu_id] = {
                    "name": parts[1],
                    "vram_mb": int(parts[2]),
                }
        return gpu_info
    except Exception as e:
        logger.warning(f"Failed to get GPU hardware info: {e}")
        return {}


def _get_gpu_realtime_metrics() -> dict[int, dict]:
    """Get real-time GPU utilization and memory usage.

    Returns:
        Dict mapping GPU ID to {"gpu_load_percent": int, "vram_used_mb": int, "vram_used_percent": float}
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return {}

        metrics = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_id = int(parts[0])
                gpu_load = int(parts[1])
                vram_used = int(parts[2])
                vram_total = int(parts[3])
                vram_percent = round((vram_used / vram_total) * 100, 1) if vram_total > 0 else 0
                metrics[gpu_id] = {
                    "gpu_load_percent": gpu_load,
                    "vram_used_mb": vram_used,
                    "vram_used_percent": vram_percent,
                }
        return metrics
    except Exception:
        return {}


class GPUState(str, Enum):
    """State of a GPU."""
    FREE = "free"
    ALLOCATED = "allocated"


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    id: int
    name: str = "Unknown"
    vram_mb: int = 0
    state: GPUState = GPUState.FREE
    allocated_to_instance: Optional[str] = None  # Instance ID
    model_loaded: Optional[str] = None
    allocated_at: Optional[datetime] = None

    def to_dict(self, realtime_metrics: Optional[dict] = None) -> dict:
        """Convert to dictionary for API response.

        Args:
            realtime_metrics: Optional dict with gpu_load_percent, vram_used_mb, vram_used_percent
        """
        result = {
            "id": self.id,
            "name": self.name,
            "vram_mb": self.vram_mb,
            "vram_gb": round(self.vram_mb / 1024, 1),
            "state": self.state.value,
            "allocated_to_instance": self.allocated_to_instance,
            "model_loaded": self.model_loaded,
            "allocated_at": self.allocated_at.isoformat() if self.allocated_at else None,
        }
        if realtime_metrics:
            result["gpu_load_percent"] = realtime_metrics.get("gpu_load_percent", 0)
            result["vram_used_mb"] = realtime_metrics.get("vram_used_mb", 0)
            result["vram_used_percent"] = realtime_metrics.get("vram_used_percent", 0.0)
        return result


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
        hw_info = _get_gpu_hardware_info()
        self._gpus: dict[int, GPUInfo] = {}
        for gpu_id in gpu_ids:
            info = hw_info.get(gpu_id, {})
            self._gpus[gpu_id] = GPUInfo(
                id=gpu_id,
                name=info.get("name", "Unknown"),
                vram_mb=info.get("vram_mb", 0),
            )
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

    def get_status(self, include_realtime: bool = True) -> dict:
        """Get current pool status for API response.

        Args:
            include_realtime: If True, fetch real-time GPU metrics (load, vram usage)
        """
        realtime_metrics = _get_gpu_realtime_metrics() if include_realtime else {}

        return {
            "total_gpus": self.total_gpus,
            "free_gpus": self.free_count,
            "allocated_gpus": self.allocated_count,
            "gpus": [
                gpu.to_dict(realtime_metrics.get(gpu.id))
                for gpu in self._gpus.values()
            ],
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
