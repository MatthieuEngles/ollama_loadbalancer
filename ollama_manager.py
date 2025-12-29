"""Ollama Instance Manager - Handles spawning, health checking, and killing Ollama instances."""

import asyncio
import logging
import os
import signal
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import httpx

from config import BehaviorConfig
from gpu_pool import GPUPool

logger = logging.getLogger(__name__)


class InstanceState(str, Enum):
    """State of an Ollama instance."""
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class OllamaInstance:
    """Represents a running Ollama instance."""
    id: str
    port: int
    gpu_ids: list[int]
    model_name: str
    process: Optional[asyncio.subprocess.Process] = None
    state: InstanceState = InstanceState.STARTING
    created_at: datetime = field(default_factory=datetime.now)
    last_request_at: Optional[datetime] = None
    request_count: int = 0
    active_requests: int = 0
    context_length: Optional[int] = None
    model_size: Optional[str] = None
    model_parameters: Optional[dict] = None
    current_request_context: Optional[int] = None  # Context size of current request
    last_request_context: Optional[int] = None  # Context size of last completed request

    @property
    def host(self) -> str:
        """Get the host:port for this instance."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def cuda_devices(self) -> str:
        """Get CUDA_VISIBLE_DEVICES string."""
        return ",".join(str(g) for g in self.gpu_ids)

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "port": self.port,
            "gpu_ids": self.gpu_ids,
            "model_name": self.model_name,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_request_at": self.last_request_at.isoformat() if self.last_request_at else None,
            "request_count": self.request_count,
            "active_requests": self.active_requests,
            "context_length": self.context_length,
            "model_size": self.model_size,
            "model_parameters": self.model_parameters,
            "current_request_context": self.current_request_context,
            "last_request_context": self.last_request_context,
        }


class OllamaManager:
    """Manages Ollama instance lifecycle."""

    BASE_PORT = 11500
    MAX_INSTANCES = 10

    def __init__(self, gpu_pool: GPUPool, behavior: BehaviorConfig):
        """Initialize the Ollama manager.

        Args:
            gpu_pool: GPU pool to use for allocation.
            behavior: Behavior configuration.
        """
        self._gpu_pool = gpu_pool
        self._behavior = behavior
        self._instances: dict[str, OllamaInstance] = {}
        self._port_to_instance: dict[int, str] = {}
        self._next_port = self.BASE_PORT
        self._lock = asyncio.Lock()
        self._ttl_tasks: dict[str, asyncio.Task] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._shutdown = False

    async def start(self):
        """Start the manager and HTTP client."""
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._shutdown = False

    async def stop(self):
        """Stop the manager and cleanup all instances."""
        self._shutdown = True

        # Cancel all TTL tasks
        for task in self._ttl_tasks.values():
            task.cancel()
        self._ttl_tasks.clear()

        # Stop all instances
        instance_ids = list(self._instances.keys())
        for instance_id in instance_ids:
            await self._stop_instance(instance_id)

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _get_next_port(self) -> int:
        """Get the next available port."""
        while self._next_port in self._port_to_instance:
            self._next_port += 1
            if self._next_port > self.BASE_PORT + self.MAX_INSTANCES:
                self._next_port = self.BASE_PORT
        port = self._next_port
        self._next_port += 1
        return port

    def _get_least_loaded_instance(self, model_name: str) -> Optional[OllamaInstance]:
        """Get the least loaded ready instance for a model.

        Args:
            model_name: Model to find instance for.

        Returns:
            Least loaded instance, or None if no ready instance exists.
        """
        instance_ids = self._gpu_pool.get_instances_for_model(model_name)
        best_instance = None
        min_load = float('inf')

        for inst_id in instance_ids:
            instance = self._instances.get(inst_id)
            if instance and instance.state == InstanceState.READY:
                if instance.active_requests < min_load:
                    min_load = instance.active_requests
                    best_instance = instance

        return best_instance

    async def get_or_create_instance(
        self,
        model_name: str,
        gpu_count: int,
    ) -> Optional[OllamaInstance]:
        """Get an existing instance for the model or create a new one.

        Load balancing strategy:
        1. If free GPUs available -> create new instance (horizontal scaling)
        2. If no free GPUs -> use least loaded existing instance
        3. If no instance exists -> queue/reject based on config

        Args:
            model_name: Name of the model to load.
            gpu_count: Number of GPUs required.

        Returns:
            OllamaInstance if successful, None otherwise.
        """
        async with self._lock:
            # Check if we can create a new instance (horizontal scaling)
            can_allocate = await self._gpu_pool.can_allocate(gpu_count)

            if can_allocate:
                # Create new instance for parallel processing
                instance_id = str(uuid.uuid4())[:8]
                allocation = await self._gpu_pool.try_allocate(
                    gpu_count=gpu_count,
                    instance_id=instance_id,
                    model_name=model_name,
                )

                if allocation.success:
                    port = self._get_next_port()
                    instance = OllamaInstance(
                        id=instance_id,
                        port=port,
                        gpu_ids=allocation.gpu_ids,
                        model_name=model_name,
                    )
                    self._instances[instance_id] = instance
                    self._port_to_instance[port] = instance_id
                    logger.info(f"Creating new instance {instance_id} on GPU(s) {allocation.gpu_ids} for {model_name}")
                    new_instance = instance
                else:
                    new_instance = None
            else:
                new_instance = None

            # If we couldn't allocate, try to reuse least loaded instance
            if new_instance is None:
                existing = self._get_least_loaded_instance(model_name)
                if existing:
                    logger.info(f"Reusing instance {existing.id} for {model_name} (active_requests={existing.active_requests})")
                    self._reset_ttl(existing.id)
                    return existing

                # Check for starting instances we can wait on
                instance_ids = self._gpu_pool.get_instances_for_model(model_name)
                for inst_id in instance_ids:
                    inst = self._instances.get(inst_id)
                    if inst and inst.state == InstanceState.STARTING:
                        logger.info(f"Waiting for starting instance {inst.id} for {model_name}")
                        starting_instance = inst
                        break
                else:
                    starting_instance = None

                if starting_instance:
                    # Release lock and wait for starting instance
                    pass
                else:
                    # No GPUs and no existing instance
                    logger.warning(f"No GPUs available and no existing instance for {model_name}")
                    return None
            else:
                starting_instance = None

        # Wait for starting instance if needed
        if starting_instance:
            ready = await self._wait_for_ready(starting_instance)
            if ready and starting_instance.state == InstanceState.READY:
                self._reset_ttl(starting_instance.id)
                return starting_instance
            return None

        # Spawn Ollama process for new instance (outside lock)
        success = await self._spawn_ollama(new_instance)
        if not success:
            await self._cleanup_failed_instance(new_instance)
            return None

        # Wait for instance to be ready
        ready = await self._wait_for_ready(new_instance)
        if not ready:
            await self._cleanup_failed_instance(new_instance)
            return None

        return new_instance

    async def get_or_create_management_instance(self) -> Optional[OllamaInstance]:
        """Get or create a lightweight instance for management operations (pull, delete, etc.).

        This instance doesn't require GPU allocation and is used for operations
        that don't need model inference.

        Returns:
            OllamaInstance if successful, None otherwise.
        """
        # First, try to use any existing ready instance
        for instance in self._instances.values():
            if instance.state == InstanceState.READY:
                logger.info(f"Reusing existing instance {instance.id} for management operation")
                return instance

        # Check if management instance already exists
        mgmt_instance = self._instances.get("__mgmt__")
        if mgmt_instance:
            if mgmt_instance.state == InstanceState.READY:
                return mgmt_instance
            elif mgmt_instance.state == InstanceState.STARTING:
                # Wait for it to be ready
                ready = await self._wait_for_ready(mgmt_instance)
                if ready:
                    return mgmt_instance
                return None

        # Create a new management instance (no GPU allocation needed)
        async with self._lock:
            # Double-check after acquiring lock
            if "__mgmt__" in self._instances:
                return self._instances["__mgmt__"]

            port = self._get_next_port()
            instance = OllamaInstance(
                id="__mgmt__",
                port=port,
                gpu_ids=[],  # No GPU needed for management
                model_name="__management__",
            )
            self._instances["__mgmt__"] = instance
            self._port_to_instance[port] = "__mgmt__"

        logger.info(f"Creating management instance on port {port}")

        # Spawn without GPU
        success = await self._spawn_management_ollama(instance)
        if not success:
            await self._cleanup_failed_instance(instance)
            return None

        # Wait for ready
        ready = await self._wait_for_ready(instance)
        if not ready:
            await self._cleanup_failed_instance(instance)
            return None

        return instance

    async def _spawn_management_ollama(self, instance: OllamaInstance) -> bool:
        """Spawn an Ollama process for management operations (no GPU needed).

        Args:
            instance: Instance to spawn process for.

        Returns:
            True if spawn succeeded.
        """
        env = os.environ.copy()
        # No CUDA_VISIBLE_DEVICES - let Ollama use CPU for management ops
        env["OLLAMA_HOST"] = f"0.0.0.0:{instance.port}"

        # Use OLLAMA_MODELS from environment if set, otherwise default to user's home
        if "OLLAMA_MODELS" not in env:
            home_dir = os.path.expanduser("~")
            env["OLLAMA_MODELS"] = os.path.join(home_dir, ".ollama", "models")

        try:
            logger.info(f"Spawning management Ollama on port {instance.port} (no GPU)")

            process = await asyncio.create_subprocess_exec(
                "ollama",
                "serve",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            instance.process = process

            # Give it a moment to start
            await asyncio.sleep(1)

            # Check if process died immediately
            if process.returncode is not None:
                stdout, stderr = await process.communicate()
                logger.error(
                    f"Management Ollama process exited immediately with code {process.returncode}. "
                    f"stderr: {stderr.decode()}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to spawn management Ollama: {e}")
            return False

    async def _spawn_ollama(self, instance: OllamaInstance) -> bool:
        """Spawn an Ollama process.

        Args:
            instance: Instance to spawn process for.

        Returns:
            True if spawn succeeded.
        """
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = instance.cuda_devices
        env["OLLAMA_HOST"] = f"0.0.0.0:{instance.port}"
        env["OLLAMA_KEEP_ALIVE"] = "-1"  # Keep model loaded

        # Use OLLAMA_MODELS from environment if set, otherwise default to user's home
        if "OLLAMA_MODELS" not in env:
            home_dir = os.path.expanduser("~")
            env["OLLAMA_MODELS"] = os.path.join(home_dir, ".ollama", "models")

        env["OLLAMA_GPU_OVERHEAD"] = "0"  # No GPU memory overhead
        env["OLLAMA_MAX_LOADED_MODELS"] = "1"  # One model per instance
        env["OLLAMA_FLASH_ATTENTION"] = "1"  # Enable flash attention
        env["OLLAMA_LLM_LIBRARY"] = "cuda_v12"  # Force CUDA
        env["OLLAMA_NUM_GPU"] = "999"  # Force all layers on GPU (no CPU offload)

        try:
            logger.info(
                f"Spawning Ollama on port {instance.port} "
                f"with CUDA_VISIBLE_DEVICES={instance.cuda_devices}"
            )

            process = await asyncio.create_subprocess_exec(
                "ollama",
                "serve",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            instance.process = process

            # Give it a moment to start
            await asyncio.sleep(1)

            # Check if process died immediately
            if process.returncode is not None:
                stdout, stderr = await process.communicate()
                logger.error(
                    f"Ollama process exited immediately with code {process.returncode}. "
                    f"stderr: {stderr.decode()}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to spawn Ollama: {e}")
            return False

    async def _wait_for_ready(self, instance: OllamaInstance) -> bool:
        """Wait for an Ollama instance to become ready.

        Args:
            instance: Instance to wait for.

        Returns:
            True if instance became ready within timeout.
        """
        timeout = self._behavior.startup_timeout
        interval = 0.5
        elapsed = 0

        while elapsed < timeout:
            if self._shutdown:
                return False

            # Check if already marked ready by another coroutine
            if instance.state == InstanceState.READY:
                return True

            try:
                url = f"{instance.host}/api/tags"
                logger.debug(f"Health check {instance.id}: GET {url}")
                response = await self._http_client.get(url, timeout=5.0)
                logger.debug(f"Health check {instance.id}: status={response.status_code}")
                if response.status_code == 200:
                    # Mark as ready (atomic state change)
                    if instance.state == InstanceState.STARTING:
                        instance.state = InstanceState.READY
                        self._schedule_ttl(instance.id)
                        logger.info(f"Instance {instance.id} ready on port {instance.port}")
                        # Fetch model info in the background
                        asyncio.create_task(self._fetch_model_info(instance))
                    return True
            except Exception as e:
                logger.debug(f"Health check {instance.id} failed: {e}")

            await asyncio.sleep(interval)
            elapsed += interval

        logger.error(f"Instance {instance.id} failed to become ready within {timeout}s")
        return False

    async def _fetch_model_info(self, instance: OllamaInstance):
        """Fetch model information from Ollama instance.

        Args:
            instance: Instance to fetch info for.
        """
        if not instance.model_name or instance.model_name.startswith("__"):
            return  # Skip for management instances

        try:
            response = await self._http_client.post(
                f"{instance.host}/api/show",
                json={"name": instance.model_name},
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                # Extract context length from model info
                if "model_info" in data:
                    model_info = data["model_info"]
                    # Try to parse context length from various fields
                    for key in model_info:
                        if "context" in key.lower():
                            try:
                                instance.context_length = int(model_info[key])
                            except (ValueError, TypeError):
                                pass

                # Extract parameters
                if "parameters" in data:
                    instance.model_parameters = data["parameters"]

                # Extract model size
                if "size" in data:
                    # Convert bytes to human readable
                    size_bytes = data["size"]
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if size_bytes < 1024.0:
                            instance.model_size = f"{size_bytes:.1f}{unit}"
                            break
                        size_bytes /= 1024.0

                logger.debug(f"Fetched model info for {instance.id}: "
                           f"context={instance.context_length}, size={instance.model_size}")
        except Exception as e:
            logger.debug(f"Failed to fetch model info for {instance.id}: {e}")

    async def _cleanup_failed_instance(self, instance: OllamaInstance):
        """Cleanup a failed instance."""
        logger.warning(f"Cleaning up failed instance {instance.id}")
        instance.state = InstanceState.ERROR

        async with self._lock:
            if instance.id in self._instances:
                del self._instances[instance.id]
            if instance.port in self._port_to_instance:
                del self._port_to_instance[instance.port]

        if instance.process:
            await self._kill_process(instance.process)

        await self._gpu_pool.release(instance.id)

    async def _stop_instance(self, instance_id: str):
        """Stop an Ollama instance.

        Args:
            instance_id: ID of instance to stop.
        """
        instance = self._instances.get(instance_id)
        if not instance:
            return

        logger.info(f"Stopping instance {instance_id}")
        instance.state = InstanceState.STOPPING

        # Cancel TTL task
        if instance_id in self._ttl_tasks:
            self._ttl_tasks[instance_id].cancel()
            del self._ttl_tasks[instance_id]

        # Kill process
        if instance.process:
            await self._kill_process(instance.process)

        instance.state = InstanceState.STOPPED

        # Cleanup
        async with self._lock:
            if instance_id in self._instances:
                del self._instances[instance_id]
            if instance.port in self._port_to_instance:
                del self._port_to_instance[instance.port]

        # Release GPUs
        await self._gpu_pool.release(instance_id)

    async def _kill_process(self, process: asyncio.subprocess.Process):
        """Kill an Ollama process gracefully."""
        try:
            # Try SIGTERM first
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                await process.wait()
        except ProcessLookupError:
            pass  # Process already dead
        except Exception as e:
            logger.warning(f"Error killing process: {e}")

    def _schedule_ttl(self, instance_id: str):
        """Schedule TTL expiration for an instance."""
        if instance_id in self._ttl_tasks:
            self._ttl_tasks[instance_id].cancel()

        async def ttl_callback():
            try:
                await asyncio.sleep(self._behavior.instance_ttl)
                instance = self._instances.get(instance_id)
                if instance and instance.active_requests == 0:
                    logger.info(f"Instance {instance_id} TTL expired, stopping")
                    await self._stop_instance(instance_id)
            except asyncio.CancelledError:
                pass

        self._ttl_tasks[instance_id] = asyncio.create_task(ttl_callback())

    def _reset_ttl(self, instance_id: str):
        """Reset TTL timer for an instance."""
        self._schedule_ttl(instance_id)

    def mark_request_start(self, instance_id: str, context_size: Optional[int] = None):
        """Mark the start of a request to an instance.

        Args:
            instance_id: ID of the instance.
            context_size: Context size requested for this request.
        """
        instance = self._instances.get(instance_id)
        if instance:
            instance.active_requests += 1
            instance.last_request_at = datetime.now()
            instance.request_count += 1
            if context_size is not None:
                instance.current_request_context = context_size

            # Cancel TTL while requests are active
            if instance_id in self._ttl_tasks:
                self._ttl_tasks[instance_id].cancel()
                del self._ttl_tasks[instance_id]

    def mark_request_end(self, instance_id: str):
        """Mark the end of a request to an instance."""
        instance = self._instances.get(instance_id)
        if instance:
            instance.active_requests = max(0, instance.active_requests - 1)
            # Save the context size of the completed request
            if instance.current_request_context is not None:
                instance.last_request_context = instance.current_request_context
                instance.current_request_context = None
            if instance.active_requests == 0:
                self._schedule_ttl(instance_id)

    def get_instance(self, instance_id: str) -> Optional[OllamaInstance]:
        """Get an instance by ID."""
        return self._instances.get(instance_id)

    def get_instance_for_model(self, model_name: str) -> Optional[OllamaInstance]:
        """Get least loaded ready instance that has model loaded."""
        return self._get_least_loaded_instance(model_name)

    def get_all_instances(self) -> list[OllamaInstance]:
        """Get all instances."""
        return list(self._instances.values())

    def get_status(self) -> dict:
        """Get manager status for API response."""
        return {
            "instances": [inst.to_dict() for inst in self._instances.values()],
            "total_instances": len(self._instances),
            "active_requests": sum(inst.active_requests for inst in self._instances.values()),
        }

    async def health_check_all(self):
        """Perform health check on all instances."""
        instances_to_cleanup = []

        for instance in list(self._instances.values()):
            # Cleanup stuck STOPPING or ERROR instances
            if instance.state in (InstanceState.STOPPING, InstanceState.ERROR):
                logger.info(f"Cleaning up {instance.state.value} instance {instance.id}")
                instances_to_cleanup.append(instance.id)
                continue

            if instance.state != InstanceState.READY:
                continue

            try:
                response = await self._http_client.get(
                    f"{instance.host}/api/tags",
                    timeout=5.0,
                )
                if response.status_code != 200:
                    logger.warning(f"Instance {instance.id} health check failed")
                    instance.state = InstanceState.ERROR
                    instances_to_cleanup.append(instance.id)
            except Exception as e:
                logger.warning(f"Instance {instance.id} health check error: {e}")
                instance.state = InstanceState.ERROR
                instances_to_cleanup.append(instance.id)

        # Cleanup dead instances
        for instance_id in instances_to_cleanup:
            await self._cleanup_dead_instance(instance_id)

    async def _cleanup_dead_instance(self, instance_id: str):
        """Cleanup a dead/stuck instance and release its resources."""
        instance = self._instances.get(instance_id)
        if not instance:
            return

        logger.info(f"Cleaning up dead instance {instance_id}")

        # Cancel TTL task
        if instance_id in self._ttl_tasks:
            self._ttl_tasks[instance_id].cancel()
            del self._ttl_tasks[instance_id]

        # Kill process if still running
        if instance.process:
            await self._kill_process(instance.process)

        # Remove from tracking
        async with self._lock:
            if instance_id in self._instances:
                del self._instances[instance_id]
            if instance.port in self._port_to_instance:
                del self._port_to_instance[instance.port]

        # Release GPUs
        await self._gpu_pool.release(instance_id)
