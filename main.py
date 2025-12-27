"""Ollama Load Balancer - Main FastAPI Application."""

import asyncio
import json
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import psutil
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import Config, load_config, set_config
from gpu_pool import GPUPool
from ollama_manager import OllamaManager
from proxy import OllamaProxy
from request_queue import RequestQueue

# Global state
config: Config = None
gpu_pool: GPUPool = None
ollama_manager: OllamaManager = None
request_queue: RequestQueue = None
proxy: OllamaProxy = None
start_time: datetime = None
health_check_task: asyncio.Task = None


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(level: str, format_type: str):
    """Setup logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    handler = logging.StreamHandler(sys.stdout)

    if format_type == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    root_logger.handlers = [handler]


async def periodic_health_check():
    """Run periodic health checks on all instances."""
    while True:
        try:
            await asyncio.sleep(config.behavior.health_check_interval)
            await ollama_manager.health_check_all()

            # Process queue if resources freed up
            await process_queue()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.error(f"Health check error: {e}")


async def process_queue():
    """Process queued requests when resources become available."""
    while True:
        next_item = await request_queue.get_next()
        if not next_item:
            break

        # Check if we can now allocate
        can_alloc = await gpu_pool.can_allocate(next_item.gpu_count)
        if can_alloc:
            await request_queue.mark_processing(next_item)
            await request_queue.mark_completed(next_item)
        else:
            break  # No more resources


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logging.info(f"Received signal {signum}, initiating shutdown...")
    raise SystemExit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global config, gpu_pool, ollama_manager, request_queue, proxy
    global start_time, health_check_task

    # Load configuration
    config = load_config()
    set_config(config)

    # Setup logging
    setup_logging(config.server.log_level, config.server.log_format)

    logger = logging.getLogger(__name__)
    logger.info("Starting Ollama Load Balancer...")
    logger.info(f"Configuration: {config.total_gpus} GPUs, "
                f"when_busy={config.behavior.when_busy}")

    # Initialize components
    start_time = datetime.now()

    gpu_pool = GPUPool(config.gpu_ids)
    logger.info(f"GPU Pool initialized with GPUs: {config.gpu_ids}")

    ollama_manager = OllamaManager(gpu_pool, config.behavior)
    await ollama_manager.start()
    logger.info("Ollama Manager started")

    request_queue = RequestQueue(
        max_size=config.behavior.max_queue_size,
        timeout=config.behavior.queue_timeout,
    )
    logger.info(f"Request Queue initialized (max_size={config.behavior.max_queue_size})")

    proxy = OllamaProxy(config, gpu_pool, ollama_manager, request_queue)
    await proxy.start()
    logger.info("Proxy started")

    # Start health check task
    health_check_task = asyncio.create_task(periodic_health_check())

    # Setup signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    logger.info(f"Ollama Load Balancer ready on {config.server.host}:{config.server.port}")

    yield

    # Shutdown
    logger.info("Shutting down...")

    if health_check_task:
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            pass

    await request_queue.clear()
    await proxy.stop()
    await ollama_manager.stop()

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Ollama Load Balancer",
    description="Intelligent GPU-aware proxy for Ollama",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================================
# Ollama API Compatible Endpoints
# ============================================================================

@app.post("/api/generate")
async def generate(request: Request):
    """Generate completion - Ollama compatible."""
    return await proxy.handle_model_request(request, "/api/generate")


@app.post("/api/chat")
async def chat(request: Request):
    """Chat completion - Ollama compatible."""
    return await proxy.handle_model_request(request, "/api/chat")


@app.post("/api/embeddings")
async def embeddings(request: Request):
    """Generate embeddings - Ollama compatible."""
    return await proxy.handle_model_request(request, "/api/embeddings")


@app.post("/api/embed")
async def embed(request: Request):
    """Generate embeddings (alternative endpoint) - Ollama compatible."""
    return await proxy.handle_model_request(request, "/api/embed")


@app.get("/api/tags")
async def list_models(request: Request):
    """List available models - Ollama compatible."""
    return await proxy.handle_management_request(request, "/api/tags")


@app.get("/api/ps")
async def list_running(request: Request):
    """List running models - Ollama compatible."""
    return await proxy.handle_management_request(request, "/api/ps")


@app.post("/api/pull")
async def pull_model(request: Request):
    """Pull a model - Ollama compatible."""
    return await proxy.handle_management_request(request, "/api/pull")


@app.delete("/api/delete")
async def delete_model(request: Request):
    """Delete a model - Ollama compatible."""
    return await proxy.handle_management_request(request, "/api/delete")


@app.post("/api/copy")
async def copy_model(request: Request):
    """Copy a model - Ollama compatible."""
    return await proxy.handle_management_request(request, "/api/copy")


@app.post("/api/show")
async def show_model(request: Request):
    """Show model info - Ollama compatible."""
    return await proxy.handle_management_request(request, "/api/show")


@app.post("/api/create")
async def create_model(request: Request):
    """Create a model from Modelfile - Ollama compatible."""
    return await proxy.handle_management_request(request, "/api/create")


@app.head("/api/blobs/{digest}")
@app.get("/api/blobs/{digest}")
async def check_blob(request: Request, digest: str):
    """Check if blob exists - Ollama compatible."""
    return await proxy.handle_management_request(request, f"/api/blobs/{digest}")


@app.post("/api/blobs/{digest}")
async def create_blob(request: Request, digest: str):
    """Create a blob - Ollama compatible."""
    return await proxy.handle_management_request(request, f"/api/blobs/{digest}")


@app.get("/api/version")
async def version():
    """Get version - Ollama compatible."""
    return JSONResponse(content={"version": "ollama-loadbalancer-0.1.0"})


@app.get("/")
async def root():
    """Root endpoint - Ollama compatible."""
    return JSONResponse(content={"status": "Ollama Load Balancer is running"})


# ============================================================================
# Custom Status/Management Endpoints
# ============================================================================

@app.get("/api/status")
async def status():
    """Get detailed load balancer status."""
    uptime = (datetime.now() - start_time).total_seconds() if start_time else 0

    # Get system resource usage
    cpu_percent = psutil.cpu_percent(interval=0.1)  # Average across all cores
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)

    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024 ** 3)
    memory_used_gb = memory.used / (1024 ** 3)
    memory_percent = memory.percent

    return JSONResponse(content={
        "status": "running",
        "uptime_seconds": uptime,
        "system": {
            "cpu": {
                "usage_percent": round(cpu_percent, 2),
                "cores_logical": cpu_count,
                "cores_physical": cpu_count_physical,
            },
            "memory": {
                "total_gb": round(memory_total_gb, 2),
                "used_gb": round(memory_used_gb, 2),
                "usage_percent": round(memory_percent, 2),
            },
        },
        "gpu_pool": gpu_pool.get_status(),
        "instances": ollama_manager.get_status(),
        "queue": request_queue.get_status(),
        "proxy_stats": proxy.get_stats(),
        "config": {
            "total_gpus": config.total_gpus,
            "when_busy": config.behavior.when_busy,
            "queue_timeout": config.behavior.queue_timeout,
            "instance_ttl": config.behavior.instance_ttl,
            "max_queue_size": config.behavior.max_queue_size,
        },
    })


@app.get("/api/status/gpu")
async def gpu_status():
    """Get GPU pool status."""
    return JSONResponse(content=gpu_pool.get_status())


@app.get("/api/status/instances")
async def instances_status():
    """Get Ollama instances status."""
    return JSONResponse(content=ollama_manager.get_status())


@app.get("/api/status/queue")
async def queue_status():
    """Get request queue status."""
    return JSONResponse(content=request_queue.get_status())


@app.delete("/api/queue/{item_id}")
async def cancel_queue_item(item_id: str):
    """Cancel a queued request."""
    success = await request_queue.cancel(item_id)
    if success:
        return JSONResponse(content={"status": "cancelled", "id": item_id})
    return JSONResponse(
        status_code=404,
        content={"error": f"Queue item {item_id} not found"},
    )


@app.post("/api/admin/clear-queue")
async def clear_queue():
    """Clear all queued requests."""
    count = await request_queue.clear()
    return JSONResponse(content={"status": "cleared", "count": count})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy"})


@app.get("/ready")
async def ready():
    """Readiness check endpoint."""
    # Check if at least one GPU is available or an instance is running
    if gpu_pool.free_count > 0 or len(ollama_manager.get_all_instances()) > 0:
        return JSONResponse(content={"status": "ready"})
    return JSONResponse(
        status_code=503,
        content={"status": "not ready", "reason": "No GPUs available"},
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the application."""
    import uvicorn

    # Load config early for server settings
    cfg = load_config()

    uvicorn.run(
        "main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
