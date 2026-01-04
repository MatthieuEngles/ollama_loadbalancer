"""Proxy Logic - Handles request proxying to Ollama instances."""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import Config
from gpu_pool import GPUPool
from ollama_manager import OllamaInstance, OllamaManager
from request_queue import RequestQueue

logger = logging.getLogger(__name__)


class OllamaProxy:
    """Handles proxying requests to Ollama instances."""

    # Endpoints that require model-based routing
    MODEL_ENDPOINTS = {
        "/api/generate",
        "/api/chat",
        "/api/embeddings",
        "/api/embed",
    }

    # OpenAI-compatible endpoints
    OPENAI_ENDPOINTS = {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
    }

    # Endpoints that can go to any instance or need special handling
    MANAGEMENT_ENDPOINTS = {
        "/api/tags",
        "/api/ps",
        "/api/pull",
        "/api/delete",
        "/api/copy",
        "/api/show",
        "/api/create",
        "/api/blobs",
        "/api/version",
        "/v1/models",
        "/",
    }

    def __init__(
        self,
        config: Config,
        gpu_pool: GPUPool,
        ollama_manager: OllamaManager,
        request_queue: RequestQueue,
    ):
        """Initialize the proxy.

        Args:
            config: Application configuration.
            gpu_pool: GPU pool manager.
            ollama_manager: Ollama instance manager.
            request_queue: Request queue.
        """
        self._config = config
        self._gpu_pool = gpu_pool
        self._ollama_manager = ollama_manager
        self._request_queue = request_queue
        self._http_client: Optional[httpx.AsyncClient] = None
        self._stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "requests_queued": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }

    async def start(self):
        """Start the proxy HTTP client."""
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=300.0,  # Long timeout for generation
                write=10.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
            ),
        )

    async def stop(self):
        """Stop the proxy HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _extract_model_from_body(self, body: bytes) -> Optional[str]:
        """Extract model name from request body.

        Args:
            body: Request body bytes.

        Returns:
            Model name or None.
        """
        try:
            data = json.loads(body)
            return data.get("model")
        except (json.JSONDecodeError, KeyError):
            return None

    def _extract_context_size(self, body: bytes) -> Optional[int]:
        """Extract context size from request body.

        Args:
            body: Request body bytes.

        Returns:
            Context size or None.
        """
        try:
            data = json.loads(body)
            # Check for explicit num_ctx in options
            if "options" in data and "num_ctx" in data["options"]:
                return data["options"]["num_ctx"]
            # Estimate from prompt/messages length
            total_length = 0
            if "prompt" in data:
                total_length = len(data["prompt"])
            elif "messages" in data:
                for msg in data["messages"]:
                    if "content" in msg:
                        total_length += len(msg["content"])
            return total_length if total_length > 0 else None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _inject_gpu_options(self, body: bytes, gpu_count: int) -> bytes:
        """Inject GPU options to force full GPU usage (no CPU offload).

        Args:
            body: Original request body.
            gpu_count: Number of GPUs allocated.

        Returns:
            Modified body with GPU options.
        """
        try:
            data = json.loads(body)
            # Set options to force all layers on GPU
            if "options" not in data:
                data["options"] = {}
            # num_gpu=999 forces all layers on GPU
            data["options"]["num_gpu"] = 999
            # Disable mmap to avoid memory mapping issues
            data["options"]["use_mmap"] = False
            return json.dumps(data).encode()
        except (json.JSONDecodeError, KeyError):
            return body

    def _is_streaming_request(self, body: bytes) -> bool:
        """Check if request wants streaming response.

        Args:
            body: Request body bytes.

        Returns:
            True if streaming is requested (default is True for Ollama).
        """
        try:
            data = json.loads(body)
            # Ollama defaults to streaming=True
            return data.get("stream", True)
        except (json.JSONDecodeError, KeyError):
            return True

    async def handle_model_request(
        self,
        request: Request,
        path: str,
        _retry_gpu_count: int | None = None,
        _cached_body: bytes | None = None,
    ) -> StreamingResponse | JSONResponse:
        """Handle a request that requires a specific model.

        Args:
            request: FastAPI request.
            path: Request path.
            _retry_gpu_count: Internal param for retry with more GPUs after memory error.
            _cached_body: Cached body for retry (can't read request body twice).

        Returns:
            Response from Ollama instance.
        """
        self._stats["requests_total"] += 1

        # Read request body (use cached on retry)
        body = _cached_body if _cached_body is not None else await request.body()
        model_name = self._extract_model_from_body(body)

        if not model_name:
            self._stats["requests_failed"] += 1
            return JSONResponse(
                status_code=400,
                content={"error": "model field is required"},
            )

        # Get model configuration
        model_config = self._config.get_model_config(model_name)
        base_gpu_count = model_config.gpu_count
        current_gpu_count = _retry_gpu_count or base_gpu_count

        # Inject GPU options to prevent CPU offload
        body = self._inject_gpu_options(body, current_gpu_count)

        logger.info(
            f"Request for model {model_name} requiring {current_gpu_count} GPU(s)"
            + (f" (retry from {base_gpu_count})" if _retry_gpu_count else "")
        )

        # Try to get or create instance
        instance = await self._ollama_manager.get_or_create_instance(
            model_name=model_name,
            gpu_count=current_gpu_count,
        )

        # If no instance available, handle based on config
        if not instance:
            if self._config.behavior.when_busy == "reject":
                self._stats["requests_failed"] += 1
                return JSONResponse(
                    status_code=429,
                    content={"error": "Insufficient GPU resources available"},
                    headers={"Retry-After": "60"},
                )

            # Queue the request
            queue_item = await self._request_queue.enqueue(
                model_name=model_name,
                gpu_count=current_gpu_count,
                priority=model_config.priority,
            )

            if not queue_item:
                self._stats["requests_failed"] += 1
                return JSONResponse(
                    status_code=429,
                    content={"error": "Queue is full, please try again later"},
                    headers={"Retry-After": "60"},
                )

            self._stats["requests_queued"] += 1
            logger.info(f"Request queued as {queue_item.id}")

            # Wait for resources
            success = await self._request_queue.wait_for_turn(queue_item)

            if not success:
                self._stats["requests_failed"] += 1
                return JSONResponse(
                    status_code=408,
                    content={"error": queue_item.error or "Request timed out in queue"},
                )

            # Try again after queue
            instance = await self._ollama_manager.get_or_create_instance(
                model_name=model_name,
                gpu_count=current_gpu_count,
            )

            if not instance:
                self._stats["requests_failed"] += 1
                return JSONResponse(
                    status_code=503,
                    content={"error": "Failed to allocate resources after queue"},
                )

        # Proxy the request to the instance
        response, memory_error = await self._proxy_to_instance(
            instance=instance,
            request=request,
            path=path,
            body=body,
        )

        # Handle memory error with auto-scaling
        if memory_error:
            next_gpu_count = current_gpu_count + 1
            max_gpus = self._gpu_pool.total_gpus

            if next_gpu_count <= max_gpus:
                logger.warning(
                    f"Memory error with {model_name} on {current_gpu_count} GPU(s). "
                    f"Destroying instance and retrying with {next_gpu_count} GPU(s)"
                )

                # Destroy the failed instance to free GPUs
                await self._ollama_manager.destroy_instance(instance.id)

                # Retry with more GPUs - get_or_create_instance will handle
                # eviction of inactive instances if needed
                return await self.handle_model_request(
                    request=request,
                    path=path,
                    _retry_gpu_count=next_gpu_count,
                    _cached_body=body,
                )
            else:
                logger.error(
                    f"Memory error with {model_name} on {current_gpu_count} GPU(s). "
                    f"Cannot scale: already at max GPUs ({max_gpus})"
                )
                # Return the original error response
                return response

        return response

    async def handle_management_request(
        self,
        request: Request,
        path: str,
    ) -> StreamingResponse | JSONResponse:
        """Handle management requests (tags, ps, pull, etc.).

        Args:
            request: FastAPI request.
            path: Request path.

        Returns:
            Response from Ollama or aggregated response.
        """
        self._stats["requests_total"] += 1

        # For pull/delete/create, we need a running instance
        # Use the first available or spawn a lightweight management instance (no GPU needed)
        if path in {"/api/pull", "/api/delete", "/api/copy", "/api/create", "/api/show"}:
            body = await request.body()

            # Get or create a management instance (doesn't require GPU)
            instance = await self._ollama_manager.get_or_create_management_instance()
            if not instance:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Failed to create management instance"},
                )

            response, _ = await self._proxy_to_instance(
                instance=instance,
                request=request,
                path=path,
                body=body,
            )
            return response

        # For /api/tags and /api/ps, aggregate from all instances or use any
        if path == "/api/tags":
            return await self._handle_tags_request()

        if path == "/api/ps":
            return await self._handle_ps_request()

        if path == "/api/version" or path == "/":
            return JSONResponse(
                content={"version": "ollama-loadbalancer-0.1.0"}
            )

        if path == "/v1/models":
            return await self._handle_openai_models_request()

        # Unknown endpoint
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown endpoint: {path}"},
        )

    async def handle_openai_request(
        self,
        request: Request,
        path: str,
        _retry_gpu_count: int | None = None,
        _cached_body: bytes | None = None,
    ) -> StreamingResponse | JSONResponse:
        """Handle OpenAI-compatible API requests.

        Args:
            request: FastAPI request.
            path: Request path (e.g., /v1/chat/completions).
            _retry_gpu_count: Internal param for retry with more GPUs after memory error.
            _cached_body: Cached body for retry.

        Returns:
            Response from Ollama instance (OpenAI format).
        """
        self._stats["requests_total"] += 1

        # Read request body (use cached on retry)
        body = _cached_body if _cached_body is not None else await request.body()
        model_name = self._extract_model_from_body(body)

        if not model_name:
            self._stats["requests_failed"] += 1
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "model field is required", "type": "invalid_request_error"}},
            )

        # Get model configuration
        model_config = self._config.get_model_config(model_name)
        base_gpu_count = model_config.gpu_count
        current_gpu_count = _retry_gpu_count or base_gpu_count

        logger.info(
            f"OpenAI request for model {model_name} requiring {current_gpu_count} GPU(s)"
            + (f" (retry from {base_gpu_count})" if _retry_gpu_count else "")
        )

        # Try to get or create instance
        instance = await self._ollama_manager.get_or_create_instance(
            model_name=model_name,
            gpu_count=current_gpu_count,
        )

        # If no instance available, handle based on config
        if not instance:
            if self._config.behavior.when_busy == "reject":
                self._stats["requests_failed"] += 1
                return JSONResponse(
                    status_code=429,
                    content={"error": {"message": "Insufficient GPU resources available", "type": "rate_limit_error"}},
                    headers={"Retry-After": "60"},
                )

            # Queue the request
            queue_item = await self._request_queue.enqueue(
                model_name=model_name,
                gpu_count=current_gpu_count,
                priority=model_config.priority,
            )

            if not queue_item:
                self._stats["requests_failed"] += 1
                return JSONResponse(
                    status_code=429,
                    content={"error": {"message": "Queue is full, please try again later", "type": "rate_limit_error"}},
                    headers={"Retry-After": "60"},
                )

            self._stats["requests_queued"] += 1
            logger.info(f"OpenAI request queued as {queue_item.id}")

            # Wait for resources
            success = await self._request_queue.wait_for_turn(queue_item)

            if not success:
                self._stats["requests_failed"] += 1
                return JSONResponse(
                    status_code=408,
                    content={"error": {"message": queue_item.error or "Request timed out in queue", "type": "timeout_error"}},
                )

            # Try again after queue
            instance = await self._ollama_manager.get_or_create_instance(
                model_name=model_name,
                gpu_count=current_gpu_count,
            )

            if not instance:
                self._stats["requests_failed"] += 1
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Failed to allocate resources after queue", "type": "server_error"}},
                )

        # Proxy the request to the instance (Ollama handles OpenAI format natively)
        response, memory_error = await self._proxy_to_instance(
            instance=instance,
            request=request,
            path=path,
            body=body,
        )

        # Handle memory error with auto-scaling
        if memory_error:
            next_gpu_count = current_gpu_count + 1
            max_gpus = self._gpu_pool.total_gpus

            if next_gpu_count <= max_gpus:
                logger.warning(
                    f"Memory error with {model_name} on {current_gpu_count} GPU(s). "
                    f"Destroying instance and retrying with {next_gpu_count} GPU(s)"
                )

                await self._ollama_manager.destroy_instance(instance.id)

                # Retry with more GPUs - get_or_create_instance will handle
                # eviction of inactive instances if needed
                return await self.handle_openai_request(
                    request=request,
                    path=path,
                    _retry_gpu_count=next_gpu_count,
                    _cached_body=body,
                )
            else:
                logger.error(
                    f"Memory error with {model_name} on {current_gpu_count} GPU(s). "
                    f"Cannot scale: already at max GPUs ({max_gpus})"
                )
                return response

        return response

    async def _handle_openai_models_request(self) -> JSONResponse:
        """Handle /v1/models - list available models in OpenAI format."""
        all_models = []

        instances = self._ollama_manager.get_all_instances()

        # Query each instance for its models (only READY instances)
        for instance in instances:
            if instance.state.value != "ready":
                continue
            try:
                response = await self._http_client.get(
                    f"{instance.host}/api/tags",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        model_name = model.get("name")
                        if model_name:
                            # Convert to OpenAI format
                            all_models.append({
                                "id": model_name,
                                "object": "model",
                                "created": 0,
                                "owned_by": "ollama",
                            })
            except Exception as e:
                logger.warning(f"Failed to get models from {instance.id}: {e}")

        # If no response, use management instance
        if not all_models:
            instance = await self._ollama_manager.get_or_create_management_instance()
            if instance:
                try:
                    response = await self._http_client.get(
                        f"{instance.host}/api/tags",
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        for model in data.get("models", []):
                            model_name = model.get("name")
                            if model_name:
                                all_models.append({
                                    "id": model_name,
                                    "object": "model",
                                    "created": 0,
                                    "owned_by": "ollama",
                                })
                except Exception as e:
                    logger.warning(f"Failed to get models: {e}")

        return JSONResponse(
            content={
                "object": "list",
                "data": all_models,
            }
        )

    async def _handle_tags_request(self) -> JSONResponse:
        """Handle /api/tags - list available models."""
        all_models = {}
        got_response = False

        instances = self._ollama_manager.get_all_instances()

        # Query each instance for its models (only READY instances)
        for instance in instances:
            if instance.state.value != "ready":
                continue
            try:
                response = await self._http_client.get(
                    f"{instance.host}/api/tags",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    got_response = True
                    data = response.json()
                    for model in data.get("models", []):
                        model_name = model.get("name")
                        if model_name and model_name not in all_models:
                            all_models[model_name] = model
            except Exception as e:
                logger.warning(f"Failed to get tags from {instance.id}: {e}")

        # If no response, use management instance (no GPU needed)
        if not got_response:
            instance = await self._ollama_manager.get_or_create_management_instance()
            if instance:
                try:
                    response = await self._http_client.get(
                        f"{instance.host}/api/tags",
                        timeout=10.0,
                    )
                    if response.status_code == 200:
                        return JSONResponse(content=response.json())
                except Exception as e:
                    logger.warning(f"Failed to get tags: {e}")

        return JSONResponse(
            content={"models": list(all_models.values())}
        )

    async def _handle_ps_request(self) -> JSONResponse:
        """Handle /api/ps - list running models."""
        running_models = []

        for instance in self._ollama_manager.get_all_instances():
            try:
                response = await self._http_client.get(
                    f"{instance.host}/api/ps",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        model["gpu_ids"] = instance.gpu_ids
                        model["instance_id"] = instance.id
                        running_models.append(model)
            except Exception as e:
                logger.warning(f"Failed to get ps from {instance.id}: {e}")

        return JSONResponse(
            content={"models": running_models}
        )

    async def _proxy_to_instance(
        self,
        instance: OllamaInstance,
        request: Request,
        path: str,
        body: bytes,
    ) -> tuple[StreamingResponse | JSONResponse, bool]:
        """Proxy a request to an Ollama instance.

        Args:
            instance: Target Ollama instance.
            request: Original request.
            path: Request path.
            body: Request body.

        Returns:
            Tuple of (proxied response, memory_error flag).
        """
        # Extract context size from request
        context_size = self._extract_context_size(body)
        self._ollama_manager.mark_request_start(instance.id, context_size)

        is_streaming = self._is_streaming_request(body) and path in self.MODEL_ENDPOINTS
        try:
            url = f"{instance.host}{path}"

            # Build headers (filter out hop-by-hop headers and content-length)
            # Content-Length is excluded because we may modify the body
            headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in {
                    "host", "connection", "keep-alive",
                    "transfer-encoding", "te", "trailer",
                    "upgrade", "proxy-authorization", "proxy-authenticate",
                    "content-length",  # Excluded: body may be modified
                }
            }

            if is_streaming:
                # For streaming, we check the first chunk for memory errors
                # before committing to the stream. This avoids preflight overhead.
                # Note: mark_request_end is called in stream_generator() for streaming
                return await self._proxy_streaming_with_error_check(
                    instance=instance,
                    url=url,
                    method=request.method,
                    headers=headers,
                    body=body,
                )
            else:
                return await self._proxy_non_streaming(
                    instance=instance,
                    url=url,
                    method=request.method,
                    headers=headers,
                    body=body,
                )

        except Exception as e:
            logger.error(f"Proxy error: {e}")
            self._stats["requests_failed"] += 1
            return JSONResponse(
                status_code=502,
                content={"error": f"Proxy error: {str(e)}"},
            ), False
        finally:
            # Only mark request end here for non-streaming requests
            # Streaming requests mark end in stream_generator() after stream completes
            if not is_streaming:
                self._ollama_manager.mark_request_end(instance.id)

    async def _proxy_streaming_with_error_check(
        self,
        instance: OllamaInstance,
        url: str,
        method: str,
        headers: dict,
        body: bytes,
    ) -> tuple[StreamingResponse | JSONResponse, bool]:
        """Proxy a streaming request, checking first chunk for memory errors.

        This method reads the first chunk to check for errors before committing
        to a streaming response. If memory error detected, returns error response
        with memory_error=True so auto-scaling can kick in.

        Args:
            instance: Target instance.
            url: Target URL.
            method: HTTP method.
            headers: Request headers.
            body: Request body.

        Returns:
            Tuple of (response, memory_error flag).
        """
        try:
            req = self._http_client.build_request(
                method=method,
                url=url,
                headers=headers,
                content=body,
            )

            # Send request and get response without reading body
            response = await self._http_client.send(req, stream=True)

            # Get the async iterator once and keep reference to it
            stream_iter = response.aiter_raw()

            # Read first chunk to check for errors
            first_chunk = b""
            try:
                first_chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                pass  # Empty response

            # Check if first chunk is a memory error
            if first_chunk:
                try:
                    first_data = json.loads(first_chunk.decode())
                    if self._is_memory_error(first_data):
                        logger.warning(f"Memory error in first chunk from instance {instance.id}")
                        await response.aclose()
                        return JSONResponse(
                            status_code=500,
                            content=first_data,
                        ), True
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # Not a JSON error, continue streaming

            # Create generator that yields first chunk then continues streaming
            async def stream_generator():
                try:
                    # Yield buffered first chunk
                    if first_chunk:
                        self._stats["bytes_received"] += len(first_chunk)
                        yield first_chunk

                    # Continue streaming remaining chunks from same iterator
                    async for chunk in stream_iter:
                        self._stats["bytes_received"] += len(chunk)
                        yield chunk

                    self._stats["requests_success"] += 1
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    self._stats["requests_failed"] += 1
                    yield json.dumps({"error": str(e)}).encode()
                finally:
                    await response.aclose()
                    # Mark request end here for streaming, not in _proxy_to_instance
                    self._ollama_manager.mark_request_end(instance.id)

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={"Transfer-Encoding": "chunked"},
            ), False

        except Exception as e:
            error_str = str(e)
            if self._is_memory_error(error_str):
                return JSONResponse(
                    status_code=500,
                    content={"error": error_str},
                ), True
            logger.error(f"Streaming setup error: {e}")
            self._stats["requests_failed"] += 1
            return JSONResponse(
                status_code=502,
                content={"error": str(e)},
            ), False

    async def _proxy_streaming(
        self,
        instance: OllamaInstance,
        url: str,
        method: str,
        headers: dict,
        body: bytes,
    ) -> StreamingResponse:
        """Proxy a streaming request.

        Args:
            instance: Target instance.
            url: Target URL.
            method: HTTP method.
            headers: Request headers.
            body: Request body.

        Returns:
            Streaming response.
        """
        # Store response info for headers
        response_headers = {}
        response_status = 200

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            nonlocal response_headers, response_status
            try:
                async with self._http_client.stream(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body,
                ) as response:
                    response_status = response.status_code
                    # Copy relevant headers (exclude hop-by-hop)
                    for key, value in response.headers.items():
                        if key.lower() not in {
                            "content-length",  # Don't set content-length for streaming
                            "transfer-encoding",
                            "connection",
                        }:
                            response_headers[key] = value

                    async for chunk in response.aiter_bytes():
                        self._stats["bytes_received"] += len(chunk)
                        yield chunk
                self._stats["requests_success"] += 1
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                self._stats["requests_failed"] += 1
                error_json = json.dumps({"error": str(e)})
                yield error_json.encode()

        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={"Transfer-Encoding": "chunked"},
        )

    def _is_memory_error(self, response_data: dict | str) -> bool:
        """Check if response indicates a GPU memory error.

        Args:
            response_data: Response data (dict or string).

        Returns:
            True if this is a GPU memory error.
        """
        error_msg = ""
        if isinstance(response_data, dict):
            error_msg = response_data.get("error", "")
        elif isinstance(response_data, str):
            error_msg = response_data

        return (
            "memory layout cannot be allocated" in error_msg
            or "out of memory" in error_msg.lower()
            or "CUDA out of memory" in error_msg
        )

    async def _proxy_non_streaming(
        self,
        instance: OllamaInstance,
        url: str,
        method: str,
        headers: dict,
        body: bytes,
    ) -> tuple[JSONResponse, bool]:
        """Proxy a non-streaming request.

        Args:
            instance: Target instance.
            url: Target URL.
            method: HTTP method.
            headers: Request headers.
            body: Request body.

        Returns:
            Tuple of (JSON response, memory_error flag).
        """
        try:
            response = await self._http_client.request(
                method=method,
                url=url,
                headers=headers,
                content=body,
            )

            self._stats["bytes_received"] += len(response.content)

            # Try to parse as JSON
            try:
                response_data = response.json()

                # Check for memory error
                if response.status_code >= 400 and self._is_memory_error(response_data):
                    logger.warning(f"Memory error detected from instance {instance.id}")
                    return JSONResponse(
                        status_code=response.status_code,
                        content=response_data,
                    ), True

                self._stats["requests_success"] += 1
                return JSONResponse(
                    status_code=response.status_code,
                    content=response_data,
                ), False
            except json.JSONDecodeError:
                self._stats["requests_success"] += 1
                return JSONResponse(
                    status_code=response.status_code,
                    content={"raw": response.text},
                ), False

        except Exception as e:
            self._stats["requests_failed"] += 1
            return JSONResponse(
                status_code=502,
                content={"error": str(e)},
            ), False

    def get_stats(self) -> dict:
        """Get proxy statistics."""
        return dict(self._stats)
