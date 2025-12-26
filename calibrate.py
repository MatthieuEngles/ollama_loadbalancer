#!/usr/bin/env python3
"""
Calibration script for Ollama Load Balancer.

This script measures actual VRAM usage for different models to determine
the optimal GPU allocation configuration.

Usage:
    python calibrate.py [--models MODEL1,MODEL2,...] [--output config.yaml]
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class CalibrationResult:
    """Result of calibrating a single model."""
    model_name: str
    success: bool
    vram_used_mb: int = 0
    vram_per_gpu_mb: int = 0
    gpus_recommended: int = 1
    error: Optional[str] = None
    load_time_seconds: float = 0


class OllamaCalibrator:
    """Calibrates VRAM usage for Ollama models."""

    GPU_VRAM_MB = 16384  # 16GB per A4000
    VRAM_BUFFER_MB = 1024  # 1GB buffer for safety
    USABLE_VRAM_MB = GPU_VRAM_MB - VRAM_BUFFER_MB  # 15GB usable

    def __init__(self, models_file: str = "models_available.json"):
        self.models_file = models_file
        self.results: list[CalibrationResult] = []
        self.process: Optional[subprocess.Popen] = None
        self.http_client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Initialize HTTP client."""
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def stop(self):
        """Cleanup."""
        if self.http_client:
            await self.http_client.aclose()
        self._kill_ollama()

    def _get_gpu_memory_usage(self) -> dict[int, int]:
        """Get current VRAM usage per GPU in MB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            usage = {}
            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) == 2:
                    gpu_id = int(parts[0].strip())
                    mem_mb = int(parts[1].strip())
                    usage[gpu_id] = mem_mb
            return usage
        except Exception as e:
            print(f"Error getting GPU memory: {e}")
            return {}

    def _spawn_ollama(self, gpu_ids: list[int], port: int = 11500) -> bool:
        """Spawn an Ollama instance on specific GPUs."""
        self._kill_ollama()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
        env["OLLAMA_KEEP_ALIVE"] = "-1"
        env["OLLAMA_MODELS"] = "/usr/share/ollama/.ollama/models"
        env["OLLAMA_GPU_OVERHEAD"] = "0"
        env["OLLAMA_MAX_LOADED_MODELS"] = "1"
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        env["OLLAMA_LLM_LIBRARY"] = "cuda_v12"
        env["OLLAMA_NUM_GPU"] = "999"

        try:
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            time.sleep(2)  # Give it time to start
            return self.process.poll() is None
        except Exception as e:
            print(f"Error spawning Ollama: {e}")
            return False

    def _kill_ollama(self):
        """Kill the current Ollama process."""
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()
                except:
                    pass
            self.process = None

        # Also kill any stray ollama processes
        subprocess.run(["pkill", "-9", "ollama"], capture_output=True)
        time.sleep(1)

    async def _wait_for_ollama(self, port: int = 11500, timeout: int = 30) -> bool:
        """Wait for Ollama to be ready."""
        url = f"http://127.0.0.1:{port}/api/tags"
        elapsed = 0
        while elapsed < timeout:
            try:
                response = await self.http_client.get(url, timeout=5.0)
                if response.status_code == 200:
                    return True
            except:
                pass
            await asyncio.sleep(1)
            elapsed += 1
        return False

    async def _load_model(self, model_name: str, port: int = 11500) -> tuple[bool, float]:
        """Load a model and return success status and load time."""
        url = f"http://127.0.0.1:{port}/api/generate"
        start_time = time.time()

        try:
            # Send a minimal request to load the model
            response = await self.http_client.post(
                url,
                json={
                    "model": model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {
                        "num_gpu": 999,
                        "use_mmap": False,
                        "num_predict": 1,
                    }
                },
                timeout=300.0,  # 5 minutes for large models
            )
            load_time = time.time() - start_time

            if response.status_code == 200:
                return True, load_time
            else:
                print(f"  Error loading model: {response.text}")
                return False, load_time

        except Exception as e:
            load_time = time.time() - start_time
            print(f"  Exception loading model: {e}")
            return False, load_time

    async def calibrate_model(
        self,
        model_name: str,
        gpu_ids: list[int],
    ) -> CalibrationResult:
        """Calibrate a single model on specific GPUs."""
        print(f"\n{'='*60}")
        print(f"Calibrating: {model_name}")
        print(f"GPUs: {gpu_ids}")
        print(f"{'='*60}")

        # Get baseline VRAM
        baseline = self._get_gpu_memory_usage()
        baseline_total = sum(baseline.get(g, 0) for g in gpu_ids)
        print(f"Baseline VRAM: {baseline_total} MB")

        # Spawn Ollama
        if not self._spawn_ollama(gpu_ids):
            return CalibrationResult(
                model_name=model_name,
                success=False,
                error="Failed to spawn Ollama",
            )

        # Wait for Ollama to be ready
        if not await self._wait_for_ollama():
            self._kill_ollama()
            return CalibrationResult(
                model_name=model_name,
                success=False,
                error="Ollama failed to start",
            )

        # Load the model
        success, load_time = await self._load_model(model_name)
        if not success:
            self._kill_ollama()
            return CalibrationResult(
                model_name=model_name,
                success=False,
                error="Failed to load model (may not be downloaded)",
                load_time_seconds=load_time,
            )

        # Give it a moment to stabilize
        await asyncio.sleep(2)

        # Measure VRAM
        current = self._get_gpu_memory_usage()
        current_total = sum(current.get(g, 0) for g in gpu_ids)
        vram_used = current_total - baseline_total

        print(f"VRAM after load: {current_total} MB")
        print(f"VRAM used by model: {vram_used} MB")
        print(f"Load time: {load_time:.1f}s")

        # Calculate recommended GPUs
        vram_per_gpu = vram_used / len(gpu_ids)
        if vram_used <= self.USABLE_VRAM_MB:
            gpus_recommended = 1
        elif vram_used <= self.USABLE_VRAM_MB * 2:
            gpus_recommended = 2
        else:
            gpus_recommended = 3

        # Kill Ollama
        self._kill_ollama()

        return CalibrationResult(
            model_name=model_name,
            success=True,
            vram_used_mb=vram_used,
            vram_per_gpu_mb=int(vram_per_gpu),
            gpus_recommended=gpus_recommended,
            load_time_seconds=load_time,
        )

    def get_installed_models(self) -> list[str]:
        """Get list of installed models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
            )
            models = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
        except Exception as e:
            print(f"Error getting installed models: {e}")
            return []

    async def run_calibration(self, models: Optional[list[str]] = None) -> list[CalibrationResult]:
        """Run calibration for specified or all installed models."""
        if models is None:
            models = self.get_installed_models()

        if not models:
            print("No models to calibrate!")
            return []

        print(f"Models to calibrate: {models}")
        print(f"Total GPUs available: 3 x A4000 (16GB each)")
        print(f"Usable VRAM per GPU: {self.USABLE_VRAM_MB} MB")

        for model in models:
            # First try with 1 GPU
            result = await self.calibrate_model(model, [0])

            if not result.success:
                # Model might not be installed, skip
                print(f"  Skipping {model}: {result.error}")
                self.results.append(result)
                continue

            # If VRAM exceeds single GPU, try with more
            if result.vram_used_mb > self.USABLE_VRAM_MB:
                print(f"  Model needs more than 1 GPU, testing with 2...")
                result = await self.calibrate_model(model, [0, 1])

                if result.vram_used_mb > self.USABLE_VRAM_MB * 2:
                    print(f"  Model needs more than 2 GPUs, testing with 3...")
                    result = await self.calibrate_model(model, [0, 1, 2])

            self.results.append(result)

        return self.results

    def generate_config(self, output_file: str = "config.yaml"):
        """Generate config.yaml from calibration results."""
        models_config = []

        for result in self.results:
            if result.success:
                models_config.append({
                    "pattern": result.model_name.split(":")[0] + ":*" if ":" in result.model_name else result.model_name,
                    "exact_name": result.model_name,
                    "gpu_count": result.gpus_recommended,
                    "priority": 5,
                    "vram_mb": result.vram_used_mb,
                })

        # Group by pattern
        patterns_seen = {}
        for m in models_config:
            pattern = m["pattern"]
            if pattern not in patterns_seen or m["gpu_count"] > patterns_seen[pattern]["gpu_count"]:
                patterns_seen[pattern] = m

        config_content = f"""# Ollama Load Balancer Configuration
# Generated by calibration script on {time.strftime('%Y-%m-%d %H:%M:%S')}

server:
  host: "0.0.0.0"
  port: 11434
  log_level: "INFO"
  log_format: "text"

# GPU configuration
gpu:
  ids: [0, 1, 2]  # 3x NVIDIA A4000 16GB

# Model configurations (from calibration)
models:
"""
        for pattern, data in sorted(patterns_seen.items()):
            config_content += f"""  - pattern: "{pattern}"
    gpu_count: {data['gpu_count']}
    priority: 5
    # Measured VRAM: {data['vram_mb']} MB
"""

        config_content += """
  # Default for unknown models
  - pattern: "*"
    gpu_count: 1
    priority: 1

# Behavior configuration
behavior:
  when_busy: "queue"  # queue or reject
  queue_timeout: 300  # 5 minutes
  max_queue_size: 100
  instance_ttl: 300   # 5 minutes idle before unload
  startup_timeout: 120  # 2 minutes to start
  health_check_interval: 30
"""

        with open(output_file, "w") as f:
            f.write(config_content)

        print(f"\nConfiguration written to {output_file}")

    def print_summary(self):
        """Print calibration summary."""
        print("\n" + "=" * 70)
        print("CALIBRATION SUMMARY")
        print("=" * 70)
        print(f"{'Model':<30} {'VRAM (MB)':<12} {'GPUs':<6} {'Load Time':<12} {'Status'}")
        print("-" * 70)

        for result in self.results:
            status = "OK" if result.success else f"FAILED: {result.error}"
            vram = str(result.vram_used_mb) if result.success else "-"
            gpus = str(result.gpus_recommended) if result.success else "-"
            load_time = f"{result.load_time_seconds:.1f}s" if result.success else "-"
            print(f"{result.model_name:<30} {vram:<12} {gpus:<6} {load_time:<12} {status}")

        print("=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Calibrate Ollama models for GPU allocation")
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to calibrate (default: all installed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config.yaml",
        help="Output config file (default: config.yaml)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Just list installed models and exit",
    )
    args = parser.parse_args()

    calibrator = OllamaCalibrator()

    if args.list:
        models = calibrator.get_installed_models()
        print("Installed models:")
        for m in models:
            print(f"  - {m}")
        return

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    await calibrator.start()

    try:
        await calibrator.run_calibration(models)
        calibrator.print_summary()
        calibrator.generate_config(args.output)
    finally:
        await calibrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
