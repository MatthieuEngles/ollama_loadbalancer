"""Configuration loading and validation for Ollama Load Balancer."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class GPUConfig(BaseModel):
    """Configuration for a single GPU."""
    id: int


class ModelConfig(BaseModel):
    """Configuration for a model's resource requirements."""
    gpu_count: int = Field(ge=1, le=8)
    priority: Literal["low", "normal", "high"] = "normal"


class BehaviorConfig(BaseModel):
    """Behavior configuration for the load balancer."""
    when_busy: Literal["queue", "reject"] = "queue"
    queue_timeout: int = Field(default=300, ge=1)
    instance_ttl: int = Field(default=60, ge=10)
    max_queue_size: int = Field(default=10, ge=1)
    health_check_interval: int = Field(default=5, ge=1)
    startup_timeout: int = Field(default=120, ge=10)


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = Field(default=11434, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"


class Config(BaseModel):
    """Main configuration for Ollama Load Balancer."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    gpu_pool: list[GPUConfig] = Field(default_factory=list)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)

    @field_validator("gpu_pool", mode="before")
    @classmethod
    def validate_gpu_pool(cls, v):
        """Convert list of dicts to list of GPUConfig."""
        if not v:
            # Auto-detect GPUs using nvidia-smi
            detected_gpus = cls._detect_gpus()
            if detected_gpus:
                logger.info(f"Auto-detected {len(detected_gpus)} GPU(s): {detected_gpus}")
                return [{"id": gpu_id} for gpu_id in detected_gpus]
            else:
                logger.warning("No GPUs detected, using default configuration with 1 GPU")
                return [{"id": 0}]
        return v

    @staticmethod
    def _detect_gpus() -> list[int]:
        """Detect available NVIDIA GPUs using nvidia-smi.

        Returns:
            List of GPU IDs.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            gpu_ids = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
            return gpu_ids
        except subprocess.CalledProcessError as e:
            logger.warning(f"nvidia-smi command failed: {e}")
            return []
        except FileNotFoundError:
            logger.warning("nvidia-smi not found - NVIDIA drivers may not be installed")
            return []
        except Exception as e:
            logger.warning(f"Failed to detect GPUs: {e}")
            return []

    @field_validator("models", mode="before")
    @classmethod
    def validate_models(cls, v):
        """Ensure default model config exists."""
        if v is None:
            v = {}
        if "default" not in v:
            v["default"] = {"gpu_count": 1, "priority": "normal"}
        return v

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model, falling back to default."""
        # Try exact match first
        if model_name in self.models:
            return self.models[model_name]

        # Try matching base name (without tag)
        base_name = model_name.split(":")[0] if ":" in model_name else model_name
        for key, config in self.models.items():
            if key == "default":
                continue
            key_base = key.split(":")[0] if ":" in key else key
            if key_base == base_name:
                return config

        # Fall back to default
        return self.models.get("default", ModelConfig(gpu_count=1))

    @property
    def gpu_ids(self) -> list[int]:
        """Get list of all GPU IDs."""
        return [gpu.id for gpu in self.gpu_pool]

    @property
    def total_gpus(self) -> int:
        """Get total number of GPUs."""
        return len(self.gpu_pool)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml
                    in current directory or uses defaults.

    Returns:
        Config object with validated settings.
    """
    if config_path is None:
        config_path = Path(os.environ.get("OLLAMA_LB_CONFIG", "config.yaml"))
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    return Config(**data)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
