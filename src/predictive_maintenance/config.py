"""
Configuration loader and settings manager.
Reads from configs/config.yaml, configs/model_config.yaml, and environment variables.
"""

from pathlib import Path
from typing import Any, Dict, List
import yaml


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Loads a YAML file safely."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings:
    """Application and pipeline settings."""
    def __init__(self):
        self.root_dir: Path = get_project_root()
        self.config_yaml_path: Path = self.root_dir / "configs" / "config.yaml"
        self.model_config_yaml_path: Path = self.root_dir / "configs" / "model_config.yaml"
        self._config = load_yaml(self.config_yaml_path)
        self._model_config = load_yaml(self.model_config_yaml_path)

    @property
    def system(self) -> Dict[str, Any]:
        return self._config.get("system", {})

    @property
    def dataset(self) -> Dict[str, Any]:
        return self._config.get("dataset", {})

    @property
    def signal_processing(self) -> Dict[str, Any]:
        return self._config.get("signal_processing", {})

    @property
    def operational_status(self) -> Dict[str, Any]:
        return self._config.get("operational_status", {})

    @property
    def storage(self) -> Dict[str, Any]:
        return self._config.get("storage", {})

    @property
    def sensor_source(self) -> Dict[str, Any]:
        return self._config.get("sensor_source", {})

    @property
    def model_config(self) -> Dict[str, Any]:
        return self._model_config

    @property
    def supported_classes(self) -> List[str]:
        return self.dataset.get("supported_classes", ["Healthy", "BrokenTooth"])


# Global singleton settings instance
settings = Settings()
