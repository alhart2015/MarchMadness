"""Load and validate config.yaml."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config from YAML file."""
    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    """Validate required config keys exist."""
    required_sections = ["data", "seasons", "efficiency", "model"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    if config["seasons"]["train_start"] >= config["seasons"]["train_end"]:
        raise ValueError("train_start must be before train_end")
