"""Tests for config loading."""

from pathlib import Path

import pytest
import yaml

from src.config import load_config


def test_load_config_valid(tmp_path, sample_config):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    result = load_config(config_path)
    assert result["seasons"]["train_start"] == 2003


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_invalid_seasons(tmp_path, sample_config):
    sample_config["seasons"]["train_start"] = 2026
    sample_config["seasons"]["train_end"] = 2003
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    with pytest.raises(ValueError, match="train_start must be before"):
        load_config(config_path)
