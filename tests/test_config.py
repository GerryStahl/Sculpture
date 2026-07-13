"""Tests for config loading."""

from pathlib import Path

import pytest

from sculpture.config import SculptureConfig, load_config


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, SculptureConfig)
    assert cfg.preprocessing.max_size == 2048
    assert cfg.meshing.method == "poisson"
    assert cfg.wireframe.feature_angle_deg == 30.0


def test_load_missing_config_returns_defaults(tmp_path):
    missing = tmp_path / "nonexistent.yaml"
    cfg = load_config(missing)
    assert isinstance(cfg, SculptureConfig)


def test_load_partial_config(tmp_path):
    yaml_content = "preprocessing:\n  max_size: 512\n"
    cfg_file = tmp_path / "partial.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(cfg_file)
    assert cfg.preprocessing.max_size == 512
    # Other fields should be defaults
    assert cfg.meshing.poisson_depth == 9
