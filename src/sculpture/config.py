"""Configuration loading and validation using Pydantic v2 + PyYAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# ── Sub-models ────────────────────────────────────────────────────────────────

class PathsConfig(BaseModel):
    photos_dir: Path = Path("photos/")
    data_raw: Path = Path("data/raw/")
    data_processed: Path = Path("data/processed/")
    calibration_dir: Path = Path("data/calibration/")
    output_dir: Path = Path("data/output/")


class PreprocessingConfig(BaseModel):
    max_size: int = 2048
    bg_removal: Literal["rembg", "grabcut", "none"] = "rembg"
    denoise_ksize: int = 0


class CalibrationConfig(BaseModel):
    board_cols: int = 9
    board_rows: int = 6
    square_size_mm: float = 25.0
    calib_file: Path = Path("data/calibration/camera_intrinsics.json")


class ReconstructionConfig(BaseModel):
    method: Literal["colmap", "opencv_sfm", "open3d"] = "open3d"
    colmap_bin: str = "colmap"
    use_depth_prior: bool = False


class MeshingConfig(BaseModel):
    method: Literal["ball_pivot", "poisson", "alpha_shape"] = "poisson"
    poisson_depth: int = 9
    min_component_frac: float = 0.01
    simplify_faces: int = 50_000


class WireframeConfig(BaseModel):
    feature_angle_deg: float = 30.0
    min_edge_frac: float = 0.005
    export_format: Literal["obj", "svg", "json_graph"] = "obj"


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Path = Path("data/output/pipeline.log")


# ── Root config ───────────────────────────────────────────────────────────────

class SculptureConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    meshing: MeshingConfig = Field(default_factory=MeshingConfig)
    wireframe: WireframeConfig = Field(default_factory=WireframeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = Path(__file__).parents[3] / "config" / "default.yaml"


def load_config(config_path: Path | str | None = None) -> SculptureConfig:
    """Load YAML config, merging with defaults.

    Args:
        config_path: Path to a YAML file.  Uses config/default.yaml if None.

    Returns:
        Validated SculptureConfig instance.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if path.exists():
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        raw = {}
    return SculptureConfig(**raw)
