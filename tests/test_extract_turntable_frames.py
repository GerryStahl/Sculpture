"""Tests for standardized turntable frame extraction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_turntable_frames.py"
_spec = importlib.util.spec_from_file_location("extract_turntable_frames", _SCRIPT_PATH)
_extract_module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_extract_module)

compute_frame_indices = _extract_module.compute_frame_indices
extract_frames = _extract_module.extract_frames
NUM_FRAMES = _extract_module.NUM_FRAMES


def _write_synthetic_video(video_path: Path, frame_count: int = 120) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("Synthetic video writer could not be opened")
    for index in range(frame_count):
        frame = np.full((64, 64, 3), index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_compute_frame_indices_returns_fixed_count():
    indices = compute_frame_indices(120)

    assert len(indices) == NUM_FRAMES
    assert indices == sorted(indices)
    assert indices[0] >= 0
    assert indices[-1] < 120


def test_extract_frames_writes_standardized_manifest(tmp_path):
    video_path = tmp_path / "turntable.mp4"
    out_dir = tmp_path / "frames"
    _write_synthetic_video(video_path)

    manifest = extract_frames(video_path, out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    saved_jpgs = sorted(out_dir.glob("*.jpg"))
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest) == NUM_FRAMES
    assert len(saved_jpgs) == NUM_FRAMES
    assert len(persisted_manifest) == NUM_FRAMES
    assert [entry["index"] for entry in manifest] == list(range(1, NUM_FRAMES + 1))
    assert all(Path(entry["file"]).exists() for entry in manifest)
