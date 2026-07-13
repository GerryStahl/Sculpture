"""Tests for image I/O utilities."""

from pathlib import Path

import numpy as np
import pytest

from sculpture.io.image_io import collect_images, load_image, save_image


PHOTOS_DIR = Path(__file__).parents[1] / "photos"


def test_collect_images_finds_heic():
    """Should find the HEIC sample image."""
    images = collect_images(PHOTOS_DIR)
    assert len(images) >= 1
    assert any(p.suffix.lower() == ".heic" for p in images)


def test_load_image_heic():
    """Should load the HEIC sample image as a uint8 RGB array."""
    imgs = [p for p in collect_images(PHOTOS_DIR) if p.suffix.lower() == ".heic"]
    assert imgs, "No HEIC image found in photos/"
    img = load_image(imgs[0])
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_save_load_roundtrip(tmp_path):
    """Save a synthetic image and reload it."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = tmp_path / "test.png"
    save_image(arr, out)
    loaded = load_image(out)
    assert loaded.shape == arr.shape
    assert loaded.dtype == arr.dtype


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_image(Path("/nonexistent/image.jpg"))
