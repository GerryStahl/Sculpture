"""Tests for image preprocessing."""

import numpy as np
import pytest

from sculpture.config import PreprocessingConfig
from sculpture.preprocessing import denoise, remove_background_grabcut, resize_to_max


def _synthetic_rgb(h=256, w=256) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_resize_to_max_downscales():
    img = _synthetic_rgb(1024, 2048)
    out = resize_to_max(img, 512)
    assert max(out.shape[:2]) <= 512


def test_resize_to_max_no_upscale():
    img = _synthetic_rgb(64, 64)
    out = resize_to_max(img, 512)
    assert out.shape == img.shape


def test_resize_to_max_zero_skips():
    img = _synthetic_rgb(1024, 1024)
    out = resize_to_max(img, 0)
    assert out.shape == img.shape


def test_denoise_returns_same_shape():
    img = _synthetic_rgb()
    out = denoise(img, ksize=5)
    assert out.shape == img.shape


def test_denoise_ksize_zero_is_identity():
    img = _synthetic_rgb()
    out = denoise(img, ksize=0)
    assert np.array_equal(out, img)


def test_grabcut_returns_rgba():
    img = _synthetic_rgb(128, 128)
    out = remove_background_grabcut(img)
    assert out.shape[2] == 4
