"""Image preprocessing: resize, denoise, background removal."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from sculpture.config import PreprocessingConfig

# Locate mask_subject binary relative to project root (parents[2] from src/sculpture/preprocessing.py)
_MASK_TOOL = Path(__file__).resolve().parents[2] / "tools" / "mask_subject" / "mask_subject"

logger = logging.getLogger(__name__)


def resize_to_max(image: np.ndarray, max_size: int) -> np.ndarray:
    """Resize *image* so its longest edge ≤ *max_size*, preserving aspect ratio."""
    if max_size <= 0:
        return image
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized %dx%d → %dx%d", w, h, new_w, new_h)
    return resized


def denoise(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply Gaussian denoising (ksize must be odd and > 0)."""
    if ksize <= 0:
        return image
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def remove_background_grabcut(image: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Remove background with GrabCut, returning an RGBA image."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    margin_x, margin_y = max(1, w // 20), max(1, h // 20)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)
    rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = fg_mask
    logger.debug("GrabCut background removal applied.")
    return rgba


def remove_background_apple_vision(image: np.ndarray) -> np.ndarray:
    """Remove background using Apple Vision VNGenerateForegroundInstanceMaskRequest.

    Returns an RGBA numpy array.
    """
    if not _MASK_TOOL.exists():
        raise RuntimeError(
            f"mask_subject binary not found at {_MASK_TOOL}. "
            "Apple Vision masking is required. Build tools/mask_subject first."
        )

    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "input.jpg"
        out = Path(tmp) / "masked.png"
        cv2.imwrite(str(inp), image)
        result = subprocess.run(
            [str(_MASK_TOOL), str(inp), str(out)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"mask_subject failed: {result.stderr.strip()}")
        masked = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
        if masked is None:
            raise RuntimeError("mask_subject output unreadable.")

    logger.debug("Apple Vision subject masking applied.")
    return masked


def remove_background_rembg(image: np.ndarray) -> np.ndarray:
    """Remove background using rembg (U2-Net); returns RGBA numpy array."""
    try:
        from PIL import Image
        from rembg import remove  # type: ignore[import]
        pil_in = Image.fromarray(image)
        pil_out = remove(pil_in)
        return np.asarray(pil_out, dtype=np.uint8)
    except ImportError:
        logger.warning("rembg not installed, falling back to GrabCut.")
        return remove_background_grabcut(image)


def preprocess_image(
    image: np.ndarray,
    cfg: PreprocessingConfig,
) -> np.ndarray:
    """Full preprocessing pipeline for a single image.

    Steps:
        1. Resize to cfg.max_size
        2. Denoise (if cfg.denoise_ksize > 0)
        3. Background removal (Apple Vision only)

    Returns:
        Preprocessed image (RGB or RGBA uint8).
    """
    image = resize_to_max(image, cfg.max_size)
    image = denoise(image, cfg.denoise_ksize)

    image = remove_background_apple_vision(image)

    return image
