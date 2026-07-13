"""I/O helpers: HEIC conversion, image loading, metadata."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Register HEIF/HEIC opener if pillow-heif is available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False
    logger.warning("pillow-heif not installed – HEIC files will not be readable.")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
               ".heic", ".heif"}


def load_image(path: Path | str) -> np.ndarray:
    """Load an image from disk as an RGB uint8 numpy array.

    Supports JPEG, PNG, TIFF, HEIC/HEIF, and other PIL-supported formats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def save_image(array: np.ndarray, path: Path | str, quality: int = 95) -> None:
    """Save a uint8 RGB numpy array as an image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=quality)
    logger.debug("Saved image → %s", path)


def collect_images(directory: Path | str, exts: set[str] | None = None) -> list[Path]:
    """Return sorted list of image paths in *directory*."""
    directory = Path(directory)
    exts = exts or _IMAGE_EXTS
    paths = sorted(p for p in directory.iterdir()
                   if p.suffix.lower() in exts)
    logger.info("Found %d image(s) in %s", len(paths), directory)
    return paths
