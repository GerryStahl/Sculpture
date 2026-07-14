"""Camera calibration from checkerboard images."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from sculpture.config import CalibrationConfig

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Pinhole camera model intrinsics."""

    camera_matrix: list[list[float]]  # 3×3
    dist_coeffs: list[float]          # k1,k2,p1,p2[,k3]
    rms_error: float
    image_size: tuple[int, int]       # (width, height)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(asdict(self), fh, indent=2)
        logger.info("Camera intrinsics saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> CameraIntrinsics:
        with path.open() as fh:
            data = json.load(fh)
        return cls(**data)

    @property
    def K(self) -> np.ndarray:
        """Return 3×3 camera matrix as numpy array."""
        return np.array(self.camera_matrix, dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        """Return distortion coefficients as numpy array."""
        return np.array(self.dist_coeffs, dtype=np.float64)


def calibrate_from_images(
    images: list[np.ndarray],
    cfg: CalibrationConfig,
) -> CameraIntrinsics:
    """Estimate camera intrinsics from checkerboard calibration images.

    Args:
        images: List of grayscale or RGB images showing a checkerboard.
        cfg:    CalibrationConfig with board dimensions and square size.

    Returns:
        CameraIntrinsics fitted to the supplied images.

    Raises:
        ValueError: If fewer than 3 usable board detections are found.
    """
    inner = (cfg.board_cols, cfg.board_rows)
    sq = cfg.square_size_mm

    # 3-D object points for one board
    objp = np.zeros((cfg.board_cols * cfg.board_rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cfg.board_cols, 0:cfg.board_rows].T.reshape(-1, 2) * sq

    obj_pts, img_pts = [], []
    image_size: tuple[int, int] | None = None

    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, inner, None)
        if not found:
            logger.debug("Board not found in calibration image %d", idx)
            continue

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners)
        logger.debug("Board detected in calibration image %d", idx)

    if len(obj_pts) < 3:
        raise ValueError(
            f"Need at least 3 board detections for calibration, got {len(obj_pts)}."
        )

    assert image_size is not None
    rms, K, D, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, image_size, None, None
    )
    logger.info("Calibration RMS reprojection error: %.4f px", rms)

    return CameraIntrinsics(
        camera_matrix=K.tolist(),
        dist_coeffs=D.flatten().tolist(),
        rms_error=float(rms),
        image_size=image_size,
    )


def load_or_calibrate(
    calib_images: list[np.ndarray],
    cfg: CalibrationConfig,
) -> CameraIntrinsics:
    """Return stored intrinsics if available; else calibrate and save."""
    if cfg.calib_file.exists():
        logger.info("Loading existing intrinsics from %s", cfg.calib_file)
        return CameraIntrinsics.load(cfg.calib_file)

    logger.info("Calibrating camera from %d image(s)…", len(calib_images))
    intrinsics = calibrate_from_images(calib_images, cfg)
    intrinsics.save(cfg.calib_file)
    return intrinsics
