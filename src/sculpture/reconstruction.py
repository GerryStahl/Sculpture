"""Multi-view 3-D reconstruction → point cloud."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from sculpture.config import ReconstructionConfig

logger = logging.getLogger(__name__)


def _alpha_mask_to_rgb(images: list[np.ndarray]) -> list[np.ndarray]:
    """Strip alpha channel; background pixels become white."""
    out = []
    for img in images:
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4] / 255.0
            rgb = img[:, :, :3]
            white = np.ones_like(rgb, dtype=np.float32) * 255
            blended = (rgb.astype(np.float32) * alpha + white * (1 - alpha)).astype(np.uint8)
            out.append(blended)
        else:
            out.append(img)
    return out


def reconstruct_open3d(
    images: list[np.ndarray],
    cfg: ReconstructionConfig,
    output_dir: Path,
) -> o3d.geometry.PointCloud:
    """Estimate a point cloud from multi-view images using Open3D RGB-D integration.

    NOTE: Without a true depth sensor this falls back to a feature-based sparse
    reconstruction heuristic.  For production, pipe images through COLMAP first
    to obtain accurate poses.

    Args:
        images:     List of RGB (or RGBA) uint8 numpy arrays.
        cfg:        ReconstructionConfig.
        output_dir: Directory to save intermediate/output files.

    Returns:
        Open3D PointCloud.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _alpha_mask_to_rgb(images)

    if len(images) < 2:
        logger.warning(
            "Only %d image(s) provided – single-image pseudo point cloud via "
            "depth estimation placeholder.", len(images)
        )
        return _single_image_pseudo_pcd(images[0], output_dir)

    # Feature-based sparse point cloud via OpenCV + Open3D
    import cv2
    from PIL import Image as PILImage

    orb = cv2.ORB_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    all_pts3d: list[np.ndarray] = []
    ref_gray = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)

    for img in images[1:]:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or des_ref is None:
            continue
        matches = bf.match(des_ref, des)
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        if len(matches) < 8:
            continue

        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches])
        pts_cur = np.float32([kp[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, 5.0)
        if H is None:
            continue

        inliers = pts_ref[mask.ravel() == 1]
        # Lift 2-D inlier points to rough 3-D (z=0 plane) – placeholder depth
        z = np.ones((len(inliers), 1), dtype=np.float32)
        pts3d = np.hstack([inliers / 1000.0, z])  # normalise to meter scale
        all_pts3d.append(pts3d)

    if not all_pts3d:
        logger.warning("No feature matches found; generating placeholder PCD.")
        return _single_image_pseudo_pcd(images[0], output_dir)

    pts = np.vstack(all_pts3d)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.02, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)

    out_path = output_dir / "point_cloud.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    logger.info("Point cloud saved → %s  (%d pts)", out_path, len(pcd.points))
    return pcd


def _single_image_pseudo_pcd(
    image: np.ndarray, output_dir: Path
) -> o3d.geometry.PointCloud:
    """Generate a placeholder point cloud from a single image (silhouette depth proxy)."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        logger.warning("No edges detected; PCD will be empty.")
        return o3d.geometry.PointCloud()

    h, w = gray.shape
    # Project edge pixels onto a unit sphere surface (crude but visualisable)
    nx = (xs / w - 0.5) * 2
    ny = (ys / h - 0.5) * 2
    r = np.sqrt(np.maximum(0, 1 - nx**2 - ny**2))
    pts = np.stack([nx, -ny, r], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals()

    out_path = output_dir / "point_cloud_single.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    logger.info("Single-image pseudo PCD saved → %s  (%d pts)", out_path, len(pcd.points))
    return pcd


def reconstruct(
    images: list[np.ndarray],
    cfg: ReconstructionConfig,
    output_dir: Path,
) -> o3d.geometry.PointCloud:
    """Entry point for reconstruction; dispatches to configured method."""
    if cfg.method in ("open3d", "opencv_sfm"):
        return reconstruct_open3d(images, cfg, output_dir)
    if cfg.method == "colmap":
        raise NotImplementedError(
            "COLMAP integration: run `colmap automatic_reconstructor` on your "
            "images, then load the resulting sparse/0/points3D.bin with Open3D."
        )
    raise ValueError(f"Unknown reconstruction method: {cfg.method!r}")
