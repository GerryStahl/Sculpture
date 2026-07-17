"""Multi-view 3-D reconstruction → point cloud.

Supported methods
-----------------
colmap      Full Structure-from-Motion + optional dense MVS via COLMAP CLI.
            Produces accurate camera poses and a high-quality point cloud.
            Requires COLMAP to be installed (brew install colmap).
open3d      Legacy ORB-homography heuristic (fast but low quality).
            Kept as a fallback when COLMAP is not available.
"""

from __future__ import annotations

import logging
import shutil
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from sculpture.config import ReconstructionConfig

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess command, streaming output to the logger."""
    logger.info("  [%s] %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.debug("    %s", line)
    if result.returncode != 0:
        logger.error("[%s] stderr:\n%s", label, result.stderr[-3000:])
        raise RuntimeError(f"COLMAP step '{label}' failed (exit {result.returncode})")


def _run_with_gpu_fallback(
    cmd_without_gpu_flag: list[str],
    gpu_flag: str,
    label: str,
) -> None:
    """Run a COLMAP command with GPU first, then CPU fallback if needed."""
    try:
        _run(cmd_without_gpu_flag + [gpu_flag, "1"], f"{label} (gpu)")
        logger.info("COLMAP: %s used GPU", label)
    except RuntimeError as exc:
        logger.warning("COLMAP: %s GPU path failed (%s); retrying on CPU", label, exc)
        _run(cmd_without_gpu_flag + [gpu_flag, "0"], f"{label} (cpu)")
        logger.info("COLMAP: %s used CPU fallback", label)


def _which_optional(binary: str) -> str | None:
    """Return resolved executable path if available, else ``None``."""
    return shutil.which(binary) or (binary if Path(binary).exists() else None)


def _read_colmap_points3d_bin(path: Path) -> np.ndarray:
    """Parse a COLMAP binary points3D.bin file → Nx3 float64 array."""
    pts = []
    with open(path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_pts):
            # point3D_id (uint64) + xyz (3×float64) + rgb (3×uint8) +
            # error (float64) + track_length (uint64) + track (2×uint32 each)
            _pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            _rgb = f.read(3)
            _err = f.read(8)
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)   # skip track (image_id, point2D_idx pairs)
            pts.append(xyz)
    return np.array(pts, dtype=np.float64)


def _read_colmap_points3d_txt(path: Path) -> np.ndarray:
    """Parse a COLMAP text points3D.txt file → Nx3 float64 array."""
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tok = line.split()
            pts.append([float(tok[1]), float(tok[2]), float(tok[3])])
    return np.array(pts, dtype=np.float64)


def _pcd_from_array(pts: np.ndarray, output_dir: Path, filename: str = "point_cloud.ply") -> o3d.geometry.PointCloud:
    """Convert an Nx3 array to an Open3D point cloud, estimate normals, save."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(20)
    except RuntimeError as exc:
        logger.warning("Normal orientation fallback: %s", exc)
    out = output_dir / filename
    o3d.io.write_point_cloud(str(out), pcd)
    logger.info("Point cloud saved → %s  (%d pts)", out, len(pcd.points))
    return pcd


def _load_and_save_dense_ply(dense_ply: Path, output_dir: Path) -> o3d.geometry.PointCloud:
    """Load a dense PLY, estimate normals, save canonical point_cloud.ply, return PCD."""
    pcd = o3d.io.read_point_cloud(str(dense_ply))
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(20)
    except RuntimeError as exc:
        logger.warning("Normal orientation fallback: %s", exc)
    final_ply = output_dir / "point_cloud.ply"
    o3d.io.write_point_cloud(str(final_ply), pcd)
    logger.info("Dense point cloud saved → %s  (%d pts)", final_ply, len(pcd.points))
    return pcd


def _try_openmvs_dense(
    cfg: ReconstructionConfig,
    colmap: str,
    sparse_model_dir: Path,
    img_dir: Path,
    dense_dir: Path,
    output_dir: Path,
) -> o3d.geometry.PointCloud | None:
    """Attempt dense reconstruction via OpenMVS using COLMAP sparse output.

    Workflow:
    1. `colmap image_undistorter`
    2. `InterfaceCOLMAP -i <dense_dir> -o scene.mvs --image-folder <dense_dir>/images`
    3. `DensifyPointCloud scene.mvs -o scene_dense.mvs`

    Returns an Open3D point cloud on success, else ``None``.
    """
    interface_bin = _which_optional(cfg.openmvs_interface_colmap_bin)
    densify_bin = _which_optional(cfg.openmvs_densify_bin)
    if not interface_bin or not densify_bin:
        logger.info(
            "OpenMVS binaries not available; skipping OpenMVS dense path "
            "(need %s and %s)",
            cfg.openmvs_interface_colmap_bin,
            cfg.openmvs_densify_bin,
        )
        return None

    openmvs_dir = output_dir / "openmvs_workspace"
    openmvs_dir.mkdir(parents=True, exist_ok=True)
    scene_mvs = openmvs_dir / "scene.mvs"
    dense_scene_mvs = openmvs_dir / "scene_dense.mvs"
    dense_ply = openmvs_dir / "scene_dense.ply"

    logger.info("OpenMVS: preparing undistorted COLMAP scene")
    _run([
        colmap, "image_undistorter",
        "--image_path", str(img_dir),
        "--input_path", str(sparse_model_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], "image_undistorter")

    logger.info("OpenMVS: importing COLMAP scene")
    _run([
        interface_bin,
        "-i", str(dense_dir),
        "-o", str(scene_mvs),
        "--image-folder", str(dense_dir / "images"),
        "-w", str(openmvs_dir),
    ], "openmvs_interface_colmap")

    logger.info("OpenMVS: densifying point cloud")
    densify_cmd = [
        densify_bin,
        str(scene_mvs),
        "-o", str(dense_scene_mvs),
        "-w", str(openmvs_dir),
        "--resolution-level", str(cfg.openmvs_resolution_level),
    ]
    if cfg.openmvs_number_views > 0:
        densify_cmd.extend(["--number-views", str(cfg.openmvs_number_views)])
    _run(densify_cmd, "openmvs_densify")

    if dense_ply.exists():
        logger.info("OpenMVS: loading dense point cloud %s", dense_ply)
        return _load_and_save_dense_ply(dense_ply, output_dir)

    logger.warning("OpenMVS finished but %s was not created", dense_ply)
    return None


def _try_colmap_dense(
    colmap: str,
    sparse_model_dir: Path,
    img_dir: Path,
    dense_dir: Path,
    output_dir: Path,
) -> o3d.geometry.PointCloud | None:
    """Attempt COLMAP dense stereo; return point cloud if successful."""
    dense_ply = dense_dir / "fused.ply"
    logger.info("COLMAP: dense MVS (patch_match_stereo + stereo_fusion)")
    _run([
        colmap, "image_undistorter",
        "--image_path", str(img_dir),
        "--input_path", str(sparse_model_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], "image_undistorter")
    _run([
        colmap, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--PatchMatchStereo.gpu_index", "-1",
    ], "patch_match_stereo")
    _run([
        colmap, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--output_path", str(dense_ply),
    ], "stereo_fusion")

    if dense_ply.exists():
        logger.info("COLMAP: loading dense point cloud %s", dense_ply)
        return _load_and_save_dense_ply(dense_ply, output_dir)
    return None


# ── COLMAP reconstruction ─────────────────────────────────────────────────────

def reconstruct_colmap(
    image_paths: list[Path],
    cfg: ReconstructionConfig,
    output_dir: Path,
) -> o3d.geometry.PointCloud:
    """Run COLMAP SfM (+ optional dense MVS) on *image_paths*.

    The function works directly with image files on disk – no in-memory arrays
    needed – which is why the pipeline passes ``image_paths`` here instead of
    decoded arrays.

    Stages
    ------
    1. feature_extractor   – SIFT features for every image
    2. sequential_matcher  – matches consecutive frames (ideal for turntable)
    3. mapper              – bundle-adjustment → sparse point cloud + poses
    4. (dense) image_undistorter + patch_match_stereo + stereo_fusion
       → dense fused.ply, which is used when available

    Falls back to the sparse point cloud if dense MVS fails or if
    ``cfg.use_dense_mvs`` is False.
    """
    colmap = cfg.colmap_bin
    output_dir.mkdir(parents=True, exist_ok=True)

    ws = output_dir / "colmap_workspace"
    ws.mkdir(exist_ok=True)
    db = ws / "database.db"
    sparse_dir = ws / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    dense_dir = ws / "dense"

    # COLMAP needs all images in a single flat directory.
    # Symlink (or copy if cross-device) to a temp image dir inside workspace.
    img_dir = ws / "images"
    img_dir.mkdir(exist_ok=True)
    for p in image_paths:
        dst = img_dir / p.name
        if not dst.exists():
            try:
                dst.symlink_to(p.resolve())
            except OSError:
                shutil.copy2(p, dst)

    # ── 1. Feature extraction ────────────────────────────────────────────────
    logger.info("COLMAP: feature extraction (%d images)", len(image_paths))
    _run_with_gpu_fallback([
        colmap, "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(img_dir),
        "--ImageReader.single_camera", "1",   # assume one camera (same intrinsics)
    ], "--FeatureExtraction.use_gpu", "feature_extractor")

    # ── 2. Sequential matching (turntable / video-frame sequence) ────────────
    overlap = 30 if len(image_paths) >= 100 else 10
    logger.info("COLMAP: sequential matching")
    _run_with_gpu_fallback([
        colmap, "sequential_matcher",
        "--database_path", str(db),
        "--SequentialMatching.overlap", str(overlap),
    ], "--FeatureMatching.use_gpu", "sequential_matcher")

    # ── 3. Sparse reconstruction (mapper / bundle adjustment) ────────────────
    logger.info("COLMAP: sparse reconstruction (mapper)")
    _run([
        colmap, "mapper",
        "--database_path", str(db),
        "--image_path", str(img_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.init_min_num_inliers", "15",
        "--Mapper.abs_pose_min_num_inliers", "10",
    ], "mapper")

    # Find the largest reconstruction sub-folder (0, 1, …)
    recon_dirs = sorted(sparse_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 999)
    if not recon_dirs:
        raise RuntimeError(
            "COLMAP mapper produced no reconstruction. "
            "Check that the images have sufficient overlap and texture."
        )
    best = recon_dirs[0]
    logger.info("COLMAP: using reconstruction in %s", best)

    # ── 4. Dense MVS (optional) ──────────────────────────────────────────────
    if cfg.use_depth_prior:
        backend = cfg.dense_backend
        if backend == "none":
            logger.info("Dense backend disabled by config; using sparse COLMAP output")
        else:
            dense_result: o3d.geometry.PointCloud | None = None
            if backend in ("auto", "openmvs"):
                try:
                    dense_result = _try_openmvs_dense(cfg, colmap, best, img_dir, dense_dir, output_dir)
                except RuntimeError as exc:
                    logger.warning("OpenMVS dense failed (%s)", exc)
                    if backend == "openmvs":
                        logger.warning("Configured dense backend is openmvs; falling back to sparse.")
            if dense_result is None and backend in ("auto", "colmap"):
                try:
                    dense_result = _try_colmap_dense(colmap, best, img_dir, dense_dir, output_dir)
                except RuntimeError as exc:
                    logger.warning("COLMAP dense failed (%s); falling back to sparse.", exc)
            if dense_result is not None:
                return dense_result

    # ── 5. Sparse fallback: read points3D binary or text ─────────────────────
    bin_path = best / "points3D.bin"
    txt_path = best / "points3D.txt"
    if bin_path.exists():
        pts = _read_colmap_points3d_bin(bin_path)
    elif txt_path.exists():
        pts = _read_colmap_points3d_txt(txt_path)
    else:
        raise RuntimeError(f"No points3D file found in {best}")

    if len(pts) == 0:
        raise RuntimeError("COLMAP sparse reconstruction produced 0 points.")

    logger.info("COLMAP: sparse cloud has %d points", len(pts))
    return _pcd_from_array(pts, output_dir)


# ── Legacy ORB fallback ───────────────────────────────────────────────────────

def reconstruct_orb_fallback(
    images: list[np.ndarray],
    output_dir: Path,
) -> o3d.geometry.PointCloud:
    """Legacy ORB-homography heuristic.  Low quality; kept as no-dependency fallback."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _alpha_mask_to_rgb(images)

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
        matches = sorted(bf.match(des_ref, des), key=lambda m: m.distance)[:200]
        if len(matches) < 8:
            continue
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches])
        pts_cur = np.float32([kp[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, 5.0)
        if H is None:
            continue
        inliers = pts_ref[mask.ravel() == 1]
        z = np.ones((len(inliers), 1), dtype=np.float32)
        all_pts3d.append(np.hstack([inliers / 1000.0, z]))

    if not all_pts3d:
        logger.warning("ORB: no matches; empty point cloud.")
        return o3d.geometry.PointCloud()

    return _pcd_from_array(np.vstack(all_pts3d).astype(np.float64), output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def reconstruct(
    images: list[np.ndarray],
    cfg: ReconstructionConfig,
    output_dir: Path,
    image_paths: list[Path] | None = None,
) -> o3d.geometry.PointCloud:
    """Dispatch to the configured reconstruction method.

    Args:
        images:      Decoded image arrays (used by the ORB fallback).
        cfg:         ReconstructionConfig.
        output_dir:  Where to save the point cloud and working files.
        image_paths: On-disk paths to the source images (required for COLMAP).
    """
    if cfg.method == "colmap":
        if image_paths is None:
            raise ValueError("image_paths must be provided for COLMAP reconstruction.")
        colmap_bin = shutil.which(cfg.colmap_bin) or cfg.colmap_bin
        if not shutil.which(colmap_bin):
            logger.warning(
                "COLMAP binary '%s' not found; falling back to ORB heuristic. "
                "Install with: brew install colmap", colmap_bin
            )
            return reconstruct_orb_fallback(images, output_dir)
        return reconstruct_colmap(image_paths, cfg, output_dir)

    # open3d / opencv_sfm → legacy ORB fallback
    return reconstruct_orb_fallback(images, output_dir)
