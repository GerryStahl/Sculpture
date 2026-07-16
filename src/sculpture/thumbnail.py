"""Thumbnail rendering for meshes and wireframes.

Produces 512×512 PNG previews using Open3D's offscreen renderer.
Falls back to a matplotlib-based projection if the renderer is unavailable
(e.g. in headless CI environments with no display).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_THUMB_SIZE = 512  # pixels


def _camera_params(mesh_or_pcd) -> tuple[list, list, list]:
    """Return (center, eye, up) for a pleasant isometric-ish view."""
    aabb = mesh_or_pcd.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = aabb.get_extent()
    diag = float(np.linalg.norm(extent))
    # Position camera at ~45° elevation, ~30° azimuth
    eye = center + np.array([diag, diag * 0.6, diag * 0.8])
    up = [0.0, 1.0, 0.0]
    return center.tolist(), eye.tolist(), up


def render_mesh_thumbnail(
    mesh_path: Path,
    out_path: Path,
    size: int = _THUMB_SIZE,
) -> Path | None:
    """Render a shaded mesh thumbnail from *mesh_path* and save to *out_path*.

    Returns the output path on success, or ``None`` if rendering failed.
    """
    try:
        o3d = importlib.import_module("open3d")
    except ImportError:
        logger.warning("open3d not available; skipping mesh thumbnail")
        return None

    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()

        # Colour the mesh a warm stone-grey
        mesh.paint_uniform_color([0.75, 0.72, 0.68])

        center, eye, up = _camera_params(mesh)

        vis = o3d.visualization.rendering.OffscreenRenderer(size, size)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"

        vis.scene.add_geometry("mesh", mesh, mat)
        vis.scene.set_background([0.15, 0.15, 0.15, 1.0])  # dark grey bg

        # One-directional key light from upper-right
        vis.scene.scene.enable_sun_light(True)
        vis.scene.scene.set_sun_light(
            [0.5, -0.7, -0.5],  # direction
            [1.0, 0.95, 0.9],   # colour (warm white)
            80_000,             # intensity
        )

        vis.setup_camera(
            60.0,    # fov_deg
            center,
            eye,
            up,
        )

        img = vis.render_to_image()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_image(str(out_path), img)
        logger.info("Mesh thumbnail → %s", out_path)
        return out_path

    except Exception as e:
        logger.warning("Mesh thumbnail failed (%s); trying fallback", e)
        return _mesh_thumbnail_fallback(mesh_path, out_path, size)


def _mesh_thumbnail_fallback(
    mesh_path: Path,
    out_path: Path,
    size: int = _THUMB_SIZE,
) -> Path | None:
    """Matplotlib-based fallback: renders a projected point cloud scatter."""
    try:
        o3d = importlib.import_module("open3d")
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        pts = np.asarray(mesh.vertices)
        if len(pts) == 0:
            return None

        # Isometric-ish projection: rotate ~30° then project XY
        theta = np.radians(30)
        phi = np.radians(25)
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        pts_rot = (pts - pts.mean(axis=0)) @ Ry.T @ Rx.T

        fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
        ax.scatter(pts_rot[:, 0], pts_rot[:, 1], s=0.4, c="#b8b0a8", linewidths=0)
        ax.set_facecolor("#262626")
        fig.patch.set_facecolor("#262626")
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Mesh thumbnail (fallback) → %s", out_path)
        return out_path

    except Exception as e:
        logger.warning("Mesh thumbnail fallback also failed: %s", e)
        return None


def render_wireframe_thumbnail(
    wireframe_obj_path: Path,
    out_path: Path,
    size: int = _THUMB_SIZE,
) -> Path | None:
    """Render a wireframe thumbnail from an OBJ file and save to *out_path*.

    Uses matplotlib so this works in headless environments.
    Returns the output path on success, or ``None`` if rendering failed.
    """
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        LineCollection = importlib.import_module("matplotlib.collections").LineCollection

        # Parse OBJ: read vertices (v) and lines (l)
        verts: list[np.ndarray] = []
        lines: list[list[int]] = []

        with wireframe_obj_path.open() as f:
            for raw in f:
                tok = raw.split()
                if not tok:
                    continue
                if tok[0] == "v" and len(tok) >= 4:
                    verts.append(np.array([float(tok[1]),
                                           float(tok[2]),
                                           float(tok[3])]))
                elif tok[0] == "l":
                    # OBJ line indices are 1-based
                    lines.append([int(i) - 1 for i in tok[1:]])

        if not verts or not lines:
            logger.warning("Wireframe OBJ has no geometry; skipping thumbnail")
            return None

        pts = np.array(verts)

        # Isometric-ish rotation matching the mesh fallback
        theta = np.radians(30)
        phi = np.radians(25)
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        pts_rot = (pts - pts.mean(axis=0)) @ Ry.T @ Rx.T

        segments = []
        for line in lines:
            for a, b in zip(line[:-1], line[1:]):
                if 0 <= a < len(pts_rot) and 0 <= b < len(pts_rot):
                    segments.append([pts_rot[a, :2], pts_rot[b, :2]])

        if not segments:
            return None

        fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
        lc = LineCollection(segments, linewidths=0.5, colors="#e8c87a", alpha=0.85)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Wireframe thumbnail → %s", out_path)
        return out_path

    except Exception as e:
        logger.warning("Wireframe thumbnail failed: %s", e)
        return None
