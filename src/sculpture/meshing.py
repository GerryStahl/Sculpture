"""Point cloud → mesh conversion and simplification."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from sculpture.config import MeshingConfig

logger = logging.getLogger(__name__)


def pcd_to_mesh_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
) -> o3d.geometry.TriangleMesh:
    """Poisson surface reconstruction from a normals-equipped point cloud."""
    if not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(10)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    # Remove low-density vertices (background / outlier artefacts)
    d = np.asarray(densities)
    keep = d > np.quantile(d, 0.1)
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    logger.info("Poisson mesh: %d vertices, %d triangles",
                len(mesh.vertices), len(mesh.triangles))
    return mesh


def pcd_to_mesh_ball_pivot(
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    """Ball-pivoting algorithm; good for dense, uniformly sampled point clouds."""
    if not pcd.has_normals():
        pcd.estimate_normals()

    # Estimate typical point spacing to set ball radii
    pts = np.asarray(pcd.points)
    dist = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    r = float(np.percentile(dist, 5)) * 0.5
    radii = [r, r * 2, r * 4]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    logger.info("BPA mesh: %d vertices, %d triangles",
                len(mesh.vertices), len(mesh.triangles))
    return mesh


def pcd_to_mesh_alpha(
    pcd: o3d.geometry.PointCloud,
    alpha: float = 0.03,
) -> o3d.geometry.TriangleMesh:
    """Alpha-shape surface reconstruction."""
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    logger.info("Alpha-shape mesh: %d vertices, %d triangles",
                len(mesh.vertices), len(mesh.triangles))
    return mesh


def remove_small_components(
    mesh: o3d.geometry.TriangleMesh,
    min_frac: float = 0.01,
) -> o3d.geometry.TriangleMesh:
    """Remove connected components smaller than *min_frac* of the largest."""
    triangle_clusters, _, _ = mesh.cluster_connected_triangles()
    counts = np.bincount(np.asarray(triangle_clusters))
    threshold = counts.max() * min_frac
    keep = np.array(triangle_clusters) >= 0  # all true initially
    for cid, cnt in enumerate(counts):
        if cnt < threshold:
            keep[np.array(triangle_clusters) == cid] = False
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def simplify_mesh(
    mesh: o3d.geometry.TriangleMesh,
    target_faces: int,
) -> o3d.geometry.TriangleMesh:
    """Decimate mesh to approximately *target_faces* triangles."""
    if target_faces <= 0 or len(mesh.triangles) <= target_faces:
        return mesh
    mesh = mesh.simplify_quadric_decimation(target_faces)
    logger.info("Simplified to %d triangles", len(mesh.triangles))
    return mesh


def build_mesh(
    pcd: o3d.geometry.PointCloud,
    cfg: MeshingConfig,
    output_dir: Path,
) -> o3d.geometry.TriangleMesh:
    """Full meshing pipeline: reconstruct → clean → simplify → save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.method == "poisson":
        mesh = pcd_to_mesh_poisson(pcd, depth=cfg.poisson_depth)
    elif cfg.method == "ball_pivot":
        mesh = pcd_to_mesh_ball_pivot(pcd)
    elif cfg.method == "alpha_shape":
        mesh = pcd_to_mesh_alpha(pcd)
    else:
        raise ValueError(f"Unknown meshing method: {cfg.method!r}")

    mesh = remove_small_components(mesh, cfg.min_component_frac)
    mesh = simplify_mesh(mesh, cfg.simplify_faces)
    mesh.compute_vertex_normals()

    out_path = output_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    logger.info("Mesh saved → %s", out_path)
    return mesh
