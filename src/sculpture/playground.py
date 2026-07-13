"""Interactive sculpture-model playground tools.

These helpers are meant for learning and experimentation:
- load meshes from the reconstruction pipeline
- rotate and inspect the model from different angles
- apply simple deformations by pulling vertices around a control point
- export an HTML 3-D viewer for quick iteration
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import trimesh


@dataclass(frozen=True)
class ControlPoint:
    """A point in model space used to deform the mesh."""

    position: np.ndarray
    radius: float = 0.25
    strength: float = 0.2



def load_mesh(path: Path | str) -> o3d.geometry.TriangleMesh:
    """Load a mesh from disk using Open3D."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Could not load mesh from {path}")
    mesh.compute_vertex_normals()
    return mesh



def save_mesh(mesh: o3d.geometry.TriangleMesh, path: Path | str) -> None:
    """Save a mesh to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh)



def mesh_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Convert an Open3D mesh to trimesh for editing and export."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)



def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert a trimesh back to Open3D."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh



def rotate_mesh(mesh: o3d.geometry.TriangleMesh, degrees_y: float = 15.0) -> o3d.geometry.TriangleMesh:
    """Rotate a mesh around the vertical Y axis."""
    rotated = mesh.clone()
    radians = np.deg2rad(degrees_y)
    rotation = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, radians, 0.0])
    rotated.rotate(rotation, center=rotated.get_center())
    rotated.compute_vertex_normals()
    return rotated



def scale_mesh(mesh: o3d.geometry.TriangleMesh, factor: float = 1.0) -> o3d.geometry.TriangleMesh:
    """Uniformly scale a mesh about its center."""
    scaled = mesh.clone()
    scaled.scale(factor, center=scaled.get_center())
    scaled.compute_vertex_normals()
    return scaled



def translate_mesh(mesh: o3d.geometry.TriangleMesh, offset: tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """Translate a mesh by an XYZ offset."""
    translated = mesh.clone()
    translated.translate(offset)
    translated.compute_vertex_normals()
    return translated



def deform_near_point(
    mesh: o3d.geometry.TriangleMesh,
    control_point: ControlPoint,
    direction: np.ndarray | None = None,
) -> o3d.geometry.TriangleMesh:
    """Pull vertices near a control point.

    Vertices inside the control radius move more than distant vertices.
    This is a simple "grab and pull" deformation that is useful for learning.
    """
    deformed = mesh.clone()
    vertices = np.asarray(deformed.vertices).copy()
    center = np.asarray(control_point.position, dtype=np.float64)
    direction_vec = np.asarray(direction if direction is not None else [0.0, 0.0, 1.0], dtype=np.float64)
    norm = np.linalg.norm(direction_vec)
    if norm == 0:
        raise ValueError("direction must not be zero")
    direction_vec = direction_vec / norm

    deltas = vertices - center
    distances = np.linalg.norm(deltas, axis=1)
    weights = np.exp(-((distances / max(control_point.radius, 1e-6)) ** 2))
    vertices = vertices + weights[:, None] * control_point.strength * direction_vec
    deformed.vertices = o3d.utility.Vector3dVector(vertices)
    deformed.compute_vertex_normals()
    return deformed



def mirror_mesh(mesh: o3d.geometry.TriangleMesh, axis: str = "x") -> o3d.geometry.TriangleMesh:
    """Mirror a mesh across the selected axis."""
    axis = axis.lower()
    scale = {
        "x": (-1.0, 1.0, 1.0),
        "y": (1.0, -1.0, 1.0),
        "z": (1.0, 1.0, -1.0),
    }.get(axis)
    if scale is None:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")
    mirrored = mesh.clone()
    mirrored.scale(1.0, center=mirrored.get_center())
    vertices = np.asarray(mirrored.vertices).copy()
    vertices *= np.array(scale, dtype=np.float64)
    mirrored.vertices = o3d.utility.Vector3dVector(vertices)
    mirrored.compute_vertex_normals()
    return mirrored



def shear_mesh(
    mesh: o3d.geometry.TriangleMesh,
    shear_xy: float = 0.0,
    shear_xz: float = 0.0,
    shear_yz: float = 0.0,
) -> o3d.geometry.TriangleMesh:
    """Apply a simple shear transformation to the mesh."""
    sheared = mesh.clone()
    vertices = np.asarray(sheared.vertices).copy()
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    vertices[:, 0] = x + shear_xy * y + shear_xz * z
    vertices[:, 1] = y + shear_yz * z
    sheared.vertices = o3d.utility.Vector3dVector(vertices)
    sheared.compute_vertex_normals()
    return sheared



def transform_vertex_subset(
    mesh: o3d.geometry.TriangleMesh,
    predicate,
    offset: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh:
    """Move only the vertices that satisfy *predicate*.

    The predicate receives a vertex position as a NumPy array and should
    return True for vertices that should be edited.
    """
    edited = mesh.clone()
    vertices = np.asarray(edited.vertices).copy()
    move = np.asarray(offset, dtype=np.float64)
    mask = np.array([bool(predicate(v)) for v in vertices], dtype=bool)
    vertices[mask] = vertices[mask] + move
    edited.vertices = o3d.utility.Vector3dVector(vertices)
    edited.compute_vertex_normals()
    return edited



def compute_bounding_box_info(mesh: o3d.geometry.TriangleMesh) -> dict[str, float]:
    """Return easy-to-read bounding-box measurements for the mesh."""
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    return {
        "width": float(extent[0]),
        "height": float(extent[1]),
        "depth": float(extent[2]),
        "diagonal": float(np.linalg.norm(extent)),
    }



def mesh_preview_figure(mesh: o3d.geometry.TriangleMesh, title: str = "Sculpture playground") -> go.Figure:
    """Create a Plotly figure for rotating and inspecting the mesh."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                color="lightsteelblue",
                opacity=0.85,
                flatshading=False,
                lighting=dict(ambient=0.55, diffuse=0.7, specular=0.3, roughness=0.8),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig



def export_viewer_html(mesh: o3d.geometry.TriangleMesh, path: Path | str, title: str = "Sculpture viewer") -> Path:
    """Export a self-contained HTML file for sharing and rotation inspection."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = mesh_preview_figure(mesh, title=title)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path



def turntable_orbit_frames(mesh: o3d.geometry.TriangleMesh, steps: int = 36) -> list[o3d.geometry.TriangleMesh]:
    """Generate rotated copies of the mesh for visual comparison."""
    frames: list[o3d.geometry.TriangleMesh] = []
    for index in range(steps):
        angle = 360.0 * index / max(steps, 1)
        frames.append(rotate_mesh(mesh, angle))
    return frames
