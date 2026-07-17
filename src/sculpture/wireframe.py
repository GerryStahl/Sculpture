"""Wireframe extraction from triangle mesh."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import open3d as o3d

from sculpture.config import WireframeConfig

logger = logging.getLogger(__name__)


def extract_feature_edges(
    mesh: o3d.geometry.TriangleMesh,
    feature_angle_deg: float = 30.0,
) -> list[tuple[int, int]]:
    """Return edge pairs whose dihedral angle exceeds *feature_angle_deg*.

    These correspond to creases, silhouettes, and structural ridges of the
    sculpture – the edges we want to keep in the wireframe.
    """
    mesh.compute_triangle_normals()
    tris = np.asarray(mesh.triangles)
    tri_normals = np.asarray(mesh.triangle_normals)
    cos_thresh = np.cos(np.radians(feature_angle_deg))

    # Build edge → triangle adjacency
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for ti, tri in enumerate(tris):
        for i in range(3):
            e = (int(min(tri[i], tri[(i + 1) % 3])),
                 int(max(tri[i], tri[(i + 1) % 3])))
            edge_to_tris.setdefault(e, []).append(ti)

    feature_edges: list[tuple[int, int]] = []
    for edge, adj_tris in edge_to_tris.items():
        if len(adj_tris) < 2:
            feature_edges.append(edge)  # boundary edge – always keep
            continue
        n1 = tri_normals[adj_tris[0]]
        n2 = tri_normals[adj_tris[1]]
        cos_angle = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        if cos_angle < cos_thresh:
            feature_edges.append(edge)

    logger.info("Feature edges extracted: %d", len(feature_edges))
    return feature_edges


def filter_short_edges(
    edges: list[tuple[int, int]],
    vertices: np.ndarray,
    min_frac: float = 0.005,
) -> list[tuple[int, int]]:
    """Remove edges shorter than *min_frac* of the bounding-box diagonal."""
    pts = vertices
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    min_len = diag * min_frac
    kept = [e for e in edges
            if np.linalg.norm(pts[e[0]] - pts[e[1]]) >= min_len]
    logger.debug("Edges after length filter: %d / %d", len(kept), len(edges))
    return kept


def build_wireframe_graph(
    edges: list[tuple[int, int]],
    vertices: np.ndarray,
) -> nx.Graph:
    """Build a NetworkX graph from wireframe edges."""
    G = nx.Graph()
    for i, v in enumerate(vertices):
        G.add_node(i, pos=v.tolist())
    for u, v in edges:
        length = float(np.linalg.norm(vertices[u] - vertices[v]))
        G.add_edge(u, v, length=length)
    # Keep only nodes that appear in at least one edge
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    logger.info("Wireframe graph: %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())
    return G


def export_wireframe_obj(
    edges: list[tuple[int, int]],
    vertices: np.ndarray,
    path: Path,
) -> None:
    """Write wireframe as a Wavefront OBJ (line elements only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# Sculpture wireframe\n")
        for v in vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for u, v_idx in edges:
            fh.write(f"l {u + 1} {v_idx + 1}\n")  # OBJ is 1-indexed
    logger.info("Wireframe OBJ saved → %s", path)


def export_wireframe_json(
    graph: nx.Graph,
    path: Path,
) -> None:
    """Serialise wireframe graph as JSON (node positions + edges)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "nodes": [{"id": n, "pos": d["pos"]} for n, d in graph.nodes(data=True)],
        "edges": [{"u": u, "v": v, "length": d.get("length", 0)}
                  for u, v, d in graph.edges(data=True)],
    }
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Wireframe JSON saved → %s", path)


def extract_wireframe(
    mesh: o3d.geometry.TriangleMesh,
    cfg: WireframeConfig,
    output_dir: Path,
) -> nx.Graph:
    """Full wireframe extraction pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guard against degenerate meshes
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        logger.warning("Mesh is degenerate; returning empty wireframe graph")
        return nx.Graph()
    
    vertices = np.asarray(mesh.vertices)

    edges = extract_feature_edges(mesh, cfg.feature_angle_deg)
    edges = filter_short_edges(edges, vertices, cfg.min_edge_frac)
    graph = build_wireframe_graph(edges, vertices)

    if cfg.export_format == "obj":
        export_wireframe_obj(edges, vertices, output_dir / "wireframe.obj")
    elif cfg.export_format == "json_graph":
        export_wireframe_json(graph, output_dir / "wireframe.json")
    # svg export: future – requires projection to 2-D view plane

    return graph
