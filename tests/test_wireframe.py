"""Tests for wireframe extraction (pure-Python logic, no Open3D required)."""

import numpy as np
import pytest

from sculpture.wireframe import build_wireframe_graph, filter_short_edges


def _make_unit_cube_vertices() -> np.ndarray:
    return np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)


def _cube_edges() -> list[tuple[int, int]]:
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def test_build_wireframe_graph():
    verts = _make_unit_cube_vertices()
    edges = _cube_edges()
    G = build_wireframe_graph(edges, verts)
    assert G.number_of_nodes() == 8
    assert G.number_of_edges() == 12


def test_filter_short_edges_removes_nothing():
    """With a very small min_frac nothing should be removed."""
    verts = _make_unit_cube_vertices()
    edges = _cube_edges()
    kept = filter_short_edges(edges, verts, min_frac=0.0)
    assert len(kept) == len(edges)


def test_filter_short_edges_removes_all():
    """With a huge min_frac everything should be removed."""
    verts = _make_unit_cube_vertices()
    edges = _cube_edges()
    kept = filter_short_edges(edges, verts, min_frac=10.0)
    assert len(kept) == 0
