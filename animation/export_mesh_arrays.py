"""Blender script: write a prepared sculpture's triangles to a .npz for analysis outside Blender.

$BLENDER --background SCULPTURE.blend --python animation/export_mesh_arrays.py -- OUT.npz
"""

import sys

import bpy
import numpy as np

out = sys.argv[sys.argv.index("--") + 1]
mesh = next(o for o in bpy.data.objects if o.type == "MESH").data
mesh.calc_loop_triangles()
count = len(mesh.loop_triangles)

verts = np.empty(len(mesh.vertices) * 3)
mesh.vertices.foreach_get("co", verts)
tris = np.empty(count * 3, dtype=np.int64)
mesh.loop_triangles.foreach_get("vertices", tris)
poly = np.empty(count, dtype=np.int64)
mesh.loop_triangles.foreach_get("polygon_index", poly)

np.savez(
    out, verts=verts.reshape(-1, 3), tris=tris.reshape(-1, 3), poly=poly, n_poly=len(mesh.polygons)
)
