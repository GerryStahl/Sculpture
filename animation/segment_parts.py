"""Split a prepared sculpture into its distinct masses so each can have its own skin.

  python animation/segment_parts.py emergent3

The solid is filled with voxels and a distance transform is taken, so thick masses show up as
peaks and the narrow necks between them become the dividing lines (a watershed). Settings per
sculpture are in animation/parts.json:
  num_peaks     how many thickness peaks to start from
  min_distance  smallest gap between peaks, in voxels
  merge         pairs of peak numbers (0-based, thickest first) to join into one mass, such as
                two peaks along one long neck

Parts are numbered from the top of the sculpture down. The result is written to
data/blender_exports/<name>/<name>_parts.json, and build_scene.py uses it when a sculpture's
skin is a list with one skin per part.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

ROOT = Path(__file__).resolve().parent.parent
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
VOXELS_ALONG_LONGEST_SIDE = 110


def load_triangles(name: str) -> dict:
    blend = ROOT / "data" / "blender_exports" / name / f"{name}.blend"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mesh.npz"
        export = ROOT / "animation" / "export_mesh_arrays.py"
        cmd = [BLENDER, "--background", str(blend), "--python", str(export), "--", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        return dict(np.load(out))


def segment(mesh: trimesh.Trimesh, settings: dict) -> np.ndarray:
    """Return a part number for every triangle, numbered from the top down."""
    voxels = mesh.voxelized(mesh.extents.max() / VOXELS_ALONG_LONGEST_SIDE).fill()
    solid = voxels.matrix
    thickness = ndimage.gaussian_filter(ndimage.distance_transform_edt(solid), 1.0)
    peaks = peak_local_max(
        thickness,
        min_distance=settings["min_distance"],
        num_peaks=settings["num_peaks"],
        labels=solid.astype(int),
        exclude_border=False,
    )

    markers = np.zeros(solid.shape, dtype=int)
    for i, peak in enumerate(peaks):
        markers[tuple(peak)] = i + 1
    for keep, drop in settings.get("merge", []):
        markers[markers == drop + 1] = keep + 1
    regions = watershed(-thickness, markers, mask=solid)

    cells = np.argwhere(solid)
    owner = regions[tuple(cells.T)]
    _, nearest = cKDTree(voxels.indices_to_points(cells)).query(mesh.triangles_center)
    labels = owner[nearest]

    # Renumber 0..n-1 from the highest mass down so part numbers are stable and meaningful.
    ids = np.unique(labels)
    height = {i: mesh.triangles_center[labels == i][:, 2].mean() for i in ids}
    order = sorted(ids, key=lambda i: -height[i])
    return np.array([order.index(i) for i in labels])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("name")
    parser.add_argument("--settings", type=Path, default=ROOT / "animation" / "parts.json")
    args = parser.parse_args()

    settings = json.loads(args.settings.read_text())[args.name]
    data = load_triangles(args.name)
    mesh = trimesh.Trimesh(data["verts"], data["tris"], process=False)
    tri_labels = segment(mesh, settings)

    # Blender polygons can be larger than a triangle; give each the majority label of its triangles.
    n_parts = int(tri_labels.max()) + 1
    votes = np.zeros((int(data["n_poly"]), n_parts))
    np.add.at(votes, (data["poly"], tri_labels), 1)
    polygon_labels = votes.argmax(axis=1)

    out = ROOT / "data" / "blender_exports" / args.name / f"{args.name}_parts.json"
    out.write_text(json.dumps({"parts": n_parts, "polygon_labels": polygon_labels.tolist()}))
    print(f"{args.name}: {n_parts} parts, polygons per part {np.bincount(polygon_labels).tolist()}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
