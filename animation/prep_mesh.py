"""Blender script: import a sculpture mesh, normalise it, and save it as a .blend.

Run:
  $BLENDER --background --python animation/prep_mesh.py -- \
      data/blender_exports/emergent4/emergent4.usdz data/blender_exports/emergent4/emergent4.blend

The result has its origin at the centre of the base, stands on z=0, and is `--height` tall,
so build_scene.py can spin it in place and move it along a path.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--name", default=None, help="Object name (default: output file stem)")
    p.add_argument("--height", type=float, default=1.0)
    p.add_argument(
        "--faces", type=int, default=150_000, help="Decimate above this face count (0 = off)"
    )
    p.add_argument(
        "--rotate-x", type=float, default=0.0, help="Degrees about X to stand the mesh upright"
    )
    return p.parse_args(argv)


def import_mesh(path: Path) -> None:
    ext = path.suffix.lower()
    if ext in {".usd", ".usda", ".usdc", ".usdz"}:
        bpy.ops.wm.usd_import(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise SystemExit(f"Unsupported mesh format: {ext}")


def join_meshes() -> bpy.types.Object:
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("No mesh objects found in the imported file")

    for obj in meshes:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world

    for obj in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def normalise(obj: bpy.types.Object, height: float, rotate_x_deg: float) -> None:
    mesh = obj.data
    if rotate_x_deg:
        mesh.transform(Matrix.Rotation(math.radians(rotate_x_deg), 4, "X"))

    zs = [v.co.z for v in mesh.vertices]
    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    centre = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min(zs))
    scale = height / (max(zs) - min(zs))
    mesh.transform(
        Matrix.Scale(scale, 4) @ Matrix.Translation((-centre[0], -centre[1], -centre[2]))
    )


def decimate(obj: bpy.types.Object, max_faces: int) -> None:
    faces = len(obj.data.polygons)
    if max_faces <= 0 or faces <= max_faces:
        return
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.ratio = max_faces / faces
    bpy.ops.object.modifier_apply(modifier=mod.name)


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    import_mesh(args.input)
    obj = join_meshes()
    normalise(obj, args.height, args.rotate_x)
    decimate(obj, args.faces)

    obj.name = args.name or args.output.stem
    obj.data.name = obj.name
    bpy.ops.object.shade_smooth()
    bpy.ops.file.pack_all()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"Saved {args.output}: {len(obj.data.polygons)} faces, height {args.height}")


if __name__ == "__main__":
    main()
