"""Blender script: import a sculpture mesh, clean it, normalize it, and save it as a .blend.

Run:
  $BLENDER --background --python animation/prep_mesh.py -- RAW.usdz OUT.blend \
      --cut-frac 0.15 --preview PREVIEW.png

Cleaning: drop floating pieces, cut away the stand below `--cut-frac` of the sculpture's height
(sealing the cut), then center on the base at z=0 and scale to `--height`. With `--guide PNG`,
nothing is saved; a front view with a labeled line every 5% of height is rendered so the cut
fraction can be chosen by eye.
"""

import argparse
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

GUIDE_STEP = 0.05
GUIDE_LINES = 12


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--name", default=None, help="Object name (default: output file stem)")
    p.add_argument("--height", type=float, default=1.0)
    p.add_argument("--faces", type=int, default=150_000, help="Decimate above this (0 = off)")
    p.add_argument("--rotate-x", type=float, default=0.0, help="Degrees about X to stand it up")
    p.add_argument("--cut-frac", type=float, default=0.0, help="Remove this fraction of the height")
    p.add_argument(
        "--max-texture", type=int, default=2048, help="Downscale larger textures (0 = off)"
    )
    p.add_argument("--guide", type=Path, default=None, help="Render a cut-height guide and stop")
    p.add_argument("--preview", type=Path, default=None, help="Render the cleaned result")
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


def normalize(obj: bpy.types.Object, height: float, rotate_x_deg: float = 0.0) -> None:
    """Stand the mesh on z=0, centered in x/y, scaled so its height is `height`."""
    mesh = obj.data
    if rotate_x_deg:
        mesh.transform(Matrix.Rotation(math.radians(rotate_x_deg), 4, "X"))

    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    zs = [v.co.z for v in mesh.vertices]
    center = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min(zs)))
    scale = height / (max(zs) - min(zs))
    mesh.transform(Matrix.Scale(scale, 4) @ Matrix.Translation(-center))


def keep_largest_island(bm: bmesh.types.BMesh) -> int:
    """Delete every connected piece except the one with the most surface area."""
    seen: set[int] = set()
    islands: list[list[bmesh.types.BMFace]] = []
    bm.faces.index_update()
    for face in bm.faces:
        if face.index in seen:
            continue
        seen.add(face.index)
        stack, island = [face], []
        while stack:
            current = stack.pop()
            island.append(current)
            for edge in current.edges:
                for neighbor in edge.link_faces:
                    if neighbor.index not in seen:
                        seen.add(neighbor.index)
                        stack.append(neighbor)
        islands.append(island)

    if len(islands) > 1:
        islands.sort(key=lambda isl: sum(f.calc_area() for f in isl), reverse=True)
        bmesh.ops.delete(bm, geom=[f for isl in islands[1:] for f in isl], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    return len(islands) - 1


def cut_below(bm: bmesh.types.BMesh, z: float) -> None:
    """Remove everything under z and seal the opening with a flat cap."""
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(
        bm, geom=geom, plane_co=(0, 0, z), plane_no=(0, 0, 1), clear_inner=True, clear_outer=False
    )
    rim = [e for e in bm.edges if e.is_boundary and all(abs(v.co.z - z) < 1e-5 for v in e.verts)]
    if rim:
        cap = bmesh.ops.holes_fill(bm, edges=rim, sides=0)["faces"]
        bmesh.ops.triangulate(bm, faces=cap)


def clean(obj: bpy.types.Object, cut_z: float | None) -> int:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    removed = keep_largest_island(bm)
    if cut_z is not None:
        cut_below(bm, cut_z)
        removed += keep_largest_island(bm)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return removed


def decimate(obj: bpy.types.Object, max_faces: int) -> None:
    faces = len(obj.data.polygons)
    if max_faces <= 0 or faces <= max_faces:
        return
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.ratio = max_faces / faces
    bpy.ops.object.modifier_apply(modifier=mod.name)


def limit_textures(max_size: int) -> None:
    """Downscale oversized textures so many sculptures fit in GPU memory and render quickly."""
    for image in bpy.data.images:
        if image.type != "IMAGE" or not image.packed_file:
            continue
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            image.scale(round(width * ratio), round(height * ratio))
            image.pack()


def glow_material(name: str, color: tuple[float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def add_guide_overlay(obj: bpy.types.Object, span: float) -> None:
    """Labelled horizontal lines every GUIDE_STEP of the height, in front of the mesh."""
    xs = [v.co.x for v in obj.data.vertices]
    height = max(v.co.z for v in obj.data.vertices)
    width = max(xs) - min(xs)
    line_mat = glow_material("GuideLine", (0.9, 0.0, 0.0))
    text_mat = glow_material("GuideText", (0.0, 0.0, 0.0))
    for i in range(1, GUIDE_LINES + 1):
        z = i * GUIDE_STEP * height
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -width, z))
        line = bpy.context.active_object
        line.scale = (width * 1.2, 0.001, span * 0.003)
        line.data.materials.append(line_mat)
        bpy.ops.object.text_add(
            location=(width * 0.62, -width, z + span * 0.004), rotation=(math.pi / 2, 0, 0)
        )
        label = bpy.context.active_object
        label.data.body = f"{round(i * GUIDE_STEP * 100)}%"
        label.data.size = min(span * 0.03, height * GUIDE_STEP * 0.7)
        label.data.materials.append(text_mat)


def render_png(obj: bpy.types.Object, path: Path, guide: bool) -> None:
    """Render a front orthographic view (guide) or a three-quarter view (preview)."""
    scene = bpy.context.scene
    xs = [v.co.x for v in obj.data.vertices]
    height = max(v.co.z for v in obj.data.vertices)
    width = max(xs) - min(xs)
    span = max(height, width) * 1.5

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.85, 0.85, 0.88, 1)

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(sun)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    target = Vector((0, 0, height / 2))
    if guide:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = span * 1.4
        cam.location = target + Vector((0, -span * 2, 0))
    else:
        cam_data.lens = 50
        cam.location = target + Vector((span * 0.9, -span * 1.6, span * 0.4))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam

    if guide:
        add_guide_overlay(obj, span)

    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_x, scene.render.resolution_y = (900, 1100)
    scene.render.image_settings.file_format = "PNG"
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    import_mesh(args.input)
    obj = join_meshes()
    clean(obj, cut_z=None)
    normalize(obj, args.height, args.rotate_x)

    if args.guide:
        bpy.ops.object.shade_smooth()
        render_png(obj, args.guide, guide=True)
        print(f"Saved guide {args.guide}")
        return

    if args.cut_frac > 0:
        clean(obj, cut_z=args.cut_frac * args.height)
        normalize(obj, args.height)
    decimate(obj, args.faces)

    obj.name = args.name or args.output.stem
    obj.data.name = obj.name
    bpy.ops.object.shade_smooth()
    bpy.ops.file.pack_all()
    if args.max_texture > 0:
        limit_textures(args.max_texture)

    if args.preview:
        render_png(obj, args.preview, guide=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"Saved {args.output}: {len(obj.data.polygons)} faces, height {args.height}")


if __name__ == "__main__":
    main()
