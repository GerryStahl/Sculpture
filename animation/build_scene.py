"""Blender script: build the animated sculpture scene from a JSON config and render it.

Run from the project root, with RUN="$BLENDER --background --python animation/build_scene.py --":
  $RUN animation/scene.json --still 1 --out data/animation/still.png    # one frame
  $RUN animation/scene.json --preview --out data/animation/preview.mp4  # fast low-res video
  $RUN animation/scene.json --out data/animation/final.mp4              # full render
  $RUN animation/scene.json --save-blend data/animation/scene.blend     # open in Blender UI
"""

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_INTERVAL = 2  # frames between location keyframes


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--preview", action="store_true", help="Half resolution, fewer samples")
    p.add_argument("--still", type=int, default=None, help="Render one frame instead of the video")
    p.add_argument("--save-blend", type=Path, default=None)
    return p.parse_args(argv)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def closed_catmull_rom(
    points: list[list[float]], samples_per_seg: int = 64
) -> list[tuple[float, float]]:
    n = len(points)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = (points[(i + k) % n] for k in (-1, 0, 1, 2))
        for s in range(samples_per_seg):
            t = s / samples_per_seg
            t2, t3 = t * t, t * t * t
            out.append(
                tuple(
                    0.5
                    * (
                        2 * p1[d]
                        + (-p0[d] + p2[d]) * t
                        + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2
                        + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3
                    )
                    for d in (0, 1)
                )
            )
    return out


class ConstantSpeedPath:
    """Closed path parametrised by distance so speed is in scene units per second."""

    def __init__(self, points: list[list[float]]):
        self.pts = closed_catmull_rom(points)
        self.cum = [0.0]
        for a, b in zip(self.pts, self.pts[1:] + self.pts[:1]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.length = self.cum[-1]

    def at(self, distance: float) -> tuple[float, float]:
        d = distance % self.length
        i = min(bisect.bisect_right(self.cum, d) - 1, len(self.pts) - 1)
        seg = self.cum[i + 1] - self.cum[i]
        f = (d - self.cum[i]) / seg if seg else 0.0
        a, b = self.pts[i], self.pts[(i + 1) % len(self.pts)]
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def set_world(background: dict) -> None:
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*background["color"], 1.0)
    bg.inputs["Strength"].default_value = background.get("strength", 1.0)


def add_ground(ground: dict) -> None:
    if not ground.get("enabled", True):
        return
    bpy.ops.mesh.primitive_plane_add(size=ground.get("size", 60))
    plane = bpy.context.active_object
    plane.name = "Ground"
    mat = bpy.data.materials.new("Ground")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*ground["color"], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    plane.data.materials.append(mat)


def add_light(light: dict) -> None:
    data = bpy.data.lights.new("Sun", "SUN")
    data.energy = light.get("sun_strength", 3.0)
    obj = bpy.data.objects.new("Sun", data)
    obj.rotation_euler = [math.radians(a) for a in light.get("sun_rotation_deg", [50, 0, 30])]
    bpy.context.scene.collection.objects.link(obj)


def add_camera(cam: dict) -> None:
    data = bpy.data.cameras.new("Camera")
    data.lens = cam.get("lens", 35)
    obj = bpy.data.objects.new("Camera", data)
    obj.location = cam["location"]
    direction = Vector(cam["target"]) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj


def apply_skin_color(obj: bpy.types.Object, color: list[float]) -> None:
    mat = bpy.data.materials.new(f"{obj.name}_skin")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1.0)
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def add_sculpture(cfg: dict, total_frames: int, fps: int) -> None:
    with bpy.data.libraries.load(str(resolve(cfg["blend"])), link=False) as (src, dst):
        dst.objects = [cfg["object"]]
    obj = dst.objects[0]
    obj.name = cfg["name"]
    bpy.context.scene.collection.objects.link(obj)

    height = cfg.get("height", 2.0)
    obj.scale = (height, height, height)
    if cfg.get("skin_color"):
        apply_skin_color(obj, cfg["skin_color"])

    path = ConstantSpeedPath(cfg["path"]["points"])
    speed = cfg["path"]["speed"]
    offset = cfg["path"].get("start_offset", 0.0) * path.length
    for frame in range(1, total_frames + 1, KEY_INTERVAL):
        x, y = path.at(offset + speed * (frame - 1) / fps)
        obj.location = (x, y, 0.0)
        obj.keyframe_insert("location", frame=frame)

    turns = (total_frames - 1) / fps / cfg["spin_period_sec"] * cfg.get("spin_direction", 1)
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert("rotation_euler", index=2, frame=1)
    obj.rotation_euler = (0, 0, math.tau * turns)
    obj.keyframe_insert("rotation_euler", index=2, frame=total_frames)

    laps = speed * total_frames / fps / path.length
    print(f"{cfg['name']}: path {path.length:.1f} units, {laps:.2f} laps")


def configure_render(cfg: dict, args: argparse.Namespace, total_frames: int) -> None:
    scene = bpy.context.scene
    scene.render.fps = cfg["fps"]
    scene.frame_start, scene.frame_end = 1, total_frames
    w, h = cfg["resolution"]
    scale = 50 if args.preview else 100
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = scale
    scene.render.engine = cfg.get("engine", "BLENDER_EEVEE")
    if hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 16 if args.preview else cfg.get("samples", 64)

    if args.still is not None:
        scene.frame_set(args.still)
        scene.render.image_settings.file_format = "PNG"
    else:
        scene.render.image_settings.media_type = "VIDEO"
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Linear interpolation keeps spin constant and avoids overshoot between location keys.
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    fps = cfg["fps"]
    total_frames = round(cfg["duration_sec"] * fps)

    set_world(cfg["background"])
    add_ground(cfg.get("ground", {"color": cfg["background"]["color"]}))
    add_light(cfg.get("light", {}))
    add_camera(cfg["camera"])
    for sculpture in cfg["sculptures"]:
        add_sculpture(sculpture, total_frames, fps)
    configure_render(cfg, args, total_frames)

    if args.save_blend:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(args.out)
        bpy.ops.render.render(write_still=args.still is not None, animation=args.still is None)


if __name__ == "__main__":
    main()
