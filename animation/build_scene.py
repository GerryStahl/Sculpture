"""Blender script: build the animated sculpture scene from a JSON config and render it.

Run from the project root, with RUN="$BLENDER --background --python animation/build_scene.py --":
  $RUN animation/scene.json --still 1 --out data/animation/still.png    # one frame
  $RUN animation/scene.json --preview --out data/animation/preview.mp4  # fast low-res video
  $RUN animation/scene.json --out data/animation/final.mp4              # full render
  $RUN animation/scene.json --frames data/animation/frames              # resumable PNG frames
  $RUN animation/scene.json --save-blend data/animation/scene.blend     # open in Blender UI

For a full multi-hour render use animation/render_video.py, which resumes after interruptions.
"""

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_INTERVAL = 2  # frames between location keyframes


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--frames", type=Path, default=None, help="Render PNG frames here, skipping done"
    )
    p.add_argument("--preview", action="store_true", help="Half resolution, fewer samples")
    p.add_argument("--seconds", type=float, default=None, help="Render only the first N seconds")
    p.add_argument("--look", default=None, help="Skin set from animation/looks/<name>.json")
    p.add_argument("--still", type=int, default=None, help="Render one frame instead of the video")
    p.add_argument("--save-blend", type=Path, default=None)
    return p.parse_args(argv)


def apply_look(cfg: dict, name: str) -> None:
    """Give each sculpture the skin listed in animation/looks/<name>.json."""
    look = json.loads((PROJECT_ROOT / "animation" / "looks" / f"{name}.json").read_text())
    names = {s["name"] for s in cfg["sculptures"]} | {
        Path(s["blend"]).stem for s in cfg["sculptures"]
    }
    unknown = set(look) - names
    if unknown:
        raise SystemExit(f"Look '{name}' names sculptures not in the scene: {sorted(unknown)}")
    for sculpture in cfg["sculptures"]:
        mesh = Path(sculpture["blend"]).stem
        sculpture["skin"] = look.get(sculpture["name"], look.get(mesh, "original"))


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
    """Closed path parametrized by distance so speed is in scene units per second."""

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
    nodes, links = world.node_tree.nodes, world.node_tree.links
    bg = nodes["Background"]
    bg.inputs["Strength"].default_value = background.get("strength", 1.0)

    if background.get("hdri"):
        coords = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value[2] = math.radians(
            background.get("rotation_deg", 0)
        )
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(resolve(background["hdri"])))
        links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (*background["color"], 1.0)


def add_ground(ground: dict, engine: str) -> None:
    # Only Cycles has a real shadow catcher; other engines would show an opaque floor.
    if not ground.get("enabled", True) or (ground.get("shadow_catcher") and engine != "CYCLES"):
        return
    bpy.ops.mesh.primitive_plane_add(size=ground.get("size", 60))
    plane = bpy.context.active_object
    plane.name = "Ground"
    if ground.get("shadow_catcher"):
        plane.is_shadow_catcher = True
        return
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


# Skin presets. Colors are linear RGB. "ramp" blends two colors with noise for natural variation,
# "bump" adds surface grain, and "transmission" makes the surface see-through like glass.
SKINS = {
    "gold": {"color": (1.000, 0.766, 0.336), "metallic": 1.0, "roughness": 0.25},
    "bronze": {"color": (0.220, 0.110, 0.045), "metallic": 1.0, "roughness": 0.42},
    "copper": {"color": (0.955, 0.638, 0.538), "metallic": 1.0, "roughness": 0.30},
    "brass": {"color": (0.880, 0.780, 0.400), "metallic": 1.0, "roughness": 0.30},
    "tin": {"color": (0.800, 0.800, 0.780), "metallic": 1.0, "roughness": 0.40},
    "silver": {"color": (0.972, 0.960, 0.915), "metallic": 1.0, "roughness": 0.20},
    "chrome": {"color": (0.550, 0.556, 0.554), "metallic": 1.0, "roughness": 0.08},
    "patina": {
        "ramp": ((0.08, 0.30, 0.22), (0.30, 0.14, 0.07)),
        "noise_scale": 18,
        "metallic": 0.3,
        "roughness": 0.60,
        "bump": 0.25,
    },
    "clay": {
        "ramp": ((0.36, 0.13, 0.065), (0.13, 0.075, 0.045)),
        "blend": (0.42, 0.58),
        "noise_scale": 9,
        "metallic": 0.0,
        "roughness": 0.68,
        "bump": 0.2,
    },
    "stone": {
        "ramp": ((0.20, 0.19, 0.18), (0.36, 0.34, 0.31)),
        "noise_scale": 40,
        "metallic": 0.0,
        "roughness": 0.85,
        "bump": 0.35,
    },
    "glass": {
        "color": (0.95, 0.98, 1.0),
        "metallic": 0.0,
        "roughness": 0.03,
        "transmission": 1.0,
        "ior": 1.5,
    },
}


# Plain rough colors, also usable as skins (and as the per-part skins of a segmented sculpture).
SKINS.update(
    {
        name: {"color": rgb, "metallic": 0.0, "roughness": 0.92, "specular": 0.2}
        for name, rgb in {
            "red": (0.75, 0.02, 0.02),
            "orange": (0.90, 0.22, 0.01),
            "pink": (0.85, 0.22, 0.40),
            "yellow": (0.90, 0.65, 0.02),
            "green": (0.03, 0.35, 0.05),
            "blue": (0.02, 0.08, 0.65),
            "indigo": (0.04, 0.01, 0.22),
            "violet": (0.30, 0.04, 0.55),
            "black": (0.012, 0.012, 0.012),
            "white": (0.80, 0.80, 0.80),
        }.items()
    }
)


def skin_spec(skin: str | None, cfg: dict) -> dict:
    if skin in SKINS:
        return SKINS[skin]
    if skin in (None, "original") and cfg.get("skin_color"):
        return {"color": cfg["skin_color"], "metallic": 0.0, "roughness": 0.6}
    raise SystemExit(f"Unknown skin '{skin}'; use original or one of {sorted(SKINS)}")


def build_material(name: str, spec: dict) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = spec["metallic"]
    bsdf.inputs["Roughness"].default_value = spec["roughness"]
    if "specular" in spec:
        bsdf.inputs["Specular IOR Level"].default_value = spec["specular"]
    if "transmission" in spec:
        bsdf.inputs["Transmission Weight"].default_value = spec["transmission"]
        bsdf.inputs["IOR"].default_value = spec["ior"]

    noise = None
    if "ramp" in spec or "bump" in spec:
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = spec.get("noise_scale", 10)
    if "ramp" in spec:
        ramp = nodes.new("ShaderNodeValToRGB")
        (low, high) = spec["ramp"]
        ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = spec.get(
            "blend", (0.46, 0.56)
        )
        ramp.color_ramp.elements[0].color = (*low, 1.0)
        ramp.color_ramp.elements[1].color = (*high, 1.0)
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*spec["color"], 1.0)
    if "bump" in spec:
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = spec["bump"]
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def load_parts(obj: bpy.types.Object, cfg: dict) -> tuple[np.ndarray, int]:
    """Part number of every polygon, from the file written by segment_parts.py."""
    blend = resolve(cfg["blend"])
    parts_file = blend.with_name(f"{blend.stem}_parts.json")
    if not parts_file.exists():
        raise SystemExit(
            f"{parts_file.name} not found; run: python animation/segment_parts.py {blend.stem}"
        )
    parts = json.loads(parts_file.read_text())
    labels = np.array(parts["polygon_labels"], dtype=np.int32)
    if len(labels) != len(obj.data.polygons):
        raise SystemExit(f"{parts_file.name} does not match the mesh; run segment_parts.py again")
    return labels, parts["parts"]


def apply_part_skins(obj: bpy.types.Object, cfg: dict, skins: list[str]) -> None:
    """Give each segmented mass its own skin."""
    labels, n_parts = load_parts(obj, cfg)
    if len(skins) != n_parts:
        raise SystemExit(f"{cfg['name']} has {n_parts} parts but {len(skins)} skins were given")
    obj.data.materials.clear()
    for i, skin in enumerate(skins):
        obj.data.materials.append(build_material(f"{obj.name}_part{i}", skin_spec(skin, cfg)))
    obj.data.polygons.foreach_set("material_index", labels)
    obj.data.update()


def part_axis(points: np.ndarray, rim: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Growth axis and anchor for one mass, and whether it grows away from a one-sided joint."""
    centroid = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - centroid).T))
    joint = rim.mean(axis=0)
    offset = centroid - joint
    if np.linalg.norm(offset) > 0.5 * np.sqrt(values[-1]):
        return offset / np.linalg.norm(offset), joint, True
    return vectors[:, -1], centroid, False


def rotate_about(points: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angles) -> np.ndarray:
    """Rotate each point about an axis through pivot by its own angle (radians)."""
    v = points - pivot
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    return pivot + v * c + np.cross(axis, v) * s + axis * ((v @ axis)[:, None]) * (1 - c)


def add_part_motions(obj: bpy.types.Object, cfg: dict, fps: int) -> None:
    """Stretch segmented masses along their own axes, one shape key per motion.

    Each motion is {"part": n, "factor": 2.0, "start_sec": a, "end_sec": b} and optionally
    "return_sec": c to shrink back by then.
    Add "swing": {"angle": 35, "start_sec": a, "end_sec": b} to swing the grown mass to one side
    and back, then the other side and back, between a and b (it stays grown until b);
    "cycles": 2 repeats that sequence. Add "repeat": {"every_sec": T, "count": N} to run the
    whole motion again every T seconds.

    A mass joined on one side (like a head on a neck) grows away from that joint; a mass joined
    all around grows both ways from its middle. The surface where it meets its neighbors stays
    put and the change fades in smoothly from there.
    """
    motions = cfg.get("motions")
    if not motions:
        return
    labels, _ = load_parts(obj, cfg)
    mesh = obj.data
    count = len(mesh.vertices)
    co = np.empty(count * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)

    touching = [set() for _ in range(count)]
    for polygon, label in zip(mesh.polygons, labels):
        for vertex in polygon.vertices:
            touching[vertex].add(int(label))

    obj.shape_key_add(name="Basis")
    # Eased growth looks more natural than a constant rate; spin and travel stay linear.
    prefs = bpy.context.preferences.edit
    prefs.keyframe_new_interpolation_type = "BEZIER"
    for i, motion in enumerate(motions):
        part = motion["part"]
        inside = np.array([touching[v] == {part} for v in range(count)])
        rim = np.array([part in touching[v] and len(touching[v]) > 1 for v in range(count)])
        if not rim.any():
            raise SystemExit(f"{cfg['name']} part {part} has no joint with another part")
        axis, anchor, one_sided = part_axis(co[inside | rim], co[rim])
        along = (co - anchor) @ axis
        if one_sided:
            along = np.maximum(along, 0.0)
        reach = np.abs(along[inside]).max()
        gap = np.linalg.norm(co[:, None, :] - co[rim][None, :, :], axis=2).min(axis=1)
        t = np.clip(gap / (0.35 * reach), 0.0, 1.0)
        weight = t * t * (3 - 2 * t)
        shift = np.where(inside, (motion["factor"] - 1.0) * along * weight, 0.0)
        stretched = co + shift[:, None] * axis
        # A mass growing downward would sink into the floor, so the whole sculpture rises.
        stretched[:, 2] += max(0.0, -stretched[:, 2].min())

        key = obj.shape_key_add(name=f"motion{i}")
        key.data.foreach_set("co", stretched.ravel())
        repeat = motion.get("repeat", {"every_sec": 0.0, "count": 1})
        offsets = [k * repeat["every_sec"] for k in range(repeat["count"])]

        def keyframes(shape_key, points):
            for offset in offsets:
                for sec, value in points:
                    shape_key.value = value
                    shape_key.keyframe_insert("value", frame=round((sec + offset) * fps) + 1)

        swing = motion.get("swing")
        grow = [(motion["start_sec"], 0.0), (motion["end_sec"], 1.0)]
        if swing:
            grow.append((swing["end_sec"], 1.0))
        if "return_sec" in motion:
            grow.append((motion["return_sec"], 0.0))
        keyframes(key, grow)

        if swing:
            # Swing about the joint: sideways for a mass that points more along than up.
            up = np.array([0.0, 0.0, 1.0])
            hinge = up if abs(axis[2]) < 0.7 else np.cross(axis, up)
            hinge = hinge / np.linalg.norm(hinge)
            cycles = swing.get("cycles", 1)
            quarter = (swing["end_sec"] - swing["start_sec"]) / (4 * cycles)
            for side, sign in (("a", 1.0), ("b", -1.0)):
                angles = np.where(inside, np.radians(swing["angle"]) * sign * weight, 0.0)
                swung = rotate_about(stretched, anchor, hinge, angles)
                swing_key = obj.shape_key_add(name=f"swing{i}{side}")
                # Offset from the grown shape, so it adds on top of the growth key; the whole
                # sculpture also moves so its lowest point stays on the floor.
                settle = np.array([0.0, 0.0, -swung[:, 2].min()])
                swing_key.data.foreach_set("co", (co + (swung - stretched) + settle).ravel())
                points = []
                for cycle in range(cycles):
                    first = swing["start_sec"] + (4 * cycle + (0 if side == "a" else 2)) * quarter
                    points += [(first, 0.0), (first + quarter, 1.0), (first + 2 * quarter, 0.0)]
                keyframes(swing_key, points)
    prefs.keyframe_new_interpolation_type = "LINEAR"


def apply_skin(obj: bpy.types.Object, cfg: dict) -> None:
    """Replace the original texture with a preset, color or per-part skins; keep it if unset."""
    skin = cfg.get("skin")
    if isinstance(skin, list):
        apply_part_skins(obj, cfg, skin)
        return
    if skin in (None, "original") and not cfg.get("skin_color"):
        return
    obj.data.materials.clear()
    obj.data.materials.append(build_material(f"{obj.name}_skin", skin_spec(skin, cfg)))


def path_track(cfg: dict, total_frames: int, fps: int) -> list[tuple[float, float]]:
    """Positions at every location keyframe for a sculpture that follows a fixed closed path."""
    path = ConstantSpeedPath(cfg["path"]["points"])
    speed = cfg["path"]["speed"]
    offset = cfg["path"].get("start_offset", 0.0) * path.length
    laps = speed * total_frames / fps / path.length
    print(f"{cfg['name']}: path {path.length:.1f} units, {laps:.2f} laps")
    return [
        path.at(offset + speed * (frame - 1) / fps)
        for frame in range(1, total_frames + 1, KEY_INTERVAL)
    ]


def traffic_tracks(cfg: dict, total_frames: int, fps: int) -> list[list[tuple[float, float]]]:
    """Positions for all sculptures when they move as lane traffic."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from traffic import Lane, simulate

    lanes = [Lane(lane["y"], lane["half"]) for lane in cfg["traffic"]["lanes"]]
    tracks, changes = simulate(
        lanes,
        cfg["sculptures"],
        (total_frames - 1) / fps,
        KEY_INTERVAL / fps,
        cfg["traffic"].get("clearance_scale", 1.0),
    )
    for name, count in changes.items():
        print(f"{name}: {count} lane changes")
    return [
        tracks[s["name"]][: len(range(1, total_frames + 1, KEY_INTERVAL))]
        for s in cfg["sculptures"]
    ]


def add_sculpture(cfg: dict, track: list[tuple[float, float]], total_frames: int, fps: int) -> None:
    """Add one sculpture that visits `track` (one position per location keyframe) and spins."""
    with bpy.data.libraries.load(str(resolve(cfg["blend"])), link=False) as (src, dst):
        dst.objects = [cfg["object"]]
    obj = dst.objects[0]
    obj.name = cfg["name"]
    bpy.context.scene.collection.objects.link(obj)

    height = cfg.get("height", 2.0)
    obj.scale = (height, height, height)
    apply_skin(obj, cfg)
    add_part_motions(obj, cfg, fps)

    for frame, (x, y) in zip(range(1, total_frames + 1, KEY_INTERVAL), track):
        obj.location = (x, y, 0.0)
        obj.keyframe_insert("location", frame=frame)

    turns = (total_frames - 1) / fps / cfg["spin_period_sec"] * cfg.get("spin_direction", 1)
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert("rotation_euler", index=2, frame=1)
    obj.rotation_euler = (0, 0, math.tau * turns)
    obj.keyframe_insert("rotation_euler", index=2, frame=total_frames)


def report_close_passes(sculptures: list[dict], tracks: list[list[tuple[float, float]]]) -> None:
    """Warn when two sculptures come closer than about the width of the pair."""
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            closest = min(math.dist(a, b) for a, b in zip(tracks[i], tracks[j]))
            limit = 0.3 * (sculptures[i].get("height", 2.0) + sculptures[j].get("height", 2.0))
            status = "TOO CLOSE" if closest < limit else "ok"
            print(
                f"{sculptures[i]['name']} / {sculptures[j]['name']}: "
                f"closest {closest:.2f} (limit {limit:.2f}) {status}"
            )


def enable_metal_gpu() -> None:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type != "CPU"


def configure_render(cfg: dict, args: argparse.Namespace, total_frames: int) -> None:
    scene = bpy.context.scene
    scene.render.fps = cfg["fps"]
    scene.frame_start, scene.frame_end = 1, total_frames
    w, h = cfg["resolution"]
    scale = 50 if args.preview else 100
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = scale
    scene.render.engine = cfg.get("engine", "BLENDER_EEVEE")
    scene.view_settings.view_transform = cfg.get("view_transform", "Standard")
    samples = 16 if args.preview else cfg.get("samples", 64)
    if scene.render.engine == "CYCLES":
        enable_metal_gpu()
        scene.cycles.device = "GPU"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.render.use_persistent_data = True
    elif hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = samples

    if args.still is not None or args.frames:
        scene.render.image_settings.file_format = "PNG"
        if args.still is not None:
            scene.frame_set(args.still)
        else:
            scene.render.use_overwrite = False
            scene.render.use_placeholder = True
    else:
        scene.render.image_settings.media_type = "VIDEO"
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    if args.look:
        apply_look(cfg, args.look)
    for sculpture in cfg["sculptures"]:
        if isinstance(sculpture.get("motions"), str):
            sculpture["motions"] = cfg["motion_sets"][sculpture["motions"]]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Linear interpolation keeps spin constant and avoids overshoot between location keys.
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    fps = cfg["fps"]
    total_frames = round((args.seconds or cfg["duration_sec"]) * fps)

    set_world(cfg["background"])
    default_ground = {"color": cfg["background"].get("color", [0.8, 0.8, 0.8])}
    add_ground(cfg.get("ground", default_ground), cfg.get("engine", "BLENDER_EEVEE"))
    add_light(cfg.get("light", {}))
    add_camera(cfg["camera"])
    if "traffic" in cfg:
        tracks = traffic_tracks(cfg, total_frames, fps)
    else:
        tracks = [path_track(s, total_frames, fps) for s in cfg["sculptures"]]
    for sculpture, track in zip(cfg["sculptures"], tracks):
        add_sculpture(sculpture, track, total_frames, fps)
    report_close_passes(cfg["sculptures"], tracks)
    configure_render(cfg, args, total_frames)

    if args.save_blend:
        args.save_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend))
    if args.frames:
        args.frames.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(args.frames / "frame_")
        bpy.ops.render.render(animation=True)
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(args.out)
        bpy.ops.render.render(write_still=args.still is not None, animation=args.still is None)


if __name__ == "__main__":
    main()
