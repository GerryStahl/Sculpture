"""Turn turntable videos into clean, animation-ready sculpture meshes.

Steps, per sculpture listed in animation/meshes.json:
  frames  extract turntable frames from the video
  mesh    build a textured 3-D model from the frames (Apple Object Capture, as in PhotoCatch)
  guide   render a front view with a line every 5% of height, to pick that sculpture's cut_frac
  prep    cut the stand off, drop floating pieces, normalise, and save <name>.blend + preview PNG

Typical use (from the project root, in the project virtualenv):
  python animation/make_meshes.py                      # frames + mesh + guide for all sculptures
  # look at data/blender_exports/<name>/<name>_guide.png, then set cut_frac in meshes.json
  python animation/make_meshes.py emergent4 --steps prep
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
PHOTOGRAMMETRY = ROOT / "tools" / "photogrammetry" / "photogrammetry"
EXTRACT_FRAMES = ROOT / "scripts" / "extract_turntable_frames.py"
PREP_MESH = ROOT / "animation" / "prep_mesh.py"
ALL_STEPS = ("frames", "mesh", "guide", "prep")
DEFAULT_STEPS = ("frames", "mesh", "guide")


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def blender_prep(args: list[str]) -> list[str]:
    return [BLENDER, "--background", "--python", str(PREP_MESH), "--", *args]


def process(name: str, cfg: dict, steps: set[str], force: bool) -> None:
    out_dir = ROOT / "data" / "blender_exports" / name
    frames_dir = ROOT / "data" / "photogrammetry" / name / "frames"
    raw_model = out_dir / f"{name}_raw.usdz"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "frames" in steps and (force or not frames_dir.exists()):
        cmd = [
            sys.executable,
            str(EXTRACT_FRAMES),
            "--video",
            cfg["video"],
            "--out",
            str(frames_dir),
            "--num-frames",
            str(cfg.get("num_frames", 120)),
        ]
        if cfg.get("start_sec") is not None:
            cmd += ["--start-sec", str(cfg["start_sec"])]
        if cfg.get("end_sec") is not None:
            cmd += ["--end-sec", str(cfg["end_sec"])]
        run(cmd)

    if "mesh" in steps and (force or not raw_model.exists()):
        if not PHOTOGRAMMETRY.exists():
            raise SystemExit(
                f"Missing {PHOTOGRAMMETRY}; build it with:\n"
                "  cd tools/photogrammetry && swiftc -O -swift-version 5 main.swift "
                "-framework RealityKit -o photogrammetry"
            )
        run([str(PHOTOGRAMMETRY), str(frames_dir), str(raw_model), cfg.get("detail", "full")])

    rotate = ["--rotate-x", str(cfg.get("rotate_x", 0))]
    if "guide" in steps:
        run(
            blender_prep(
                [str(raw_model), str(out_dir / f"{name}.blend"), *rotate]
                + ["--guide", str(out_dir / f"{name}_guide.png")]
            )
        )

    if "prep" in steps:
        run(
            blender_prep(
                [str(raw_model), str(out_dir / f"{name}.blend"), "--name", name, *rotate]
                + ["--height", str(cfg.get("height", 1.0))]
                + ["--cut-frac", str(cfg.get("cut_frac", 0.0))]
                + ["--preview", str(out_dir / f"{name}_preview.png")]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("names", nargs="*", help="Sculptures to process (default: all)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "animation" / "meshes.json")
    parser.add_argument(
        "--steps", default=",".join(DEFAULT_STEPS), help=f"Comma list of {ALL_STEPS}"
    )
    parser.add_argument("--force", action="store_true", help="Redo frames and mesh even if present")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    steps = set(args.steps.split(","))
    unknown = steps - set(ALL_STEPS)
    if unknown:
        raise SystemExit(f"Unknown steps: {sorted(unknown)}")

    names = args.names or list(manifest)
    for name in names:
        if name not in manifest:
            raise SystemExit(f"{name} is not in {args.manifest}")
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        process(name, manifest[name], steps, args.force)


if __name__ == "__main__":
    main()
