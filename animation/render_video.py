"""Render the scene to an MP4 in a way that survives interruptions.

  python animation/render_video.py                   # full render -> data/animation/final.mp4
  python animation/render_video.py --preview         # half-size quick check -> final_preview.mp4

Frames are written as PNGs to data/animation/frames_<name>/ and then encoded with ffmpeg.
If the run stops (crash, closed lid, Ctrl+C), run the same command again; finished frames are kept.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
BUILD_SCENE = ROOT / "animation" / "build_scene.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "animation" / "scene.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "animation" / "final.mp4")
    parser.add_argument("--preview", action="store_true", help="Half resolution, fewer samples")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    fps = cfg["fps"]
    total = round(cfg["duration_sec"] * fps)
    out = args.out.with_name(args.out.stem + "_preview.mp4") if args.preview else args.out
    frames_dir = out.parent / f"frames_{out.stem}"

    # caffeinate keeps the Mac awake for the length of the render.
    render = ["caffeinate", "-i", BLENDER, "--background", "--python", str(BUILD_SCENE), "--"]
    render += [str(args.config), "--frames", str(frames_dir)]
    if args.preview:
        render.append("--preview")
    subprocess.run(render, cwd=ROOT, check=True)

    missing = [n for n in range(1, total + 1) if not (frames_dir / f"frame_{n:04d}.png").exists()]
    if missing:
        sys.exit(f"{len(missing)} frames missing (first: {missing[0]}); run this command again")

    out.parent.mkdir(parents=True, exist_ok=True)
    encode = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png")]
    encode += ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(encode, check=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
