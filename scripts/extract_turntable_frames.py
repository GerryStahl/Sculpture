from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2  # noqa: E402

VIDEO_PATH = Path("photos/emergent4.mp4")
OUT_DIR = Path("photos/emergent4_frames")
NUM_FRAMES = 30


def compute_frame_indices(frame_count: int, num_frames: int = NUM_FRAMES) -> list[int]:
    """Compute standardized frame indices for a turntable video."""
    if frame_count <= 0:
        raise ValueError("Video has no frames")

    start_idx = int(frame_count * 0.05)
    end_idx = int(frame_count * 0.95)
    if end_idx <= start_idx:
        start_idx, end_idx = 0, frame_count - 1

    indices = [
        round(start_idx + i * (end_idx - start_idx) / (num_frames - 1))
        for i in range(num_frames)
    ]
    return [max(0, min(frame_count - 1, idx)) for idx in indices]


def extract_frames(video_path: Path, out_dir: Path, num_frames: int = NUM_FRAMES) -> list[dict[str, float | int | str | None]]:
    """Extract standardized frames and return manifest entries."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = compute_frame_indices(frame_count, num_frames)

        manifest: list[dict[str, float | int | str | None]] = []
        for i, frame_idx in enumerate(indices, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                print(f"Skipping unreadable frame {frame_idx}")
                continue

            out_path = out_dir / f"{video_path.stem}_{i:02d}_frame{frame_idx:06d}.jpg"
            cv2.imwrite(
                str(out_path),
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            manifest.append(
                {
                    "index": i,
                    "frame": frame_idx,
                    "time_s": frame_idx / fps if fps else None,
                    "file": str(out_path),
                }
            )
        return manifest
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a standardized 30-frame turntable set (~12° step)."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help="Path to input video (default: photos/emergent4.mp4)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="Output directory for extracted frames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path: Path = args.video
    out_dir: Path = args.out
    num_frames: int = NUM_FRAMES
    manifest = extract_frames(video_path, out_dir, num_frames)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        {
            "video": str(video_path),
            "frame_count": frame_count,
            "fps": fps,
            "saved_frames": len(manifest),
            "out_dir": str(out_dir),
            "manifest": str(manifest_path),
        }
    )


if __name__ == "__main__":
    main()
