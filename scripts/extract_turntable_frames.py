from __future__ import annotations

import json
from pathlib import Path

import cv2  # noqa: E402

VIDEO_PATH = Path("photos/emergent4.mp4")
OUT_DIR = Path("photos/emergent4_frames")
NUM_FRAMES = 30


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {VIDEO_PATH}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_count <= 0:
        raise SystemExit("Video has no frames")

    start_idx = int(frame_count * 0.05)
    end_idx = int(frame_count * 0.95)
    if end_idx <= start_idx:
        start_idx, end_idx = 0, frame_count - 1

    raw_indices = [
        round(start_idx + i * (end_idx - start_idx) / (NUM_FRAMES - 1))
        for i in range(NUM_FRAMES)
    ]
    indices = sorted({max(0, min(frame_count - 1, idx)) for idx in raw_indices})

    manifest = []
    for i, frame_idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"Skipping unreadable frame {frame_idx}")
            continue

        out_path = OUT_DIR / f"emergent4_{i:02d}_frame{frame_idx:06d}.jpg"
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

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        {
            "video": str(VIDEO_PATH),
            "frame_count": frame_count,
            "fps": fps,
            "saved_frames": len(manifest),
            "out_dir": str(OUT_DIR),
            "manifest": str(manifest_path),
        }
    )


if __name__ == "__main__":
    main()
