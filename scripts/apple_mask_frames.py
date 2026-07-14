"""Extract one frame from a video, apply Apple Vision subject mask, save results."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


MASK_TOOL = Path(__file__).resolve().parents[1] / "tools" / "mask_subject" / "mask_subject"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "output" / "renders"


def extract_mid_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(total // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame from: {video_path}")
    return frame


def apple_mask(input_jpg: Path, output_png: Path) -> None:
    result = subprocess.run(
        [str(MASK_TOOL), str(input_jpg), str(output_png)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mask_subject failed:\n{result.stderr}")
    print(result.stdout.strip())


def side_by_side_png(original_bgr: np.ndarray, masked_png: Path) -> Path:
    import cv2 as _cv2
    masked_bgra = _cv2.imread(str(masked_png), _cv2.IMREAD_UNCHANGED)
    if masked_bgra is None:
        raise RuntimeError(f"Cannot read masked image: {masked_png}")

    # Composite masked image on white background for display
    if masked_bgra.ndim == 3 and masked_bgra.shape[2] == 4:
        alpha = masked_bgra[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(masked_bgra[:, :, :3], 255)
        composited = (
            masked_bgra[:, :, :3].astype(np.float32) * alpha
            + white.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
    else:
        composited = masked_bgra[:, :, :3]

    # Resize composited to match original height if needed
    if composited.shape[0] != original_bgr.shape[0]:
        composited = _cv2.resize(
            composited,
            (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=_cv2.INTER_LINEAR,
        )

    side = np.concatenate([original_bgr, composited], axis=1)
    stem = masked_png.stem.replace("_masked", "")
    out_path = masked_png.parent / f"{stem}_side_by_side.jpg"
    _cv2.imwrite(str(out_path), side)
    return out_path


def process(video_name: str) -> None:
    video_path = Path(__file__).resolve().parents[1] / "photos" / video_name
    if not video_path.exists():
        print(f"[skip] not found: {video_path}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem.lower()

    print(f"\n── {video_name} ──────────────────────────────────")
    bgr = extract_mid_frame(video_path)
    orig_path = OUT_DIR / f"{stem}_frame_original.jpg"
    cv2.imwrite(str(orig_path), bgr)
    print(f"original:  {orig_path}")

    masked_path = OUT_DIR / f"{stem}_frame_masked.png"
    apple_mask(orig_path, masked_path)

    side_path = side_by_side_png(bgr, masked_path)
    print(f"side-by-side: {side_path}")


if __name__ == "__main__":
    videos = sys.argv[1:] if len(sys.argv) > 1 else ["adam.mp4", "athena.mp4"]
    for v in videos:
        process(v)
