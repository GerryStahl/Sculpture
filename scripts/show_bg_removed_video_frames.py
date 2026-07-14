from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sculpture.config import PreprocessingConfig
from sculpture.preprocessing import preprocess_image


@dataclass
class ResultPaths:
    original: Path
    removed: Path
    side_by_side: Path


def extract_mid_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_idx = max(frame_count // 2, 0)
    if frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from: {video_path}")
    return frame


def remove_bg_and_save(video_path: Path, out_dir: Path, cfg: PreprocessingConfig) -> ResultPaths:
    bgr = extract_mid_frame(video_path)
    out = preprocess_image(bgr, cfg)

    if out.ndim == 3 and out.shape[2] == 4:
        alpha = out[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(out[:, :, :3], 255)
        removed = (
            out[:, :, :3].astype(np.float32) * alpha
            + white.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
    else:
        removed = out

    if removed.shape[:2] != bgr.shape[:2]:
        removed = cv2.resize(removed, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    side = np.concatenate([bgr, removed], axis=1)

    stem = video_path.stem.lower().replace(" ", "_")
    original_path = out_dir / f"{stem}_frame_original.jpg"
    removed_path = out_dir / f"{stem}_frame_bg_removed.jpg"
    side_path = out_dir / f"{stem}_frame_bg_removed_side_by_side.jpg"

    cv2.imwrite(str(original_path), bgr)
    cv2.imwrite(str(removed_path), removed)
    cv2.imwrite(str(side_path), side)

    return ResultPaths(original=original_path, removed=removed_path, side_by_side=side_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    photos_dir = repo_root / "photos"
    out_dir = repo_root / "data" / "output" / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessingConfig(max_size=4096, bg_removal="rembg", denoise_ksize=0)

    for name in ["adam.mp4", "athena.mp4"]:
        video_path = photos_dir / name
        if not video_path.exists():
            print(f"missing video: {video_path}")
            continue

        paths = remove_bg_and_save(video_path, out_dir, cfg)
        print(f"video: {name}")
        print(f"saved: {paths.original}")
        print(f"saved: {paths.removed}")
        print(f"saved: {paths.side_by_side}")


if __name__ == "__main__":
    main()
