from pathlib import Path

import cv2
import numpy as np

from sculpture.config import PreprocessingConfig
from sculpture.preprocessing import preprocess_image

frame_path = Path("/Users/GStahl2/AI/sculpture/photos/birdinspace_frames/birdinspace_15_frame000676.jpg")
out_dir = Path("/Users/GStahl2/AI/sculpture/data/output/renders")
out_dir.mkdir(parents=True, exist_ok=True)

bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
if bgr is None:
    raise FileNotFoundError(f"Could not read frame: {frame_path}")

cfg = PreprocessingConfig(max_size=4096, bg_removal="rembg", denoise_ksize=0)
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

side_by_side = np.concatenate([bgr, removed], axis=1)

orig_path = out_dir / "birdinspace_frame_original.jpg"
removed_path = out_dir / "birdinspace_frame_bg_removed.jpg"
side_path = out_dir / "birdinspace_frame_bg_removed_side_by_side.jpg"

cv2.imwrite(str(orig_path), bgr)
cv2.imwrite(str(removed_path), removed)
cv2.imwrite(str(side_path), side_by_side)

print(f"saved: {orig_path}")
print(f"saved: {removed_path}")
print(f"saved: {side_path}")
