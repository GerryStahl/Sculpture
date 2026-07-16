from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "data" / "output" / "renders"
sets = ["review_birdinspace", "review_emergent4", "review_adam", "review_athena"]

for name in sets:
    d = base / name
    imgs = sorted([p for p in d.glob("*.jpg")])
    if not imgs:
        continue
    selected = imgs[:16]
    thumbs = []
    for p in selected:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        target_w = 320
        target_h = int(h * (target_w / w))
        thumbs.append(cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        continue
    rows = []
    cols = 4
    pad = 6
    for r in range(0, len(thumbs), cols):
        row_imgs = thumbs[r:r+cols]
        h = max(x.shape[0] for x in row_imgs)
        fixed = []
        for x in row_imgs:
            if x.shape[0] != h:
                x = cv2.resize(x, (x.shape[1], h), interpolation=cv2.INTER_AREA)
            fixed.append(x)
        while len(fixed) < cols:
            fixed.append(np.full_like(fixed[0], 255))
        row = cv2.hconcat(fixed)
        rows.append(row)
    sheet = cv2.vconcat(rows)
    out = base / f"{name}_contact_sheet.jpg"
    cv2.imwrite(str(out), sheet)
    print(out)
