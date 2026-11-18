"""Generate an abstract flowing HDRI (equirectangular, pale green and blue) for the background.

  python animation/make_hdri.py                      # writes data/animation/hdri/pale_flow.hdr
  python animation/make_hdri.py --seed 7 --width 2048

The pattern is built from warped sine waves of the 3-D view direction, so it wraps seamlessly
at the left/right edge. A soft bright lobe gives the scene a main light direction.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PALENESS = 0.5  # 0 = original tones, 1 = white

# sRGB palette stops, from pale green through mint white to pale blue.
PALETTE = np.array(
    [
        [0.66, 0.90, 0.78],
        [0.80, 0.96, 0.88],
        [0.93, 0.99, 0.97],
        [0.78, 0.91, 0.98],
        [0.62, 0.80, 0.94],
        [0.66, 0.90, 0.78],
    ]
)


def view_directions(width: int, height: int) -> np.ndarray:
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)], axis=-1)


def flow_field(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scalar field in [0, 1] with long, swirling bands."""
    q = p * 1.4
    for freq, amp in ((1.2, 0.8), (2.2, 0.35), (3.6, 0.12)):
        phase = rng.uniform(0, 2 * np.pi, size=3)
        q = q + amp * np.sin(freq * q[..., [1, 2, 0]] + phase)
    weights = rng.normal(size=3)
    weights /= np.linalg.norm(weights)
    return 0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * (q @ weights))


def colorize(t: np.ndarray) -> np.ndarray:
    stops = np.linspace(0, 1, len(PALETTE))
    srgb = np.stack([np.interp(t, stops, PALETTE[:, c]) for c in range(3)], axis=-1)
    srgb = srgb * (1 - PALENESS) + PALENESS
    return srgb**2.2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data/animation/hdri/pale_flow.hdr")
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    p = view_directions(args.width, args.width // 2)
    rgb = colorize(flow_field(p, rng))

    # Fine ridges along the bands add the "flowing" look without adding noise.
    ridge = np.exp(-(((flow_field(p * 1.15, rng) - 0.5) * 6) ** 2))
    rgb = rgb * (0.96 + 0.06 * ridge[..., None])

    # Soft light from upper right, slightly in front; gentle darkening toward the horizon below.
    light_dir = np.array([0.55, 0.65, -0.52])
    light_dir /= np.linalg.norm(light_dir)
    glow = np.exp(-((1 - p @ light_dir) / 0.04))
    rgb = rgb * (0.85 + 0.25 * np.clip(p[..., 1], -1, 1)[..., None]) + glow[..., None] * np.array(
        [1.2, 1.3, 1.2]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.ascontiguousarray(rgb[..., ::-1], dtype=np.float32))
    print(f"Wrote {args.out} ({args.width}x{args.width // 2}), peak {rgb.max():.2f}")


if __name__ == "__main__":
    main()
