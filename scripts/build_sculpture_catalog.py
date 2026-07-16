"""Build a structured sculpture catalog (JSON + PKL)."""

from __future__ import annotations

import argparse
from pathlib import Path

from sculpture.catalog import build_catalog, save_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sculpture catalog files.")
    parser.add_argument("--photos", type=Path, default=Path("photos"), help="Photos/video directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/repository"),
        help="Output directory for catalog files",
    )
    parser.add_argument(
        "--frame-samples",
        type=int,
        default=30,
        help="Max frame sample paths per sculpture",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    photos_dir = (project_root / args.photos).resolve() if not args.photos.is_absolute() else args.photos
    out_dir = (project_root / args.out).resolve() if not args.out.is_absolute() else args.out

    catalog = build_catalog(
        project_root=project_root,
        photos_dir=photos_dir,
        max_frame_samples=max(1, args.frame_samples),
    )
    json_path, pkl_path = save_catalog(catalog, out_dir)

    print(f"sculptures: {catalog['sculpture_count']}")
    print(f"json: {json_path}")
    print(f"pkl:  {pkl_path}")


if __name__ == "__main__":
    main()
