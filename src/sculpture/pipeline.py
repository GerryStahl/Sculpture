"""End-to-end sculpture reconstruction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from sculpture.config import SculptureConfig, load_config
from sculpture.io.image_io import collect_images, load_image, save_image
from sculpture.meshing import build_mesh
from sculpture.preprocessing import preprocess_image
from sculpture.reconstruction import reconstruct
from sculpture.utils.logging import setup_logging
from sculpture.wireframe import extract_wireframe

logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: Path | str | None = None,
    photos_dir: Path | str | None = None,
) -> dict:
    """Execute the full image → wireframe pipeline.

    Args:
        config_path: Optional path to a custom YAML config.
        photos_dir:  Override the photos directory from config.

    Returns:
        Dict with keys ``point_cloud``, ``mesh``, ``wireframe_graph``.
    """
    cfg = load_config(config_path)

    # ── Logging ────────────────────────────────────────────────────────────
    setup_logging(cfg.logging.level, cfg.logging.file)
    logger.info("=" * 60)
    logger.info("Sculpture pipeline starting")
    logger.info("=" * 60)

    # ── Paths ───────────────────────────────────────────────────────────────
    root = Path(config_path).parent.parent if config_path else Path.cwd()
    photos = Path(photos_dir) if photos_dir else root / cfg.paths.photos_dir
    output_dir = root / cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load images ──────────────────────────────────────────────────────
    logger.info("Step 1/4 – Loading images from %s", photos)
    image_paths = collect_images(photos)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {photos}")

    raw_images = [load_image(p) for p in image_paths]
    logger.info("Loaded %d image(s)", len(raw_images))

    # ── 2. Preprocess ───────────────────────────────────────────────────────
    logger.info("Step 2/4 – Preprocessing images")
    processed: list = []
    proc_dir = root / cfg.paths.data_processed
    proc_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, src_path) in enumerate(zip(raw_images, image_paths)):
        logger.info("  Preprocessing %s …", src_path.name)
        processed_img = preprocess_image(img, cfg.preprocessing)
        out_path = proc_dir / f"{src_path.stem}_processed.png"
        save_image(processed_img[:, :, :3] if processed_img.ndim == 3
                   and processed_img.shape[2] == 4 else processed_img, out_path)
        processed.append(processed_img)

    # ── 3. Reconstruct point cloud ──────────────────────────────────────────
    logger.info("Step 3/4 – Reconstructing point cloud (%s)", cfg.reconstruction.method)
    recon_dir = output_dir / "reconstruction"
    pcd = reconstruct(processed, cfg.reconstruction, recon_dir)

    if len(pcd.points) == 0:
        logger.error("Point cloud is empty – pipeline cannot continue.")
        return {"point_cloud": pcd, "mesh": None, "wireframe_graph": None}

    # ── 4. Mesh + wireframe ─────────────────────────────────────────────────
    logger.info("Step 4/4 – Building mesh and extracting wireframe")
    mesh = build_mesh(pcd, cfg.meshing, output_dir / "meshes")
    wf_graph = extract_wireframe(mesh, cfg.wireframe, output_dir / "wireframes")

    logger.info("Pipeline complete.")
    logger.info("  Point cloud : %d points", len(pcd.points))
    logger.info("  Mesh        : %d vertices, %d triangles",
                len(mesh.vertices), len(mesh.triangles))
    logger.info("  Wireframe   : %d nodes, %d edges",
                wf_graph.number_of_nodes(), wf_graph.number_of_edges())

    return {"point_cloud": pcd, "mesh": mesh, "wireframe_graph": wf_graph}
