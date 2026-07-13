"""Typer CLI entry-point for the sculpture package."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(help="Sculpture image → 3-D wireframe pipeline.")


@app.command()
def run(
    photos: Optional[Path] = typer.Option(None, "--photos", "-p",
                                           help="Directory of input images."),
    config: Optional[Path] = typer.Option(None, "--config", "-c",
                                           help="Path to YAML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full reconstruction pipeline."""
    from sculpture.pipeline import run_pipeline
    from sculpture.utils.logging import setup_logging
    if verbose:
        setup_logging("DEBUG")
    result = run_pipeline(config_path=config, photos_dir=photos)
    rprint("[bold green]Pipeline finished.[/bold green]")
    if result.get("wireframe_graph"):
        g = result["wireframe_graph"]
        rprint(f"  Wireframe: [cyan]{g.number_of_nodes()}[/cyan] nodes, "
               f"[cyan]{g.number_of_edges()}[/cyan] edges")


@app.command()
def preprocess_only(
    photos: Path = typer.Argument(..., help="Directory of input images."),
    output: Path = typer.Option(Path("data/processed"), "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Preprocess images only (resize + background removal)."""
    from sculpture.config import load_config
    from sculpture.io.image_io import collect_images, load_image, save_image
    from sculpture.preprocessing import preprocess_image
    from sculpture.utils.logging import setup_logging

    setup_logging("INFO")
    cfg = load_config(config)
    output.mkdir(parents=True, exist_ok=True)

    for p in collect_images(photos):
        img = load_image(p)
        out = preprocess_image(img, cfg.preprocessing)
        save_path = output / f"{p.stem}_processed.png"
        save_image(out[:, :, :3] if out.ndim == 3 and out.shape[2] == 4 else out,
                   save_path)
        rprint(f"  Saved [cyan]{save_path}[/cyan]")


if __name__ == "__main__":
    app()
