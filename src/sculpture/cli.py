"""Typer CLI entry-point for the sculpture package."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

app = typer.Typer(help="Sculpture image → 3-D wireframe pipeline.")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


@app.command()
def playground(
    viewer: bool = typer.Option(
        False,
        "--viewer",
        help="Open the exported HTML viewer instead of the notebook.",
    ),
) -> None:
    """Launch the sculpture playground quickly.

    By default this opens the interactive notebook.
    Use --viewer to open the exported HTML preview if it already exists.
    """
    notebook_path = _project_root() / "notebooks" / "02_playground.ipynb"
    viewer_path = _project_root() / "data" / "output" / "renders" / "sculpture_playground_export.html"

    target = viewer_path if viewer else notebook_path
    if viewer and not target.exists():
        rprint("[yellow]No exported HTML viewer found yet; opening the notebook instead.[/yellow]")
        target = notebook_path

    rprint(f"[bold green]Opening:[/bold green] {target}")
    subprocess.run(["open", str(target)], check=False)


if __name__ == "__main__":
    app()
