"""Typer CLI entry-point for the sculpture package."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import typer
from rich import print as rprint

app = typer.Typer(help="Sculpture image → 3-D wireframe pipeline.")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@app.command()
def run(
    photos: Path | None = typer.Option(None, "--photos", "-p",
                                           help="Directory of input images."),
    masked_dir: Path | None = typer.Option(None, "--masked-dir", "-m",
                                           help="Directory of pre-masked (bg-removed) images. "
                                                "Skips preprocessing entirely."),
    sculpture_id: str | None = typer.Option(None, "--sculpture-id", "-s",
                                           help="Explicit sculpture ID (auto-detected from "
                                                "directory name when not supplied)."),
    config: Path | None = typer.Option(None, "--config", "-c",
                                           help="Path to YAML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full reconstruction pipeline."""
    from sculpture.pipeline import run_pipeline
    from sculpture.utils.logging import setup_logging
    if verbose:
        setup_logging("DEBUG")
    result = run_pipeline(
        config_path=config,
        photos_dir=photos,
        masked_dir=masked_dir,
        sculpture_id=sculpture_id,
    )
    rprint("[bold green]Pipeline finished.[/bold green]")
    if result.get("wireframe_graph"):
        g = result["wireframe_graph"]
        rprint(f"  Wireframe: [cyan]{g.number_of_nodes()}[/cyan] nodes, "
               f"[cyan]{g.number_of_edges()}[/cyan] edges")


@app.command()
def preprocess_only(
    photos: Path = typer.Argument(..., help="Directory of input images."),
    output: Path = typer.Option(Path("data/processed"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
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
    viewer_path = (
        _project_root() / "data" / "output" / "renders" / "sculpture_playground_export.html"
    )

    target = viewer_path if viewer else notebook_path
    if viewer and not target.exists():
        rprint("[yellow]No exported HTML viewer found yet; opening the notebook instead.[/yellow]")
        target = notebook_path

    rprint(f"[bold green]Opening:[/bold green] {target}")
    subprocess.run(["open", str(target)], check=False)


@app.command("build-catalog")
def build_catalog_cmd(
    photos: Path = typer.Option(Path("photos"), "--photos", "-p", help="Directory of videos/frames."),
    out: Path = typer.Option(Path("data/repository"), "--out", "-o", help="Catalog output directory."),
    frame_samples: int = typer.Option(30, "--frame-samples", help="Max frame sample paths per sculpture."),
) -> None:
    """Build structured sculpture repository files (JSON + PKL)."""
    from sculpture.catalog import build_catalog, load_catalog, records_by_id, save_catalog

    root = _project_root()
    photos_dir = photos if photos.is_absolute() else root / photos
    out_dir = out if out.is_absolute() else root / out
    existing_json = out_dir / "sculpture_catalog.json"

    existing_records = None
    if existing_json.exists():
        existing_records = records_by_id(load_catalog(existing_json))

    catalog = build_catalog(
        root,
        photos_dir,
        max_frame_samples=max(1, frame_samples),
        existing_records=existing_records,
    )
    json_path, pkl_path = save_catalog(catalog, out_dir)

    rprint(f"[bold green]Catalog built.[/bold green] sculptures={catalog['sculpture_count']}")
    rprint(f"  JSON: [cyan]{json_path}[/cyan]")
    rprint(f"  PKL:  [cyan]{pkl_path}[/cyan]")


@app.command("add-sculpture")
def add_sculpture_cmd(
    sculpture_id: str = typer.Argument(..., help="Stable ID, usually the video stem."),
    title: str | None = typer.Option(None, "--title", help="Display title."),
    year: int | None = typer.Option(None, "--year", help="Creation year."),
    medium: str | None = typer.Option(None, "--medium", help="Material/medium."),
    dimensions: str | None = typer.Option(None, "--dimensions", help="Dimensions, e.g. 18 x 9 x 7 in."),
    tag: list[str] | None = typer.Option(None, "--tag", help="Repeatable tag field."),
    photography_date: str | None = typer.Option(None, "--photography-date", help="Photo/video date, YYYY-MM-DD."),
    notes: str | None = typer.Option(None, "--notes", help="Curatorial or workflow notes."),
    critic_description: str | None = typer.Option(None, "--critic-description", help="Optional art-critical description."),
    catalog_dir: Path = typer.Option(Path("data/repository"), "--catalog-dir", help="Catalog directory containing sculpture_catalog.json."),
) -> None:
    """Add or update sculpture metadata directly in the catalog."""
    from sculpture.catalog import load_catalog, save_catalog, upsert_sculpture

    root = _project_root()
    resolved_catalog_dir = catalog_dir if catalog_dir.is_absolute() else root / catalog_dir
    catalog_json = resolved_catalog_dir / "sculpture_catalog.json"

    if catalog_json.exists():
        catalog = load_catalog(catalog_json)
    else:
        catalog = {
            "generated_at_utc": "",
            "schema_version": "1.0",
            "project_root": str(root),
            "sculpture_count": 0,
            "sculptures": [],
        }

    catalog = upsert_sculpture(
        catalog,
        sculpture_id=sculpture_id,
        title=title,
        year=year,
        medium=medium,
        dimensions=dimensions,
        tags=tag,
        photography_date=photography_date,
        notes=notes,
        critic_description=critic_description,
    )
    json_path, pkl_path = save_catalog(catalog, resolved_catalog_dir)

    rprint(f"[bold green]Sculpture saved.[/bold green] id=[cyan]{sculpture_id}[/cyan]")
    rprint(f"  JSON: [cyan]{json_path}[/cyan]")
    rprint(f"  PKL:  [cyan]{pkl_path}[/cyan]")


@app.command("apple-vision-preflight")
def apple_vision_preflight_cmd(
    fix_permissions: bool = typer.Option(
        False,
        "--fix-permissions",
        help="Attempt to mark the mask_subject binary as executable if needed.",
    ),
) -> None:
    """Validate Apple Vision masking prerequisites and run a smoke test.

    This command fails early with actionable guidance if the local environment
    cannot run the Apple Vision subject masking binary.
    """
    import cv2
    import numpy as np

    root = _project_root()
    tool_dir = root / "tools" / "mask_subject"
    swift_source = tool_dir / "main.swift"
    binary = tool_dir / "mask_subject"

    checks_ok = True

    rprint("[bold]Apple Vision preflight[/bold]")
    rprint(f"  Project root: [cyan]{root}[/cyan]")
    rprint(f"  Tool dir:     [cyan]{tool_dir}[/cyan]")

    if sys.platform != "darwin":
        rprint("[bold red]Fail:[/bold red] Apple Vision masking requires macOS.")
        raise typer.Exit(code=1)

    if not swift_source.exists():
        checks_ok = False
        rprint(f"[bold red]Fail:[/bold red] Missing Swift source: [cyan]{swift_source}[/cyan]")

    if not binary.exists():
        checks_ok = False
        rprint(f"[bold red]Fail:[/bold red] Missing binary: [cyan]{binary}[/cyan]")
    elif not binary.is_file():
        checks_ok = False
        rprint(f"[bold red]Fail:[/bold red] Not a file: [cyan]{binary}[/cyan]")
    elif not binary.stat().st_mode & 0o111:
        if fix_permissions:
            binary.chmod(binary.stat().st_mode | 0o755)
            rprint(f"[yellow]Updated execute permission:[/yellow] [cyan]{binary}[/cyan]")
        else:
            checks_ok = False
            rprint(f"[bold red]Fail:[/bold red] Binary is not executable: [cyan]{binary}[/cyan]")

    xcode = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True)
    if xcode.returncode == 0:
        rprint(f"  Xcode path:   [cyan]{xcode.stdout.strip()}[/cyan]")
    else:
        checks_ok = False
        rprint("[bold red]Fail:[/bold red] Xcode command-line tools not configured.")

    if not checks_ok:
        rprint("\n[bold]Actionable guidance[/bold]")
        rprint("  1) Install/launch Xcode and accept licenses")
        rprint("  2) Build mask tool:")
        rprint("     [cyan]cd tools/mask_subject && swiftc main.swift -framework Vision -framework CoreImage -framework ImageIO -o mask_subject[/cyan]")
        rprint("  3) Ensure executable bit:")
        rprint("     [cyan]chmod +x tools/mask_subject/mask_subject[/cyan]")
        raise typer.Exit(code=1)

    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "preflight_input.jpg"
        out = Path(tmp) / "preflight_masked.png"

        img = np.full((96, 96, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (24, 24), (72, 72), (0, 0, 0), -1)
        cv2.imwrite(str(inp), img)

        proc = subprocess.run([str(binary), str(inp), str(out)], capture_output=True, text=True)
        if proc.returncode != 0:
            rprint("[bold red]Fail:[/bold red] mask_subject execution failed.")
            if proc.stderr.strip():
                rprint(f"  stderr: [yellow]{proc.stderr.strip()}[/yellow]")
            rprint("  Rebuild with:")
            rprint("  [cyan]cd tools/mask_subject && swiftc main.swift -framework Vision -framework CoreImage -framework ImageIO -o mask_subject[/cyan]")
            raise typer.Exit(code=1)

        if not out.exists():
            rprint("[bold red]Fail:[/bold red] mask_subject did not write output PNG.")
            raise typer.Exit(code=1)

        masked = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
        if masked is None:
            rprint("[bold red]Fail:[/bold red] Output PNG unreadable.")
            raise typer.Exit(code=1)

        if masked.ndim != 3 or masked.shape[2] != 4:
            rprint(
                "[bold red]Fail:[/bold red] Output PNG missing alpha channel "
                f"(shape={masked.shape})."
            )
            raise typer.Exit(code=1)

    rprint("[bold green]Apple Vision preflight passed.[/bold green]")


@app.command("import-edited-mesh")
def import_edited_mesh_cmd(
    sculpture_id: str = typer.Argument(..., help="Target sculpture ID."),
    mesh_path: Path = typer.Argument(..., help="Path to edited mesh file (PLY, OBJ, etc.)."),
    editor: str = typer.Option("Blender", "--editor", "-e", help="Editor tool name (e.g., Blender, Meshmixer)."),
    editor_notes: str | None = typer.Option(None, "--notes", "-n", help="Optional provenance notes about the edit."),
    source_mesh: str | None = typer.Option(None, "--source-mesh", "-s", help="Optional: original mesh path that was edited."),
    catalog_dir: Path = typer.Option(Path("data/repository"), "--catalog-dir", help="Catalog directory containing sculpture_catalog.json."),
) -> None:
    """Import a Blender-edited (or otherwise modified) mesh into the catalog.
    
    This makes Blender edits first-class assets, tracking provenance and enabling
    round-trip workflows where edited meshes become indexed sculpture assets.
    
    Example:
        python -m sculpture.cli import-edited-mesh adam \\
            data/output/adam/meshes/adam_edited_v2.ply \\
            --editor Blender \\
            --notes "Refined nose and ear details" \\
            --source-mesh data/output/adam/meshes/mesh.ply
    """
    from sculpture.catalog import import_edited_mesh, load_catalog, save_catalog

    root = _project_root()
    
    # Resolve paths
    resolved_catalog_dir = catalog_dir if catalog_dir.is_absolute() else root / catalog_dir
    resolved_mesh_path = mesh_path if mesh_path.is_absolute() else root / mesh_path
    catalog_json = resolved_catalog_dir / "sculpture_catalog.json"

    if not catalog_json.exists():
        rprint(f"[bold red]Error:[/bold red] Catalog not found at {catalog_json}")
        raise typer.Exit(code=1)

    try:
        catalog = load_catalog(catalog_json)
        catalog = import_edited_mesh(
            catalog,
            root,
            sculpture_id=sculpture_id,
            mesh_path=resolved_mesh_path,
            editor=editor,
            editor_notes=editor_notes,
            source_mesh=source_mesh,
        )
        json_path, pkl_path = save_catalog(catalog, resolved_catalog_dir)

        rprint(f"[bold green]Edited mesh imported.[/bold green] id=[cyan]{sculpture_id}[/cyan]")
        rprint(f"  Mesh:    [cyan]{resolved_mesh_path.name}[/cyan]")
        rprint(f"  Editor:  [cyan]{editor}[/cyan]")
        if editor_notes:
            rprint(f"  Notes:   [cyan]{editor_notes}[/cyan]")
        rprint(f"  Catalog: [cyan]{json_path}[/cyan]")
    except FileNotFoundError as e:
        rprint(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        rprint(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("blender-export")
def blender_export_cmd(
    sculpture_id: str = typer.Argument(..., help="Sculpture ID (e.g., 'adam', 'athena')."),
    open_blender: bool = typer.Option(False, "--open", "-o", help="Open Blender with the mesh pre-loaded (if available)."),
    export_dir: Path | None = typer.Option(None, "--export-dir", "-e", help="Custom export directory (default: data/blender_exports)."),
) -> None:
    """One-click Blender export: copy mesh and generate startup presets.
    
    Automates the handoff of reconstructed meshes to Blender by:
    1. Locating the mesh from the catalog or pipeline output
    2. Copying to a Blender-friendly export directory
    3. Creating optional startup presets/batch files
    4. Optionally launching Blender with the mesh pre-loaded
    
    Example:
        sculpture blender-export adam
        sculpture blender-export adam --open
        sculpture blender-export athena --export-dir ~/blender_projects/sculptures
    """
    from pathlib import Path as PathlibPath
    import shutil
    from sculpture.catalog import load_catalog

    root = _project_root()
    catalog_json = root / "data" / "repository" / "sculpture_catalog.json"
    export_base = PathlibPath(export_dir) if export_dir else root / "data" / "blender_exports"
    export_base.mkdir(parents=True, exist_ok=True)
    
    # Try to find mesh from catalog or pipeline output
    mesh_path = None
    
    # 1. Try catalog first (preferred for versioned meshes)
    if catalog_json.exists():
        try:
            catalog = load_catalog(catalog_json)
            sculptures = catalog.get("sculptures", [])
            sculpture_record = next((s for s in sculptures if s.get("sculpture_id") == sculpture_id), None)
            if sculpture_record:
                mesh_candidates = sculpture_record.get("meshes", [])
                if mesh_candidates:
                    # Use the first (most recent/original) mesh
                    candidate = PathlibPath(mesh_candidates[0])
                    if not candidate.is_absolute():
                        candidate = root / candidate
                    if candidate.exists():
                        mesh_path = candidate
        except Exception as e:
            rprint(f"[yellow]Warning:[/yellow] Could not load catalog: {e}")
    
    # 2. Try direct pipeline output path
    if not mesh_path:
        pipeline_mesh = root / "data" / "output" / sculpture_id / "meshes" / "mesh.ply"
        if pipeline_mesh.exists():
            mesh_path = pipeline_mesh
    
    if not mesh_path or not mesh_path.exists():
        rprint(f"[bold red]Error:[/bold red] Mesh not found for sculpture_id=[cyan]{sculpture_id}[/cyan]")
        rprint("  Ensure the pipeline has run: [cyan]sculpture run --photos photos/[sculpture]_frames[/cyan]")
        raise typer.Exit(code=1)
    
    # Create sculpture-specific export subdirectory
    sculpture_export_dir = export_base / sculpture_id
    sculpture_export_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy mesh to export directory
    export_mesh = sculpture_export_dir / f"{sculpture_id}_mesh.ply"
    shutil.copy2(mesh_path, export_mesh)
    rprint(f"[bold green]Mesh exported:[/bold green] [cyan]{export_mesh}[/cyan]")
    
    # Create startup preset Python script for Blender
    preset_script = sculpture_export_dir / f"{sculpture_id}_blender_import.py"
    preset_content = f'''"""
Auto-generated Blender import script for {sculpture_id}.
Place this file in Blender's startup folder or run manually in Blender's Python console.

For Blender 3.x+:
  1. Copy this file to ~/.config/blender/<VERSION>/scripts/startup/
  2. Restart Blender and the mesh will auto-load
  
Or run manually:
  1. Open Blender
  2. Go to: Scripting tab
  3. Open this file and press [Run Script]
"""

import bpy
from pathlib import Path

# Auto-import the mesh on startup
mesh_path = Path({repr(str(export_mesh))})

if mesh_path.exists():
    # Clear default cube
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Import PLY
    bpy.ops.wm.ply_import(filepath=str(mesh_path))
    print(f"✓ Imported: {{mesh_path}}")
    
    # Frame view on imported object
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = {{'area': area, 'region': region}}
                    bpy.ops.view3d.view_all(override, use_all_regions=False)
else:
    print(f"✗ Mesh not found: {{mesh_path}}")
'''
    
    preset_script.write_text(preset_content)
    rprint(f"[bold green]Preset generated:[/bold green] [cyan]{preset_script.name}[/cyan]")
    
    # Create a quick-start guide
    guide_file = sculpture_export_dir / f"{sculpture_id}_BLENDER_QUICKSTART.txt"
    guide_content = f"""BLENDER IMPORT GUIDE: {sculpture_id}
{'=' * 60}

Files in this directory:
  • {export_mesh.name}           — Reconstructed PLY mesh (ready to import)
  • {preset_script.name}  — Auto-import Python preset (optional)

OPTION 1: Manual Import (Fastest)
  1. Open Blender
  2. File → Import → PLY ({export_mesh.name})
  3. Start editing!

OPTION 2: Auto-Import Preset (Hands-free)
  1. Copy {preset_script.name} to Blender's startup folder:
     macOS:   ~/.config/blender/<VERSION>/scripts/startup/
     Windows: %APPDATA%\\Blender Foundation\\Blender\\<VERSION>\\scripts\\startup\\
     Linux:   ~/.config/blender/<VERSION>/scripts/startup/
  2. Restart Blender
  3. The mesh loads automatically with viewport framed

OPTION 3: Python Console (For batch operations)
  1. Open Blender
  2. Open the Scripting workspace (top menu)
  3. File → Open: {preset_script.name}
  4. Click [Run Script]
  5. Mesh imports and viewport frames

Next Steps:
  1. Edit the mesh in Blender (sculpt, smooth, refine geometry)
  2. Export back to PLY: File → Export → PLY
  3. Register the edited mesh in the catalog:
     sculpture import-edited-mesh {sculpture_id} <edited_mesh.ply> \\
         --notes "Your edit description" \\
         --source-mesh {export_mesh}

Tips:
  • Use Blender's sculpting tools (Dyntopo, Remesh, etc.)
  • Preserve mesh topology for best reconstruction compatibility
  • Save your .blend file in this directory for version control

For help: See README.md "Blender Integration" section
"""
    
    guide_file.write_text(guide_content)
    rprint(f"[bold green]Guide created:[/bold green] [cyan]{guide_file.name}[/cyan]")
    
    # Display next steps
    rprint("\n[bold cyan]── Next Steps ──[/bold cyan]")
    rprint(f"  1. [bold]Import in Blender[/bold] → File → Import → PLY → [cyan]{export_mesh.name}[/cyan]")
    rprint(f"  2. [bold]Edit in Blender[/bold] → sculpt, smooth, refine geometry")
    rprint(f"  3. [bold]Export back[/bold] → File → Export → PLY")
    rprint(f"  4. [bold]Register edit[/bold] → [cyan]sculpture import-edited-mesh {sculpture_id} <edited.ply> --notes '...'[/cyan]")
    
    # Show export directory
    rprint(f"\n[bold]Export location:[/bold] [cyan]{sculpture_export_dir}[/cyan]")
    rprint(f"  • Mesh:     [cyan]{export_mesh.name}[/cyan] (import this in Blender)")
    rprint(f"  • Preset:   [cyan]{preset_script.name}[/cyan] (optional auto-import)")
    rprint(f"  • Guide:    [cyan]{guide_file.name}[/cyan] (open in text editor)")
    
    # Optionally open Blender
    if open_blender:
        if sys.platform == "darwin":  # macOS
            blender_app = Path("/Applications/Blender.app")
            if blender_app.exists():
                rprint(f"\n[yellow]Launching Blender...[/yellow]")
                subprocess.run(["open", "-a", "Blender", str(export_mesh)], check=False)
            else:
                rprint("[yellow]Blender app not found at /Applications/Blender.app[/yellow]")
        elif sys.platform == "linux":
            subprocess.run(["blender", str(export_mesh)], check=False)
        elif sys.platform == "win32":
            subprocess.run(["blender.exe", str(export_mesh)], check=False)
