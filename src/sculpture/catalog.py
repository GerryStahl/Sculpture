"""Structured sculpture repository catalog (PKL + JSON)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
from typing import Any


@dataclass
class EditHistory:
    """Track provenance of mesh edits."""
    timestamp: str  # ISO format UTC
    editor: str  # e.g., "Blender", "Meshmixer"
    source_mesh: str | None  # original mesh path that was edited
    edited_mesh: str  # output mesh path
    editor_notes: str | None = None
    version: int = 1  # edit iteration


@dataclass
class SculptureRecord:
    sculpture_id: str
    title: str
    year: int | None
    medium: str | None
    dimensions: str | None
    tags: list[str]
    photography_date: str | None
    source_video: str | None
    frames_dir: str | None
    frame_count: int
    frame_samples: list[str]
    masked_samples: list[str]
    side_by_side_samples: list[str]
    meshes: list[str]
    wireframes: list[str]
    point_clouds: list[str]
    mesh_thumbnails: list[str]
    wireframe_thumbnails: list[str]
    notes: str | None
    critic_description: str | None
    edited_meshes: list[str] | None = None  # Blender edits, imports, etc.
    edit_history: list[dict] | None = None  # Provenance tracking


METADATA_KEYS = [
    "title",
    "year",
    "medium",
    "dimensions",
    "tags",
    "photography_date",
    "critic_description",
]

ASSET_KEYS = [
    "edited_meshes",
    "edit_history",
    "masked_frames",
    "masked_frames_preprocessed_date",
    "mesh_thumbnails",
    "wireframe_thumbnails",
]

AUTO_FRAME_NOTE = "Frame set not extracted yet; run extract_turntable_frames.py for full per-view set."


def _rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _existing(paths: list[Path], root: Path) -> list[str]:
    return [_rel(p, root) for p in paths if p.exists()]


def _file_date(path: Path) -> str | None:
    if not path.exists():
        return None
    dt = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return dt.date().isoformat()


def _critic_descriptions() -> dict[str, str]:
    return {
        "birdinspace": (
            "A disciplined vertical form that reads as aerodynamic and architectural in the round, "
            "with subtle torsion and a strong contrast between luminous body and dark base."
        ),
        "emergent4": (
            "An abstract, interlocking composition where voids are as active as mass; "
            "the sculpture unfolds through rotation as a choreography of compression and release."
        ),
        "adam": (
            "Grounded and block-like, this torso emphasizes tactile surface and weight over idealization, "
            "creating a monolithic, primordial presence."
        ),
        "athena": (
            "More lyrical and figurative in silhouette, with asymmetrical transitions that animate the form "
            "across angles while maintaining a composed classical poise."
        ),
    }


def load_catalog(catalog_path: Path) -> dict[str, Any]:
    with catalog_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def records_by_id(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        record["sculpture_id"]: record
        for record in catalog.get("sculptures", [])
        if isinstance(record, dict) and record.get("sculpture_id")
    }


def _merged_metadata(existing: dict[str, Any] | None, generated: SculptureRecord) -> SculptureRecord:
    if not existing:
        return generated

    payload = asdict(generated)
    for key in METADATA_KEYS:
        if key in existing and existing[key] is not None:
            payload[key] = existing[key]
    if "tags" in existing and isinstance(existing["tags"], list):
        payload["tags"] = existing["tags"]
    existing_notes = existing.get("notes")
    if existing_notes and existing_notes != AUTO_FRAME_NOTE:
        payload["notes"] = existing_notes
    
    # Preserve asset keys (edited meshes, edit history, thumbnails, masks) from existing records
    for key in ASSET_KEYS:
        if key in existing and existing[key] is not None:
            payload[key] = existing[key]
    
    # Strip any keys from old catalog records that are not in the current dataclass
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(SculptureRecord)}
    payload = {k: v for k, v in payload.items() if k in valid_keys}
    
    return SculptureRecord(**payload)


def build_catalog(
    project_root: Path,
    photos_dir: Path,
    max_frame_samples: int = 30,
    existing_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    critics = _critic_descriptions()
    records: list[SculptureRecord] = []

    for video in sorted(photos_dir.glob("*.mp4")):
        sid = video.stem.lower().strip()
        frames_dir = photos_dir / f"{sid}_frames"
        review_dir = project_root / "data" / "output" / "renders" / f"review_{sid}"

        selected_frames_dir = frames_dir if frames_dir.exists() else (review_dir if review_dir.exists() else None)
        frame_files: list[Path] = []
        if selected_frames_dir:
            frame_files = sorted(
                [*selected_frames_dir.glob("*.jpg"), *selected_frames_dir.glob("*.png")]
            )

        renders = project_root / "data" / "output" / "renders"
        masked_candidates = [
            renders / f"{sid}_frame_masked.png",
            renders / f"{sid}_frame_bg_removed.png",
        ]
        side_by_side_candidates = [
            renders / f"{sid}_frame_side_by_side.jpg",
            renders / f"{sid}_frame_bg_removed_side_by_side.jpg",
        ]

        output = project_root / "data" / "output"
        # Primary location: sculpture-specific subdirectories
        mesh_candidates = [
            output / sid / "meshes" / "mesh.ply",
            output / f"{sid}_nobg" / "meshes" / "mesh.ply",
        ]
        wire_candidates = [
            output / sid / "wireframes" / "wireframe.obj",
            output / f"{sid}_nobg" / "wireframes" / "wireframe.obj",
        ]
        pcd_candidates = [
            output / sid / "reconstruction" / "point_cloud.ply",
            output / f"{sid}_nobg" / "reconstruction" / "point_cloud.ply",
        ]

        # Fallback: default output paths (for sculptures processed without ID detection)
        mesh_candidates.append(output / "meshes" / "mesh.ply")
        wire_candidates.append(output / "wireframes" / "wireframe.obj")
        pcd_candidates.append(output / "reconstruction" / "point_cloud.ply")

        # Thumbnail candidates (sculpture-specific then default fallback)
        mesh_thumb_candidates = [
            output / sid / "thumbnails" / "mesh_thumb.png",
            output / f"{sid}_nobg" / "thumbnails" / "mesh_thumb.png",
            output / "thumbnails" / "mesh_thumb.png",
        ]
        wire_thumb_candidates = [
            output / sid / "thumbnails" / "wireframe_thumb.png",
            output / f"{sid}_nobg" / "thumbnails" / "wireframe_thumb.png",
            output / "thumbnails" / "wireframe_thumb.png",
        ]

        rec = SculptureRecord(
            sculpture_id=sid,
            title=video.stem,
            year=None,
            medium=None,
            dimensions=None,
            tags=[],
            photography_date=_file_date(video),
            source_video=_rel(video, project_root),
            frames_dir=_rel(selected_frames_dir, project_root) if selected_frames_dir else None,
            frame_count=len(frame_files),
            frame_samples=_existing(frame_files[:max_frame_samples], project_root),
            masked_samples=_existing(masked_candidates, project_root),
            side_by_side_samples=_existing(side_by_side_candidates, project_root),
            meshes=_existing(mesh_candidates, project_root),
            wireframes=_existing(wire_candidates, project_root),
            point_clouds=_existing(pcd_candidates, project_root),
            mesh_thumbnails=_existing(mesh_thumb_candidates, project_root),
            wireframe_thumbnails=_existing(wire_thumb_candidates, project_root),
            notes=(
                AUTO_FRAME_NOTE
                if not frame_files else None
            ),
            critic_description=critics.get(sid),
        )
        records.append(_merged_metadata((existing_records or {}).get(sid), rec))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "project_root": str(project_root),
        "sculpture_count": len(records),
        "sculptures": [asdict(r) for r in records],
    }


def save_catalog(catalog: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sculpture_catalog.json"
    pkl_path = out_dir / "sculpture_catalog.pkl"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    with pkl_path.open("wb") as f:
        pickle.dump(catalog, f)

    return json_path, pkl_path


def upsert_sculpture(
    catalog: dict[str, Any],
    *,
    sculpture_id: str,
    title: str | None = None,
    year: int | None = None,
    medium: str | None = None,
    dimensions: str | None = None,
    tags: list[str] | None = None,
    photography_date: str | None = None,
    notes: str | None = None,
    critic_description: str | None = None,
) -> dict[str, Any]:
    records = catalog.setdefault("sculptures", [])
    existing = None
    for record in records:
        if record.get("sculpture_id") == sculpture_id:
            existing = record
            break

    if existing is None:
        existing = {
            "sculpture_id": sculpture_id,
            "title": title or sculpture_id,
            "year": year,
            "medium": medium,
            "dimensions": dimensions,
            "tags": tags or [],
            "photography_date": photography_date,
            "source_video": None,
            "frames_dir": None,
            "frame_count": 0,
            "frame_samples": [],
            "masked_samples": [],
            "side_by_side_samples": [],
            "meshes": [],
            "wireframes": [],
            "point_clouds": [],
            "notes": notes,
            "critic_description": critic_description,
            "edited_meshes": [],
            "edit_history": [],
        }
        records.append(existing)

    if title is not None:
        existing["title"] = title
    if year is not None:
        existing["year"] = year
    if medium is not None:
        existing["medium"] = medium
    if dimensions is not None:
        existing["dimensions"] = dimensions
    if tags is not None:
        existing["tags"] = tags
    if photography_date is not None:
        existing["photography_date"] = photography_date
    if notes is not None:
        existing["notes"] = notes
    if critic_description is not None:
        existing["critic_description"] = critic_description

    catalog["sculpture_count"] = len(records)
    catalog["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return catalog


def import_edited_mesh(
    catalog: dict[str, Any],
    project_root: Path,
    *,
    sculpture_id: str,
    mesh_path: Path,
    editor: str = "Blender",
    editor_notes: str | None = None,
    source_mesh: str | None = None,
) -> dict[str, Any]:
    """Import a Blender-edited (or otherwise modified) mesh into the catalog.
    
    Args:
        catalog: Catalog dict to update
        project_root: Project root for relative path resolution
        sculpture_id: Target sculpture ID
        mesh_path: Path to the edited mesh file (PLY, OBJ, etc.)
        editor: Editor tool name (default: "Blender")
        editor_notes: Optional provenance notes
        source_mesh: Original mesh path that was edited (if known)
    
    Returns:
        Updated catalog dict
    
    Raises:
        FileNotFoundError: If mesh file doesn't exist
        ValueError: If sculpture not found in catalog
    """
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    records = catalog.get("sculptures", [])
    existing = None
    for record in records:
        if record.get("sculpture_id") == sculpture_id:
            existing = record
            break
    
    if existing is None:
        raise ValueError(f"Sculpture not found in catalog: {sculpture_id}")
    
    # Store the mesh as a relative path
    rel_mesh_path = _rel(mesh_path, project_root)
    
    # Ensure edited_meshes list exists
    if "edited_meshes" not in existing:
        existing["edited_meshes"] = []
    
    # Add mesh if not already present
    if rel_mesh_path not in existing["edited_meshes"]:
        existing["edited_meshes"].append(rel_mesh_path)
    
    # Create edit history entry
    now = datetime.now(timezone.utc).isoformat()
    version = len(existing.get("edit_history", [])) + 1
    
    edit_entry = {
        "timestamp": now,
        "editor": editor,
        "source_mesh": source_mesh,
        "edited_mesh": rel_mesh_path,
        "editor_notes": editor_notes,
        "version": version,
    }
    
    if "edit_history" not in existing:
        existing["edit_history"] = []
    
    existing["edit_history"].append(edit_entry)
    
    catalog["generated_at_utc"] = now
    return catalog

