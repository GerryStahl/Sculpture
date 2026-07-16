"""Tests for sculpture catalog module."""

import json
from pathlib import Path
import pickle
import tempfile
import pytest

from sculpture.catalog import (
    EditHistory,
    SculptureRecord,
    build_catalog,
    import_edited_mesh,
    load_catalog,
    save_catalog,
    upsert_sculpture,
)


@pytest.fixture
def temp_catalog_dir():
    """Create a temporary directory for catalog testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_catalog():
    """Create a sample catalog for testing."""
    return {
        "generated_at_utc": "2026-07-15T00:00:00",
        "schema_version": "1.0",
        "project_root": "/test/root",
        "sculpture_count": 1,
        "sculptures": [
            {
                "sculpture_id": "test_sculpture",
                "title": "Test Sculpture",
                "year": 2024,
                "medium": "Stone",
                "dimensions": "10 x 10 x 10",
                "tags": ["test"],
                "photography_date": "2024-01-01",
                "source_video": "test.mp4",
                "frames_dir": None,
                "frame_count": 0,
                "frame_samples": [],
                "masked_samples": [],
                "side_by_side_samples": [],
                "meshes": [],
                "wireframes": [],
                "point_clouds": [],
                "notes": None,
                "critic_description": None,
                "edited_meshes": [],
                "edit_history": [],
            }
        ],
    }


def test_edit_history_dataclass():
    """Test EditHistory dataclass creation."""
    edit = EditHistory(
        timestamp="2026-07-15T20:00:00",
        editor="Blender",
        source_mesh="original.ply",
        edited_mesh="edited.ply",
        editor_notes="Test edit",
        version=1,
    )
    assert edit.timestamp == "2026-07-15T20:00:00"
    assert edit.editor == "Blender"
    assert edit.version == 1


def test_sculpture_record_includes_edit_fields():
    """Test that SculptureRecord dataclass includes edited_meshes and edit_history."""
    rec = SculptureRecord(
        sculpture_id="test",
        title="Test",
        year=2024,
        medium="Stone",
        dimensions="10x10x10",
        tags=[],
        photography_date="2024-01-01",
        source_video=None,
        frames_dir=None,
        frame_count=0,
        frame_samples=[],
        masked_samples=[],
        side_by_side_samples=[],
        meshes=[],
        wireframes=[],
        point_clouds=[],
        mesh_thumbnails=[],
        wireframe_thumbnails=[],
        notes=None,
        critic_description=None,
        edited_meshes=[],
        edit_history=[],
    )
    assert hasattr(rec, "edited_meshes")
    assert hasattr(rec, "edit_history")
    assert hasattr(rec, "mesh_thumbnails")
    assert hasattr(rec, "wireframe_thumbnails")
    assert rec.edited_meshes == []
    assert rec.edit_history == []
    assert rec.mesh_thumbnails == []
    assert rec.wireframe_thumbnails == []


def test_import_edited_mesh_adds_entry(sample_catalog, temp_catalog_dir):
    """Test that import_edited_mesh creates edit history entry."""
    # Create a test mesh file
    mesh_file = temp_catalog_dir / "test_mesh.ply"
    mesh_file.write_text("ply\nformat ascii 1.0\nend_header\n")

    # Import the mesh
    updated_catalog = import_edited_mesh(
        sample_catalog,
        temp_catalog_dir,
        sculpture_id="test_sculpture",
        mesh_path=mesh_file,
        editor="Blender",
        editor_notes="Test import",
        source_mesh="original.ply",
    )

    # Verify the sculpture was updated
    sculpture = updated_catalog["sculptures"][0]
    assert len(sculpture["edited_meshes"]) == 1
    assert "test_mesh.ply" in sculpture["edited_meshes"][0]
    assert len(sculpture["edit_history"]) == 1

    # Check edit history entry
    edit = sculpture["edit_history"][0]
    assert edit["editor"] == "Blender"
    assert edit["editor_notes"] == "Test import"
    assert edit["version"] == 1
    assert edit["source_mesh"] == "original.ply"


def test_import_edited_mesh_nonexistent_file(sample_catalog, temp_catalog_dir):
    """Test that import_edited_mesh raises error for missing file."""
    with pytest.raises(FileNotFoundError):
        import_edited_mesh(
            sample_catalog,
            temp_catalog_dir,
            sculpture_id="test_sculpture",
            mesh_path=temp_catalog_dir / "nonexistent.ply",
        )


def test_import_edited_mesh_nonexistent_sculpture(sample_catalog, temp_catalog_dir):
    """Test that import_edited_mesh raises error for missing sculpture."""
    mesh_file = temp_catalog_dir / "test_mesh.ply"
    mesh_file.write_text("ply\nformat ascii 1.0\nend_header\n")

    with pytest.raises(ValueError, match="Sculpture not found"):
        import_edited_mesh(
            sample_catalog,
            temp_catalog_dir,
            sculpture_id="nonexistent",
            mesh_path=mesh_file,
        )


def test_import_edited_mesh_multiple_versions(sample_catalog, temp_catalog_dir):
    """Test version tracking for multiple edits."""
    # Create multiple mesh files
    mesh_v1 = temp_catalog_dir / "mesh_v1.ply"
    mesh_v2 = temp_catalog_dir / "mesh_v2.ply"
    mesh_v1.write_text("ply\nformat ascii 1.0\nend_header\n")
    mesh_v2.write_text("ply\nformat ascii 1.0\nend_header\n")

    # Import first version
    catalog = import_edited_mesh(
        sample_catalog,
        temp_catalog_dir,
        sculpture_id="test_sculpture",
        mesh_path=mesh_v1,
        editor="Blender",
        editor_notes="Initial edit",
    )

    # Import second version
    catalog = import_edited_mesh(
        catalog,
        temp_catalog_dir,
        sculpture_id="test_sculpture",
        mesh_path=mesh_v2,
        editor="Blender",
        editor_notes="Refinement",
    )

    # Verify version numbers
    sculpture = catalog["sculptures"][0]
    assert len(sculpture["edit_history"]) == 2
    assert sculpture["edit_history"][0]["version"] == 1
    assert sculpture["edit_history"][1]["version"] == 2
    assert len(sculpture["edited_meshes"]) == 2


def test_upsert_sculpture_includes_edit_fields():
    """Test that upsert_sculpture initializes edit tracking fields."""
    catalog = {
        "generated_at_utc": "",
        "schema_version": "1.0",
        "project_root": "/test",
        "sculpture_count": 0,
        "sculptures": [],
    }

    updated = upsert_sculpture(
        catalog,
        sculpture_id="new_sculpture",
        title="New Sculpture",
    )

    sculpture = updated["sculptures"][0]
    assert "edited_meshes" in sculpture
    assert "edit_history" in sculpture
    assert sculpture["edited_meshes"] == []
    assert sculpture["edit_history"] == []


def test_save_and_load_catalog_preserves_edits(sample_catalog, temp_catalog_dir):
    """Test that catalog serialization preserves edit history."""
    # Add edit data
    sample_catalog["sculptures"][0]["edited_meshes"] = ["mesh_v1.ply"]
    sample_catalog["sculptures"][0]["edit_history"] = [
        {
            "timestamp": "2026-07-15T20:00:00",
            "editor": "Blender",
            "source_mesh": "original.ply",
            "edited_mesh": "mesh_v1.ply",
            "editor_notes": "Test",
            "version": 1,
        }
    ]

    # Save and load
    save_catalog(sample_catalog, temp_catalog_dir)
    loaded = load_catalog(temp_catalog_dir / "sculpture_catalog.json")

    # Verify
    assert loaded["sculptures"][0]["edited_meshes"] == ["mesh_v1.ply"]
    assert len(loaded["sculptures"][0]["edit_history"]) == 1
    assert loaded["sculptures"][0]["edit_history"][0]["editor"] == "Blender"


def test_save_catalog_keeps_json_and_pkl_in_sync(sample_catalog, temp_catalog_dir):
    """Test that JSON and PKL outputs serialize the same catalog payload."""
    json_path, pkl_path = save_catalog(sample_catalog, temp_catalog_dir)

    with json_path.open(encoding="utf-8") as fh:
        json_payload = json.load(fh)
    with pkl_path.open("rb") as fh:
        pkl_payload = pickle.load(fh)

    assert json_payload == pkl_payload == sample_catalog
