# Sculpture AI

An end-to-end Python pipeline for reconstructing 3-D wireframe models of sculptures from video or multi-view image sequences.

## Project Goals

** Primary Objective:** 
- This project models my sculptures and then manipulates the models. 

- The sculptures rotating 1 or 2 full turns on a turntable are captured by iPhone in mp4 videos. 

- The videos are loaded in PhotoCatch, where a first and last frame are selected and a mesh model is generated. Much of the procedure for creating meshes of sculpture was also developed in VSC, but PhotoCatch does a much faster and better job of it.

- Visual Studio Code (VSC) (with Claude) is used as a JSON repository of the videos, meshes and other project products. 

- The mesh model is loaded into Blender. The Claude App assists in performing tasks in Blender. Blender can be used to manipulate and pose the sculpture mesh model, apply the original or new skins, add a background scene, distort or animate the sculpture, etc. 

**Key Targets:**
- Capture 20–40 images of a sculpture rotating on a turntable (covering 360° rotations)
- Automatically extract camera parameters via checkerboard calibration
- Reconstruct dense 3-D geometry using multi-view structure-from-motion (SfM)
- Mesh point clouds and simplify for downstream use
- Extract and export structural wireframes (sharp edges, creases, silhouettes)

## Pipeline Overview

### Technical Pipeline (Manual vs Automated)

This project follows the workflow below (matching the collection-to-Blender round-trip process).

1. **Add turntable videos (`.mp4`) to catalog (JSON)**
   - **Manual user action:** Record/copy `.mp4` files into `photos/`.
   - **Automated action:** Build or refresh catalog index.
   - **Project components:** `sculpture build-catalog`, [scripts/build_sculpture_catalog.py](scripts/build_sculpture_catalog.py), [src/sculpture/catalog.py](src/sculpture/catalog.py), [src/sculpture/cli.py](src/sculpture/cli.py)
   - **Status:** **Functioning**

2. **Frame extraction (default 120 frames over 360°, configurable)**
   - **Manual user action:** Choose source video(s).
   - **Automated action:** Extract a standardized set of evenly spaced frames (default: 120, ~3° angular step).
   - **Practical guidance:** Use ~60 frames for quick iteration and 120 for final/high-detail runs or difficult subjects.
   - **Project components:** [scripts/extract_turntable_frames.py](scripts/extract_turntable_frames.py)
   - **Status:** **Functioning**

3. **Background removal and store in catalog**
   - **Manual user action:** Ensure Apple Vision masking tool is built on macOS.
   - **Automated action:** Subject masking with Apple Vision during preprocessing + catalog indexing of masked outputs.
   - **Project components:** [src/sculpture/preprocessing.py](src/sculpture/preprocessing.py), [tools/mask_subject/main.swift](tools/mask_subject/main.swift), [scripts/apple_mask_frames.py](scripts/apple_mask_frames.py), [src/sculpture/catalog.py](src/sculpture/catalog.py)
   - **Status:** **Functioning**

4. **Mesh creation and store original mesh in catalog**
   - **Manual user action:** Trigger run from CLI.
   - **Automated action:** COLMAP sparse SfM → optional dense reconstruction (OpenMVS/COLMAP) → meshing → thumbnail generation → catalog asset discovery.
   - **Project components:** `sculpture run`, [src/sculpture/pipeline.py](src/sculpture/pipeline.py), [src/sculpture/reconstruction.py](src/sculpture/reconstruction.py), [src/sculpture/meshing.py](src/sculpture/meshing.py), [src/sculpture/catalog.py](src/sculpture/catalog.py)
   - **Status:** **Functioning**

5. **Export mesh to Blender**
   - **Manual user action:** Run `sculpture blender-export` command (one-click handoff).
   - **Automated action:** Copy mesh, generate startup presets, create quick-start guide, optionally launch Blender.
   - **Project components:** [src/sculpture/cli.py](src/sculpture/cli.py), [src/sculpture/meshing.py](src/sculpture/meshing.py)
   - **Status:** **Functioning (automated one-click export)**

6. **Use Blender to manipulate models**
   - **Manual user action:** Sculpt/edit in Blender; export edited mesh.
   - **Automated action:** None inside this repo (Blender is external).
   - **Project components:** External tool (Blender)
   - **Status:** **Manual external step (Functioning workflow)**

7. **Import manipulated models from Blender and store in catalog**
   - **Manual user action:** Provide edited mesh path and provenance metadata.
   - **Automated action:** Register edited mesh as first-class catalog asset with edit history and versioning.
   - **Project components:** `sculpture import-edited-mesh`, [src/sculpture/catalog.py](src/sculpture/catalog.py), [src/sculpture/cli.py](src/sculpture/cli.py), [tests/test_catalog.py](tests/test_catalog.py)
   - **Status:** **Functioning**

### Internal Reconstruction Stages (Automated)

Within steps 3–4, the pipeline automatically performs:

- Image preprocessing (resize / denoise / masking)
- Feature extraction and matching (COLMAP SIFT + geometric verification)
- Point cloud reconstruction (COLMAP SfM + optional dense reconstruction)
- Mesh generation (Poisson + cleanup + simplification)
- Wireframe extraction (feature/boundary edges)

### Step-by-Step Details

#### 1. **Preprocessing**
Standardizes input images and removes unwanted background:
- **Resize:** Constrains longest edge to 2048px (configurable)
- **Denoise:** Optional Gaussian blur to reduce noise
- **Background Removal:** 
   - **Apple Vision only** (macOS 14+): Uses `VNGenerateForegroundInstanceMaskRequest` for on-device ML subject isolation

#### 2. **Feature Matching**
Establishes correspondences between images:
- **SIFT (COLMAP):** Extracts robust keypoints across all views
- **Sequential Matching:** Prioritizes neighboring turntable frames (overlap configurable)
- **Geometric Verification:** Rejects outliers before bundle adjustment

#### 3. **3-D Reconstruction**
Lifts 2-D feature matches into 3-D space:
- **Sparse SfM (COLMAP):** Estimates camera poses + sparse points via mapper/bundle adjustment
- **Dense stage (optional):** Uses OpenMVS when available (`dense_backend: auto`), otherwise attempts COLMAP dense stereo
- **Automatic fallback chain:** If dense output is too small, the pipeline falls back to sparse COLMAP; if that is still too sparse, it can try the legacy ORB path as a last resort for a usable mesh/wireframe
- **Downsampling:** Voxel grid (size 5mm) reduces point count before meshing
- **Normals:** Estimates surface normals for mesh generation

#### 4. **Meshing**
Converts point cloud to watertight surface:
- **Poisson Reconstruction:** Implicit function fitting at depth 10 (default)
- **Component Filtering:** Removes isolated pieces < 1% of largest
- **Simplification:** Disabled by default for high-detail runs (`simplify_faces: 0`)
- **Cleanup:** Removes degenerate triangles and non-manifold edges

#### 5. **Wireframe Extraction**
Identifies and exports structural edges:
- **Feature Edges:** Dihedral angle > 30° (configurable)
- **Boundary Edges:** Edges with <2 adjacent triangles
- **Length Filter:** Removes edges < 0.5% of bounding box diagonal
- **Export Formats:** OBJ (line segments) or JSON (graph)

---

## Technology Stack

### Core Libraries
- **OpenCV** — Feature detection (ORB), camera calibration, image preprocessing
- **Open3D** — Point cloud processing, Poisson meshing, mesh utilities
- **NumPy / SciPy** — Numerical computing and linear algebra
- **scikit-image** — Advanced image filtering and morphology
- **Trimesh** — Mesh analysis and simplification
- **NetworkX** — Wireframe graph representation and export

### Configuration & Orchestration
- **Pydantic v2** — Type-safe configuration with validation
- **PyYAML** — Human-readable config files (YAML format)
- **Typer** — CLI framework for user-facing commands
- **Rich** — Terminal formatting and progress output

### Development
- **PyTest** — Unit testing framework
- **MLflow** — Experiment tracking (optional)
- **Ruff** — Code linting and formatting

### Media Support
- **Pillow / pillow-heif** — Image I/O with HEIC/HEIF support (native iPhone photo format)
- **imageio** — Video frame extraction

---

## Project Structure

```
sculpture/
├── config/
│   └── default.yaml              ← All pipeline parameters (YAML)
├── photos/
│   ├── emergent4g.HEIC           ← Sample single image
│   ├── emergent4.mp4             ← Sample turntable video
│   ├── adam.mp4                  ← Torso sculpture video
│   ├── athena.mp4                ← Torso sculpture video
│   └── emergent4_frames/         ← Extracted frames (generated)
├── data/
│   ├── raw/                      ← Original images
│   ├── processed/                ← Preprocessed images (RGB)
│   ├── calibration/              ← Camera intrinsics (JSON)
│   └── output/
│       ├── meshes/               ← Mesh PLY files
│       ├── wireframes/           ← Wireframe OBJ/JSON
│       ├── reconstruction/       ← Point cloud PLY
│       ├── thumbnails/           ← Mesh / wireframe preview PNGs
│       ├── renders/              ← Preview images
│       └── pipeline.log          ← Execution logs
├── src/sculpture/
│   ├── config.py                 ← Pydantic config loader
│   ├── io/image_io.py            ← HEIC-aware image loading
│   ├── preprocessing.py          ← Resize, denoise, bg removal
│   ├── calibration.py            ← Checkerboard camera calibration
│   ├── reconstruction.py         ← Multi-view SfM → point cloud
│   ├── meshing.py                ← Poisson/BPA/alpha-shape
│   ├── wireframe.py              ← Feature edges → NetworkX graph
│   ├── playground.py             ← 3D mesh manipulation tools
│   ├── pipeline.py               ← End-to-end orchestrator
│   └── cli.py                    ← Typer CLI commands
├── scripts/
│   ├── extract_turntable_frames.py    ← Extract frames from MP4
│   └── apple_mask_frames.py          ← Test Apple Vision masking on frames
├── tools/
│   └── mask_subject/
│       ├── main.swift            ← Apple Vision VisionKit CLI (Swift)
│       └── mask_subject          ← Compiled binary
├── notebooks/
│   ├── 01_explore.py             ← Exploration script
│   └── 02_playground.ipynb       ← Interactive 3D model editor + previews
├── tests/
│   ├── test_config.py            ← Config loading tests
│   ├── test_image_io.py          ← Image I/O tests
│   ├── test_catalog.py           ← Catalog sync + provenance tests
│   ├── test_extract_turntable_frames.py ← Frame extraction standard tests
│   ├── test_preprocessing.py     ← Preprocessing unit tests
│   ├── test_wireframe.py         ← Wireframe logic tests
│   └── conftest.py               ← Pytest configuration
├── pyproject.toml                ← Dependencies, build config, CLI entry-point
├── .gitignore                    ← Git exclusions
└── README.md                     ← This file
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- ffmpeg (for MP4 processing)

### Installation

```bash
# Clone and enter the workspace
cd ~/AI/sculpture

# Install all dependencies (including dev/test tools)
pip install -e ".[dev]"
```

### Running the Pipeline

#### From MP4 (Extract frames first)
```bash
# Extract default frame count (120)
python scripts/extract_turntable_frames.py

# Extract a custom frame count
python scripts/extract_turntable_frames.py --video photos/adam.mp4 --out photos/adam_frames60 --num-frames 60

# Run full pipeline on extracted frames
sculpture run --photos photos/emergent4_frames
```

#### From Curated Masked Frames
```bash
sculpture run --masked-dir data/processed/adam_masked120 --sculpture-id adam120
```

#### From Image Directory
```bash
sculpture run --photos photos/emergent4_frames
```

#### Preprocessing Only
```bash
sculpture preprocess-only photos/emergent4_frames --output data/processed
```

#### Open the Playground
```bash
sculpture playground
```

Use `sculpture playground --viewer` to open the exported HTML viewer once you have saved a modified model.

#### With Custom Config
```bash
sculpture run --config config/custom.yaml --photos photos/
```

### Output Files

After a successful run, check:
- **Point Cloud:** `data/output/<sculpture_id>/reconstruction/point_cloud.ply` (Open3D/Meshlab)
- **Mesh:** `data/output/<sculpture_id>/meshes/mesh.ply` (Blender/CAD)
- **Wireframe (OBJ):** `data/output/<sculpture_id>/wireframes/wireframe.obj` (CAD/3D editors)
- **Wireframe (JSON):** `data/output/<sculpture_id>/wireframes/wireframe.json` (Post-processing)
- **Thumbnails:** `data/output/<sculpture_id>/thumbnails/mesh_thumb.png` and `data/output/<sculpture_id>/thumbnails/wireframe_thumb.png` (quick visual review)
- **Logs:** `data/output/pipeline.log` (Execution details)

### Structured Sculpture Repository (JSON + PKL)

As your collection grows, build a unified catalog that links each sculpture to its media and outputs.

```bash
# Build/refresh repository index
python scripts/build_sculpture_catalog.py --photos photos --out data/repository --frame-samples 30

# or via CLI
sculpture build-catalog --photos photos --out data/repository --frame-samples 30
```

Generated files:
- `data/repository/sculpture_catalog.json` — human-readable index
- `data/repository/sculpture_catalog.pkl` — fast Python load for tooling

Each entry includes:
- source video path
- frame set location + sampled frame paths
- masked preview paths
- mesh / wireframe / point-cloud paths
- mesh / wireframe thumbnail preview paths
- photography date metadata
- curator notes / critic description fields

Add or update sculpture metadata directly in the catalog:

```bash
sculpture add-sculpture nike \
   --title "Nike" \
   --year 2022 \
   --medium "Ceramic sculpture" \
   --dimensions "unknown" \
   --tag winged --tag figurative \
   --photography-date 2022-11-26 \
   --critic-description "A winged fragment of victory, classical in outline but modern in its incompleteness." \
   --catalog-dir data/repository
```

Supported metadata fields:
- `title`
- `year`
- `medium`
- `dimensions`
- repeatable `--tag`
- `photography_date`
- `notes`
- `critic_description`

---

## Blender Integration: Import Edited Meshes

The `import-edited-mesh` command enables a round-trip workflow where edited sculptures become first-class catalog assets with full provenance tracking.

### Workflow

1. **Export mesh from reconstruction**:
   ```bash
   # From the pipeline output
   cp data/output/adam/meshes/mesh.ply data/output/adam/meshes/adam_original.ply
   ```

2. **Edit in Blender**:
   - Import PLY: `File → Import → PLY` 
   - Refine geometry (sculpting, smoothing, etc.)
   - Export: `File → Export → PLY`
   - Save to: `data/output/adam/meshes_edited/adam_edited_v1.ply`

3. **Register edit in catalog**:
   ```bash
   sculpture import-edited-mesh adam \
       data/output/adam/meshes_edited/adam_edited_v1.ply \
       --editor Blender \
       --notes "Refined nose geometry; smoothed ear transitions" \
       --source-mesh data/output/adam/meshes/mesh.ply
   ```

4. **Verify in catalog**:
   ```python
   import json
   with open('data/repository/sculpture_catalog.json') as f:
       catalog = json.load(f)
   
   adam = next(r for r in catalog['sculptures'] if r['sculpture_id'] == 'adam')
   print(f"Edit history: {len(adam['edit_history'])} versions")
   for edit in adam['edit_history']:
       print(f"  v{edit['version']}: {edit['editor']} @ {edit['timestamp']}")
   ```

### Import Command Details

```bash
sculpture import-edited-mesh <sculpture_id> <mesh_path> \
    --editor Blender                              # (default: "Blender")
    --notes "Description of changes"              # Optional provenance notes
    --source-mesh <original_mesh_path>            # Optional: original mesh reference
    --catalog-dir data/repository                 # (default: data/repository)
```

**Arguments:**
- `sculpture_id` — Target sculpture (e.g., `adam`, `athena`)
- `mesh_path` — Path to edited mesh file (supports PLY, OBJ, etc.)

**Options:**
- `--editor` — Editor tool name (Blender, Meshmixer, etc.)
- `--notes` — Provenance notes (e.g., edit intent, artist name)
- `--source-mesh` — Reference to original mesh that was edited
- `--catalog-dir` — Catalog directory containing `sculpture_catalog.json`

### Catalog Schema: Edit History

Each sculpture record includes:
- **`edited_meshes`**: `list[str]` — Relative paths to Blender-edited mesh files
- **`edit_history`**: `list[dict]` — Provenance log with entries:
  ```json
  {
    "timestamp": "2026-07-15T20:19:28",
    "editor": "Blender",
    "source_mesh": "data/output/adam/meshes/mesh.ply",
    "edited_mesh": "data/output/adam/meshes_edited/adam_edited_v1.ply",
    "editor_notes": "Refined nose; smoothed transitions",
    "version": 1
  }
  ```

### Multi-Edit Workflow

Each `import-edited-mesh` call appends to the edit history, creating version lineage:

```bash
# v1: Initial refinement
sculpture import-edited-mesh adam mesh_v1.ply --notes "First pass geometry refinement"

# v2: Further detail work
sculpture import-edited-mesh adam mesh_v2.ply --notes "Added surface detail pass"

# v3: Final export for printing
sculpture import-edited-mesh adam mesh_v3_final.ply --notes "Optimized for 3D printing"
```

Resulting catalog entry:
```python
adam['edit_history']  # 3 entries (v1, v2, v3)
adam['edited_meshes'] # [path_v1, path_v2, path_v3]
```

---

## One-Click Blender Export

### Workflow

The `blender-export` command automates the entire handoff from reconstruction to Blender editing:

```bash
# Simple export (copies mesh + generates presets)
sculpture blender-export adam

# Export and automatically launch Blender with mesh pre-loaded
sculpture blender-export adam --open

# Export to custom directory
sculpture blender-export athena --export-dir ~/my_blender_projects
```

### What It Does

1. **Locates mesh** — Searches catalog or pipeline output for the reconstructed mesh
2. **Copies mesh** — Exports to a Blender-friendly directory (e.g., `data/blender_exports/adam/`)
3. **Generates presets** — Creates:
   - **Startup script** — Auto-import script for hands-free Blender launch
   - **Quick-start guide** — Plain-text instructions for manual vs. preset import
4. **Optional launch** — Opens Blender with mesh pre-loaded (macOS/Linux/Windows)

### Export Output

After running `sculpture blender-export adam`, you'll find in `data/blender_exports/adam/`:

```
adam_mesh.ply                      ← Import this in Blender
adam_blender_import.py             ← Optional auto-import preset
adam_BLENDER_QUICKSTART.txt        ← Import instructions
```

### Blender Import Options

**Option 1: Manual Import (Fastest)**
```
1. Open Blender
2. File → Import → PLY → adam_mesh.ply
3. Start editing
```

**Option 2: Auto-Import Preset (Hands-free)**
```
1. Copy adam_blender_import.py to Blender's startup scripts folder
2. Restart Blender
3. Mesh auto-loads with viewport framed
```

**Option 3: Python Console (Batch operations)**
```
1. Open Blender Scripting workspace
2. File → Open: adam_blender_import.py
3. Click [Run Script]
```

### Complete Round-Trip Workflow

```bash
# 1. Reconstruct
sculpture run --photos photos/adam_frames

# 2. Export to Blender
sculpture blender-export adam --open

# 3. [In Blender] Edit the mesh
#    - Sculpt, smooth, refine geometry
#    - Save your work
#    - File → Export → PLY → adam_edited_v1.ply

# 4. Register edited mesh back in catalog
sculpture import-edited-mesh adam \
    data/blender_exports/adam/adam_edited_v1.ply \
    --notes "Refined nose and ear transitions" \
    --source-mesh data/blender_exports/adam/adam_mesh.ply

# 5. Verify in catalog
python -c "
import json
with open('data/repository/sculpture_catalog.json') as f:
    cat = json.load(f)
    adam = next(s for s in cat['sculptures'] if s['sculpture_id'] == 'adam')
    print(f\"Edit history: {len(adam['edit_history'])} entries\")
"
```

---

## Configuration


Edit `config/default.yaml` to tune the pipeline:

```yaml
preprocessing:
  max_size: 2048                  # Resize longest edge
  bg_removal: apple_vision        # Apple Vision only (required)
  denoise_ksize: 0                # Gaussian kernel (0 = skip)

reconstruction:
  method: colmap                  # "colmap" | "open3d"
  use_depth_prior: true           # Request dense reconstruction when backend is available
  dense_backend: auto             # "auto" | "openmvs" | "colmap" | "none"
  min_dense_points_for_accept: 5000
  min_points_for_legacy_fallback: 100
  openmvs_interface_colmap_bin: InterfaceCOLMAP
  openmvs_densify_bin: DensifyPointCloud

meshing:
  method: poisson                 # "poisson" | "ball_pivot" | "alpha_shape"
  poisson_depth: 10               # Higher = finer detail
  simplify_faces: 0               # 0 disables simplification for max detail

wireframe:
  feature_angle_deg: 30.0         # Dihedral angle threshold
  min_edge_frac: 0.005            # Minimum edge length
  export_format: obj              # "obj" | "svg" | "json_graph"
```

---

## Testing

Run the full test suite (27 unit tests):

```bash
pytest tests/ -v
```

Expected output:
```
27 passed in ~2s
```

Tests cover:
- Config loading and validation
- Catalog JSON/PKL synchronization
- Standardized frame extraction + manifest generation
- Image I/O (HEIC/PNG loading, roundtrip)
- Preprocessing (resize, denoise, background removal)
- Wireframe logic (graph construction, edge filtering)

---

## Example Workflow: Sculpture Turntable Capture

**Best Practices for 3-D Reconstruction:**

1. **Capture Setup**
   - Place sculpture on a turntable
   - Use consistent, bright lighting (no harsh shadows)
   - Keep camera at a fixed height (eye-level relative to object)
   - Capture 20–40 images over full 360° rotation (ideally 2 rotations for redundancy)

2. **Save as MP4**
   - Record video at 30fps (1080p+)
   - Use H.264 codec (compatibility)
   - Ensure smooth rotation (no jerky motion)

3. **Extract Frames**
   ```bash
   python scripts/extract_turntable_frames.py
   ```
   Generates 120 evenly spaced frames by default. For faster iteration, use `--num-frames 60`; for final runs, keep 120.

4. **Run Pipeline**
   ```bash
   sculpture run --photos photos/emergent4_frames
   ```
   Processes all frames through reconstruction steps.

5. **Inspect Results**
   - Open `mesh.ply` in Blender/Meshlab to verify 3-D geometry
   - Check `wireframe.obj` for structural edges
   - Review `pipeline.log` for warnings/errors

---

## Apple Vision Subject Masking

### Overview

**New in v1.1:** Native macOS subject isolation using Apple's `VNGenerateForegroundInstanceMaskRequest` (on-device ML, no internet required).

### Building the Masking Tool

Requires full Xcode with Vision framework:

```bash
cd tools/mask_subject/
swiftc main.swift -framework Vision -framework CoreImage -framework ImageIO -o mask_subject
```

### Usage

The masking is automatic in preprocessing. `bg_removal` is standardized to `apple_vision` and treated as required.

Run preflight before processing to validate local setup and fail early with actionable guidance:

```bash
sculpture apple-vision-preflight
```

Direct invocation for spot checks:

```bash
tools/mask_subject/mask_subject <input_jpg> <output_png>
```

Outputs a PNG with subject foreground on transparent background.

### Policy

- Apple Vision masking is the only supported background-removal path.
- `grabcut` and `rembg` are no longer supported pipeline modes.
- If `tools/mask_subject/mask_subject` is missing or fails, preprocessing fails fast.

### Why Apple Vision

Apple Vision produced the most reliable masks for this sculpture dataset and is now the production baseline.

---

## Interactive Playground

### Notebook: `02_playground.ipynb`

A Jupyter notebook for loading, viewing, and editing reconstructed 3D models:

```bash
jupyter notebook notebooks/02_playground.ipynb
```

#### Features
1. **Load & Preview** — Auto-loads the latest `mesh.ply`, displays with de-flatten preview for thin objects
2. **Interactive 3D Viewer** — Rotate, zoom, pan using Plotly
3. **Parametric Controls** — Sliders for:
   - Rotation (yaw, pitch, roll)
   - Translation (X, Y, Z)
   - Scale
4. **Geometry Transforms** — Mirror, shear, vertex-pull deformation, subset transformation
5. **Camera & Lighting** — Adjust eye position, opacity, material color
6. **Side-by-Side Comparison** — Original vs. modified mesh inspection
7. **Export** — Save edited models as PLY or HTML viewer
8. **Background Removal Previews** — View original + masked frames for quality inspection

#### Example Workflow
```
1. Run pipeline: sculpture run --photos photos/emergent4_frames
2. Open notebook: jupyter notebook notebooks/02_playground.ipynb
3. Cell 2 auto-loads mesh.ply and displays initial model
4. Use sliders in Section 3–4 to rotate and inspect
5. Try deformations in Section 5
6. Export in Section 8
```

---

## API Reference

### Command-Line Interface

```bash
sculpture --help                           # Show all commands
sculpture run --help                       # Full options for run
sculpture preprocess-only --help           # Preprocessing only
sculpture playground --help                # Launch interactive playground
sculpture build-catalog --help             # Rebuild JSON/PKL repository index
sculpture add-sculpture --help             # Add or update sculpture metadata
sculpture blender-export --help            # One-click Blender mesh export + presets
sculpture apple-vision-preflight --help    # Validate Apple Vision toolchain + smoke test
sculpture import-edited-mesh --help        # Import Blender-edited meshes into catalog
```

### Python API

```python
from sculpture.pipeline import run_pipeline
from pathlib import Path

# Run full pipeline
result = run_pipeline(
    config_path=None,                    # Use default config
    photos_dir=Path('photos/my_images')
)

# Access results
pcd = result['point_cloud']              # Open3D PointCloud
mesh = result['mesh']                    # Open3D TriangleMesh
wf_graph = result['wireframe_graph']     # NetworkX Graph
```

---

## Known Limitations & Future Work

### Current
- **Single-image fallback:** When <2 images provided, sparse fallback is still coarse
- **Apple Vision dependency:** Requires macOS + compiled `tools/mask_subject` binary
- **COLMAP dense stereo on macOS:** requires CUDA; falls back unless OpenMVS is installed
- **Dense-output variability:** Some subjects still produce weak OpenMVS clouds; the pipeline now falls back to sparse COLMAP and can try legacy ORB as a last resort

### Future Enhancements
- [x] COLMAP automatic pose estimation (RANSAC-robust SfM)
- [ ] Depth estimation prior (MonoDepth2 / Depth Anything)
- [ ] NeRF-based implicit reconstruction
- [ ] Metal-native dense backend documentation/packaging for macOS arm64
- [ ] Web UI for visualization and parameter tuning
- [ ] SVG wireframe projection for technical drawings

---

## Roadmap (Prioritized)

### Now
- [x] Add one-click Blender export helper command/presets (replace manual handoff step)
- [x] Add Apple Vision preflight check command (validate binary and fail early with actionable guidance)

### Next
- [x] Add automatic COLMAP wrapper for robust pose estimation path
- [x] Add dense backend selection (`auto` / `openmvs` / `colmap` / `none`)
- [ ] Add optional GPU acceleration for reconstruction/meshing hot paths

### Later
- [ ] Add NeRF-based implicit reconstruction branch
- [ ] Add lightweight web UI for visualization/parameter tuning
- [ ] Add SVG wireframe projection/export pipeline

### CI Coverage

- GitHub Actions now runs the test suite on every push and pull request.
- CI now checks that standardized frame extraction still produces exactly `NUM_FRAMES` frames plus a manifest.
- CI now checks that `sculpture_catalog.json` and `sculpture_catalog.pkl` serialize the same catalog payload.

---

## Contributing

1. Add tests in `tests/`
2. Follow code style: `ruff check` and `ruff format`
3. Update `config/default.yaml` docs if adding parameters
4. Commit with clear messages

---

## License

MIT (included in project)

---

## Contact & Support

For issues, questions, or feature requests, check the project logs:
```bash
tail -f data/output/pipeline.log
```

Happy reconstructing! 🗿
