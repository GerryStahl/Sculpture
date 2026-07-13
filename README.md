# Sculpture AI: 3-D Wireframe Reconstruction from Multi-View Images

An end-to-end Python pipeline for reconstructing 3-D wireframe models of sculptures from video or multi-view image sequences.

## Project Goals

**Primary Objective:** Transform 2-D images of sculptures (captured from multiple angles) into 3-D point clouds, surface meshes, and wireframe representations suitable for CAD, 3-D modeling, and digital asset creation.

**Key Targets:**
- Capture 20–40 images of a sculpture rotating on a turntable (covering 360° rotations)
- Automatically extract camera parameters via checkerboard calibration
- Reconstruct dense 3-D geometry using multi-view structure-from-motion (SfM)
- Mesh point clouds and simplify for downstream use
- Extract and export structural wireframes (sharp edges, creases, silhouettes)

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
- **PyTest** — Unit testing framework (16 tests included)
- **MLflow** — Experiment tracking (optional)
- **Ruff** — Code linting and formatting

### Media Support
- **Pillow / pillow-heif** — Image I/O with HEIC/HEIF support (native iPhone photo format)
- **imageio** — Video frame extraction

---

## Pipeline Overview

```
Multi-view Images (MP4 / Image Sequence)
                    ↓
    [1. Image Preprocessing]
       • Resize to max 2048px
       • Denoise (Gaussian blur)
       • Background removal (rembg / GrabCut)
                    ↓
    [2. Feature Extraction & Matching]
       • ORB keypoint detection
       • Brute-force feature matching
       • Pose estimation (homography + RANSAC)
                    ↓
    [3. Point Cloud Reconstruction]
       • Sparse 3-D point triangulation
       • Voxel downsampling
       • Normal estimation
                    ↓
    [4. Mesh Generation]
       • Poisson surface reconstruction (depth=9)
       • Density filtering & cleanup
       • Mesh simplification (↓ target face count)
                    ↓
    [5. Wireframe Extraction]
       • Dihedral angle feature detection
       • Edge filtering by length
       • NetworkX graph construction
       • Export (OBJ / JSON)
                    ↓
         Outputs: Point Cloud, Mesh, Wireframe Graph
```

### Step-by-Step Details

#### 1. **Preprocessing**
Standardizes input images and removes unwanted background:
- **Resize:** Constrains longest edge to 2048px (configurable)
- **Denoise:** Optional Gaussian blur to reduce noise
- **Background Removal:** rembg (U2-Net segmentation) or GrabCut (traditional)

#### 2. **Feature Matching**
Establishes correspondences between images:
- **ORB Detector:** Extracts ~5000 keypoints per image
- **Brute-Force Matching:** Finds correspondences between consecutive frames
- **RANSAC Filtering:** Removes outlier matches (threshold: 5px reprojection error)

#### 3. **3-D Reconstruction**
Lifts 2-D feature matches into 3-D space:
- **Triangulation:** Uses matched feature positions across views
- **Downsampling:** Voxel grid (size 5mm) reduces point count
- **Normals:** Estimates surface normals for mesh generation

#### 4. **Meshing**
Converts point cloud to watertight surface:
- **Poisson Reconstruction:** Implicit function fitting at depth 9 (fine detail)
- **Component Filtering:** Removes isolated pieces < 1% of largest
- **Simplification:** Target ~50k triangles (configurable)
- **Cleanup:** Removes degenerate triangles and non-manifold edges

#### 5. **Wireframe Extraction**
Identifies and exports structural edges:
- **Feature Edges:** Dihedral angle > 30° (configurable)
- **Boundary Edges:** Edges with <2 adjacent triangles
- **Length Filter:** Removes edges < 0.5% of bounding box diagonal
- **Export Formats:** OBJ (line segments) or JSON (graph)

---

## Project Structure

```
sculpture/
├── config/
│   └── default.yaml              ← All pipeline parameters (YAML)
├── photos/
│   ├── emergent4g.HEIC           ← Sample single image
│   ├── emergent4.mp4             ← Sample turntable video
│   └── emergent4_frames/         ← Extracted 30 frames (generated)
├── data/
│   ├── raw/                      ← Original images
│   ├── processed/                ← Preprocessed images (RGB)
│   ├── calibration/              ← Camera intrinsics (JSON)
│   └── output/
│       ├── meshes/               ← Mesh PLY files
│       ├── wireframes/           ← Wireframe OBJ/JSON
│       ├── reconstruction/       ← Point cloud PLY
│       └── pipeline.log          ← Execution logs
├── src/sculpture/
│   ├── config.py                 ← Pydantic config loader
│   ├── io/image_io.py            ← HEIC-aware image loading
│   ├── preprocessing.py          ← Resize, denoise, bg removal
│   ├── calibration.py            ← Checkerboard camera calibration
│   ├── reconstruction.py         ← Multi-view SfM → point cloud
│   ├── meshing.py                ← Poisson/BPA/alpha-shape
│   ├── wireframe.py              ← Feature edges → NetworkX graph
│   ├── pipeline.py               ← End-to-end orchestrator
│   └── cli.py                    ← Typer CLI commands
├── scripts/
│   └── extract_turntable_frames.py  ← Extract frames from MP4
├── notebooks/
│   └── 01_explore.py             ← Interactive exploration (with matplotlib)
├── tests/
│   ├── test_config.py            ← Config loading tests
│   ├── test_image_io.py          ← Image I/O tests (16 passing)
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
# Extract 30 frames from turntable video
python scripts/extract_turntable_frames.py

# Run full pipeline on extracted frames
sculpture run --photos photos/emergent4_frames
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
- **Point Cloud:** `data/output/reconstruction/point_cloud.ply` (Open3D/Meshlab)
- **Mesh:** `data/output/meshes/mesh.ply` (Blender/CAD)
- **Wireframe (OBJ):** `data/output/wireframes/wireframe.obj` (CAD/3D editors)
- **Wireframe (JSON):** `data/output/wireframes/wireframe.json` (Post-processing)
- **Logs:** `data/output/pipeline.log` (Execution details)

---

## Configuration

Edit `config/default.yaml` to tune the pipeline:

```yaml
preprocessing:
  max_size: 2048              # Resize longest edge
  bg_removal: rembg           # "rembg" | "grabcut" | "none"
  denoise_ksize: 0            # Gaussian kernel (0 = skip)

reconstruction:
  method: open3d              # "colmap" | "opencv_sfm" | "open3d"
  use_depth_prior: false      # Future: depth estimation

meshing:
  method: poisson             # "poisson" | "ball_pivot" | "alpha_shape"
  poisson_depth: 9            # Higher = finer detail
  simplify_faces: 50000       # Target triangle count

wireframe:
  feature_angle_deg: 30.0     # Dihedral angle threshold
  min_edge_frac: 0.005        # Minimum edge length
  export_format: obj          # "obj" | "svg" | "json_graph"
```

---

## Testing

Run the full test suite (16 unit tests):

```bash
pytest tests/ -v
```

Expected output:
```
16 passed in 1.87s
```

Tests cover:
- Config loading and validation
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
   Generates 30 evenly spaced frames covering the full video duration.

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

## API Reference

### Command-Line Interface

```bash
sculpture --help                           # Show all commands
sculpture run --help                       # Full options for run
sculpture preprocess-only --help           # Preprocessing only
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
- **Single-image fallback:** When <2 images provided, creates pseudo-PCD from silhouette edges (placeholder depth)
- **COLMAP integration:** Requires manual export; planned for automatic wrapper
- **rembg dependency:** Large model (~250MB); GrabCut fallback available
- **No GPU acceleration:** Uses CPU for all operations (slow for large point clouds)

### Future Enhancements
- [ ] COLMAP automatic pose estimation (RANSAC-robust SfM)
- [ ] Depth estimation prior (MonoDepth2 / Depth Anything)
- [ ] NeRF-based implicit reconstruction
- [ ] GPU-accelerated point cloud processing (CUDA)
- [ ] Web UI for visualization and parameter tuning
- [ ] SVG wireframe projection for technical drawings

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
