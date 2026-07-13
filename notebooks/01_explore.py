# Quick-start exploration notebook
# Run cell-by-cell after: pip install -e ".[dev]"

# %% [markdown]
# # Sculpture – End-to-End Exploration
# This notebook walks through the pipeline interactively.

# %% Imports
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sculpture.config import load_config
from sculpture.io.image_io import collect_images, load_image
from sculpture.preprocessing import preprocess_image

# %% Config
cfg = load_config()
print(cfg.model_dump())

# %% Load sample image
photos = Path("../photos")
image_paths = collect_images(photos)
print(f"Found {len(image_paths)} image(s)")

raw = load_image(image_paths[0])
print(f"Shape: {raw.shape}, dtype: {raw.dtype}")

plt.figure(figsize=(8, 6))
plt.imshow(raw)
plt.title("Raw input")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% Preprocess
processed = preprocess_image(raw, cfg.preprocessing)
print(f"Processed shape: {processed.shape}")

plt.figure(figsize=(8, 6))
plt.imshow(processed[:, :, :3])
plt.title("After preprocessing")
plt.axis("off")
plt.tight_layout()
plt.show()
