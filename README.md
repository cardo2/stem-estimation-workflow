# Stem Estimation Workflow

**Mixture-of-Experts Deep Learning for Large-Scale Tree Stem Enumeration**

This repository contains the complete, reproducible workflow for the paper:

**Ardohain, C., Willsey, S., & Fei, S. (2026).** *A Novel Architecture with Mixture of Deep Learning Experts for Large Scale Stem Enumeration. Pending Publication*

---

## Overview

The workflow converts publicly available remote sensing data into tree stem density estimates (stems/ha) using a Mixture-of-Experts (MoE) convolutional neural network.

**Required Inputs (all three):**
- `naip/` — Folder containing 4-band NAIP imagery (RGB + NIR)
- `las/` — Folder containing 3DEP LiDAR files (.las or .laz)
- `dem/` — Folder containing DEM files (.tif or .img)

The processing extent is the **intersection** of the NAIP, LAS, and DEM. Height slices are calculated as **height above ground** using the DEM.

Height-above-ground percentage layers: 1–2 m, 2–3 m, and 3–4 m AGL
(percentage of non-ground returns per 1 m cell; DEM is the 1 m pixel-grid anchor).

---

## Key Features

- Height-above-ground calculation using DEM
- Supports multiple DEM tiles (they are merged automatically)
- Configurable tile size (default 100 m)
- Outputs stem density predictions + which expert model was used
- Vector output (GeoPackage) by default + optional raster

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/cardo2/stem-estimation-workflow.git
cd stem-estimation-workflow

# Recommended
conda env create -f environment.yml
conda activate stem_estimation
2. Prepare Your Data
Organize your data like this:
textyour_data/
├── naip/           # Folder with 4-band NAIP .tif files
├── las/            # Folder with 3DEP .las or .laz files
└── dem/            # Folder with DEM .tif or .img files
NAIP Requirements
The workflow requires 4-band (Red, Green, Blue, Near-Infrared) GeoTIFFs.
Most modern NAIP data is available directly as 4-band. If you only have separate RGB and CIR downloads, merge them into a single 4-band file first.
DEM Requirement
A dem/ folder is required. It can contain one or multiple DEM tiles — the code will merge them automatically.

Configuration
Edit config.yaml:
YAMLinput:
  naip_dir: "path/to/your/naip"
  las_dir:  "path/to/your/las"
  dem_dir:  "path/to/your/dem"        # Folder containing DEM files

output:
  out_dir: "./output"
  format: "vector"          # "vector", "raster", or "both"
  tile_size_m: 100
  basename: "stem_predictions"

models:
  discriminator: "models/discriminator.keras"
  low_expert:    "models/low_expert.keras"
  base_expert:   "models/base_expert.keras"
  high_expert:   "models/high_expert.keras"

Running the Pipeline
Bashpython -m stem_estimation.pipeline --config config.yaml
The pipeline will:

Merge DEMs (if multiple) and calculate height above ground
Process LAS into rasters using height-above-ground slices
Stack with NAIP and normalize
Create tiles over the common extent
Run Mixture-of-Experts inference
Save predictions


Output
Default output:

stem_predictions.gpkg — Vector file with predicted stem density per tile
Optional raster output (change in config.yaml)


Models
Download the trained models from GitHub Releases and place them in the models/ folder.

Citation
Please cite the original paper if you use this workflow:
Ardohain, C., Willsey, S., & Fei, S. (2026). A Novel Architecture with Mixture of Deep Learning Experts for Large Scale Stem Enumeration. *Pending Publication*