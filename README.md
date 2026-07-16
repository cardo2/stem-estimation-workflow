# Stem Estimation Workflow: Mixture-of-Experts Deep Learning for Large-Scale Tree Stem Enumeration

**From public NAIP imagery + 3DEP LiDAR point clouds to stem density (stems/ha)**

This repository provides a **validated, reproducible, and user-friendly workflow** to replicate and extend the methods from:

> Ardohain, C., Willsey, S., & Fei, S. (2026). A Novel Architecture with Mixture of Deep Learning Experts for Large Scale Stem Enumeration. *International Journal of Applied Earth Observation and Geoinformation*.

It directly addresses reviewer requests for **model availability + documented code** by supplying:
- Clear end-to-end pipeline from raw public data → predictions.
- Modular, configurable Python code (no county hard-coding).
- Placeholder structure for the trained models (to be hosted on GitHub Releases).
- Full transparency on preprocessing, tiling, normalization, and Mixture-of-Experts (MoE) inference.

---

## 1. Overview of the Method

The workflow implements a **Mixture-of-Experts (MoE)** convolutional neural network:

1. **Input stack (9 bands @ 1 m resolution)**:
   - 5 bands from 3DEP LiDAR: DSM, Intensity, + three height-interval layers (1–2 m, 2–3 m, 3–4 m).
   - 4 bands from NAIP: Red, Green, Blue, Near-Infrared.

2. **Tiling**: Regular square tiles (default **100 m** × 100 m = 1 ha; configurable to 50/200 m etc.).

3. **Discriminator CNN**: Routes each tile to the most appropriate expert based on raster features (low-density, base, or high-density stem regimes).

4. **Expert CNNs** (transfer-learned):
   - **Low expert**: Optimized for sparse/agricultural/urban areas (transfer-learned on Allen County data).
   - **Base expert**: General model trained on the core ~44 km² West Lafayette dataset.
   - **High expert**: Optimized for dense hardwood forests (transfer-learned on Hardwood Ecosystem Experiment sites).

5. **Output**: Predicted stem density (stems/ha, DBH ≥ ~10.16 cm / 4 inches) per tile.

**Key performance (Indiana statewide, validated vs Continuous Forest Inventory)**: MAE ≈ 27.5 stems/ha, RRMSE ≈ 10% when aggregated to county level with sufficient plots.

**Important limitations** (see Discussion in paper):
- Best performance in Central Hardwood Forest region (similar structure/species to Indiana).
- Data shifts in NAIP/3DEP acquisition parameters across states or years can degrade performance.
- Requires additional labeled data + fine-tuning for substantially different forest types, point densities, or sensor characteristics.
- GPU strongly recommended for large areas.

---

## 2. Quick Start

### 2.1 Installation

```bash
# Recommended: Python 3.10 – 3.12
python -m venv stem_env
source stem_env/bin/activate          # Linux/Mac
# stem_env\Scripts\activate           # Windows

pip install -r requirements.txt
```

**Note on TensorFlow & GPU**:
- The models use TensorFlow/Keras.
- For GPU acceleration (highly recommended): `pip install tensorflow[and-cuda]` or follow official TF GPU install guide for your CUDA/cuDNN versions.
- CPU inference works but will be slow on large areas (tens of minutes to hours per county depending on size and tile size).

If you encounter version conflicts (code originally developed late 2024 – early 2025), try pinning older compatible versions or use the provided `environment.yml` as a starting point and adjust.

### 2.2 Prepare Input Data

**You provide two folders** (no county structure required):

```
your_data/
├── naip/                  # Folder containing one or more NAIP GeoTIFFs (any tiling OK)
│   ├── tile1.tif
│   ├── tile2.tif
│   └── ...
└── las/                   # Folder containing 3DEP .las or .laz files
    ├── tileA.las
    ├── tileB.laz
    └── ...
```

- The **spatial union** of all NAIP and LAS files automatically defines the processing extent.
- NAIP establishes the base pixel grid and CRS for alignment.
- All outputs will be in the CRS of the NAIP data (recommended: projected CRS with meter units).

**Recommended data sources**:
- **NAIP**: USDA NRCS Geospatial Data Gateway, AWS Registry of Open Data, or USGS EarthExplorer.
  **NAIP Imagery Requirements**
  The workflow expects **4-band (Red, Green, Blue, Near-Infrared) GeoTIFFs**.
  Most modern NAIP collections are available directly as 4-band files.  
  If you only have separate RGB and CIR downloads, you must merge them into one 4-band file first (using QGIS, GDAL, or rasterio).
  Place all your final **4-band GeoTIFFs** in the `naip/` folder.
- **3DEP LiDAR (LAS)**: USGS The National Map, AWS, or state repositories. Prefer leaf-off collections when possible.
- Optional but helpful: Co-located DEMs from 3DEP for more accurate height-above-ground slices (future enhancement).

### 2.3 Configure & Run

Edit `config.yaml` (or override via CLI in future versions):

```yaml
input:
  naip_dir: "/path/to/your/naip"
  las_dir: "/path/to/your/las"

output:
  out_dir: "/path/to/output"
  format: "vector"          # "vector" (default, GeoPackage), "raster", or "both"
  tile_size_m: 100          # 50, 100 (default), 200, etc.

models:
  # Placeholders — replace with actual paths after downloading models
  discriminator: "models/discriminator.keras"
  low_expert: "models/low_expert.keras"
  base_expert: "models/base_expert.keras"
  high_expert: "models/high_expert.keras"

preprocessing:
  normalize:
    dsm: [0, 118]
    intensity: [0, 167]
    h1_2: [0, 100]
    h2_3: [0, 100]
    h3_4: [0, 100]
  # Future: ground_dem_dir if you want height-above-ground slices
```

Then run:

```bash
python -m stem_estimation.pipeline --config config.yaml
```

Or import in Python:

```python
from stem_estimation.pipeline import run_pipeline
run_pipeline(config_path="config.yaml")
```

**Expected outputs** (in `out_dir`):
- `predictions.gpkg` (or `.shp` / `.geojson`) — Vector tiles with columns: `tile_id`, `predicted_stems_per_ha`, `model_used` ("low"|"base"|"high"), geometry (square polygons).
- Optional raster `stem_density.tif` (if `format: raster` or `both`).
- `processing_log.txt` and summary statistics.

---

## 3. Detailed Pipeline Steps

### 3.1 Preprocessing (`stem_estimation/preprocessing/`)

1. **LAS → 1 m rasters** (`las_to_rasters.py`):
   - DSM (max Z)
   - Intensity (mean)
   - Height interval count layers for 1–2 m, 2–3 m, 3–4 m (points falling in these Z ranges; future versions will support DEM-relative heights).
   - Uses `laspy` + `rasterio` for binning. Aligned to a common 1 m grid.

2. **Stack + Normalize** (`stack_normalize.py`):
   - Reproject/align all layers to NAIP grid and resolution (1 m).
   - Clip values using the thresholds in `config.yaml`.
   - Scale each band to [0, 1] (standard for CNN inputs).
   - Create 9-band stacked GeoTIFF(s) covering the full extent.

3. **Tiling** (`tile_raster.py`):
   - Generate regular square grid of user-specified `tile_size_m`.
   - Only create tiles that overlap valid data (mask nodata-heavy tiles).
   - Output tile index (vector) + individual chip rasters or on-the-fly reading for inference.

### 3.2 MoE Inference (`stem_estimation/models/moe_inference.py`)

- Load discriminator + three expert models (Keras format).
- For each tile:
  - Run discriminator → softmax probabilities → argmax class.
  - Route to the corresponding expert CNN.
  - Expert outputs predicted stem count for the tile.
  - Convert to stems/ha (divide by tile area in hectares; for 100 m default this is 1:1).
- Batch processing for efficiency.
- Record which expert was used for traceability.

**Model input shape**: `(batch, tile_size_px, tile_size_px, 9)` — channels-last, float32 in [0,1].

### 3.3 Post-processing & Outputs (`stem_estimation/postprocessing/outputs.py`)

- Vector (default): GeoPackage with polygon tiles + prediction attributes. Easy to visualize in QGIS/ArcGIS or aggregate further (e.g., zonal stats to stands).
- Raster: Burn predictions into a GeoTIFF (same grid as input or aggregated).
- Summary CSV: Total stems, mean density, % tiles routed to each expert, basic stats.

---

## 4. Model Hosting & Download (Placeholders)

Trained models will be released on **GitHub Releases** (free, no LFS storage costs for reasonable model sizes ~ tens of MB each).

**Planned release structure** (when models are uploaded):
```
https://github.com/<your-org>/stem-estimation-models/releases/download/v1.0/
├── discriminator.keras
├── low_expert.keras
├── base_expert.keras
├── high_expert.keras
└── model_metadata.json   # training params, input shape, normalization thresholds, performance metrics
```

Until then, the code will raise clear errors telling you exactly where to place the files.

If you prefer another host (Zenodo, Hugging Face, institutional repo), update the `models/README.md` and download instructions accordingly.

---

## 5. Extending the Workflow

- **New regions / forest types**: Collect new labeled tiles (stem counts from field plots or high-density LiDAR), fine-tune the experts (or add new ones), retrain the discriminator on the expanded set. The MoE design is naturally extensible.
- **Different tile sizes**: Just change `tile_size_m` in config. Models were primarily developed/tested at 50/100/200 m; 100 m is a good balance.
- **Height-above-ground slices**: Future enhancement — provide a DEM folder and the preprocessor will compute relative heights.
- **Optical-only path** (for problematic 3DEP areas): Not included in v1.0. Contact authors if needed.
- **Batch / parallel processing**: The current design processes tiles sequentially or in small batches. For very large areas, wrap with Dask, GNU Parallel, or SLURM.

---

## 6. Hardware, Performance & Troubleshooting

- **Recommended**: NVIDIA GPU with ≥ 8 GB VRAM (e.g., RTX 3070+ as used in original work). Whole-county runs: 30–90+ minutes depending on area and tile size.
- **CPU fallback**: Works but expect 5–10× slowdown.
- **Memory**: Large NAIP/LAS collections benefit from 32+ GB RAM. Tile-by-tile processing keeps peak memory reasonable.
- **Common issues**:
  - CRS mismatches → ensure NAIP and LAS are in a common projected meter-based CRS.
  - Version conflicts with TensorFlow/rasterio → create a fresh conda env from `environment.yml` and `pip install -r requirements.txt --upgrade-strategy only-if-needed`.
  - Out-of-memory on very large tiles (200 m+) → reduce `tile_size_m` or batch size in inference.
  - Nodata/edge tiles → automatically filtered or flagged in output.

See `processing_log.txt` for detailed diagnostics.

---

## 7. Citation & Acknowledgments

If you use this workflow or models in your work, please cite the original paper:

Ardohain, C., Willsey, S., Fei, S. (2026). A Novel Architecture with Mixture of Deep Learning Experts for Large Scale Stem Enumeration. *Int. J. Appl. Earth Obs. Geoinf.*

This code refactors and generalizes the original research implementation for broader use while preserving the validated logic.

**Data credits**: NAIP (USDA), 3DEP (USGS), training labels derived from Geiger-mode LiDAR (original study) and field/CFI data (Indiana DNR).

---

## 8. Roadmap / Future Versions

- v1.1: DEM-relative height slices, better ground filtering.
- v1.2: Optical-only expert path for bad 3DEP regions.
- v2.0: PyTorch version option, Docker container, web demo / Hugging Face Space.
- Community contributions welcome (especially new regional experts).

---

**Questions or issues?** Open a GitHub issue or contact the authors.

*This workflow was prepared to maximize reproducibility and impact of the published research.*

---

**Current Status (v1.0, June 2026)**:
- Preprocessing (LAS → DSM/Intensity/Height slices) implemented and tested in structure.
- Stacking + normalization to 9-band [0,1] input implemented.
- Configurable tiling (default 100 m) with valid-data filtering implemented.
- MoE inference module ready (loads models + routing logic; awaits actual model files).
- Vector output (GeoPackage) + optional raster implemented.
- Full `run_pipeline()` now wires the major components together.

The workflow is **usable end-to-end once you provide the four .keras model files** and point `config.yaml` at real NAIP + 3DEP data.

**Version**: 1.0 (June 2026)  
**Compatible with paper**: Yes (core MoE architecture, 100 m default tile, 9-band fused input, transfer-learned experts).