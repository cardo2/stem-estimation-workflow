Stem Estimation Workflow
Mixture-of-Experts Deep Learning for Large-Scale Tree Stem Enumeration
This repository contains the complete, reproducible workflow for the paper:
Ardohain, C., Willsey, S., & Fei, S. (2026). A Novel Architecture with Mixture of Deep Learning Experts for Large Scale Stem Enumeration. *Yet to be published*

Overview
The workflow takes publicly available remote sensing data and produces tree stem density estimates (stems/ha) using a Mixture-of-Experts (MoE) convolutional neural network.
Required Inputs:

NAIP imagery (4-band: RGB + NIR)
3DEP LiDAR point clouds (.las / .laz)
DEM (Digital Elevation Model)

The processing extent is automatically determined by the intersection of the NAIP, LAS, and DEM. Height slices are calculated as height above ground using the DEM.
Default height bins: 2–2.66 m, 2.67–3.33 m, 3.34–4 m

Key Features

Height-above-ground calculation using DEM
Configurable tile size (default 100 m)
Outputs stem density predictions with model used (low / base / high)
Vector output (GeoPackage) by default + optional raster output
Fully documented and modular Python code


Quick Start
1. Clone and Install
Bashgit clone https://github.com/cardo2/stem-estimation-workflow.git
cd stem-estimation-workflow

# Recommended
conda env create -f environment.yml
conda activate stem_estimation
2. Prepare Data
Organize your data like this:
textyour_data/
├── naip/           # Folder with 4-band NAIP .tif files
├── las/            # Folder with 3DEP .las or .laz files
└── dem.tif         # Your DEM file
NAIP Requirements
The code requires 4-band (Red, Green, Blue, Near-Infrared) GeoTIFFs.
Most modern NAIP data is available as 4-band. If you only have separate RGB and CIR downloads, you must merge them into a single 4-band file first.
DEM Requirement
A DEM is required. It is used to calculate height above ground for the vegetation structure layers.

Configuration
Edit the config.yaml file:
YAMLinput:
  naip_dir: "path/to/your/naip"
  las_dir:  "path/to/your/las"
  dem_path: "path/to/your/dem.tif"

output:
  out_dir: "./output"
  format: "vector"        # Options: vector, raster, both
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

Process LAS + DEM into rasters (DSM, Intensity, height-above-ground slices)
Stack with NAIP and normalize the data
Create analysis tiles over the common extent
Run the Mixture-of-Experts model
Save results (vector predictions by default)


Output
By default, the workflow produces:

stem_predictions.gpkg — Vector file with tile polygons and predicted stem density
Intermediate rasters in the output folder (optional)

You can change the output format in config.yaml.

Models
The trained models are available via GitHub Releases.
Download the four .keras files and place them in the models/ folder, or update the paths in your config.yaml.

Citation
If you use this code or workflow, please cite the original paper:
Ardohain, C., Willsey, S., & Fei, S. (2026). A Novel Architecture with Mixture of Deep Learning Experts for Large Scale Stem Enumeration. International Journal of Applied Earth Observation and Geoinformation.