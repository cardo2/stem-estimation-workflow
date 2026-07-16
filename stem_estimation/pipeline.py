"""
High-level pipeline orchestrator for stem density estimation.

Usage:
    from stem_estimation.pipeline import run_pipeline
    run_pipeline(config_path="config.yaml")

Or via CLI (future):
    python -m stem_estimation.pipeline --config config.yaml
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Internal imports
from .preprocessing.las_to_rasters import process_las_folder
from .preprocessing.stack_normalize import align_and_stack
from .preprocessing.tile_raster import generate_tile_grid
from .models.moe_inference import run_moe_on_tiles
from .postprocessing.outputs import write_vector_output, write_raster_output, write_summary

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Basic validation / defaults
    config.setdefault("output", {}).setdefault("format", "vector")
    config.setdefault("output", {}).setdefault("tile_size_m", 100)
    config.setdefault("inference", {}).setdefault("batch_size", 16)
    config.setdefault("inference", {}).setdefault("use_gpu", True)

    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging based on config."""
    level = getattr(logging, config.get("logging", {}).get("level", "INFO").upper())
    log_file = config.get("logging", {}).get("log_file", "processing_log.txt")
    out_dir = Path(config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / log_file, mode="w"),
        ],
    )
    logger.info("Logging initialized.")


def run_pipeline(config_path: str | Path = "config.yaml") -> None:
    """
    Main entry point for the stem estimation MoE workflow.
    """
    config = load_config(config_path)
    setup_logging(config)

    logger.info("=" * 70)
    logger.info("STEM ESTIMATION MoE PIPELINE v1.0")
    logger.info(f"Config file: {config_path}")
    logger.info("=" * 70)

    out_dir = Path(config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    naip_dir = Path(config["input"]["naip_dir"])
    las_dir = Path(config["input"]["las_dir"])
    tile_size_m = config["output"]["tile_size_m"]
    output_format = config["output"]["format"]
    basename = config["output"].get("basename", "stem_predictions")

    # 1. Preprocess LAS → rasters (DSM, intensity, height slices)
    lidar_out = out_dir / "lidar_rasters"
    lidar_rasters = process_las_folder(
        las_dir=las_dir,
        output_dir=lidar_out,
        resolution=1.0,
        dem_path=config["input"].get("dem_path")
    )

    # 2. Stack + normalize with NAIP
    stack_path = out_dir / f"{basename}_stack_9band.tif"
    stack_path = align_and_stack(
        naip_dir=naip_dir,
        lidar_rasters=lidar_rasters,
        output_path=stack_path,
        normalize_config=config["preprocessing"]["normalize"]
    )

    # 3. Generate analysis tiles
    tile_index_path = out_dir / f"{basename}_tiles.gpkg"
    tile_gdf = generate_tile_grid(
        stack_path=stack_path,
        tile_size_m=tile_size_m,
        min_valid_fraction=config["preprocessing"].get("min_valid_fraction", 0.5),
        output_vector=tile_index_path
    )

    if len(tile_gdf) == 0:
        logger.error("No valid tiles generated. Check input data extent and quality.")
        return

    # 4. MoE Inference
    model_paths = {
        "discriminator": config["models"]["discriminator"],
        "low_expert": config["models"]["low_expert"],
        "base_expert": config["models"]["base_expert"],
        "high_expert": config["models"]["high_expert"],
    }
    predictions = run_moe_on_tiles(
        stack_path=stack_path,
        tile_gdf=tile_gdf,
        model_paths=model_paths,
        batch_size=config["inference"]["batch_size"],
        tile_size_px=int(tile_size_m)  # assumes 1 m/px
    )

    # 5. Write outputs
    vector_out = out_dir / f"{basename}.gpkg"
    write_vector_output(tile_gdf, predictions, vector_out, basename)

    if output_format in ["raster", "both"]:
        raster_out = out_dir / f"{basename}_density.tif"
        write_raster_output(tile_gdf, predictions, stack_path, raster_out)

    summary_csv = out_dir / f"{basename}_summary.csv"
    write_summary(predictions, summary_csv)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Results saved to: {out_dir}")
    logger.info(f"Total tiles processed: {len(predictions)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stem Estimation MoE Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration YAML file")
    args = parser.parse_args()

    run_pipeline(args.config)
