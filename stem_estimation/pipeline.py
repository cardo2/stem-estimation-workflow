"""
High-level pipeline orchestrator for stem density estimation.

Usage:
    from stem_estimation.pipeline import run_pipeline
    run_pipeline(config_path="config.yaml")

CLI:
    python -m stem_estimation.pipeline --config config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from .preprocessing.las_to_rasters import process_las_folder
from .preprocessing.stack_normalize import align_and_stack, DEFAULT_THRESHOLDS
from .preprocessing.stack_index import build_stack_index
from .preprocessing.tile_raster import tile_mosaic
from .models.moe_inference import run_moe_on_chip_folder
from .postprocessing.outputs import write_summary

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("output", {}).setdefault("format", "vector")
    config.setdefault("output", {}).setdefault("tile_size_m", 100)
    config.setdefault("output", {}).setdefault("basename", "stem_predictions")
    config.setdefault("inference", {}).setdefault("batch_size", 16)
    config.setdefault("inference", {}).setdefault("use_gpu", True)
    config.setdefault("preprocessing", {}).setdefault("fill_intensity_gaps", True)
    return config


def setup_logging(config: Dict[str, Any]) -> None:
    level = getattr(logging, config.get("logging", {}).get("level", "INFO").upper())
    log_file = config.get("logging", {}).get("log_file", "processing_log.txt")
    out_dir = Path(config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    logger.info("Logging initialized.")


def _thresholds_from_config(config: Dict[str, Any]) -> Dict[str, tuple]:
    norm = config.get("preprocessing", {}).get("normalize", {})
    out = dict(DEFAULT_THRESHOLDS)
    for key in ("chm", "intensity", "h1_2", "h2_3", "h3_4"):
        if key in norm and isinstance(norm[key], (list, tuple)) and len(norm[key]) == 2:
            out[key] = (float(norm[key][0]), float(norm[key][1]))
    return out


def _write_chip_vector(results: List[Dict], output_path: Path) -> Path:
    if not results:
        raise ValueError("No predictions to write")
    crs = results[0].get("crs")
    gdf = gpd.GeoDataFrame(
        {
            "chip_id": [r["chip_id"] for r in results],
            "stems_ha": [r["predicted_stems_per_ha"] for r in results],
            "model_used": [r["model_used"] for r in results],
        },
        geometry=[
            box(r["west"], r["south"], r["east"], r["north"]) for r in results
        ],
        crs=crs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GPKG")
    logger.info(f"Vector output: {output_path}")
    return output_path


def run_pipeline(config_path: str | Path = "config.yaml") -> None:
    config = load_config(config_path)
    setup_logging(config)

    logger.info("=" * 70)
    logger.info("STEM ESTIMATION MoE PIPELINE")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 70)

    out_dir = Path(config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    naip_dir = Path(config["input"]["naip_dir"])
    las_dir = Path(config["input"]["las_dir"])
    dem_dir = Path(config["input"]["dem_dir"])
    basename = config["output"]["basename"]
    tile_size = int(config["output"].get("tile_size_m", 100))
    min_valid = float(
        config.get("preprocessing", {})
        .get("normalize", {})
        .get("min_valid_fraction", 0.5)
    )
    thresholds = _thresholds_from_config(config)

    # 1. LAS + DEM → per-tile LiDAR layers
    lidar_dir = out_dir / "lidar_layers"
    logger.info("Step 1/5: LAS → rasters (CHM, intensity, height %)")
    process_las_folder(
        las_dir=las_dir,
        dem_dir=dem_dir,
        output_dir=lidar_dir,
        fill_intensity_gaps=config["preprocessing"].get("fill_intensity_gaps", True),
    )

    # 2. Stack + normalize with NAIP
    stacked_dir = out_dir / "stacked"
    logger.info("Step 2/5: Stack LiDAR + NAIP and normalize to 0–254")
    stacked_paths = align_and_stack(
        lidar_dir=lidar_dir,
        naip_dir=naip_dir,
        output_dir=stacked_dir,
        thresholds=thresholds,
    )
    if not stacked_paths:
        logger.error("No stacked tiles produced. Check NAIP/LAS overlap.")
        return

    # 3. Spatial index of stacked tiles
    index_path = out_dir / "stack_index.geojson"
    logger.info("Step 3/5: Build stacked-tile spatial index")
    build_stack_index(stacked_dir, index_path)

    # 4. Mosaic-aware chipping
    chips_dir = out_dir / "chips"
    logger.info("Step 4/5: Mosaic tile grid → chips")
    chips = tile_mosaic(
        index_geojson=index_path,
        output_dir=chips_dir,
        tile_size=tile_size,
        min_valid_fraction=min_valid,
    )
    if not chips:
        logger.error("No chips produced. Check extents and min_valid_fraction.")
        return

    # 5. MoE inference on chips
    model_paths = {
        "discriminator": config["models"]["discriminator"],
        "low_expert": config["models"]["low_expert"],
        "base_expert": config["models"]["base_expert"],
        "high_expert": config["models"]["high_expert"],
    }
    logger.info("Step 5/5: MoE inference on chips")
    predictions = run_moe_on_chip_folder(
        chip_dir=chips_dir,
        model_paths=model_paths,
        batch_size=int(config["inference"]["batch_size"]),
    )

    # Outputs
    vector_out = out_dir / f"{basename}.gpkg"
    _write_chip_vector(predictions, vector_out)

    summary_csv = out_dir / f"{basename}_summary.csv"
    write_summary(predictions, summary_csv)

    stems = np.array([p["predicted_stems_per_ha"] for p in predictions], dtype=float)
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Results: {out_dir}")
    logger.info(f"Chips: {len(predictions)}")
    if len(stems):
        logger.info(
            f"stems/ha  min={stems.min():.2f}  median={np.median(stems):.2f}  "
            f"mean={stems.mean():.2f}  max={stems.max():.2f}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stem Estimation MoE Pipeline")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()
    run_pipeline(args.config)