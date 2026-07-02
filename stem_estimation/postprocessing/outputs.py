"""
Write model predictions to vector (default) or raster formats.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

logger = logging.getLogger(__name__)


def write_vector_output(tile_gdf: gpd.GeoDataFrame,
                        predictions: List[Dict],
                        output_path: Path,
                        basename: str = "stem_predictions") -> Path:
    """Join predictions to tile geometries and save as GeoPackage (recommended) or Shapefile."""
    pred_df = pd.DataFrame(predictions)
    merged = tile_gdf.merge(pred_df, on="tile_id", how="left")

    if output_path.suffix.lower() in [".gpkg", ".geojson"]:
        driver = "GPKG" if output_path.suffix == ".gpkg" else "GeoJSON"
    else:
        output_path = output_path.with_suffix(".gpkg")
        driver = "GPKG"

    merged.to_file(output_path, driver=driver)
    logger.info(f"Vector output written: {output_path} ({len(merged)} features)")
    return output_path


def write_raster_output(tile_gdf: gpd.GeoDataFrame,
                        predictions: List[Dict],
                        reference_raster: Path,
                        output_path: Path) -> Path:
    """Burn predictions into a GeoTIFF aligned with the input stack."""
    with rasterio.open(reference_raster) as src:
        out_profile = src.profile.copy()
        out_profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")

    shapes = []
    for pred, geom in zip(predictions, tile_gdf.geometry):
        if pd.notna(pred.get("predicted_stems_per_ha")):
            shapes.append((geom, pred["predicted_stems_per_ha"]))

    if not shapes:
        logger.warning("No valid predictions to rasterize.")
        return None

    with rasterio.open(output_path, "w", **out_profile) as dst:
        burned = rasterize(
            shapes=shapes,
            out_shape=(dst.height, dst.width),
            transform=dst.transform,
            fill=-9999.0,
            dtype="float32",
        )
        dst.write(burned, 1)

    logger.info(f"Raster output written: {output_path}")
    return output_path


def write_summary(predictions: List[Dict], output_csv: Path) -> None:
    """Write a simple CSV summary of results."""
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    logger.info(f"Summary CSV written: {output_csv}")