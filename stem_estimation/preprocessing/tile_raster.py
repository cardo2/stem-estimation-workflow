"""
Generate square analysis tiles (chips) from a large stacked raster.

Default: 100 m tiles at 1 m/px → 100×100 pixel chips.
Only tiles with sufficient valid data are kept.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd

logger = logging.getLogger(__name__)


def generate_tile_grid(stack_path: Path,
                       tile_size_m: float = 100.0,
                       min_valid_fraction: float = 0.5,
                       output_vector: Optional[Path] = None) -> gpd.GeoDataFrame:
    """
    Create a regular grid of square tiles covering the raster extent.

    Returns a GeoDataFrame with tile polygons and attributes for filtering.
    """
    with rasterio.open(stack_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.shape
        res_x = transform.a
        res_y = -transform.e  # usually positive

        pixel_size_m = res_x  # assume square pixels
        tile_size_px = int(round(tile_size_m / pixel_size_m))

        if tile_size_px < 1:
            raise ValueError(f"tile_size_m {tile_size_m} too small for resolution {pixel_size_m}")

        logger.info(f"Creating {tile_size_m} m tiles ({tile_size_px}×{tile_size_px} px)")

        tiles = []
        tile_id = 0

        for row_off in range(0, height, tile_size_px):
            for col_off in range(0, width, tile_size_px):
                win_height = min(tile_size_px, height - row_off)
                win_width = min(tile_size_px, width - col_off)

                if win_height < tile_size_px // 2 or win_width < tile_size_px // 2:
                    continue  # skip tiny edge tiles

                window = Window(col_off, row_off, win_width, win_height)
                # Read a quick nodata mask (use first band)
                data = src.read(1, window=window)
                valid = np.sum(~np.isnan(data) & (data != src.nodata)) / data.size

                if valid < min_valid_fraction:
                    continue

                # Compute geographic bounds of this tile
                x0 = transform.c + col_off * transform.a
                y1 = transform.f + row_off * transform.e
                x1 = x0 + win_width * transform.a
                y0 = y1 + win_height * transform.e

                geom = box(x0, y0, x1, y1)

                tiles.append({
                    "tile_id": f"tile_{tile_id:06d}",
                    "row_off": row_off,
                    "col_off": col_off,
                    "width_px": win_width,
                    "height_px": win_height,
                    "valid_fraction": round(valid, 3),
                    "geometry": geom
                })
                tile_id += 1

    gdf = gpd.GeoDataFrame(tiles, crs=crs)

    if output_vector:
        gdf.to_file(output_vector, driver="GPKG")
        logger.info(f"Tile index saved to {output_vector} ({len(gdf)} tiles)")

    return gdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("tile_raster module ready.")