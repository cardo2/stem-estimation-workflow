"""
Mosaic-aware chipping of stacked 9-band GeoTIFFs.

Builds a regular grid over the union of all stacked footprints.
Each chip is mosaicked on-the-fly from intersecting sources so seams
do not produce mixed half-tiles or artificial gaps inside the grid.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, shape
from shapely.strtree import STRtree
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _valid_fraction(chip: np.ndarray, nodata: int = 255) -> float:
    if chip.size == 0:
        return 0.0
    bad = np.any(chip == nodata, axis=0)
    return float(1.0 - bad.mean())


def _load_index(geojson_path: Path):
    with open(geojson_path, encoding="utf-8") as fh:
        fc = json.load(fh)
    geoms = []
    props = []
    for feat in fc["features"]:
        geoms.append(shape(feat["geometry"]))
        props.append(feat["properties"])
    tree = STRtree(geoms)
    return geoms, props, tree


def _mosaic_window(
    paths: List[Path],
    west: float,
    south: float,
    east: float,
    north: float,
    tile_size: int,
    nodata: int = 255,
) -> Optional[Tuple[np.ndarray, object, object]]:
    """
    Mosaic intersecting sources into a tile_size × tile_size array
    covering [west, south, east, north]. Returns (data, transform, crs) or None.
    """
    srcs = []
    try:
        for p in paths:
            srcs.append(rasterio.open(p))

        # Target transform for this chip
        dst_transform = from_bounds(west, south, east, north, tile_size, tile_size)
        crs = srcs[0].crs
        count = srcs[0].count

        # Merge to a slightly larger mosaic then reproject into exact chip grid
        mosaic, out_trans = merge(
            srcs,
            bounds=(west, south, east, north),
            nodata=nodata,
            dtype="uint8",
        )

        dest = np.full((count, tile_size, tile_size), nodata, dtype=np.uint8)
        reproject(
            source=mosaic,
            destination=dest,
            src_transform=out_trans,
            src_crs=crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
            src_nodata=nodata,
            dst_nodata=nodata,
        )
        return dest, dst_transform, crs
    except Exception as e:
        logger.debug(f"Mosaic window failed: {e}")
        return None
    finally:
        for s in srcs:
            s.close()


def tile_mosaic(
    index_geojson: str | Path,
    output_dir: str | Path,
    tile_size: int = 100,
    min_valid_fraction: float = 0.5,
    nodata: int = 255,
    resolution: Optional[float] = None,
) -> List[Path]:
    """
    Chip a regular grid over the union of all indexed stacked tiles.
    resolution: pixel size in CRS units (e.g. 1.0 for 1 m). If None,
    taken from the first source file.
    """
    index_geojson = Path(index_geojson)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geoms, props, tree = _load_index(index_geojson)
    if not geoms:
        raise ValueError("Empty stack index")

    # Pixel size from first raster if not provided
    if resolution is None:
        with rasterio.open(props[0]["path"]) as src:
            resolution = abs(src.transform.a)

    # Union bounds
    minx = min(g.bounds[0] for g in geoms)
    miny = min(g.bounds[1] for g in geoms)
    maxx = max(g.bounds[2] for g in geoms)
    maxy = max(g.bounds[3] for g in geoms)

    chip_m = tile_size * resolution  # chip size in map units
    n_cols = int(np.floor((maxx - minx) / chip_m))
    n_rows = int(np.floor((maxy - miny) / chip_m))

    logger.info(
        f"Grid: {n_cols}×{n_rows} chips | tile_size={tile_size} px | "
        f"res={resolution} | extent=({minx:.1f},{miny:.1f})–({maxx:.1f},{maxy:.1f})"
    )

    written: List[Path] = []

    for i in tqdm(range(n_rows), desc="Tiling rows"):
        for j in range(n_cols):
            west = minx + j * chip_m
            east = west + chip_m
            north = maxy - i * chip_m
            south = north - chip_m
            cell = box(west, south, east, north)

            # Intersecting stacked tiles
            hits = tree.query(cell)
            # STRtree.query returns indices (shapely 2) or geoms (shapely 1)
            paths = []
            for h in hits:
                if isinstance(h, (int, np.integer)):
                    geom = geoms[int(h)]
                    prop = props[int(h)]
                else:
                    geom = h
                    idx = geoms.index(h)
                    prop = props[idx]
                if geom.intersects(cell):
                    paths.append(Path(prop["path"]))

            if not paths:
                continue

            result = _mosaic_window(paths, west, south, east, north, tile_size, nodata)
            if result is None:
                continue
            data, transform, crs = result

            if _valid_fraction(data, nodata) < min_valid_fraction:
                continue

            out_name = f"chip_r{i:04d}_c{j:04d}.tif"
            out_path = output_dir / out_name
            profile = {
                "driver": "GTiff",
                "height": tile_size,
                "width": tile_size,
                "count": data.shape[0],
                "dtype": "uint8",
                "crs": crs,
                "transform": transform,
                "nodata": nodata,
                "compress": "deflate",
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)
                dst.descriptions = (
                    "CHM", "Intensity", "h1_2", "h2_3", "h3_4",
                    "NAIP_R", "NAIP_G", "NAIP_B", "NAIP_NIR",
                )
            written.append(out_path)

    logger.info(f"Wrote {len(written)} mosaic chips → {output_dir}")
    return written