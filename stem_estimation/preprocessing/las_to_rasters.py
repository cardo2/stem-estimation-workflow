"""
LAS/LAZ → raster layers for the stem-estimation workflow.

Design rules (debugged July 2026):
- DEM is the master pixel grid and must be 1 m resolution.
- Process one LAS file at a time (memory efficient).
- Full CRS handling (LAS often State-Plane feet, DEM usually UTM meters).
- Output is clipped to the LAS × DEM intersection and snapped to the DEM grid.
- Percentage layers use non-ground points (classification != 2) as denominator,
  matching the original research code. Falls back to h > 0 when classification
  is missing.
- Layers written per LAS tile:
    *_chm.tif        – canopy height model (height above ground, feet)
    *_intensity.tif  – mean intensity (thin gaps filled)
    *_h1_2.tif       – % of non-ground returns in 1–2 m AGL
    *_h2_3.tif       – % of non-ground returns in 2–3 m AGL
    *_h3_4.tif       – % of non-ground returns in 3–4 m AGL
Height-slice spatial coverage can differ slightly when a different
DEM product is used than the one that generated training data.
The percentage formula matches the original research code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import laspy
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from pyproj import Transformer
from scipy import ndimage
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Height bins in meters (above ground) – matches training data
HEIGHT_BINS_M = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
BIN_NAMES = ["h1_2", "h2_3", "h3_4"]

FT_PER_M = 3.28084


def _find_intersecting_dems(
    las_bounds_native: Tuple[float, float, float, float],
    las_crs,
    dem_files: List[Path],
) -> List[dict]:
    """Return list of DEM info dicts that intersect the LAS bounds."""
    from shapely.geometry import box

    intersecting = []
    for dem_path in dem_files:
        try:
            with rasterio.open(dem_path) as src:
                reproj = transform_bounds(las_crs, src.crs, *las_bounds_native)
                if box(*reproj).intersects(
                    box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                ):
                    intersecting.append(
                        {
                            "path": dem_path,
                            "crs": src.crs,
                            "transform": src.transform,
                            "bounds": src.bounds,
                            "width": src.width,
                            "height": src.height,
                        }
                    )
        except Exception as e:
            logger.debug(f"Could not check {dem_path.name}: {e}")
    return intersecting


def _clip_window_to_las(
    dem_info: dict,
    las_bounds_native: Tuple[float, float, float, float],
    las_crs,
) -> Tuple[Window, Affine, int, int]:
    """
    Compute a Window on the DEM that covers the LAS extent,
    snapped to the DEM pixel grid.
    """
    las_bounds_m = transform_bounds(las_crs, dem_info["crs"], *las_bounds_native)

    inter_minx = max(las_bounds_m[0], dem_info["bounds"].left)
    inter_miny = max(las_bounds_m[1], dem_info["bounds"].bottom)
    inter_maxx = min(las_bounds_m[2], dem_info["bounds"].right)
    inter_maxy = min(las_bounds_m[3], dem_info["bounds"].top)

    inv = ~dem_info["transform"]
    col_start, row_start = inv * (inter_minx, inter_maxy)
    col_stop, row_stop = inv * (inter_maxx, inter_miny)

    col_start = max(0, int(np.floor(col_start)))
    row_start = max(0, int(np.floor(row_start)))
    col_stop = min(dem_info["width"], int(np.ceil(col_stop)))
    row_stop = min(dem_info["height"], int(np.ceil(row_stop)))

    width = max(1, col_stop - col_start)
    height = max(1, row_stop - row_start)
    window = Window(col_start, row_start, width, height)
    dst_transform = dem_info["transform"] * Affine.translation(col_start, row_start)

    return window, dst_transform, width, height


def process_single_las(
    las_path: Path,
    dem_files: List[Path],
    output_dir: Path,
    fill_intensity_gaps: bool = True,
) -> Dict[str, Path]:
    """
    Process one LAS/LAZ file against intersecting DEM(s).
    Returns dict of output raster paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = las_path.stem

    # ----- LAS header -----
    with laspy.open(las_path) as las:
        hdr = las.header
        minx, miny = float(hdr.mins[0]), float(hdr.mins[1])
        maxx, maxy = float(hdr.maxs[0]), float(hdr.maxs[1])
        try:
            las_crs = las.header.parse_crs()
        except Exception:
            logger.error(f"{stem}: could not parse CRS – skipping")
            return {}

    las_bounds = (minx, miny, maxx, maxy)

    # ----- Find intersecting DEM -----
    intersecting = _find_intersecting_dems(las_bounds, las_crs, dem_files)
    if not intersecting:
        logger.warning(f"{stem}: no intersecting DEM – skipping")
        return {}

    dem = intersecting[0]
    logger.info(f"{stem}: using DEM {dem['path'].name}")

    # ----- Clip window on DEM grid -----
    window, dst_transform, width, height = _clip_window_to_las(dem, las_bounds, las_crs)
    logger.info(f"{stem}: grid {width} × {height}")

    # ----- Load DEM window (meters) -----
    with rasterio.open(dem["path"]) as src:
        ground_m = src.read(1, window=window).astype(np.float32)

    ground_ft = ground_m * FT_PER_M

    # ----- Accumulators -----
    chm = np.zeros((height, width), dtype=np.float32)
    intensity_sum = np.zeros((height, width), dtype=np.float64)
    intensity_cnt = np.zeros((height, width), dtype=np.uint32)
    total_cnt = np.zeros((height, width), dtype=np.uint32)   # denominator for %
    bin_cnt = {name: np.zeros((height, width), dtype=np.uint32) for name in BIN_NAMES}

    transformer = Transformer.from_crs(las_crs, dem["crs"], always_xy=True)
    a, e, c, f = dst_transform.a, dst_transform.e, dst_transform.c, dst_transform.f

    # ----- Rasterize points -----
    with laspy.open(las_path) as las:
        # Check whether classification is usable
        has_classification = False
        try:
            sample = next(las.chunk_iterator(10_000))
            cls = np.asarray(getattr(sample, "classification", []))
            if cls.size > 0 and not np.all(cls == 0):
                has_classification = True
        except Exception:
            pass

        if has_classification:
            logger.info(f"{stem}: using classification != 2 for percentage denominator")
        else:
            logger.info(f"{stem}: no usable classification – falling back to h > 0")

        for points in las.chunk_iterator(500_000):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z_ft = np.asarray(points.z)
            inten = np.asarray(getattr(points, "intensity", np.zeros_like(z_ft)))

            # Optional classification filter (original behaviour)
            if has_classification:
                cls = np.asarray(points.classification)
                non_ground = cls != 2
                if not np.any(non_ground):
                    continue
                x, y, z_ft, inten = x[non_ground], y[non_ground], z_ft[non_ground], inten[non_ground]

            x2, y2 = transformer.transform(x, y)
            col = np.floor((x2 - c) / a).astype(np.int32)
            row = np.floor((y2 - f) / e).astype(np.int32)

            valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            if not np.any(valid):
                continue

            col, row = col[valid], row[valid]
            z_ft, inten = z_ft[valid], inten[valid]
            z_m = z_ft / FT_PER_M

            g_m = ground_m[row, col]
            valid_g = np.isfinite(g_m)
            if not np.any(valid_g):
                continue

            col, row = col[valid_g], row[valid_g]
            z_ft, z_m, inten = z_ft[valid_g], z_m[valid_g], inten[valid_g]
            g_m = g_m[valid_g]
            g_ft = g_m * FT_PER_M

            h_ft = z_ft - g_ft
            h_m = z_m - g_m

            # CHM (feet) – max height above ground
            pos = h_ft > 0
            if np.any(pos):
                np.maximum.at(chm, (row[pos], col[pos]), h_ft[pos])

            # Intensity (all points that reached here)
            np.add.at(intensity_sum, (row, col), inten)
            np.add.at(intensity_cnt, (row, col), 1)

            # Percentage denominator
            if has_classification:
                # Already filtered to non-ground; count all remaining points
                np.add.at(total_cnt, (row, col), 1)
            else:
                # Fallback: only points with height > 0
                above = h_m > 0
                if np.any(above):
                    np.add.at(total_cnt, (row[above], col[above]), 1)

            # Bin counts (1–2, 2–3, 3–4 m)
            for (lo, hi), name in zip(HEIGHT_BINS_M, BIN_NAMES):
                mask = (h_m >= lo) & (h_m < hi)
                if np.any(mask):
                    np.add.at(bin_cnt[name], (row[mask], col[mask]), 1)

    # ----- Post-process intensity (optional thin-gap fill) -----
    intensity = np.full((height, width), np.nan, dtype=np.float32)
    mask = intensity_cnt > 0
    intensity[mask] = (intensity_sum[mask] / intensity_cnt[mask]).astype(np.float32)

    if fill_intensity_gaps:
        filled = intensity.copy()
        nan_mask = ~np.isfinite(filled)
        for _ in range(5):
            kernel = np.ones((3, 3), dtype=np.float32)
            neighbor_sum = ndimage.convolve(
                np.nan_to_num(filled, nan=0.0), kernel, mode="constant", cval=0.0
            )
            neighbor_cnt = ndimage.convolve(
                (~nan_mask).astype(np.float32), kernel, mode="constant", cval=0.0
            )
            fill_vals = np.divide(
                neighbor_sum, neighbor_cnt,
                out=np.zeros_like(neighbor_sum), where=neighbor_cnt > 0
            )
            still_nan = ~np.isfinite(filled)
            filled[still_nan & (neighbor_cnt > 0)] = fill_vals[still_nan & (neighbor_cnt > 0)]
            nan_mask = ~np.isfinite(filled)
        intensity = np.nan_to_num(filled, nan=0.0)
    else:
        intensity = np.nan_to_num(intensity, nan=0.0)

    # ----- Percentages -----
    percentages = {}
    for name in BIN_NAMES:
        pct = np.zeros((height, width), dtype=np.float32)
        m = total_cnt > 0
        pct[m] = (bin_cnt[name][m] / total_cnt[m]) * 100.0
        percentages[name] = pct

    # ----- Write outputs -----
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": dem["crs"],
        "transform": dst_transform,
        "nodata": -9999.0,
        "compress": "deflate",
    }

    outputs = {}

    def _write(name: str, data: np.ndarray) -> Path:
        path = output_dir / f"{stem}_{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)
        return path

    outputs["chm"] = _write("chm", chm)
    outputs["intensity"] = _write("intensity", intensity)
    for name in BIN_NAMES:
        outputs[name] = _write(name, percentages[name])

    logger.info(
        f"{stem}: CHM max={chm.max():.1f} ft | "
        f"intensity max={intensity.max():.1f} | "
        f"h1_2 max={percentages['h1_2'].max():.1f}% | "
        f"class_filter={'yes' if has_classification else 'no'}"
    )
    return outputs


def process_las_folder(
    las_dir: str | Path,
    dem_dir: str | Path,
    output_dir: str | Path,
    fill_intensity_gaps: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Process every LAS/LAZ file in a folder, one at a time.
    """
    las_dir = Path(las_dir)
    dem_dir = Path(dem_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    las_files = sorted(list(las_dir.glob("*.las")) + list(las_dir.glob("*.laz")))
    dem_files = list(dem_dir.glob("*.tif")) + list(dem_dir.glob("*.img"))

    if not las_files:
        raise ValueError(f"No LAS/LAZ files found in {las_dir}")
    if not dem_files:
        logger.warning(f"No DEM files found in {dem_dir}")

    logger.info(f"Found {len(las_files)} LAS files and {len(dem_files)} DEM files")
    logger.info("NOTE: DEM must be 1 m resolution – it anchors the pixel grid.")

    results = {}
    for las_path in tqdm(las_files, desc="Processing LAS files"):
        try:
            outs = process_single_las(
                las_path=las_path,
                dem_files=dem_files,
                output_dir=output_dir,
                fill_intensity_gaps=fill_intensity_gaps,
            )
            if outs:
                results[las_path.stem] = outs
        except Exception as e:
            logger.error(f"Failed on {las_path.name}: {e}")

    return results