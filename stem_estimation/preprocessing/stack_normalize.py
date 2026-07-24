"""
Align LiDAR layers + NAIP, normalize to 0–254 uint8, write stacked GeoTIFFs.

Band order (fixed):
  1 CHM | 2 Intensity | 3 h1_2 | 4 h2_3 | 5 h3_4
  6 NAIP-R | 7 NAIP-G | 8 NAIP-B | 9 NAIP-NIR
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from shapely.geometry import box
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    "chm": (0.0, 118.0),
    "intensity": (0.0, 43000.0),
    "h1_2": (0.0, 100.0),
    "h2_3": (0.0, 100.0),
    "h3_4": (0.0, 100.0),
}

LIDAR_KEYS = ["chm", "intensity", "h1_2", "h2_3", "h3_4"]


def _norm_to_254(data: np.ndarray, mn: float, mx: float) -> np.ndarray:
    data = np.clip(data.astype(np.float32), mn, mx)
    scaled = (data - mn) * (254.0 / max(mx - mn, 1e-6))
    return np.clip(scaled, 0, 254).astype(np.uint8)


def _find_files(folder: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(folder.glob(pat)))
    return files


def stack_one_tile(
    stem: str,
    lidar_dir: Path,
    naip_files: List[Path],
    out_dir: Path,
    thresholds: Dict[str, Tuple[float, float]] = None,
) -> Optional[Path]:
    """
    Stack one complete set of LiDAR layers + overlapping NAIP.
    Grid is taken from the CHM (which is already DEM-aligned).
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    required = {k: lidar_dir / f"{stem}_{k}.tif" for k in LIDAR_KEYS}
    if not all(p.exists() for p in required.values()):
        logger.warning(f"{stem}: missing LiDAR layer(s) – skipped")
        return None

    with rasterio.open(required["chm"]) as src:
        master_crs = src.crs
        master_transform = src.transform
        master_w, master_h = src.width, src.height
        master_bounds = src.bounds

    # --- LiDAR bands ---
    lidar_norm = []
    for key in LIDAR_KEYS:
        with rasterio.open(required[key]) as src:
            data = src.read(1).astype(np.float32)
        mn, mx = thresholds[key]
        lidar_norm.append(_norm_to_254(data, mn, mx))

    # --- NAIP (first overlapping file) ---
    naip_stack = np.zeros((4, master_h, master_w), dtype=np.float32)
    found = False
    for naip_path in naip_files:
        with rasterio.open(naip_path) as src:
            try:
                nb = transform_bounds(src.crs, master_crs, *src.bounds)
            except Exception:
                continue
            if not box(*nb).intersects(box(*master_bounds)):
                continue
            for b in range(min(4, src.count)):
                reproject(
                    source=rasterio.band(src, b + 1),
                    destination=naip_stack[b],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=master_transform,
                    dst_crs=master_crs,
                    resampling=Resampling.bilinear,
                    dst_nodata=0,
                )
            found = True
            logger.info(f"{stem}: NAIP = {naip_path.name}")
            break

    if not found:
        logger.warning(f"{stem}: no overlapping NAIP – skipped")
        return None

    naip_norm = np.clip(naip_stack, 0, 255).astype(np.uint8)
    stacked = np.concatenate([np.stack(lidar_norm, axis=0), naip_norm], axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_stacked.tif"
    profile = {
        "driver": "GTiff",
        "height": master_h,
        "width": master_w,
        "count": 9,
        "dtype": "uint8",
        "crs": master_crs,
        "transform": master_transform,
        "nodata": 255,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)
        dst.descriptions = (
            "CHM", "Intensity", "h1_2", "h2_3", "h3_4",
            "NAIP_R", "NAIP_G", "NAIP_B", "NAIP_NIR",
        )

    logger.info(f"{stem}: wrote {out_path.name}  shape={stacked.shape}")
    return out_path


def align_and_stack(
    lidar_dir: str | Path,
    naip_dir: str | Path,
    output_dir: str | Path,
    thresholds: Dict[str, Tuple[float, float]] = None,
    max_tiles: Optional[int] = None,
) -> List[Path]:
    """
    Stack every complete LiDAR tile that has overlapping NAIP.
    Set max_tiles to limit for testing.
    """
    lidar_dir = Path(lidar_dir)
    naip_dir = Path(naip_dir)
    output_dir = Path(output_dir)
    thresholds = thresholds or DEFAULT_THRESHOLDS

    chm_files = sorted(lidar_dir.glob("*_chm.tif"))
    naip_files = _find_files(naip_dir, ["*.tif", "*.tiff", "*.TIF", "*.TIFF"])
    if not chm_files:
        raise ValueError(f"No *_chm.tif files in {lidar_dir}")
    if not naip_files:
        raise ValueError(f"No NAIP files in {naip_dir}")

    results = []
    for chm in tqdm(chm_files, desc="Stacking"):
        if max_tiles is not None and len(results) >= max_tiles:
            break
        stem = chm.name.replace("_chm.tif", "")
        path = stack_one_tile(stem, lidar_dir, naip_files, output_dir, thresholds)
        if path is not None:
            results.append(path)

    logger.info(f"Stacked {len(results)} tiles → {output_dir}")
    return results