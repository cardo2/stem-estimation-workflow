"""
LAS / LAZ to 1 m raster layers for stem estimation model input.

Produces:
    - DSM.tif          (max Z per 1 m cell)
    - Intensity.tif    (mean intensity per cell)
    - h1_2.tif, h2_3.tif, h3_4.tif  (point counts in height ranges 1-2 m, 2-3 m, 3-4 m)

These layers (plus NAIP) form the 9-band input stack.

Uses laspy + numpy + rasterio for pure-Python processing (no PDAL required).
Future enhancement: optional DEM for height-above-ground calculation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_las_bounds_and_crs(las_files: List[Path]) -> Tuple[Tuple[float, float, float, float], CRS]:
    """Compute union bounds and common CRS from a list of LAS files."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    common_crs = None

    for las_path in las_files:
        with laspy.open(las_path) as las:
            header = las.header
            minx = min(minx, header.mins[0])
            miny = min(miny, header.mins[1])
            maxx = max(maxx, header.maxs[0])
            maxy = max(maxy, header.maxs[1])

            if common_crs is None:
                common_crs = CRS.from_wkt(header.vlr[0].record.crs.to_wkt()) if hasattr(header, 'vlr') else None
            # Fallback: try to read from laspy 2.x
            if common_crs is None:
                try:
                    common_crs = las.header.parse_crs()
                except Exception:
                    pass

    if common_crs is None:
        logger.warning("Could not determine CRS from LAS headers. Assuming EPSG:32616 (UTM 16N) or specify manually.")
        common_crs = CRS.from_epsg(32616)

    bounds = (minx, miny, maxx, maxy)
    return bounds, common_crs


def create_empty_raster(bounds: Tuple[float, float, float, float],
                        resolution: float = 1.0,
                        crs: CRS = None,
                        nodata: float = -9999.0) -> Tuple[np.ndarray, rasterio.Affine, Dict]:
    """Create an empty numpy array and transform for the given bounds and resolution."""
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    transform = from_origin(minx, maxy, resolution, resolution)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    data = np.full((height, width), nodata, dtype=np.float32)
    return data, transform, profile


def rasterize_las_points(las_path: Path,
                         bounds: Tuple[float, float, float, float],
                         resolution: float = 1.0,
                         height_bins: Optional[List[Tuple[float, float]]] = None) -> Dict[str, np.ndarray]:
    """
    Rasterize a single LAS file into DSM, intensity, and height-slice count layers.

    height_bins example: [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    """
    if height_bins is None:
        height_bins = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]

    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    # Initialize accumulators
    dsm = np.full((height, width), -9999.0, dtype=np.float32)
    intensity_sum = np.zeros((height, width), dtype=np.float64)
    intensity_count = np.zeros((height, width), dtype=np.uint32)

    h_slices = {f"h{int(b[0])}_{int(b[1])}": np.zeros((height, width), dtype=np.uint16)
                for b in height_bins}

    with laspy.open(las_path) as las:
        # Read in chunks to handle large files
        for points in las.chunk_iterator(1_000_000):
            x = points.x
            y = points.y
            z = points.z
            intensity = getattr(points, 'intensity', np.zeros_like(z))

            # Compute pixel indices
            col = ((x - minx) / resolution).astype(int)
            row = ((maxy - y) / resolution).astype(int)

            valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            if not np.any(valid):
                continue

            col = col[valid]
            row = row[valid]
            z = z[valid]
            intensity = intensity[valid]

            # DSM (max Z)
            np.maximum.at(dsm, (row, col), z)

            # Intensity accumulation
            np.add.at(intensity_sum, (row, col), intensity)
            np.add.at(intensity_count, (row, col), 1)

            # Height slices (absolute Z for v1.0)
            for (lo, hi), name in zip(height_bins, h_slices.keys()):
                mask = (z >= lo) & (z < hi)
                if np.any(mask):
                    np.add.at(h_slices[name], (row[mask], col[mask]), 1)

    # Finalize rasters
    result = {
        "dsm": np.where(dsm > -9998, dsm, np.nan).astype(np.float32),
        "intensity": np.where(intensity_count > 0, intensity_sum / intensity_count, np.nan).astype(np.float32),
    }
    for name, arr in h_slices.items():
        result[name] = arr.astype(np.float32)

    return result


def process_las_folder(las_dir: str | Path,
                       output_dir: str | Path,
                       resolution: float = 1.0,
                       height_bins: Optional[List[Tuple[float, float]]] = None,
                       overwrite: bool = False) -> Dict[str, Path]:
    """
    Process all .las/.laz files in a folder and produce merged 1 m rasters.

    Returns dict of output paths: {"dsm": Path, "intensity": Path, "h1_2": Path, ...}
    """
    las_dir = Path(las_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    las_files = sorted(list(las_dir.glob("*.las")) + list(las_dir.glob("*.laz")))
    if not las_files:
        raise ValueError(f"No .las or .laz files found in {las_dir}")

    logger.info(f"Found {len(las_files)} LAS/LAZ files in {las_dir}")

    # Compute global bounds
    bounds, crs = get_las_bounds_and_crs(las_files)
    logger.info(f"Global bounds: {bounds}, CRS: {crs}")

    # Create output rasters (initialize with nodata)
    dsm_full, transform, profile = create_empty_raster(bounds, resolution, crs)
    intensity_full = np.full_like(dsm_full, np.nan)
    h_full = {f"h{int(b[0])}_{int(b[1])}": np.full_like(dsm_full, 0, dtype=np.float32)
              for b in (height_bins or [(1.,2.), (2.,3.), (3.,4.)]) }

    # Process each file and merge (simple max/mean for overlapping areas)
    for las_path in tqdm(las_files, desc="Processing LAS files"):
        try:
            rasters = rasterize_las_points(las_path, bounds, resolution, height_bins)
        except Exception as e:
            logger.warning(f"Failed to process {las_path.name}: {e}")
            continue

        # Merge logic (simple: take max for DSM, mean where overlap for others)
        valid = ~np.isnan(rasters["dsm"])
        dsm_full[valid] = np.maximum(dsm_full[valid], rasters["dsm"][valid])

        valid_int = ~np.isnan(rasters["intensity"])
        # Weighted mean would be better; simple overwrite for v1
        intensity_full[valid_int] = rasters["intensity"][valid_int]

        for name in h_full:
            if name in rasters:
                h_full[name] = np.maximum(h_full[name], rasters[name])  # or add for counts

    # Write outputs
    outputs = {}
    profile.update(count=1)

    # DSM
    dsm_path = output_dir / "dsm.tif"
    with rasterio.open(dsm_path, "w", **profile) as dst:
        dst.write(dsm_full, 1)
    outputs["dsm"] = dsm_path
    logger.info(f"Written DSM: {dsm_path}")

    # Intensity
    int_path = output_dir / "intensity.tif"
    with rasterio.open(int_path, "w", **profile) as dst:
        dst.write(np.nan_to_num(intensity_full, nan=-9999).astype(np.float32), 1)
    outputs["intensity"] = int_path

    # Height slices
    for name, arr in h_full.items():
        p = output_dir / f"{name}.tif"
        with rasterio.open(p, "w", **profile) as dst:
            dst.write(arr, 1)
        outputs[name] = p
        logger.info(f"Written {name}: {p}")

    return outputs


if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--las_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    process_las_folder(args.las_dir, args.out_dir)