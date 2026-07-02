"""
Stack NAIP + LiDAR-derived rasters into a single 9-band GeoTIFF and normalize.

Band order (channels last):
0: DSM
1: Intensity
2: h1_2 (1-2 m)
3: h2_3 (2-3 m)
4: h3_4 (3-4 m)
5: Red   (NAIP)
6: Green (NAIP)
7: Blue  (NAIP)
8: NIR   (NAIP)

Normalization: clip using thresholds in config, then scale each band to [0, 1].
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_files(folder: Path, patterns: List[str]) -> List[Path]:
    files = []
    for pat in patterns:
        files.extend(sorted(folder.glob(pat)))
    return files


def get_reference_profile(naip_files: List[Path]) -> Dict:
    """Use the first NAIP file as the reference grid / CRS / resolution."""
    with rasterio.open(naip_files[0]) as src:
        profile = src.profile.copy()
        profile.update(count=9, dtype="float32", nodata=-9999.0)
        return profile, src.transform, src.crs, src.res


def align_and_stack(naip_dir: Path,
                    lidar_rasters: Dict[str, Path],
                    output_path: Path,
                    normalize_config: Dict[str, List[float]],
                    resolution: float = 1.0) -> Path:
    """
    Align all LiDAR layers to NAIP grid, stack with NAIP bands, normalize, and write 9-band stack.
    """
    naip_files = find_files(naip_dir, ["*.tif", "*.tiff"])
    if not naip_files:
        raise ValueError(f"No NAIP GeoTIFFs found in {naip_dir}")

    ref_profile, ref_transform, ref_crs, ref_res = get_reference_profile(naip_files)

    # For simplicity in v1: assume NAIP is already ~1 m and we reproject LiDAR to it.
    # In production, use a more robust mosaic + reproject workflow.

    logger.info(f"Reference NAIP CRS: {ref_crs}, transform: {ref_transform}")

    # Read and reproject each LiDAR layer to reference grid
    band_data = []
    band_names = ["dsm", "intensity", "h1_2", "h2_3", "h3_4"]

    for bname in band_names:
        if bname not in lidar_rasters:
            logger.warning(f"Missing LiDAR band: {bname}. Filling with zeros.")
            # Create zero array matching reference shape
            with rasterio.open(naip_files[0]) as src:
                zeros = np.zeros((src.height, src.width), dtype=np.float32)
            band_data.append(zeros)
            continue

        src_path = lidar_rasters[bname]
        with rasterio.open(src_path) as src:
            # Reproject to reference
            dst_array = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            band_data.append(dst_array)

    # Now add NAIP bands (assume 4-band NAIP)
    for naip_path in tqdm(naip_files, desc="Processing NAIP tiles"):
        with rasterio.open(naip_path) as src:
            for b in range(1, 5):
                arr = src.read(b).astype(np.float32)
                # Simple mosaic: take first valid or last tile wins for overlapping NAIP
                # For production use rasterio.merge or exact grid handling
                if len(band_data) < 5 + b:
                    band_data.append(arr)
                else:
                    # Placeholder: last tile wins
                    mask = arr != src.nodata
                    band_data[4 + b][mask] = arr[mask]

    # At this point band_data should have 9 arrays. Stack them.
    if len(band_data) != 9:
        logger.warning(f"Expected 9 bands, got {len(band_data)}. Padding/truncating.")
        while len(band_data) < 9:
            band_data.append(np.zeros_like(band_data[0]))
        band_data = band_data[:9]

    stack = np.stack(band_data, axis=0).astype(np.float32)  # (9, H, W)

    # Normalize (clip + scale to [0,1])
    thresholds = {
        0: normalize_config.get("dsm", [0, 118]),
        1: normalize_config.get("intensity", [0, 167]),
        2: normalize_config.get("h1_2", [0, 100]),
        3: normalize_config.get("h2_3", [0, 100]),
        4: normalize_config.get("h3_4", [0, 100]),
        # NAIP bands: usually 0-255 or 0-10000; simple min-max or assume already suitable
        5: normalize_config.get("red", [0, 255]),
        6: normalize_config.get("green", [0, 255]),
        7: normalize_config.get("blue", [0, 255]),
        8: normalize_config.get("nir", [0, 255]),
    }

    for b in range(9):
        lo, hi = thresholds.get(b, [0, 1])
        band = stack[b]
        band = np.clip(band, lo, hi)
        if hi > lo:
            band = (band - lo) / (hi - lo)
        stack[b] = np.nan_to_num(band, nan=0.0).astype(np.float32)

    # Write multi-band stack
    out_profile = ref_profile.copy()
    out_profile.update(count=9, dtype="float32", nodata=-9999.0, compress="deflate")

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(stack)

    logger.info(f"9-band normalized stack written to {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example call would go here
    print("stack_normalize module loaded.")