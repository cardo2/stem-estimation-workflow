"""Build a spatial index of stacked GeoTIFF footprints for mosaic tiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import rasterio
from rasterio.features import bounds as feature_bounds
from shapely.geometry import box, mapping
from tqdm import tqdm


def build_stack_index(
    stacked_dir: str | Path,
    output_geojson: str | Path,
    pattern: str = "*_stacked.tif",
) -> Path:
    """
    Scan stacked tiles and write a GeoJSON FeatureCollection.
    Each feature: geometry = footprint, properties = {path, stem, width, height, crs}
    """
    stacked_dir = Path(stacked_dir)
    output_geojson = Path(output_geojson)
    files = sorted(stacked_dir.glob(pattern))
    if not files:
        raise ValueError(f"No files matching {pattern} in {stacked_dir}")

    features = []
    crs_wkt = None

    for f in tqdm(files, desc="Indexing stacked tiles"):
        with rasterio.open(f) as src:
            b = src.bounds
            geom = box(b.left, b.bottom, b.right, b.top)
            if crs_wkt is None and src.crs is not None:
                crs_wkt = src.crs.to_wkt()
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {
                        "path": str(f.resolve()),
                        "stem": f.stem,
                        "width": src.width,
                        "height": src.height,
                        "crs": src.crs.to_string() if src.crs else None,
                    },
                }
            )

    fc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_wkt}} if crs_wkt else None,
        "features": features,
    }
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    with open(output_geojson, "w", encoding="utf-8") as fh:
        json.dump(fc, fh, indent=2)

    print(f"Indexed {len(features)} tiles → {output_geojson}")
    return output_geojson