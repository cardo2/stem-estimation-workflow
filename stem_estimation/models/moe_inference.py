"""
Mixture-of-Experts inference module.

- Loads discriminator + 3 expert CNNs (Keras .keras format)
- For each tile: discriminator decides routing → expert predicts stem density
- Returns per-tile predictions + which model was used
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)

# Suppress TF warnings in production
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


def load_models(model_paths: Dict[str, str]) -> Dict[str, keras.Model]:
    """Load the four Keras models. Paths come from config."""
    models = {}
    for name, path in model_paths.items():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Model '{name}' not found at {path}. "
                "Please download the models from the release and update config.yaml."
            )
        models[name] = keras.models.load_model(p, compile=False)
        logger.info(f"Loaded {name} from {path}")
    return models


def predict_tile(tile_array: np.ndarray,
                 models: Dict[str, keras.Model],
                 batch_size: int = 16) -> Tuple[float, str]:
    """
    Run discriminator on a single tile (or batch) and route to the best expert.

    tile_array: shape (H, W, 9) or (N, H, W, 9) float32 in [0,1]
    Returns: (predicted_stems_per_ha, model_used)
    """
    if tile_array.ndim == 3:
        tile_array = np.expand_dims(tile_array, axis=0)

    # Discriminator expects same input shape
    disc = models["discriminator"]
    probs = disc.predict(tile_array, batch_size=batch_size, verbose=0)
    class_idx = int(np.argmax(probs[0]))

    # Map index to expert name (adjust if your training used different ordering)
    class_map = {0: "low_expert", 1: "base_expert", 2: "high_expert"}
    expert_name = class_map.get(class_idx, "base_expert")
    model_used = expert_name.replace("_expert", "")

    expert = models[expert_name]
    pred = expert.predict(tile_array, batch_size=batch_size, verbose=0)[0][0]

    # Ensure non-negative
    pred = max(0.0, float(pred))

    return pred, model_used


def run_moe_on_tiles(stack_path: Path,
                     tile_gdf: "gpd.GeoDataFrame",
                     model_paths: Dict[str, str],
                     batch_size: int = 16,
                     tile_size_px: int = 100) -> List[Dict]:
    """
    Iterate over tiles, read chips from the stack, run MoE, collect predictions.
    """
    models = load_models(model_paths)
    results = []

    with rasterio.open(stack_path) as src:
        for _, row in tile_gdf.iterrows():
            window = Window(row.col_off, row.row_off, row.width_px, row.height_px)
            chip = src.read(window=window)  # (9, H, W)
            chip = np.moveaxis(chip, 0, -1)  # → (H, W, 9)
            chip = np.nan_to_num(chip, nan=0.0).astype(np.float32)

            # Pad if needed (edge tiles)
            if chip.shape[0] != tile_size_px or chip.shape[1] != tile_size_px:
                pad_h = tile_size_px - chip.shape[0]
                pad_w = tile_size_px - chip.shape[1]
                chip = np.pad(chip, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

            pred_density, model_used = predict_tile(chip, models, batch_size=batch_size)

            results.append({
                "tile_id": row.tile_id,
                "predicted_stems_per_ha": round(pred_density, 2),
                "model_used": model_used,
                "valid_fraction": row.valid_fraction
            })

    logger.info(f"MoE inference completed on {len(results)} tiles.")
    return results


# Quick shape check helper
def check_model_input_shape(model: keras.Model, expected_bands: int = 9):
    input_shape = model.input_shape
    logger.info(f"Model expects input shape: {input_shape}")
    if input_shape[-1] != expected_bands:
        logger.warning(f"Model channel dimension {input_shape[-1]} != expected {expected_bands}")