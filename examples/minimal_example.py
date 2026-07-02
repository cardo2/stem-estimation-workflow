"""
Minimal example to test the stem estimation workflow on a small dataset.

1. Create tiny test folders with sample NAIP and LAS (or use real small subsets).
2. Update config.yaml with test paths.
3. Run this script or the full pipeline.
"""

from pathlib import Path
from stem_estimation.pipeline import run_pipeline

if __name__ == "__main__":
    # Example: point to a config that uses small test data
    config_path = Path(__file__).parent.parent / "config.yaml"

    print("Running minimal pipeline test...")
    print(f"Using config: {config_path}")

    # You can also pass a modified dict instead of editing the yaml
    run_pipeline(config_path)