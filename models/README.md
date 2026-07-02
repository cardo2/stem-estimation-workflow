# Model Files (Placeholders)

This folder should contain the four trained Keras models:

- `discriminator.keras` — Routes tiles to the best expert (Low / Base / High).
- `low_expert.keras`   — Transfer-learned expert for low-density / sparse / agricultural landscapes.
- `base_expert.keras`  — Core model trained on the primary West Lafayette ~44 km² dataset.
- `high_expert.keras`  — Transfer-learned expert for high-density hardwood forests.

**Until the models are publicly released**, this folder contains only this README.

## Planned Release (Free Hosting)

Models will be made available via **GitHub Releases** of the companion repository (or the main workflow repo). This is free and does not require Git LFS for model sizes typical of these CNNs (~10–50 MB each).

When released, you will be able to:

```bash
# Example (adjust owner/repo/tag when available)
wget https://github.com/<org>/stem-estimation-workflow/releases/download/v1.0/discriminator.keras -P models/
# ... repeat for the other three files
```

Or use the Python helper (future):

```python
from stem_estimation.models import download_models
download_models(version="v1.0", dest="models/")
```

## Model Metadata

A `model_metadata.json` will accompany the release with:
- Expected input shape `(None, tile_px, tile_px, 9)`
- Class mapping for discriminator
- Normalization thresholds used during training
- Recommended tile sizes
- Performance metrics on held-out test sets
- Training date / commit hash for traceability

## Important Notes

- All models expect **float32 input in [0, 1]** after the clipping + scaling described in `config.yaml` and the main README.
- Input bands order (channels last): `[DSM, Intensity, h1_2, h2_3, h3_4, R, G, B, NIR]`.
- Do **not** fine-tune these models for production use without also updating/retraining the discriminator on the new data distribution.

If you need the models urgently for replication or collaboration before public release, please contact the corresponding author.

---

*These models are provided for research and non-commercial use in accordance with the original paper's license/terms.*