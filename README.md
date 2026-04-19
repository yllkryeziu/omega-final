# Detecting Deforestation from Space

**osapiens Makeathon 2026** — Pixel-level deforestation detection using multi-modal satellite time series.

![Deforestation event example](content/deforestation.png)

---

## Architecture

```
                         Sentinel-2 (optical, 10m)
                                  |
                    +-------------+-------------+
                    |                           |
              [ U-TAE Model ]           [ Feature Extraction ]
              Spatial-temporal            78 handcrafted
              deep learning               temporal features
                    |                           |
              probability map           [ XGBoost Model ]
                    |                           |
                    |                     probability map
                    |                           |
                    +-------------+-------------+
                                  |
                         [ Gated Union Ensemble ]
                                  |
                    +------- binary mask -------+
                    |             |              |
              NDVI change    NBR change    S1 VV change
              (vegetation)   (burn/clear)   (structure)
                    |             |              |
                    +------  spectral gates ----+
                                  |
                    [ Morphological Cleanup ]
                    closing (fill) + opening (denoise)
                                  |
                    [ Time-of-Change Estimation ]
                    first month NDVI drops from baseline
                                  |
                         submission.geojson
```

---

## Pipeline

The full pipeline is automated in [`recipe.sh`](recipe.sh):

```bash
bash recipe.sh          # run everything
bash recipe.sh train    # run only training
bash recipe.sh predict  # run only prediction + ensemble
```

| Step | Script | What it does |
|------|--------|--------------|
| **1. Fuse labels** | [`build_dataset.py`](build_dataset.py) | Majority-vote fusion of 3 weak label sources (RADD, GLAD-L, GLAD-S2) into consensus training targets with gold/silver confidence tiers |
| **2a. Train XGBoost** | [`train_xgb.py`](train_xgb.py) | Extracts 78 temporal features per pixel (NDVI/NBR/NDWI change, S1 backscatter, AEF embedding diffs) and trains a gradient-boosted tree classifier |
| **2b. Train U-TAE** | [`train_utae.py`](train_utae.py) | Trains a U-Net with Temporal Attention Encoder on monthly S2 patches across 4 cross-validation folds |
| **3a. Predict XGBoost** | [`predict_xgb.py`](predict_xgb.py) | Generates per-pixel probability maps for all 5 test tiles |
| **3b. Predict U-TAE** | [`predict_utae.py`](predict_utae.py) | 4-fold ensemble prediction with probability maps |
| **4. Ensemble** | [`ensemble_v3.py`](ensemble_v3.py) | Gated union of 5 signals + morphological cleanup + time-of-change estimation |
| **5. Visualize** | [`visualize.py`](visualize.py) | 8-panel overview per tile: S2 before/after, predictions, labels, spectral indices |

---

## Data

```
data/makeathon-challenge/
  sentinel-2/         12-band optical imagery, monthly, 2020-2025 (~10m)
  sentinel-1/         VV radar backscatter, monthly, 2020-2025 (~30m)
  aef-embeddings/     64-dim AlphaEarth foundation model features, annual (~10m)
  labels/train/
    radd/             Radar-based deforestation alerts
    gladl/            Landsat-based forest loss alerts
    glads2/           Sentinel-2-based forest loss alerts (8 of 16 tiles)
```

**Train tiles**: 16 (8 South America + 8 Southeast Asia)
**Test tiles**: 5 (2 South America + 1 West Africa + 2 Southeast Asia)

---

## Models

### U-TAE

Spatial-temporal segmentation network from [Garnot & Landrieu (ICCV 2021)](https://github.com/VSainteuf/utae-paps). Architecture defined in [`models_utae.py`](models_utae.py).

- **Input**: Monthly S2 patches (128x128, 12 bands, up to 36 timesteps)
- **Fusion**: AEF embedding difference (2020 vs 2025) injected as skip-connection features
- **Training**: Consensus-weighted BCE loss (gold pixels weighted higher than silver)

### XGBoost

Pixel-level gradient-boosted trees on handcrafted temporal features.

| Feature group | Count | Source |
|--------------|-------|--------|
| Spectral indices (NDVI, NBR, NDWI) early/late/diff | 9 | Sentinel-2 |
| VV backscatter early/late/diff | 3 | Sentinel-1 |
| AEF embedding diffs + distance metrics | 66 | AlphaEarth |
| **Total** | **78** | |

### Ensemble Strategy

```
final = ML_union | (spectral_union & S1_mask) | (NDVI_mask & NBR_mask)
```

| Signal | Threshold | Role |
|--------|-----------|------|
| U-TAE probability | > 0.15 | Primary detector |
| XGBoost probability | > 0.25 | Primary detector |
| NDVI drop | > 0.12 | Supporting (vegetation loss) |
| NBR drop | > 0.10 | Supporting (burn/clearing) |
| S1 VV drop | > 2.0 dB | Gate for spectral signals |

ML models are trusted alone. Spectral signals require corroboration from radar or each other.

---

## Label Denoising

Three independent weak label sources are fused into consensus tiers:

```
           RADD    GLAD-L    GLAD-S2
             \       |       /
              majority vote
             /             \
     gold (all agree)    silver (majority)    uncertain (disagree)
         |                    |                     |
    full weight          reduced weight          excluded
```

| Tier | Description | Training weight |
|------|-------------|-----------------|
| Gold positive | All sources agree: deforestation | 1.0 |
| Gold negative | All sources agree: no change | 1.0 |
| Silver | Majority agrees | 0.5 |
| Uncertain | Sources disagree | 0.1 |

---

## Output Format

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Polygon", "coordinates": [[...]] },
      "properties": { "time_step": "2301" }
    }
  ]
}
```

- **geometry**: Deforestation polygons in EPSG:4326
- **time_step**: Estimated date in YYMM format (e.g. `"2301"` = January 2023), derived from first month NDVI drops below 2020 baseline
- **min area**: 0.25 ha (filters noise polygons)

---

## File Map

```
makeathon-challenge-2026/
  recipe.sh                  full pipeline script
  build_dataset.py           label fusion + CV splits
  train_xgb.py               XGBoost training
  train_utae.py              U-TAE training (per fold)
  models_utae.py             U-TAE architecture definition
  predict_xgb.py             XGBoost inference -> probability maps
  predict_utae.py            U-TAE inference -> probability maps
  ensemble_v3.py             gated union ensemble -> submission
  visualize.py               multi-panel tile visualization
  submission_utils.py        raster -> GeoJSON conversion
  model.md                   model description for presentation
  models/                    saved model weights
  submission/                prediction rasters + submission.geojson
  submission_xgb/            XGBoost probability maps
  figures/                   visualization PNGs
  data/                      satellite imagery + labels
```

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | AMD Instinct MI300X (206 GB VRAM) |
| Framework | PyTorch 2.5.1 + ROCm 7.0 |
| XGBoost | CPU (`tree_method="hist"`) |

---

## Quick Start

```bash
make install
make download_data_from_s3
source .venv/bin/activate
bash recipe.sh
# upload submission/submission.geojson to the leaderboard
```
