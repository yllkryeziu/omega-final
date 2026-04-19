# osapiens Makeathon 2026 - Detecting Deforestation from Space

## Challenge Summary

Build an ML system that detects **deforestation events after 2020** at the pixel level using multimodal satellite time series. Deforestation = permanent removal of tree cover (forest in 2020 -> non-forest). The system must generalize across unseen geographic regions.

## Evaluation

- **Quantitative**: Hidden test set leaderboard (live during hackathon)
- **Qualitative**: Jury evaluates model design, noise handling, generalization, scalability, clarity
- Final ranking = combination of both

## Repository Structure

```
makeathon-challenge-2026/
├── challenge.ipynb              # Main walkthrough: data, labels, visualization, submission example
├── submission_utils.py          # raster_to_geojson() - converts binary prediction raster -> GeoJSON
├── osapiens-challenge-full-description.md
├── alphaearth_workshop.ipynb    # Workshop notebook: AEF embeddings, PCA/UMAP, SVM/MLP classifiers
├── build_dataset.py             # Fuses 3 weak label sources -> consensus training dataset
├── train_xgb.py                 # Tier 1: feature extraction + XGBoost CV training
├── predict_xgb.py               # Tier 1: test predictions -> GeoJSON submission
├── download_data.py             # Data download entrypoint (run via `make download_data_from_s3`)
├── Makefile                     # `make install` (venv + deps), `make download_data_from_s3`
├── requirements.txt             # geopandas, numpy, matplotlib, pandas, rasterio, shapely, boto3, tqdm
└── data/
    ├── makeathon-challenge/     # Challenge data (21 tiles: 16 train + 5 test)
    │   ├── sentinel-1/          # Radar (SAR) time series
    │   ├── sentinel-2/          # Optical imagery time series
    │   ├── aef-embeddings/      # AlphaEarth foundation model embeddings
    │   ├── labels/train/        # 3 weak label sources (NO test labels)
    │   └── metadata/            # train_tiles.geojson, test_tiles.geojson
    ├── fused-labels/            # Fused consensus labels (built by build_dataset.py)
    │   ├── {tile_id}/           # Per-tile: fused_binary.tif, consensus_mask.tif, source_*.tif
    │   └── dataset_meta.json    # Stats, CV folds, encoding docs
    └── alphaearth-workshop/     # Workshop data (3 tiles with HARD labels)
        ├── aef_segmentation_embeddings/consolidated_tiles/  # 3 AEF tiles (2022 only)
        └── annotations/alphaearth/                          # 420 hand-drawn polygons
```

---

## Challenge Data (21 tiles)

### Tile IDs

Format: `{MGRS_grid}_{x}_{y}` (e.g. `18NWG_6_6`). MGRS = Military Grid Reference System.

**16 train tiles** (with weak labels):

| Region | Tiles | UTM Zone |
|--------|-------|----------|
| South America (Colombia/Peru) | 18NWG_6_6, 18NWH_1_4, 18NWJ_8_9, 18NWM_9_4, 18NXH_6_8, 18NXJ_7_6, 18NYH_9_9, 19NBD_4_4 | EPSG:32618/32619 |
| Southeast Asia (Myanmar/Thailand/Laos/Cambodia) | 47QMB_0_8, 47QQV_2_4, 48PUT_0_8, 48PWV_7_8, 48PXC_7_7, 48PYB_3_6, 48QVE_3_0, 48QWD_2_2 | EPSG:32647/32648 |

**5 test tiles** (no labels):

| Tile | Region | Generalization Challenge |
|------|--------|--------------------------|
| 18NVJ_1_6 | South America | New MGRS grid in known zone |
| 18NYH_2_1 | South America | Same MGRS grid as train tile 18NYH_9_9, different sub-tile |
| 33NTE_5_1 | **West Africa** | **Completely unseen region** (EPSG:32633) |
| 47QMA_6_2 | Southeast Asia | New MGRS grid in known zone |
| 48PWA_0_6 | Southeast Asia | New MGRS grid in known zone |

### Sentinel-2 (Optical)

- **Path**: `sentinel-2/{train|test}/{tile_id}__s2_l2a/{tile_id}__s2_l2a_{year}_{month}.tif`
- **Bands**: 12 spectral bands (B01 Aerosol, B02 Blue, B03 Green, B04 Red, B05-B07 Red Edge, B08 NIR, B8A Narrow NIR, B09 Water Vapour, B10 Cirrus, B11-B12 SWIR), all upsampled to 10m
- **Shape**: ~1002x1002 pixels
- **Dtype**: uint16, value range 0-~8500
- **Nodata**: 0
- **CRS**: Local UTM projected (e.g. EPSG:32618)
- **Temporal**: Monthly best cloud-free scene, 2020-2025 (~72 files per tile)
- **Total files**: ~1150 train + ~343 test

### Sentinel-1 (Radar/SAR)

- **Path**: `sentinel-1/{train|test}/{tile_id}__s1_rtc/{tile_id}__s1_rtc_{year}_{month}_{ascending|descending}.tif`
- **Bands**: 1 (VV polarization, linear scale backscatter)
- **Shape**: ~334x335 pixels (~30m resolution)
- **Dtype**: float32, value range ~0.01-2.2
- **Nodata**: NaN
- **CRS**: Local UTM (aligned with S2)
- **Temporal**: Monthly, 2020-2025. Both ascending/descending orbits in 2020-2021; descending-only 2022-2024; mixed 2025 (~118 files per train tile)
- **Processing**: Radiometrically Terrain Corrected (RTC)
- **Key advantage**: All-weather, day-and-night (penetrates clouds)
- **Total files**: ~1882 train + ~563 test

### AlphaEarth Foundation Embeddings (AEF)

- **Path**: `aef-embeddings/{train|test}/{tile_id}_{year}.tiff`
- **Bands**: 64 embedding dimensions
- **Shape**: ~1004x998 pixels (~10m resolution)
- **Dtype**: float32, value range ~-0.27 to ~0.2
- **Nodata**: NaN (~0.2% of pixels)
- **CRS**: EPSG:4326 (WGS-84) -- DIFFERENT from S1/S2! Must reproject to UTM before combining
- **Temporal**: Annual (one per year), 2020-2025 (6 files per tile)
- **Total files**: 96 train + 30 test

### Weak Labels (Train Only)

All labels are **weak/noisy** (model predictions, not ground truth). They may conflict with each other. All in EPSG:4326.

#### RADD (Radar for Detecting Deforestation)

- **Path**: `labels/train/radd/radd_{tile_id}_labels.tif`
- **Shape**: ~905x899 pixels
- **Dtype**: uint16
- **Coverage**: All 16 train tiles (16 files)
- **Encoding**: Single integer encodes confidence + date
  - `0` = no alert
  - Leading digit `2` = low confidence, `3` = high confidence
  - Remaining digits = days since 2014-12-31
  - Example: `21847` = low-confidence alert on 2020-01-21
- **Sample stats** (18NWG_6_6): 532K no-alert pixels, 281K alert pixels, 222 unique timestamps

#### GLAD-L (Landsat-based)

- **Path**: `labels/train/gladl/gladl_{tile_id}_alert{YY}.tif` + `gladl_{tile_id}_alertDate{YY}.tif`
- **Shape**: ~362x360 pixels (coarser than RADD/GLAD-S2 due to Landsat resolution)
- **Dtype**: uint8 (alert), uint16 (alertDate)
- **Coverage**: All 16 train tiles x 5 years = 160 files
- **Years**: YY = 21, 22, 23, 24, 25 (separate file per year)
- **alert encoding**: 0=no loss, 2=probable, 3=confirmed
- **alertDate encoding**: day-of-year within year 20YY (0=no alert)
- **Sample stats** (18NWG_6_6, 2021): 124K no-loss, 89 probable, 5324 confirmed

#### GLAD-S2 (Sentinel-2-based)

- **Path**: `labels/train/glads2/glads2_{tile_id}_alert.tif` + `glads2_{tile_id}_alertDate.tif`
- **Shape**: ~905x899 pixels
- **Dtype**: uint8 (alert), uint16 (alertDate)
- **Coverage**: Only 8 of 16 train tiles (not all tiles!) -- 16 files total
- **alert encoding**: 0=no loss, 1=recent-only, 2=low confidence, 3=medium, 4=high confidence
- **alertDate encoding**: days since 2019-01-01 (0=no alert)
- **Single file** covers all years (no YY suffix)
- **Sample stats** (18NWG_6_6): 425K no-loss, 2K recent, 52K low, 112K medium, 223K high confidence

---

## Workshop Data (3 tiles with HARD labels)

The workshop provides **hand-drawn polygon annotations** for land cover classification on 3 tiles in Southeast Asia. Unlike the challenge's weak labels, these are **human-curated ground truth**.

### AEF Embedding Tiles

- **Path**: `data/alphaearth-workshop/aef_segmentation_embeddings/consolidated_tiles/`
- **Files**:
  - `aef_2022_48PUB_3_0_1262780ares.tif` (1123x1137 pixels, ~91 MB)
  - `aef_2022_48PVC_3_0_1287913ares.tif` (1137x1137 pixels, ~95 MB)
  - `aef_2022_48QUD_2_4_1250519ares.tif` (1138x1111 pixels, ~91 MB)
- **Bands**: 64 AEF embedding dimensions
- **Dtype**: uint8 (quantized -- must be dequantized, unlike the challenge AEF tiles which are float32)
- **CRS**: EPSG:4326
- **Temporal**: 2022 only
- **Dequantization**: `dequantized = -sign(val - 127) * |abs(val - 127) / 127.5|^2`. Groups of 64 consecutive zeros = NaN (missing data).

### Hand-Drawn Annotations

- **Path**: `data/alphaearth-workshop/annotations/alphaearth/`
- **Files**:
  - `workshop_annotations.geojson` (current, with corrections)
  - `workshop_annotations_original.geojson` (original baseline)
- **CRS**: EPSG:4326
- **Total**: 420 polygons across 3 tiles
- **Properties**: `class` (string), `tiff_file` (string path to AEF tile)

**Class distribution**:

| Class | Count | Description |
|-------|-------|-------------|
| plantation | 218 | Tree plantations (e.g. palm oil, rubber) |
| crops | 73 | Cropland |
| forest | 66 | Natural forest |
| water | 31 | Water bodies |
| built | 19 | Built-up / urban |
| shrubs_and_scrubs | 13 | Shrubland |

**Per-tile coverage**: 48PVC has 231 polygons (55%), 48PUB has 99 (24%), 48QUD has 90 (21%)

**Difference between files**: One polygon in 48PUB was reclassified from "built" to "water" in the corrected version.

### Workshop Tiles vs Challenge Tiles

The 3 workshop tiles (48PUB, 48PVC, 48QUD) are **different tiles** from the 21 challenge tiles but are in the same Southeast Asia region (MGRS zone 48). They have:
- Only AEF embeddings (no S1/S2 imagery)
- Only 2022 data (no time series)
- Hard labels for **land cover classification** (6 classes), not deforestation detection
- Potential use: pre-training, transfer learning, or understanding what AEF embeddings encode about forest vs non-forest

### AEF Embedding Semantic Dimensions

Key learned associations from the workshop:

| Dimensions | Land Cover Class |
|-----------|------------------|
| A16, A23 | Tree Cover (high NIR, closed canopy) |
| A12, A50 | Cropland (phenological regularity) |
| A05, A27 | Mangroves (intertidal, waterlogging) |
| A09, A35 | Built-up/Urban |
| A04, A11, A25, A29 | Herbaceous Wetland |
| A18, A21, A26 | Shrubland |
| A41 | Grassland |
| A63 | Bare/Sparse Vegetation |
| A64 | Permanent Water |

---

## CRS Alignment

- **S1 and S2**: Local UTM projected CRS (e.g. EPSG:32618, 32647, 32648)
- **AEF, all labels, workshop data**: EPSG:4326 (geodetic lat/lon)
- When combining data sources, **reproject to the same CRS** (typically UTM). Use `rasterio.warp.reproject` with `Resampling.nearest` for labels, `Resampling.bilinear` for continuous data.

## Resolution Summary

| Source | Pixel Size | Tile Dimensions | Notes |
|--------|-----------|-----------------|-------|
| Sentinel-2 | 10m | ~1002x1002 | Reference grid |
| Sentinel-1 | 30m | ~334x335 | 3x coarser than S2 |
| AEF (challenge) | ~10m | ~1004x998 | Close to S2 |
| AEF (workshop) | ~10m | ~1130x1130 | Larger tiles |
| RADD labels | ~10m | ~905x899 | - |
| GLAD-L labels | ~30m | ~362x360 | Coarsest labels |
| GLAD-S2 labels | ~10m | ~905x899 | Same res as RADD |

## Submission Pipeline

1. Generate binary prediction raster per test tile (1=deforestation, 0=no deforestation)
2. Use `submission_utils.raster_to_geojson(raster_path, output_path, min_area_ha=0.5)`:
   - Vectorizes binary raster into polygons
   - Filters out polygons < 0.5 ha
   - Reprojects to EPSG:4326
   - Outputs GeoJSON FeatureCollection
3. Each Feature has a `time_step` property (set to None by default; can optionally encode when deforestation occurred)
4. Upload `.geojson` files to leaderboard

## File Count Summary

| Source | Train | Test | Total |
|--------|-------|------|-------|
| Sentinel-2 | ~1150 | ~343 | ~1493 |
| Sentinel-1 | ~1882 | ~563 | ~2445 |
| AEF (challenge) | 96 | 30 | 126 |
| RADD labels | 16 | 0 | 16 |
| GLAD-L labels | 160 | 0 | 160 |
| GLAD-S2 labels | 16 | 0 | 16 |
| Workshop AEF | 3 | - | 3 |
| Workshop annotations | 2 | - | 2 |

## Key Technical Considerations

- **Noisy labels**: All 3 challenge label sources are weak/noisy predictions, not ground truth. They may conflict with each other. Combining them intelligently (label fusion, confidence weighting) is critical.
- **GLAD-S2 partial coverage**: Only 8 of 16 train tiles have GLAD-S2 labels.
- **S1 orbit gaps**: Ascending orbit data is missing 2022-2024 for some tiles. Only descending orbits available.
- **CRS mismatch**: AEF embeddings and labels are EPSG:4326; S1/S2 are UTM. Always reproject before combining.
- **Resolution mismatch**: S2=10m, S1=30m, AEF=~10m. Resample to common grid.
- **Cloud contamination**: S2 optical data affected by clouds; S1 radar is cloud-penetrating.
- **Generalization**: Test tile 33NTE_5_1 (West Africa) is a completely unseen region. Model must not overfit to specific geographies.
- **Temporal**: Deforestation = change detection. Comparing pre-2020 (forest baseline) vs. post-2020 states is fundamental.
- **Workshop data as auxiliary**: The 420 hard-labeled polygons can potentially help with pre-training or understanding AEF embedding semantics for forest vs non-forest discrimination.

## Bonus Objectives

- Predict **when** deforestation occurred (month/year) via the `time_step` field
- Handle label uncertainty / estimate confidence
- Build a visualization or monitoring tool

## Fused Label Dataset (`data/fused-labels/`)

Built by `build_dataset.py`. For each of the 16 train tiles, fuses the 3 weak label sources into a single training target and consensus evaluation mask.

### Per-Tile Outputs

| File | Description |
|------|-------------|
| `fused_binary.tif` | Binary deforestation label (0/1) from majority vote |
| `consensus_mask.tif` | Evaluation tier: 0=uncertain, 1=gold neg, 2=silver neg, 3=silver pos, 4=gold pos |
| `source_radd.tif` | Binarised RADD (post-2020, any confidence) |
| `source_gladl.tif` | Binarised GLAD-L (confirmed loss, union of 2021-2025) |
| `source_glads2.tif` | Binarised GLAD-S2 (confidence >= 2). Only 8 South America tiles. |

All outputs are on the RADD/GLAD-S2 native grid (EPSG:4326, ~10m).

### Consensus Tiers

- **Gold positive (4)**: All available sources agree deforestation + high confidence in >= 1 source. Use for evaluation.
- **Gold negative (1)**: All sources agree no deforestation. Use for evaluation.
- **Silver positive/negative (3/2)**: Majority vote with disagreement. Use for training but not strict evaluation.
- **Uncertain (0)**: Sources disagree (2-way tiles only). Exclude from evaluation.

### Dataset Statistics

| Tier | Pixels | % of Total |
|------|--------|------------|
| Gold negative | 10,989,171 | 82.3% |
| Gold positive | 747,201 | 5.6% |
| Silver negative | 630,405 | 4.7% |
| Silver positive | 400,460 | 3.0% |
| Uncertain | 590,812 | 4.4% |
| **Evaluable (gold)** | **11,736,372** | **87.9%** |

### CV Folds (region-stratified, tile-level)

Each fold holds out 2 South America + 2 Southeast Asia tiles:

| Fold | Validation Tiles |
|------|-----------------|
| 0 | 18NWG_6_6, 18NWH_1_4, 47QMB_0_8, 47QQV_2_4 |
| 1 | 18NWJ_8_9, 18NWM_9_4, 48PUT_0_8, 48PWV_7_8 |
| 2 | 18NXH_6_8, 18NXJ_7_6, 48PXC_7_7, 48PYB_3_6 |
| 3 | 18NYH_9_9, 19NBD_4_4, 48QVE_3_0, 48QWD_2_2 |

Full metadata in `data/fused-labels/dataset_meta.json`.

### Evaluation Protocol

1. Train on fold's train tiles using `fused_binary.tif` as labels
2. Predict on fold's val tiles
3. Evaluate only on gold pixels (`consensus_mask == 1 or 4`): compute F1, precision, recall
4. Additionally check physical plausibility on predictions (temporal monotonicity, NDVI consistency, spatial coherence)

## Modeling Roadmap

### Tier 1: XGBoost Baseline (IMPLEMENTED)

Pixel-level temporal feature extraction + XGBoost. No spatial context.

- **Scripts**: `train_xgb.py` (train + CV), `predict_xgb.py` (test predictions + GeoJSON)
- **Features** (~78 per pixel):
  - S2: NDVI, NBR, NDWI early mean (2020) vs late mean (2024-2025) + diffs (9 features)
  - S1: VV backscatter dB early/late/diff (3 features)
  - AEF: 64-dim embedding diffs (2020 vs 2025) + cosine distance + L2 distance (66 features)
- **GPU acceleration**: AEF reprojection (66 bands) and S1 reprojection (3 bands) use `torch.grid_sample` on MI300X. Coordinate mapping via pyproj (CPU, ~0.15s), pixel resampling on GPU.
- **XGBoost**: CPU `tree_method="hist"`, 500 trees, depth 8, lr=0.05
- **Known issues**: Tiles 18NWM_9_4 (all S2 corrupt) and 18NYH_9_9 (partially corrupt) are skipped/degraded
- **Output**: `models/xgb_baseline.json`, `models/cv_results.json`, `submission/pred_{tile_id}.geojson`

### Tier 2: U-TAE (NEXT STEP)

U-Net with Temporal Attention Encoder — SOTA for satellite image time series segmentation.

- **Paper**: Garnot & Landrieu, "Panoptic Segmentation of Satellite Image Time Series" (ICCV 2021)
- **Repo**: github.com/VSainteuf/utae-paps
- **Why**: Captures spatial context (U-Net encoder-decoder) + temporal dynamics (L-TAE attention over monthly observations). XGBoost treats each pixel independently — U-TAE sees spatial patches.
- **Architecture**: Spatial U-Net backbone with Lightweight Temporal Attention Encoder (L-TAE) replacing LSTMs. Uses date-based positional encoding → handles irregular time series natively.
- **Input**: Monthly S2 patches (e.g. 128x128 crops), all 12 bands, up to 72 timesteps. Subsample to 24-36 timesteps if memory is tight (though 206GB VRAM is generous).
- **Multi-modal fusion**: S1 as separate temporal encoder branch at bottleneck. AEF embedding diffs as skip-connection features.
- **Training**: PyTorch + ROCm on MI300X. Use fused consensus labels, weight gold pixels higher.
- **Key considerations**:
  - Use normalized indices (NDVI, NBR) not raw reflectance for cross-region generalization
  - Data augmentation: random spatial flips/rotations, temporal jitter
  - Patch overlap at inference time to avoid edge artifacts
  - Install: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` (already done)

### Tier 3: Ensemble

Weighted average of Tier 1 (XGBoost probabilities) + Tier 2 (U-TAE probabilities). Simple pixel-level averaging or stacking.

### Other Approaches to Consider

- **Presto** (github.com/nasaharvest/presto): Pre-trained pixel-level transformer for S1+S2. Fine-tune on our labels.
- **TSViT** (Tarasiou et al., CVPR 2023): Pure transformer for SITS — alternative to U-TAE.
- **Simple NDVI thresholding**: As a sanity-check baseline — flag pixels where NDVI dropped > 0.3 between 2020 and 2025.

## Hardware

- **GPU**: AMD Instinct MI300X VF, 206GB VRAM
- **ROCm**: 7.0, HIP 7.0
- **PyTorch**: 2.5.1+rocm6.2 (installed, GPU verified)
- **XGBoost**: 3.2.0 (CPU only — pip XGBoost doesn't support ROCm)
- GPU reprojection: pyproj (CPU coord mapping) + torch.grid_sample (GPU resampling)

## Setup Commands

```bash
make install                  # Create venv + install deps
make download_data_from_s3    # Download dataset to data/
source .venv/bin/activate     # Activate environment
```

## Key Libraries

- `rasterio` - reading/writing GeoTIFFs, CRS reprojection
- `geopandas` - vector data (GeoJSON), spatial operations
- `numpy` - array operations
- `shapely` - geometry operations
- `matplotlib` - visualization
