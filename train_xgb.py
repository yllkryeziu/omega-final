"""
Tier 1 Baseline: Temporal feature extraction + XGBoost for deforestation detection.

Uses GPU (AMD MI300X) for raster reprojection via torch.grid_sample.
Uses CPU XGBoost (ROCm XGBoost not available via pip).

Per-pixel features (~80):
  - S2: NDVI, NBR, NDWI temporal change (2020 vs 2024-2025)
  - S1: VV backscatter temporal change
  - AEF: 64-dim embedding diffs + cosine/L2 distance (2020 vs 2025)

Run: python3 train_xgb.py
"""

import json
import glob
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import torch
import torch.nn.functional as F
from pyproj import Transformer
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb

DATA_ROOT = Path("data/makeathon-challenge")
FUSED_ROOT = Path("data/fused-labels")
MODEL_DIR = Path("models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_TILES = [
    "18NWG_6_6", "18NWH_1_4", "18NWJ_8_9", "18NWM_9_4",
    "18NXH_6_8", "18NXJ_7_6", "18NYH_9_9", "19NBD_4_4",
    "47QMB_0_8", "47QQV_2_4", "48PUT_0_8", "48PWV_7_8",
    "48PXC_7_7", "48PYB_3_6", "48QVE_3_0", "48QWD_2_2",
    "AF_MAINDOMBE_01",
]

# S2 band indices (1-based)
S2_B03, S2_B04, S2_B08, S2_B11, S2_B12 = 3, 4, 8, 11, 12


# ---------------------------------------------------------------------------
# GPU reprojection
# ---------------------------------------------------------------------------

def build_reproject_grid(src_path, dst_transform, dst_crs, dst_h, dst_w):
    """Build a normalized sample grid for torch.grid_sample (CPU, ~0.15s)."""
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_h, src_w = src.shape

    cols, rows = np.meshgrid(np.arange(dst_w, dtype=np.float64),
                              np.arange(dst_h, dtype=np.float64))
    dst_xs, dst_ys = dst_transform * (cols + 0.5, rows + 0.5)

    transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    src_xs, src_ys = transformer.transform(dst_xs, dst_ys)

    inv = ~src_transform
    src_cols, src_rows = inv * (src_xs, src_ys)

    grid_x = (2.0 * src_cols / (src_w - 1) - 1.0).astype(np.float32)
    grid_y = (2.0 * src_rows / (src_h - 1) - 1.0).astype(np.float32)
    grid = torch.from_numpy(np.stack([grid_x, grid_y], axis=-1)).unsqueeze(0).to(DEVICE)
    return grid


def gpu_reproject(data_np, grid, mode="bilinear"):
    """Reproject a (C, H, W) numpy array using a prebuilt grid on GPU."""
    data_clean = np.nan_to_num(data_np, nan=0.0).astype(np.float32)
    tensor = torch.from_numpy(data_clean).unsqueeze(0).to(DEVICE)
    result = F.grid_sample(tensor, grid, mode=mode, padding_mode="zeros", align_corners=True)
    return result.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_s2_ref(tile_id, split="train"):
    """Get S2 reference grid metadata. Filters out corrupt files (any dim < 100).
    Returns None if no valid files exist."""
    s2_dir = DATA_ROOT / f"sentinel-2/{split}/{tile_id}__s2_l2a"
    all_files = sorted(glob.glob(str(s2_dir / "*.tif")))
    shapes = Counter()
    metas = {}
    for f in all_files:
        with rasterio.open(f) as src:
            s = src.shape
            if s[0] < 100 or s[1] < 100:
                continue
            shapes[s] += 1
            if s not in metas:
                metas[s] = {"shape": s, "transform": src.transform, "crs": src.crs}
    if not shapes:
        return None
    dominant = shapes.most_common(1)[0][0]
    return metas[dominant]


def extract_s2_features(tile_id, split, ref):
    """Extract S2 vegetation index temporal change features."""
    s2_dir = DATA_ROOT / f"sentinel-2/{split}/{tile_id}__s2_l2a"
    expected = ref["shape"]
    H, W = expected

    def read_indices(year_months):
        ndvi_sum, nbr_sum, ndwi_sum = np.zeros((H, W), np.float64), np.zeros((H, W), np.float64), np.zeros((H, W), np.float64)
        count = np.zeros((H, W), np.float64)
        for year, month in year_months:
            path = s2_dir / f"{tile_id}__s2_l2a_{year}_{month}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                if src.shape != expected:
                    continue
                b03 = src.read(S2_B03).astype(np.float32)
                b04 = src.read(S2_B04).astype(np.float32)
                b08 = src.read(S2_B08).astype(np.float32)
                b12 = src.read(S2_B12).astype(np.float32)
            valid = (b04 > 0) | (b08 > 0)
            eps = 1e-6
            ndvi = (b08 - b04) / (b08 + b04 + eps)
            nbr = (b08 - b12) / (b08 + b12 + eps)
            ndwi = (b03 - b08) / (b03 + b08 + eps)
            ndvi_sum[valid] += ndvi[valid]
            nbr_sum[valid] += nbr[valid]
            ndwi_sum[valid] += ndwi[valid]
            count[valid] += 1
        mask = count > 0
        ndvi_mean = np.full((H, W), np.nan, np.float32)
        nbr_mean = np.full((H, W), np.nan, np.float32)
        ndwi_mean = np.full((H, W), np.nan, np.float32)
        ndvi_mean[mask] = (ndvi_sum[mask] / count[mask]).astype(np.float32)
        nbr_mean[mask] = (nbr_sum[mask] / count[mask]).astype(np.float32)
        ndwi_mean[mask] = (ndwi_sum[mask] / count[mask]).astype(np.float32)
        return ndvi_mean, nbr_mean, ndwi_mean

    early_ym = [(2020, m) for m in range(1, 13)]
    late_ym = [(y, m) for y in [2024, 2025] for m in range(1, 13)]

    ndvi_e, nbr_e, ndwi_e = read_indices(early_ym)
    ndvi_l, nbr_l, ndwi_l = read_indices(late_ym)

    return {
        "ndvi_early": ndvi_e, "ndvi_late": ndvi_l, "ndvi_diff": ndvi_l - ndvi_e,
        "nbr_early": nbr_e, "nbr_late": nbr_l, "nbr_diff": nbr_l - nbr_e,
        "ndwi_early": ndwi_e, "ndwi_late": ndwi_l, "ndwi_diff": ndwi_l - ndwi_e,
    }


def extract_s1_features(tile_id, split, ref):
    """Extract S1 VV backscatter temporal change, reprojected to S2 grid via GPU."""
    s1_dir = DATA_ROOT / f"sentinel-1/{split}/{tile_id}__s1_rtc"
    ref_shape = ref["shape"]

    # Find dominant S1 shape
    all_files = sorted(glob.glob(str(s1_dir / "*.tif")))[:24]
    if not all_files:
        nan = np.full(ref_shape, np.nan, np.float32)
        return {"vv_db_early": nan, "vv_db_late": nan, "vv_db_diff": nan}

    shapes = Counter()
    for f in all_files:
        with rasterio.open(f) as src:
            shapes[src.shape] += 1
    dominant = shapes.most_common(1)[0][0]

    # Get S1 metadata for grid building
    s1_meta = None
    for f in all_files:
        with rasterio.open(f) as src:
            if src.shape == dominant:
                s1_meta = {"transform": src.transform, "crs": src.crs}
                break

    # Accumulate VV on S1 native grid
    def accum_vv(year_months):
        vv_sum = np.zeros(dominant, np.float64)
        count = np.zeros(dominant, np.float64)
        for year, month in year_months:
            for orbit in ["descending", "ascending"]:
                path = s1_dir / f"{tile_id}__s1_rtc_{year}_{month}_{orbit}.tif"
                if not path.exists():
                    continue
                with rasterio.open(path) as src:
                    if src.shape != dominant:
                        continue
                    vv = src.read(1).astype(np.float32)
                valid = np.isfinite(vv) & (vv > 0)
                vv_sum[valid] += vv[valid]
                count[valid] += 1
        mask = count > 0
        result = np.full(dominant, np.nan, np.float32)
        result[mask] = (vv_sum[mask] / count[mask]).astype(np.float32)
        return result

    early_ym = [(2020, m) for m in range(1, 13)]
    late_ym = [(y, m) for y in [2024, 2025] for m in range(1, 13)]
    vv_early = accum_vv(early_ym)
    vv_late = accum_vv(late_ym)

    # Convert to dB
    with np.errstate(all='ignore'):
        db_early = np.where(vv_early > 0, 10 * np.log10(vv_early), np.nan).astype(np.float32)
        db_late = np.where(vv_late > 0, 10 * np.log10(vv_late), np.nan).astype(np.float32)

    # Stack and GPU-reproject to S2 grid in one call
    stack = np.stack([db_early, db_late, db_late - db_early], axis=0)  # (3, h, w)

    # Build a temp file path reference for grid building — or build grid from metadata
    grid = _build_grid_from_meta(s1_meta["transform"], s1_meta["crs"], dominant,
                                  ref["transform"], ref["crs"], ref_shape)
    reprojected = gpu_reproject(stack, grid)  # (3, H, W)

    return {
        "vv_db_early": reprojected[0],
        "vv_db_late": reprojected[1],
        "vv_db_diff": reprojected[2],
    }


def _build_grid_from_meta(src_transform, src_crs, src_shape, dst_transform, dst_crs, dst_shape):
    """Build torch grid_sample grid from rasterio metadata (no file needed)."""
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape

    cols, rows = np.meshgrid(np.arange(dst_w, dtype=np.float64),
                              np.arange(dst_h, dtype=np.float64))
    dst_xs, dst_ys = dst_transform * (cols + 0.5, rows + 0.5)

    transformer = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
    src_xs, src_ys = transformer.transform(dst_xs, dst_ys)

    inv = ~src_transform
    src_cols, src_rows = inv * (src_xs, src_ys)

    grid_x = (2.0 * src_cols / (src_w - 1) - 1.0).astype(np.float32)
    grid_y = (2.0 * src_rows / (src_h - 1) - 1.0).astype(np.float32)
    return torch.from_numpy(np.stack([grid_x, grid_y], axis=-1)).unsqueeze(0).to(DEVICE)


def extract_aef_features(tile_id, split, ref):
    """Extract AEF embedding change features, GPU-reprojected to S2 grid."""
    aef_2020_path = DATA_ROOT / f"aef-embeddings/{split}/{tile_id}_2020.tiff"
    aef_2025_path = DATA_ROOT / f"aef-embeddings/{split}/{tile_id}_2025.tiff"

    if not aef_2020_path.exists() or not aef_2025_path.exists():
        H, W = ref["shape"]
        features = {}
        for i in range(64):
            features[f"aef_diff_{i:02d}"] = np.full((H, W), np.nan, dtype=np.float32)
        features["aef_cosine_dist"] = np.full((H, W), np.nan, dtype=np.float32)
        features["aef_l2_dist"] = np.full((H, W), np.nan, dtype=np.float32)
        return features

    with rasterio.open(aef_2020_path) as src:
        emb_2020 = src.read().astype(np.float32)
        aef_meta = {"transform": src.transform, "crs": src.crs, "shape": src.shape}

    with rasterio.open(aef_2025_path) as src:
        emb_2025 = src.read().astype(np.float32)

    # Compute features on AEF native grid (fast, avoids reprojecting raw 128 bands)
    diff = emb_2025 - emb_2020  # (64, H, W)

    dot = np.nansum(emb_2020 * emb_2025, axis=0)
    norm_2020 = np.sqrt(np.nansum(emb_2020 ** 2, axis=0)) + 1e-8
    norm_2025 = np.sqrt(np.nansum(emb_2025 ** 2, axis=0)) + 1e-8
    cosine_dist = (1.0 - dot / (norm_2020 * norm_2025)).astype(np.float32)
    l2_dist = np.sqrt(np.nansum(diff ** 2, axis=0)).astype(np.float32)

    # Stack all 66 features and reproject in ONE gpu call
    stack = np.concatenate([diff, cosine_dist[np.newaxis], l2_dist[np.newaxis]], axis=0)  # (66, H, W)

    grid = _build_grid_from_meta(aef_meta["transform"], aef_meta["crs"], aef_meta["shape"],
                                  ref["transform"], ref["crs"], ref["shape"])
    reprojected = gpu_reproject(stack, grid)  # (66, H, W)

    features = {}
    for i in range(64):
        features[f"aef_diff_{i:02d}"] = reprojected[i]
    features["aef_cosine_dist"] = reprojected[64]
    features["aef_l2_dist"] = reprojected[65]
    return features


def extract_all_features(tile_id, split="train"):
    """Extract and align all features on S2 grid. Returns (N_pixels, N_features)."""
    t0 = time.time()
    ref = get_s2_ref(tile_id, split)
    if ref is None:
        return None, None, None

    s2_feats = extract_s2_features(tile_id, split, ref)
    s1_feats = extract_s1_features(tile_id, split, ref)
    aef_feats = extract_aef_features(tile_id, split, ref)

    all_feats = {}
    all_feats.update(s2_feats)
    all_feats.update(s1_feats)
    all_feats.update(aef_feats)

    feature_names = sorted(all_feats.keys())
    H, W = ref["shape"]
    X = np.stack([all_feats[k] for k in feature_names], axis=-1).reshape(-1, len(feature_names))

    print(f"  {len(feature_names)} features, {H*W:,} pixels, {time.time()-t0:.1f}s")
    return X, feature_names, ref


def load_labels_on_s2_grid(tile_id, ref):
    """Load fused labels + consensus mask, reprojected to S2 grid."""
    results = {}
    for name, filename in [("labels", "fused_binary.tif"), ("consensus", "consensus_mask.tif")]:
        path = FUSED_ROOT / tile_id / filename
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            dst = np.zeros(ref["shape"], dtype=np.float32)
            reproject(data, dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref["transform"], dst_crs=ref["crs"],
                      resampling=Resampling.nearest, src_nodata=255, dst_nodata=0)
        results[name] = dst.astype(np.uint8).ravel()
    return results["labels"], results["consensus"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(FUSED_ROOT / "dataset_meta.json") as f:
        meta = json.load(f)
    folds = meta["cv_folds"]

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)

    tile_data = {}
    skipped = []
    feature_names = None
    for tile_id in TRAIN_TILES:
        print(f"\n{tile_id}:")
        X, fnames, ref = extract_all_features(tile_id)
        if X is None:
            print(f"  SKIPPED (corrupt S2 data)")
            skipped.append(tile_id)
            continue
        y, consensus = load_labels_on_s2_grid(tile_id, ref)
        tile_data[tile_id] = {"X": X, "y": y, "consensus": consensus}
        if feature_names is None:
            feature_names = fnames
        gold = (consensus == 1) | (consensus == 4)
        print(f"  Labels: {y.sum():,} pos / {(y==0).sum():,} neg | gold: {gold.sum():,}")
    if skipped:
        print(f"\nSkipped tiles (corrupt): {skipped}")

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)

    fold_results = []
    for fold_info in folds:
        fold_idx = fold_info["fold"]
        val_tiles = fold_info["val"]
        train_tiles = fold_info["train"]
        print(f"\n--- Fold {fold_idx} (val: {val_tiles}) ---")

        avail_train = [t for t in train_tiles if t in tile_data]
        X_train = np.concatenate([tile_data[t]["X"] for t in avail_train])
        y_train = np.concatenate([tile_data[t]["y"] for t in avail_train])
        valid = ~np.any(np.isnan(X_train), axis=1)
        X_train, y_train = X_train[valid], y_train[valid]

        pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        print(f"  Train: {len(y_train):,} px, pos_weight: {pos_w:.1f}")

        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=pos_w, reg_alpha=0.1, reg_lambda=1.0,
            device="cpu", tree_method="hist", eval_metric="logloss", random_state=42,
        )
        t0 = time.time()
        model.fit(X_train, y_train, verbose=10)
        print(f"  XGBoost trained in {time.time()-t0:.1f}s")
        del X_train, y_train

        for vt in val_tiles:
            if vt not in tile_data:
                print(f"    {vt}: skip (corrupt)")
                continue
            X_v, y_v, c = tile_data[vt]["X"], tile_data[vt]["y"], tile_data[vt]["consensus"]
            gold = (c == 1) | (c == 4)
            valid = gold & ~np.any(np.isnan(X_v), axis=1)
            if valid.sum() < 100:
                print(f"    {vt}: skip ({valid.sum()} gold px)")
                continue
            y_pred = model.predict(X_v[valid])
            y_true = y_v[valid]
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            print(f"    {vt}: F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  (pos={y_true.sum():,})")
            fold_results.append({"fold": fold_idx, "tile": vt, "f1": float(f1),
                                  "precision": float(prec), "recall": float(rec)})

    # Summary
    print("\n" + "=" * 60)
    f1s = [r["f1"] for r in fold_results]
    print(f"MEAN F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"MEAN P:   {np.mean([r['precision'] for r in fold_results]):.4f}")
    print(f"MEAN R:   {np.mean([r['recall'] for r in fold_results]):.4f}")
    per_tile = [f'{r["tile"]}={r["f1"]:.3f}' for r in fold_results]
    print(f"Per-tile: {per_tile}")

    # Final model on all data
    print("\n" + "=" * 60)
    print("FINAL MODEL")
    avail_all = [t for t in TRAIN_TILES if t in tile_data]
    X_all = np.concatenate([tile_data[t]["X"] for t in avail_all])
    y_all = np.concatenate([tile_data[t]["y"] for t in avail_all])
    valid = ~np.any(np.isnan(X_all), axis=1)
    X_all, y_all = X_all[valid], y_all[valid]
    pos_w = (y_all == 0).sum() / max((y_all == 1).sum(), 1)

    final = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w,
        device="cpu", tree_method="hist", eval_metric="logloss", random_state=42,
    )
    t0 = time.time()
    final.fit(X_all, y_all, verbose=10)
    print(f"Trained on {len(y_all):,} pixels in {time.time()-t0:.1f}s")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final.save_model(str(MODEL_DIR / "xgb_baseline.json"))
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open(MODEL_DIR / "cv_results.json", "w") as f:
        json.dump({"fold_results": fold_results, "mean_f1": float(np.mean(f1s))}, f, indent=2)
    print(f"Saved to {MODEL_DIR}/")

    imp = final.feature_importances_
    top = np.argsort(imp)[::-1][:15]
    print("\nTop 15 features:")
    for i in top:
        print(f"  {feature_names[i]:25s} {imp[i]:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
