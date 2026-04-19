"""
Generate deforestation predictions for test tiles using trained XGBoost model.

For each test tile:
  1. Extract the same features as training (S2, S1, AEF temporal change)
  2. Predict with the saved XGBoost model
  3. Save binary raster (GeoTIFF)
  4. Convert to GeoJSON via submission_utils.raster_to_geojson()

Run: python3 predict_xgb.py
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
import xgboost as xgb

from train_xgb import extract_all_features, get_s2_ref, DATA_ROOT
from submission_utils import raster_to_geojson

MODEL_DIR = Path("models")
SUBMISSION_DIR = Path("submission")
SUBMISSION_XGB_DIR = Path("submission_xgb")
TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]


def predict_tile(tile_id, model, feature_names):
    """Generate predictions for one test tile."""
    print(f"\n{tile_id}:")

    X, fnames, ref = extract_all_features(tile_id, split="test")
    if X is None:
        print(f"  SKIPPED (corrupt S2 data)")
        return None

    assert fnames == feature_names, f"Feature mismatch: {len(fnames)} vs {len(feature_names)}"

    H, W = ref["shape"]

    # Handle NaN: predict 0 for pixels with missing features
    nan_mask = np.any(np.isnan(X), axis=1)
    valid_mask = ~nan_mask

    y_pred = np.zeros(X.shape[0], dtype=np.uint8)
    y_prob = np.zeros(X.shape[0], dtype=np.float32)
    if valid_mask.sum() > 0:
        y_pred[valid_mask] = model.predict(X[valid_mask]).astype(np.uint8)
        y_prob[valid_mask] = model.predict_proba(X[valid_mask])[:, 1]

    pred_2d = y_pred.reshape(H, W)
    prob_2d = y_prob.reshape(H, W)
    n_pos = pred_2d.sum()
    pct = 100 * n_pos / pred_2d.size
    print(f"  Predictions: {n_pos:,} deforested pixels ({pct:.1f}%), {nan_mask.sum():,} NaN skipped")

    # Save probability map for ensemble use
    SUBMISSION_XGB_DIR.mkdir(parents=True, exist_ok=True)
    prob_path = SUBMISSION_XGB_DIR / f"prob_{tile_id}.tif"
    prob_meta = {
        "driver": "GTiff", "dtype": "float32", "width": W, "height": H,
        "count": 1, "crs": ref["crs"], "transform": ref["transform"],
        "nodata": 0, "compress": "lzw",
    }
    with rasterio.open(prob_path, "w", **prob_meta) as dst:
        dst.write(prob_2d, 1)
    print(f"  Probability map: {prob_path}")

    # Save binary raster as GeoTIFF
    raster_path = SUBMISSION_DIR / f"pred_{tile_id}.tif"
    raster_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": W,
        "height": H,
        "count": 1,
        "crs": ref["crs"],
        "transform": ref["transform"],
        "nodata": 0,
        "compress": "lzw",
    }
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(pred_2d, 1)
    print(f"  Raster: {raster_path}")

    # Convert to GeoJSON
    geojson_path = SUBMISSION_DIR / f"pred_{tile_id}.geojson"
    try:
        geojson = raster_to_geojson(str(raster_path), output_path=str(geojson_path), min_area_ha=0.5)
        n_polygons = len(geojson["features"])
        print(f"  GeoJSON: {geojson_path} ({n_polygons} polygons)")
    except ValueError as e:
        print(f"  GeoJSON: skipped ({e})")
        geojson_path = None

    return geojson_path


def main():
    print("Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "xgb_baseline.json"))

    with open(MODEL_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    print(f"Model loaded: {model.n_estimators} trees, {len(feature_names)} features")

    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 60)

    results = {}
    t0 = time.time()
    for tile_id in TEST_TILES:
        geojson_path = predict_tile(tile_id, model, feature_names)
        results[tile_id] = str(geojson_path) if geojson_path else None

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Submission files in {SUBMISSION_DIR}/")

    # Summary
    for tile_id, path in results.items():
        status = path if path else "FAILED"
        print(f"  {tile_id}: {status}")


if __name__ == "__main__":
    main()
