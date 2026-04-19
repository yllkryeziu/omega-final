"""
Generate deforestation predictions for test tiles using trained U-TAE model(s).

Supports single-model or multi-fold ensemble (averages probabilities).

Run:
  python3 predict_utae.py                                    # ensemble all 4 folds
  python3 predict_utae.py --model models/utae_fold0_best.pt  # single model
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

from models_utae import build_utae
from train_utae import load_tile_s2, load_tile_aef, DEVICE
from train_xgb import get_s2_ref, DATA_ROOT
from submission_utils import raster_to_geojson

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]
SUBMISSION_DIR = Path("submission")


def predict_tile_single(model, s2_t, positions_t, aef, H, W, patch_size):
    """Run one model on a tile, return probability map."""
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)
    ps = patch_size

    rows = list(range(0, H - ps + 1, ps))
    cols = list(range(0, W - ps + 1, ps))
    if rows[-1] + ps < H:
        rows.append(H - ps)
    if cols[-1] + ps < W:
        cols.append(W - ps)

    for r in rows:
        for c in cols:
            s2_patch = s2_t[:, :, :, r:r+ps, c:c+ps]
            aef_patch = None
            if aef is not None:
                aef_patch = torch.from_numpy(aef[:, r:r+ps, c:c+ps].copy()).unsqueeze(0).to(DEVICE)

            logits = model(s2_patch, positions_t, aef_patch)
            probs = logits.squeeze().sigmoid().cpu().numpy()
            pred_sum[r:r+ps, c:c+ps] += probs
            pred_count[r:r+ps, c:c+ps] += 1.0

    return pred_sum / np.maximum(pred_count, 1.0)


def predict_tile(tile_id, models, patch_size=128, use_aef=True, threshold=0.5):
    print(f"\n{tile_id}:")

    ref = get_s2_ref(tile_id, split="test")
    if ref is None:
        print("  SKIPPED (corrupt S2)")
        return None

    s2, positions = load_tile_s2(tile_id, ref, split="test")
    if s2 is None:
        print("  SKIPPED (too few timesteps)")
        return None

    H, W = ref["shape"]
    print(f"  S2: {s2.shape}, grid: {H}x{W}")

    aef = None
    if use_aef:
        aef = load_tile_aef(tile_id, ref, split="test")
        if aef is not None:
            print(f"  AEF: {aef.shape}")

    # Ensemble: average probabilities across all models
    ensemble_prob = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        positions_t = torch.from_numpy(positions).unsqueeze(0).to(DEVICE)
        s2_t = torch.from_numpy(s2).unsqueeze(0).to(DEVICE)

        for i, model in enumerate(models):
            model.eval()
            prob = predict_tile_single(model, s2_t, positions_t, aef, H, W, patch_size)
            ensemble_prob += prob
            print(f"  Model {i}: mean_prob={prob.mean():.4f}, deforest={100*(prob>0.5).mean():.1f}%")

    ensemble_prob /= len(models)

    # Binarize
    pred_binary = (ensemble_prob > threshold).astype(np.uint8)
    n_pos = pred_binary.sum()
    pct = 100 * n_pos / pred_binary.size
    print(f"  Ensemble ({len(models)} models): {n_pos:,} deforested ({pct:.1f}%)")

    # Save binary raster
    raster_path = SUBMISSION_DIR / f"pred_{tile_id}.tif"
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff", "dtype": "uint8", "width": W, "height": H,
        "count": 1, "crs": ref["crs"], "transform": ref["transform"],
        "nodata": 0, "compress": "lzw",
    }
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(pred_binary, 1)
    print(f"  Raster: {raster_path}")

    # Save probability map for visualization / debugging
    prob_path = SUBMISSION_DIR / f"prob_{tile_id}.tif"
    prob_meta = {**meta, "dtype": "float32"}
    with rasterio.open(prob_path, "w", **prob_meta) as dst:
        dst.write(ensemble_prob, 1)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Single model path. If omitted, ensembles all utae_fold*_best.pt")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--no_aef", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    aef_ch = 0 if args.no_aef else 66

    # Find model files
    if args.model:
        model_paths = [Path(args.model)]
    else:
        model_paths = sorted(Path("models").glob("utae_fold*_best.pt"))
        if not model_paths:
            print("No models found in models/utae_fold*_best.pt")
            print("Train first: python3 -u train_utae.py --epochs 50 --batch_size 8")
            return

    # Load all models
    models = []
    for mp in model_paths:
        model = build_utae(in_channels=12, aef_channels=aef_ch, small=args.small).to(DEVICE)
        state = torch.load(mp, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        print(f"Loaded: {mp}")

    n_params = sum(p.numel() for p in models[0].parameters())
    print(f"Ensemble: {len(models)} models, {n_params:,} params each")

    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 60)

    results = {}
    t0 = time.time()
    for tile_id in TEST_TILES:
        geojson_path = predict_tile(tile_id, models, args.patch_size, use_aef=not args.no_aef, threshold=args.threshold)
        results[tile_id] = str(geojson_path) if geojson_path else None

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    for tile_id, path in results.items():
        print(f"  {tile_id}: {path or 'FAILED'}")
    print(f"\nUpload .geojson files from {SUBMISSION_DIR}/")


if __name__ == "__main__":
    main()
