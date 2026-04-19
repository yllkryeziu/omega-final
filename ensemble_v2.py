"""
Advanced ensemble: U-TAE (TTA) + XGBoost + NDVI + S1 change.

Probability fusion with morphological cleanup.
"""
import glob
import json
import time
import numpy as np
import rasterio
import torch
from pathlib import Path
from scipy import ndimage
from rasterio.warp import reproject, Resampling
from models_utae import build_utae
from train_utae import load_tile_s2, load_tile_aef, DEVICE
from train_xgb import get_s2_ref, DATA_ROOT
from submission_utils import raster_to_geojson

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]
SUBMISSION_DIR = Path("submission")


def predict_utae_tta(models, s2_t, positions_t, aef, H, W, ps=128):
    """U-TAE prediction with Test-Time Augmentation (8x: 4 rotations x 2 flips)."""
    rows = list(range(0, H - ps + 1, ps))
    cols = list(range(0, W - ps + 1, ps))
    if rows[-1] + ps < H:
        rows.append(H - ps)
    if cols[-1] + ps < W:
        cols.append(W - ps)

    ensemble_prob = np.zeros((H, W), dtype=np.float32)

    for model in models:
        model.eval()
        pred_sum = np.zeros((H, W), dtype=np.float32)
        pred_count = np.zeros((H, W), dtype=np.float32)

        for r in rows:
            for c in cols:
                s2_patch = s2_t[:, :, :, r:r+ps, c:c+ps]
                aef_patch = None
                if aef is not None:
                    aef_patch = torch.from_numpy(
                        aef[:, r:r+ps, c:c+ps].copy()
                    ).unsqueeze(0).to(DEVICE)

                aug_probs = []
                for rot in range(4):
                    for flip in [False, True]:
                        s2_aug = torch.rot90(s2_patch, rot, dims=[-2, -1])
                        aef_aug = None
                        if aef_patch is not None:
                            aef_aug = torch.rot90(aef_patch, rot, dims=[-2, -1])
                        if flip:
                            s2_aug = torch.flip(s2_aug, dims=[-1])
                            if aef_aug is not None:
                                aef_aug = torch.flip(aef_aug, dims=[-1])

                        logits = model(s2_aug, positions_t, aef_aug)
                        prob = logits.squeeze().sigmoid()

                        # Reverse augmentation
                        if flip:
                            prob = torch.flip(prob, dims=[-1])
                        prob = torch.rot90(prob, -rot, dims=[-2, -1])
                        aug_probs.append(prob.cpu().numpy())

                avg_prob = np.mean(aug_probs, axis=0)
                pred_sum[r:r+ps, c:c+ps] += avg_prob
                pred_count[r:r+ps, c:c+ps] += 1.0

        ensemble_prob += pred_sum / np.maximum(pred_count, 1.0)

    ensemble_prob /= len(models)
    return ensemble_prob


def compute_ndvi_change(tile_id, ref):
    """NDVI change (late - early), returns values in [-1, 1]. Negative = loss."""
    s2_dir = DATA_ROOT / f"sentinel-2/test/{tile_id}__s2_l2a"
    H, W = ref["shape"]

    def load_ndvi_mean(years):
        arrays = []
        for year in years:
            files = sorted(glob.glob(str(s2_dir / f"{tile_id}__s2_l2a_{year}_*.tif")))
            for f in files:
                try:
                    with rasterio.open(f) as src:
                        nir = src.read(8).astype(np.float32)
                        red = src.read(4).astype(np.float32)
                    ndvi = (nir - red) / (nir + red + 1e-10)
                    ndvi[(nir == 0) & (red == 0)] = np.nan
                    if ndvi.shape == (H, W):
                        arrays.append(ndvi)
                except Exception:
                    continue
        if not arrays:
            return None
        return np.nanmean(np.stack(arrays), axis=0)

    early = load_ndvi_mean([2020])
    late = load_ndvi_mean([2024, 2025])
    if early is None or late is None:
        return None
    change = late - early
    return np.nan_to_num(change, nan=0.0)


def compute_s1_change(tile_id, ref):
    """S1 VV backscatter change in dB (late - early). Negative = loss of structure."""
    s1_dir = DATA_ROOT / f"sentinel-1/test/{tile_id}__s1_rtc"
    H, W = ref["shape"]

    def load_vv_mean_db(years):
        arrays = []
        for year in years:
            files = sorted(glob.glob(str(s1_dir / f"{tile_id}__s1_rtc_{year}_*_descending.tif")))
            if not files:
                files = sorted(glob.glob(str(s1_dir / f"{tile_id}__s1_rtc_{year}_*.tif")))
            for f in files:
                try:
                    with rasterio.open(f) as src:
                        vv = src.read(1).astype(np.float32)
                        s1_transform = src.transform
                        s1_crs = src.crs
                        s1_shape = src.shape
                    vv_db = np.where(vv > 0, 10 * np.log10(vv), np.nan)
                    arrays.append((vv_db, s1_transform, s1_crs, s1_shape))
                except Exception:
                    continue
        if not arrays:
            return None, None, None, None
        vv_stack = np.stack([a[0] for a in arrays])
        return np.nanmean(vv_stack, axis=0), arrays[0][1], arrays[0][2], arrays[0][3]

    early_db, s1_t, s1_c, s1_s = load_vv_mean_db([2020])
    late_db, _, _, _ = load_vv_mean_db([2024, 2025])
    if early_db is None or late_db is None:
        return None

    change_db = late_db - early_db  # negative = backscatter dropped

    # Reproject to S2 grid
    change_reproj = np.zeros((H, W), dtype=np.float32)
    reproject(
        source=np.nan_to_num(change_db, nan=0.0),
        destination=change_reproj,
        src_transform=s1_t,
        src_crs=s1_c,
        dst_transform=ref["transform"],
        dst_crs=ref["crs"],
        resampling=Resampling.bilinear,
    )
    return change_reproj


def morphological_cleanup(binary, open_size=3, close_size=2):
    """Remove small noise (opening) and fill small holes (closing)."""
    struct_open = ndimage.generate_binary_structure(2, 1)
    struct_close = ndimage.generate_binary_structure(2, 1)
    cleaned = ndimage.binary_opening(binary, structure=struct_open, iterations=open_size)
    cleaned = ndimage.binary_closing(cleaned, structure=struct_close, iterations=close_size)
    return cleaned.astype(np.uint8)


def main():
    t_start = time.time()

    # Load U-TAE models
    model_paths = sorted(Path("models").glob("utae_fold*_best.pt"))
    models = []
    for mp in model_paths:
        m = build_utae(in_channels=12, aef_channels=66).to(DEVICE)
        m.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=True))
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} U-TAE models")

    combined_geojson = {"type": "FeatureCollection", "features": []}

    for tile_id in TEST_TILES:
        print(f"\n{'='*60}")
        print(f"Tile: {tile_id}")
        t0 = time.time()

        ref = get_s2_ref(tile_id, split="test")
        if ref is None:
            print("  SKIPPED")
            continue
        H, W = ref["shape"]

        # --- 1. U-TAE with TTA ---
        s2, positions = load_tile_s2(tile_id, ref, split="test")
        if s2 is None:
            print("  SKIPPED (S2)")
            continue
        aef = load_tile_aef(tile_id, ref, split="test")

        print(f"  Running U-TAE TTA (8 augmentations x {len(models)} models)...")
        with torch.no_grad():
            positions_t = torch.from_numpy(positions).unsqueeze(0).to(DEVICE)
            s2_t = torch.from_numpy(s2).unsqueeze(0).to(DEVICE)
            utae_prob = predict_utae_tta(models, s2_t, positions_t, aef, H, W)
        print(f"  U-TAE TTA: mean={utae_prob.mean():.4f}, >0.3={100*(utae_prob>0.3).mean():.1f}%")

        # --- 2. XGBoost probabilities ---
        xgb_path = Path(f"submission_xgb/prob_{tile_id}.tif")
        with rasterio.open(xgb_path) as src:
            xgb_raw = src.read(1)
            xgb_t, xgb_c, xgb_s = src.transform, src.crs, src.shape

        if xgb_s != (H, W):
            xgb_prob = np.zeros((H, W), dtype=np.float32)
            reproject(xgb_raw, xgb_prob,
                      src_transform=xgb_t, src_crs=xgb_c,
                      dst_transform=ref["transform"], dst_crs=ref["crs"],
                      resampling=Resampling.bilinear)
        else:
            xgb_prob = xgb_raw
        print(f"  XGBoost: mean={xgb_prob.mean():.4f}, >0.3={100*(xgb_prob>0.3).mean():.1f}%")

        # --- 3. NDVI change signal ---
        ndvi_change = compute_ndvi_change(tile_id, ref)
        if ndvi_change is not None:
            # Convert to soft deforestation signal: large negative NDVI change → high probability
            ndvi_signal = np.clip(-ndvi_change / 0.3, 0, 1).astype(np.float32)
            print(f"  NDVI signal: mean={ndvi_signal.mean():.4f}, >0.3={100*(ndvi_signal>0.3).mean():.1f}%")
        else:
            ndvi_signal = np.zeros((H, W), dtype=np.float32)
            print("  NDVI signal: N/A")

        # --- 4. S1 change signal ---
        s1_change = compute_s1_change(tile_id, ref)
        if s1_change is not None:
            # Negative dB change → deforestation. Convert to soft signal.
            s1_signal = np.clip(-s1_change / 3.0, 0, 1).astype(np.float32)
            print(f"  S1 signal: mean={s1_signal.mean():.4f}, >0.3={100*(s1_signal>0.3).mean():.1f}%")
        else:
            s1_signal = np.zeros((H, W), dtype=np.float32)
            print("  S1 signal: N/A")

        # --- 5. Probability fusion ---
        # Weighted average with model-based signals getting more weight
        fused = (0.40 * utae_prob +
                 0.30 * xgb_prob +
                 0.15 * ndvi_signal +
                 0.15 * s1_signal)

        # Also boost with max approach: if either strong model is confident, trust it
        model_max = np.maximum(utae_prob, xgb_prob)
        # Blend: mostly use fused average but let strong individual predictions through
        final_prob = np.maximum(fused, 0.7 * model_max)

        print(f"  Fused: mean={final_prob.mean():.4f}")

        # --- 6. Binarize + morphological cleanup ---
        threshold = 0.22
        binary = (final_prob > threshold).astype(np.uint8)
        pre_cleanup = binary.sum()

        binary = morphological_cleanup(binary, open_size=1, close_size=1)
        post_cleanup = binary.sum()

        pct = 100 * binary.sum() / binary.size
        print(f"  Binary@{threshold}: {binary.sum():,} px ({pct:.1f}%), "
              f"cleanup removed {pre_cleanup - post_cleanup:,} px")

        # --- 7. Save ---
        raster_path = SUBMISSION_DIR / f"pred_{tile_id}.tif"
        raster_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "driver": "GTiff", "dtype": "uint8", "width": W, "height": H,
            "count": 1, "crs": ref["crs"], "transform": ref["transform"],
            "nodata": 0, "compress": "lzw",
        }
        with rasterio.open(raster_path, "w", **meta) as dst:
            dst.write(binary, 1)

        # Save fused probability map
        prob_path = SUBMISSION_DIR / f"prob_{tile_id}.tif"
        with rasterio.open(prob_path, "w", **{**meta, "dtype": "float32"}) as dst:
            dst.write(final_prob, 1)

        geojson_path = SUBMISSION_DIR / f"pred_{tile_id}.geojson"
        try:
            geojson = raster_to_geojson(str(raster_path), output_path=str(geojson_path), min_area_ha=0.5)
            combined_geojson["features"].extend(geojson["features"])
            print(f"  GeoJSON: {len(geojson['features'])} polygons")
        except ValueError as e:
            print(f"  GeoJSON: skipped ({e})")

        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")

    # Save combined submission
    with open(SUBMISSION_DIR / "submission.geojson", "w") as f:
        json.dump(combined_geojson, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total: {len(combined_geojson['features'])} polygons in {elapsed:.1f}s")
    print(f"Saved to submission/submission.geojson")


if __name__ == "__main__":
    main()
