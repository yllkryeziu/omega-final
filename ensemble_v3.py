"""
Ensemble v3: Aggressive union with more signals to maximize recall.

Strategy: v1 union worked best (52.71%). Build on it by adding S1 and NBR
change signals, lowering thresholds, and using morphological closing to
fill gaps (NOT opening which removes true positives).

Signals:
  1. U-TAE probabilities (TTA from v2 saved as prob maps)
  2. XGBoost probabilities
  3. NDVI change (vegetation loss)
  4. NBR change (burn/clearing detection)
  5. S1 VV change (radar backscatter drop)

Union: any signal fires → positive. Then morphological closing to fill gaps.
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
from train_xgb import get_s2_ref, DATA_ROOT
from submission_utils import raster_to_geojson

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]
SUBMISSION_DIR = Path("submission")
NDVI_DROP_THRESHOLD = 0.15


def load_prob_map(path, ref, H, W):
    """Load a probability raster and reproject to ref grid if needed."""
    with rasterio.open(path) as src:
        prob = src.read(1)
        if src.shape != (H, W):
            reproj = np.zeros((H, W), dtype=np.float32)
            reproject(prob, reproj,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref["transform"], dst_crs=ref["crs"],
                      resampling=Resampling.bilinear)
            return reproj
        return prob


def compute_spectral_change(tile_id, ref, early_years, late_years, compute_fn):
    """Generic function to compute spectral index change between early and late periods."""
    s2_dir = DATA_ROOT / f"sentinel-2/test/{tile_id}__s2_l2a"
    H, W = ref["shape"]

    def load_mean(years):
        arrays = []
        for year in years:
            files = sorted(glob.glob(str(s2_dir / f"{tile_id}__s2_l2a_{year}_*.tif")))
            for f in files:
                try:
                    with rasterio.open(f) as src:
                        bands = src.read().astype(np.float32)
                    idx = compute_fn(bands)
                    if idx.shape == (H, W):
                        arrays.append(idx)
                except Exception:
                    continue
        if not arrays:
            return None
        return np.nanmean(np.stack(arrays), axis=0)

    early = load_mean(early_years)
    late = load_mean(late_years)
    if early is None or late is None:
        return None
    return np.nan_to_num(late - early, nan=0.0)


def ndvi_fn(bands):
    """NDVI from S2 bands array (12, H, W). B08=band[7], B04=band[3]."""
    nir, red = bands[7], bands[3]
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi[(nir == 0) & (red == 0)] = np.nan
    return ndvi


def nbr_fn(bands):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). B08=band[7], B12=band[11]."""
    nir, swir2 = bands[7], bands[11]
    nbr = (nir - swir2) / (nir + swir2 + 1e-10)
    nbr[(nir == 0) & (swir2 == 0)] = np.nan
    return nbr


def compute_time_of_change(tile_id, ref):
    """Estimate when deforestation occurred per pixel. Returns YYMM raster.

    Requires sustained NDVI drop: 2+ consecutive observations below baseline - threshold.
    This filters out transient cloud shadows and seasonal variation.
    The recorded date is the FIRST month of the sustained drop.
    """
    s2_dir = DATA_ROOT / f"sentinel-2/test/{tile_id}__s2_l2a"
    H, W = ref["shape"]

    baseline_arrays = []
    for f in sorted(glob.glob(str(s2_dir / f"{tile_id}__s2_l2a_2020_*.tif"))):
        try:
            with rasterio.open(f) as src:
                bands = src.read().astype(np.float32)
            ndvi = ndvi_fn(bands)
            if ndvi.shape == (H, W):
                baseline_arrays.append(ndvi)
        except Exception:
            continue
    if not baseline_arrays:
        return None

    baseline = np.nanmean(np.stack(baseline_arrays), axis=0)
    time_raster = np.zeros((H, W), dtype=np.int16)
    prev_dropped = np.zeros((H, W), dtype=bool)
    prev_yymm = np.zeros((H, W), dtype=np.int16)

    for year in [2021, 2022, 2023, 2024, 2025]:
        yy = year - 2000
        for month in range(1, 13):
            fname = s2_dir / f"{tile_id}__s2_l2a_{year}_{month:02d}.tif"
            if not fname.exists():
                continue
            try:
                with rasterio.open(fname) as src:
                    bands = src.read().astype(np.float32)
                ndvi = ndvi_fn(bands)
                if ndvi.shape != (H, W):
                    continue
            except Exception:
                continue

            drop = baseline - ndvi
            currently_dropped = drop > NDVI_DROP_THRESHOLD

            sustained = currently_dropped & prev_dropped & (time_raster == 0)
            time_raster[sustained] = prev_yymm[sustained]

            prev_yymm[currently_dropped & ~prev_dropped] = (yy * 100 + month)
            prev_dropped = currently_dropped

    return time_raster


def compute_s1_change(tile_id, ref):
    """S1 VV backscatter change in dB. Negative = structure loss."""
    s1_dir = DATA_ROOT / f"sentinel-1/test/{tile_id}__s1_rtc"
    H, W = ref["shape"]

    def load_vv_db(years):
        arrays = []
        meta = None
        for year in years:
            files = sorted(glob.glob(str(s1_dir / f"{tile_id}__s1_rtc_{year}_*.tif")))
            for f in files:
                try:
                    with rasterio.open(f) as src:
                        vv = src.read(1).astype(np.float32)
                        if meta is None:
                            meta = (src.transform, src.crs, src.shape)
                    vv_db = np.where(vv > 0, 10 * np.log10(vv), np.nan)
                    arrays.append(vv_db)
                except Exception:
                    continue
        if not arrays or meta is None:
            return None, None
        return np.nanmean(np.stack(arrays), axis=0), meta

    early_db, meta_e = load_vv_db([2020])
    late_db, _ = load_vv_db([2024, 2025])
    if early_db is None or late_db is None:
        return None

    change_db = late_db - early_db
    reproj = np.zeros((H, W), dtype=np.float32)
    reproject(
        source=np.nan_to_num(change_db, nan=0.0),
        destination=reproj,
        src_transform=meta_e[0], src_crs=meta_e[1],
        dst_transform=ref["transform"], dst_crs=ref["crs"],
        resampling=Resampling.bilinear,
    )
    return reproj


def main():
    t_start = time.time()
    combined_geojson = {"type": "FeatureCollection", "features": []}

    for tile_id in TEST_TILES:
        print(f"\n{'='*60}")
        print(f"Tile: {tile_id}")

        ref = get_s2_ref(tile_id, split="test")
        if ref is None:
            print("  SKIPPED")
            continue
        H, W = ref["shape"]

        # --- Signal 1: U-TAE ---
        utae_prob = load_prob_map(SUBMISSION_DIR / f"prob_{tile_id}.tif", ref, H, W)
        utae_mask = utae_prob > 0.10
        print(f"  U-TAE@0.10: {100*utae_mask.mean():.1f}%")

        # --- Signal 2: XGBoost ---
        xgb_prob = load_prob_map(f"submission_xgb/prob_{tile_id}.tif", ref, H, W)
        xgb_mask = xgb_prob > 0.25
        print(f"  XGBoost@0.25: {100*xgb_mask.mean():.1f}%")

        # --- Signal 3: NDVI change ---
        ndvi_change = compute_spectral_change(tile_id, ref, [2020], [2024, 2025], ndvi_fn)
        if ndvi_change is not None:
            ndvi_mask = ndvi_change < -0.12
            print(f"  NDVI drop>0.12: {100*ndvi_mask.mean():.1f}%")
        else:
            ndvi_mask = np.zeros((H, W), dtype=bool)
            print("  NDVI: N/A")

        # --- Signal 4: NBR change ---
        nbr_change = compute_spectral_change(tile_id, ref, [2020], [2024, 2025], nbr_fn)
        if nbr_change is not None:
            nbr_mask = nbr_change < -0.10
            print(f"  NBR drop>0.10: {100*nbr_mask.mean():.1f}%")
        else:
            nbr_mask = np.zeros((H, W), dtype=bool)
            print("  NBR: N/A")

        # --- Signal 5: S1 VV change ---
        s1_change = compute_s1_change(tile_id, ref)
        if s1_change is not None:
            s1_mask = s1_change < -2.0  # >2 dB drop
            print(f"  S1 VV drop>2dB: {100*s1_mask.mean():.1f}%")
        else:
            s1_mask = np.zeros((H, W), dtype=bool)
            print("  S1: N/A")

        # --- Union ---
        # ML models are primary, spectral indices are supporting
        # Require spectral signals to agree with at least one ML model OR both indices
        ml_union = utae_mask | xgb_mask
        spectral_union = ndvi_mask | nbr_mask
        s1_strong = s1_mask & (s1_change < -3.0 if s1_change is not None else np.zeros((H,W), dtype=bool))

        # Final: ML model fires, OR (spectral + S1 agree), OR (both spectral indices agree)
        union = ml_union | (spectral_union & s1_mask) | (ndvi_mask & nbr_mask)

        n_union_pre = union.sum()
        print(f"  Union (pre-cleanup): {100*union.mean():.1f}%")

        # Morphological closing to fill small gaps in detected areas
        union = ndimage.binary_closing(union, iterations=1).astype(np.uint8)

        pct = 100 * union.sum() / union.size
        print(f"  Final: {union.sum():,} px ({pct:.1f}%)")

        # --- Time of change estimation ---
        time_raster = compute_time_of_change(tile_id, ref)
        if time_raster is not None:
            detected_times = time_raster[union == 1]
            detected_times = detected_times[detected_times > 0]
            if len(detected_times) > 0:
                counts = np.bincount(detected_times)
                mode_yymm = counts.argmax()
                print(f"  Time of change: mode={mode_yymm // 100 + 2000}-{mode_yymm % 100:02d}, "
                      f"coverage={100 * (detected_times > 0).sum() / max(1, (union == 1).sum()):.0f}%")
            else:
                print("  Time of change: no NDVI-based dates found")
        else:
            print("  Time of change: N/A (no S2 baseline)")

        # Save raster
        raster_path = SUBMISSION_DIR / f"pred_{tile_id}.tif"
        meta = {
            "driver": "GTiff", "dtype": "uint8", "width": W, "height": H,
            "count": 1, "crs": ref["crs"], "transform": ref["transform"],
            "nodata": 0, "compress": "lzw",
        }
        with rasterio.open(raster_path, "w", **meta) as dst:
            dst.write(union, 1)

        # GeoJSON with time_step
        geojson_path = SUBMISSION_DIR / f"pred_{tile_id}.geojson"
        try:
            geojson = raster_to_geojson(
                str(raster_path), output_path=str(geojson_path),
                min_area_ha=0.25,
                time_step_raster=time_raster,
                time_step_transform=ref["transform"],
            )
            combined_geojson["features"].extend(geojson["features"])
            print(f"  GeoJSON: {len(geojson['features'])} polygons")
        except ValueError as e:
            print(f"  GeoJSON: skipped ({e})")

    # Save combined submission
    with open(SUBMISSION_DIR / "submission.geojson", "w") as f:
        json.dump(combined_geojson, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total: {len(combined_geojson['features'])} polygons in {elapsed:.1f}s")
    print(f"Saved to submission/submission.geojson")


if __name__ == "__main__":
    main()
