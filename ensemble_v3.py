"""
Ensemble v3 — Final submission.

Changes from #18 (54.27%):
  1. U-TAE threshold 0.15 → 0.12 (recall boost)
  2. Remove (ndvi & nbr) pathway — no ML/radar confirmation, caused FPR on 48PWA
  3. Keep closing + restore opening (proven in best submission)
  4. Robust year estimation: yearly NDVI means (cloud-robust), first year with
     significant drop, selective assignment (null when uncertain)
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


def load_prob_map(path, ref, H, W):
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
    nir, red = bands[7], bands[3]
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi[(nir == 0) & (red == 0)] = np.nan
    return ndvi


def nbr_fn(bands):
    nir, swir2 = bands[7], bands[11]
    nbr = (nir - swir2) / (nir + swir2 + 1e-10)
    nbr[(nir == 0) & (swir2 == 0)] = np.nan
    return nbr


def compute_time_of_change(tile_id, ref):
    """Yearly NDVI analysis for robust change dating.

    Uses annual mean NDVI (averages out cloud noise) to find the first year
    with a clear vegetation drop from the 2020 baseline. Only assigns a date
    when the drop is unambiguous (>0.15). Returns YYMM raster with YY01 encoding.
    """
    s2_dir = DATA_ROOT / f"sentinel-2/test/{tile_id}__s2_l2a"
    H, W = ref["shape"]

    yearly_ndvi = {}
    for year in range(2020, 2026):
        arrays = []
        for f in sorted(glob.glob(str(s2_dir / f"{tile_id}__s2_l2a_{year}_*.tif"))):
            try:
                with rasterio.open(f) as src:
                    bands = src.read().astype(np.float32)
                ndvi = ndvi_fn(bands)
                if ndvi.shape == (H, W):
                    arrays.append(ndvi)
            except Exception:
                continue
        if arrays:
            yearly_ndvi[year] = np.nanmean(np.stack(arrays), axis=0)

    if 2020 not in yearly_ndvi:
        return None

    baseline = yearly_ndvi[2020]
    time_raster = np.zeros((H, W), dtype=np.int16)

    for year in [2021, 2022, 2023, 2024, 2025]:
        if year not in yearly_ndvi:
            continue
        drop = baseline - yearly_ndvi[year]
        newly_detected = (drop > 0.15) & (time_raster == 0)
        if not newly_detected.any():
            continue
        yy = year - 2000
        time_raster[newly_detected] = yy * 100 + 1

    return time_raster


def compute_s1_change(tile_id, ref):
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

        # --- Signal 1: U-TAE (lowered from 0.15 to 0.12 for recall) ---
        utae_prob = load_prob_map(SUBMISSION_DIR / f"prob_{tile_id}.tif", ref, H, W)
        utae_mask = utae_prob > 0.12
        print(f"  U-TAE@0.12: {100*utae_mask.mean():.1f}%")

        # --- Signal 2: XGBoost (keep 0.25 — proven clean separation) ---
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
            s1_mask = s1_change < -2.0
            print(f"  S1 VV drop>2dB: {100*s1_mask.mean():.1f}%")
        else:
            s1_mask = np.zeros((H, W), dtype=bool)
            print("  S1: N/A")

        # --- Gated Union ---
        # ML models trusted alone (primary detectors)
        ml_union = utae_mask | xgb_mask

        # Spectral only fires when confirmed by S1 radar (removes cloud/seasonal FP)
        # REMOVED: (ndvi_mask & nbr_mask) pathway — no ML/radar confirmation,
        # caused 26.9% NBR FP on 48PWA_0_6
        spectral_confirmed = (ndvi_mask | nbr_mask) & s1_mask

        union = ml_union | spectral_confirmed

        print(f"  Union (pre-cleanup): {100*union.mean():.1f}%")

        # Morphological closing fills small gaps within deforested areas
        union = ndimage.binary_closing(union, iterations=1).astype(np.uint8)

        # Opening removes truly isolated single pixels (10m = never real clearing)
        union = ndimage.binary_opening(union, iterations=1).astype(np.uint8)

        pct = 100 * union.sum() / union.size
        print(f"  Final: {union.sum():,} px ({pct:.1f}%)")

        # --- Time of change (yearly NDVI, robust to clouds) ---
        time_raster = compute_time_of_change(tile_id, ref)
        if time_raster is not None:
            detected_times = time_raster[union == 1]
            detected_times = detected_times[detected_times > 0]
            if len(detected_times) > 0:
                counts = np.bincount(detected_times)
                mode_yymm = counts.argmax()
                print(f"  Time: mode={mode_yymm // 100 + 2000}-{mode_yymm % 100:02d}, "
                      f"coverage={100 * len(detected_times) / max(1, (union == 1).sum()):.0f}%")
            else:
                print("  Time: no dates found")
        else:
            print("  Time: N/A")

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
