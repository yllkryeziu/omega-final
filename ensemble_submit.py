"""Union ensemble: U-TAE + XGBoost + NDVI change heuristic."""
import json
import numpy as np
import rasterio
from pathlib import Path
from rasterio.warp import reproject, Resampling
from train_xgb import get_s2_ref, DATA_ROOT
from submission_utils import raster_to_geojson

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]
SUBMISSION_DIR = Path("submission")


def compute_ndvi_change(tile_id, ref):
    """Compute NDVI change (late - early) on the S2 grid."""
    import glob
    s2_dir = DATA_ROOT / f"sentinel-2/test/{tile_id}__s2_l2a"
    H, W = ref["shape"]

    def load_ndvi_mean(years):
        arrays = []
        for year in years:
            files = sorted(glob.glob(str(s2_dir / f"{tile_id}__s2_l2a_{year}_*.tif")))
            for f in files:
                try:
                    with rasterio.open(f) as src:
                        nir = src.read(8).astype(np.float32)  # B08
                        red = src.read(4).astype(np.float32)  # B04
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
    return late - early  # negative = vegetation loss


for tile_id in TEST_TILES:
    ref = get_s2_ref(tile_id, split="test")
    if ref is None:
        print(f"{tile_id}: SKIPPED (no S2 ref)")
        continue
    H, W = ref["shape"]

    # 1. U-TAE probabilities
    utae_path = SUBMISSION_DIR / f"prob_{tile_id}.tif"
    with rasterio.open(utae_path) as src:
        utae_prob = src.read(1)

    # 2. XGBoost probabilities (may be different grid size)
    xgb_path = Path(f"submission_xgb/prob_{tile_id}.tif")
    with rasterio.open(xgb_path) as src:
        xgb_raw = src.read(1)
        xgb_crs = src.crs
        xgb_transform = src.transform
        xgb_shape = src.shape

    # Reproject XGB to U-TAE grid if sizes differ
    if xgb_shape != (H, W):
        xgb_prob = np.zeros((H, W), dtype=np.float32)
        reproject(
            source=xgb_raw,
            destination=xgb_prob,
            src_transform=xgb_transform,
            src_crs=xgb_crs,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            resampling=Resampling.bilinear,
        )
    else:
        xgb_prob = xgb_raw

    # 3. NDVI change heuristic
    ndvi_change = compute_ndvi_change(tile_id, ref)
    ndvi_mask = np.zeros((H, W), dtype=bool)
    if ndvi_change is not None:
        ndvi_mask = ndvi_change < -0.15  # NDVI dropped by more than 0.15
        ndvi_pct = 100 * ndvi_mask.sum() / ndvi_mask.size
    else:
        ndvi_pct = 0.0

    # Union: U-TAE > 0.2 OR XGBoost > 0.3 OR NDVI drop > 0.15
    utae_mask = utae_prob > 0.2
    xgb_mask = xgb_prob > 0.3
    union = (utae_mask | xgb_mask | ndvi_mask).astype(np.uint8)

    n_union = union.sum()
    pct = 100 * n_union / union.size
    print(f"{tile_id}: utae@0.2={100*utae_mask.mean():.1f}%  xgb@0.3={100*xgb_mask.mean():.1f}%  "
          f"ndvi={ndvi_pct:.1f}%  union={pct:.1f}%")

    # Save binary raster
    raster_path = SUBMISSION_DIR / f"pred_{tile_id}.tif"
    meta = {
        "driver": "GTiff", "dtype": "uint8", "width": W, "height": H,
        "count": 1, "crs": ref["crs"], "transform": ref["transform"],
        "nodata": 0, "compress": "lzw",
    }
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(union, 1)

    # GeoJSON
    geojson_path = SUBMISSION_DIR / f"pred_{tile_id}.geojson"
    try:
        geojson = raster_to_geojson(str(raster_path), output_path=str(geojson_path), min_area_ha=0.5)
        print(f"  -> {len(geojson['features'])} polygons")
    except ValueError as e:
        print(f"  -> skipped ({e})")

# Merge into single submission
combined = {"type": "FeatureCollection", "features": []}
for f in sorted(SUBMISSION_DIR.glob("pred_*.geojson")):
    with open(f) as fh:
        data = json.load(fh)
    combined["features"].extend(data["features"])
with open(SUBMISSION_DIR / "submission.geojson", "w") as fh:
    json.dump(combined, fh)
print(f"\nSubmission: {len(combined['features'])} total polygons -> submission/submission.geojson")
