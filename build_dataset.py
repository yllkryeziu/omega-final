"""
Build fused label dataset from 3 weak label sources (RADD, GLAD-L, GLAD-S2).

For each train tile, produces:
  - fused_binary.tif:    binary deforestation label (0/1) from majority vote
  - consensus_mask.tif:  evaluation tier per pixel:
                           0 = uncertain (excluded from eval)
                           1 = gold negative (all sources agree: no deforestation)
                           2 = silver negative (majority says no)
                           3 = silver positive (majority says yes)
                           4 = gold positive (all sources agree: deforestation, high confidence)
  - source_radd.tif:     binarised RADD (post-2020, any confidence)
  - source_gladl.tif:    binarised GLAD-L (union of all years, confirmed only)
  - source_glads2.tif:   binarised GLAD-S2 (confidence >= 2)

All outputs are on the RADD/GLAD-S2 native grid (EPSG:4326, ~10m).
GLAD-L is upsampled from its coarser grid via nearest-neighbour.

Also writes metadata/cv_splits.json with region-stratified tile-level folds.
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

DATA_ROOT = Path("data/makeathon-challenge")
OUT_ROOT = Path("data/fused-labels")

TRAIN_TILES = [
    "18NWG_6_6", "18NWH_1_4", "18NWJ_8_9", "18NWM_9_4",
    "18NXH_6_8", "18NXJ_7_6", "18NYH_9_9", "19NBD_4_4",
    "47QMB_0_8", "47QQV_2_4", "48PUT_0_8", "48PWV_7_8",
    "48PXC_7_7", "48PYB_3_6", "48QVE_3_0", "48QWD_2_2",
]

SOUTH_AMERICA = [t for t in TRAIN_TILES if t.startswith("18") or t.startswith("19")]
SOUTHEAST_ASIA = [t for t in TRAIN_TILES if t.startswith("47") or t.startswith("48")]

# Days from 2014-12-31 to 2020-12-31
RADD_POST2020_THRESHOLD = (date(2020, 12, 31) - date(2014, 12, 31)).days  # 2192


def binarise_radd(tile_id: str, ref_shape=None, ref_transform=None, ref_crs=None):
    """Binarise RADD: post-2020, any confidence -> 1."""
    path = DATA_ROOT / f"labels/train/radd/radd_{tile_id}_labels.tif"
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {"shape": src.shape, "transform": src.transform, "crs": src.crs}

    mask = data > 0
    days = data % 10000
    post2020 = days > RADD_POST2020_THRESHOLD
    binary = (mask & post2020).astype(np.uint8)
    return binary, meta


def binarise_radd_high_conf(tile_id: str):
    """Binarise RADD: post-2020 AND high confidence (leading digit 3) -> 1."""
    path = DATA_ROOT / f"labels/train/radd/radd_{tile_id}_labels.tif"
    with rasterio.open(path) as src:
        data = src.read(1)

    mask = data > 0
    leading = data // 10000
    days = data % 10000
    post2020 = days > RADD_POST2020_THRESHOLD
    high_conf = leading == 3
    return (mask & post2020 & high_conf).astype(np.uint8)


def binarise_gladl(tile_id: str, ref_shape, ref_transform, ref_crs):
    """Binarise GLAD-L: confirmed loss (value=3) in any year -> 1. Upsample to ref grid."""
    combined = None
    src_meta = None
    for yy in [21, 22, 23, 24, 25]:
        path = DATA_ROOT / f"labels/train/gladl/gladl_{tile_id}_alert{yy}.tif"
        with rasterio.open(path) as src:
            data = src.read(1)
            if src_meta is None:
                src_meta = {"transform": src.transform, "crs": src.crs}
        yearly = (data == 3).astype(np.uint8)
        if combined is None:
            combined = yearly
        else:
            combined = np.maximum(combined, yearly)

    # Upsample to reference grid
    upsampled = np.zeros(ref_shape, dtype=np.uint8)
    reproject(
        source=combined,
        destination=upsampled,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )
    return upsampled


def binarise_glads2(tile_id: str, ref_shape, ref_transform, ref_crs):
    """Binarise GLAD-S2: confidence >= 2 -> 1. Returns None if file doesn't exist."""
    path = DATA_ROOT / f"labels/train/glads2/glads2_{tile_id}_alert.tif"
    if not path.exists():
        return None

    with rasterio.open(path) as src:
        data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    binary = (data >= 2).astype(np.uint8)

    if binary.shape == ref_shape:
        return binary

    reprojected = np.zeros(ref_shape, dtype=np.uint8)
    reproject(
        source=binary,
        destination=reprojected,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )
    return reprojected


def binarise_glads2_high_conf(tile_id: str, ref_shape, ref_transform, ref_crs):
    """Binarise GLAD-S2: confidence >= 3 (medium+high) -> 1."""
    path = DATA_ROOT / f"labels/train/glads2/glads2_{tile_id}_alert.tif"
    if not path.exists():
        return None

    with rasterio.open(path) as src:
        data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    binary = (data >= 3).astype(np.uint8)

    if binary.shape == ref_shape:
        return binary

    reprojected = np.zeros(ref_shape, dtype=np.uint8)
    reproject(
        source=binary,
        destination=reprojected,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )
    return reprojected


def build_consensus(radd, gladl, glads2, radd_high, glads2_high):
    """
    Build consensus mask from binarised sources.

    Returns:
      fused_binary: majority-vote binary label
      consensus_mask:
        0 = uncertain
        1 = gold negative (all agree no deforestation)
        2 = silver negative (majority no)
        3 = silver positive (majority yes)
        4 = gold positive (all agree deforestation + high confidence in >=1 source)
    """
    n_sources = 2 if glads2 is None else 3
    vote_sum = radd.astype(np.int8) + gladl.astype(np.int8)
    if glads2 is not None:
        vote_sum += glads2.astype(np.int8)

    fused_binary = (vote_sum > (n_sources / 2)).astype(np.uint8)

    consensus = np.zeros_like(radd, dtype=np.uint8)

    if n_sources == 3:
        # Gold negative: all 3 say no
        gold_neg = vote_sum == 0
        # Gold positive: all 3 say yes AND at least one high-confidence source
        has_high_conf = (radd_high == 1)
        if glads2_high is not None:
            has_high_conf = has_high_conf | (glads2_high == 1)
        gold_pos = (vote_sum == 3) & has_high_conf
        # Silver: 2/3 agree
        silver_neg = (vote_sum <= 1) & ~gold_neg
        silver_pos = (vote_sum >= 2) & ~gold_pos
    else:
        # 2-source: RADD + GLAD-L only
        gold_neg = vote_sum == 0
        gold_pos = (vote_sum == 2) & (radd_high == 1)
        silver_neg = np.zeros_like(radd, dtype=bool)
        silver_pos = np.zeros_like(radd, dtype=bool)
        # Disagreement = uncertain
        uncertain = vote_sum == 1
        # For 2-source, upgrade: both agree but no high conf -> silver
        both_yes_no_high = (vote_sum == 2) & (radd_high == 0)
        silver_pos = both_yes_no_high | uncertain  # 1/2 is uncertain, keep as silver
        # Actually: 1/2 disagree = uncertain, 2/2 no high conf = silver positive
        silver_pos = both_yes_no_high
        # 1/2 disagree: uncertain (stays 0)

    consensus[gold_neg] = 1
    consensus[silver_neg] = 2
    consensus[silver_pos] = 3
    consensus[gold_pos] = 4

    return fused_binary, consensus


def save_raster(data, path, ref_transform, ref_crs):
    """Save a single-band uint8 raster."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": 255,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)


def process_tile(tile_id: str):
    """Process one tile: fuse labels, build consensus, save outputs."""
    print(f"\n{'='*60}")
    print(f"Processing tile: {tile_id}")

    # Use RADD grid as reference (same as GLAD-S2 where available)
    radd_path = DATA_ROOT / f"labels/train/radd/radd_{tile_id}_labels.tif"
    with rasterio.open(radd_path) as src:
        ref_shape = src.shape
        ref_transform = src.transform
        ref_crs = src.crs

    # Binarise each source
    radd, _ = binarise_radd(tile_id)
    radd_high = binarise_radd_high_conf(tile_id)
    gladl = binarise_gladl(tile_id, ref_shape, ref_transform, ref_crs)
    glads2 = binarise_glads2(tile_id, ref_shape, ref_transform, ref_crs)
    glads2_high = binarise_glads2_high_conf(tile_id, ref_shape, ref_transform, ref_crs)

    n_sources = 3 if glads2 is not None else 2
    print(f"  Sources: RADD + GLAD-L{' + GLAD-S2' if glads2 is not None else ''} ({n_sources}-way)")
    print(f"  Grid: {ref_shape} @ {ref_crs}")
    print(f"  RADD positives: {radd.sum():,} (high-conf: {radd_high.sum():,})")
    print(f"  GLAD-L positives: {gladl.sum():,}")
    if glads2 is not None:
        print(f"  GLAD-S2 positives: {glads2.sum():,} (high-conf: {glads2_high.sum():,})")

    # Build consensus
    fused, consensus = build_consensus(radd, gladl, glads2, radd_high, glads2_high)

    # Stats
    total = consensus.size
    tier_names = {0: "uncertain", 1: "gold_neg", 2: "silver_neg", 3: "silver_pos", 4: "gold_pos"}
    for val, name in tier_names.items():
        count = (consensus == val).sum()
        pct = 100 * count / total
        print(f"  {name}: {count:>8,} ({pct:5.1f}%)")
    print(f"  Fused positive: {fused.sum():,} ({100*fused.sum()/total:.1f}%)")

    # Save
    out_dir = OUT_ROOT / tile_id
    save_raster(fused, out_dir / "fused_binary.tif", ref_transform, ref_crs)
    save_raster(consensus, out_dir / "consensus_mask.tif", ref_transform, ref_crs)
    save_raster(radd, out_dir / "source_radd.tif", ref_transform, ref_crs)
    save_raster(gladl, out_dir / "source_gladl.tif", ref_transform, ref_crs)
    if glads2 is not None:
        save_raster(glads2, out_dir / "source_glads2.tif", ref_transform, ref_crs)

    return {
        "tile_id": tile_id,
        "n_sources": n_sources,
        "shape": list(ref_shape),
        "crs": str(ref_crs),
        "fused_pos": int(fused.sum()),
        "fused_neg": int(total - fused.sum()),
        "gold_pos": int((consensus == 4).sum()),
        "gold_neg": int((consensus == 1).sum()),
        "silver_pos": int((consensus == 3).sum()),
        "silver_neg": int((consensus == 2).sum()),
        "uncertain": int((consensus == 0).sum()),
        "total_pixels": int(total),
    }


def build_cv_splits():
    """Create region-stratified CV folds."""
    # Fold 1: hold out 2 SA + 2 SEA
    # Fold 2: rotate
    # Fold 3: rotate
    sa = SOUTH_AMERICA.copy()
    sea = SOUTHEAST_ASIA.copy()

    folds = [
        {
            "fold": 0,
            "val": [sa[0], sa[1], sea[0], sea[1]],
            "train": sa[2:] + sea[2:],
            "note": "Balanced holdout"
        },
        {
            "fold": 1,
            "val": [sa[2], sa[3], sea[2], sea[3]],
            "train": sa[:2] + sa[4:] + sea[:2] + sea[4:],
            "note": "Balanced holdout"
        },
        {
            "fold": 2,
            "val": [sa[4], sa[5], sea[4], sea[5]],
            "train": sa[:4] + sa[6:] + sea[:4] + sea[6:],
            "note": "Balanced holdout"
        },
        {
            "fold": 3,
            "val": [sa[6], sa[7], sea[6], sea[7]],
            "train": sa[:6] + sea[:6],
            "note": "Balanced holdout"
        },
    ]
    return folds


def main():
    print("Building fused label dataset...")
    print(f"South America tiles: {SOUTH_AMERICA}")
    print(f"Southeast Asia tiles: {SOUTHEAST_ASIA}")

    tile_stats = []
    for tile_id in TRAIN_TILES:
        stats = process_tile(tile_id)
        tile_stats.append(stats)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    total_gold_pos = sum(s["gold_pos"] for s in tile_stats)
    total_gold_neg = sum(s["gold_neg"] for s in tile_stats)
    total_silver_pos = sum(s["silver_pos"] for s in tile_stats)
    total_silver_neg = sum(s["silver_neg"] for s in tile_stats)
    total_uncertain = sum(s["uncertain"] for s in tile_stats)
    total_pixels = sum(s["total_pixels"] for s in tile_stats)

    print(f"Total pixels: {total_pixels:,}")
    print(f"Gold positive (eval): {total_gold_pos:,} ({100*total_gold_pos/total_pixels:.1f}%)")
    print(f"Gold negative (eval): {total_gold_neg:,} ({100*total_gold_neg/total_pixels:.1f}%)")
    print(f"Silver positive: {total_silver_pos:,}")
    print(f"Silver negative: {total_silver_neg:,}")
    print(f"Uncertain (excluded): {total_uncertain:,} ({100*total_uncertain/total_pixels:.1f}%)")
    eval_pixels = total_gold_pos + total_gold_neg
    print(f"Evaluable pixels (gold only): {eval_pixels:,} ({100*eval_pixels/total_pixels:.1f}%)")

    # CV splits
    folds = build_cv_splits()
    print(f"\nCV folds: {len(folds)}")
    for f in folds:
        print(f"  Fold {f['fold']}: val={f['val']}")

    # Save metadata
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta = {
        "tile_stats": tile_stats,
        "cv_folds": folds,
        "label_encoding": {
            "fused_binary": "0=no deforestation, 1=deforestation (majority vote)",
            "consensus_mask": {
                "0": "uncertain (exclude from eval)",
                "1": "gold negative (all sources agree: no deforestation)",
                "2": "silver negative (majority no, some disagree)",
                "3": "silver positive (majority yes, or both agree without high conf)",
                "4": "gold positive (all sources agree + high confidence)",
            }
        },
        "sources": {
            "radd": "post-2020, any confidence",
            "gladl": "confirmed loss (value=3) in any year 2021-2025",
            "glads2": "confidence >= 2 (low/medium/high). Only 8 South America tiles.",
        },
    }
    with open(OUT_ROOT / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {OUT_ROOT / 'dataset_meta.json'}")
    print(f"All outputs in {OUT_ROOT}/")


if __name__ == "__main__":
    main()
