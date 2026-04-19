"""
Improve weak label fusion using Dawid-Skene and CROWDLAB.

Dawid-Skene: EM algorithm that learns per-source confusion matrices
and produces probabilistic consensus labels. No trained classifier needed.

CROWDLAB (cleanlab): Multi-annotator aggregation that incorporates
model predictions for tie-breaking. Uses vote-ratio as pred_probs.

For each train tile, produces:
  - ds_labels.tif:   Dawid-Skene consensus binary labels
  - ds_quality.tif:  Dawid-Skene per-pixel quality score (float32)
  - cl_labels.tif:   CROWDLAB consensus binary labels
  - cl_quality.tif:  CROWDLAB per-pixel quality score (float32)

Run: python3 build_improved_labels.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from build_dataset import (
    DATA_ROOT, OUT_ROOT, TRAIN_TILES,
    binarise_radd, binarise_radd_high_conf,
    binarise_gladl, binarise_glads2, save_raster,
)

try:
    from cleanlab.multiannotator import get_label_quality_multiannotator
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False
    print("cleanlab not installed, skipping CROWDLAB")


def dawid_skene(labels, n_iter=50, tol=1e-4):
    """
    Dawid-Skene EM for binary labels with multiple annotators.

    Args:
        labels: (N, M) array, 0/1/NaN. NaN = annotator did not label this item.
        n_iter: max EM iterations
        tol: convergence threshold on label change rate

    Returns:
        probs: (N,) array, P(true label = 1) for each item
        confusion: (M, 2, 2) array, confusion[j][k][l] = P(annotator j says l | true = k)
        prior: (2,) array, class prior [P(0), P(1)]
    """
    N, M = labels.shape
    observed = ~np.isnan(labels)
    labs = np.nan_to_num(labels, nan=0).astype(np.int8)

    # Initialize with majority vote
    vote_sum = np.nansum(labels, axis=1)
    vote_count = observed.sum(axis=1).clip(1)
    probs = vote_sum / vote_count  # P(true=1)
    probs = np.clip(probs, 0.01, 0.99)

    for it in range(n_iter):
        old_probs = probs.copy()

        # M-step: estimate confusion matrices and class prior
        prior = np.array([1.0 - probs.mean(), probs.mean()])
        prior = prior.clip(0.01)
        prior /= prior.sum()

        confusion = np.zeros((M, 2, 2))
        for j in range(M):
            mask = observed[:, j]
            if mask.sum() == 0:
                confusion[j] = np.eye(2)
                continue
            for true_k in range(2):
                w = probs[mask] if true_k == 1 else (1.0 - probs[mask])
                w_total = w.sum() + 1e-10
                for obs_l in range(2):
                    match = (labs[mask, j] == obs_l).astype(np.float64)
                    confusion[j, true_k, obs_l] = (w * match).sum() / w_total
            # Ensure rows sum to 1
            for k in range(2):
                row_sum = confusion[j, k].sum()
                if row_sum > 0:
                    confusion[j, k] /= row_sum

        # E-step: update P(true=1) for each item
        log_prior = np.log(prior + 1e-30)
        log_lik = np.zeros((N, 2))
        log_lik[:, 0] = log_prior[0]
        log_lik[:, 1] = log_prior[1]

        for j in range(M):
            mask = observed[:, j]
            for true_k in range(2):
                obs = labs[mask, j]
                log_lik[mask, true_k] += np.log(confusion[j, true_k, obs] + 1e-30)

        # Normalize to get probabilities
        log_lik -= log_lik.max(axis=1, keepdims=True)
        lik = np.exp(log_lik)
        lik_sum = lik.sum(axis=1, keepdims=True)
        probs = lik[:, 1] / lik_sum.ravel()
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

        # Check convergence
        change = np.abs(probs - old_probs).mean()
        if change < tol:
            print(f"    Dawid-Skene converged at iteration {it+1} (change={change:.6f})")
            break
    else:
        print(f"    Dawid-Skene: {n_iter} iterations (change={change:.6f})")

    return probs, confusion, prior


def run_crowdlab(labels, pred_probs):
    """Run CROWDLAB via cleanlab."""
    df = pd.DataFrame(labels, columns=[f"ann_{i}" for i in range(labels.shape[1])])
    # Replace NaN properly for cleanlab (expects pd.NA or np.nan for missing)

    results = get_label_quality_multiannotator(
        df, pred_probs,
        quality_method="crowdlab",
        verbose=False,
    )

    consensus = results["label_quality"]["consensus_label"].values.astype(np.int8)
    quality = results["label_quality"]["consensus_quality_score"].values.astype(np.float32)

    return consensus, quality


def save_float_raster(data, path, ref_transform, ref_crs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data, 1)


def process_tile(tile_id):
    print(f"\n{'='*60}")
    print(f"Tile: {tile_id}")

    # Load reference grid
    radd_path = DATA_ROOT / f"labels/train/radd/radd_{tile_id}_labels.tif"
    with rasterio.open(radd_path) as src:
        ref_shape = src.shape
        ref_transform = src.transform
        ref_crs = src.crs

    H, W = ref_shape

    # Load binarized sources
    radd, _ = binarise_radd(tile_id)
    gladl = binarise_gladl(tile_id, ref_shape, ref_transform, ref_crs)
    glads2 = binarise_glads2(tile_id, ref_shape, ref_transform, ref_crs)

    n_sources = 3 if glads2 is not None else 2
    N = H * W

    # Build annotator matrix (N, M) with NaN for missing
    if glads2 is not None:
        labels = np.stack([radd.ravel(), gladl.ravel(), glads2.ravel()], axis=1).astype(np.float64)
    else:
        labels = np.stack([radd.ravel(), gladl.ravel()], axis=1).astype(np.float64)
        # For Dawid-Skene: add NaN column for missing GLAD-S2
        labels = np.column_stack([labels, np.full(N, np.nan)])
    labels_2col = np.stack([radd.ravel(), gladl.ravel()], axis=1).astype(np.float64) if glads2 is None else labels

    print(f"  Sources: {n_sources}, pixels: {N:,}")

    # ---- Dawid-Skene ----
    t0 = time.time()
    ds_probs, ds_confusion, ds_prior = dawid_skene(labels, n_iter=50)
    dt = time.time() - t0

    ds_labels = (ds_probs > 0.5).astype(np.uint8)
    ds_quality = np.where(ds_probs > 0.5, ds_probs, 1.0 - ds_probs).astype(np.float32)

    ds_labels_2d = ds_labels.reshape(H, W)
    ds_quality_2d = ds_quality.reshape(H, W)

    n_pos = ds_labels.sum()
    print(f"  Dawid-Skene [{dt:.1f}s]: {n_pos:,} positive ({100*n_pos/N:.1f}%)")
    print(f"    Prior: P(0)={ds_prior[0]:.4f}, P(1)={ds_prior[1]:.4f}")
    for j, name in enumerate(["RADD", "GLAD-L", "GLAD-S2"]):
        if j < ds_confusion.shape[0]:
            sens = ds_confusion[j, 1, 1]
            spec = ds_confusion[j, 0, 0]
            print(f"    {name}: sensitivity={sens:.4f}, specificity={spec:.4f}")

    # Compare with majority vote
    mv_path = OUT_ROOT / tile_id / "fused_binary.tif"
    with rasterio.open(mv_path) as ds:
        mv = ds.read(1).ravel()
    changed = (ds_labels != mv).sum()
    print(f"    Changed from majority vote: {changed:,} pixels ({100*changed/N:.2f}%)")

    # Save DS results
    out_dir = OUT_ROOT / tile_id
    save_raster(ds_labels_2d, out_dir / "ds_labels.tif", ref_transform, ref_crs)
    save_float_raster(ds_quality_2d, out_dir / "ds_quality.tif", ref_transform, ref_crs)

    # ---- CROWDLAB ----
    cl_consensus = None
    cl_quality_2d = None
    if HAS_CLEANLAB:
        t0 = time.time()
        # Build pred_probs from vote ratio (simple heuristic)
        vote_sum = np.nansum(labels, axis=1)
        vote_count = (~np.isnan(labels)).sum(axis=1).clip(1)
        p1 = vote_sum / vote_count
        # Smooth slightly to avoid 0/1
        p1 = np.clip(p1 * 0.9 + 0.05, 0.01, 0.99)
        pred_probs = np.column_stack([1 - p1, p1]).astype(np.float32)

        try:
            cl_input = labels_2col if glads2 is None else labels
            cl_pred_probs = pred_probs
            cl_labels_flat, cl_quality_flat = run_crowdlab(cl_input, cl_pred_probs)
            dt = time.time() - t0

            cl_consensus = cl_labels_flat.reshape(H, W).astype(np.uint8)
            cl_quality_2d = cl_quality_flat.reshape(H, W)

            n_pos_cl = cl_consensus.sum()
            changed_cl = (cl_labels_flat != mv).sum()
            print(f"  CROWDLAB [{dt:.1f}s]: {n_pos_cl:,} positive ({100*n_pos_cl/N:.1f}%)")
            print(f"    Changed from majority vote: {changed_cl:,} pixels ({100*changed_cl/N:.2f}%)")

            save_raster(cl_consensus, out_dir / "cl_labels.tif", ref_transform, ref_crs)
            save_float_raster(cl_quality_2d, out_dir / "cl_quality.tif", ref_transform, ref_crs)
        except Exception as e:
            print(f"  CROWDLAB failed: {e}")

    return {
        "tile_id": tile_id,
        "n_sources": n_sources,
        "ds_positive": int(ds_labels.sum()),
        "ds_changed": int(changed),
        "ds_prior": ds_prior.tolist(),
        "ds_confusion": ds_confusion.tolist(),
    }


def main():
    print("Building improved labels with Dawid-Skene + CROWDLAB")
    print(f"cleanlab available: {HAS_CLEANLAB}")

    results = []
    t_total = time.time()

    for tile_id in TRAIN_TILES:
        r = process_tile(tile_id)
        results.append(r)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s")

    total_changed = sum(r["ds_changed"] for r in results)
    total_pixels = sum(r["ds_positive"] + r["ds_changed"] for r in results)
    print(f"\nDawid-Skene changed {total_changed:,} pixels total from majority vote")

    # Save summary
    out = OUT_ROOT / "improved_labels_meta.json"
    with open(out, "w") as f:
        json.dump({"results": results, "elapsed": elapsed}, f, indent=2)
    print(f"Metadata saved to {out}")


if __name__ == "__main__":
    main()
