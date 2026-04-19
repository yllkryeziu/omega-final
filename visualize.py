#!/usr/bin/env python3
"""
Visualization script for deforestation detection project.
Generates a multi-panel figure showing input data, derived indices, and predictions
for a given tile.

Panels:
  1. S2 True Color (R,G,B = B04,B03,B02)
  2. S2 False Color (R,G,B = B08,B04,B03)
  3. NDVI Change (late minus early)
  4. S1 VV Change (late minus early, dB)
  5. AEF PCA (64-dim embeddings projected to 3 PCA components -> RGB)
  6. Prediction overlay on S2 true color
  7. Labels overlay on S2 true color (train only)
"""

import argparse
import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "makeathon-challenge")
LABEL_DIR = os.path.join(BASE_DIR, "data", "fused-labels")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

TRAIN_TILES = [
    "18NWG_6_6", "18NWH_1_4", "18NWJ_8_9", "18NWM_9_4",
    "18NXH_6_8", "18NXJ_7_6", "18NYH_9_9", "19NBD_4_4",
    "47QMB_0_8", "47QQV_2_4", "48PUT_0_8", "48PWV_7_8",
    "48PXC_7_7", "48PYB_3_6", "48QVE_3_0", "48QWD_2_2",
]
TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]

# S2 band indices (1-based for rasterio): B01=1, B02=2, B03=3, B04=4, ..., B08=8
S2_BLUE = 2   # B02
S2_GREEN = 3  # B03
S2_RED = 4    # B04
S2_NIR = 8    # B08


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_split(tile_id):
    """Auto-detect whether tile is train or test."""
    if tile_id in TRAIN_TILES:
        return "train"
    elif tile_id in TEST_TILES:
        return "test"
    else:
        # Try to find it in file system
        for split in ("train", "test"):
            s2_dir = os.path.join(DATA_DIR, "sentinel-2", split, f"{tile_id}__s2_l2a")
            if os.path.isdir(s2_dir):
                return split
        print(f"WARNING: Could not detect split for tile {tile_id}, defaulting to 'test'")
        return "test"


def find_s2_files(tile_id, split, years):
    """Find all S2 files for given tile and years. Returns list of paths sorted by (year, month)."""
    s2_dir = os.path.join(DATA_DIR, "sentinel-2", split, f"{tile_id}__s2_l2a")
    files = []
    for year in years:
        pattern = os.path.join(s2_dir, f"{tile_id}__s2_l2a_{year}_*.tif")
        matches = glob.glob(pattern)
        for f in matches:
            # Extract month from filename: {tile}__s2_l2a_{year}_{month}.tif
            basename = os.path.basename(f)
            parts = basename.replace(".tif", "").split("_")
            month = int(parts[-1])
            files.append((year, month, f))
    files.sort()
    return files


def find_s1_files(tile_id, split, years):
    """Find all S1 files for given tile and years. Returns list of paths."""
    s1_dir = os.path.join(DATA_DIR, "sentinel-1", split, f"{tile_id}__s1_rtc")
    files = []
    for year in years:
        pattern = os.path.join(s1_dir, f"{tile_id}__s1_rtc_{year}_*.tif")
        matches = glob.glob(pattern)
        for f in matches:
            files.append(f)
    files.sort()
    return files


def read_s2_bands(filepath, bands):
    """Read specific bands from an S2 file. bands is a list of 1-based band indices.
    Returns array of shape (len(bands), H, W) as float32, scaled to reflectance."""
    with rasterio.open(filepath) as ds:
        data = ds.read(bands).astype(np.float32) / 10000.0
    return data


def percentile_stretch(img, low=2, high=98):
    """Apply percentile stretch to a 3-band image (3, H, W). Returns (H, W, 3) clipped to [0, 1]."""
    out = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
    for i in range(3):
        band = img[i]
        valid = band[np.isfinite(band) & (band > 0)]
        if len(valid) == 0:
            out[:, :, i] = 0
            continue
        lo = np.percentile(valid, low)
        hi = np.percentile(valid, high)
        if hi <= lo:
            hi = lo + 1e-6
        stretched = (band - lo) / (hi - lo)
        out[:, :, i] = np.clip(stretched, 0, 1)
    return out


def compute_ndvi(nir, red):
    """Compute NDVI from NIR and Red bands. Returns 2D array."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
    return ndvi


def safe_mean_stack(file_list, read_func):
    """Compute mean of rasters from file list, skipping files that fail to read."""
    arrays = []
    for f in file_list:
        try:
            arr = read_func(f)
            if arr is not None:
                arrays.append(arr)
        except Exception:
            continue
    if len(arrays) == 0:
        return None
    target_shape = arrays[0].shape
    arrays = [a for a in arrays if a.shape == target_shape]
    if len(arrays) == 0:
        return None
    return np.nanmean(np.stack(arrays, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Panel generators
# ---------------------------------------------------------------------------

def _score_clarity(filepath):
    """Score an S2 image by how cloud-free it is. Higher = clearer.
    Uses green band (B03): valid pixels with moderate reflectance indicate clear sky.
    Very bright pixels (>0.3 reflectance) are likely clouds."""
    try:
        with rasterio.open(filepath) as ds:
            green = ds.read(3).astype(np.float32) / 10000.0  # B03
        total = green.size
        nodata = (green == 0).sum()
        bright = (green > 0.25).sum()  # clouds are very bright
        clear_frac = 1.0 - (nodata + bright) / total
        return clear_frac
    except Exception:
        return -1.0


def _pick_s2_rgb(tile_id, split, years, bands):
    """Pick the clearest S2 image from the given years, return percentile-stretched RGB."""
    best_score = -1.0
    best_path = None
    for year in years:
        files = find_s2_files(tile_id, split, [year])
        for _, _, filepath in files:
            score = _score_clarity(filepath)
            if score > best_score:
                best_score = score
                best_path = filepath
    if best_path is not None:
        try:
            data = read_s2_bands(best_path, bands)
            return percentile_stretch(data)
        except Exception:
            pass
    return None


def make_s2_early(tile_id, split):
    """S2 True Color from 2020 (before deforestation)."""
    return _pick_s2_rgb(tile_id, split, [2020], [S2_RED, S2_GREEN, S2_BLUE])


def make_s2_late(tile_id, split):
    """S2 True Color from 2024-2025 (after deforestation)."""
    return _pick_s2_rgb(tile_id, split, [2025, 2024], [S2_RED, S2_GREEN, S2_BLUE])


def make_s2_false_color(tile_id, split):
    """S2 False Color (B08, B04, B03 = NIR, R, G) from late period."""
    return _pick_s2_rgb(tile_id, split, [2025, 2024], [S2_NIR, S2_RED, S2_GREEN])


def make_ndvi_change(tile_id, split):
    """Panel 3: NDVI Change (late mean - early mean).
    Early = mean of 2020 NDVI. Late = mean of 2024-2025 NDVI."""
    def read_ndvi(filepath):
        data = read_s2_bands(filepath, [S2_NIR, S2_RED])
        return compute_ndvi(data[0], data[1])

    early_files = find_s2_files(tile_id, split, [2020])
    late_files = find_s2_files(tile_id, split, [2024, 2025])

    early_paths = [f for _, _, f in early_files]
    late_paths = [f for _, _, f in late_files]

    if not early_paths or not late_paths:
        return None

    early_ndvi = safe_mean_stack(early_paths, read_ndvi)
    late_ndvi = safe_mean_stack(late_paths, read_ndvi)

    if early_ndvi is None or late_ndvi is None:
        return None

    return late_ndvi - early_ndvi


def make_s1_change(tile_id, split):
    """Panel 4: S1 VV Change (late mean dB - early mean dB).
    Early = 2020. Late = 2024-2025."""
    def read_vv(filepath):
        with rasterio.open(filepath) as ds:
            return ds.read(1).astype(np.float32)

    early_files = find_s1_files(tile_id, split, [2020])
    late_files = find_s1_files(tile_id, split, [2024, 2025])

    if not early_files or not late_files:
        return None

    early_vv = safe_mean_stack(early_files, read_vv)
    late_vv = safe_mean_stack(late_files, read_vv)

    if early_vv is None or late_vv is None:
        return None

    # Convert to dB: 10 * log10(linear power)
    with np.errstate(divide="ignore", invalid="ignore"):
        early_db = 10.0 * np.log10(np.clip(early_vv, 1e-10, None))
        late_db = 10.0 * np.log10(np.clip(late_vv, 1e-10, None))

    change = late_db - early_db
    change = np.where(np.isfinite(change), change, 0)
    return change


def make_aef_pca(tile_id, split):
    """Panel 5: AEF PCA visualization. Read 2025 AEF (64 dims), PCA to 3 components -> RGB."""
    aef_path = os.path.join(DATA_DIR, "aef-embeddings", split, f"{tile_id}_2025.tiff")
    if not os.path.exists(aef_path):
        # Fallback to 2024
        aef_path = os.path.join(DATA_DIR, "aef-embeddings", split, f"{tile_id}_2024.tiff")
    if not os.path.exists(aef_path):
        return None

    try:
        from sklearn.decomposition import PCA

        with rasterio.open(aef_path) as ds:
            data = ds.read().astype(np.float32)  # (64, H, W)

        n_bands, h, w = data.shape
        # Reshape to (H*W, 64)
        pixels = data.reshape(n_bands, -1).T  # (H*W, 64)

        # Handle NaN/Inf
        valid_mask = np.all(np.isfinite(pixels), axis=1)
        if valid_mask.sum() < 10:
            return None

        # Fit PCA on valid pixels only
        pca = PCA(n_components=3)
        transformed = np.zeros((h * w, 3), dtype=np.float32)
        transformed[valid_mask] = pca.fit_transform(pixels[valid_mask])

        # Reshape back
        rgb = transformed.reshape(h, w, 3)

        # Normalize each channel to [0, 1] using percentile stretch
        for c in range(3):
            channel = rgb[:, :, c]
            valid_vals = channel[valid_mask.reshape(h, w)]
            if len(valid_vals) == 0:
                continue
            lo = np.percentile(valid_vals, 2)
            hi = np.percentile(valid_vals, 98)
            if hi <= lo:
                hi = lo + 1e-6
            rgb[:, :, c] = np.clip((channel - lo) / (hi - lo), 0, 1)

        return rgb
    except Exception as e:
        print(f"  Warning: AEF PCA failed: {e}")
        return None


def make_prediction_overlay(tile_id, split, s2_rgb):
    """Panel 6: Prediction overlaid on S2 true color.
    Returns RGBA image or None."""
    pred_path = os.path.join(SUBMISSION_DIR, f"pred_{tile_id}.tif")
    if not os.path.exists(pred_path):
        return None

    try:
        with rasterio.open(pred_path) as ds:
            pred = ds.read(1)  # binary uint8

        if s2_rgb is None:
            return None

        # pred may have different dimensions from s2_rgb if CRS matches but size differs.
        # Resize pred to match s2_rgb using nearest neighbor.
        h_s2, w_s2 = s2_rgb.shape[:2]
        h_pred, w_pred = pred.shape

        if (h_pred != h_s2) or (w_pred != w_s2):
            from PIL import Image
            pred_pil = Image.fromarray(pred, mode="L")
            pred_pil = pred_pil.resize((w_s2, h_s2), resample=Image.NEAREST)
            pred = np.array(pred_pil)

        # Create overlay: S2 as base, deforestation pixels in red with transparency
        overlay = s2_rgb.copy()
        mask = pred > 0
        # Blend: deforestation pixels become reddish
        overlay[mask] = overlay[mask] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6

        return overlay
    except Exception as e:
        print(f"  Warning: Prediction overlay failed: {e}")
        return None


def make_label_overlay(tile_id, split, s2_rgb):
    """Panel 7: Fused labels overlaid on S2 true color (train only).
    Returns RGBA image, None, or 'no_labels' string for test tiles."""
    if split == "test":
        return "no_labels"

    label_path = os.path.join(LABEL_DIR, tile_id, "fused_binary.tif")
    if not os.path.exists(label_path):
        return None

    try:
        with rasterio.open(label_path) as ds:
            labels = ds.read(1)  # binary uint8

        if s2_rgb is None:
            return None

        h_s2, w_s2 = s2_rgb.shape[:2]
        h_lbl, w_lbl = labels.shape

        if (h_lbl != h_s2) or (w_lbl != w_s2):
            from PIL import Image
            lbl_pil = Image.fromarray(labels, mode="L")
            lbl_pil = lbl_pil.resize((w_s2, h_s2), resample=Image.NEAREST)
            labels = np.array(lbl_pil)

        overlay = s2_rgb.copy()
        mask = labels > 0
        overlay[mask] = overlay[mask] * 0.4 + np.array([1.0, 0.0, 0.0]) * 0.6

        return overlay
    except Exception as e:
        print(f"  Warning: Label overlay failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def create_figure(tile_id, split, output_path):
    """Create the multi-panel visualization figure.

    Layout (2 rows x 4 columns):
        Row 0: S2 2020 (before) | Prediction        | NDVI Change  | S1 VV Change
        Row 1: S2 2025 (after)  | Labels            | AEF PCA      | False Color
    """
    print(f"Generating visualization for tile {tile_id} (split={split})")
    print(f"Output: {output_path}")

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 4, figure=fig, wspace=0.15, hspace=0.20)
    fig.suptitle(f"Tile: {tile_id} ({split})", fontsize=18, fontweight="bold", y=0.98)

    def _show(ax, img, title, fallback_text="N/A"):
        ax.set_title(title, fontsize=11, fontweight="bold")
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, fallback_text, ha="center", va="center", fontsize=14,
                    transform=ax.transAxes, color="gray")
        ax.set_xticks([]); ax.set_yticks([])

    def _show_diverging(ax, data, title, cbar_label, default_vmax=0.5):
        ax.set_title(title, fontsize=11, fontweight="bold")
        if data is not None:
            vmax = max(abs(np.nanpercentile(data, 2)),
                       abs(np.nanpercentile(data, 98)))
            if vmax < 0.01:
                vmax = default_vmax
            im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14,
                    transform=ax.transAxes, color="gray")
        ax.set_xticks([]); ax.set_yticks([])

    # ---- Row 0, Col 0: S2 2020 (before) ----
    print("  [1/8] S2 Early (2020)...")
    s2_early = make_s2_early(tile_id, split)
    _show(fig.add_subplot(gs[0, 0]), s2_early, "S2 True Color — 2020 (before)")

    # ---- Row 1, Col 0: S2 2024-25 (after) ----
    print("  [2/8] S2 Late (2024-25)...")
    s2_late = make_s2_late(tile_id, split)
    _show(fig.add_subplot(gs[1, 0]), s2_late, "S2 True Color — 2024-25 (after)")

    s2_rgb = s2_late if s2_late is not None else s2_early

    # ---- Row 0, Col 1: Prediction ----
    print("  [3/8] Prediction overlay...")
    pred_overlay = make_prediction_overlay(tile_id, split, s2_rgb)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Prediction (red = deforestation)", fontsize=11, fontweight="bold")
    if pred_overlay is not None:
        ax.imshow(pred_overlay)
    else:
        if s2_rgb is not None:
            ax.imshow(s2_rgb, alpha=0.5)
        ax.text(0.5, 0.5, "No predictions\navailable", ha="center", va="center",
                fontsize=12, transform=ax.transAxes, color="black",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Row 1, Col 1: Labels ----
    print("  [4/8] Labels overlay...")
    label_result = make_label_overlay(tile_id, split, s2_rgb)
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Labels (red = deforestation)", fontsize=11, fontweight="bold")
    if isinstance(label_result, str) and label_result == "no_labels":
        if s2_rgb is not None:
            ax.imshow(s2_rgb, alpha=0.5)
        ax.text(0.5, 0.5, "No labels\n(test tile)", ha="center", va="center",
                fontsize=12, transform=ax.transAxes, color="black",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    elif label_result is not None:
        ax.imshow(label_result)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14,
                transform=ax.transAxes, color="gray")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- Row 0, Col 2: NDVI Change ----
    print("  [5/8] NDVI Change...")
    ndvi_change = make_ndvi_change(tile_id, split)
    _show_diverging(fig.add_subplot(gs[0, 2]), ndvi_change,
                    "NDVI Change (2024-25 minus 2020)", "NDVI diff")

    # ---- Row 0, Col 3: S1 VV Change ----
    print("  [6/8] S1 VV Change...")
    s1_change = make_s1_change(tile_id, split)
    _show_diverging(fig.add_subplot(gs[0, 3]), s1_change,
                    "S1 VV Change dB (2024-25 minus 2020)", "dB diff", 5.0)

    # ---- Row 1, Col 2: AEF PCA ----
    print("  [7/8] AEF PCA...")
    aef_rgb = make_aef_pca(tile_id, split)
    _show(fig.add_subplot(gs[1, 2]), aef_rgb, "AEF Embeddings (PCA -> RGB)")

    # ---- Row 1, Col 3: S2 False Color ----
    print("  [8/8] S2 False Color...")
    s2_fc = make_s2_false_color(tile_id, split)
    _show(fig.add_subplot(gs[1, 3]), s2_fc, "S2 False Color (NIR, R, G)")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-panel visualization for a deforestation detection tile."
    )
    parser.add_argument(
        "--tile", required=True,
        help="Tile ID (e.g. 18NWG_6_6)"
    )
    parser.add_argument(
        "--split", choices=["train", "test"], default=None,
        help="Data split (default: auto-detect from tile ID)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path (default: figures/{tile_id}_overview.png)"
    )
    args = parser.parse_args()

    tile_id = args.tile
    split = args.split if args.split else detect_split(tile_id)

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(BASE_DIR, "figures", f"{tile_id}_overview.png")

    create_figure(tile_id, split, output_path)


if __name__ == "__main__":
    main()
