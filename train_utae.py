"""
Tier 2: U-TAE training for deforestation detection.

Trains U-Net with Temporal Attention Encoder on Sentinel-2 time series patches.
Uses fused consensus labels with quality-aware weighting.
Optional AEF embedding fusion at bottleneck.
GPU-accelerated on AMD MI300X via ROCm.

Run:
  python3 train_utae.py --small --fold 0          # Quick single-fold test
  python3 train_utae.py --fold 0                   # Full model, one fold
  python3 train_utae.py                             # Full 4-fold CV
"""

import argparse
import json
import glob
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pyproj import Transformer
from sklearn.metrics import f1_score, precision_score, recall_score

from models_utae import build_utae
from train_xgb import get_s2_ref, build_reproject_grid, gpu_reproject, DATA_ROOT, FUSED_ROOT, TRAIN_TILES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("models")

SELECTED_MONTHS = list(range(0, 72, 3))  # quarterly: 24 timesteps


class TileData:
    __slots__ = ["tile_id", "s2", "positions", "label", "mask", "quality", "aef", "H", "W"]

    def __init__(self, tile_id, s2, positions, label, mask, quality, aef, shape):
        self.tile_id = tile_id
        self.s2 = s2
        self.positions = positions
        self.label = label
        self.mask = mask
        self.quality = quality
        self.aef = aef
        self.H, self.W = shape


def load_tile_s2(tile_id, ref, split="train"):
    s2_dir = DATA_ROOT / f"sentinel-2/{split}/{tile_id}__s2_l2a"
    H, W = ref["shape"]

    file_map = {}
    for f in sorted(s2_dir.glob("*.tif")):
        parts = f.stem.split("_")
        year, month = int(parts[-2]), int(parts[-1])
        mi = (year - 2020) * 12 + (month - 1)
        file_map[mi] = f

    selected_files = []
    positions = []
    for mi in SELECTED_MONTHS:
        if mi in file_map:
            selected_files.append(file_map[mi])
            positions.append(mi)

    if len(selected_files) < 4:
        return None, None

    data = np.zeros((len(selected_files), 12, H, W), dtype=np.float32)
    valid = np.ones(len(selected_files), dtype=bool)

    for t, f in enumerate(selected_files):
        try:
            with rasterio.open(f) as ds:
                arr = ds.read()
                if arr.shape[1:] != (H, W):
                    valid[t] = False
                    continue
                data[t] = arr.astype(np.float32) / 10000.0
        except Exception:
            valid[t] = False

    data = data[valid]
    positions = np.array([p for p, v in zip(positions, valid) if v], dtype=np.int64)
    if len(data) < 4:
        return None, None
    return data, positions


LABEL_FILES = {
    "majority": "fused_binary.tif",
    "ds": "ds_labels.tif",
    "cl": "cl_labels.tif",
}
QUALITY_FILES = {
    "ds": "ds_quality.tif",
    "cl": "cl_quality.tif",
}


def load_tile_labels(tile_id, ref, label_source="majority", load_quality=False):
    label_file = LABEL_FILES[label_source]
    label_path = FUSED_ROOT / tile_id / label_file
    mask_path = FUSED_ROOT / tile_id / "consensus_mask.tif"
    if not label_path.exists():
        label_path = FUSED_ROOT / tile_id / "fused_binary.tif"

    H, W = ref["shape"]
    grid = build_reproject_grid(str(label_path), ref["transform"], ref["crs"], H, W)

    with rasterio.open(label_path) as ds:
        label = ds.read(1).astype(np.float32)
    with rasterio.open(mask_path) as ds:
        mask = ds.read(1).astype(np.float32)

    label_r = gpu_reproject(label[np.newaxis], grid, mode="nearest")[0]
    mask_r = gpu_reproject(mask[np.newaxis], grid, mode="nearest")[0]

    quality_r = None
    if load_quality and label_source in QUALITY_FILES:
        qpath = FUSED_ROOT / tile_id / QUALITY_FILES[label_source]
        if qpath.exists():
            with rasterio.open(qpath) as ds:
                q = ds.read(1).astype(np.float32)
            quality_r = gpu_reproject(q[np.newaxis], grid, mode="bilinear")[0]

    return label_r.round().astype(np.uint8), mask_r.round().astype(np.uint8), quality_r


def load_tile_aef(tile_id, ref, split="train"):
    aef_dir = DATA_ROOT / f"aef-embeddings/{split}"
    p2020 = aef_dir / f"{tile_id}_2020.tiff"
    p2025 = aef_dir / f"{tile_id}_2025.tiff"
    if not p2020.exists() or not p2025.exists():
        return None

    H, W = ref["shape"]
    grid = build_reproject_grid(str(p2020), ref["transform"], ref["crs"], H, W)

    with rasterio.open(p2020) as ds:
        e20 = ds.read().astype(np.float32)
    with rasterio.open(p2025) as ds:
        e25 = ds.read().astype(np.float32)

    diff = e25 - e20
    n20 = np.sqrt((e20 ** 2).sum(0, keepdims=True)).clip(1e-8)
    n25 = np.sqrt((e25 ** 2).sum(0, keepdims=True)).clip(1e-8)
    cos_d = 1.0 - (e20 * e25).sum(0, keepdims=True) / (n20 * n25)
    l2_d = np.sqrt((diff ** 2).sum(0, keepdims=True))
    features = np.concatenate([diff, cos_d, l2_d], axis=0)  # (66, h, w)
    return gpu_reproject(features, grid, mode="bilinear")


def load_all_tiles(tile_ids, split="train", label_source="majority", load_quality=False):
    tiles = []
    for tile_id in tile_ids:
        ref = get_s2_ref(tile_id, split)
        if ref is None:
            print(f"  {tile_id}: skipped (corrupt S2)")
            continue

        result = load_tile_s2(tile_id, ref, split)
        if result[0] is None:
            print(f"  {tile_id}: skipped (too few timesteps)")
            continue
        s2, positions = result

        label, mask, quality = load_tile_labels(tile_id, ref, label_source, load_quality)
        if label is None:
            print(f"  {tile_id}: skipped (no labels)")
            continue

        aef = load_tile_aef(tile_id, ref, split)

        tiles.append(TileData(tile_id, s2, positions, label, mask, quality, aef, ref["shape"]))
        pct = 100 * label.mean()
        q_tag = ", quality=yes" if quality is not None else ""
        print(f"  {tile_id}: S2 {s2.shape}, deforest {pct:.1f}%, AEF={'yes' if aef is not None else 'no'}{q_tag}")

    return tiles


class PatchDataset(Dataset):
    def __init__(self, tiles, patch_size=128, patches_per_tile=64, augment=True, use_aef=True):
        self.tiles = tiles
        self.ps = patch_size
        self.augment = augment
        self.use_aef = use_aef
        self.n = len(tiles) * patches_per_tile

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        tile = self.tiles[idx % len(self.tiles)]
        ps = self.ps
        r = np.random.randint(0, tile.H - ps + 1)
        c = np.random.randint(0, tile.W - ps + 1)

        s2 = tile.s2[:, :, r:r+ps, c:c+ps].copy()
        lab = tile.label[r:r+ps, c:c+ps].copy()
        msk = tile.mask[r:r+ps, c:c+ps].copy()
        aef = tile.aef[:, r:r+ps, c:c+ps].copy() if self.use_aef and tile.aef is not None else None
        qual = tile.quality[r:r+ps, c:c+ps].copy() if tile.quality is not None else None

        if self.augment:
            if np.random.rand() > 0.5:
                s2 = np.flip(s2, -1).copy()
                lab = np.flip(lab, -1).copy()
                msk = np.flip(msk, -1).copy()
                if aef is not None:
                    aef = np.flip(aef, -1).copy()
                if qual is not None:
                    qual = np.flip(qual, -1).copy()
            if np.random.rand() > 0.5:
                s2 = np.flip(s2, -2).copy()
                lab = np.flip(lab, -2).copy()
                msk = np.flip(msk, -2).copy()
                if aef is not None:
                    aef = np.flip(aef, -2).copy()
                if qual is not None:
                    qual = np.flip(qual, -2).copy()
            k = np.random.randint(0, 4)
            if k:
                s2 = np.rot90(s2, k, (-2, -1)).copy()
                lab = np.rot90(lab, k, (-2, -1)).copy()
                msk = np.rot90(msk, k, (-2, -1)).copy()
                if aef is not None:
                    aef = np.rot90(aef, k, (-2, -1)).copy()
                if qual is not None:
                    qual = np.rot90(qual, k, (-2, -1)).copy()

        out = {
            "s2": torch.from_numpy(s2),
            "positions": torch.from_numpy(tile.positions.copy()),
            "label": torch.from_numpy(lab.astype(np.float32)),
            "mask": torch.from_numpy(msk),
        }
        if aef is not None:
            out["aef"] = torch.from_numpy(aef)
        if qual is not None:
            out["quality"] = torch.from_numpy(qual)
        return out


class ValPatchDataset(Dataset):
    def __init__(self, tiles, patch_size=128, use_aef=True):
        self.tiles = tiles
        self.ps = patch_size
        self.use_aef = use_aef
        self.patches = []
        for ti, t in enumerate(tiles):
            for r in range(0, t.H - patch_size + 1, patch_size):
                for c in range(0, t.W - patch_size + 1, patch_size):
                    self.patches.append((ti, r, c))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        ti, r, c = self.patches[idx]
        tile = self.tiles[ti]
        ps = self.ps

        out = {
            "s2": torch.from_numpy(tile.s2[:, :, r:r+ps, c:c+ps].copy()),
            "positions": torch.from_numpy(tile.positions.copy()),
            "label": torch.from_numpy(tile.label[r:r+ps, c:c+ps].astype(np.float32).copy()),
            "mask": torch.from_numpy(tile.mask[r:r+ps, c:c+ps].copy()),
        }
        if self.use_aef and tile.aef is not None:
            out["aef"] = torch.from_numpy(tile.aef[:, r:r+ps, c:c+ps].copy())
        if tile.quality is not None:
            out["quality"] = torch.from_numpy(tile.quality[r:r+ps, c:c+ps].copy())
        return out


def collate_fn(batch):
    max_t = max(b["s2"].shape[0] for b in batch)
    B = len(batch)
    C = batch[0]["s2"].shape[1]
    ps = batch[0]["s2"].shape[2]

    s2 = torch.zeros(B, max_t, C, ps, ps)
    pos = torch.zeros(B, max_t, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["s2"].shape[0]
        s2[i, :t] = b["s2"]
        pos[i, :t] = b["positions"]

    out = {
        "s2": s2,
        "positions": pos,
        "label": torch.stack([b["label"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
    }
    if "aef" in batch[0]:
        out["aef"] = torch.stack([b["aef"] for b in batch])
    if "quality" in batch[0]:
        out["quality"] = torch.stack([b["quality"] for b in batch])
    return out


def evaluate(model, loader, device, pos_weight):
    model.eval()
    all_preds, all_labels, all_masks = [], [], []
    total_loss = 0.0
    n = 0

    pw = torch.tensor([pos_weight]).to(device)

    with torch.no_grad():
        for batch in loader:
            s2 = batch["s2"].to(device)
            positions = batch["positions"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"]
            aef = batch.get("aef")
            if aef is not None:
                aef = aef.to(device)

            logits = model(s2, positions, aef).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw, reduction="mean")
            total_loss += loss.item()
            n += 1

            preds = (logits.sigmoid() > 0.5).cpu().numpy().ravel()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().ravel())
            all_masks.append(masks.numpy().ravel())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_masks = np.concatenate(all_masks)

    gold = (all_masks == 1) | (all_masks == 4)
    if gold.sum() == 0:
        return {"loss": total_loss / max(n, 1), "f1": 0, "precision": 0, "recall": 0, "n_gold": 0}

    y_true = all_labels[gold].astype(int)
    y_pred = all_preds[gold].astype(int)

    return {
        "loss": total_loss / max(n, 1),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "n_gold": int(gold.sum()),
        "n_pos_true": int(y_true.sum()),
        "n_pos_pred": int(y_pred.sum()),
    }


def train_fold(fold_idx, all_tiles, folds, args):
    val_ids = set(folds[fold_idx])
    train_tiles = [t for t in all_tiles if t.tile_id not in val_ids]
    val_tiles = [t for t in all_tiles if t.tile_id in val_ids]

    print(f"\nFold {fold_idx}: train={len(train_tiles)}, val={len(val_tiles)}")
    if not val_tiles:
        print("  No val tiles, skipping")
        return None

    train_ds = PatchDataset(train_tiles, args.patch_size, args.patches_per_tile, True, args.use_aef)
    val_ds = ValPatchDataset(val_tiles, args.patch_size, args.use_aef)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=collate_fn)

    aef_ch = 66 if args.use_aef else 0
    model = build_utae(in_channels=12, aef_channels=aef_ch, small=args.small).to(DEVICE)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    pw = torch.tensor([args.pos_weight]).to(DEVICE)
    best_f1 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        nb = 0
        t0 = time.time()

        for bi, batch in enumerate(train_loader):
            s2 = batch["s2"].to(DEVICE)
            positions = batch["positions"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            aef = batch.get("aef")
            if aef is not None:
                aef = aef.to(DEVICE)

            logits = model(s2, positions, aef).squeeze(1)

            loss_raw = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw, reduction="none")

            if "quality" in batch and args.use_quality:
                weight = batch["quality"].to(DEVICE)
            else:
                weight = torch.ones_like(labels)
                weight[masks == 0] = 0.1
                weight[(masks == 2) | (masks == 3)] = 0.5

            loss = (loss_raw * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            nb += 1

            if (bi + 1) % 20 == 0:
                print(f"    batch {bi+1}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        avg = epoch_loss / max(nb, 1)
        dt = time.time() - t0

        do_eval = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        if do_eval:
            m = evaluate(model, val_loader, DEVICE, args.pos_weight)
            tag = ""
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_epoch = epoch + 1
                tag = " *BEST*"
                save_p = MODEL_DIR / f"utae_fold{fold_idx}_best.pt"
                save_p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_p)
            print(f"  Epoch {epoch+1}/{args.epochs} [{dt:.0f}s] "
                  f"train={avg:.4f} val={m['loss']:.4f} "
                  f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"gold={m['n_gold']:,} pos={m['n_pos_true']:,}/{m['n_pos_pred']:,}{tag}")
        else:
            print(f"  Epoch {epoch+1}/{args.epochs} [{dt:.0f}s] train={avg:.4f}")

    print(f"  Best F1={best_f1:.4f} at epoch {best_epoch}")
    return {"fold": fold_idx, "best_f1": best_f1, "best_epoch": best_epoch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--patches_per_tile", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=5.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--use_aef", action="store_true", default=True)
    parser.add_argument("--no_aef", dest="use_aef", action="store_false")
    parser.add_argument("--labels", choices=["majority", "ds", "cl"], default="majority",
                        help="Label source: majority vote, dawid-skene, or crowdlab")
    parser.add_argument("--use-quality", action="store_true",
                        help="Use per-pixel quality scores for loss weighting (requires ds/cl labels)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, VRAM: {props.total_memory / 1e9:.0f} GB")

    with open(FUSED_ROOT / "dataset_meta.json") as f:
        folds_list = json.load(f)["cv_folds"]
    folds = {f["fold"]: f["val"] for f in folds_list}

    print(f"\nLabels: {args.labels}, quality weighting: {args.use_quality}")
    print("\n" + "=" * 60)
    print("LOADING TILES")
    print("=" * 60)
    all_tiles = load_all_tiles(TRAIN_TILES, label_source=args.labels,
                               load_quality=args.use_quality)
    total_mem = sum(t.s2.nbytes + t.label.nbytes + t.mask.nbytes +
                    (t.aef.nbytes if t.aef is not None else 0) for t in all_tiles)
    print(f"\nLoaded {len(all_tiles)} tiles, {total_mem / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    fold_range = [args.fold] if args.fold is not None else range(4)
    results = []
    for fi in fold_range:
        r = train_fold(fi, all_tiles, folds, args)
        if r:
            results.append(r)

    if results:
        mean_f1 = np.mean([r["best_f1"] for r in results])
        print(f"\n{'=' * 60}")
        print(f"Mean F1 = {mean_f1:.4f}")
        for r in results:
            print(f"  Fold {r['fold']}: F1={r['best_f1']:.4f} (epoch {r['best_epoch']})")

        out = MODEL_DIR / "utae_cv_results.json"
        with open(out, "w") as f:
            json.dump({"results": results, "mean_f1": float(mean_f1), "args": vars(args)}, f, indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
