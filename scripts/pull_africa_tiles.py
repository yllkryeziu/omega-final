"""
Pull African training tiles from Google Earth Engine to close the training gap
for test tile 33NTE_5_1 (Congo basin). The existing training set has zero African
coverage; this script exports the same modalities used by the makeathon pipeline
(AEF + RADD + GLAD-L + Hansen + optionally S1/S2) for the tiles listed in
scripts/africa_tiles.json.

Stages (run as separate invocations):
  --submit        : submits GEE export tasks to your Drive. Prints task IDs.
  --monitor       : polls submitted tasks and prints progress.
  --restructure   : given a local folder where you downloaded the Drive outputs,
                    moves files into the makeathon data/ layout.

Prerequisites (one-time):
    pip install earthengine-api
    earthengine authenticate    # opens a browser, then paste code back

Optional stages:
  --stages aef,radd,gladl,hansen        (single-modality path: ~70 tasks, ~6 GB)
  --stages aef,radd,gladl,hansen,s1,s2  (multimodal U-Net path: ~1540 tasks,
                                          ~26 GB; emits per-month per-scene
                                          S1/S2 matching the makeathon layout
                                          exactly — same folder tree, same
                                          filenames, same band order, same dtype)

Why these modalities, why these stages:
  - AEF(2020)+AEF(latest)  → base embedding features (all model variants).
  - RADD + GLAD-L          → 2 of the 3 weak-label sources. GLAD-S2 is Amazon-only
                              so it is EXCLUDED for Africa. Fusion threshold
                              effectively becomes "RADD and GLAD-L both fire"
                              (higher precision, lower recall) — acceptable,
                              optionally augmented by Hansen as a 3rd vote.
  - Hansen lossyear        → optional 3rd vote to recover recall in Africa.
  - S2 L2A monthly median  → 12 bands per month, cloud-masked via SCL.
  - S1 VV monthly mean     → LINEAR power (S1_GRD_FLOAT), ASCENDING orbit,
                              matches the makeathon `dst>0` + `10*log10` convention.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


AEF_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
RADD_ASSET = "projects/radar-wur/raddalert/v1"
GLADL_ASSET_TMPL = "projects/glad/alert/{yy}final"   # e.g. 21final, 22final, ...
HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"
S2_ASSET = "COPERNICUS/S2_SR_HARMONIZED"
S1_ASSET = "COPERNICUS/S1_GRD_FLOAT"   # linear power; matches makeathon convention

DRIVE_FOLDER = "makeathon_africa_v1"

# Years matching the makeathon schema
AEF_YEARS = [2020]        # always need 2020 baseline
AEF_LATEST_CANDIDATES = [2025, 2024]   # try 2025 first, fall back to 2024
GLADL_YEARS = [21, 22, 23, 24, 25]
S2S1_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


# ─── GEE authentication + geometry ──────────────────────────────────────────

def init_ee():
    import ee
    try:
        ee.Initialize()
    except Exception:
        print("earthengine not initialized — run `earthengine authenticate` first.",
              file=sys.stderr)
        raise
    return ee


def tile_region(ee, lon: float, lat: float, km: float = 10.0):
    """10 km x 10 km bbox centered on (lon, lat), in EPSG:4326."""
    pt = ee.Geometry.Point([lon, lat])
    return pt.buffer(km * 500).bounds()   # half-extent = km * 500 m


def utm_epsg(lon: float, lat: float) -> str:
    """Local UTM EPSG for (lon, lat). Challenge S1/S2 are delivered in local UTM
    (challenge.ipynb, sections 4.1 and 4.2) — NOT EPSG:4326. AEF stays 4326."""
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


# ─── AEF ─────────────────────────────────────────────────────────────────────

def submit_aef(ee, tile, year):
    region = tile_region(ee, tile["lon"], tile["lat"])
    coll = (ee.ImageCollection(AEF_ASSET)
              .filterBounds(region)
              .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    size = coll.size().getInfo()
    if size == 0:
        return None
    img = coll.mosaic().clip(region).toFloat()
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"aef_{tile['tile_id']}_{year}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"{tile['tile_id']}_{year}",
        region=region,
        scale=10,
        crs="EPSG:4326",
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def resolve_latest_aef_year(ee, tile) -> int:
    region = tile_region(ee, tile["lon"], tile["lat"])
    for yr in AEF_LATEST_CANDIDATES:
        n = (ee.ImageCollection(AEF_ASSET)
               .filterBounds(region)
               .filterDate(f"{yr}-01-01", f"{yr + 1}-01-01")
               .size().getInfo())
        if n > 0:
            return yr
    raise RuntimeError(f"No AEF available for any of {AEF_LATEST_CANDIDATES} "
                       f"at {tile['tile_id']}")


# ─── RADD ────────────────────────────────────────────────────────────────────

def submit_radd(ee, tile):
    """
    RADD is an ImageCollection; African alerts are tagged geography='africa'.
    Mosaic the most recent alert per pixel. Encoding: leading digit = conf
    (2=low, 3=high); trailing digits = days since 2014-12-31.
    """
    region = tile_region(ee, tile["lon"], tile["lat"])
    coll = (ee.ImageCollection(RADD_ASSET)
              .filterMetadata("geography", "equals", "africa")
              .filterMetadata("layer", "equals", "alert")
              .filterBounds(region))
    if coll.size().getInfo() == 0:
        return None
    # Max preserves the most recent / highest-confidence alert per pixel
    img = coll.select("Alert").max().clip(region).toInt32()
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"radd_{tile['tile_id']}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"radd_{tile['tile_id']}_labels",
        region=region,
        scale=10,
        crs="EPSG:4326",
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


# ─── GLAD-L ──────────────────────────────────────────────────────────────────

def submit_gladl(ee, tile, yy):
    """
    Per-year GLAD-L alert raster. Exports just the conf{yy} band so the file is
    a single-channel {0,2,3} raster matching the makeathon gladl schema.
    """
    region = tile_region(ee, tile["lon"], tile["lat"])
    asset = GLADL_ASSET_TMPL.format(yy=yy)
    try:
        img = ee.Image(asset).select(f"conf{yy}").clip(region).toUint8()
    except Exception:
        return None
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"gladl_{tile['tile_id']}_{yy}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"gladl_{tile['tile_id']}_alert{yy}",
        region=region,
        scale=30,       # GLAD-L is natively 30 m; matches makeathon
        crs="EPSG:4326",
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


# ─── Hansen GFC (optional 4th vote) ──────────────────────────────────────────

def submit_hansen(ee, tile):
    region = tile_region(ee, tile["lon"], tile["lat"])
    img = (ee.Image(HANSEN_ASSET)
             .select(["lossyear", "treecover2000"])
             .clip(region)
             .toUint8())
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"hansen_{tile['tile_id']}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"hansen_{tile['tile_id']}_loss",
        region=region,
        scale=30,
        crs="EPSG:4326",
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


# ─── S2 — per-month median composite (matches makeathon monthly layout) ─────

_S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
             "B8A", "B9", "B11", "B12"]   # 12-band L2A, no B10; ESA index order


def _month_date_range(year: int, month: int) -> tuple[str, str]:
    """Return (start, end) ISO dates spanning exactly one calendar month."""
    start = f"{year}-{month:02d}-01"
    if month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month + 1:02d}-01"
    return start, end


def submit_s2_month(ee, tile, year, month):
    """
    Per-month median composite of Sentinel-2 L2A, cloud-masked via SCL.

    Output: 12-band uint16 GeoTIFF in band order
        B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
    (rasterio index 1..12 — matches `build_multimodal_features.py:52`).

    Filename: {tile_id}__s2_l2a_{year}_{month}.tif   (month 1..12, no zero-pad)
    """
    region = tile_region(ee, tile["lon"], tile["lat"])
    start, end = _month_date_range(year, month)

    def mask_scl(img):
        # Mask cloud shadow (3), cloud med/high (8,9), thin cirrus (10), snow (11).
        scl = img.select("SCL")
        clear = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
                          .And(scl.neq(10)).And(scl.neq(11)))
        return img.updateMask(clear)

    coll = (ee.ImageCollection(S2_ASSET)
              .filterBounds(region)
              .filterDate(start, end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 95))
              .map(mask_scl)
              .select(_S2_BANDS))   # select AFTER mask so SCL is still available

    if coll.size().getInfo() == 0:
        return None  # month has zero usable scenes — skip, pipeline glob allows gaps
    img = coll.median().clip(region).toUint16()

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"s2_{tile['tile_id']}_{year}_{month:02d}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"{tile['tile_id']}__s2_l2a_{year}_{month}",  # NO zero-pad
        region=region,
        scale=10,
        crs=utm_epsg(tile["lon"], tile["lat"]),   # local UTM, per challenge.ipynb §4.1
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


# ─── S1 — per-month VV mean in LINEAR power (matches repo convention) ───────

def submit_s1_month(ee, tile, year, month):
    """
    Per-month mean VV backscatter in LINEAR power from S1_GRD_FLOAT, ASCENDING
    orbit only (matches the primary glob in `build_multimodal_features.py:78`).

    Linear (not dB) is required — the repo does:
        db = np.where(dst > 0, 10 * np.log10(dst + 1e-10), np.nan)
    which is a no-op on dB data but correct on linear power. Exporting from
    S1_GRD_FLOAT gives linear natively; S1_GRD would be wrong.

    Output: 1-band float32 GeoTIFF.
    Filename: {tile_id}__s1_rtc_{year}_{month}_ascending.tif
    """
    region = tile_region(ee, tile["lon"], tile["lat"])
    start, end = _month_date_range(year, month)

    coll = (ee.ImageCollection(S1_ASSET)
              .filterBounds(region)
              .filterDate(start, end)
              .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
              .filter(ee.Filter.listContains(
                  "transmitterReceiverPolarisation", "VV"))
              .select("VV"))

    if coll.size().getInfo() == 0:
        return None  # no ASCENDING coverage this month (rare in tropics, happens)
    img = coll.mean().clip(region).toFloat()

    task = ee.batch.Export.image.toDrive(
        image=img,
        description=f"s1_{tile['tile_id']}_{year}_{month:02d}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"{tile['tile_id']}__s1_rtc_{year}_{month}_ascending",
        region=region,
        scale=10,
        crs=utm_epsg(tile["lon"], tile["lat"]),   # local UTM, per challenge.ipynb §4.2
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


# ─── Driver ─────────────────────────────────────────────────────────────────

def submit_all(tiles, stages: set[str]):
    ee = init_ee()
    task_log = []

    for tile in tiles:
        print(f"\n── {tile['tile_id']} ({tile['region']}, {tile['biome']}) ──")

        if "aef" in stages:
            latest = resolve_latest_aef_year(ee, tile)
            print(f"  AEF years: 2020, {latest}")
            for yr in [2020, latest]:
                t = submit_aef(ee, tile, yr)
                if t:
                    task_log.append({"tile": tile["tile_id"], "kind": f"aef_{yr}",
                                     "task_id": t.id, "desc": t.config["description"]})

        if "radd" in stages:
            t = submit_radd(ee, tile)
            if t:
                task_log.append({"tile": tile["tile_id"], "kind": "radd",
                                 "task_id": t.id, "desc": t.config["description"]})
            else:
                print("  WARN: no RADD (Africa) coverage at this centroid")

        if "gladl" in stages:
            for yy in GLADL_YEARS:
                t = submit_gladl(ee, tile, yy)
                if t:
                    task_log.append({"tile": tile["tile_id"], "kind": f"gladl_{yy}",
                                     "task_id": t.id, "desc": t.config["description"]})

        if "hansen" in stages:
            t = submit_hansen(ee, tile)
            if t:
                task_log.append({"tile": tile["tile_id"], "kind": "hansen",
                                 "task_id": t.id, "desc": t.config["description"]})

        if "s2" in stages:
            n_s2 = 0
            for yr in S2S1_YEARS:
                for mo in range(1, 13):
                    t = submit_s2_month(ee, tile, yr, mo)
                    if t:
                        task_log.append({"tile": tile["tile_id"],
                                         "kind": f"s2_{yr}_{mo:02d}",
                                         "task_id": t.id,
                                         "desc": t.config["description"]})
                        n_s2 += 1
            print(f"  S2: submitted {n_s2}/{12 * len(S2S1_YEARS)} monthly tasks")

        if "s1" in stages:
            n_s1 = 0
            for yr in S2S1_YEARS:
                for mo in range(1, 13):
                    t = submit_s1_month(ee, tile, yr, mo)
                    if t:
                        task_log.append({"tile": tile["tile_id"],
                                         "kind": f"s1_{yr}_{mo:02d}",
                                         "task_id": t.id,
                                         "desc": t.config["description"]})
                        n_s1 += 1
            print(f"  S1: submitted {n_s1}/{12 * len(S2S1_YEARS)} monthly tasks")

    log_path = Path("data/derived/africa_export_tasks.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(task_log, indent=2))
    print(f"\n{len(task_log)} export tasks submitted.")
    print(f"Task log -> {log_path}")
    print(f"Outputs will appear in Google Drive folder: {DRIVE_FOLDER}")
    print("Run with --monitor to poll progress.")


def monitor():
    ee = init_ee()
    log_path = Path("data/derived/africa_export_tasks.json")
    if not log_path.exists():
        print(f"No task log at {log_path}. Submit first.")
        return
    tasks = json.loads(log_path.read_text())
    while True:
        done = running = failed = 0
        for t in tasks:
            status = ee.data.getTaskStatus([t["task_id"]])[0]
            state = status["state"]
            if state == "COMPLETED":
                done += 1
            elif state in ("READY", "RUNNING"):
                running += 1
            else:
                failed += 1
        total = len(tasks)
        print(f"[{time.strftime('%H:%M:%S')}] done={done}/{total}  "
              f"running={running}  failed={failed}")
        if done + failed == total:
            break
        time.sleep(60)


def restructure(drive_local: Path):
    """
    Move files from the flat Drive download folder into the makeathon data/
    layout so the existing pipeline finds them without modification.
    """
    drive_local = Path(drive_local)
    if not drive_local.exists():
        print(f"Not found: {drive_local}")
        return

    dest_aef = Path("data/makeathon-challenge/aef-embeddings/train")
    dest_radd = Path("data/makeathon-challenge/labels/train/radd")
    dest_gladl = Path("data/makeathon-challenge/labels/train/gladl")
    dest_hansen = Path("data/derived/hansen")
    for d in (dest_aef, dest_radd, dest_gladl, dest_hansen):
        d.mkdir(parents=True, exist_ok=True)

    # Routing ORDER matters: S1/S2 monthly filenames contain "_2020"/"_2024"/"_2025"
    # via the year-month suffix, so they must be matched BEFORE the AEF branch.
    moved = 0
    for f in drive_local.glob("*.tif*"):
        name = f.name
        if name.startswith("radd_") and name.endswith("_labels.tif"):
            dst = dest_radd / name
        elif name.startswith("gladl_") and "_alert" in name:
            dst = dest_gladl / name
        elif name.startswith("hansen_"):
            dst = dest_hansen / name
        elif "__s2_l2a_" in name:
            tile_id = name.split("__s2_l2a_")[0]
            tile_dir = Path("data/makeathon-challenge/sentinel-2/train") / f"{tile_id}__s2_l2a"
            tile_dir.mkdir(parents=True, exist_ok=True)
            dst = tile_dir / name
        elif "__s1_rtc_" in name:
            tile_id = name.split("__s1_rtc_")[0]
            tile_dir = Path("data/makeathon-challenge/sentinel-1/train") / f"{tile_id}__s1_rtc"
            tile_dir.mkdir(parents=True, exist_ok=True)
            dst = tile_dir / name
        elif name.endswith(".tif") and ("_2020" in name or "_2024" in name
                                        or "_2025" in name):
            # AEF file — rename .tif -> .tiff to match makeathon convention
            dst = dest_aef / (f.stem + ".tiff")
        else:
            print(f"  skip (unknown): {name}")
            continue
        shutil.move(str(f), str(dst))
        moved += 1
        print(f"  {name}  ->  {dst}")
    print(f"\nMoved {moved} files.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--restructure", action="store_true")
    ap.add_argument("--drive-local", type=str, default=None,
                    help="Local path where you downloaded the Drive folder "
                         "(for --restructure).")
    ap.add_argument("--tiles", type=str, default="scripts/africa_tiles.json")
    ap.add_argument("--stages", type=str, default="aef,radd,gladl,hansen",
                    help="Comma-separated subset of {aef,radd,gladl,hansen,s1,s2}.")
    args = ap.parse_args()

    if args.submit:
        tiles = json.loads(Path(args.tiles).read_text())["tiles"]
        stages = set(s.strip() for s in args.stages.split(",") if s.strip())
        submit_all(tiles, stages)
    elif args.monitor:
        monitor()
    elif args.restructure:
        if not args.drive_local:
            print("--restructure requires --drive-local /path/to/downloaded/folder")
            sys.exit(1)
        restructure(args.drive_local)
    else:
        ap.print_help()
