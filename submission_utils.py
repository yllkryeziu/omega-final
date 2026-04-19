"""Utilities for converting deforestation prediction rasters into submittable GeoJSON."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def raster_to_geojson(
    raster_path: str | Path,
    output_path: str | Path | None = None,
    min_area_ha: float = 0.5,
    time_step_raster: np.ndarray | None = None,
    time_step_transform=None,
) -> dict:
    """Convert a binary deforestation prediction raster to a GeoJSON FeatureCollection.

    Reads a single-band GeoTIFF where 1 indicates deforestation and 0 indicates
    no deforestation, vectorises the foreground pixels into polygons, removes
    polygons smaller than ``min_area_ha``, reprojects the result to EPSG:4326,
    and returns (and optionally writes) a GeoJSON FeatureCollection.

    The caller is responsible for binarising their model output before passing
    it to this function. This function is designed to be the final step in the
    submission pipeline: take your binarised prediction raster, call this
    function, and upload the resulting ``.geojson`` file to the leaderboard.

    Args:
        raster_path: Path to the input GeoTIFF. Must be a single-band raster
            with binary values (0 = no deforestation, 1 = deforestation).
        output_path: Optional path at which to write the GeoJSON file. Parent
            directories are created automatically. If ``None``, the result is
            returned but not written to disk.
        min_area_ha: Minimum polygon area in hectares. Polygons smaller than
            this threshold are removed before the output is written. Area is
            computed in the appropriate UTM projection so the filter is
            metric-accurate regardless of the raster's native CRS. Defaults
            to ``0.5``.

    Returns:
        A GeoJSON-compatible ``dict`` representing a FeatureCollection. Each
        Feature corresponds to one contiguous deforestation polygon in
        EPSG:4326 (longitude/latitude, WGS-84).

    Raises:
        FileNotFoundError: If ``raster_path`` does not point to an existing file.
        ValueError: If the raster contains no deforestation pixels (all zeros),
            or if all polygons are smaller than ``min_area_ha``.

    Example:
        >>> geojson = raster_to_geojson(
        ...     raster_path="predictions/tile_18NVJ.tif",
        ...     output_path="submission/tile_18NVJ.geojson",
        ...     min_area_ha=0.5,
        ... )
        >>> print(len(geojson["features"]), "deforestation polygons")
    """
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    if data.sum() == 0:
        raise ValueError(
            f"No deforestation pixels (value=1) found in {raster_path}. "
            "Ensure the raster has been binarised before calling this function."
        )

    # Vectorise connected foreground regions into polygons
    polygons = [
        shape(geom)
        for geom, value in shapes(data, mask=data, transform=transform)
        if value == 1
    ]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf = gdf.to_crs("EPSG:4326")

    # Filter by area: project to UTM for metric-accurate ha calculation
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    gdf = gdf[gdf_utm.area / 10_000 >= min_area_ha].reset_index(drop=True)

    if gdf.empty:
        raise ValueError(
            f"All polygons are smaller than min_area_ha={min_area_ha} ha. "
            "Lower the threshold or check your prediction raster."
        )

    if time_step_raster is not None and time_step_transform is not None:
        from rasterio.features import geometry_mask
        gdf_native = gdf.to_crs(crs)
        time_steps = []
        for geom in gdf_native.geometry:
            try:
                mask = ~geometry_mask([geom], out_shape=time_step_raster.shape,
                                      transform=time_step_transform)
                vals = time_step_raster[mask]
                vals = vals[vals > 0]
                if len(vals) > 0:
                    counts = np.bincount(vals.astype(int))
                    mode_val = counts.argmax()
                    yy = mode_val // 100
                    mm = mode_val % 100
                    time_steps.append(f"{yy:02d}{mm:02d}")
                else:
                    time_steps.append(None)
            except Exception:
                time_steps.append(None)
        gdf["time_step"] = time_steps
    else:
        gdf["time_step"] = None

    geojson = json.loads(gdf.to_json())

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f)

    return geojson
