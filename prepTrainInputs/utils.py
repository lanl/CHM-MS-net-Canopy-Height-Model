"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

# Standard library
import json
import logging
import math
import multiprocessing
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from datetime import datetime, timezone
from glob import glob
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass
import warnings
import stat
import traceback

# Third-party
import geopandas as gpd
import numpy as np
import pandas as pd
import pdal
import rasterio
import requests
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import (
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
    shape,
    LineString
)
from rasterio.vrt import WarpedVRT
from shapely.ops import transform as shp_transform, unary_union, nearest_points
from shapely.wkt import loads
import shapely
# add these to your existing block
from functools import partial
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
#TODO: remove unnecessary imports


def saveRasterToUTM(rasterPath, epsg, savePath):
    """
    Reproject a raster file to the specified UTM EPSG code, always resampling to 0.5m resolution
    using bicubic interpolation. Replaces all 'no data' pixels with 0.

    Args:
        rasterPath (str): The file path of the input raster.
        epsg (int): The EPSG code of the desired UTM projection.
        savePath (str): Path to save the output raster.
    """
    start_time = time.time()
    
    with rasterio.open(rasterPath) as src:
        # Force a 0.5m resolution resampling and reprojection
        transform, width, height = calculate_default_transform(
            src.crs,
            f'EPSG:{epsg}',
            src.width,
            src.height,
            *src.bounds,
            resolution=(0.5, 0.5)
        )

        # Update metadata (set nodata to 0)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': f'EPSG:{epsg}',
            'transform': transform,
            'width': width,
            'height': height,
            'no data': 0
        })

        with rasterio.open(savePath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                data = np.empty((height, width), dtype=src.dtypes[i - 1])

                reproject(
                    source=rasterio.band(src, i),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=f'EPSG:{epsg}',
                    resampling=Resampling.cubic
                )

                # Replace nodata or NaN with 0
                if src.nodata is not None:
                    data[data == src.nodata] = 0
                data = np.nan_to_num(data, nan=0)

                dst.write(data, i)

    total_time = time.time() - start_time
    # print(f"Reprojection and resampling complete in {total_time:.2f} seconds.")

def genWKT(aoi, shp_path):
    """
    Generate a Well-Known Text (WKT) representation of the bounding box that encloses
    both an Area of Interest (AOI) and a shapefile.

    This function calculates the smallest bounding box that contains both the given AOI
    and the extent of the specified shapefile. The resulting bounding box is then
    converted to WGS84 coordinates and returned as a WKT string.

    Args:
        aoi (geopandas.GeoDataFrame): A GeoDataFrame containing the AOI geometry.
        shp_path (str): The file path to the shapefile.

    Returns:
        str: A WKT string representing the bounding box in WGS84 coordinates.
    """
    # Get the bounding box of the AOI
    aoi_bbox = aoi.total_bounds
    
    # Read the shapefile
    shp_gdf = gpd.read_file(shp_path)
    
    # Get the bounding box of the shapefile
    shp_bbox = shp_gdf.total_bounds
    shp_crs = shp_gdf.crs
    
    # Ensure both AOI and shapefile are in the same CRS
    if aoi.crs != shp_crs:
        aoi = aoi.to_crs(shp_crs)
        aoi_bbox = aoi.total_bounds
    
    # Calculate the smallest bbox that encloses both the AOI and shapefile
    min_x = min(aoi_bbox[0], shp_bbox[0]) - 512
    min_y = min(aoi_bbox[1], shp_bbox[1]) - 512
    max_x = max(aoi_bbox[2], shp_bbox[2]) + 512
    max_y = max(aoi_bbox[3], shp_bbox[3]) + 512
    
    # Create a shapely box geometry from the combined bbox
    combined_bbox = box(min_x, min_y, max_x, max_y)
    
    # Create a GeoDataFrame from the combined bbox
    gdf = gpd.GeoDataFrame(geometry=[combined_bbox], crs=shp_crs)
    
    # Convert to WGS84 (EPSG:4326) for lat/lon
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    
    # Get the WKT in lat/lon
    wkt = gdf_wgs84.geometry[0].wkt
    
    return wkt

def getRasterShape(raster_path, target_resolution=200):
    """
    Downsample a raster to 100m resolution and extract its shape into a GeoDataFrame.

    Parameters:
    raster_path (str): Path to the input raster file.
    target_resolution (float): Target resolution in meters (default is 100m).

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the shape of the downsampled raster.
    """
    try:
        with rasterio.open(raster_path) as src:
            # Calculate the downscaling factor
            scale_factor = target_resolution / src.res[0]
            
            # Calculate new dimensions
            new_width = int(src.width / scale_factor)
            new_height = int(src.height / scale_factor)
            
            # Resample the raster
            data = src.read(
                1,  # Read the first band
                out_shape=(new_height, new_width),
                resampling=Resampling.average
            )
            
            # Update the transform
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )
            
            # Create a mask for valid data
            mask = data != src.nodata
            
            # Get shapes of the resampled raster
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                    shapes(data, mask=mask, transform=transform))
            )
            
            # Convert shapes to a list
            geoms = list(results)
            
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(geoms)
            
            # Set the coordinate reference system (CRS) of the GeoDataFrame
            gdf.crs = src.crs
            
            return gdf
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    

def saveInfShape(input_path, output_path, buffer_meters=0, epsg=None):
    """
    Read an AOI (SHP/GeoJSON/anything GeoPandas reads), reproject to the REQUIRED EPSG
    (e.g., a UTM zone), make a buffered axis-aligned rectangle in that CRS,
    and save it (default: GeoJSON).

    Args:
        input_path: path to AOI vector file.
        output_path: file path (extension decides driver) or a directory.
        buffer_meters: buffer to grow the bbox (meters, since EPSG is projected).
        epsg: REQUIRED int, the target projected CRS (e.g., 32610).

    Returns:
        str: path to the saved file.
    """
    if epsg is None:
        raise ValueError("You must specify epsg (e.g., 32610 for UTM Zone 10N).")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"AOI not found: {input_path}")

    # 1) Load and project AOI to target EPSG (meters)
    aoi = gpd.read_file(input_path)
    if aoi.empty:
        raise ValueError("AOI is empty.")
    if aoi.crs is None:
        raise ValueError("AOI has no CRS; set a CRS before running.")

    working = aoi.to_crs(epsg=epsg)

    # 2) Compute bbox and buffer in meters (axis-aligned in target CRS)
    minx, miny, maxx, maxy = working.total_bounds
    if buffer_meters:
        minx -= buffer_meters
        miny -= buffer_meters
        maxx += buffer_meters
        maxy += buffer_meters

    rect = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=working.crs)

    # 3) Decide output path/driver
    if output_path.is_dir() or output_path.suffix == "":
        output_path = output_path / f"{input_path.stem}_bbox.geojson"

    ext = output_path.suffix.lower()
    if ext in (".geojson", ".json"):
        driver = "GeoJSON"       # Note: GeoJSON spec prefers EPSG:4326; many tools still read projected.
    elif ext == ".gpkg":
        driver = "GPKG"
    elif ext == ".shp":
        driver = "ESRI Shapefile"
    else:
        driver = None  # Let Fiona guess; or set explicitly if you prefer.

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rect.to_file(output_path, driver=driver)

    return str(output_path)

def snap(x, res):
    # snap coordinate to nearest multiple of res
    return math.ceil(round(x / res) * res)

def build_train_square_near_inference(
    inference_geojson_path,
    output_geojson_path,
    num_tiles=1024,
    halo_m=16,
    res=0.5,
    epsg=None
):
    """
    Read an inference AOI, reproject to REQUIRED EPSG (meters), build a square training
    footprint sized to hold >= num_tiles tiles (256 m) with stride = 256 - 2*halo_m,
    snap to a 'res' grid, then AFTER placement expand the bbox outward by exactly
    2*STRIDE (axis-aligned), re-snap edges to 'res', and save as GeoJSON.

    Notes:
      - OUTER_BUFFER = 2*STRIDE and is applied after placement (more than num_tiles may fit).
      - Uses global snap(x, res) you already defined.
    """
    TILE = 256.0

    # --- Validate inputs
    in_path = Path(inference_geojson_path)
    out_path = Path(output_geojson_path)

    if in_path.suffix.lower() != ".geojson":
        raise ValueError("inference_geojson_path must be a .geojson file.")
    if epsg is None or not isinstance(epsg, int):
        raise ValueError("You must provide a projected EPSG integer (e.g., epsg=32610).")
    if num_tiles <= 0:
        raise ValueError("num_tiles must be >= 1")

    stride = TILE - 2 * halo_m
    if stride <= 0:
        raise ValueError("halo_m too large: stride must be positive (i.e., < 128).")
    OUTER_BUFFER = 1 * stride  # fixed outer buffer

    # --- Load inference AOI (GeoJSON), reproject to requested EPSG (meters)
    inf_gdf = gpd.read_file(in_path)
    if inf_gdf.empty:
        raise ValueError("Inference GeoJSON is empty.")
    if inf_gdf.crs is None:
        raise ValueError("Inference GeoJSON has no CRS defined.")
    inf_proj = inf_gdf.to_crs(epsg=epsg)

    # --- Center of inference bounds
    minx, miny, maxx, maxy = inf_proj.total_bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    # Snap center to tile grid for stable tiling symmetry
    cx = snap(cx, TILE)
    cy = snap(cy, TILE)

    # Tiles per side and exact footprint side length (N×N layout)
    N = math.ceil(math.sqrt(num_tiles))
    side = TILE + (N - 1) * stride  # also equals N*TILE - 2*halo_m*(N-1)

    # Snap edges to 'res' so bounds fall on whole "pixels"
    half_px = round((side / 2.0) / res)
    half = half_px * res
    cx = snap(cx, res)
    cy = snap(cy, res)

    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    # --- Initial square footprint (pre-buffer)
    rect = box(xmin, ymin, xmax, ymax)

    # --- POST-PLACEMENT OUTER BUFFER (expand bbox by 2*STRIDE), then re-snap to res
    bxmin, bymin, bxmax, bymax = rect.bounds
    bxmin -= OUTER_BUFFER
    bymin -= OUTER_BUFFER
    bxmax += OUTER_BUFFER
    bymax += OUTER_BUFFER

    bxmin = snap(bxmin, res)
    bymin = snap(bymin, res)
    bxmax = snap(bxmax, res)
    bymax = snap(bymax, res)

    rect = box(bxmin, bymin, bxmax, bymax)
    final_side = bxmax - bxmin  # (square)

    # --- Build GeoDataFrame in the requested EPSG
    out_gdf = gpd.GeoDataFrame(
        {
            "id": [1],
            "tiles_per_side": [N],
            "tile_m": [TILE],
            "halo_m": [halo_m],
            "stride_m": [stride],
            "outer_buffer_m": [OUTER_BUFFER],
            "res": [res],
            "base_side_m": [2 * half],
            "final_side_m": [final_side],
        },
        geometry=[rect],
        crs=f"EPSG:{epsg}",
    )

    # --- Force .geojson output and write
    if out_path.suffix.lower() != ".geojson":
        out_path = out_path.with_suffix(".geojson")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(out_path, driver="GeoJSON")

    return str(out_path)


def fetch_DEM(geojson_path, save_path, api_key, demtype="COP30", buffer_m=0):
    """
    Fetch DEM data from the OpenTopography /globaldem API as a GeoTIFF,
    using the bounding box of a GeoJSON file, optionally expanded with a buffer.

    Args:
        geojson_path (str | Path): Path to a GeoJSON file defining the AOI.
        save_path (str | Path): Path to save the output GeoTIFF file.
        api_key (str): Your OpenTopography API key.
        demtype (str): DEM type to request (default="COP30").
        buffer_m (float): Buffer in meters to expand the bounding box on each side (default=512).

    Returns:
        str: Path to the saved GeoTIFF file.

    Raises:
        RuntimeError: If the request fails or returns no data.
    """
    geojson_path = Path(geojson_path)
    save_path = Path(save_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    # --- Load AOI
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError("GeoJSON is empty.")
    if gdf.crs is None:
        raise ValueError("GeoJSON has no CRS defined.")

    # --- Project to a meter-based CRS for buffering (Web Mercator fallback)
    gdf_m = gdf.to_crs(epsg=3857)
    minx, miny, maxx, maxy = gdf_m.total_bounds

    # --- Apply buffer in meters
    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m
    buffered = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=3857)

    # --- Convert buffered bbox back to WGS84
    buffered = buffered.to_crs(epsg=4326)
    minx, miny, maxx, maxy = buffered.total_bounds

    # --- API request
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": miny,
        "north": maxy,
        "west": minx,
        "east": maxx,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    response = requests.get(url, params=params, stream=True)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(save_path)
    elif response.status_code == 204:
        raise RuntimeError("No data available for given bounding box.")
    elif response.status_code == 401:
        raise RuntimeError("Unauthorized – check your API key.")
    else:
        raise RuntimeError(f"API request failed with code {response.status_code}: {response.text}")

def _load_aoi_from_shapefile(shp_path: str) -> Dict[str, Any]:
    """
    Reads a Shapefile, unions all geometries, reprojects to WGS84, and
    returns a GeoJSON-like geometry dict suitable for STAC 'intersects'.
    """
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("AOI shapefile is empty.")
    if gdf.crs is None:
        raise ValueError("AOI shapefile has no CRS. Please define one or reproject to WGS84.")
    gdf = gdf.to_crs(4326)  # WGS84 lon/lat
    geom = unary_union(gdf.geometry)  # merge all features into one
    if geom.is_empty:
        raise ValueError("AOI geometry is empty after union.")
    return mapping(geom)  # GeoJSON geometry dict

def search_maxar_discovery_from_shapefile(
    shp_path: str,
    api_key: str,
    target_year: int,
    *,
    page_size: int = 100,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None,
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    """
    Search Maxar Discovery (WV01/WV02/WV03) for scenes intersecting the AOI from a Shapefile,
    with off-nadir < 20°, and return features sorted by closeness to the target year.

    Parameters
    ----------
    shp_path : str
        Path to AOI Shapefile (.shp). Can contain multiple features; they will be merged.
    api_key : str
        Maxar API Bearer token.
    target_year : int
        Year to which results should be temporally closest (e.g., 2022).
    page_size : int
        Results per page (Discovery supports ~50–100). Default 100.
    start_date : str
        Beginning of the search window (YYYY-MM-DD). Default "2000-01-01".
    end_date : str | None
        End of the search window (YYYY-MM-DD). Defaults to today (UTC).
    timeout : int
        Per-request timeout (seconds). Default 120.

    Returns
    -------
    list[dict]
        STAC Features with two helper fields added:
          - feature["__datetime"] (datetime, UTC)
          - feature["__days_from_target"] (int)
        Sorted ascending by __days_from_target (and then by off-nadir).
    """
    DISCOVERY_URL = "https://api.maxar.com/discovery/v1/catalogs/imagery/search"
    aoi_geom = _load_aoi_from_shapefile(shp_path)

    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dt_range = f"{start_date}/{end_date}"

    # Fixed collections and server-side off-nadir filter
    body: Dict[str, Any] = {
        "collections": ["wv01", "wv02", "wv03"],
        "limit": page_size,
        "datetime": dt_range,
        "intersects": aoi_geom,
        "query": {
            "view:off_nadir": {"lt": 20}
        },
        "fields": {
            "include": [
                "id",
                "collection",
                "geometry",
                "bbox",
                "assets",
                "properties.datetime",
                "properties.view:off_nadir",
                "properties.eo:cloud_cover",
                "properties.gsd",
                "properties.platform",
            ]
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Always enable AOI-based calculations in Discovery
    params = {"area-based-calc": "true"}

    # Page through results up to a hard cap of 100 items
    max_items = 100
    items: List[Dict[str, Any]] = []
    url = DISCOVERY_URL

    while url and len(items) < max_items:
        resp = requests.post(url, headers=headers, json=body, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        items.extend(features)
        if len(items) >= max_items:
            items = items[:max_items]
            break

        # Follow STAC 'next' link if present
        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next" and link.get("href"):
                next_url = link["href"]
                break
        url = next_url

        # After the first request, 'next' is a fully formed URL; clear body/params
        if url:
            body = None
            params = None

    # Sort by temporal proximity to midpoint (July 1) of target_year
    target_dt = datetime(target_year, 7, 1, tzinfo=timezone.utc)

    def _parse_iso(dt_str: str) -> datetime:
        # Robust ISO-8601 parsing without extra deps
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return datetime.fromisoformat(dt_str)

    cleaned: List[Dict[str, Any]] = []
    for feat in items:
        dt_str = feat.get("properties", {}).get("datetime")
        if not dt_str:
            continue
        dt = _parse_iso(dt_str)
        days = abs((dt - target_dt).days)
        feat["__datetime"] = dt
        feat["__days_from_target"] = days
        cleaned.append(feat)

    cleaned.sort(
        key=lambda f: (
            f["__days_from_target"],
            f.get("properties", {}).get("view:off_nadir", 9999),
        )
    )

    return cleaned

def test_maxar_discovery_connection(api_key: str, timeout: int = 30) -> bool:
    """
    Quick connectivity/auth test for Maxar Discovery API.
    Calls the root catalog endpoint and checks for HTTP 200.

    Parameters
    ----------
    api_key : str
        Maxar Bearer token (or API key).
    timeout : int
        Timeout in seconds. Default 30.

    Returns
    -------
    bool
        True if connection succeeds, False otherwise.
    """
    url = "https://api.maxar.com/discovery/v1"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            print("✅ Connection successful.")
            print("Catalog ID:", data.get("id"))
            # Show a few available links for confirmation
            for link in data.get("links", [])[:5]:
                print("-", link.get("title"), ":", link.get("href"))
            return True
        else:
            print(f"❌ Connection failed. Status {resp.status_code}")
            print("Response text:", resp.text[:500])
            return False
    except Exception as e:
        print("❌ Exception during request:", str(e))
        return False
    

def snap_down(v, step): return math.floor(v / step) * step
def snap_up(v, step): return math.ceil(v / step) * step

def _largest_rect_in_binary_grid(grid):
    """
    Given a binary grid (rows = Y, cols = X), find the max-area axis-aligned
    rectangle of 1's. Returns (top_row, left_col, bottom_row_excl, right_col_excl, area_cells).
    """
    if not grid or not grid[0]:
        return None
    n_rows, n_cols = len(grid), len(grid[0])
    heights = [0]*n_cols
    best = (0, 0, 0, 0, 0)  # area, top, left, bot_excl, right_excl
    for r in range(n_rows):
        # Build histogram heights
        for c in range(n_cols):
            heights[c] = heights[c] + 1 if grid[r][c] else 0

        # Largest rectangle in histogram (classic stack)
        stack = []  # (col_idx, height_start_row)
        c = 0
        while c <= n_cols:
            h = heights[c] if c < n_cols else 0
            start = c
            while stack and stack[-1][1] > h:
                idx, prev_h = stack.pop()
                width = c - (stack[-1][0] if stack else 0)
                area = prev_h * width
                # bottom row inclusive is r; top row = r - prev_h + 1
                top = r - prev_h + 1
                left = stack[-1][0] if stack else 0
                right = c
                if area > best[0]:
                    best = (area, top, left, r+1, right)
                start = idx
            if not stack or stack and stack[-1][1] < h:
                stack.append((c, h))
            c += 1
    if best[0] == 0:
        return None
    _, top, left, bot_excl, right = best
    return (top, left, bot_excl, right, best[0])

def place_train_rect_within_mask(
    inference_path,
    rough_train_path,
    output_path,
    n_tiles,
    epsg,
    buffer=16,
    res=0.5,
):
    """
    1) Find the largest axis-aligned rectangle fully inside the training AOI (aligned to STRIDE grid).
    2) Place a stride-snapped, pixel-snapped rectangle that fits >= n_tiles tiles fully inside that inner bbox.
    Saves the placed rectangle as GeoJSON (EPSG:epsg).
    """
    TILE = 256.0
    STRIDE = TILE - 2.0 * buffer
    if STRIDE <= 0:
        raise ValueError(f"Invalid buffer={buffer}. Must be <128 (so STRIDE stays positive).")

    # --- Load & project
    trn = gpd.read_file(rough_train_path).to_crs(epsg=epsg)
    inf = gpd.read_file(inference_path).to_crs(epsg=epsg)  # not strictly needed here, but kept for parity
    if trn.empty:
        raise ValueError("Rough training shape is empty.")
    trn_union = getattr(trn, "unary_union", None) or unary_union(trn.geometry.tolist())
    if trn_union.is_empty:
        raise ValueError("Training AOI is empty after reprojection.")

    # --- STRIDE-aligned grid covering the AOI bbox
    tb_minx, tb_miny, tb_maxx, tb_maxy = trn_union.bounds
    gx0 = snap_down(tb_minx, STRIDE)
    gy0 = snap_down(tb_miny, STRIDE)
    gx1 = snap_up(tb_maxx, STRIDE)
    gy1 = snap_up(tb_maxy, STRIDE)

    n_cols = int(round((gx1 - gx0) / STRIDE))
    n_rows = int(round((gy1 - gy0) / STRIDE))
    if n_cols <= 0 or n_rows <= 0:
        raise RuntimeError("Degenerate AOI bounds for grid construction.")

    # Build a binary grid: cell is 1 iff the entire STRIDE cell is covered by AOI
    grid = [[0]*n_cols for _ in range(n_rows)]
    for r in range(n_rows):
        y0 = gy0 + r * STRIDE
        y1 = y0 + STRIDE
        for c in range(n_cols):
            x0 = gx0 + c * STRIDE
            x1 = x0 + STRIDE
            cell = box(x0, y0, x1, y1)
            # covers == AOI completely covers the cell (including boundaries)
            if trn_union.covers(cell):
                grid[r][c] = 1

    # Largest rectangle of 1's (in cell units)
    rect_cells = _largest_rect_in_binary_grid(grid)
    if rect_cells is None:
        raise RuntimeError("Could not find any STRIDE-sized region fully inside the training AOI.")
    top, left, bot_excl, right, area_cells = rect_cells

    # Convert to coordinates (largest inner bbox)
    inner_minx = gx0 + left * STRIDE
    inner_miny = gy0 + top * STRIDE
    inner_maxx = gx0 + right * STRIDE
    inner_maxy = gy0 + bot_excl * STRIDE
    inner_bbox = box(inner_minx, inner_miny, inner_maxx, inner_maxy)
    inner_w = inner_maxx - inner_minx
    inner_h = inner_maxy - inner_miny

    # --- Choose (Nx, Ny): minimal area, but must FIT within inner bbox
    best = None
    max_try = max(1, int(math.ceil(math.sqrt(n_tiles)) * 3))
    for nx in range(1, max_try + 1):
        ny = math.ceil(n_tiles / nx)
        w = TILE + (nx - 1) * STRIDE
        h = TILE + (ny - 1) * STRIDE
        if w <= inner_w and h <= inner_h:
            area = w * h
            if best is None or area < best[0]:
                best = (area, nx, ny, w, h)

    if best is None:
        # Compute the maximum Nx, Ny, and total tiles that can fit within the inner box
        max_nx = int((inner_w - TILE) // STRIDE) + 1
        max_ny = int((inner_h - TILE) // STRIDE) + 1
        max_tiles = max_nx * max_ny if max_nx > 0 and max_ny > 0 else 0

        if max_tiles <= 0:
            raise RuntimeError(
                f"Inner bbox is {inner_w:.1f}×{inner_h:.1f} m. "
                f"No rectangle for n_tiles={n_tiles} (TILE=256, STRIDE={STRIDE:.1f}) fits. "
                "Reduce n_tiles or decrease buffer."
            )

        print(
            f"Requested {n_tiles} tiles cannot fit. "
            f"Reducing to max possible: {max_tiles} (Nx={max_nx}, Ny={max_ny})"
        )

        # Update n_tiles to max_tiles and retry finding best fit
        n_tiles = max_tiles

        # Re-run best-fit search using new tile count
        best = None
        max_try = max(1, int(math.ceil(math.sqrt(n_tiles)) * 3))
        for nx in range(1, max_try + 1):
            ny = math.ceil(n_tiles / nx)
            w = TILE + (nx - 1) * STRIDE
            h = TILE + (ny - 1) * STRIDE
            if w <= inner_w and h <= inner_h:
                area = w * h
                if best is None or area < best[0]:
                    best = (area, nx, ny, w, h)

        if best is None:
            raise RuntimeError(
                "Unexpected: even after adjustment, no fitting rectangle found."
            )

    _, nx, ny, req_w, req_h = best

    # --- Place the rectangle anchored at the inner bbox's SW corner (stride-aligned by construction)
    xmin = snap(inner_minx, res)
    ymin = snap(inner_miny, res)
    xmax = snap(xmin + req_w, res)
    ymax = snap(ymin + req_h, res)
    placed = box(xmin, ymin, xmax, ymax)

    print(f'placed: {placed}')

    # Final containment guard (should succeed by construction; keeps us safe with res rounding)
    if not placed.within(inner_bbox): # or not placed.within(trn_union):
        raise RuntimeError(
            "Placement failed containment after pixel snapping. "
            "Try a finer 'res' or adjust n_tiles/buffer."
        )

    # --- Save
    out_gdf = gpd.GeoDataFrame(geometry=[placed], crs=f"EPSG:{epsg}")
    out_path = Path(output_path).with_suffix(".geojson")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(out_path, driver="GeoJSON")
    return str(out_path)


def mergeTifs(inputPath, savePath, target_resolution=0.5):
    # If inputPath contains only a single zip file, unzip it and use the extracted contents
    if os.path.isdir(inputPath):
        items = [os.path.join(inputPath, item) for item in os.listdir(inputPath)]
        files = [item for item in items if os.path.isfile(item)]
        dirs = [item for item in items if os.path.isdir(item)]
        zip_files = [item for item in files if item.lower().endswith(".zip")]

        if len(zip_files) == 1 and len(files) == 1 and len(dirs) == 0:
            zip_path = zip_files[0]
            extract_dir = os.path.join(inputPath, "unzipped_contents")

            # Optional: clear old extracted contents before re-extracting
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)

            print(f"Only zip found in inputPath. Extracting: {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            inputPath = extract_dir
            print(f"Using extracted contents from: {inputPath}")

    # Identify each subfolder within this folder
    subfolders = [f.path for f in os.scandir(inputPath) if f.is_dir()]

    os.makedirs(savePath, exist_ok=True)

    for subfolder in subfolders:
        tif_files = []
        metadata_xml = ""

        # Recursively search for all tif tiles and metadata xml in the subfolder
        for root, _, files in os.walk(subfolder):
            for file in files:
                if file.lower().endswith(".tif") and "P" in root:
                    tif_files.append(os.path.join(root, file))

                if file.lower().endswith(".xml") and "P" in root and "tif" not in file.lower():
                    metadata_xml = os.path.join(root, file)

        if not tif_files:
            print(f"No TIF files found in the subfolder: {subfolder}")
            continue

        # Parse metadata for tile ID
        tileID = ""
        if metadata_xml:
            print(f"metadata_xml: {metadata_xml}")
            tree = ET.parse(metadata_xml)
            root = tree.getroot()
            catid_elem = root.find(".//IMD/IMAGE/CATID")
            if catid_elem is not None:
                tileID = catid_elem.text
                print(f"tileID: {tileID}")
            else:
                print("catid_elem is none!")
        else:
            print("METADATA FILE NOT FOUND")

        # Output filename
        new_savePath = os.path.join(savePath, f"{tileID}.tif")

        # Merge the TIFs
        mosaic, out_trans = merge(tif_files)

        with rasterio.open(tif_files[0]) as src:
            src_crs = src.crs
            src_dtype = src.dtypes[0]
            count = src.count

        # Calculate aligned bounds
        left, top = out_trans.c, out_trans.f
        right = left + mosaic.shape[2] * out_trans.a
        bottom = top + mosaic.shape[1] * out_trans.e

        # Align bounds to target grid
        aligned_left = np.floor(left / target_resolution) * target_resolution
        aligned_bottom = np.floor(bottom / target_resolution) * target_resolution
        aligned_right = np.ceil(right / target_resolution) * target_resolution
        aligned_top = np.ceil(top / target_resolution) * target_resolution

        # New dimensions
        dst_width = int((aligned_right - aligned_left) / target_resolution)
        dst_height = int((aligned_top - aligned_bottom) / target_resolution)

        # New transform
        dst_transform = rasterio.transform.from_origin(
            aligned_left,
            aligned_top,
            target_resolution,
            target_resolution
        )

        # Metadata for the resampled output
        dst_meta = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": count,
            "dtype": src_dtype,
            "crs": src_crs,
            "transform": dst_transform
        }

        # Prepare destination array
        dst_array = np.zeros((count, dst_height, dst_width), dtype=src_dtype)

        # Resample
        for i in range(count):
            reproject(
                source=mosaic[i],
                destination=dst_array[i],
                src_transform=out_trans,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=src_crs,
                resampling=Resampling.bilinear
            )

        # Save the aligned and resampled raster
        with rasterio.open(new_savePath, "w", **dst_meta) as dst:
            dst.write(dst_array)

        print(f"Merged and resampled TIF saved to: {new_savePath}")


def checkCRS(directory, epsg):
    """
    Checks the Coordinate Reference System (CRS) of all TIF files in a given directory.
    
    Parameters:
    directory (str): The path to the directory containing the TIF files.
    epsg (str): The expected EPSG code for the CRS.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".tif") or filename.endswith(".TIF"):
            tifPath = os.path.join(directory, filename)
            with rasterio.open(tifPath) as src:
                crs = src.crs
                assert crs.is_epsg_code, "The CRS is not in EPSG format."

                actualEPSG = crs.to_epsg()
                assert str(actualEPSG) == str(epsg), f"Expected EPSG code: {epsg}, Actual EPSG code: {actualEPSG}"
                print(f"EPSG code for {filename} is correct: {actualEPSG}")

def cropTif(inputTif, shp, outputTif, epsg):
    # Load the lidar shapefile
    lidar_shape = gpd.read_file(shp)

    # Check the CRS
    target_crs = CRS.from_epsg(int(epsg))
    if lidar_shape.crs != target_crs:
        print(f"Reprojecting from {lidar_shape.crs} to EPSG:{epsg}")
        lidar_shape = lidar_shape.to_crs(target_crs)
    else:
        print(f"Shapefile is already in EPSG:{epsg}")

    # Get the bounding box of the reprojected shapefile
    minx, miny, maxx, maxy = lidar_shape.total_bounds

    print(f'inputTif: {inputTif}')

    # First, crop the raster to the bounding box
    cropCmd = [
        "gdal_translate", "-projwin",
        str(minx), str(maxy), str(maxx), str(miny),
        inputTif, outputTif
    ]
    subprocess.run(cropCmd, check=True)

    # # Then, perform the mask on the cropped raster
    # warpCmd = [
    #     "gdalwarp", "-overwrite", "-of", "GTiff",
    #     "-tr", "0.5", "-0.5", "-tap",
    #     "-cutline", shp,
    #     inputTif, outputTif
    # ]
    # subprocess.run(warpCmd, check=True)

    # Remove the temporary cropped file
    # os.remove("temp_cropped.tif")

def saveWvimgMetadata(wvimgPath, savePath, prewvimgPath):
    metadata_dict = {}

    # Walk through the directory
    for root, dirs, files in os.walk(wvimgPath):
        for file in files:
            if file.endswith('XML') and 'PAN' in os.path.abspath(os.path.join(root, file)):
                full_path = os.path.join(root, file)
                x_vals = []
                y_vals = []
                
                # Parse the XML file
                tree = ET.parse(full_path)
                root = tree.getroot()

                # Extract the required information
                first_line_time = root.find('.//FIRSTLINETIME').text
                date = datetime.strptime(first_line_time[:10], '%Y-%m-%d').strftime('%Y-%m-%d')

                off_nadir = float(root.find('.//MEANOFFNADIRVIEWANGLE').text)
                target_azimuth = float(root.find('.//MEANSATAZ').text)
                solar_azimuth = float(root.find('.//MEANSUNAZ').text)
                solar_elevation = float(root.find('.//MEANSUNEL').text)
                tileID = root.find('.//IMD/IMAGE/CATID').text
                print(f'tileID: {tileID}')

                # extract the bounds of the tile (UTM)
                x_vals.append(float(root.find('.//IMD/MAP_PROJECTED_PRODUCT/ULX').text))
                x_vals.append(float(root.find('.//IMD/MAP_PROJECTED_PRODUCT/URX').text))
                x_vals.append(float(root.find('.//IMD/MAP_PROJECTED_PRODUCT/LLX').text))
                x_vals.append(float(root.find('.//IMD/MAP_PROJECTED_PRODUCT/LRX').text))

                y_vals.append(float(root.find('.//IMD/MAP_PROJECTED_PRODUCT/ULY').text))
                y_vals.append(float(root.find('.//IMD/MAP_PROJECTED_PRODUCT/URY').text))
                y_vals.append(float(root.find('.//IMD/MAP_PROJECTED_PRODUCT/LLY').text))
                y_vals.append(float(root.find('.//IMD/MAP_PROJECTED_PRODUCT/LRY').text))

                min_x = min(x_vals)
                min_y = min(y_vals)
                max_x = max(x_vals)
                max_y = max(y_vals)

                # Store the information in the dictionary
                metadata_dict[tileID] = {
                    'filePath': os.path.join(prewvimgPath, f'{tileID}.tif'),
                    'date': date,
                    'offNadir': off_nadir,
                    'targetAzimuth': target_azimuth,
                    'solarAzimuth': solar_azimuth,
                    'solarElevation': solar_elevation,
                    'min_x': min_x,
                    'min_y': min_y,
                    'max_x': max_x,
                    'max_y': max_y
                }

    dir_ = os.path.dirname(savePath)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

    # Save the metadata dictionary to a JSON file
    with open(savePath, 'w') as f:
        json.dump(metadata_dict, f, indent=4)


def generate_sensor_solar_tiles(metadata_json_path, output_directory):
    """
    Generate sensor and solar tiles for all elements in the metadata JSON file.
    
    :param metadata_json_path: Path to the JSON file containing metadata
    :param output_directory: Directory to save the output files
    """
    # Create 'sensor' and 'solar' subdirectories within the output directory
    sensor_dir = os.path.join(output_directory, 'sensor')
    solar_dir = os.path.join(output_directory, 'solar')
    os.makedirs(sensor_dir, exist_ok=True)
    os.makedirs(solar_dir, exist_ok=True)

    # Load metadata from file
    with open(metadata_json_path, 'r') as file:
        metadata = json.load(file)

    solar_max_global = float('-inf')
    sensor_max_global = float('-inf')

    # First pass to find global max values
    for file_path, data in metadata.items():
        _, solarNormal = getSolarNormal(data)
        _, sensorNormal = getSensorNormal(data)
        
        solar_max_global = max(solar_max_global, np.max(np.abs(solarNormal)))
        sensor_max_global = max(sensor_max_global, np.max(np.abs(sensorNormal)))

    solar_scale = solar_max_global / 128
    sensor_scale = 128 / sensor_max_global

    # Second pass to generate and save tiles
    for tileID, data in metadata.items():
        # get the tileID (key of the metadata)
        _, solarNormal = getSolarNormal(data)
        _, sensorNormal = getSensorNormal(data)

        solarNormal = (solarNormal / solar_scale + 128).astype(np.uint8)
        sensorNormal = (sensorNormal * sensor_scale + 128).astype(np.uint8)

        # Extract site name and date from file path
        site = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(data['filePath']))))
        date = datetime.strptime(data['date'], '%Y-%m-%d').strftime('%Y-%m-%d')

        # Save images in their respective subdirectories
        Image.fromarray(solarNormal).save(os.path.join(solar_dir, f"{tileID}.tif"))
        Image.fromarray(sensorNormal).save(os.path.join(sensor_dir, f"{tileID}.tif"))


def getSolarNormal(data):
    theta = data['solarElevation']
    phi = data['solarAzimuth']
    z0 = 10
    x0 = np.sqrt(z0**2 / (((np.tan(np.radians(theta)))**2) + (((np.tan(np.radians(theta)))**2) / (np.tan(np.radians(phi)))**2)))
    if phi > 180:
        x0 = x0 * -1
    y0 = x0 / np.tan(np.radians(phi))
    s = np.sqrt(x0**2 + y0**2 + z0**2)
    xn = x0 / s
    yn = y0 / s
    zn = z0 / s
    xg, yg = np.meshgrid(np.arange(-127.75, 128.25, 0.5), np.arange(-127.75, 128.25, 0.5))
    yg = np.flipud(yg)
    zg = -xn/zn * xg + -yn/zn * yg
    normal = zg
    return data, normal

def getSensorNormal(data):
    theta = 90 - data['offNadir']
    phi = (data['targetAzimuth']+180)%360
    z0 = 10
    x0 = np.sqrt(z0**2 / (((np.tan(np.radians(theta)))**2) + (((np.tan(np.radians(theta)))**2) / (np.tan(np.radians(phi)))**2)))
    if phi > 180:
        x0 = x0 * -1
    y0 = x0 / np.tan(np.radians(phi))
    s = np.sqrt(x0**2 + y0**2 + z0**2)
    xn = x0 / s
    yn = y0 / s
    zn = z0 / s
    xg, yg = np.meshgrid(np.arange(-127.75, 128.25, 0.5), np.arange(-127.75, 128.25, 0.5))
    yg = np.flipud(yg)
    zg = -xn/zn * xg + -yn/zn * yg
    normal = zg
    return data, normal

def snap(x, res):
    # snap coordinate to nearest multiple of res
    return math.ceil(round(x / res) * res)

def process_tile(tile_info, pathToRaster, outputPath, crs, tileSize, res, datatype):
    i, j, left, top, stride_x, stride_y = tile_info

    # Compute tile origin using stride (NOT tileSize)
    tile_left = left + i * stride_x
    tile_top  =  top - j * stride_y
    tile_right  = tile_left + tileSize
    tile_bottom = tile_top  - tileSize

    # Snap to the pixel grid to avoid fractional pixels
    tile_left   = snap(tile_left,   res)
    tile_right  = snap(tile_right,  res)
    tile_top    = snap(tile_top,    res)
    tile_bottom = snap(tile_bottom, res)

    bbox = f"{tile_left} {tile_bottom} {tile_right} {tile_top}"

    # Name tiles by their upper-left corner
    if datatype == 'wvimg':
        outfile = os.path.join(
            outputPath,
            f"{os.path.splitext(os.path.basename(pathToRaster))[0]}_{int(tile_left)}_{int(tile_top)}.tif"
        )
    else:
        outfile = os.path.join(outputPath, f"{int(tile_left)}_{int(tile_top)}.tif")

    # keep output size fixed via bounds + res (produces 512x512 given 256m and 0.5m/px)
    projcmd = (
        f"rio warp {pathToRaster} {outfile} "
        f"--dst-crs {crs.to_string()} "
        f"--bounds {bbox} "
        f"--res {res} "
        f"--resampling cubic "
        f"--overwrite"
    )

    devnull = open(os.devnull, 'w')
    subprocess.call(projcmd, shell=True, stdout=devnull, stderr=devnull)

    # Optional quality checks
    with rasterio.open(outfile) as tile:
        data = tile.read()
        nodata_value = tile.nodata
        if datatype == 'wvimg':
            nodata_value = 0

        if nodata_value is not None:
            contains_nodata = (data == nodata_value).any()
            if contains_nodata:
                print(f"Tile {os.path.basename(outfile)} contains NoData ({nodata_value}).")
                os.remove(outfile)
                if datatype.lower() == 'chm':
                    print(f'REJECTING A CHM TILE WITH NO DATA')
                return False

    return True

def tileRaster(
    pathToRaster: str,
    outputPath: str,
    dataType: str,
    tileSize: int = 256,
    anchors_csv: str | Path = None,
    res: float = 0.5,
):
    """
    pathToRaster: input raster
    outputPath: output folder
    dataType: 'wvimg' or 'chm'
    tileSize: width/height in meters
    anchors_csv: path to CSV with columns ["X","Y"] for NW-corner anchors (in raster CRS)
    res: target pixel size (m/px)
    """
    if anchors_csv is None:
        raise ValueError("anchors_csv is required and must point to a CSV with columns ['X','Y'].")

    anchors_csv = Path(anchors_csv)
    if not anchors_csv.exists():
        raise FileNotFoundError(f"Anchors CSV not found: {anchors_csv}")

    # Load and validate anchors
    anchors = pd.read_csv(anchors_csv)
    if not {"X", "Y"}.issubset(anchors.columns):
        raise ValueError(f"Anchors CSV must contain columns ['X','Y']; got {list(anchors.columns)}")

    # Clean up anchors a bit
    anchors = anchors[["X", "Y"]].dropna().drop_duplicates()
    anchors["X"] = anchors["X"].astype(float)
    anchors["Y"] = anchors["Y"].astype(float)

    os.makedirs(outputPath, exist_ok=True)

    # Detect number of CPU cores
    num_cores = multiprocessing.cpu_count()
    num_workers = max(1, num_cores - 2)

    with rasterio.open(pathToRaster) as src:
        src_left, src_bottom, src_right, src_top = src.bounds
        crs = src.crs

        if crs is None or getattr(crs, "is_geographic", False):
            raise ValueError("Raster must be in a projected CRS (meters).")

        # Define tiles directly from anchors
        tile_info_list = []
        for x, y in anchors[["X", "Y"]].itertuples(index=False, name=None):
            x_left   = snap(x, res)
            y_top    = snap(y, res)
            x_right  = x_left + tileSize
            y_bottom = y_top  - tileSize

            # Skip anchors that would produce a partial tile outside raster
            if (x_left < src_left) or (x_right > src_right) or (y_bottom < src_bottom) or (y_top > src_top):
                continue

            # pass (0,0,left,top,0,0) → ensures process_tile uses left/top as-is
            tile_info_list.append((0, 0, x_left, y_top, 0.0, 0.0))

    process_tile_partial = partial(
        process_tile,
        pathToRaster=pathToRaster,
        outputPath=outputPath,
        crs=crs,
        tileSize=tileSize,
        res=res,
        datatype=dataType
    )

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(process_tile_partial, tile_info_list)

    return results

def findIntTiles(metadataPath, lidarShapePath, epsg):
    tiles = []

    # Load the lidar shapefile
    lidar_shape = gpd.read_file(lidarShapePath)

    # Check the CRS
    target_crs = CRS.from_epsg(int(epsg))
    if lidar_shape.crs != target_crs:
        print(f"Reprojecting from {lidar_shape.crs} to EPSG:{epsg}")
        lidar_shape = lidar_shape.to_crs(target_crs)
    else:
        print(f"Shapefile is already in EPSG:{epsg}")

    # Get the geometry of the shapefile
    lidar_geometry = lidar_shape.geometry.unary_union

    # Load metadata
    with open(metadataPath, 'r') as f:
        metadata = json.load(f)

    # Check each metadata tile for intersection with the shapefile geometry
    for tile_id, tile_data in metadata.items():
        tile_minx = tile_data["min_x"]
        tile_maxx = tile_data["max_x"]
        tile_miny = tile_data["min_y"]
        tile_maxy = tile_data["max_y"]
        print(f'Tile {tile_id} bounds: {tile_minx}, {tile_miny}, {tile_maxx}, {tile_maxy}')

        # Create a box geometry for the tile
        tile_box = box(tile_minx, tile_miny, tile_maxx, tile_maxy)

        # Check if the tile intersects with the shapefile geometry
        if lidar_geometry.intersects(tile_box):
            tiles.append(tile_id)

    return tiles

def resolveOverlaps(wvimgInfPath, metadataPath):
    # Load metadata
    with open(metadataPath, 'r') as f:
        metadata = json.load(f)

    # Create a list of tiles with their geometries and dates
    tiles = []
    for tile_id, tile_data in metadata.items():
        tile_box = box(tile_data['min_x'], tile_data['min_y'], tile_data['max_x'], tile_data['max_y'])
        tile_date = datetime.strptime(tile_data['date'], '%Y-%m-%d')
        tiles.append((tile_id, tile_box, tile_date, tile_data['filePath']))

    # Find pairs of tiles that overlap
    for tile1, tile2 in combinations(tiles, 2):
        if tile1[1].intersects(tile2[1]):
            # Determine which tile is newer
            if tile1[2] > tile2[2]:
                newer_tile, older_tile = tile1, tile2
            else:
                newer_tile, older_tile = tile2, tile1

            # Calculate the difference between the older tile and the newer tile
            difference = older_tile[1].difference(newer_tile[1])

            # If there's a difference (i.e., the older tile is not completely covered)
            if not difference.is_empty:
                # Prepare the gdalwarp command
                input_file = os.path.join(wvimgInfPath, older_tile[3])
                
                # Create a temporary file in the same directory
                output_file = input_file + '_temp.tif'
                
                # Create a WKT string for the difference geometry
                wkt = difference.wkt
                
                # Construct the gdalwarp command
                command = [
                    'gdalwarp',
                    '-cutline', wkt,
                    '-crop_to_cutline',
                    '-overwrite',
                    input_file,
                    output_file
                ]

                # Execute the gdalwarp command
                try:
                    subprocess.run(command, check=True)
                    print(f"Successfully cropped {older_tile[0]}")
                    
                    # Replace the original file with the cropped version
                    os.remove(input_file)
                    os.rename(output_file, input_file)
                    print(f"Replaced original file for {older_tile[0]}")
                except subprocess.CalledProcessError as e:
                    print(f"Error cropping {older_tile[0]}: {e}")
                    # Clean up the temporary file if an error occurred
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except OSError as e:
                    print(f"Error replacing file for {older_tile[0]}: {e}")
                    # Clean up the temporary file if an error occurred
                    if os.path.exists(output_file):
                        os.remove(output_file)


def renameTiles(ms_data_path):
    wvimg_path = os.path.join(ms_data_path, "wvimg")
    dem_path = os.path.join(ms_data_path, "dem")
    lidar_path = os.path.join(ms_data_path, "chm")

    # Helper to rename or symlink tiles in target_dir based on wvimg
    def process_against_wvimg(target_dir):
        for wvimg_file in os.listdir(wvimg_path):
            if '_' not in wvimg_file:
                print('_ not in wvimg_file')
                continue

            tile_id, rest = wvimg_file.split('_', 1)
            reference_name = f"{tile_id}_{rest}"
            # Find matching tiles in DEM or lidar that include the tile_id
            matches = [f for f in os.listdir(target_dir) if rest in f]

            for match in matches:
                match_path = os.path.join(target_dir, match)

                if not os.path.isfile(match_path):
                    continue

                underscore_count = match.count('_')

                # If not renamed: do so
                if underscore_count == 1:
                    new_name = f"{tile_id}_{match}"
                    new_path = os.path.join(target_dir, new_name)
                    os.rename(match_path, new_path)

                # If already renamed: make symlink to avoid duplicates
                elif underscore_count >= 2:
                    existing_file = match_path
                    base_name = '_'.join(match.split('_')[1:])
                    symlink_name = os.path.join(target_dir, f"{tile_id}_{base_name}")
                    if not os.path.exists(symlink_name):
                        # Create relative symlink
                        link_target = os.path.relpath(existing_file, os.path.dirname(symlink_name))
                        os.symlink(link_target, symlink_name)

    # Process DEM and lidar using wvimg as reference
    process_against_wvimg(dem_path)
    # Only rename lidar data if it exists (not inference)
    if os.path.isdir(lidar_path):
        process_against_wvimg(lidar_path)

    # # Clean up any files in DEM and lidar that don't have exactly 2 underscores
    # for cleanup_dir in [dem_path, lidar_path]:
    #     for fname in os.listdir(cleanup_dir):
    #         full_path = os.path.join(cleanup_dir, fname)
    #         if os.path.isfile(full_path) and fname.count('_') != 2:
    #             os.remove(full_path)

def makeLists(base_dir: str, site, random_seed=42):
    """
    Args:
        base_dir: Base directory containing subdirectories
        site: Site name for output files
        random_seed: Seed for random number generator (default=42)
    """
    # Define the subdirectories
    subdirs = ['dem', 'chm', 'wvimg']
    
    # Get the set of files for each subdirectory
    file_sets = []
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if os.path.isdir(path):
            files = set(f for f in os.listdir(path) 
                        if f.lower().endswith('.tif') and not f.lower().endswith('.tif.aux.xml'))
            file_sets.append(files)
    
    # Find the intersection of all file sets
    common_files = set.intersection(*file_sets)
    
    # IMPORTANT: Sort first to ensure consistent order, THEN shuffle
    common_files_list = sorted(list(common_files))
    print(f'size of set intersection: {len(common_files_list)}')
    
    # Set random seed and shuffle
    random.seed(random_seed)
    random.shuffle(common_files_list)
    
    # Calculate split sizes
    total_files = len(common_files_list)
    train_size = int(0.7 * total_files)
    val_size = int(0.15 * total_files)
    
    # Split the files
    train_files = common_files_list[:train_size]
    val_files = common_files_list[train_size:train_size+val_size]
    test_files = common_files_list[train_size+val_size:]
    
    # Write to output files
    write_to_file(train_files, os.path.join(base_dir, f'{site}_trainlist.txt'))
    write_to_file(val_files, os.path.join(base_dir, f'{site}_vallist.txt'))
    write_to_file(test_files, os.path.join(base_dir, f'{site}_testlist.txt'))

def write_to_file(file_list: List[str], output_file: str):
    with open(output_file, 'w') as f:
        for file_name in file_list:
            f.write(f"{file_name}\n")

def get_max_difference_in_dir(tif_dir):
    max_diff = -np.inf
    max_file = None

    for filename in os.listdir(tif_dir):
        if filename.lower().endswith(".tif"):
            filepath = os.path.join(tif_dir, filename)
            try:
                with rasterio.open(filepath) as src:
                    data = src.read(1, masked=True)
                    if data.mask.all():
                        continue
                    local_max = data.max()
                    local_min = data.min()
                    diff = local_max - local_min
                    if diff > max_diff:
                        max_diff = diff
                        max_file = filename
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return max_diff, max_file


def normalize_and_save_tile(src_path, dst_path, global_max_range):
    with rasterio.open(src_path) as src:
        profile = src.profile
        data = src.read(1, masked=True)

        if data.mask.all():
            return None, None

        local_min = data.min()
        local_max = data.max()
        local_range = local_max - local_min
        midpoint = (local_min + local_max) / 2.0

        # Each pixel is shifted based on deviation from midpoint
        scale = 127.0 / (global_max_range / 2.0)  # 127 steps from midpoint to either edge

        # Shift values so midpoint = 128
        pixel_data = 128 + ((data - midpoint) * scale)

        # Clip and convert to uint8
        pixel_data = np.clip(pixel_data, 0, 255).astype(np.uint8)

        # Update profile for uint8 image
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)

        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(pixel_data.filled(0), 1)

        return local_range, os.path.basename(dst_path)
    

def normDEMs(src_path, dst_path):
    global_max_range, max_range_file = get_max_difference_in_dir(src_path)

    for filename in os.listdir(src_path):
        if filename.lower().endswith(".tif"):
            src_tif_path = os.path.join(src_path, filename)
            dst_tif_path = os.path.join(dst_path, filename)
            try:
                diff, norm_file = normalize_and_save_tile(src_tif_path, dst_tif_path, global_max_range)
            except Exception as e:
                print(f"Error normalizing {filename}: {e}")

def findLidarResources(
    train_shp: str | Path,
    catalog_geojson: str | Path | dict,
    return_full: bool = False,
    verbose: bool = False
) -> List[Union[str, Dict[str, Any]]]:
    """
    Search a catalog GeoJSON (FeatureCollection) for resources whose
    geometries intersect the AOI. No name/state filtering is applied.

    Parameters
    ----------
    train_shp : path to AOI vector (any format readable by GeoPandas)
    catalog_geojson : path to the catalog GeoJSON *or* an already-loaded dict
    return_full : if True, return list of dicts with name/url/id/count/bounds; else list of URLs
    verbose : print a few debug lines

    Returns
    -------
    List[str]            (URLs)                     if return_full=False (default)
    List[Dict[str, Any]] (name/url/id/count/bounds) if return_full=True
    """

    # --- 1) Load AOI and normalize to EPSG:4326
    aoi = gpd.read_file(train_shp)
    if aoi.crs is None:
        raise ValueError("AOI has no CRS defined. Set it (aoi.set_crs) before running.")
    aoi_4326 = aoi.to_crs(4326)
    aoi_union = aoi_4326.union_all() if hasattr(aoi_4326, "union_all") else aoi_4326.unary_union

    # --- 2) Load the catalog GeoJSON (path or preloaded dict)
    if isinstance(catalog_geojson, (str, Path)):
        with open(catalog_geojson, "r") as f:
            obj = json.load(f)
    elif isinstance(catalog_geojson, dict):
        obj = catalog_geojson
    else:
        raise TypeError("catalog_geojson must be a path or a dict-like object.")

    if not isinstance(obj, dict) or obj.get("type") != "FeatureCollection":
        raise ValueError("Catalog must be a GeoJSON FeatureCollection.")

    feats = []
    for ft in obj.get("features", []):
        geom = ft.get("geometry")
        props = ft.get("properties", {}) or {}
        if not geom:
            continue
        try:
            shp = shape(geom)
            if shp.is_empty:
                continue
            feats.append({
                "name": props.get("name"),
                "url": props.get("url"),
                "id": props.get("id"),
                "count": props.get("count"),
                "geometry": shp
            })
        except Exception:
            # skip malformed features
            continue

    if not feats:
        return []

    gdf = gpd.GeoDataFrame(feats, geometry="geometry", crs=4326)

    # --- 3) Spatial prefilter via bbox index, then exact intersects
    try:
        idx = gdf.sindex.query(aoi_union, predicate="intersects")
        candidates = gdf.iloc[idx]
    except Exception:
        # fallback without spatial index
        aoi_bbox_poly = box(*aoi_union.bounds)
        candidates = gdf[gdf.intersects(aoi_bbox_poly)]

    hits = candidates[candidates.intersects(aoi_union)]
    if hits.empty:
        return []

    if verbose:
        print(f"AOI bounds: {aoi_union.bounds}")
        print(f"Found {len(hits)} intersecting resources in catalog.")
        print(hits[["name", "url"]].head())

    if return_full:
        out = []
        for _, r in hits.iterrows():
            minx, miny, maxx, maxy = r.geometry.bounds
            out.append({
                "name": r["name"],
                "url": r["url"],
                "id": r["id"],
                "count": r["count"],
                "bounds": (minx, miny, maxx, maxy),
            })
        return out

    # default: list of URLs (strings)
    return hits["url"].dropna().tolist()

def genTileAnchors(
    shp_path: str,
    out_path: str | Path,
    *,
    buffer: int = 16,
    margin: int = 256,
    target_epsg: int | None = None,
    include_boundary: bool = True,
) -> None:
    """
    Create a grid of snapped points inside a polygonal area and write them to `out_path`.

    Grid points are placed at coordinates divisible by `spacing = 256 - 2*buffer`
    (e.g., spacing=224 for buffer=16), and only if they are at least 256 meters
    away from the AOI boundary (i.e., inside AOI buffered inward by 256).

    Parameters
    ----------
    shp_path : str
        Path to the AOI vector (anything GeoPandas can read). Geometry must be polygonal.
    out_path : str | Path
        Output file path. Extension determines format:
          - .csv -> CSV with columns X,Y
          - .geojson/.json/.gpkg/.shp -> vector points with CRS
    buffer : int
        Tile halo in meters (per edge). spacing = 256 - 2*buffer. Default 16.
    target_epsg : int | None
        If provided, AOI is reprojected before gridding.
    include_boundary : bool
        If True, include points on the *inner eroded* boundary (the 256m-inset edge).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    # --- Compute spacing and margin-to-edge
    spacing = 256 - 2 * buffer
    if spacing <= 0:
        raise ValueError(f"Buffer too large ({buffer}); spacing must stay positive.")

    # --- Load AOI
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        if ext == ".csv":
            pd.DataFrame(columns=["X", "Y"]).to_csv(out_path, index=False)
        elif ext in (".geojson", ".json"):
            out_path.write_text('{"type":"FeatureCollection","features":[]}')  # minimal FC
        return

    # Optional reprojection
    if target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)

    # Require projected CRS (meters)
    if gdf.crs is None or getattr(gdf.crs, "is_geographic", False):
        raise ValueError(
            "Input must be in a projected CRS (meters). "
            "Use target_epsg (e.g., 32617) or reproject before calling."
        )

    # Union geometry
    geom = gdf.unary_union
    if geom.is_empty:
        if ext == ".csv":
            pd.DataFrame(columns=["X", "Y"]).to_csv(out_path, index=False)
        elif ext in (".geojson", ".json"):
            out_path.write_text('{"type":"FeatureCollection","features":[]}') 
        return

    # --- Erode AOI by fixed 256m margin
    inner = geom.buffer(-margin)
    try:
        inner = inner.buffer(0)  # clean artifacts
    except Exception:
        pass

    if inner.is_empty:
        # Nothing remains after erosion → no anchors
        if ext == ".csv":
            pd.DataFrame(columns=["X", "Y"]).to_csv(out_path, index=False)
        elif ext in (".geojson", ".json"):
            out_path.write_text('{"type":"FeatureCollection","features":[]}') 
        return

    # --- Generate a global spacing grid (coordinates divisible by `spacing`)
    # Use inner bounds to limit the loops, but every candidate coord is k*spacing.
    minx, miny, maxx, maxy = inner.bounds

    # k ranges so that x = k*spacing spans [minx, maxx]
    kx0 = int(np.ceil(minx / spacing))
    kx1 = int(np.floor(maxx / spacing))
    ky0 = int(np.ceil(miny / spacing))
    ky1 = int(np.floor(maxy / spacing))

    if kx1 < kx0 or ky1 < ky0:
        if ext == ".csv":
            pd.DataFrame(columns=["X", "Y"]).to_csv(out_path, index=False)
        elif ext in (".geojson", ".json"):
            out_path.write_text('{"type":"FeatureCollection","features":[]}') 
        return

    xs = (np.arange(kx0, kx1 + 1, dtype=np.int64) * spacing).astype(float)
    ys = (np.arange(ky0, ky1 + 1, dtype=np.int64) * spacing).astype(float)

    # Mesh → coords (all coords are multiples of spacing by construction)
    X, Y = np.meshgrid(xs, ys)
    coords = np.column_stack((X.ravel(), Y.ravel()))

    # --- Keep only points inside the 256m-eroded AOI
    pts = gpd.GeoSeries(gpd.points_from_xy(coords[:, 0], coords[:, 1]), crs=gdf.crs)

    mask = None
    try:
        # shapely >= 2 vectorized
        if hasattr(shapely, "covers"):
            mask = shapely.covers(inner, pts.array) if include_boundary else shapely.contains(inner, pts.array)
    except Exception:
        pass
    if mask is None:
        # Fallback predicates
        mask = pts.within(inner) | pts.touches(inner) if include_boundary else pts.within(inner)

    inside = coords[mask]
    if inside.size == 0:
        if ext == ".csv":
            pd.DataFrame(columns=["X", "Y"]).to_csv(out_path, index=False)
        elif ext in (".geojson", ".json"):
            out_path.write_text('{"type":"FeatureCollection","features":[]}') 
        return

    # --- Write outputs
    if ext == ".csv":
        pd.DataFrame(inside, columns=["X", "Y"]).to_csv(out_path, index=False)
    else:
        anchors_gdf = gpd.GeoDataFrame(
            {"X": inside[:, 0], "Y": inside[:, 1]},
            geometry=gpd.points_from_xy(inside[:, 0], inside[:, 1]),
            crs=gdf.crs,
        )
        if ext in (".geojson", ".json"):
            anchors_gdf.to_file(out_path, driver="GeoJSON")
        elif ext == ".gpkg":
            anchors_gdf.to_file(out_path, driver="GPKG", layer=out_path.stem)
        elif ext == ".shp":
            anchors_gdf.to_file(out_path, driver="ESRI Shapefile")
        else:
            anchors_gdf.to_file(out_path.with_suffix(".geojson"), driver="GeoJSON")

@dataclass(frozen=True)
class WorkerCfg:
    epsg: int
    ept_urls: Tuple[str, ...]
    output_dir: str
    tmp_laz_dir: str
    start_date: str
    end_date: str
    r_script_path: Optional[str] = None   # None => skip R entirely
    run_r: bool = False                   # default: LAZ-only mode
    r_timeout_sec: int = 600
    size_threshold_bytes: int = 3 * 1024  # skip tiny files
    buf: float = 20.0
    tile_size: float = 256.0
    resolution: float = 1.0


# --------------------------------
# Small helpers (pure functions)
# --------------------------------
def _normalize_url(u: str) -> str:
    return u if u.startswith("http") else f"https://{u}"


def _expected_tif_path(output_dir: str, x: float, y: float) -> str:
    return os.path.join(output_dir, f"{int(x)}_{int(y)}.tif")


def _choose_laz(laz_candidates: Dict[str, str]) -> str:
    # Deterministic pick: latest in lexical order by EPT key
    if len(laz_candidates) == 1:
        return next(iter(laz_candidates.values()))
    return sorted(laz_candidates.items(), key=lambda kv: kv[0])[-1][1]


# --------------------------------
# PDAL fetcher (runs in subprocess)
# --------------------------------
def laz(
    x: float,
    y: float,
    epsg: int,
    url: str,
    out_dir: str = ".",
    buf: float = 20.0,
    tile_size: float = 256.0,
    resolution: float = 1.0,
) -> str:
    """
    Fetch a buffered LAZ for a 256x256 m tile whose UPPER-LEFT is (x, y) in UTM.
    """
    utm = f"EPSG:{epsg}"
    ept_crs = "EPSG:3857"

    tile_utm = box(x, y - tile_size, x + tile_size, y)
    tile_utm_buf = gpd.GeoSeries([tile_utm], crs=utm).buffer(buf).iloc[0]

    tile_3857 = (
        gpd.GeoSeries([tile_utm], crs=utm)
        .to_crs(ept_crs)
        .buffer(buf)
        .iloc[0]
    )

    site_name = os.path.basename(os.path.dirname(urlparse(url).path)) or "ept"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    filename = os.path.abspath(os.path.join(out_dir, f"{int(x)}_{int(y)}_{site_name}.laz"))

    pipeline = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": str(url),
                "polygon": tile_3857.wkt,
                "resolution": resolution,
            },
            {
                "type": "filters.range",
                "limits": "Classification![7:7],Classification![18:18]",
            },
            {"type": "filters.reprojection", "out_srs": utm},
            {"type": "filters.crop", "polygon": tile_utm_buf.wkt},
            {"type": "writers.las", "compression": "laszip", "filename": filename},
        ]
    }

    pdal.Pipeline(json.dumps(pipeline)).execute_streaming(chunk_size=1_000_000)
    return filename


# --------------------------------
# Worker (must be top-level & picklable)
# --------------------------------
def _worker_point(args: Tuple[WorkerCfg, float, float]) -> Tuple[Tuple[float, float], str, Optional[str]]:
    """
    Returns ((x,y), status, message_or_path)
    status: "done" | "skip" | "error"
    message_or_path: reason (skip/error) OR chosen laz path (done)
    """
    cfg, x, y = args
    try:
        # print(f"[START] Point ({x:.2f}, {y:.2f})", flush=True)
        laz_candidates: Dict[str, str] = {}

        for ept in cfg.ept_urls:
            try:
                p = laz(
                    x=x, y=y, epsg=cfg.epsg, url=ept,
                    out_dir=cfg.tmp_laz_dir, buf=cfg.buf,
                    tile_size=cfg.tile_size, resolution=cfg.resolution
                )
                size = os.path.getsize(p)
                if size > cfg.size_threshold_bytes:
                    laz_candidates[ept] = p
                    # print(f"[FETCHED] {os.path.basename(p)} ({size} bytes)", flush=True)
                else:
                    os.remove(p)
                    # print(f"[EMPTY] Removed {os.path.basename(p)}", flush=True)
            except Exception as e:
                print(f"[ERROR] Fetch failed at ({x:.2f}, {y:.2f}) from {ept}: {e}", flush=True)

        if not laz_candidates:
            msg = f"No valid LAZ for ({x:.2f}, {y:.2f})"
            # print(f"[SKIP] {msg}", flush=True)
            return ((x, y), "skip", msg)

        chosen_laz = _choose_laz(laz_candidates)
        # print(f"[CHOSEN] {os.path.basename(chosen_laz)}", flush=True)

        try:
            proc = subprocess.run(
                [
                    "Rscript", "--vanilla", "R_chm_cli.R",
                    "chm_create",
                    "--las", chosen_laz,
                    "--epsg", str(cfg.epsg),
                    "--outdir", cfg.output_dir,
                ],
                text=True,
                check=False,
                capture_output=True,
                timeout=cfg.r_timeout_sec,
            )
            if proc.returncode != 0:
                print(f"[R ERROR] {proc.stderr.strip()}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[R TIMEOUT] ({x:.2f},{y:.2f}) exceeded {cfg.r_timeout_sec}s", flush=True)
        except Exception as e:
            print(f"[R FATAL] ({x:.2f},{y:.2f}): {e}", flush=True)

        # Cleanup any extra candidates (keep the chosen)
        for ept, p in laz_candidates.items():
            if p != chosen_laz:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception as ce:
                    print(f"[WARN] Could not remove {p}: {ce}", flush=True)

        # print(f"[DONE] Point ({x:.2f}, {y:.2f})", flush=True)
        return ((x, y), "done", chosen_laz)

    except Exception as e:
        tb = traceback.format_exc(limit=4)
        msg = f"Unhandled error at ({x:.2f},{y:.2f}): {e}\n{tb}"
        print(f"[FATAL] {msg}", flush=True)
        return ((x, y), "error", msg)


# --------------------------------
# Public API
# --------------------------------
def generate_chm_tiles(
    epsg: int,
    ept_urls: Iterable[str],
    points_df: pd.DataFrame,
    output_dir: str,
    r_script_path: Optional[str],  # keep in signature; pass None to skip
    *,
    run_r: bool = False,                 # default LAZ-only mode
    max_workers: Optional[int] = None,   # None => conservative default
    r_timeout_sec: int = 600,
    size_threshold_bytes: int = 3 * 1024,
    buf: float = 20.0,
    tile_size: float = 256.0,
    resolution: float = 1.0,
    start_date: str = "2000-01-01",
    end_date: str = "2025-01-01",
) -> List[Tuple[Tuple[float, float], str, Optional[str]]]:
    """
    Stable, process-based version.
    LAZ-only by default (run_r=False). Returns a list of per-point results:
    [ ((x,y), status, message_or_path), ... ]
    """
    output_dir = os.path.abspath(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tmp_laz_dir = os.path.join(output_dir, "_laz_tmp")
    Path(tmp_laz_dir).mkdir(parents=True, exist_ok=True)

    urls = tuple(_normalize_url(u) for u in ept_urls)

    # Conservative worker count (avoid I/O thrash)
    if max_workers is None:
        cores = mp.cpu_count()
        max_workers = max(1, min(6, cores // 3))  # e.g., 2 on 6 cores, 4 on 16 cores

    cfg = WorkerCfg(
        epsg=epsg,
        ept_urls=urls,
        output_dir=output_dir,
        tmp_laz_dir=tmp_laz_dir,
        start_date=start_date,
        end_date=end_date,
        r_script_path=r_script_path,
        run_r=run_r,
        r_timeout_sec=r_timeout_sec,
        size_threshold_bytes=size_threshold_bytes,
        buf=buf,
        tile_size=tile_size,
        resolution=resolution,
    )

    records = points_df[["X", "Y"]].dropna().to_records(index=False)
    tasks = [(cfg, float(x), float(y)) for (x, y) in records]

    results: List[Tuple[Tuple[float, float], str, Optional[str]]] = []

    # Use "spawn" for safety across platforms and C++ libs
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker_point, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                res = fut.result()  # surfaces exceptions from the worker
                results.append(res)
            except Exception as e:
                # This is extremely rare because _worker_point catches most errors
                results.append(((math.nan, math.nan), "error", f"Executor error: {e}"))

    return results

def createLidarData(train_shp, catalog_geojson, epsg, lidarTilesPath, anchors_csv, projectPath):
    # find intersecting lidar scan(s)

    lidarScans = findLidarResources(train_shp, catalog_geojson)
    # print(f'lidarScans: {lidarScans}')

    # download lidar tifs
    pathToRScript = os.path.join(projectPath, 'SatCHM', 'prepTrainInputs', 'Rutils.R')
    anchors = pd.read_csv(anchors_csv)
    generate_chm_tiles(epsg=epsg, ept_urls=lidarScans, points_df=anchors, output_dir=lidarTilesPath, r_script_path=pathToRScript)
    
    alignmentYear = round(sum(int(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', u)[-1]) for u in lidarScans) / len(lidarScans), 1)

    return alignmentYear


def merge_chm_tiles(
    input_folder: str,
    output_tif: str,
    b: float = 16.0,      # meters: feather fully "on" by b meters from an edge
    c: float = None,       # meters: fully "off" within c meters of an edge
    nodata: float = -9999 # output nodata
):
    """
    Merge georeferenced CHM tiles with edge feathering:
      - weight = 0 for pixels within c meters of any tile edge
      - weight ramps linearly 0→1 between c and b meters from the edge
      - weight = 1 for pixels ≥ b meters from every edge
    Overlaps are blended by weighted mean.

    Assumes all inputs share CRS and pixel size (square pixels).
    """

    if c is None:
        c = b / 4.0

    tif_paths = sorted(glob(os.path.join(input_folder, "*.tif")))
    if not tif_paths:
        raise ValueError(f"No GeoTIFFs found in {input_folder}")

    # Read base metadata
    with rasterio.open(tif_paths[0]) as ds0:
        crs = ds0.crs
        resx = abs(ds0.transform.a)
        resy = abs(ds0.transform.e)
        if not np.isclose(resx, resy):
            raise ValueError("Non-square pixels not supported.")
        px = resx  # meters per pixel

    # Validate consistency
    for p in tif_paths[1:]:
        with rasterio.open(p) as d:
            if d.crs != crs:
                raise ValueError(f"CRS mismatch: {p}")
            if not (np.isclose(abs(d.transform.a), resx) and np.isclose(abs(d.transform.e), resy)):
                raise ValueError(f"Resolution mismatch: {p}")

    # Union bounds
    def _bounds(path):
        with rasterio.open(path) as d:
            return d.bounds
    bounds_list = [_bounds(p) for p in tif_paths]
    minx = min(b.left for b in bounds_list)
    miny = min(b.bottom for b in bounds_list)
    maxx = max(b.right for b in bounds_list)
    maxy = max(b.top for b in bounds_list)

    width  = int(np.ceil((maxx - minx) / px))
    height = int(np.ceil((maxy - miny) / px))
    out_transform = from_origin(minx, maxy, px, px)

    sum_arr = np.zeros((height, width), dtype=np.float64)
    wsum_arr = np.zeros((height, width), dtype=np.float64)

    # Precompute denominator for ramp (avoid divide-by-zero)
    ramp_den = max(1e-6, (b - c))

    for path in tif_paths:
        with rasterio.open(path) as d:
            tile = d.read(1).astype(np.float64)

            # Map to output window
            win: Window = from_bounds(*d.bounds, transform=out_transform, width=width, height=height)
            win = win.round_offsets().round_lengths()
            r0, c0 = int(win.row_off), int(win.col_off)
            h, w = int(win.height), int(win.width)

            r1 = min(r0 + h, height)
            c1 = min(c0 + w, width)
            tile = tile[: (r1 - r0), : (c1 - c0)]

            th, tw = tile.shape

            # Distance-to-nearest-edge (in meters) at each pixel
            rr = np.minimum(np.arange(th), np.arange(th)[::-1])  # pixels to top/bottom
            cc = np.minimum(np.arange(tw), np.arange(tw)[::-1])  # pixels to left/right
            dist_edge_px = np.minimum(rr[:, None], cc[None, :]).astype(np.float64)
            dist_edge_m = dist_edge_px * px

            # Feathered weights: 0 (<=c), linear to 1 (>=b)
            w = (dist_edge_m - c) / ramp_den
            np.clip(w, 0.0, 1.0, out=w)

            # Respect source nodata if present
            if d.nodata is not None:
                w = np.where(tile == d.nodata, 0.0, w)

            # Accumulate
            sum_arr[r0:r1, c0:c1] += tile * w
            wsum_arr[r0:r1, c0:c1] += w

    # Final mosaic (weighted mean), else nodata
    out = np.full((height, width), nodata, dtype=np.float32)
    mask = wsum_arr > 0
    out[mask] = (sum_arr[mask] / wsum_arr[mask]).astype(np.float32)

    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": height,
        "width": width,
        "crs": crs,
        "transform": out_transform,
        "compress": "lzw",
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(out, 1)

    return output_tif

def merge_tiles_with_avg_height_feathered(
    input_folder: str,
    output_tif: str,
    threshold: float = 2.0,  # keep only pixels > threshold when computing each tile's average
    b: float = 16.0,         # meters: weight ramps up to 1 by this distance from edges
    c: float = None,          # meters: completely discard within this distance of edges
    nodata: float = -9999.0, # output NoData
    exclude_edges_in_average: bool = True  # ignore the outer c m when computing per-tile averages
):
    """
    Build a mosaic where each tile is represented by its own average CHM height
    (computed over pixels > `threshold`). The per-tile constant images are
    feather-blended across overlaps using the same c→b ramp as before.

    If `exclude_edges_in_average` is True, pixels within `c` meters of any tile edge
    are NOT used when computing each tile's average (to avoid edge artifacts).
    """

    if c is None:
        c = b / 4.0

    tif_paths = sorted(glob(os.path.join(input_folder, "*.tif")))
    if not tif_paths:
        raise ValueError(f"No GeoTIFFs found in {input_folder}")

    # Read base metadata
    with rasterio.open(tif_paths[0]) as ds0:
        crs = ds0.crs
        resx = abs(ds0.transform.a)
        resy = abs(ds0.transform.e)
        if not np.isclose(resx, resy):
            raise ValueError("Non-square pixels not supported.")
        px = resx  # meters per pixel

    # Validate consistency
    for p in tif_paths[1:]:
        with rasterio.open(p) as d:
            if d.crs != crs:
                raise ValueError(f"CRS mismatch: {p}")
            if not (np.isclose(abs(d.transform.a), resx) and np.isclose(abs(d.transform.e), resy)):
                raise ValueError(f"Resolution mismatch: {p}")

    # Union bounds
    def _bounds(path):
        with rasterio.open(path) as d:
            return d.bounds
    bounds_list = [_bounds(p) for p in tif_paths]
    minx = min(bd.left for bd in bounds_list)
    miny = min(bd.bottom for bd in bounds_list)
    maxx = max(bd.right for bd in bounds_list)
    maxy = max(bd.top for bd in bounds_list)

    width  = int(np.ceil((maxx - minx) / px))
    height = int(np.ceil((maxy - miny) / px))
    out_transform = from_origin(minx, maxy, px, px)

    # Accumulators for feather-blended mosaic of constant tiles
    sum_arr = np.zeros((height, width), dtype=np.float64)
    wsum_arr = np.zeros((height, width), dtype=np.float64)

    ramp_den = max(1e-6, (b - c))  # avoid div-by-zero

    for path in tif_paths:
        with rasterio.open(path) as d:
            tile = d.read(1).astype(np.float64)
            th, tw = tile.shape

            # Build mask for averaging: > threshold, not nodata, and (optionally) not in the outer c m ring
            avg_mask = (tile > threshold)
            if d.nodata is not None:
                avg_mask &= (tile != d.nodata)

            if exclude_edges_in_average and c > 0:
                rr = np.minimum(np.arange(th), np.arange(th)[::-1])
                cc = np.minimum(np.arange(tw), np.arange(tw)[::-1])
                dist_edge_px = np.minimum(rr[:, None], cc[None, :]).astype(np.float64)
                dist_edge_m = dist_edge_px * px
                avg_mask &= (dist_edge_m >= c)

            # Compute per-tile average height
            if not np.any(avg_mask):
                # No valid data > threshold: skip this tile entirely
                continue
            tile_avg = float(tile[avg_mask].mean())

            # Create a constant-image tile filled with tile_avg
            const_tile = np.full_like(tile, tile_avg, dtype=np.float64)

            # Feathering weights for blending this tile into the mosaic
            rr = np.minimum(np.arange(th), np.arange(th)[::-1])
            cc = np.minimum(np.arange(tw), np.arange(tw)[::-1])
            dist_edge_px = np.minimum(rr[:, None], cc[None, :]).astype(np.float64)
            dist_edge_m = dist_edge_px * px
            w = (dist_edge_m - c) / ramp_den
            np.clip(w, 0.0, 1.0, out=w)

            # Respect source nodata: give weight 0 where nodata OR <= threshold (optional)
            # (We generally let the whole tile contribute, but you could zero out <=threshold if desired.)
            if d.nodata is not None:
                w = np.where(tile == d.nodata, 0.0, w)

            # Map to output window
            win: Window = from_bounds(*d.bounds, transform=out_transform, width=width, height=height)
            win = win.round_offsets().round_lengths()
            r0, c0 = int(win.row_off), int(win.col_off)
            h, w_cols = int(win.height), int(win.width)
            r1, c1 = min(r0 + h, height), min(c0 + w_cols, width)

            const_tile = const_tile[: (r1 - r0), : (c1 - c0)]
            w = w[: (r1 - r0), : (c1 - c0)]

            # Accumulate
            sum_arr[r0:r1, c0:c1] += const_tile * w
            wsum_arr[r0:r1, c0:c1] += w

    # Final blended mosaic
    out = np.full((height, width), nodata, dtype=np.float32)
    mask = wsum_arr > 0
    out[mask] = (sum_arr[mask] / wsum_arr[mask]).astype(np.float32)

    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": height,
        "width": width,
        "crs": crs,
        "transform": out_transform,
        "compress": "lzw",
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(out, 1)

    return output_tif

def makeInfList(base_dir: str, site):
    # Define the subdirectories
    # subdirs = ['dem', 'wvimg', 'lidar']
    subdirs = ['dem', 'chm', 'wvimg']
    
    # Get the set of files for each subdirectory
    file_sets = []
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if os.path.isdir(path):
            files = set(f for f in os.listdir(path) 
                        if f.lower().endswith('.tif') and not f.lower().endswith('.tif.aux.xml'))
            file_sets.append(files)
    
    # Find the intersection of all file sets
    common_files = set.intersection(*file_sets)
    common_files_list = list(common_files)
    
    # Write to output files
    write_to_file(common_files_list, os.path.join(base_dir, f'{site}_inflist.txt'))

def genTreelist(tifPath, projectPath, rdsPath=None, epsg=None, filename = 'treelist.csv'):
    pathToRScript = os.path.join(projectPath, 'SatCHM', 'prepTrainInputs', 'Rutils.R')
    outdir = os.path.dirname(tifPath)
    treelist_csv = os.path.join(outdir, filename)
    crowns_gpkg = os.path.join(outdir, 'final_detected_crowns.gpkg')

    proc = subprocess.run(
        ["Rscript", "--vanilla", pathToRScript, "treelist",
         "--chm", tifPath, "--outdir", outdir],
        text=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    if filename != 'treelist.csv':
        os.rename(os.path.join(outdir, 'treelist.csv'), treelist_csv)

    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")

    #TODO: Add col for crown radius derived from crown area

    # Add additional attributes with cloud2trees
    df = trivHMD(treelistCSV=treelist_csv, crownsGPKG=crowns_gpkg, chm_raster=tifPath)
    df.to_csv(treelist_csv)

    # If we don't have rds data, add the hmd with a trivial hmd method
    if rdsPath != None and epsg != None:
        proc = subprocess.run(
            [
                "Rscript", "--vanilla", pathToRScript, "cbh",
                "--csv", treelist_csv,
                "--model", rdsPath,
                "--epsg", str(epsg)
            ],
            text=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


def trivHMD(
    treelistCSV: str,
    crownsGPKG: str,
    chm_raster: str,
    tree_id_col: str = "treeID",
    tree_x_col: str = "tree_x",
    tree_y_col: str = "tree_y",
    densify_step: float = 0.25,
    hmd_col: str = "HMD_m",
    furthest_x_col: str = "HMD_x",
    furthest_y_col: str = "HMD_y",
    crown_join_key: str = None,
    search_tolerance: float = 2.0
):
    def _densify(ls: LineString, step: float) -> LineString:
        if step <= 0 or ls.length == 0:
            return ls
        L = ls.length
        n = max(1, int(math.ceil(L / step)))
        pts = [ls.interpolate(i * L / n) for i in range(n + 1)]
        return LineString(pts)

    def _furthest_point(tree_pt: Point, poly, step: float) -> Point:
        boundary = poly.boundary
        if boundary.geom_type == "MultiLineString":
            coords = [xy for seg in boundary.geoms for xy in _densify(seg, step).coords]
        else:
            coords = list(_densify(boundary, step).coords)
        max_d, max_xy = -1, None
        for x, y in coords:
            d = tree_pt.distance(Point(x, y))
            if d > max_d:
                max_d, max_xy = d, (x, y)
        return Point(max_xy)

    def _sample_chm(x: float, y: float, chm_path: str, dst_crs) -> float | None:
        with rasterio.open(chm_path) as src:
            if src.crs is None or (dst_crs is not None and src.crs == dst_crs):
                sampler = src
            else:
                sampler = WarpedVRT(src, crs=dst_crs)
            val = list(sampler.sample([(x, y)]))[0][0]
            if src.nodata is not None and val == src.nodata:
                return None
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return float(val)

    # --- Load data ---
    df = pd.read_csv(treelistCSV)
    crowns = gpd.read_file(crownsGPKG)
    if crowns.empty or crowns.geometry.is_empty.all():
        raise ValueError("No valid crown polygons found.")
    if crowns.crs is None:
        raise ValueError("Crowns layer has no CRS.")

    crowns = crowns[[crowns.geometry.name]].copy()  # NEW: only keep crown geometry

    gdf_trees = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[tree_x_col], df[tree_y_col]),
        crs=crowns.crs
    )

    # --- Match trees to crowns ---
    pip = gpd.sjoin(gdf_trees, crowns, how="left", predicate="within")
    unmatched = pip[pip.index_right.isna()].copy()
    if not unmatched.empty:
        nearest = gpd.sjoin_nearest(
            unmatched.drop(columns=["index_right"]),
            crowns,
            how="left",
            max_distance=search_tolerance,
            distance_col="_nn_dist"
        )
        pip.update(nearest)

    out = pip.copy()
    out[hmd_col] = pd.NA
    # (We’ll compute HMD_x/y internally, but won’t return them if you don’t want them)
    out[furthest_x_col] = pd.NA
    out[furthest_y_col] = pd.NA

    # --- Compute HMD ---
    for idx, row in out.iterrows():
        crown_idx = row.get("index_right")
        if pd.isna(crown_idx):
            continue
        crown_geom = crowns.geometry.iloc[int(crown_idx)]
        tree_pt = row.geometry
        if crown_geom is None or crown_geom.is_empty or tree_pt is None:
            continue
        try:
            far_pt = _furthest_point(tree_pt, crown_geom, densify_step)
            out.at[idx, furthest_x_col] = far_pt.x
            out.at[idx, furthest_y_col] = far_pt.y
            h = _sample_chm(far_pt.x, far_pt.y, chm_raster, crowns.crs)
            out.at[idx, hmd_col] = float(h) if h is not None else pd.NA
        except Exception:
            continue

    # --- Return only desired columns ---
    desired_cols = ["treeID", "tree_height_m", "tree_x", "tree_y", "crown_area_m2", hmd_col]
    # Build from original CSV columns to avoid _left/_right, then add HMD from out
    final = df.copy()
    final[hmd_col] = out[hmd_col].values  # align by index
    final = final[desired_cols]           # NEW: select exactly what you want
    final.index.name = "index"
    return final

def filter_trees_and_crowns(treelist_csv, crowns_gpkg, crowns_layer=None, min_area=4.0):
    # --- Load treelist ---
    df = pd.read_csv(treelist_csv)

    # Filter by crown area
    df_filtered = df[df["crown_area_m2"] >= min_area].copy()

    # Overwrite treelist CSV
    df_filtered.to_csv(treelist_csv, index=False)

    # --- Load crowns ---
    crowns = gpd.read_file(crowns_gpkg, layer=crowns_layer)

    # Build GeoDataFrame of tree tops (from filtered treelist)
    gdf_trees = gpd.GeoDataFrame(
        df_filtered,
        geometry=gpd.points_from_xy(df_filtered["tree_x"], df_filtered["tree_y"]),
        crs=crowns.crs,
    )

    # Spatial join: keep only crowns that contain the filtered trees
    joined = gpd.sjoin(crowns, gdf_trees, how="inner", predicate="contains")

    # Keep only the crown geometries
    crowns_filtered = crowns.loc[joined.index]

    # Overwrite the crowns GPKG
    crowns_filtered.to_file(crowns_gpkg, driver="GPKG", layer=crowns_layer or "crowns")

    return df_filtered, crowns_filtered

def remove_rasters_with_nodata(folder_path, dry_run=True, extensions=(".tif", ".tiff")):
    """
    Scans a folder for raster files and removes those containing any 'no data' pixels.
    Handles both numeric and NaN nodata definitions.
    
    Parameters
    ----------
    folder_path : str
        Path to folder containing raster tiles.
    dry_run : bool, optional
        If True, only prints which files would be deleted. If False, actually deletes them.
    extensions : tuple, optional
        File extensions to check (default: .tif, .tiff).
    """
    total_rasters = 0
    to_delete = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                total_rasters += 1
                raster_path = os.path.join(root, file)

                try:
                    with rasterio.open(raster_path) as src:
                        data = src.read(1)
                        nodata_val = src.nodata

                        # Handle both numeric and NaN nodata properly
                        if nodata_val is None:
                            nodata_mask = np.isnan(data)
                        elif np.isnan(nodata_val):
                            nodata_mask = np.isnan(data)
                        else:
                            nodata_mask = (data == nodata_val)

                        if np.any(nodata_mask):
                            to_delete.append(raster_path)

                except Exception as e:
                    print(f"⚠️ Could not read {raster_path}: {e}")

    # --- Perform or simulate deletion ---
    if dry_run:
        print(f"🔎 Dry run: Found {len(to_delete)} rasters with no-data pixels (out of {total_rasters}).")
        for path in to_delete:
            print(f" - {path}")
    else:
        deleted = 0
        for path in to_delete:
            try:
                os.remove(path)
                deleted += 1
            except Exception as e:
                print(f"❌ Could not delete {path}: {e}")
        print(f"🗑️ Deleted {deleted}/{len(to_delete)} rasters with no-data pixels.")
        print(f"✅ Remaining rasters: {total_rasters - deleted}")

    return to_delete