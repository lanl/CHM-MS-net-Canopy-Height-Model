"""
rasterAE.py — Fetch Google AlphaEarth embeddings from GEE for SatCHM

Downloads 64-band AlphaEarth embeddings from Google Earth Engine at native 10m 
resolution as int8 (quantized). Tiling to 0.5m happens via utils.tileAlphaEarth().

This file is separate to keep `import ee` optional - utils.py remains importable
without earthengine-api installed.
"""

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from shapely.geometry import box
import warnings

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
AE_BANDS = 64
AE_BAND_NAMES = [f"A{i:02d}" for i in range(AE_BANDS)]
VALID_YEAR_RANGE = (2017, 2025)

# ────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Spatial chunking helper functions
# ────────────────────────────────────────────────────────────────────
def _calculate_grid_layout(width_m, height_m, target_mb=35, scale=10, bands=64):
    """
    Calculate grid dimensions to keep each chunk under target size.
    
    GEE serves AlphaEarth as float64 over the wire (8 bytes/value), even though
    it's stored as int8. This is why requests are ~8x larger than expected.
    
    Args:
        width_m: AOI width in meters
        height_m: AOI height in meters
        target_mb: Target chunk size in MB (default 35, safe margin below 48 MB limit)
        scale: Pixel resolution in meters (default 10)
        bands: Number of bands (default 64)
    
    Returns:
        tuple: (n_cols, n_rows, chunk_width_m, chunk_height_m)
    """
    bytes_per_pixel = bands * 8  # float64 over GEE API
    target_bytes = target_mb * 1024 * 1024
    target_pixels = target_bytes / bytes_per_pixel
    
    # Calculate square chunk size in pixels
    chunk_side_pixels = int(target_pixels ** 0.5)
    chunk_side_m = chunk_side_pixels * scale
    
    # Calculate how many chunks needed
    n_cols = int(np.ceil(width_m / chunk_side_m))
    n_rows = int(np.ceil(height_m / chunk_side_m))
    
    # Actual chunk dimensions (divide evenly)
    chunk_width_m = width_m / n_cols
    chunk_height_m = height_m / n_rows
    
    return n_cols, n_rows, chunk_width_m, chunk_height_m


def _split_bbox_into_chunks(minx, miny, maxx, maxy, n_cols, n_rows):
    """
    Split bounding box into grid of smaller rectangles.
    
    Args:
        minx, miny, maxx, maxy: Bounding box coordinates
        n_cols: Number of columns in grid
        n_rows: Number of rows in grid
    
    Returns:
        list: List of (chunk_id, minx, miny, maxx, maxy) tuples
    """
    width = maxx - minx
    height = maxy - miny
    chunk_width = width / n_cols
    chunk_height = height / n_rows
    
    chunks = []
    chunk_id = 0
    for row in range(n_rows):
        for col in range(n_cols):
            chunk_minx = minx + col * chunk_width
            chunk_maxx = chunk_minx + chunk_width
            chunk_miny = miny + row * chunk_height
            chunk_maxy = chunk_miny + chunk_height
            
            chunks.append((chunk_id, chunk_minx, chunk_miny, chunk_maxx, chunk_maxy))
            chunk_id += 1
    
    return chunks



def _download_chunk(chunk_id, chunk_bounds, ae_image, epsg, scale, temp_dir):
    """
    Download a single chunk from GEE.
    
    Args:
        chunk_id: Chunk identifier (for logging/filenames)
        chunk_bounds: (minx, miny, maxx, maxy) tuple
        ae_image: ee.Image with AlphaEarth bands
        epsg: Target EPSG code
        scale: Pixel resolution in meters
        temp_dir: Directory to save chunk file
    
    Returns:
        str: Path to downloaded chunk file
    """
    minx, miny, maxx, maxy = chunk_bounds
    
    # Create ee.Geometry for this chunk
    chunk_aoi = ee.Geometry.Rectangle(
        [minx, miny, maxx, maxy],
        proj=f'EPSG:{epsg}',
        geodesic=False
    )
    
    export_params = {
        "image": ae_image,
        "scale": scale,
        "region": chunk_aoi,
        "fileFormat": "GeoTIFF",
        "maxPixels": 1e9,
        "crs": f"EPSG:{epsg}",
        "filePerBand": False,
    }
    
    try:
        url = ae_image.getDownloadURL(export_params)
    except ee.EEException as e:
        raise RuntimeError(f"GEE export failed for chunk {chunk_id}: {e}") from e
    
    response = requests.get(url, stream=True, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed for chunk {chunk_id} with status {response.status_code}"
        )
    
    chunk_path = Path(temp_dir) / f"chunk_{chunk_id:03d}.tif"
    with open(chunk_path, "wb") as f:
        for chunk_data in response.iter_content(chunk_size=8192):
            f.write(chunk_data)
    
    # Check if GEE returned a ZIP archive (common behavior)
    if zipfile.is_zipfile(chunk_path):
        logger.info(f"Chunk {chunk_id} is a ZIP archive, extracting...")
        
        # Extract the TIFF from the ZIP
        with zipfile.ZipFile(chunk_path, 'r') as zip_ref:
            # Find .tif files in the ZIP
            tif_files = [f for f in zip_ref.namelist() if f.endswith('.tif') or f.endswith('.tiff')]
            
            if not tif_files:
                raise RuntimeError(f"No TIFF file found in ZIP for chunk {chunk_id}")
            
            if len(tif_files) > 1:
                logger.warning(f"Multiple TIFFs in ZIP, using first: {tif_files[0]}")
            
            # Extract to temp directory
            extracted_name = zip_ref.extract(tif_files[0], temp_dir)
            extracted_path = Path(temp_dir) / extracted_name
            
            # Replace the ZIP with the extracted TIFF
            zip_backup = chunk_path.with_suffix('.zip')
            chunk_path.rename(zip_backup)  # Keep ZIP as .zip
            extracted_path.rename(chunk_path)  # Rename extracted to .tif
            
            logger.info(f"✓ Extracted TIFF from ZIP for chunk {chunk_id}")
    
    # Convert float64 to int8 if needed (GEE exports as float64)
    with rasterio.open(chunk_path) as src:
        if src.dtypes[0] == 'float64':
            logger.info(f"Converting chunk {chunk_id} from float64 to int8...")
            
            # Read all bands
            data = src.read()
            profile = src.profile.copy()
            
            # Convert to int8
            # Quantize float64 → int8 for storage efficiency (8x compression)
            data_int8 = np.round(np.sign(data) * np.sqrt(np.abs(data)) * 127.5).astype('int8')
            
            # Update profile
            profile.update(dtype='int8')
            
            # Write converted data to temporary file
            temp_converted = chunk_path.with_suffix('.tmp.tif')
            with rasterio.open(temp_converted, 'w', **profile) as dst:
                dst.write(data_int8)
            
            # Replace original with converted
            chunk_path.unlink()
            temp_converted.rename(chunk_path)
            
            logger.info(f"✓ Converted chunk {chunk_id} to int8")
    
    # Validate chunk
    with rasterio.open(chunk_path) as src:
        if src.count != AE_BANDS:
            raise RuntimeError(
                f"Chunk {chunk_id} has wrong band count: expected {AE_BANDS}, got {src.count}"
            )
        if src.dtypes[0] != 'int8':
            raise RuntimeError(
                f"Chunk {chunk_id} has wrong dtype: expected int8, got {src.dtypes[0]}"
            )
        chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded chunk {chunk_id} ({chunk_size_mb:.1f} MB)")
    
    return str(chunk_path)


def _merge_chunks(chunk_files, output_path, epsg):
    """
    Merge chunks using rasterio.merge (same pattern as utils.py line 936).
    
    Args:
        chunk_files: List of paths to chunk GeoTIFFs
        output_path: Path to write merged file
        epsg: Target EPSG code
    """
    # Open all chunk files
    src_files = [rasterio.open(f) for f in chunk_files]
    
    try:
        # Merge (rasterio handles overlaps and alignment)
        mosaic, out_transform = merge(src_files)
        
        # Get metadata from first chunk
        with rasterio.open(chunk_files[0]) as src:
            out_meta = src.meta.copy()
        
        # Update metadata for merged image
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "crs": f"EPSG:{epsg}",
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
        })
        
        # Write merged file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic.astype(np.int8))
            # Set band descriptions
            for band_idx, band_name in enumerate(AE_BAND_NAMES, start=1):
                dest.set_band_description(band_idx, band_name)
        
        logger.info(f"✓ Merged {len(chunk_files)} chunks into {output_path}")
    
    finally:
        # Close all source files
        for src in src_files:
            src.close()


# ────────────────────────────────────────────────────────────────────
# Main fetch function
# ────────────────────────────────────────────────────────────────────
def fetch_alphaEarth(
    geojson_path: str,
    save_path: str,
    year: int,
    epsg: int,
    sa_key_path: str,
    project: str,
    buffer_m: int = 512,
    scale: int = 10,
    temp_dir: str = None,
) -> str:
    """
    Fetch 64-band AlphaEarth embeddings from Google Earth Engine.
    
    Downloads at native 10m resolution as int8 (quantized by Google).
    De-quantization happens during training in pore_utils_2D.load_samples().
    
    For large AOIs (>40 MB estimated request size), automatically splits the
    download into spatial chunks to stay under GEE's 48 MB limit, then merges
    them locally.
    
    Args:
        geojson_path: Path to UTM GeoJSON defining AOI
        save_path: Where to save output (e.g., downloads/{site}/ae/{site}_ae_{year}.tif)
        year: 2017-2025 (validated)
        epsg: Target UTM EPSG code (e.g., 32617)
        sa_key_path: Path to GEE service account JSON key file
        project: GEE project ID (e.g., "satchm")
        buffer_m: Buffer in meters to expand AOI (default 512, matches fetch_DEM)
        scale: Native resolution in meters (default 10)
        temp_dir: Directory for temporary chunk storage (default: CHM_2/temp_chunks)
    
    Returns:
        str: Path to saved GeoTIFF file
        
    Raises:
        ValueError: If year is out of valid range
        FileNotFoundError: If geojson_path or sa_key_path don't exist
        RuntimeError: If GEE request fails
    """
    # Validate inputs
    geojson_path = Path(geojson_path)
    save_path = Path(save_path)
    sa_key_path = Path(sa_key_path)
    
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    
    if not sa_key_path.exists():
        raise FileNotFoundError(f"GEE service account key not found: {sa_key_path}")
    
    # if not (VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]):
    #     raise ValueError(
    #         f"alphaEarthYear must be {VALID_YEAR_RANGE[0]}-{VALID_YEAR_RANGE[1]}, got {year}"
    #     )
    
    if year >= VALID_YEAR_RANGE[1] :    # 2025
        raise ValueError(
            f"alphaEarthYear must be {VALID_YEAR_RANGE[0]}-{VALID_YEAR_RANGE[1]}, got {year}"
        )
    
    if year < VALID_YEAR_RANGE[0] :     # 2017
        warnings.warn(
            f"alphaEarthYear must be {VALID_YEAR_RANGE[0]}-{VALID_YEAR_RANGE[1]}, got {year}",
            category=UserWarning,
            stacklevel=2
        )
        year = VALID_YEAR_RANGE[0]      # earliest year AE provides
    
    year = int(year)        # ensure type int for GEE
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize Earth Engine
    logger.info("Initializing Earth Engine...")
    try:
        with open(sa_key_path) as f:
            key_data = json.load(f)
        
        credentials = ee.ServiceAccountCredentials(
            key_data['client_email'],
            str(sa_key_path)
        )
        ee.Initialize(credentials, project=project)
        logger.info(f"✓ Earth Engine initialized (project: {project})")
        
    except ee.EEException as e:
        if "USER_PROJECT_DENIED" in str(e):
            raise RuntimeError(
                f"GEE project '{project}' access denied. Required IAM roles:\n"
                f"  - roles/serviceusage.serviceUsageConsumer\n"
                f"  - Earth Engine API must be enabled\n"
                f"Original error: {e}"
            ) from e
        raise
    
    # Load AOI and apply buffer
    logger.info(f"Loading AOI from {geojson_path}...")
    gdf = gpd.read_file(geojson_path)
    
    if gdf.empty:
        raise ValueError("GeoJSON is empty")
    if gdf.crs is None:
        raise ValueError("GeoJSON has no CRS defined")
    
    # Buffer in meters via projected CRS
    #gdf_m = gdf.to_crs(epsg=3857)              # I don't know why we do this in utils.fetch_DEM() but it works there?
    gdf_m = gdf.to_crs(epsg=epsg)
    minx, miny, maxx, maxy = gdf_m.total_bounds

    # Calculate unbuffered bbox dimensions
    width_m = maxx - minx
    height_m = maxy - miny
    bbox_area_km2 = (width_m * height_m) / 1_000_000
    
    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m

    # ========== DIAGNOSTIC: Insert after line 133 ==========
    # Calculate unbuffered area
    unbuffered_area_m2 = gdf_m.geometry.area.sum()
    unbuffered_area_km2 = unbuffered_area_m2 / 1_000_000

    # Calculate buffered bbox dimensions
    width_m = maxx - minx
    height_m = maxy - miny
    buffered_bbox_area_km2 = (width_m * height_m) / 1_000_000

    print(f"\n=== AOI SIZE DIAGNOSTIC ===")
    print(f"Unbuffered polygon dimensions: {width_m/1000:.2f} km * {height_m/1000:.2f} km")
    print(f"Unbuffered polygon area: {unbuffered_area_km2:.2f} km²")
    print(f"Buffered bbox dimensions: {width_m/1000:.2f} km * {height_m/1000:.2f} km")
    print(f"Buffered bbox area: {buffered_bbox_area_km2:.2f} km²")
    print(f"Buffer applied: {buffer_m} m on each side")

    # Calculate expected request size
    pixels_at_10m = int(width_m / 10) * int(height_m / 10)
    request_size_mb = (pixels_at_10m * 64 * 1) / (1024 * 1024)  # 64 bands, int8=1 byte
    # GEE actually serves as float64 (8 bytes), so actual request is 8x larger
    actual_request_mb = (pixels_at_10m * 64 * 8) / (1024 * 1024)
    print(f"Expected pixels (10m): {pixels_at_10m:,}")
    print(f"Expected request size (int8): {request_size_mb:.1f} MB")
    print(f"Actual request size (float64 over GEE): {actual_request_mb:.1f} MB (limit: 48 MB)")
    print(f"===========================\n")
    # ========== END DIAGNOSTIC ==========

    # Use UTM coordinates directly (don't convert to WGS84)
    minx, miny, maxx, maxy = gdf_m.total_bounds  # Already buffered above
    buffered = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=epsg)

    # Create EE geometry with explicit UTM projection
    aoi = ee.Geometry.Rectangle(
        [minx, miny, maxx, maxy],
        proj=f'EPSG:{epsg}',
        geodesic=False
    )

    logger.info(f"AOI bounds (UTM): {minx}, {miny}, {maxx}, {maxy} (buffered {buffer_m}m)")
    
    # Fetch AlphaEarth image from GEE
    logger.info(f"Fetching AlphaEarth embeddings for year {year}...")
    
    collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"
    
    filtered = (
        collection
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .mosaic()
    )
    
    # Hard-lock band order (critical for consistency)
    ae_image = filtered.select(AE_BAND_NAMES)
    
    # Decide if chunking is needed
    GEE_LIMIT_MB = 48
    SAFETY_MARGIN_MB = 40  # Conservative target
    
    if actual_request_mb > SAFETY_MARGIN_MB:
        # Use spatial chunking to stay under limit
        logger.info(
            f"Request size ({actual_request_mb:.1f} MB) exceeds safety margin ({SAFETY_MARGIN_MB} MB). "
            "Using chunked download..."
        )
        
        # Calculate grid layout
        n_cols, n_rows, chunk_w, chunk_h = _calculate_grid_layout(width_m, height_m)
        total_chunks = n_cols * n_rows
        
        print(f"Splitting into {n_cols}*{n_rows} = {total_chunks} chunks")
        print(f"Each chunk: ~{chunk_w/1000:.1f} * {chunk_h/1000:.1f} km\n")
        
        # Set up temp directory for chunks
        # Default: ae_tiles_path/ae_chunks
        temp_base = Path(temp_dir)
        # Create temp directory structure
        temp_base.mkdir(parents=True, exist_ok=True)
        chunk_temp_dir = temp_base / "ae_chunks"
        chunk_temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Temp directory: {chunk_temp_dir}")
        
        try:
            # Split and download chunks
            chunks = _split_bbox_into_chunks(minx, miny, maxx, maxy, n_cols, n_rows)
            chunk_files = []
            
            for chunk_id, cx_min, cy_min, cx_max, cy_max in chunks:
                logger.info(f"Downloading chunk {chunk_id + 1}/{total_chunks}...")
                chunk_path = _download_chunk(
                    chunk_id, (cx_min, cy_min, cx_max, cy_max),
                    ae_image, epsg, scale, str(chunk_temp_dir)
                )
                chunk_files.append(chunk_path)
            
            # Merge chunks
            logger.info("Merging chunks...")
            _merge_chunks(chunk_files, save_path, epsg)
            
        finally:
            # Clean up temp directory
            shutil.rmtree(chunk_temp_dir, ignore_errors=True)
            logger.info("✓ Cleaned up temporary files")
    
    else:
        # Direct download (existing code path)
        logger.info(
            f"Request size ({actual_request_mb:.1f} MB) is within limit. "
            "Using direct download..."
        )
        
        # Download via getDownloadURL (streaming, mirrors fetch_DEM pattern)
        export_params = {
            "image": ae_image,
            "scale": scale,
            "region": aoi,
            "fileFormat": "GeoTIFF",
            "maxPixels": 1e9,
            "crs": f"EPSG:{epsg}",
            "filePerBand": False,
        }
        
        logger.info(f"Requesting download URL from GEE (scale={scale}m, crs=EPSG:{epsg})...")
        
        try:
            url = ae_image.getDownloadURL(export_params)
        except ee.EEException as e:
            raise RuntimeError(f"GEE export failed: {e}") from e
        
        logger.info("Downloading...")
        response = requests.get(url, stream=True, timeout=600)
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Download failed with status {response.status_code}: {response.text}"
            )
        
        # Stream to file (same pattern as fetch_DEM)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✓ Downloaded to {save_path}")
    
    # Post-validate
    with rasterio.open(save_path) as src:
        if src.count != AE_BANDS:
            raise RuntimeError(
                f"Expected {AE_BANDS} bands, got {src.count}"
            )
        
        if src.dtypes[0] != 'int8':
            raise RuntimeError(
                f"Expected int8 dtype, got {src.dtypes[0]}"
            )
        
        if src.crs is None or src.crs.to_epsg() != epsg:
            raise RuntimeError(
                f"Expected EPSG:{epsg}, got {src.crs}"
            )
        
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"✓ Validated: {AE_BANDS} bands, int8, EPSG:{epsg}, "
            f"{src.width}*{src.height} pixels, {file_size_mb:.1f} MB"
        )
    
    return str(save_path)
