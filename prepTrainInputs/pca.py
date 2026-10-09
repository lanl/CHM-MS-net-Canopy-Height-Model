"""
pca.py — PCA dimensionality reduction for AlphaEarth embeddings

Performs PCA on native 10m int8 AlphaEarth embeddings to reduce from 64 
dimensions to 4/8/16/32 dimensions. Operates in embedding space [-1,1] via
de-quantization, then re-quantizes output for storage efficiency.

WORKFLOW:
    1. Sample pixels from ae_source.tif (int8 at 10m resolution)
    2. De-quantize to float32 in [-1,1] range using Google's formula
    3. Fit sklearn PCA on sampled data
    4. Transform full raster in memory-efficient chunks:
       int8 → float32 → PCA transform → float32 → int8
    5. Save reduced-dimension rasters and PCA models

FILE STRUCTURE CREATED:
    {site}_data/ae/
    ├── ae_source.tif               # Original 64-band (unchanged)
    ├── pca_models/
    │   ├── pca_4d.pkl              # Pickled PCA models
    │   ├── pca_4d_metadata.json    # Training metadata
    │   └── pca_summary.json        # Comparison across dimensions
    ├── 4d/ae_source_4d.tif         # 4-band reduced raster
    ├── 8d/ae_source_8d.tif         # 8-band reduced raster
    ├── 16d/ae_source_16d.tif       # 16-band reduced raster
    ├── 32d/ae_source_32d.tif       # 32-band reduced raster
    └── 64d/ae_source.tif           # Symlink to original

USAGE:
    From command line:
        python pca.py --site fs_train_ae
        python pca.py --site fs_train_ae --dimensions 16 32
        python pca.py --site fs_train_ae --sample-size 200000 --force-refit
    
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.decomposition import PCA

# Import from sibling module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from site_config import load_site_config
finally:
    sys.path.pop(0)

# ────────────────────────────────────────────────────────────────────
# Constants (from rasterAE.py line 30-31)
# ────────────────────────────────────────────────────────────────────
AE_BANDS = 64
AE_BAND_NAMES = [f"A{i:02d}" for i in range(AE_BANDS)]

# Target dimensions for PCA reduction
TARGET_DIMENSIONS = [4, 8, 16, 32, 64]
# Sampling parameters
DEFAULT_SAMPLE_SIZE = 100000  # 100k pixels for stable covariance estimation
DEFAULT_CHUNK_SIZE = 1000

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
# Utility Functions: De-quantization and Re-quantization
# ────────────────────────────────────────────────────────────────────

def dequantize_int8_to_float32(data_int8: np.ndarray) -> np.ndarray:
    """
    De-quantize int8 AE embeddings to float32 in [-1, 1] range.
    
    This reverses Google's quantization: int8 = sign(x) * sqrt(|x|) * 127.5
    Formula from ms_net/pore_utils_2D.py line 680.
    
    Args:
        data_int8: Array of int8 values, shape (bands, height, width) or (bands, n_pixels)
        
    Returns:
        Array of float32 values in [-1, 1] range, same shape as input
        
    Example:
        >>> int8_data = np.array([64, -32, 0], dtype='int8')
        >>> float_data = dequantize_int8_to_float32(int8_data)
        >>> # Returns approximately [0.252, -0.063, 0.0]
    """
    # Cast to float32 for computation
    data_float = data_int8.astype(np.float32)
    
    # De-quantize: ((x / 127.5) ** 2) * sign(x)
    # From ms_net/pore_utils_2D.py line 680
    dequantized = ((data_float / 127.5) ** 2) * np.sign(data_float)
    
    return dequantized


# FIXME: not really needed
def quantize_float32_to_int8(data_float32: np.ndarray) -> np.ndarray:
    """
    Quantize float32 embeddings in [-1, 1] to int8 for storage.
    
    This applies Google's quantization: int8 = sign(x) * sqrt(|x|) * 127.5
    Formula from prepTrainInputs/rasterAE.py line 204.
    
    Args:
        data_float32: Array of float32 values in [-1, 1], shape (bands, height, width)
        
    Returns:
        Array of int8 values, same shape as input
        
    Example:
        >>> float_data = np.array([0.25, -0.0625, 0.0], dtype='float32')
        >>> int8_data = quantize_float32_to_int8(float_data)
        >>> # Returns approximately [63, -32, 0]
    """
    # Quantize: sign(x) * sqrt(|x|) * 127.5
    # From rasterAE.py line 204
    quantized = np.round(
        np.sign(data_float32) * np.sqrt(np.abs(data_float32)) * 127.5
    ).astype('int8')
    
    return quantized


# ────────────────────────────────────────────────────────────────────
# Sampling Functions
# ────────────────────────────────────────────────────────────────────

def sample_pixels_from_raster(
    raster_path: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42
) -> np.ndarray:
    """
    Sample random pixels from AE raster for PCA fitting.
    
    Reads int8 raster, de-quantizes to float32, samples uniformly across
    spatial extent (includes edges/artifacts). Returns in shape (n_samples, 64).
    
    Args:
        raster_path: Path to ae_source.tif (64-band int8)
        sample_size: Number of pixels to sample (default 100k)
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (sample_size, 64) with float32 values in [-1, 1]
        
    Raises:
        ValueError: If raster doesn't have 64 bands or wrong dtype
    """
    np.random.seed(seed)
    
    with rasterio.open(raster_path) as src:
        # Validate raster
        if src.count != AE_BANDS:
            raise ValueError(
                f"Expected {AE_BANDS} bands, got {src.count} in {raster_path}"
            )
        if src.dtypes[0] != 'int8':
            raise ValueError(
                f"Expected int8 dtype, got {src.dtypes[0]} in {raster_path}"
            )
        
        height, width = src.height, src.width
        total_pixels = height * width
        
        # Determine actual sample size
        actual_sample_size = min(sample_size, total_pixels)
        
        print(
            f"Sampling {actual_sample_size:,} pixels from {width}*{height} raster "
            f"({total_pixels:,} total pixels)"
        )
        
        # Generate random pixel coordinates
        sample_rows = np.random.randint(0, height, size=actual_sample_size)
        sample_cols = np.random.randint(0, width, size=actual_sample_size)
        
        # Read all bands at once (efficient for random access)
        # Pattern similar to utils.py line 1048: src.read(window=window)
        data_int8 = src.read()  # Shape: (64, height, width)
        
        # Extract sampled pixels
        sampled_int8 = data_int8[:, sample_rows, sample_cols]  # Shape: (64, n_samples)
        
        # De-quantize to float32
        sampled_float32 = dequantize_int8_to_float32(sampled_int8)
        
        # Transpose to (n_samples, 64) for sklearn
        samples = sampled_float32.T
        
        print(
            f"Sampled data shape: {samples.shape}, "
            f"range: [{samples.min():.3f}, {samples.max():.3f}]"
        )
        
        return samples


# ────────────────────────────────────────────────────────────────────
# PCA Operations
# ────────────────────────────────────────────────────────────────────

def fit_pca(
    samples: np.ndarray,
    n_components: int
) -> PCA:
    """
    Fit PCA on sampled embedding data.
    
    Args:
        samples: Array of shape (n_samples, 64) with float32 embeddings
        n_components: Target dimensionality (4, 8, 16, or 32)
        
    Returns:
        Fitted sklearn PCA object
    """
    print(f"Fitting PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components)
    pca.fit(samples)
    
    # Report variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(
        f"PCA-{n_components}: "
        f"Explained variance = {cumulative_variance[-1]:.2%} "
        f"({n_components}/{AE_BANDS} components)"
    )
    
    return pca


def save_pca_model(
    pca: PCA,
    save_dir: Path,
    n_components: int,
    metadata: Dict
) -> Tuple[Path, Path]:
    """
    Save PCA model and metadata to disk.
    
    Args:
        pca: Fitted PCA object
        save_dir: Directory to save model (e.g., {site}_data/ae/pca_models/)
        n_components: Number of components (for filename)
        metadata: Dict with training info (site, sample_size, timestamp, etc.)
        
    Returns:
        Tuple of (model_path, metadata_path)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f"pca_{n_components}d.pkl"
    metadata_path = save_dir / f"pca_{n_components}d_metadata.json"
    
    # Save PCA object
    with open(model_path, 'wb') as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Augment metadata with PCA statistics
    full_metadata = {
        **metadata,
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'singular_values': pca.singular_values_.tolist(),
    }
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    print(f"Saved PCA model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    
    return model_path, metadata_path


def load_pca_model(model_path: str) -> Tuple[PCA, Dict]:
    """
    Load PCA model and metadata from disk.
    
    Args:
        model_path: Path to .pkl file
        
    Returns:
        Tuple of (PCA object, metadata dict)
        
    Raises:
        FileNotFoundError: If model or metadata doesn't exist
    """
    model_path = Path(model_path)
    metadata_path = model_path.with_name(model_path.stem + '_metadata.json')
    
    if not model_path.exists():
        raise FileNotFoundError(f"PCA model not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    # Load PCA
    with open(model_path, 'rb') as f:
        pca = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(
        f"Loaded PCA-{pca.n_components_}: "
        f"{metadata['cumulative_variance'][-1]:.2%} variance explained"
    )
    
    return pca, metadata


# ────────────────────────────────────────────────────────────────────
# Raster Transformation
# ────────────────────────────────────────────────────────────────────

def transform_ae_raster(
    input_raster: str,
    pca: PCA,
    output_raster: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """
    Transform full AE raster using fitted PCA, processing in memory-efficient chunks.
    
    Workflow per chunk:
        1. Read int8 chunk
        2. De-quantize to float32 [-1, 1]
        3. Apply PCA transformation
        4. Re-quantize to int8
        5. Write chunk to output
    
    Args:
        input_raster: Path to ae_source.tif (64-band int8)
        pca: Fitted PCA object
        output_raster: Path to save reduced raster (e.g., ae_source_16d.tif)
        chunk_size: Window size in pixels (default 1000*1000)
        
    Memory usage per chunk:
        Input:  64 bands * 1000*1000 pixels * 1 byte  = 64 MB
        Float:  64 bands * 1000*1000 pixels * 4 bytes = 256 MB
        Output: n bands  * 1000*1000 pixels * 4 bytes = 64 MB (for 16d)
        Total:  ~400-500 MB per chunk (safe for most systems)
    """
    n_components = pca.n_components_
    
    print(f"Transforming raster: {input_raster}")
    print(f"PCA: 64 → {n_components} dimensions")
    print(f"Output: {output_raster}")
    
    with rasterio.open(input_raster) as src:
        # Validate input
        if src.count != AE_BANDS:
            raise ValueError(f"Expected {AE_BANDS} bands, got {src.count}")
        if src.dtypes[0] != 'int8':
            raise ValueError(f"Expected int8, got {src.dtypes[0]}")
        
        # Prepare output profile
        profile = src.profile.copy()
        profile.update({
            'count': n_components,
            'dtype': 'int8',
            'compress': 'zstd',  # Match rasterAE.py line 271
            'predictor': 2,
            'tiled': True,
        })
        
        height, width = src.height, src.width
        total_windows = int(np.ceil(height / chunk_size) * np.ceil(width / chunk_size))
        
        print(
            f"Processing {width}*{height} raster in {total_windows} chunks "
            f"({chunk_size}*{chunk_size} pixels each)"
        )
        
        with rasterio.open(output_raster, 'w', **profile) as dst:
            processed_windows = 0
            
            # Iterate over windows
            for row_off in range(0, height, chunk_size):
                for col_off in range(0, width, chunk_size):
                    # Calculate window bounds
                    window_height = min(chunk_size, height - row_off)
                    window_width = min(chunk_size, width - col_off)
                    window = Window(col_off, row_off, window_width, window_height)
                    
                    # Read int8 chunk (64, H, W)
                    chunk_int8 = src.read(window=window)
                    
                    # De-quantize to float32
                    chunk_float32 = dequantize_int8_to_float32(chunk_int8)
                    
                    # Reshape for PCA: (64, H, W) → (H*W, 64)
                    n_pixels = window_height * window_width
                    pixels_2d = chunk_float32.reshape(AE_BANDS, n_pixels).T
                    
                    # Apply PCA transformation
                    transformed_2d = pca.transform(pixels_2d)  # (H*W, n_components)
                    
                    # Reshape back: (H*W, n_components) → (n_components, H, W)
                    transformed_cube = transformed_2d.T.reshape(
                        n_components, window_height, window_width
                    )
                    
                    # Re-quantize to int8
                    chunk_out_int8 = quantize_float32_to_int8(transformed_cube)

                    # Write chunk
                    dst.write(chunk_out_int8, window=window)
                    
                    processed_windows += 1
                    if processed_windows % 10 == 0:
                        progress = (processed_windows / total_windows) * 100
                        print(f"Progress: {progress:.1f}% ({processed_windows}/{total_windows} windows)")
            
            print(f"✓ Transformation complete: {output_raster}")


# ────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────

def validate_pca_reconstruction(
    original_raster: str,
    reduced_raster: str,
    pca: PCA,
    n_test_pixels: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Validate PCA transformation by comparing original vs reconstructed embeddings.
    
    Samples random pixels, reconstructs embeddings using PCA inverse transform,
    computes reconstruction error metrics.
    
    Args:
        original_raster: Path to ae_source.tif (64-band)
        reduced_raster: Path to ae_source_Nd.tif (N-band)
        pca: Fitted PCA object used for reduction
        n_test_pixels: Number of pixels to test
        seed: Random seed
        
    Returns:
        Dict with validation metrics:
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - max_error: Maximum absolute error
            - explained_variance: From PCA object
            - test_pixels: Number of pixels tested
    """
    np.random.seed(seed)
    
    print(f"Validating PCA reconstruction on {n_test_pixels} random pixels...")
    
    with rasterio.open(original_raster) as src_orig, \
         rasterio.open(reduced_raster) as src_reduced:
        
        height, width = src_orig.height, src_orig.width
        
        # Sample random pixels
        test_rows = np.random.randint(0, height, size=n_test_pixels)
        test_cols = np.random.randint(0, width, size=n_test_pixels)
        
        # Read original pixels (64-band int8)
        orig_data = src_orig.read()
        orig_pixels_int8 = orig_data[:, test_rows, test_cols]  # (64, n_test)
        orig_pixels_float32 = dequantize_int8_to_float32(orig_pixels_int8).T  # (n_test, 64)
        
        # Read reduced pixels (n_components-band int8)
        reduced_data = src_reduced.read()
        reduced_pixels_int8 = reduced_data[:, test_rows, test_cols]  # (n_components, n_test)
        reduced_pixels_float32 = dequantize_int8_to_float32(reduced_pixels_int8).T  # (n_test, n_components)
        
        # Reconstruct using PCA inverse transform
        reconstructed = pca.inverse_transform(reduced_pixels_float32)  # (n_test, 64)
        
        # Compute errors
        errors = orig_pixels_float32 - reconstructed
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))
        
        # Per-band error
        band_rmse = np.sqrt(np.mean(errors ** 2, axis=0))
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'max_error': float(max_error),
            'band_rmse_mean': float(band_rmse.mean()),
            'band_rmse_max': float(band_rmse.max()),
            'explained_variance': float(np.sum(pca.explained_variance_ratio_)),
            'n_components': pca.n_components_,
            'test_pixels': n_test_pixels,
        }
        
        print(f"Reconstruction RMSE: {rmse:.4f}")
        print(f"Reconstruction MAE: {mae:.4f}")
        print(f"Max error: {max_error:.4f}")
        print(f"Explained variance: {metrics['explained_variance']:.2%}")
        
        return metrics


# ────────────────────────────────────────────────────────────────────
# Main Workflow
# ────────────────────────────────────────────────────────────────────

def reduce_ae_dimensions(
    ae_source_path: str,
    output_dir: str,
    site_name: str,
    target_dims: List[int] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    validate: bool = True,
    seed: int = 42
) -> Dict[int, Dict]:
    """
    Complete PCA workflow: fit models and generate reduced-dimension rasters.
    
    This is the main entry point for PCA dimensionality reduction. It:
        1. Samples pixels from ae_source.tif
        2. Fits PCA models for each target dimension
        3. Transforms full raster to create ae_source_Nd.tif files
        4. Optionally validates reconstruction quality
        5. Saves PCA models and metadata
    
    Args:
        ae_source_path: Path to ae_source.tif (64-band int8)
        output_dir: Directory to save outputs (e.g., fs_train_ae_data/ae/)
        site_name: Site identifier for metadata (e.g., 'fs_train')
        target_dims: List of target dimensions (default [4, 8, 16, 32])
        sample_size: Number of pixels for PCA fitting (default 100k)
        chunk_size: Window size for transformation (default 1000*1000)
        validate: Whether to run validation (default True)
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping n_components → result dict with:
            - model_path: Path to saved PCA model
            - metadata_path: Path to metadata JSON
            - output_raster: Path to reduced raster
            - validation_metrics: Validation results (if enabled)
            
    Directory structure created:
        {output_dir}/
            4d/ae_source_4d.tif
            8d/ae_source_8d.tif
            16d/ae_source_16d.tif
            32d/ae_source_32d.tif
            pca_models/
                pca_4d.pkl, pca_4d_metadata.json
                pca_8d.pkl, pca_8d_metadata.json
                ...
    """
    if target_dims is None:
        target_dims = TARGET_DIMENSIONS
    
    output_dir = Path(output_dir)
    model_dir = output_dir / 'pca_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"PCA Dimensionality Reduction Pipeline")
    print("="*80)
    print(f"Site: {site_name}")
    print(f"Input: {ae_source_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target dimensions: {target_dims}")
    print(f"Sample size: {sample_size:,} pixels")
    print("="*80)
    
    # Step 1: Sample pixels for PCA fitting
    print("\n[1/4] Sampling pixels for PCA fitting...")
    samples = sample_pixels_from_raster(ae_source_path, sample_size=sample_size, seed=seed)
    
    results = {}
    
    # Step 2-3: Fit and transform for each target dimension
    for n_components in target_dims:
        print("\n" + "="*80)
        print(f"Processing {n_components}-dimensional reduction...")
        print("="*80)
        
        # Fit PCA
        print(f"\n[2/4] Fitting PCA with {n_components} components...")
        pca = fit_pca(samples, n_components=n_components)
        
        # Save model
        print(f"\n[3/4] Saving PCA model...")
        metadata = {
            'site': site_name,
            'source_raster': ae_source_path,
            'sample_size': sample_size,
            'seed': seed,
            'timestamp': Path(ae_source_path).stat().st_mtime,
            'input_bands': AE_BANDS,
        }
        model_path, metadata_path = save_pca_model(
            pca, model_dir, n_components, metadata
        )

        # # override pca dir with custom path
        # custom_pca_dir = "/project/wildfirehydro/ltiede/CHM_2/fs_train_ae_data/ae/pca_models"
        # curr_pca = os.path.join(custom_pca_dir, f"pca_{n_components}d.pkl")
        # pca, metadata = load_pca_model(os.fspath(curr_pca))
        
        # Transform raster
        print(f"\n[4/4] Transforming raster...")
        dim_dir = output_dir / f'{n_components}d'
        dim_dir.mkdir(parents=True, exist_ok=True)
        output_raster = dim_dir / f'ae_source_{n_components}d.tif'
        
        transform_ae_raster(
            input_raster=ae_source_path,
            pca=pca,
            output_raster=str(output_raster),
            chunk_size=chunk_size
        )
        
        # Store results
        result = {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'output_raster': str(output_raster),
            'n_components': n_components,
            'explained_variance': float(np.sum(pca.explained_variance_ratio_)),
        }
        
        # Validate if requested
        if validate:
            print(f"\n[Validation] Testing reconstruction quality...")
            validation_metrics = validate_pca_reconstruction(
                original_raster=ae_source_path,
                reduced_raster=str(output_raster),
                pca=pca,
                n_test_pixels=1000,
                seed=seed
            )
            result['validation_metrics'] = validation_metrics
        
        results[n_components] = result
        print(f"\nCompleted {n_components}d reduction")
    
    # Final summary
    print("\n" + "="*80)
    print("PCA Pipeline Complete!")
    print("="*80)
    print(f"\nGenerated files:")
    for n_comp, res in results.items():
        print(f"  {n_comp}d: {res['output_raster']}")
        print(f"       Variance: {res['explained_variance']:.2%}")
        if 'validation_metrics' in res:
            print(f"       RMSE: {res['validation_metrics']['rmse']:.4f}")
    print(f"\nPCA models saved in: {model_dir}")
    print("="*80)
    
    return results


# ────────────────────────────────────────────────────────────────────
# Command-line Interface
# ────────────────────────────────────────────────────────────────────

def main():
    """Command-line interface for PCA dimensionality reduction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reduce AlphaEarth embeddings from 64 to 4/8/16/32 dimensions using PCA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process fs_train site with defaults
  python pca.py --input fs_train_ae_data/ae/ae_source.tif --site fs_train
  
  # Custom dimensions and sample size
  python pca.py --input fs_train_ae_data/ae/ae_source.tif --site fs_train --pca-dims 8 16 --sample-size 50000
  
  # Skip validation for faster processing
  python pca.py --input fs_train_ae_data/ae/ae_source.tif --site fs_train --no-validate
        """
    )
    
    # FIXME: eventually may remove this independent file functionality
    parser.add_argument(
        '--input',
        required=False,
        help='Path to ae_source.tif (64-band int8 raster)'
    )
    parser.add_argument(
        '--site',
        required=True,
        help='Site name for metadata (e.g., fs_train, mm_train)'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (default: parent directory of input)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()
    
    site = args.site

    # Path definitions
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    print(f"project_path: {project_path}")
    site_data_path = os.path.join(project_path, f'{site}_data')
    lidar_tiles_path = os.path.join(site_data_path, 'chm')
    pathToLidarResources = os.path.join(os.path.dirname(__file__), "resources.geojson")

    # Folder creations
    os.makedirs(site_data_path, exist_ok=True)
    os.makedirs(os.path.join(site_data_path, 'wvimg'), exist_ok=True)
    wvimg_train_path = os.path.join(project_path, 'downloads', site, 'wvimgTrain')
    wvimg_inf_path = os.path.join(project_path, 'downloads', site, 'wvimgInf')
    print(f'CREATING FOLDER: {wvimg_train_path}')
    print(f'CREATING FOLDER: {wvimg_inf_path}')
    os.makedirs(wvimg_train_path, exist_ok=True)
    os.makedirs(wvimg_inf_path, exist_ok=True)

    # AlphaEarth paths and folder creations
    ae_tiles_path = os.path.join(site_data_path, 'ae')
    os.makedirs(ae_tiles_path, exist_ok=True)
    source_raster = os.path.join(ae_tiles_path, 'ae_source.tif')

    # use this site's source_raster by default
    if args.input is None :
        input_file = source_raster
        # Determine output directory
        if args.output_dir is None:
            output_dir = Path(input_file).parent
        else:
            output_dir = Path(args.output_dir)
    else :
        input_file = Path(args.input)
        # Determine output directory
        if args.output_dir is None:
            output_dir = Path(args.input).parent
        else:
            output_dir = Path(args.output_dir)
    
    
    
    # Run pipeline
    try:
        results = reduce_ae_dimensions(
            ae_source_path=str(input_file),
            output_dir=str(output_dir),
            site_name=args.site,
            target_dims=TARGET_DIMENSIONS,
            sample_size=DEFAULT_SAMPLE_SIZE,
            chunk_size=DEFAULT_CHUNK_SIZE,
            validate=True,
            seed=args.seed
        )
        
        print("\nPCA pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
