"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare, derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

def find_and_plot_tiles(coord_pattern, sites_data_dir='./sites_data', metric='all'):
    """
    Find tiles matching coordinate pattern across all sites and create a plot.
    
    All metrics are normalized to [0, 1] where 1 = perfect match:
    - cosine: Pattern similarity (angle between vectors)
    - ssim: Structural similarity (luminance + contrast + structure)
    - norm_corr: Normalized correlation (linear relationship)
    - similarity: Inverse normalized MAE (pixel-wise accuracy)
    
    Args:
        coord_pattern: Coordinate pattern to match (e.g., "491008_7199136")
        sites_data_dir: Path to sites_data directory
        metric: Which metric to display ('cosine', 'ssim', 'norm_corr', 'similarity', or 'all')
    """
    sites_data_path = Path(sites_data_dir)
    
    # Find all matching files
    ground_truth = None
    predictions = []
    
    # Search through all site directories
    for site_dir in sites_data_path.glob("*_data"):
        if not site_dir.is_dir():
            continue
        
        # Search in chm folder for ground truth
        chm_dir = site_dir / "chm"
        if chm_dir.exists():
            for file in chm_dir.glob(f"*{coord_pattern}*"):
                if file.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg']:
                    if ground_truth is None:
                        ground_truth = str(file)
                        print(f"Found ground truth: {file.name}")
                    else:
                        print(f"Warning: Multiple ground truth files found, using first one")
        
        # Search in all chm_preds_train_* folders for predictions
        for pred_dir in site_dir.glob("chm_preds_train_*"):
            if not pred_dir.is_dir():
                continue
            
            # Extract training site from folder name
            train_site = pred_dir.name.replace("chm_preds_train_", "")
            
            for file in pred_dir.glob(f"*{coord_pattern}*"):
                if file.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg']:
                    predictions.append((str(file), train_site))
                    print(f"Found prediction: {train_site} -> {file.name}")
    
    # Verify we found files
    if ground_truth is None:
        raise ValueError(f"No ground truth file found with pattern '{coord_pattern}' in any chm folder")
    if len(predictions) == 0:
        raise ValueError(f"No prediction files found with pattern '{coord_pattern}'")
    
    print(f"\nFound {len(predictions)} predictions for coordinate {coord_pattern}")
    
    # Sort predictions by training site name
    predictions.sort(key=lambda x: x[1])
    
    # Load all images
    all_paths = [ground_truth] + [p[0] for p in predictions]
    images = []
    
    for path in all_paths:
        img = imread(path)
        # If image is RGB/RGBA, take first channel or convert to grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # RGB
                img = np.mean(img, axis=2)
            elif img.shape[2] == 4:  # RGBA
                img = np.mean(img[:, :, :3], axis=2)
        images.append(img)
    
    # Ground truth is the first image
    gt_image = images[0]
    gt_flat = gt_image.flatten()
    
    # Calculate metrics for each prediction (all bounded [0, 1])
    metrics_list = []
    for pred_img in images[1:]:
        pred_flat = pred_img.flatten()
        
        # 1. COSINE SIMILARITY [0, 1]
        # Measures the cosine of the angle between two vectors
        # 1 = vectors point in same direction (identical patterns)
        # 0 = vectors are orthogonal (no pattern similarity)
        # For non-negative data like images, naturally bounded [0, 1]
        cos_sim = cosine_similarity(gt_flat.reshape(1, -1), pred_flat.reshape(1, -1))[0, 0]
        
        # 2. SSIM [0, 1]
        # Structural Similarity Index - designed for image quality assessment
        # Compares: luminance (brightness), contrast (dynamic range), structure (patterns)
        # 1 = structurally identical images
        # 0 = completely different structure
        data_range = gt_image.max() - gt_image.min()
        ssim_val = ssim(gt_image, pred_img, data_range=data_range)
        
        # 3. NORMALIZED CORRELATION [0, 1]
        # Pearson correlation transformed from [-1, 1] to [0, 1]
        # Measures strength of linear relationship
        # 1 = perfect positive correlation
        # 0.5 = no correlation
        # 0 = perfect negative correlation
        corr, _ = pearsonr(gt_flat, pred_flat)
        norm_corr = (corr + 1) / 2
        
        # 4. NORMALIZED SIMILARITY [0, 1]
        # Based on Mean Absolute Error, normalized and inverted
        # Measures pixel-wise accuracy relative to data range
        # 1 = perfect pixel-wise match (MAE = 0)
        # 0 = maximum possible error (MAE = data_range)
        mae = np.mean(np.abs(gt_flat - pred_flat))
        similarity = 1 - (mae / data_range) if data_range > 0 else 1
        
        metrics_list.append({
            'cosine': cos_sim,
            'ssim': ssim_val,
            'norm_corr': norm_corr,
            'similarity': similarity
        })
    
    # Find the image with the greatest range for colorbar normalization
    max_range = 0
    vmin, vmax = 0, 1
    
    for img in images:
        img_min = np.min(img)
        img_max = np.max(img)
        img_range = img_max - img_min
        
        if img_range > max_range:
            max_range = img_range
            vmin = img_min
            vmax = img_max
    
    print(f"Normalizing colormap to range: [{vmin:.2f}, {vmax:.2f}]")
    
    # Determine grid size based on number of predictions
    n_total = len(predictions) + 1  # +1 for ground truth
    n_cols = min(3, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols  # Ceiling division
    
    # Create plot
    fig = plt.figure(figsize=(5.5 * n_cols, 5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3, right=0.85)
    
    # Plot ground truth first
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(images[0], cmap='viridis', vmin=vmin, vmax=vmax)
    ax0.set_title('Ground Truth', fontsize=12, fontweight='bold', pad=10)
    ax0.axis('off')
    
    # Plot predictions
    for idx, ((pred_path, train_site), img, metrics_dict) in enumerate(zip(predictions, images[1:], metrics_list), start=1):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Format title based on selected metric
        if metric == 'all':
            title = (f'Train: {train_site}\n'
                    f'Cosine={metrics_dict["cosine"]:.3f} | SSIM={metrics_dict["ssim"]:.3f}\n'
                    f'Corr={metrics_dict["norm_corr"]:.3f} | Sim={metrics_dict["similarity"]:.3f}')
        elif metric == 'cosine':
            title = f'Train: {train_site}\nCosine Similarity = {metrics_dict["cosine"]:.3f}'
        elif metric == 'ssim':
            title = f'Train: {train_site}\nSSIM = {metrics_dict["ssim"]:.3f}'
        elif metric == 'norm_corr':
            title = f'Train: {train_site}\nNorm. Correlation = {metrics_dict["norm_corr"]:.3f}'
        elif metric == 'similarity':
            title = f'Train: {train_site}\nSimilarity = {metrics_dict["similarity"]:.3f}'
        
        ax.set_title(title, fontsize=10, pad=10)
        ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Save and show
    output_file = f'comparison_{coord_pattern}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.show()
    
    return fig

# Usage:
find_and_plot_tiles(
    "726880_4709152", 
    sites_data_dir="/mnt/c/Users/402630/Desktop/SatCHM_copy/sites_data"
)

# Or with specific metric:
# find_and_plot_tiles(
#     "726880_4709152", 
#     sites_data_dir="/mnt/c/Users/402630/Desktop/SatCHM_copy/sites_data",
#     metric='cosine'  # Options: 'cosine', 'ssim', 'norm_corr', 'similarity', 'all'
# )