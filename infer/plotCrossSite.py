import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

def find_and_plot_tiles(key_pattern, base_dir='.', metric='all'):
    """
    Find 9 tiles matching key_pattern and create a 3x3 plot with normalized viridis colormap.
    
    Args:
        key_pattern: Substring to match (e.g., "⋆⋆X_YYY")
        base_dir: Base directory to search in
        metric: Which metric to display ('r2', 'correlation', 'ssim', 'nrmse', or 'all')
    """
    # Find all matching files
    matching_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if key_pattern in file and (file.endswith('.png') or file.endswith('.tif') or file.endswith('.jpg')):
                matching_files.append(os.path.join(root, file))
    
    # Separate ground truth from predictions
    ground_truth = None
    predictions = []
    
    for file_path in matching_files:
        # Ground truth: in 'chm' folder but NOT in 'chm_preds'
        if 'chm_preds' in file_path:
            predictions.append(file_path)
        elif 'chm' in file_path:
            ground_truth = file_path
    
    # Verify we have the right number of files
    if ground_truth is None:
        raise ValueError(f"No ground truth file found in 'chm' folder (excluding 'chm_preds')")
    if len(predictions) != 8:
        raise ValueError(f"Expected 8 prediction files in 'chm_preds', found {len(predictions)}")
    
    # Extract site names and sort predictions
    pred_info = []
    for pred_path in predictions:
        filename = os.path.basename(pred_path)
        # Extract site from pattern "train_{SITE}_*"
        if 'train_' in filename:
            site = filename.split('train_')[1].split('_')[0]
        else:
            site = "Unknown"
        pred_info.append((pred_path, site))
    
    pred_info.sort(key=lambda x: x[1])  # Sort by site name
    
    # Load all images and find the one with greatest range
    all_paths = [ground_truth] + [p[0] for p in pred_info]
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
    gt_mean = np.mean(gt_flat)
    gt_std = np.std(gt_flat)
    
    # Calculate metrics for each prediction
    metrics_list = []
    for pred_img in images[1:]:
        pred_flat = pred_img.flatten()
        
        # R² score
        ss_res = np.sum((gt_flat - pred_flat) ** 2)
        ss_tot = np.sum((gt_flat - gt_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Pearson correlation
        corr, _ = pearsonr(gt_flat, pred_flat)
        
        # SSIM (requires same data range)
        data_range = gt_image.max() - gt_image.min()
        ssim_val = ssim(gt_image, pred_img, data_range=data_range)
        
        # Normalized RMSE
        rmse = np.sqrt(np.mean((gt_flat - pred_flat) ** 2))
        nrmse = rmse / gt_std if gt_std != 0 else 0
        
        metrics_list.append({
            'r2': r2,
            'corr': corr,
            'ssim': ssim_val,
            'nrmse': nrmse
        })
    
    # Find the image with the greatest range (max - min)
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
    
    print(f"Normalizing to range: [{vmin:.2f}, {vmax:.2f}]")
    
    # Create 3x3 plot with space for colorbar
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, right=0.85)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)])
    
    # Plot ground truth in top-left (0, 0)
    im = axes[0, 0].imshow(images[0], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Plot predictions in remaining positions
    positions = [(i, j) for i in range(3) for j in range(3)]
    positions.remove((0, 0))  # Remove top-left
    
    for idx, ((pred_path, site), img, metrics_dict) in enumerate(zip(pred_info, images[1:], metrics_list)):
        i, j = positions[idx]
        axes[i, j].imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Format title based on selected metric
        if metric == 'all':
            title = f'Site: {site}\nR²={metrics_dict["r2"]:.3f} | ρ={metrics_dict["corr"]:.3f} | SSIM={metrics_dict["ssim"]:.3f}'
        elif metric == 'r2':
            title = f'Site: {site} | R² = {metrics_dict["r2"]:.3f}'
        elif metric == 'correlation':
            title = f'Site: {site} | ρ = {metrics_dict["corr"]:.3f}'
        elif metric == 'ssim':
            title = f'Site: {site} | SSIM = {metrics_dict["ssim"]:.3f}'
        elif metric == 'nrmse':
            title = f'Site: {site} | NRMSE = {metrics_dict["nrmse"]:.3f}'
        
        axes[i, j].set_title(title, fontsize=10)
        axes[i, j].axis('off')
    
    # Add colorbar to the right of all subplots
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    
    plt.savefig(f'comparison_{key_pattern}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

# Usage - same as before:
find_and_plot_tiles("10300100632D9700_431536_3544816", 
                    base_dir="/mnt/c/Users/402630/Desktop/SatCHM_copy/fortStewart_data")

# Or try different metrics:
# find_and_plot_tiles("10300100632D9700_431536_3544816", 
#                     base_dir="/mnt/c/Users/402630/Desktop/SatCHM_copy/fortStewart_data",
#                     metric='r2')