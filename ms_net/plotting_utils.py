"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

import numpy as np
import torch
import matplotlib
import h5py
from hdf5storage import loadmat
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import os
from PIL import Image


params = {
    #'text.latex.preamble': '\\usepackage{gensymb}',
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'inferno',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'figure.dpi': 300,
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8, # was 10
    'legend.fontsize': 6, # was 10
    'xtick.labelsize': 4,
    'ytick.labelsize': 4,
    #'text.usetex': True,
    'figure.figsize': [3.39, 2.10],
    'font.family': 'serif',
}
matplotlib.rcParams.update(params)




def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar




def plot_site_inference(
    predicted_raster: str,
    observed_raster: str = None,
    imagery_raster: str = None,
    output_path: str = None,
    title: str = None,
    downsample_factor: int = 4
):
    """
    Plot full-site inference results from georeferenced rasters.
    
    Parameters
    ----------
    predicted_raster : str
        Path to predicted CHM GeoTIFF (required)
    observed_raster : str, optional
        Path to observed CHM GeoTIFF (for evaluation)
    imagery_raster : str, optional
        Path to panchromatic imagery GeoTIFF (for context)
    output_path : str, optional
        Path to save the plot. If None, displays interactively
    title : str, optional
        Main title for the figure
    downsample_factor : int, optional
        Factor to downsample large rasters for visualization (default: 4)
    
    Returns
    -------
    None
        Saves plot to output_path or displays it
    """
    
    def read_and_downsample(raster_path, factor=1, window=None):
        """Read a raster and optionally downsample it."""
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Raster not found: {raster_path}")
        
        with rasterio.open(raster_path) as src:
            # Read at reduced resolution if downsampling
            if factor > 1:
                if window:
                    # Adjust window for downsampling
                    from rasterio.windows import Window
                    data = src.read(
                        1,
                        window=window,
                        out_shape=(
                            window.height // factor,
                            window.width // factor
                        ),
                        resampling=Resampling.average
                    )
                else:
                    data = src.read(
                        1,
                        out_shape=(
                            src.height // factor,
                            src.width // factor
                        ),
                        resampling=Resampling.average
                    )
            else:
                if window:
                    data = src.read(1, window=window)
                else:
                    data = src.read(1)
            
            # Handle nodata
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            
            # DEFENSIVE: Also mask extreme negative values that might be corrupted NoData
            # (e.g., scaled NoData like -5710 from -9999 * 0.572)
            data = np.where(data < -1000, np.nan, data)
            
            return data.astype(np.float32)
    
    # Load predicted CHM (required)
    print(f"Loading predicted CHM: {predicted_raster}")
    pred = read_and_downsample(predicted_raster, downsample_factor)
    
    # Get predicted raster bounds for alignment
    with rasterio.open(predicted_raster) as pred_src:
        pred_bounds = pred_src.bounds
        pred_crs = pred_src.crs
    
    # Load observed CHM (optional) - align to predicted extent
    obs = None
    pred_display = pred
    if observed_raster and os.path.exists(observed_raster):
        print(f"Loading observed CHM: {observed_raster}")
        
        # Read only the portion that overlaps with predicted
        from rasterio.windows import from_bounds
        with rasterio.open(observed_raster) as obs_src:
            # Calculate window in observed raster that corresponds to predicted bounds
            try:
                window = from_bounds(*pred_bounds, transform=obs_src.transform)
                # Round window bounds to integers
                from rasterio.windows import Window
                window = Window(
                    col_off=int(window.col_off),
                    row_off=int(window.row_off),
                    width=int(window.width),
                    height=int(window.height)
                )
                obs = read_and_downsample(observed_raster, downsample_factor, window=window)
                
                # Ensure exact shape match (handle rounding differences)
                if obs.shape != pred.shape:
                    min_h = min(obs.shape[0], pred.shape[0])
                    min_w = min(obs.shape[1], pred.shape[1])
                    obs = obs[:min_h, :min_w]
                    pred_display = pred[:min_h, :min_w]
                    
            except Exception as e:
                print(f"Warning: Could not align observed to predicted bounds: {e}")
                print("Reading full observed raster")
                obs = read_and_downsample(observed_raster, downsample_factor)
    
    # Load imagery (optional)
    img = None
    if imagery_raster and os.path.exists(imagery_raster):
        print(f"Loading imagery: {imagery_raster}")
        img = read_and_downsample(imagery_raster, downsample_factor)
    
    # Determine layout based on available data
    if obs is not None:
        # Full evaluation layout: imagery, observed, predicted, error
        n_panels = 4 if img is not None else 3
        fig_width = 12 if n_panels == 4 else 9
    else:
        # Prediction-only layout: imagery, predicted
        n_panels = 2 if img is not None else 1
        fig_width = 6 if n_panels == 2 else 3
    
    plt.figure(figsize=(fig_width, 3))
    
    panel_idx = 1
    
    # Panel: Panchromatic imagery
    if img is not None:
        plt.subplot(1, n_panels, panel_idx)
        im = plt.imshow(img, cmap='gray', origin='upper')
        colorbar(im)
        plt.title('Panchromatic')
        plt.axis('off')
        panel_idx += 1
    
    # Panel: Observed CHM
    if obs is not None:
        plt.subplot(1, n_panels, panel_idx)
        vmin, vmax = np.nanpercentile(obs, [2, 98])
        im = plt.imshow(obs, cmap='inferno', vmin=vmin, vmax=vmax, origin='upper')
        colorbar(im)
        plt.title('Observed CHM (m)')
        plt.axis('off')
        panel_idx += 1
    
    # Panel: Predicted CHM
    plt.subplot(1, n_panels, panel_idx)
    if obs is not None:
        # Use same scale as observed for comparison
        vmin, vmax = np.nanpercentile(obs, [2, 98])
        pred_to_plot = pred_display
    else:
        vmin, vmax = np.nanpercentile(pred_display, [2, 98])
        pred_to_plot = pred_display
    im = plt.imshow(pred_to_plot, cmap='inferno', vmin=vmin, vmax=vmax, origin='upper')
    colorbar(im)
    plt.title('Predicted CHM (m)')
    plt.axis('off')
    panel_idx += 1
    
    # Panel: Error (only if observed is available)
    if obs is not None:
        plt.subplot(1, n_panels, panel_idx)
        
        # Use the aligned versions
        obs_plot = obs
        pred_plot = pred_display
        
        # Ensure shapes match
        if obs_plot.shape != pred_plot.shape:
            print(f"Warning: Shape mismatch after alignment - Observed: {obs_plot.shape}, Predicted: {pred_plot.shape}")
            # Crop to minimum common shape
            min_h = min(obs_plot.shape[0], pred_plot.shape[0])
            min_w = min(obs_plot.shape[1], pred_plot.shape[1])
            obs_plot = obs_plot[:min_h, :min_w]
            pred_plot = pred_plot[:min_h, :min_w]
        
        # Signed error: observed - predicted
        error = obs_plot - pred_plot
        
        # Symmetric color scale
        max_abs_error = np.nanpercentile(np.abs(error), 98)
        im = plt.imshow(
            error,
            cmap='RdBu_r',
            vmin=-max_abs_error,
            vmax=max_abs_error,
            origin='upper'
        )
        colorbar(im)
        plt.title('Error: Obs - Pred (m)')
        plt.axis('off')
        
        # Calculate and print metrics
        valid_mask = np.isfinite(obs_plot) & np.isfinite(pred_plot)
        if np.any(valid_mask):
            obs_valid = obs_plot[valid_mask]
            pred_valid = pred_plot[valid_mask]
            error_valid = error[valid_mask]
            
            mae = np.mean(np.abs(error_valid))
            rmse = np.sqrt(np.mean(error_valid**2))
            bias = np.mean(error_valid)
            
            print(f"\nSite-wide metrics:")
            print(f"  MAE:  {mae:.2f} m")
            print(f"  RMSE: {rmse:.2f} m")
            print(f"  Bias: {bias:.2f} m")
            print(f"  Valid pixels: {np.sum(valid_mask):,}")
    
    if title:
        plt.suptitle(title, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cs(y, yhat, pan=None, error='L1', title=None, save_as=None):


    y_cs = y.cpu().squeeze()
    yh_cs = yhat.cpu().squeeze()

    # Absolute error
    absolute_error = torch.abs(y_cs - yh_cs)

    # Signed error: observed - predicted
    signed_error = y_cs - yh_cs

    # Convert to numpy for plotting/statistics
    y_np = y_cs.numpy()
    yh_np = yh_cs.numpy()
    abs_np = absolute_error.numpy()
    signed_np = signed_error.numpy()

    # Symmetric color scale for signed error
    max_abs_error = np.nanmax(np.abs(signed_np))

    plt.figure(figsize=(12, 3))

    # --------------------------------------------------
    # 1. Panchromatic Image
    # --------------------------------------------------

    plt.subplot(1, 4, 1)
    if pan.ndim == 2:
        im = plt.imshow(pan, cmap='gray')
    else:
        im = plt.imshow(pan)
    colorbar(im)
    plt.title('Panchromatic')

    # --------------------------------------------------
    # 2. Observed
    # --------------------------------------------------
    plt.subplot(1, 4, 2)
    im = plt.imshow(y_np)
    colorbar(im)
    plt.title('Observed')

    # --------------------------------------------------
    # 3. Predicted
    # --------------------------------------------------
    plt.subplot(1, 4, 3)
    im = plt.imshow(
        yh_np,
        clim=(np.nanmin(y_np), np.nanmax(y_np))
    )
    colorbar(im)
    plt.title('Predicted')

    # # --------------------------------------------------
    # # 4. Absolute error
    # # --------------------------------------------------
    # plt.subplot(1, 4, 3)
    # im = plt.imshow(
    #     abs_np,
    #     clim=(0, np.nanmax(abs_np))
    # )
    # colorbar(im)
    # plt.title('Absolute error')

    # --------------------------------------------------
    # 5. Signed error
    # --------------------------------------------------
    plt.subplot(1, 4, 4)

    im = plt.imshow(
        signed_np,
        cmap='RdBu_r',
        clim=(-max_abs_error, max_abs_error)
    )

    colorbar(im)
    plt.title('Signed error\nObserved - Predicted')

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')

    return plt.gcf()
    
    
    
def evaluate_validation_model(
    model,
    val_dataloader,
    norm_const=46,
    device=None,
    max_samples=None,
    plot_sample=True,
    save_plot=None,
    data_point_index=0
):
    """
    Evaluate a trained MS-Net model on the validation dataset.

    Calculates:
        - RMSE
        - Mean absolute error (MAE)
        - Mean signed error (bias)
        - Standard deviation of signed error
        - R²
        - Mean observed height
        - Mean predicted height
        - Standard deviation of observed height
        - Standard deviation of predicted height

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    val_dataloader : DataLoader
        Validation dataloader.

    norm_const : float
        Normalization constant used for CHM data.

    device : torch.device, optional
        Device on which to run the model.

    max_samples : int, optional
        Maximum number of validation samples to evaluate.

    plot_sample : bool
        Whether to generate a four-panel observed/predicted/error plot.

    save_plot : str, optional
        Path for saving the plot.

    Returns
    -------
    stats : dict
        Dictionary containing validation metrics.
    """

    import numpy as np
    import torch

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_observed = []
    all_predicted = []

    plotted = False

    with torch.no_grad():

        for batch_idx, batch in enumerate(val_dataloader):

            if max_samples is not None and batch_idx >= max_samples:
                break

            sample, masks, xy = batch

            x, y = xy[0], xy[1]

            x = [
                xi.to(device) if torch.is_tensor(xi) else xi
                for xi in x
            ]

            masks = [
                m.to(device) if torch.is_tensor(m) else m
                for m in masks
            ]

            # # Debug code
            # print("Number of x features:", len(x))
            # for i, feature in enumerate(x):
            #     print(f"x[{i}] type={type(feature)}")
            #     if isinstance(feature, (list, tuple)):
            #         print(f"  number of scales: {len(feature)}")
            #         for j, scale in enumerate(feature):
            #             print(f"  x[{i}][{j}] shape={scale.shape}")
            #     else:
            #         print(f"  shape={feature.shape}")
            
            # Highest-resolution input
            pan = x[2][data_point_index]
            #print("Finest-resolution input shape:", pan.shape)
            pan = pan.detach().cpu()
            pan = pan[0]

            y_pred = model(x, masks)

            # Finest scale is the final scale
            observed = y[-1]
            predicted = y_pred[-1]

            # Convert from normalized CHM back to meters
            observed = observed.detach().cpu() * norm_const
            predicted = predicted.detach().cpu() * norm_const

            all_observed.append(observed.numpy().ravel())
            all_predicted.append(predicted.numpy().ravel())

            # Plot first validation example
            if plot_sample and not plotted:
                plot_cs(
                    observed[data_point_index],
                    predicted[data_point_index],
                    pan=pan,
                    error='L1',
                    title=f'Validation sample {batch_idx}',
                    save_as=save_plot
                )
                plotted = True

    observed = np.concatenate(all_observed)
    predicted = np.concatenate(all_predicted)

    # Remove invalid values
    valid = np.isfinite(observed) & np.isfinite(predicted)

    observed = observed[valid]
    predicted = predicted[valid]

    # Signed error: observed - predicted
    signed_error = observed - predicted

    # Absolute error
    absolute_error = np.abs(signed_error)

    # Squared error
    squared_error = signed_error ** 2

    # Metrics
    mse = np.mean(squared_error)
    rmse = np.sqrt(mse)
    mae = np.mean(absolute_error)

    mean_signed_error = np.mean(signed_error)
    std_signed_error = np.std(signed_error)

    observed_mean = np.mean(observed)
    predicted_mean = np.mean(predicted)

    observed_std = np.std(observed)
    predicted_std = np.std(predicted)

    # R²
    ss_res = np.sum(squared_error)
    ss_tot = np.sum((observed - observed_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    stats = {
        "n_pixels": len(observed),
        "mse_m2": mse,
        "rmse_m": rmse,
        "mae_m": mae,
        "mean_signed_error_m": mean_signed_error,
        "std_signed_error_m": std_signed_error,
        "observed_mean_m": observed_mean,
        "predicted_mean_m": predicted_mean,
        "observed_std_m": observed_std,
        "predicted_std_m": predicted_std,
        "r2": r2,
    }
    
    return stats


def save_clean_prediction_png(
    predicted_raster,
    output_path,
    downsample_factor=4,
    colormap='inferno',
    percentile_range=(2, 98)
):
    """
    Generate clean colormap PNG of prediction without matplotlib decorations.
    
    Creates a pure RGB image with colormap applied to prediction values,
    without any axes, labels, titles, or plot decorations. Suitable for
    use as a basemap overlay or standalone visualization.
    
    Parameters
    ----------
    predicted_raster : str
        Path to predicted CHM GeoTIFF.
    output_path : str
        Path to save output PNG.
    downsample_factor : int, optional
        Downsampling factor for visualization (default: 4).
    colormap : str, optional
        Matplotlib colormap name (default: 'inferno').
    percentile_range : tuple, optional
        (min, max) percentiles for auto-scaling (default: (2, 98)).
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'output_path': Path to saved PNG
        - 'shape': (width, height) of output image
        - 'value_range': (vmin, vmax) used for scaling
    """
    
    # Read and downsample raster
    with rasterio.open(predicted_raster) as src:
        if downsample_factor > 1:
            data = src.read(
                1,
                out_shape=(
                    src.height // downsample_factor,
                    src.width // downsample_factor
                ),
                resampling=Resampling.average
            )
        else:
            data = src.read(1)
        
        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
    
    data = data.astype(np.float32)
    
    # DEFENSIVE: Also mask extreme negative values that might be corrupted NoData
    data = np.where(data < -1000, np.nan, data)
    
    # Auto-scale using percentiles
    vmin, vmax = np.nanpercentile(data, percentile_range)
    
    # Normalize to [0, 1]
    data_norm = (data - vmin) / (vmax - vmin)
    data_norm = np.clip(data_norm, 0, 1)
    
    # Handle NaN - set to 0 (black)
    data_norm = np.nan_to_num(data_norm, nan=0.0)
    
    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    rgba = cmap(data_norm)
    
    # Convert to uint8 RGB (drop alpha channel)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    
    # Save with PIL (no matplotlib figure overhead)
    img = Image.fromarray(rgb)
    img.save(output_path)
    
    print(f"Saved clean prediction image to: {output_path}")
    print(f"  Shape: {rgb.shape[1]}x{rgb.shape[0]} pixels")
    print(f"  Value range: [{vmin:.2f}, {vmax:.2f}] m")
    
    return {
        'output_path': output_path,
        'shape': (rgb.shape[1], rgb.shape[0]),
        'value_range': (vmin, vmax)
    }


def main() :
    evaluate_validation_model(
        
    )


if __name__ == '__main__' :
    main()