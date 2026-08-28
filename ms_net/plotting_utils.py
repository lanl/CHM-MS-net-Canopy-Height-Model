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




def plot_cs(y, yhat, error='L1', title=None, save_as=None):

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
    # 1. Observed
    # --------------------------------------------------
    plt.subplot(1, 4, 1)
    im = plt.imshow(y_np)
    colorbar(im)
    plt.title('Observed')

    # --------------------------------------------------
    # 2. Predicted
    # --------------------------------------------------
    plt.subplot(1, 4, 2)
    im = plt.imshow(
        yh_np,
        clim=(np.nanmin(y_np), np.nanmax(y_np))
    )
    colorbar(im)
    plt.title('Predicted')

    # --------------------------------------------------
    # 3. Absolute error
    # --------------------------------------------------
    plt.subplot(1, 4, 3)
    im = plt.imshow(
        abs_np,
        clim=(0, np.nanmax(abs_np))
    )
    colorbar(im)
    plt.title('Absolute error')

    # --------------------------------------------------
    # 4. Signed error
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

    # plt.subplot(1,3,1)
    # im = plt.imshow(y_cs); 
    # colorbar(im)
    # plt.title('y')
    # plt.subplot(1,3,2)
    # im = plt.imshow(yh_cs, clim=(y_cs.min(), y_cs.max()));
    # colorbar(im)
    # plt.title('$\hat{y}$')
    # plt.subplot(1,3,3)
    # im = plt.imshow(e_cs, clim=(0, y_cs.max()));
    # colorbar(im)
    # plt.title(f'{error} error')
    
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(9, 3)
    # #plt.show()
    
    # if title:
    #     plt.suptitle(title)
    
    # if save_as:    
    #     plt.savefig(save_as)
    
    
    
def evaluate_validation_model(
    model,
    val_dataloader,
    norm_const=46,
    device=None,
    max_samples=None,
    plot_sample=True,
    save_plot=None
):
    """
    Evaluate a trained MS-Net model on the validation dataset.

    Calculates:
        - RMSE
        - MAE
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
                    observed[0],
                    predicted[0],
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

    print("\n" + "=" * 60)
    print("VALIDATION MODEL STATISTICS")
    print("=" * 60)

    print(f"Pixels evaluated:       {len(observed):,}")
    print(f"MSE:                    {mse:.4f} m²")
    print(f"RMSE:                   {rmse:.4f} m")
    print(f"MAE:                    {mae:.4f} m")
    print(f"Mean signed error:      {mean_signed_error:.4f} m")
    print(f"Std. signed error:      {std_signed_error:.4f} m")
    print(f"R²:                     {r2:.4f}")

    print("\nObserved:")
    print(f"  Mean:                 {observed_mean:.4f} m")
    print(f"  Std:                  {observed_std:.4f} m")

    print("\nPredicted:")
    print(f"  Mean:                 {predicted_mean:.4f} m")
    print(f"  Std:                  {predicted_std:.4f} m")

    print("=" * 60)

    return stats