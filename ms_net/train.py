"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

"""
The main trainer script for MS-net.
"""
from glob import glob as gb
import os
import argparse
from dotenv import load_dotenv, find_dotenv
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from multiprocessing import freeze_support

try:
    from .network_2D_lightning import MS_Net
    from .ms_parser import parse_args
    from .pore_utils_2D import get_dataloader, load_hparams, num_input_channels
    from .plotting_utils import evaluate_validation_model
except (ImportError, ModuleNotFoundError):
    from network_2D_lightning import MS_Net
    from ms_parser import parse_args
    from pore_utils_2D import get_dataloader, load_hparams, num_input_channels
    from plotting_utils import evaluate_validation_model

import re
import warnings
import io


def setup_environment():
    """
    Description
    ___________
    This reads the .env and finds the project directory and site.

    Returns
    _______
    directory : str
        path to project directory
    site : str
        site being processed
    """
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    directory = os.getenv('project_path')
    site = os.getenv('site')
    return directory, site

def setup_params(sites):
    """
    Setup parameters for multiple sites.
    
    Parameters
    ----------
    sites : list of str
        List of site codes (e.g., ['fs_train', 'fw_large', 'qm'])
    
    Returns
    -------
    params : namespace
        Parameters including multi-site data path mapping
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from site_config import load_site_config
    finally:
        sys.path.pop(0)
    
    params = parse_args()
    
    # Store sites list
    params.sites = sites
    
    # Build data path mapping for each site
    params.data_paths = {}
    for site in sites:
        site_config = load_site_config(site)
        site_data_path = site_config['data_path']
        
        if not os.path.exists(site_data_path):
            raise ValueError(
                f"Data directory for site '{site}' not found: {site_data_path}\n"
                f"Run prepTrainInputs/main2.py for this site first."
            )
        params.data_paths[site] = site_data_path
        print(f"Site '{site}' data path: {site_data_path}")
    
    channel = params.channel
    if not channel :
        raise ValueError("Model channel cannot be empty or missing.")
    # received model architecture
    else :
        # contains base-ae-
        if "base-ae-" in channel :
            ae_dims = f"ae_{int(channel.removeprefix("base-ae-"))}"
            params.x_array = ['wvimg', 'solar', 'sensor', 'dem', ae_dims]
        # contains base-ae
        elif "base-ae" == channel :
            params.x_array = ['wvimg', 'solar', 'sensor', 'dem', 'ae']
        # contains base
        elif "base" == channel :
            params.x_array = ['wvimg', 'solar', 'sensor', 'dem']
        # will catch if an incorrect arg is passed, but that is handled within ms_parser.py anywasy
        else :
            raise ValueError(f"Incorrect model channel: {channel} cannot be used.")
    
    params.y_array = ['chm']
    params.x_xform = [None] * len(params.x_array)
    params.y_xform = [None]
    params.c_xform = [None]

    return params



def load_sample_lists(sites, data_paths):
    """
    Load and combine training/validation sample lists from multiple sites.
    
    This function reads site-specific trainlist.txt and vallist.txt files,
    prefixes each sample with its site code, and returns combined lists.
    Sample lists are kept in memory only and never written to disk or hparams.
    
    Parameters
    ----------
    sites : list of str
        List of site codes
    data_paths : dict
        Mapping of site codes to data directory paths
    
    Returns
    -------
    train_samples : list of str
        Combined training samples with site prefixes (e.g., ['fs_train:tile_0_0_0.tif', ...])
    val_samples : list of str
        Combined validation samples with site prefixes
    """
    train_samples = []
    val_samples = []
    
    for site in sites:
        site_train_list = os.path.join(data_paths[site], f'{site}_trainlist.txt')
        site_val_list = os.path.join(data_paths[site], f'{site}_vallist.txt')
        
        if not os.path.exists(site_train_list):
            raise ValueError(
                f"Training list not found for site '{site}': {site_train_list}\n"
                f"Run prepTrainInputs/main2.py for this site first."
            )
        
        if not os.path.exists(site_val_list):
            raise ValueError(
                f"Validation list not found for site '{site}': {site_val_list}\n"
                f"Run prepTrainInputs/main2.py for this site first."
            )
        
        # Read and prefix samples with site code
        with open(site_train_list, 'r') as f:
            site_train_samples = [f"{site}:{line.strip()}" for line in f if line.strip()]
            train_samples.extend(site_train_samples)
        
        with open(site_val_list, 'r') as f:
            site_val_samples = [f"{site}:{line.strip()}" for line in f if line.strip()]
            val_samples.extend(site_val_samples)
    
    print(f"Loaded {len(train_samples)} training samples from {len(sites)} site(s)")
    print(f"Loaded {len(val_samples)} validation samples from {len(sites)} site(s)")
    
    return train_samples, val_samples


def setup_net_dict(params):
    """
    Description
    ___________
    This creates a dictionary for hyperparameters of the neural network.
    
    Parameters
    __________
    params : namespace
        this contains specific arguments for the neural network

    Returns
    _______
    net_dict : dict
        a dictionary of hyperparameters for the neural network
    """
    net_dict = vars(params).copy()
    
    # Remove runtime data that shouldn't be in hparams.yaml
    runtime_keys = ['train_samples', 'val_samples', 'train_list', 'val_list']
    for key in runtime_keys:
        net_dict.pop(key, None)
    
    # Add normalization stats
    net_dict['uz_stats'] = {'scalar': 1}
    net_dict['edist_stats'] = {'scalar': 2.7}
    net_dict['c_stats'] = {'scalar': 1}
    net_dict['p_stats'] = {'scalar': 0}
    net_dict['D_stats'] = {'scalar': 0}
    return net_dict


# def find_latest_checkpoint(site):
#     """
#     Find the most recent best-val checkpoint for the given site.
    
#     Parameters
#     ----------
#     site : str
#         Site name (e.g., 'fs_train_ae')
    
#     Returns
#     -------
#     str
#         Path to latest best-val checkpoint
    
#     Raises
#     ------
#     RuntimeError
#         If no checkpoint found
#     """
#     model_dir = os.path.join('lightning_logs', f'{site}_model')
    
#     if not os.path.isdir(model_dir):
#         raise RuntimeError(
#             f"No model directory found: {model_dir}\n"
#             f"Train a model first before using --eval-only"
#         )
    
#     # Find all version directories (version_0, version_1, etc.)
#     versions = [
#         name for name in os.listdir(model_dir)
#         if os.path.isdir(os.path.join(model_dir, name))
#         and re.fullmatch(r'version_\d+', name)
#     ]
    
#     if not versions:
#         raise RuntimeError(f"No version directories found in {model_dir}")
    
#     # Get latest version
#     latest_version = max(versions, key=lambda x: int(x.split('_')[-1]))
#     latest_version_dir = os.path.join(model_dir, latest_version)
#     checkpoints_dir = os.path.join(latest_version_dir, 'checkpoints')
    
#     if not os.path.isdir(checkpoints_dir):
#         raise RuntimeError(f"No checkpoints directory in {latest_version_dir}")
    
#     # Find best-val checkpoint
#     ckpt_files = [
#         name for name in os.listdir(checkpoints_dir)
#         if name.startswith('best-val-epoch=') and name.endswith('.ckpt')
#     ]
    
#     if not ckpt_files:
#         raise RuntimeError(f"No best-val checkpoint found in {checkpoints_dir}")
    
#     if len(ckpt_files) > 1:
#         print(f"Warning: Multiple best-val checkpoints found, using first: {ckpt_files[0]}")
    
#     checkpoint_path = os.path.join(checkpoints_dir, ckpt_files[0])
#     print(f"Found latest checkpoint: {checkpoint_path}")
    
#     return checkpoint_path


def load_checkpoint_with_hparams(checkpoint_path):
    """
    Load model from checkpoint using its saved hyperparameters.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to .ckpt file
    
    Returns
    -------
    MS_Net
        Loaded model
    """
    # Derive hparams.yaml location (up 2 dirs from checkpoint)
    version_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    yaml_path = os.path.join(version_dir, 'hparams.yaml')
    
    if not os.path.exists(yaml_path):
        print(f"Warning: hparams.yaml not found at {yaml_path}")
        print("This appears to be a legacy checkpoint. Attempting to load with defaults...")
        
        # Legacy fallback: try to load with minimal params
        # PyTorch Lightning may auto-infer from checkpoint
        model = MS_Net.load_from_checkpoint(checkpoint_path)
        return model
    
    # Load hyperparameters from YAML
    print(f"Loading hyperparameters from {yaml_path}")
    hparams = load_hparams(yaml_path)
    
    # Extract architecture params
    net_name = hparams.get('net_name')
    num_scales = hparams.get('num_scales')
    num_features = hparams.get('num_features')
    num_filters = hparams.get('num_filters')
    f_mult = hparams.get('f_mult')
    
    # Load checkpoint with explicit architecture
    model = MS_Net.load_from_checkpoint(
        checkpoint_path,
        net_name=net_name,
        num_scales=num_scales,
        num_features=num_features,
        num_filters=num_filters,
        f_mult=f_mult
    )
    
    print(f"Model loaded: {num_features} input channels")
    return model


def setup_trainer(params, site, net_dict=None, new_model=False):
    """
    Setup PyTorch Lightning trainer with site-specific logging.
    
    Parameters
    __________
    params : namespace
        training parameters
    site : str
        site code (e.g., 'ws', 'lm', 'qm')
    
    Returns
    _______
    Trainer : pytorch_lightning.Trainer
        configured trainer with site-specific model directory
    """
    cbs = [
        # saves best model based on val_loss
        ModelCheckpoint(
            monitor="val_loss",
            filename="best-val-{epoch:02d}-{step:07d}",
            save_top_k=1,
            mode="min",
            save_on_train_epoch_end=False  
        ),

        # saves every 10 steps 
        ModelCheckpoint(
            filename="epoch-{epoch:02d}",
	    save_on_train_epoch_end=True,
            every_n_epochs=10,
            save_top_k=-1, 
        ),

        # EarlyStopping(
        #     monitor="val_loss",
        #     check_finite=False,
        #     mode="min",
        #     patience=300,
        #     min_delta=0.001,
        # ),
    ] 
    
    # Create site-specific logger - THIS IS THE KEY CHANGE!
    # Models will be saved to: lightning_logs/{site}_model/version_0/, version_1/, etc.
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{site}_model",
        version=None  # auto-increment version within this site's directory
    )
    
    # when there exists no hyperparameters file, create new one (for legacy models)
    if new_model and net_dict is not None :
        logger.log_hyperparams(net_dict)

    print(f"Models will be saved to: lightning_logs/{site}_model/")

    return Trainer(
        max_epochs=params.max_epochs,
        callbacks=cbs,
        logger=logger,  # site-specific logger
        plugins=None,
        precision="16-mixed",
        devices=1,
        accelerator="gpu",
        log_every_n_steps=10,
    )

def train_main(sites, NORM_CONST):
    """
    Main training function supporting multiple sites.
    
    Parameters
    ----------
    sites : list of str
        List of site codes to train on
    NORM_CONST : float
        Normalization constant for CHM
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from prepTrainInputs.utils import order_sites
    finally:
        sys.path.pop(0)
    
    directory, env_site = setup_environment()
    params = setup_params(sites)
    
    # Load sample lists separately (not part of params/net_dict)
    train_samples, val_samples = load_sample_lists(sites, params.data_paths)
    
    net_dict = setup_net_dict(params)

    if not params.model_loc :
        # create new model using channel
        model = MS_Net(
            net_name=params.net_name,
            num_scales=params.num_scales,
            num_features=num_input_channels(params.x_array),
            num_filters=params.num_filters,
            f_mult=params.f_mult,
            lr=params.LR,
            hparams=net_dict,
            steps=params.steps,
        )
        print(f"New MS_Net instantiated")
        print(f"  - Input channels: {num_input_channels(params.x_array)}")
        print(f"  - Scales: {params.num_scales}")
        print(f"  - Filters: {params.num_filters}")
        new_model = True

    else :
        checkpoint_path = params.model_loc
        if not os.path.isfile(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        # Load model from checkpoint (uses checkpoint's own architecture)
        model = load_checkpoint_with_hparams(checkpoint_path)
        new_model = False
    print(f"new_model: {new_model}")
    
    # Determine model name for logging
    if len(sites) == 1:
        model_name = sites[0]
    else:
        # Multi-site model naming
        model_name = order_sites(sites)
        model_name = f"multisite_{model_name}"
    
    print("\nLoading validation samples...")
    val_dataloader = get_dataloader(
        net_dict, ['val'],
        data_path=params.data_paths,
        NORM_CONST=NORM_CONST,
        train_samples=train_samples,
        val_samples=val_samples
    )
    if not params.eval_only and new_model :
        print("Training and Evaluating")
        # in train mode
        print("\nLoading training samples...")
        train_dataloader = get_dataloader(
            net_dict, ['train'],
            data_path=params.data_paths,
            NORM_CONST=NORM_CONST,
            train_samples=train_samples,
            val_samples=val_samples
        )
        trainer = setup_trainer(
            params,
            model_name,             # Use combined name for multi-site
            net_dict=net_dict,      # pass dictionary for writing hparams.yaml
            new_model=new_model     # pass whether this is a new model or not
        )
        try:
            trainer.fit(model, train_dataloader, val_dataloader['val'])
        except KeyboardInterrupt:
            print("\nTraining stopped by user (Ctrl+C).")
            # stats = evaluate_validation_model(
            #     model=model,
            #     val_dataloader=val_dataloader['val'],
            #     norm_const=NORM_CONST,
            #     plot_sample=True,
            #     data_point_index=0,
            #     save_plot=f"validation_comparison_{model_name}.png"
            # )
    else :
        # in eval-only mode
        print("Only Evaluating")

    stats = evaluate_validation_model(
        model=model,
        val_dataloader=val_dataloader['val'],
        norm_const=NORM_CONST,
        plot_sample=True,
        data_point_index=0,
        save_plot=f"validation_comparison_{model_name}.png"
    )
    buffer = io.StringIO()
    buffer.write("\n" + "=" * 60 + "\n")
    buffer.write("VALIDATION MODEL STATISTICS")
    buffer.write("\n" + "=" * 60 + "\n")
    buffer.write(f"Pixels evaluated:       {stats["n_pixels"]:,}\n")
    buffer.write(f"MSE:                    {stats["mse_m2"]:.4f} m²\n")
    buffer.write(f"RMSE:                   {stats["rmse_m"]:.4f} m\n")
    buffer.write(f"MAE:                    {stats["mae_m"]:.4f} m\n")
    buffer.write(f"Mean signed error:      {stats["mean_signed_error_m"]:.4f} m\n")
    buffer.write(f"Std. signed error:      {stats["std_signed_error_m"]:.4f} m\n")
    buffer.write(f"R²:                     {stats["r2"]:.4f}\n")
    buffer.write("\nObserved:\n")
    buffer.write(f"  Mean:                 {stats["observed_mean_m"]:.4f} m\n")
    buffer.write(f"  Std:                  {stats["observed_std_m"]:.4f} m\n")
    buffer.write("\nPredicted:\n")
    buffer.write(f"  Mean:                 {stats["predicted_mean_m"]:.4f} m\n")
    buffer.write(f"  Std:                  {stats["predicted_std_m"]:.4f} m\n")
    buffer.write("=" * 60 + "\n")
    print(buffer.getvalue())
    with open(f"model_metrics_{model_name}.txt", "w", encoding="utf-8") as file :
        file.write(buffer.getvalue())
    buffer.close()

def main():
    # freeze_support()
    # Parse arguments using ms_parser (includes --sites flag)
    params = parse_args()
    
    # Get sites from command line (now required)
    sites = params.sites
    if not sites :
        raise ValueError("Sites cannot be empty or missing.")
    print(f"Training on site(s): {', '.join(sites)}")
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from site_config import load_site_config
    finally:
        sys.path.pop(0)
    
    # Validate that site data exists
    for site in sites:
        site_config = load_site_config(site)
        data_path = site_config['data_path']
        if not os.path.exists(data_path):
            raise ValueError(
                f"Data directory not found for site '{site}': {data_path}\n"
                f"Run data preparation for this site first."
            )
        print(f"Site '{site}' data directory: {data_path}")
    
    # Use norm_const from params (with underscore, not hyphen)
    NORM_CONST = getattr(params, 'norm_const', 46)
    train_main(sites=sites, NORM_CONST=NORM_CONST)

    
if __name__ == '__main__':
    main()
