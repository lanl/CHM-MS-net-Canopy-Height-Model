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

from network_2D_lightning import MS_Net
from ms_parser import parse_args
from pore_utils_2D import get_dataloader, load_hparams, num_input_channels
from plotting_utils import evaluate_validation_model
import re
import warnings


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
    
    # Build in-memory combined trainlist/vallist (no file saved)
    params.train_samples = []
    params.val_samples = []
    
    for site in sites:
        site_train_list = os.path.join(params.data_paths[site], f'{site}_trainlist.txt')
        site_val_list = os.path.join(params.data_paths[site], f'{site}_vallist.txt')
        
        if not os.path.exists(site_train_list):
            raise ValueError(
                f"Training list not found for site '{site}': {site_train_list}\n"
                f"Run prepTrainInputs/main2.py for this site first."
            )
        
        # Read and prefix with site code
        with open(site_train_list, 'r') as f:
            site_train_samples = [f"{site}:{line.strip()}" for line in f if line.strip()]
            params.train_samples.extend(site_train_samples)
        
        with open(site_val_list, 'r') as f:
            site_val_samples = [f"{site}:{line.strip()}" for line in f if line.strip()]
            params.val_samples.extend(site_val_samples)
    
    print(f"✓ Loaded {len(params.train_samples)} training samples from {len(sites)} site(s)")
    print(f"✓ Loaded {len(params.val_samples)} validation samples from {len(sites)} site(s)")
    
    # Create temporary list files for backward compatibility with get_dataloader
    # (These will be read by return_fields in pore_utils_2D.py)
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix='msnet_multisite_')
    
    params.train_list = os.path.join(tmpdir, 'combined_trainlist.txt')
    params.val_list = os.path.join(tmpdir, 'combined_vallist.txt')
    
    with open(params.train_list, 'w') as f:
        f.write('\n'.join(params.train_samples))
    
    with open(params.val_list, 'w') as f:
        f.write('\n'.join(params.val_samples))
    
    # Set x_array based on --use-ae and --pca-dims (as before)
    if getattr(params, 'use_ae', False):
        if getattr(params, 'pca_dims', None) is None:
            params.x_array = ['wvimg', 'solar', 'sensor', 'dem', 'ae']
        else:
            ae_dims = f'ae_{params.pca_dims}'
            params.x_array = ['wvimg', 'solar', 'sensor', 'dem', ae_dims]
    else:
        if getattr(params, 'pca_dims', None) is not None:
            warnings.warn("--pca-dims must only be specified alongside --use-ae.\nUsing non-ae dims...")
        params.x_array = ['wvimg', 'solar', 'sensor', 'dem']
    
    params.y_array = ['chm']
    params.x_xform = [None] * len(params.x_array)
    params.y_xform = [None]
    params.c_xform = [None]
    
    return params


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
    net_dict = vars(params)
    net_dict['uz_stats'] = {'scalar': 1}
    net_dict['edist_stats'] = {'scalar': 2.7}
    net_dict['c_stats'] = {'scalar': 1}
    net_dict['p_stats'] = {'scalar': 0}
    net_dict['D_stats'] = {'scalar': 0}
    return net_dict

def find_latest_checkpoint(site):
    """
    Find the most recent best-val checkpoint for the given site.
    
    Parameters
    ----------
    site : str
        Site name (e.g., 'fs_train_ae')
    
    Returns
    -------
    str
        Path to latest best-val checkpoint
    
    Raises
    ------
    RuntimeError
        If no checkpoint found
    """
    model_dir = os.path.join('lightning_logs', f'{site}_model')
    
    if not os.path.isdir(model_dir):
        raise RuntimeError(
            f"No model directory found: {model_dir}\n"
            f"Train a model first before using --eval-only"
        )
    
    # Find all version directories (version_0, version_1, etc.)
    versions = [
        name for name in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, name))
        and re.fullmatch(r'version_\d+', name)
    ]
    
    if not versions:
        raise RuntimeError(f"No version directories found in {model_dir}")
    
    # Get latest version
    latest_version = max(versions, key=lambda x: int(x.split('_')[-1]))
    latest_version_dir = os.path.join(model_dir, latest_version)
    checkpoints_dir = os.path.join(latest_version_dir, 'checkpoints')
    
    if not os.path.isdir(checkpoints_dir):
        raise RuntimeError(f"No checkpoints directory in {latest_version_dir}")
    
    # Find best-val checkpoint
    ckpt_files = [
        name for name in os.listdir(checkpoints_dir)
        if name.startswith('best-val-epoch=') and name.endswith('.ckpt')
    ]
    
    if not ckpt_files:
        raise RuntimeError(f"No best-val checkpoint found in {checkpoints_dir}")
    
    if len(ckpt_files) > 1:
        print(f"Warning: Multiple best-val checkpoints found, using first: {ckpt_files[0]}")
    
    checkpoint_path = os.path.join(checkpoints_dir, ckpt_files[0])
    print(f"Found latest checkpoint: {checkpoint_path}")
    
    return checkpoint_path


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


def warn_if_flags_mismatch(checkpoint_path, params):
    """
    Warn user if current CLI flags don't match checkpoint architecture.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint file
    params : namespace
        Current command-line parameters
    """
    version_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    yaml_path = os.path.join(version_dir, 'hparams.yaml')
    
    if not os.path.exists(yaml_path):
        # Can't check without hparams - skip warning
        return
    
    hparams = load_hparams(yaml_path)
    checkpoint_channels = hparams.get('num_features')
    current_channels = num_input_channels(params.x_array)
    
    if checkpoint_channels != current_channels:
        print(f"Architecture Mismatch Warning")
        print(f"Checkpoint was trained with: {checkpoint_channels} input channels")
        print(f"Your current flags specify:  {current_channels} input channels")
        print(f"\nThe checkpoint's architecture will be used (flags ignored).")


def create_new_model(params, net_dict):
    """
    Create a brand new MS_Net model for training.
    
    Parameters
    ----------
    params : namespace
        Training parameters
    net_dict : dict
        Hyperparameter dictionary
    
    Returns
    -------
    MS_Net
        Newly instantiated model
    """
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
    
    return model

def load_or_create_model(params, net_dict):
    """
    Load existing checkpoint (eval mode) or create new model (train mode).
    
    Parameters
    __________
    params : namespace
        Command line arguments
    net_dict : dict
        Dictionary of hyperparameters for the neural network
    
    Returns
    _______
    tuple(MS_Net, bool)
        Model and whether it's newly created (True) or loaded (False)
    """
    
    if params.eval_only:
        # ============ EVALUATION MODE ============
        # Load existing checkpoint, ignore current architecture flags
        
        if params.model_loc:
            checkpoint_path = params.model_loc
            if not os.path.isfile(checkpoint_path):
                raise ValueError(f"Checkpoint not found: {checkpoint_path}")
        else:
            checkpoint_path = find_latest_checkpoint(params.site)
        
        print(f"\n{'='*70}")
        print(f"EVALUATION MODE: Loading checkpoint")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*70}\n")
        
        # Warn if user-provided flags don't match checkpoint
        warn_if_flags_mismatch(checkpoint_path, params)
        
        # Load model from checkpoint (uses checkpoint's own architecture)
        model = load_checkpoint_with_hparams(checkpoint_path)
        return model, False
    
    else:
        # ============ TRAINING MODE ============
        # Create new model using current flags
        
        if params.model_loc:
            raise ValueError(
                "--model-loc can only be used with --eval-only\n"
                "To train a new model, remove --model-loc flag"
            )
        
        print(f"\n{'='*70}")
        print(f"TRAINING MODE: Creating new model")
        print(f"Architecture: {num_input_channels(params.x_array)} input channels")
        print(f"{'='*70}\n")
        
        model = create_new_model(params, net_dict)
        return model, True


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
    net_dict = setup_net_dict(params)
    model, new_model = load_or_create_model(params, net_dict)
    print(f"new_model: {new_model}")
    
    # Determine model name for logging
    if len(sites) == 1:
        model_name = sites[0]
    else:
        # Multi-site model naming
        model_name = order_sites(sites)
        model_name = f"multisite_{model_name}"
    
    # only create trainer when we are training a model
    if not params.eval_only:
        print("\nLoading training and validation samples...\n")
        train_dataloader = get_dataloader(net_dict, ['train'], data_path=params.data_paths, NORM_CONST=NORM_CONST)
        val_dataloader = get_dataloader(net_dict, ['val'], data_path=params.data_paths, NORM_CONST=NORM_CONST)
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
    else:
        print("\nLoading only validation samples...\n")
        val_dataloader = get_dataloader(net_dict, ['val'], data_path=params.data_paths, NORM_CONST=NORM_CONST)
    
    evaluate_validation_model(
        model=model,
        val_dataloader=val_dataloader['val'],
        norm_const=NORM_CONST,
        plot_sample=True,
        data_point_index=0,
        save_plot=f"validation_comparison_{model_name}.png"
    )

def main():
    # freeze_support()
    # Parse arguments using ms_parser (includes --sites flag)
    params = parse_args()
    
    # Get sites from command line (now required)
    sites = params.sites
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
        print(f"  ✓ Site '{site}' data directory: {data_path}")
    
    # Use norm_const from params (with underscore, not hyphen)
    NORM_CONST = getattr(params, 'norm_const', 46)
    train_main(sites=sites, NORM_CONST=NORM_CONST)

    
if __name__ == '__main__':
    main()
