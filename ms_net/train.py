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
from pore_utils_2D import get_dataloader, load_hparams

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

def setup_model_storage(site):
    """
    Set up model storage outside the repo with symlink back to expected location.
    
    Parameters
    __________
    site : str
        Site code (e.g., 'ws', 'qm', 'lm')
    
    Returns
    _______
    save_dir : str
        Directory where models will be saved
    """
    import pathlib
    
    # Get the repo root (SatCHM directory)
    repo_root = pathlib.Path(__file__).parent.parent.resolve()
    
    # Check if we're in the standard project structure
    # Standard: /project/es14-stor/canopy/SatCHM/
    parent_dir = repo_root.parent
    
    # Create models directory outside repo
    models_dir = parent_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Site-specific model directory outside repo
    site_model_dir = models_dir / f"{site}_model"
    
    # Expected location inside repo
    repo_lightning_logs = repo_root / "ms_net" / "lightning_logs"
    repo_site_model = repo_lightning_logs / f"{site}_model"
    
    # If symlink doesn't exist, create it
    if not repo_site_model.exists():
        # Create the symlink
        repo_site_model.symlink_to(site_model_dir, target_is_directory=True)
        print(f"✓ Created symlink: {repo_site_model} -> {site_model_dir}")
    elif not repo_site_model.is_symlink():
        # If it's a real directory (not symlink), warn user
        print(f"⚠ Warning: {repo_site_model} exists but is not a symlink.")
        print(f"  Models will be saved inside repo. Consider moving to {site_model_dir}")
        return "lightning_logs"  # Use default location
    
    print(f"✓ Models will be saved to: {site_model_dir}")
    print(f"✓ Accessible via symlink: {repo_site_model}")
    
    # Return the parent directory for TensorBoardLogger
    # Logger will create {save_dir}/{name}/version_X/
    return str(models_dir)

def setup_params(directory, site):
    """
    Description
    ___________
    This sets up the parameters for the argument parser and documents the training and validation lists.

    Parameters
    _______
    directory : str
        path to project directory
    site : str
        site being processed

    Returns
    ______
    params : namespace
        this contains specific arguments for the neural network
    """
    params = parse_args()
    params.train_list = os.path.join(directory, f'{site}_trainlist.txt')
    params.val_list = os.path.join(directory, f'{site}_vallist.txt')
    params.x_array = ['wvimg', 'solar', 'sensor', 'dem']
    params.y_array = ['chm']
    params.x_xform = [None, None, None, None]
    params.y_xform = [None]
    params.c_xform = [None]
    params.model_loc = 'chks'
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

def load_or_create_model(params, net_dict):
    """
    Description
    ___________
    This loads and creates the model

    Parameters
    __________
    params : namespace
        this contains specific arguments for the neural network
    net_dict : dict
        a dictionary of hyperparameters for the neural network

    Returns
    _______
    Returns the created model
    """
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    # directory = os.path.join('..', '..', os.getcwd())
    directory = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    output_directory = os.path.join(directory, "outputs")
    os.makedirs(output_directory, exist_ok=True)

    try:
        model_dir = os.path.join('lightning_logs',f'version_{params.net_name}')
        model_dir = os.path.normpath(os.path.join(model_dir, '..', '..'))
        model_loc = gb(f'{model_dir}/checkpoints/*.ckpt')[params.num_model]
        print(f'Loading {model_loc}')
        yaml_loc = gb(f'{model_dir}/*.yaml')[0]
        yaml_dict = load_hparams(yaml_loc)
        model = MS_Net().load_from_checkpoint(
            model_loc,
            net_name=yaml_dict['net_name'],
            num_scales=yaml_dict['num_scales'],
            num_features=len(params.x_array) + 1,
            num_filters=yaml_dict['num_filters'],
            f_mult=yaml_dict['f_mult'])
    except IndexError:
        print('Instantiating a new MS-NET()')
        model = MS_Net(
            net_name=params.net_name,
            num_scales=params.num_scales,
            num_features=len(params.x_array),
            num_filters=params.num_filters,
            f_mult=params.f_mult,
            lr=params.LR,
            hparams=net_dict,
            steps=params.steps,
        )

    return model

def setup_trainer(params, site):
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

        EarlyStopping(
            monitor="val_loss",
            check_finite=False,
            patience=9999
        )
    ] 
    
    # Set up model storage outside repo with symlink
    model_save_dir = setup_model_storage(site)
    
    # Create site-specific logger
    # Models will be saved to: /project/es14-stor/canopy/models/{site}_model/version_X/
    logger = TensorBoardLogger(
        save_dir=model_save_dir,
        name=f"{site}_model",
        version=None  # auto-increment version within this site's directory
    )

    return Trainer(
        max_epochs=params.max_epochs,
        callbacks=cbs,
        logger=logger,  # Add the site-specific logger
        plugins=None,
        precision="16-mixed",
        devices=1,
        accelerator="gpu",
        log_every_n_steps=10,
    )

def train_main(data_path, NORM_CONST, site):
    """
    Main training function.
    
    Parameters
    __________
    data_path : str
        path to training data
    NORM_CONST : float
        normalization constant for CHM
    site : str
        site code for site-specific model saving
    """
    directory, env_site = setup_environment()
    params = setup_params(data_path, site)
    net_dict = setup_net_dict(params)
    model = load_or_create_model(params, net_dict)
    
    print("\nLoading training and validation samples...\n")
    print(f'net_dict: {net_dict}')
    train_dataloader = get_dataloader(net_dict, ['train'], data_path=data_path, NORM_CONST=NORM_CONST)
    val_dataloader = get_dataloader(net_dict, ['val'], data_path=data_path, NORM_CONST=NORM_CONST)

    trainer = setup_trainer(params, site)  # Pass site to setup_trainer
    #trainer.fit(model, train_dataloader, val_dataloader['val'])
    
    try:
            trainer.fit(model, train_dataloader, val_dataloader['val'])
    except KeyboardInterrupt:
            print("\nTraining stopped by user (Ctrl+C).")
    finally:
         pass
    
def main():
    # freeze_support()
    # Parse arguments using ms_parser (includes --site flag)
    params = parse_args()
    
    # Get site from command line or fall back to .env
    if params.site:
        site = params.site
        print(f"✓ Using site from command line: {site}")
    else:
        site = os.getenv('site')
        if not site:
            raise ValueError("Site must be specified via --site flag or in .env file")
        print(f"✓ Using site from .env file: {site}")
    
    # Validate that site data exists
    data_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', f'{site}_data'))
    if not os.path.exists(data_path):
        raise ValueError(f"Data directory not found: {data_path}\n"
                        f"Run data preparation for site '{site}' first.")
    
    print(f"✓ Data directory found: {data_path}")
    
    # Use norm_const from params (with underscore, not hyphen)
    NORM_CONST = getattr(params, 'norm_const', 46)
    train_main(data_path=data_path, NORM_CONST=NORM_CONST, site=site)
    
if __name__ == '__main__':
    main()
