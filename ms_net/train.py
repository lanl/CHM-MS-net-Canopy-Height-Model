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
from plotting_utils import evaluate_validation_model
import re


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

    model_dir = os.path.join(
        'lightning_logs',
        f'{params.site}_model'
    )
    print(f'model_dir: {model_dir}')
    # Check whether a model directory exists
    if not os.path.isdir(model_dir):
        if params.eval_only:
            raise FileNotFoundError(
                f"--eval-only was requested, but no model directory "
                f"was found: {model_dir}"
            )
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
        return model, True

    versions = [
        name for name in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, name))
        and re.fullmatch(r'version_\d+', name)
    ]
    if not versions:
        if params.eval_only:
            raise FileNotFoundError(
                f"--eval-only was requested, but no model versions "
                f"were found in: {model_dir}"
            )
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
        return model, True

    latest_version = max(
        versions,
        key=lambda x: int(x.split('_')[-1])
    )
    latest_version_dir = os.path.join(
        model_dir,
        latest_version
    )
    print(f'Latest model version: {latest_version_dir}')

    latest_model_dir = os.path.join(
        latest_version_dir,
        'checkpoints'
    )

    if not os.path.isdir(latest_model_dir):
        if params.eval_only:
            raise FileNotFoundError(
                f"--eval-only was requested, but no checkpoint "
                f"directory was found: {latest_model_dir}"
            )

        print('No checkpoint found. Instantiating a new MS-NET()')
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

        return model, True

    ckpt_files = [
        name for name in os.listdir(latest_model_dir)
        if os.path.isfile(os.path.join(latest_model_dir, name))
        and name.startswith('best-val-epoch=')
    ]

    if len(ckpt_files) == 0:
        if params.eval_only:
            raise FileNotFoundError(
                f"--eval-only was requested, but no best-val "
                f"checkpoint was found in: {latest_model_dir}"
            )

        print('No best-val checkpoint found. Instantiating a new MS-NET()')
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

        return model, True

    if len(ckpt_files) > 1:
        raise RuntimeError(
            f"Expected exactly one best-val checkpoint, "
            f"but found {len(ckpt_files)}: {ckpt_files}"
        )

    model_loc = os.path.join(
        latest_model_dir,
        ckpt_files[0]
    )

    print(f'model_loc: {model_loc}')

    # ---------------------------------------------------------
    # Load hyperparameters
    # ---------------------------------------------------------

    yaml_loc = os.path.join(
        latest_version_dir,
        'hparams.yaml'
    )

    print(f'yaml_loc: {yaml_loc}')

    yaml_dict = load_hparams(yaml_loc)

    if yaml_dict:
        print("Loading architecture parameters from hparams.yaml")

        net_name = yaml_dict['net_name']
        num_scales = yaml_dict['num_scales']
        num_filters = yaml_dict['num_filters']
        f_mult = yaml_dict['f_mult']

    else:
        print("hparams.yaml is empty.")
        print("Using current params to reconstruct legacy checkpoint.")

        net_name = params.net_name
        num_scales = params.num_scales
        num_filters = params.num_filters
        f_mult = params.f_mult

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    model_loc = "/project/wildfirehydro/ltiede/CHM_2/CHM-MS-net-Canopy-Height-Model/ms_net/lightning_logs/fs_nov_9_model/version_0/checkpoints/epoch-epoch=969.ckpt"

    model = MS_Net.load_from_checkpoint(
        model_loc,
        net_name=net_name,
        num_scales=num_scales,
        num_features=len(params.x_array),
        num_filters=num_filters,
        f_mult=f_mult
    )

    new_model = False

    return model, new_model


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
    
    if new_model and net_dict is not None :
        logger.log_hyperparams(net_dict)

    print(f"✓ Models will be saved to: lightning_logs/{site}_model/")

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
    model, new_model = load_or_create_model(params, net_dict)
    print(f'net_dict: {net_dict}')
    
    # only create trainer when we are training a model
    if not params.eval_only :
        print("\nLoading training and validation samples...\n")
        train_dataloader = get_dataloader(net_dict, ['train'], data_path=data_path, NORM_CONST=NORM_CONST)
        val_dataloader = get_dataloader(net_dict, ['val'], data_path=data_path, NORM_CONST=NORM_CONST)
        trainer = setup_trainer(
            params,
            site,                   # Pass site to setup_trainer
            net_dict=net_dict,      # pass dictrionary for writing hparams.yaml
            new_model=new_model     # pass whether this is a new model or not
        )
        try:
                trainer.fit(model, train_dataloader, val_dataloader['val'])
        except KeyboardInterrupt:
                print("\nTraining stopped by user (Ctrl+C).")
    else :
        print("\nLoading only validation samples...\n")
        val_dataloader = get_dataloader(net_dict, ['val'], data_path=data_path, NORM_CONST=NORM_CONST)
    
    evaluate_validation_model(
        model=model,
        val_dataloader=val_dataloader['val'],
        norm_const=NORM_CONST,
        plot_sample=True,
        data_point_index=10,                    # change to view a different data point
        save_plot="validation_comparison.png"
    )
    
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
