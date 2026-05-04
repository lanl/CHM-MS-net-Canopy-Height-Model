"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

"""
The main trainer script for MS-net.
"""
from glob import glob as gb
import os
from dotenv import load_dotenv, find_dotenv
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from multiprocessing import freeze_support

from .network_2D_lightning import MS_Net
from .ms_parser import parse_args
from .pore_utils_2D import get_dataloader, load_hparams

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
    params.train_list = os.path.join(directory, "inputs", f'{site}_train.txt')
    params.val_list = os.path.join(directory, "inputs", f'{site}_val.txt')
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
    directory = os.getenv('project_path')
    output_directory = os.path.join(directory, "outputs")
    os.makedirs(output_directory, exist_ok=True)

    try:
        model_dir = f'lightning_logs/version_{params.net_name}'
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

def setup_trainer(params):
    cbs = [
        # saves best model based on val_loss
        ModelCheckpoint(
            monitor="val_loss",
            filename="best-val-{epoch:02d}-{step:07d}",
            save_top_k=1,
            mode="min",
            save_on_train_epoch_end=False  
        ),

        # saves every 1000 steps 
        ModelCheckpoint(
            filename="step-{step:07d}",
            every_n_train_steps=1000,
            save_top_k=-1, 
        ),

        EarlyStopping(
            monitor="val_loss",
            check_finite=False,
            patience=99999
        )
    ] 

    return Trainer(
        max_epochs=params.max_epochs,
        callbacks=cbs,
        plugins=None,
        precision="16-mixed",
        devices='auto',
        accelerator=params.accelerator,
        log_every_n_steps=10,
    )

def train_main():
    directory, site = setup_environment()
    params = setup_params(directory, site)
    net_dict = setup_net_dict(params)
    model = load_or_create_model(params, net_dict)
    
    print("\nLoading training and validation samples...\n")
    train_dataloader = get_dataloader(net_dict, ['train'])
    val_dataloader = get_dataloader(net_dict, ['val'])

    trainer = setup_trainer(params)
    #trainer.fit(model, train_dataloader, val_dataloader['val'])
    
    try:
            trainer.fit(model, train_dataloader, val_dataloader['val'])
    except KeyboardInterrupt:
            print("\nTraining stopped by user (Ctrl+C).")
    finally:
         pass
def main():
    freeze_support()
    train_main()
    

if __name__ == '__main__':
    main()