"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


'''
Argument parser for MS-NET. 
'''

import os
import argparse
import torch
from dotenv import load_dotenv, find_dotenv

code_version = 0.00123
project_directory=os.getenv('project_directory')

def parse_args():
    
    '''
    Description
    ___________
    Function to parse boolean arguments. 
    
    '''
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # Command-line argument parser
    parser = argparse.ArgumentParser(description="MS-Net properties.")

    # GPU PROPERTIES
    parser.add_argument("--gpu_device", default=0, type=int)

    # NETWORK PROPERTIES
    parser.add_argument("--net_name",    default="CHMer", type=str)  
    # Name of the neural network
    parser.add_argument("--num_scales",  default=3, type=int)            
    # Number of scales (multi-scale architecture)
    parser.add_argument("--num_filters", default=2, type=int)             
    # Number of filters in each convolutional layer
    parser.add_argument("--f_mult",      default=4, type=int)             
    # Filter multiplier to control network complexity

    # TRAINING PARAMETERS
    parser.add_argument("--train", default=True, type=str2bool)   
    # Whether to train the network
    parser.add_argument("--LR", default=1e-4, type=float)         
    # Learning rate for gradient descent
    parser.add_argument("--max_epochs", default=10000, type=int)  
    # Maximum number of epochs
    parser.add_argument("--min_epochs", default=10, type=int)     
    # Minimum number of epochs
    parser.add_argument("--steps", default=4, type=int)         
    # Number of steps 
    parser.add_argument("--batch_size", default=32, type=int)     
    # Number of samples per batch
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)  
    # For gradient accumulation
    parser.add_argument("--check_val_every_n_epoch", default=2, type=int)  
    # How often to run validation
    parser.add_argument("--gradient_clip_val", default=0.00, type=float)  
    # Clipping gradients to avoid large updates

    # TESTING/RESTART PARAMETERS
    parser.add_argument("--num_model", default=0, type=int) 
    # Model number for testing or restarting

    # DATA PARAMETERS
    parser.add_argument("--rnd_data", default=False, type=str2bool)    
    # Use random sampling
    parser.add_argument("--training_size", default=1, type=int)        
    # Proportion of data to use for training
    parser.add_argument("--validation_size", default=1, type=int)       
    # Proportion of data to use for validation
    parser.add_argument("--data_aug", default=False, type=str2bool)    
    # Apply data augmentation
    parser.add_argument("--data_loc", default='new_dataset', type=str)  
    # Path to dataset location

    # SYSTEM CONFIGS
    # Sets the random seed for reproducibility. A consistent seed ensures that the random operations will produce the same results every time the code is run.
    parser.add_argument("--seed", type=int)
    #Specifies the number of GPUs to use for training. Default is set to 2, meaning the script is configured to utilize 2 GPUs for training. If the machine doesn't have GPUs (checked later in the script), this will be set to None.
    parser.add_argument("--gpus", default=2, type=int)
    # Defines the number of worker threads for data loading. More workers mean faster data loading because multiple threads are used to read the data in parallel. Default is 4.
    parser.add_argument("--num_workers", default=4, type=int)
    #  Used for multi-node training. By default, only one node is used, but this could be increased for distributed setups where multiple machines are working together
    parser.add_argument("--num_nodes", default=1, type=int)

    args = parser.parse_args()

    # Code checks if GPUs are available on the machine's Pytorch's torch.cuda.is.available(), if not --gpus is set to None
    print("Searching for a GPU...")
    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        args.gpus = None
    if torch.cuda.is_available():
        args.accelerator = "gpu"
        gpus = [args.gpu_device]
    else:
        args.accelerator = "cpu"
        gpus = None
    
    # Sets torch version and code version
    args.torch_version = torch.__version__
    args.code_version  = code_version

    # Setting the computer name variable, if there is none set in the os environment, then it is default set to my computer

    try: 
        args.pc = os.environ._data['COMPUTERNAME']
    except KeyError:
        args.pc = 'PC'

    print("Model Arguments:",args)
    return args 