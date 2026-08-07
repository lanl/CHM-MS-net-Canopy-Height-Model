"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

"""
Argument parser for MS-NET.
"""

import os
import argparse
import torch
from dotenv import load_dotenv, find_dotenv

# Optional: load .env if present (harmless if none exists)
load_dotenv(find_dotenv())

code_version = 0.00123
project_directory = os.getenv("project_directory")


def parse_args(argv=None):
    """
    Description
    ___________
    Function to parse boolean arguments.

    Accepts an optional `argv` list so callers can override defaults
    programmatically, e.g. parse_args(['--data_loc', '/my/data']).
    """

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
    parser.add_argument("--net_name", default="CHMer", type=str)
    parser.add_argument("--num_scales", default=3, type=int)
    parser.add_argument("--num_filters", default=8, type=int)
    parser.add_argument("--f_mult", default=4, type=int)

    # TRAINING PARAMETERS
    parser.add_argument("--train", default=True, type=str2bool)
    parser.add_argument("--LR", default=1e-4, type=float)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--min_epochs", default=10, type=int)
    parser.add_argument("--steps", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=2, type=int)
    parser.add_argument("--gradient_clip_val", default=1.0, type=float)

    # TESTING/RESTART PARAMETERS
    parser.add_argument("--num_model", default=0, type=int)

    # DATA PARAMETERS
    parser.add_argument("--rnd_data", default=False, type=str2bool)
    parser.add_argument("--training_size", default=657, type=int)
    parser.add_argument("--validation_size", default=140, type=int)
    parser.add_argument("--data_aug", default=False, type=str2bool)
    parser.add_argument("--data_loc", default='../../ms-data/', type=str)

    # SYSTEM CONFIGS
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    
    # SITE MANAGEMENT
    parser.add_argument("--site", type=str, required=False,
                       help='Site code (e.g., ws, lm, qm). Overrides .env file if provided.')
    parser.add_argument("--norm-const", type=float, default=46,
                       help='Normalization constant for CHM data (default: 46)')

    # NOTE: parse provided argv if given; else parse from sys.argv
    args = parser.parse_args(argv)

    # GPU detection & accelerator setup
    print("Searching for a GPU...")
    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        args.gpus = None
        args.accelerator = "cpu"
    else:
        args.accelerator = "gpu"
        # (Optional) if your trainer expects devices, you could add:
        # args.devices = [args.gpu_device]
    print(f'args.accelerator: {args.accelerator}')

    # Meta
    args.torch_version = torch.__version__
    args.code_version = code_version

    # Computer name (robust)
    args.pc = os.environ.get("COMPUTERNAME", "PC")

    print("Model Arguments:", args)
    return args
