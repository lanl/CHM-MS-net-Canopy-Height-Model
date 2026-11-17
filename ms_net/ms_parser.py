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
from pathlib import Path

# Optional: load .env if present (harmless if none exists)
load_dotenv(find_dotenv())

code_version = 0.00123
project_directory = os.getenv("project_directory")

# FIXME: idk if this should be duplicated here, imported from sites_config.py, or just make one centralized parsing file
def load_config():
    import json
    # Find config.json in project directory
    script_dir = Path(__file__).resolve()
    project_dir = script_dir.parent.parent
    config_json_path = project_dir / 'config.json'

    if not config_json_path.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_json_path}\n"
            "Please create config.json in the project root directory."
        )

    # Load config.json
    with open(config_json_path, 'r') as f:
        config = json.load(f)

    config['project_dir'] = str(project_dir)

    return config


def parse_args(argv=None):
    """
    Description
    ___________
    Function to parse boolean arguments.

    Accepts an optional `argv` list so callers can override defaults
    programmatically, e.g. parse_args(['--data_loc', '/my/data']).
    """

    config = load_config()
    channels_opt = config['channels']

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
    parser.add_argument("--gpu-device", default=0, type=int)

    # NETWORK PROPERTIES
    parser.add_argument("--net-name", default="CHMer", type=str)
    parser.add_argument("--num-scales", default=3, type=int)
    parser.add_argument("--num-filters", default=8, type=int)
    parser.add_argument("--f-mult", default=4, type=int)

    # TRAINING PARAMETERS
    parser.add_argument("--train", default=True, type=str2bool)
    parser.add_argument("--LR", default=1e-4, type=float)
    parser.add_argument("--max-epochs", default=1000, type=int)
    parser.add_argument("--min-epochs", default=10, type=int)
    parser.add_argument("--steps", default=4, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--accumulate-grad-batches", default=1, type=int)
    parser.add_argument("--check-val-every-n-epoch", default=2, type=int)
    parser.add_argument("--gradient-clip-val", default=1.0, type=float)

    # MODEL CHANNELS
    parser.add_argument("--channel", type=str, choices=channels_opt, default=None)

    # EVALUATION PARAMETERS
    parser.add_argument("--eval-only", action="store_true")         # defaults to False. True when passed       # FIXME: to be removed
    parser.add_argument("--model-loc", default=None, type=str)       # used alongside --eval-only to specify your model_loc
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument('--outlier-filter', type=int, default=None, choices=[95, 99])

    # TESTING/RESTART PARAMETERS
    parser.add_argument("--num-model", default=0, type=int)

    # DATA PARAMETERS
    parser.add_argument("--rnd-data", default=False, type=str2bool)
    parser.add_argument("--training-size", default=657, type=int)
    parser.add_argument("--validation-size", default=140, type=int)
    parser.add_argument("--data-aug", default=False, type=str2bool)
    parser.add_argument("--data-loc", default='../../ms-data/', type=str)

    # SYSTEM CONFIGS
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--num-nodes", default=1, type=int)
    
    # SITE MANAGEMENT
    parser.add_argument("--sites", type=str, nargs='+',
                       help='Site code(s) (e.g., fs_train fw_large qm)')
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
