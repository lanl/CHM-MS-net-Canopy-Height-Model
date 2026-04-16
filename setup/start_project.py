"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


"""
This script sets up starting directories and the satellite angle metadata for pre-processing. Written by Mia Mitchell in Santa Fe, New Mexico. Spring 2025.
"""

import os
from dotenv import load_dotenv, find_dotenv
import shutil
from tools.styles import style

def start():
    """
    Description
    ___________

    This is the main function that creates starting directories and .csv files.

    Returns
    _______

    paths : dict
        a dictionary to hold directories for the project 
    """
    paths = {}

    # Locates .env file that contains project_path and utm code
    # Uses ls -a to see .env in commandline
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    directory=os.getenv('project_dir')

    w, h = shutil.get_terminal_size()
    print("-" * w)
    print("\n" + style.BOLD + "Project Directory:" + style.RESET, directory + "\n")
   
    os.makedirs(directory, exist_ok=True)
    site=os.getenv('site')

    # Making directories: 'ms-data,' 'metadata,', 'site-polygons' and 'downloads'
    paths["new_project"] = os.path.join(directory, 'ms-data')
    os.makedirs(paths["new_project"], exist_ok=True)
    
    paths["metadata"] = os.path.join(directory, 'metadata')
    os.makedirs(paths["metadata"], exist_ok=True)
    
    paths["site_polygons"] = os.path.join(directory, 'site-polygons')
    os.makedirs(paths["site_polygons"], exist_ok=True)
    
    paths["with_crs"] = os.path.join(paths["site_polygons"], 'with_crs', f'{site}') 
    os.makedirs(paths["with_crs"], exist_ok=True)

    paths["inputs"] = os.path.join(directory, 'inputs')
    os.makedirs(paths["inputs"], exist_ok=True)
    
    return paths


if __name__ == "__main__":
    start()
