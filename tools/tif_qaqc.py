"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


""" Some QAQC and processing tools for GeoTIFFS"""

import os
import rasterio
import numpy as np
import zipfile
from tqdm import tqdm

def delete_non_tif(folder):
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.tif', '.tiff')):
            file_path = os.path.join(folder, filename)
            os.remove(file_path)

def has_tif_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.tif', '.tiff', '.TIF', '.TIFF')):
            return True
    return False

def tif_qaqc(folder_path):
    l=0
    k=0
    tif_files = [
    filename for filename in os.listdir(folder_path)
    if filename.lower().endswith(('.tif', '.tiff')) and not filename.startswith("._")]
    filename_pbar = tqdm(tif_files, desc="Processing files", unit="files", leave=True)
    for filename in tif_files:
        filename_pbar.update(1)
        with rasterio.open(os.path.join(folder_path, filename)) as src:
            data = src.read(1)
            invalid_data = (data == -9999) | (data == 0)
            invalid_ratio = np.count_nonzero(invalid_data) / data.size
        if invalid_ratio > 0.95:
            os.remove(os.path.join(folder_path, filename))
            k+=1
        else: 
            #print(filename)
            l+=1
            pass
    else:
        pass
    #filename_pbar.write(f"Greater than 95%: {l}")
    filename_pbar.write(f"\nFiles removed with significant bad data: {k} files\n")

def unzip_and_remove(zip_filename):
    """
    Description
    ___________
    This function extracts the contents of a zip file and then removes the original zip file.

    Parameters
    __________
    zip_filename : str
        The name of the zip file to be processed.

    """
    try:
        # Get the directory of the zip file
        zip_dir = os.path.dirname(zip_filename)
        
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Extract to the same directory as the zip file
            zip_ref.extractall(path=zip_dir)
        print(f"Successfully unzipped: {zip_filename}")
        os.remove(zip_filename)
        print(f"Removed zip file: {zip_filename}")
    except Exception as e:
        print(f"Error processing {zip_filename}: {str(e)}")

