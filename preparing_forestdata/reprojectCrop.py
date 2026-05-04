"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


"""
This script reprojects and crops lidar data and digital elevation models (DEM) into correct projected coordinate system and polygon bounds, respectively. Written by Mia Mitchell. Los Alamos, NM. Winter 2025.
"""

import os
import subprocess
import argparse
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

def reprojectTif(input_file, output_dir, site_shapefile_path, resolution):
    """
    Description
    ___________
    Main function for reprojecting and cropping GeoTIFF data. This function uses gdalwarp from GDAL to reproject and crop. Files will be overwritten if it already exists.

    Parameters
    __________
        input_file : str
            The path to GeoTIFF (Digital Elevation Model or Lidar CHM)
        output_dir : str
            The path to output directory for the projected and cropped GeoTIFF
        site_shapefile_path : str
            The path to the site shapefile provided in the .env file
        resolution : int
            the resolution of the GeoTIFF wanted
    Returns
    _______
        output_file_cropped : str
            The path to the reprojected, cropped shapefile
    """
    # Pulled from the .env 
    env_path = find_dotenv()
    load_dotenv(env_path)
    utm = os.getenv("utm")
    site = os.getenv("site")

    # Reprojecting 
    output_file_reprojected = os.path.join(output_dir, f"{site}_reprojected.tif")

    reproj_cmd = [
    "gdalwarp",
    "-overwrite",
    "-t_srs", f"{utm}",
    "-tr", f"{resolution}", f"{resolution}",
    "-r", "near",
    "-of", "GTiff",
    f"{input_file}",
    f"{output_file_reprojected}"]  

    print(f"\nReprojecting {os.path.split(input_file)[-1]}...\n")
    reproj_process = subprocess.Popen(reproj_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
   
    # Progress Bar
    with tqdm(unit="lines", desc="Processing", bar_format="{l_bar}{bar}| {n_fmt} lines") as pbar:
        for line in reproj_process.stdout:
            pbar.update(1)
            pbar.set_postfix_str(line.strip()[:40]) 
        
    return output_file_reprojected
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reprojects and crops a GeoTIFF file based on UTM listed in .env to the projected shapefile provided')

    parser.add_argument('--site_shapefile_path', type=str, help='Path to the projected shapefile')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory for the reprojected, cropped GeoTIFF')
    parser.add_argument('--input_file', type=str, help='Path to the input GeoTIFF')
    
    args = parser.parse_args()
    reprojectTif(args.site_shapefile_path, args.output_dir, args.input_file)




