"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


'''
 This script is for processing the satellite imagery into 2040 x 2040 pixels 2D arrays. Written originally by Chuck Abolt in MATLAB, but written and edited by Mia Mitchell in Santa Fe, New Mexico. Winter 2025.

'''
import os
import rasterio
import pandas as pd
import numpy as np
import shutil
import argparse
from process_imagery.manageDirectories import Manager
from tqdm import tqdm
import subprocess
from dotenv import load_dotenv, find_dotenv


def projectSatelliteImagery(raw_directory, angle_metadata, site_metadata):
    """
    Description
    ___________
    This is the main function for processing the satellite imagery

    Parameters
    __________
    raw_directory : str
        The directory for where GeoTIFFs are located for subtiling
        
    angle_metadata : str
        The CSV that has information about site and angle

    site_metadata : str
        The CSV that has information about the bounds of tge site
    Returns
    ______

    projected_data_directory : str
        The path to the projected data directory that contains the first subtiling of the GeoTIFFs

    """
    # Pulled from the .env 
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    project_directory=os.getenv('project_path')
    utm = os.getenv("utm")
    site = os.getenv("site")

    # Functions 
    def find_folder(start_path): 
        """
        Description
        ___________
        This function searches for a folder containing the GeoTIFF files

        Parameters
        __________
        start_path : str
            The starting path of the directory tree to search.
        
        Returns
        _______
        list
             GeoTIFF files is found
        
        """
        wvfiles = []
        for root, dirs, files in os.walk(start_path):
            for file in files:
                if not file.startswith('.') and file.lower().endswith(('.tif', '.tiff', '.TIF', '.TIFF')):
                    wvfiles.append(os.path.join(root, file))
        return wvfiles

    def getting_site_information(input_path):
        """
        Description
        ___________
        This function reads a CSV and processes the data to extract site-specific information
        
        Parameters
        __________
        input_path : str
            The file path of the input CSV file.

        Returns
        _______
        
        dict
            The `site_data` dictionary, where the keys are the site names, and the values are dictionaries containing the spatial and coordinate information for each site.
            
        """
        metadata = pd.read_csv(input_path)
        if "chunk bounds" in metadata["chunk_or_whole"].values:
            metadata = metadata[metadata["chunk_or_whole"] != "whole bounds"]
        else:
            print("chunk == whole_bounds\n")
        metadata = metadata.dropna(subset=["name", "e0", "e1", "n0", "n1", "utm_code"])
        site_data = {}
        for _, row in metadata.iterrows():
            name = row["name"]
            e0, e1, n0, n1 = row["e0"], row["e1"], row["n0"], row["n1"]
            utmcode = row["utm_code"]
            site_data[name] = {"e0": e0, "e1": e1, "n0": n0, "n1": n1, "utm_code": utmcode}
        return site_data

    chunk_folders = [f for f in os.listdir(raw_directory) if os.path.isdir(os.path.join(raw_directory, f)) and f not in [".DS_Store", "projected-data"]]
    print("\nTotal chunks:", chunk_folders)
    
    # Creating projected-data folder
    projected_data_directory = os.path.join(raw_directory, 'projected-data')
    os.makedirs(projected_data_directory, exist_ok=True)

    # Creating projected-data directories
    manager = Manager(projected_data_directory)
    main_folder_paths = manager.create_mainfolders(angle_metadata, site)

    # Progress Bar
    chunk_pbar = tqdm(chunk_folders, desc="Processing chunks", unit="chunk", leave=False)
    
    for chunk in chunk_folders:
        chunk_path = os.path.join(raw_directory, chunk)

        chunk_pbar.write(f"\n\nWorking on chunk: {chunk}\n")
        
        subsub_folders = [f for f in os.listdir(chunk_path) if os.path.isdir(os.path.join(chunk_path, f)) and f != ".DS_Store"]
        chunk_pbar.write(f"Chunk subfolders: {subsub_folders}\n")
        
        tif_files = []
        for folder in subsub_folders:
            folder_path = os.path.join(chunk_path, folder)
            tif_files = find_folder(folder_path) 
            outdir = os.path.join(projected_data_directory, chunk)
            merged_tif_path = os.path.join(projected_data_directory, chunk, "merged_temp.tif")
            if os.path.exists(merged_tif_path):
                os.remove(merged_tif_path)
            chunk_pbar.write(f"Merging {chunk}.")
            mrg_cmd = f"rio merge {' '.join(tif_files)} {merged_tif_path}"

            result = subprocess.run(
                    mrg_cmd.split(" "),
                    capture_output = True,
                    text = True 
                )
            if result.returncode != 0:
                    raise RuntimeError(f"\n\ncommand:\n\n\t>{mrg_cmd}<\n\nFAILED with stdout:\n\n"
                                        f"{result.stdout}"
                                        "\n\nSTDERR:\n\n"
                                        f"{result.stderr}\n\n")
            # Getting site data
            site_data = getting_site_information(site_metadata)
            dataframe = site_data.get(chunk)
            e0, e1, n0, n1 = int(dataframe["e0"]), int(dataframe["e1"]), int(dataframe["n0"]), int(dataframe["n1"])
            utmcode = dataframe["utm_code"]
    
            eLL = range(int(e0), int(e1), 1000)
            nLL = range(int(n0), int(n1), 1000)
            total_iterations = len(eLL) * len(nLL)
            
            tile_pbar = tqdm(
                total=total_iterations,
                desc="Processing tiles",
                unit="tile")
            
            tile_pbar.write(f"\nWorking on images from {folder}...\n")
        
            for e in eLL:
                for n in nLL:
                    sqkm    = str(e)[:3] + '_' + str(n)[:4]
                    bbox = " ".join(
                    [str(e - 10), str(n - 10), str(e + 1010), str(n + 1010)])
                
                    outfile = os.path.join(outdir, f"{folder}_{sqkm}.tif")

                    tile_pbar.update(1)
                    tile_pbar.set_description(
                        f"Processing easting={e}, northing={n}")
                    
                    projcmd = f'rio warp {merged_tif_path} {outfile} --dst-crs {utmcode} --bounds {bbox} --resampling cubic --res 0.5 --overwrite'
                    
                    result = subprocess.run(
                    projcmd.split(" "),
                    capture_output = True,
                    text = True)

                    if result.returncode != 0:
                        raise RuntimeError(f"\n\ncommand:\n\n\t>{projcmd}<\n\nFAILED with stdout:\n\n"
                                        f"{result.stdout}"
                                        "\n\nSTDERR:\n\n"
                                        f"{result.stderr}\n\n")
                    if os.path.exists(outfile):
                        with rasterio.open(outfile) as src: 
                            data = src.read(1)
                            nodata_value = src.nodata
                            if nodata_value is not None and np.all(
                                data == nodata_value 
                            ):
                                os.remove(outfile)
                            else:
                                continue
                            if np.all(data == 0):
                                os.remove(outfile)
                            else:
                                continue
                    else:
                        tile_pbar.write("Continuing...")
        chunk_pbar.update(1) 
    
    chunk_pbar.write(
                f"Finished creating  2020 x 2020 tiles for the satellite imagery..."
            )
    
    return projected_data_directory


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Projecting Satellite Imagery into 1010 x 1010 tiles')
    parser.add_argument('--raw_directory', type=str, required=True,
                        help='Path to the directory containing the raw satellite imagery')
    parser.add_argument('--angle_metadata', type=str, required=True,
                        help='Path to the file containing the angle metadata')
    parser.add_argument('--site_metadata', type=str, required=True,
                        help='Path to the file containing the site metadata')
    args = parser.parse_args()
    projectSatelliteImagery(args.raw_directory, args.angle_metadata, args.site_metadata)


