"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


"""

This script is for processing the lidar data and digital elevation model (DEM) into 1010 x 1010 m tiles. Originally written by Chuck Abolt in Matlab, but edited and written by Mia Mitchell in Santa Fe, New Mexico. Fall 2024.

"""

import os
import subprocess
from shapely.geometry import box
import pandas as pd
import rasterio
import argparse
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

def processAuxTifs(raw_directory, metadata_path, input_tif, output_directory):
    """
    Description
    ___________
    This is the main function for processing the lidar and DEM input files.

    Parameters
    __________

    metadata_path : str
        The path to the site_metadata.csv
    input_tif : str
        The input .tif file being processed
    output_directory : str
        The output directory for the file being processed
    """

    def getting_site_information(metadata_path):
        """
        Description
        ___________
        This function reads a CSV and processes the data to extract site-specific information
        
        Parameters
        __________
        metadata_path : str
            The path to the site_metadata.csv

        Returns
        _______
        
        dict
            The `site_data` dictionary, where the keys are the site names, and the values are dictionaries containing the spatial and coordinate information for each site.
            
        """
        metadata = pd.read_csv(metadata_path)
        metadata = metadata.dropna(subset=["name", "e0", "e1", "n0", "n1", "utm_code"])
        site_data = {}
        for _, row in metadata.iterrows():
            name = row["name"]
            e0, e1, n0, n1 = row["e0"], row["e1"], row["n0"], row["n1"]
            utmcode = row["utm_code"]
            site_data[name] = {"e0": e0, "e1": e1, "n0": n0, "n1": n1, "utm_code": utmcode}
        return site_data

    
    # get raster bbox
    def get_file_bbox(file_path):
        """
        Description
        ___________
        This function gets the bounding box of the input raster.

        Parameters
        __________
        file_path : str
            The path to the input raster (ends in .tiff or .tif)

        Returns
        _______
        
        bounds
            xmin, ymin, xmax, ymax
            
        """  
        with rasterio.open(file_path) as src:
            bounds = src.bounds
        return bounds

    raster_bounds = get_file_bbox(input_tif)
    raster_geom = box(*raster_bounds)

    chunks = [f for f in os.listdir(raw_directory) if os.path.isdir(os.path.join(raw_directory, f)) and f not in [".DS_Store", "projected-data"]]
    chunk_pbar = tqdm(chunks, desc="Processing chunks", unit="chunk")  

    for chunk in chunks:
        chunk_pbar.set_description(f"Working on site: {chunks}")
        site_data = getting_site_information(metadata_path)
        dataframe = site_data.get(chunk)
        e0, e1, n0, n1 = int(dataframe["e0"]), int(dataframe["e1"]), int(dataframe["n0"]), int(dataframe["n1"])
        utmcode = dataframe["utm_code"]
        eLL = range(int(e0), int(e1), 1000)
        nLL = range(int(n0), int(n1), 1000)
        total_iterations = len(eLL) * len(nLL)

        tile_pbar = tqdm(
            total=total_iterations,
            desc="Processing tiles",
            unit="tile",
            leave=False,
        )

        for e in eLL:
            for n in nLL:
                sqkm = str(e)[:3] + "_" + str(n)[:4]
                bbox_coords = [e - 10, n - 10, e + 1010, n + 1010]
                bbox_geom = box(*bbox_coords)
                bbox = " ".join(
                    [str(e - 10), str(n - 10), str(e + 1010), str(n + 1010)]
                )
                outfile = os.path.join(output_directory, f"{sqkm}.tif")

                tile_pbar.update(1)
                tile_pbar.set_description(
                    f"Processing easting={e}, northing={n}"
                )
                
                if raster_geom.intersects(bbox_geom):
                    outfile = os.path.join(output_directory, f"{sqkm}.tif")
                else:
                    #tile_pbar.write("Doesn't intersect...continuing...")
                    continue

                projcmd = (
                    f"rio warp {input_tif} {outfile} "
                    f"--dst-crs {utmcode} "
                    f"--bounds {bbox} "
                    "--resampling cubic "
                    "--res 0.5 "
                    "--overwrite"
                )

            
                result = subprocess.run(
                    projcmd.split(" "),
                    capture_output = True,
                    text = True 
                )
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
                else:
                    tile_pbar.write("Continuing...")
        chunk_pbar.update(1)
    chunk_pbar.write(
            f"Finished creating 2020 x 2020 tiles for {os.path.split(input_tif)[-1]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Projecting a GeoTIFF into 1010 x 1010 tiles')
    parser.add_argument('--raw_directory', type=str, required=True,
                        help='Path to the directory containing the raw satellite imagery')
    parser.add_argument('--input_tif', type=str, required=True,
                        help='Path to the projected GeoTIFF')
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to the file containing the site metadata')
    parser.add_argument('--output_directory', type=str, required=True, 
                        help='Path to the output directory for the the tiled GeoTIFFs')

    args = parser.parse_args()

    processAuxTifs(args.input_tif, args.output_directory)
