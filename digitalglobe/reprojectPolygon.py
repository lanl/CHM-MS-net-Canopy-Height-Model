"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


"""

This script reprojects the shp file into UTM and saves the metadata about the site's bounding box into a csv. Written by Mia Mitchell & Zachary Crennen in Los Alamos, New Mexico. Winter 2025.

"""
import geopandas as gpd
import os
import argparse
import csv
from dotenv import load_dotenv, find_dotenv

def reproject_shapefile(shp_file_path):
    """
    Description
    ___________
    Main function reprojects the shp file & saves the metadata of the bounds

    Parameters
    __________
    shp_file_path : str
        The path to the shapefile
    
    Returns
    ______
    output_shp_path : str
        The output path for the reprojected shapefile
    output_csv_path : str
        The output path for the site metadata csv
    """
    # Pulled from the .env 
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    project_directory=os.getenv('project_dir')
    utm = os.getenv('utm')
    site = os.getenv('site')

    # Output paths
    output_path = os.path.join(project_directory, f'site-polygons/with_crs/{site}')

    # Reprojecting polygon to utm determined in .env
    gdf = gpd.read_file(shp_file_path)
    gdf = gdf[["geometry"]]
    gdf=gdf.to_crs(utm) 
    output_shp_path = os.path.join("/", output_path, f'{site}.shp')
    gdf.to_file(output_shp_path)
    #print(f"Reprojected {site} polygon...\n")
    #print(f"Obtained bounds metadata for {site} polygon...\n")


    # Obtaining bounds for shapefile
    minx, miny, maxx, maxy = gdf.total_bounds


    data = { 
    'name': [site],
    'e0': [minx],
    'e1': [maxx],
    'n0': [miny],
    'n1': [maxy],
    'utm_code': ['EPSG:32610'],
    'chunk_or_whole' : ['whole bounds'] #ZACH
    }

    # Save site metadata to a csv
    output_csv_path = os.path.join(project_directory, "metadata", "site-metadata.csv")

    if os.path.exists(output_csv_path):
         os.remove(output_csv_path)
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow({k: str(v[0]) for k, v in data.items()})

    #print(f"Saved metadata to {os.path.basename(output_csv_path)}...\n")

    return output_shp_path, output_csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reprojects a polygon based on UTM listed in .env and saves bounds metadata in a csv file.')

    parser.add_argument('--shp_file_path', type=str, help='Path to the unprojected shapefile', required=True)

    args = parser.parse_args()

    reproject_shapefile(args.shp_file_path)
