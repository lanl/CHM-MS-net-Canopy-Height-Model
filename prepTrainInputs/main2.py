"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

import os
import argparse
from dotenv import load_dotenv
import shutil
import utils
import time
from utils import tileAlphaEarth
import rasterio
import sys
import glob

# import site_config from higher scope
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from site_config import load_site_config
finally:
    sys.path.pop(0)


# Target dimensions for PCA reduction
TARGET_DIMENSIONS = [4, 8, 16, 32, 64]

def main() :
    
    ################### SETUP #################
    
    # load global variables
    global_vars = load_global_vars()
    project_dir = global_vars['project_dir']
    default_channels = global_vars['channels']
    openTopoAPIkey = global_vars['openTopoAPIkey']
    geeKey = global_vars['geeKey']
    geeProject = global_vars['geeProject']

    # variables not currently implemented within config.json  (will consider adding later)
    # maxarAPIkey
    # pathToLidarResources
    numTrainImages = 1000

    # define base paths (maybe load project_dir from load_global_vars)
    project_parent = os.path.abspath(os.path.join(project_dir, ".."))
    print(f"project_parent: {project_parent}")
    site_data_path = os.path.join(project_parent, f'{site}_data')
    lidar_tiles_path = os.path.join(site_data_path, 'chm')
    pathToLidarResources = os.path.join(project_dir, "prepTrainInputs", "resources.geojson")

    # Folder creations
    os.makedirs(site_data_path, exist_ok=True)
    os.makedirs(os.path.join(site_data_path, 'wvimg'), exist_ok=True)
    wvimg_train_path = os.path.join(project_parent, 'downloads', site, 'wvimgTrain')
    wvimg_inf_path = os.path.join(project_parent, 'downloads', site, 'wvimgInf')
    os.makedirs(wvimg_train_path, exist_ok=True)
    os.makedirs(wvimg_inf_path, exist_ok=True)

    # AlphaEarth paths and folder creations
    ae_tiles_path = os.path.join(site_data_path, 'ae')
    source_raster = os.path.join(ae_tiles_path, 'ae_source.tif')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare training data (step 2) for a list of sites')
    parser.add_argument('--sites', type=str, nargs='+', required=True,
                       help='Site code (e.g., ws, lm, qm). Overrides .env file if provided.')
    parser.add_argument('--channels', type=str, choices=default_channels, nargs='+', default=default_channels, required=False
                        help=f'Used within the pca process. Choices are specified in config.json')
    args = parser.parse_args()


    sites = args.sites
    # for each site in sites, generate data
    for site in sites :
        print(f"Current site is:  {site}")

        site_config = load_site_config(site)
        epsg = site_config['epsg']
        # FIXME: right now train and inf shape paths are entangled, but eventually we'll just be using
        #        one shpPath per site, and decide which site(s) to train and infer elsewhere.
        inferenceShpPath = site_config['inferenceShpPath']
        customTrainShpPath = site_config['trainShpPath']
        customTrainShpPath = None      # keep None for now to avoid alternate training behavior


        # merge wvimg (will take a while)
        print('Merging wvimg tiles')
        wvimgMergedPath = os.path.join(site_data_path, 'prewvimg')
        print(f"wvimgMergedPath: {wvimgMergedPath}")
        pathToWvimg = os.path.join(project_path, 'downloads', site, 'wvimgTrain')
        os.makedirs(wvimgMergedPath, exist_ok=True)
        utils.mergeTifs(pathToWvimg, wvimgMergedPath)
        print(f'Saved merged wvimg tiles to {wvimgMergedPath}')

        #FIXME: There really should be a check to make sure that sites.json contains the correct EPSG code in main1.py so that this doesn't fail
        # verify that wvimg is in correct crs
        utils.checkCRS(wvimgMergedPath, epsg)

        # get wvimg metadata
        print('Saving wvimg metadata')
        metadataPath = os.path.join(project_path, 'downloads', site, 'metadata', 'DGTilesMetadata.json')
        os.makedirs(os.path.dirname(metadataPath), exist_ok=True)
        utils.saveWvimgMetadata(wvimgPath=pathToWvimg, savePath=metadataPath, prewvimgPath=wvimgMergedPath)
        print(f'Saved metadata to: {metadataPath}')

        # generate sensor and solar tiles
        print('Generating Sensor and Solar Tiles')
        utils.generate_sensor_solar_tiles(metadata_json_path=metadataPath, output_directory=site_data_path)
        print(f'Saved sensor and solar tiles to: {site_data_path}')

        # tile out wvimg data for model partition
        # NOTE: this requires multiple wvimg tiles to be partitioned, as we are using both tiles for areas with intersecting tiles
        print('Tiling wvimg data for model partition')
        trainAnchorsPath = os.path.join(project_path, 'downloads', site, 'trainShape', f'{site}_trainAnchors.csv')
        for tif in [f for f in os.listdir(wvimgMergedPath) if f.endswith('.tif')]:
            # only select tiles that have been cropped to the model partition
            utils.tileRaster(pathToRaster=os.path.join(wvimgMergedPath, tif), outputPath=os.path.join(site_data_path, 'wvimg'), dataType = 'wvimg', anchors_csv=trainAnchorsPath)
        print(f'Saved wvimg tiles to: {site_data_path}/wvimg/')


        # conditional if AE is being used
        if any("ae" in item for item in channels) :
            # tile out non-PCA AlpahEarth
            print("Tiling non-PCA AlphaEarth embeddings...")
            tileAlphaEarth(
                pathToRaster=str(source_raster),
                outputPath=str(ae_tiles_path),
                anchors_csv=trainAnchorsPath,
                epsg=epsg,
            )

            # if we are using AE and PCA
            if any("ae-" in item for item in channels) :
                target_dims = get_pca_dims(channels)

                # iterate through each listed dimension specified in target_dims
                for n_components in target_dims :
                    dim_ae_tiles_path = os.path.join(ae_tiles_path, f'{n_components}d')
                    dim_source_raster = os.path.join(dim_ae_tiles_path, f'ae_source_{n_components}d.tif')

                    # check if folder for current n_component exists
                    if not os.path.isfile(dim_source_raster):
                        raise FileNotFoundError(f"The source raster at: '{dim_source_raster}' does not exist.")

                    print(f"Tiling {n_components}-dimensional AlphaEarth embeddings...")

                    dim_bands = n_components
                    # tile out AlpahEarth
                    tileAlphaEarth(
                        pathToRaster=str(dim_source_raster),
                        outputPath=str(dim_ae_tiles_path),
                        anchors_csv=trainAnchorsPath,
                        epsg=epsg,
                        expected_bands=dim_bands
                    )
                    # Validate tiles
                    tiles = glob.glob(os.path.join(dim_ae_tiles_path, "*.tif"))
                    print(f"Generated {len(tiles)} tiles")
                    with rasterio.open(tiles[0]) as src:
                        print(f"Tile: {src.count} bands, {src.dtypes[0]}, {src.width}x{src.height}, {src.res}")

            print("Done tiling AlphaEarth embeddings")

        # rename tiles to preserve associations between input tiles and sat/solar angle tiles
        print('Renaming tiles')
        utils.renameTiles(site_data_path)
        print(f'Renamed input tiles, saved in {site_data_path}')

        # Clean up unnecessary dirs
        shutil.rmtree(wvimgMergedPath, ignore_errors=True)
        shutil.rmtree(os.path.join(site_data_path, 'dem_prenorm'), ignore_errors=True)

        # Create lists to feed to models
        print('Creating model lists')
        utils.makeLists(site_data_path, site)
        print(f'Created model lists, saved in {site_data_path}')


# Standard multiprocessing guard
if __name__ == "__main__":
    main()
