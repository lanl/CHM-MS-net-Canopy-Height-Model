"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

# SatCHM/infer/main.py
from pathlib import Path
import sys
import shutil 
import os
import re
import argparse
# import geopandas as gpd
from dotenv import load_dotenv
import time
import json
import rasterio
import glob

# import site_config from higher scope
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from site_config import load_site_config
    from site_config import load_global_vars
    import prepTrainInputs.utils as utils
    from ms_net.infer import run_inference
    from prepTrainInputs.pca import reduce_ae_dimensions
    from prepTrainInputs.rasterAE import fetch_alphaEarth
    from prepTrainInputs.utils import tileAlphaEarth
    from ms_net.pore_utils_2D import load_hparams
    from ms_net.plotting_utils import plot_site_inference, save_clean_prediction_png
finally:
    sys.path.pop(0)


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
    pathToLidarResources = os.path.join(project_parent, "prepTrainInputs", "resources.geojson")

    # Parse command line arguments FIRST
    parser = argparse.ArgumentParser(description='Run SatCHM inference for a specific site')
    parser.add_argument('--sites', type=str, required=True, nargs='+',
                    help='Site code (e.g., ws, lm, qm). Overrides .env file if provided.')
    parser.add_argument('--scale', type=str, default='mean', choices=['mean', 'max'],
                    help='Scaling method for output CHM: "mean" or "max" (default: mean)')
    parser.add_argument('--norm-const', type=float, default=46,
                    help='Normalization constant for CHM data (default: 46)')
    parser.add_argument('--feather-const', type=int, default=40,
                    help='Feathering constant for tile merging (default: 40)')
    parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--model-loc', type=str, required=True)
    parser.add_argument("--eval-only", action="store_true")

    args = parser.parse_args()
    sites = args.sites
    # Use command line args or defaults
    SCALING_METHOD = args.scale
    NORM_CONST = args.norm_const
    FEATHER_CONST = args.feather_const
    print(f"Using scaling method: {SCALING_METHOD}")
    # Optional, only needed if trying to compute CBH in treelist
    rdsPath = None

    # get model weights from --model-loc
    pathToWeights = args.model_loc
    checkpoint_epoch = 'unknown'

    # need to read hparams.yaml to get model architecture, and ensure that all the correct site data has been generated during main1 and main2
    hparamsPath = os.path.join(os.path.dirname(os.path.dirname(pathToWeights)), 'hparams.yaml')
    channel = load_hparams(hparamsPath).get('channel')
    print(f"read channel from hparams: \'{channel}\'")
    pca_dims = re.search(r"\d+", channel)

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
        customLidarTifPath = None

        ############### PATH DEFINITIONS ###############
        inf_data_path = os.path.join(project_parent, f'{site}_INF_data')
        
        # When using --eval-only, find existing scaled CHM instead of using default path
        if args.eval_only:
            # Look for existing scaled CHM files
            pattern = os.path.join(inf_data_path, f'{site}_infer_chm_{SCALING_METHOD}_epoch*.tif')
            existing_files = glob.glob(pattern)
            if existing_files:
                # Use the most recent file
                scaledOutputRasterPath = max(existing_files, key=os.path.getmtime)
                print(f"Found existing scaled CHM: {scaledOutputRasterPath}")
            else:
                # Fallback to default path (will error later if file doesn't exist)
                scaledOutputRasterPath = os.path.join(inf_data_path, f'{site}_infer_chm_{SCALING_METHOD}_epoch{checkpoint_epoch}.tif')
        else:
            scaledOutputRasterPath = os.path.join(inf_data_path, f'{site}_infer_chm_{SCALING_METHOD}_epoch{checkpoint_epoch}.tif')

        root, _ = os.path.splitext(inferenceShpPath) # Find inf shape as defined during training input prep
        inf_shp_output_path = root + "_utm.geojson"
        infAnchorsPath = os.path.join(project_parent, 'downloads', site, 'infShape', f'{site}_infAnchors.csv')
        lidarTilesPath = os.path.join(inf_data_path, 'chm')
        dem_download_path = os.path.join(project_parent, 'downloads', site, 'dem', f'{site}_dem_INF_download.tif')
        dem_UTM_path = os.path.join(project_parent, 'downloads', site, 'dem', f'{site}_dem_INF_UTM.tif')
        dem_prenorm_tiles_path = os.path.join(inf_data_path, 'dem_prenorm')
        dem_tiles_path = os.path.join(inf_data_path, 'dem')
        pathToWvimg = os.path.join(project_parent, 'downloads', site, 'wvimgInf')
        prewvimgPath = os.path.join(inf_data_path, 'prewvimg')
        metadataPath = os.path.join(project_parent, 'downloads', site, 'metadata', 'DGTilesMetadata.json')
        outputRasterPath = os.path.join(inf_data_path, 'INF_chm_pred_merged.tif')
        croppedOutputRasterPath = os.path.join(inf_data_path, f'{site}_merged_CHM_inf.tif')

        # AlphaEarth paths
        ae_tiles_path = os.path.join(inf_data_path, 'ae')
        source_raster = os.path.join(ae_tiles_path, 'ae_source.tif')


        # try:
        #     # SITE-SPECIFIC weight selection - THIS IS THE KEY CHANGE!
        #     # Looks for models in: lightning_logs/{site}_model/version_X/checkpoints/
        #     weightsRoot = os.path.join(project_parent, 'CHM-MS-net-Canopy-Height-Model', 'ms_net', 'lightning_logs')
        #     site_model_dir = os.path.join(weightsRoot, f"{site}_model")
            
        #     if not os.path.exists(site_model_dir):
        #         raise FileNotFoundError(
        #             f"No model directory found for site '{site}' at: {site_model_dir}\n"
        #             f"Train a model for this site first using: python ms_net/train.py --site {site}"
        #         )
            
        #     # Find the latest version for THIS site
        #     weightsVersion = max(
        #         (
        #             os.path.join(site_model_dir, d)
        #             for d in os.listdir(site_model_dir)
        #             if d.startswith("version_") and load_hparams(os.path.join(site_model_dir, d, 'hparams.yaml')).get('pca_dims') == args.pca_dims
        #         ),
        #         key=lambda p: int(p.split("_")[-1])
        #     )
            
        #     ckpt_dir = Path(weightsVersion) / "checkpoints"
        #     epoch_re = re.compile(r"best-val-epoch=(\d+)-step=(\d+)\.ckpt")       # searching for best-val
        #     matches = []
        #     for p in ckpt_dir.glob("*.ckpt"):
        #         m = epoch_re.search(p.name)
        #         if m:
        #             matches.append((int(m.group(1)), p))
        #     if not matches:
        #         raise FileNotFoundError(f"No weights files found in {ckpt_dir}")
            
        #     # Get the checkpoint with the best-val-epoch
        #     checkpoint_epoch, pathToWeights = max(matches, key=lambda t: t[0])
        #     print(f"Automatically selected weights for site '{site}': {pathToWeights}")
            
        #     # Create output filename with scaling method and checkpoint info
        #     scaledOutputRasterPath = os.path.join(inf_data_path, f'{site}_infer_chm_{SCALING_METHOD}_epoch{checkpoint_epoch}.tif')
            
        # except Exception as e:
        #     print(f"Failed to automatically select weights: {e}")
        #     print("Using fallback weights path...")
        #     pathToWeights = ''
        #     checkpoint_epoch = 'unknown'
        #     scaledOutputRasterPath = os.path.join(inf_data_path, f'{site}_infer_chm_{SCALING_METHOD}_epoch{checkpoint_epoch}.tif')


        if not args.eval_only:
            ############### RUN INFERENCE PIPELINE ######################
            
            print('\n' + '='*60)
            print(f'STARTING INFERENCE PIPELINE FOR SITE: {site}')
            print('='*60 + '\n')

            # folder creations
            for path in [
                inf_data_path,
                dem_tiles_path,
                dem_prenorm_tiles_path,
                prewvimgPath
            ]:
                if not os.path.isdir(path):
                    print(f"Creating directory: {path}")
                    os.makedirs(path, exist_ok=True)

                if not os.path.isdir(path):
                    warnings.warn(f"Directory could not be verified: {path}")


            ############ PREPARE ANCHORS #############

            # Create points to center tiles along (anchors)
            print('Creating set of tile anchors')

            if not os.path.isfile(inf_shp_output_path) :
                # Create inference shape
                print('No existing utm projection of inference shape. Creating new utm inference shape...')
                print(f'Using inferenceShpPath: {inferenceShpPath}')
                utils.saveInfShape(
                    input_path=inferenceShpPath,
                    output_path=inf_shp_output_path,
                    buffer_meters=512,
                    epsg=epsg,
                )
                print(f'Saved inference shape to: {inf_shp_output_path}')
            else :
                print("Using existing utm inference shape...")

            tileAnchors = utils.genTileAnchors(shp_path=inf_shp_output_path, out_path=infAnchorsPath, buffer = 32)
            print(f'Saved tile anchors to {infAnchorsPath}')

            # ################## PROCESS dem DATA ###########################

            # fetch dem data from openTopo
            print('Requesting dem data from OpenTopography')
            utils.fetch_DEM(geojson_path=inf_shp_output_path, save_path=dem_download_path, api_key=openTopoAPIkey)
            print(f'Saved dem data to: {dem_download_path}')

            # reproject dem data
            print('Reprojecting dem data. This will take a while...')
            utils.saveRasterToUTM(rasterPath=dem_download_path, epsg=epsg, savePath=dem_UTM_path)
            print(f'Saved reprojected dem data to: {dem_UTM_path}')

            # tile out dem data
            print('Tiling dem data')
            utils.tileRaster(pathToRaster=dem_UTM_path, outputPath=dem_prenorm_tiles_path, dataType='dem', anchors_csv=infAnchorsPath)
            print(f'Saved dem tiles to: {dem_prenorm_tiles_path}')

            # normalize dem tiles
            print(f'Normalizing dem data from {dem_prenorm_tiles_path}')
            utils.normDEMs(src_path=dem_prenorm_tiles_path, dst_path=dem_tiles_path)
            shutil.rmtree(dem_prenorm_tiles_path)
            print(f'Saved normalized dem tiles to {dem_tiles_path}')


            ################### PROCESS WVIMG DATA #####################

            # merge wvimg (will take a while)
            print('Merging wvimg tiles')
            utils.mergeTifs(pathToWvimg, prewvimgPath)
            utils.checkCRS(prewvimgPath, epsg) # verify that wvimg is in correct crs
            print(f'Saved merged wvimg tiles to {prewvimgPath}')

            # get wvimg metadata
            print('Saving wvimg metadata')
            utils.saveWvimgMetadata(wvimgPath=pathToWvimg, savePath=metadataPath, prewvimgPath=prewvimgPath)
            print(f'Saved metadata to: {metadataPath}')

            # generate sensor and solar tiles
            print('Generating Sensor and Solar Tiles')
            utils.generate_sensor_solar_tiles(metadata_json_path=metadataPath, output_directory=inf_data_path)
            print(f'Saved sensor and solar tiles to: {inf_data_path}')

            # tile out wvimg data for model partition
            # NOTE: this requires multiple wvimg tiles to be partitioned, as we are using both tiles for areas with intersecting tiles
            print('Tiling wvimg data for model partition')
            for tif in [f for f in os.listdir(prewvimgPath) if f.endswith('.tif')]:
                utils.tileRaster(pathToRaster=os.path.join(prewvimgPath, tif), outputPath=os.path.join(inf_data_path, 'wvimg'), dataType = 'wvimg', anchors_csv=infAnchorsPath)
            print(f'Saved wvimg tiles to: {os.path.join(inf_data_path, "wvimg")}')

            # remove prewvimg folder
            shutil.rmtree(prewvimgPath)


            ################## RENAME AND CREATE LISTS ##############

            # Extract year from wvimg metadata for AlphaEarth alignment
            with open(metadataPath, 'r') as f:
                metadata = json.load(f)
            # Get the first tile's date and extract the year
            first_tile = next(iter(metadata.values()))
            date_str = first_tile['date']  # Format: 'YYYY-MM-DD'
            inferenceYear = float(date_str.split('-')[0])
            print(f'Extracted inference year from wvimg metadata: {inferenceYear}')

            ################### PROCESS ALPHAEARTH DATA (CONDITIONAL) #####################
            
            # FIXME: somewhere we need to verify that the site data paths actually exist for AE, but ultimately everything

            # conditional if AE is being used
            if "base-ae" in channel :
                print('=== AlphaEarth Processing Enabled ===')
                os.makedirs(ae_tiles_path, exist_ok=True)

                # Download AlphaEarth Embeddings
                print(f'Fetching AlphaEarth data to {source_raster}')
                
                result = fetch_alphaEarth(
                    geojson_path=inferenceShpPath,
                    save_path=str(source_raster),
                    year=int(inferenceYear),
                    epsg=epsg,
                    sa_key_path=geeKey,
                    project=geeProject,
                    temp_dir=ae_tiles_path
                )
                # Validate source
                with rasterio.open(result) as src:
                    print(f"Source: {src.count} bands, {src.dtypes[0]}, {src.width}x{src.height}, {src.res}")
                    # Expected: Source: 64 bands, int8, WxH, (10.0, 10.0)

                if "base-ae-" in channel :
                    print(f"Inferring with AE channel: {channel}")
                    output_dir = Path(source_raster).parent
                    # call PCA for all default dimensions
                    results = reduce_ae_dimensions(
                        ae_source_path=str(source_raster),
                        output_dir=str(output_dir),
                        site_name=site,
                        target_dims=pca_dims,
                        validate=False,
                    )
                    dim_ae_tiles_path = os.path.join(ae_tiles_path, f'{pca_dims}d')
                    dim_source_raster = os.path.join(dim_ae_tiles_path, f'ae_source_{pca_dims}d.tif')
                    # Tile AlphaEarth data
                    print('Tiling AlphaEarth data for model partition')
                    tileAlphaEarth(
                        pathToRaster=str(dim_source_raster),
                        outputPath=str(dim_ae_tiles_path),
                        anchors_csv=infAnchorsPath,
                        epsg=epsg,
                        expected_bands=pca_dims
                    )
                    print(f'Saved AlphaEarth tiles to {dim_ae_tiles_path}')
                    tiles = glob.glob(os.path.join(dim_ae_tiles_path, "*.tif"))
                    print(f"Generated {len(tiles)} AlphaEarth tiles")
                    if tiles:
                        with rasterio.open(tiles[0]) as src:
                            print(f"Tile: {src.count} bands, {src.dtypes[0]}, {src.width}*{src.height}, {src.res}")
                            # Expected: Tile: 64 bands, int8, 512x512, (0.5, 0.5)

                else :
                    print("Inferring with default 64 AE channels")
                    # Tile AlphaEarth data
                    print('Tiling AlphaEarth data for model partition')
                    tileAlphaEarth(
                        pathToRaster=str(source_raster),
                        outputPath=str(ae_tiles_path),
                        anchors_csv=infAnchorsPath,
                        epsg=epsg,
                    )
                    print(f'Saved AlphaEarth tiles to {ae_tiles_path}')
                    # Validate tiles
                    tiles = glob.glob(os.path.join(ae_tiles_path, "*.tif"))
                    print(f"Generated {len(tiles)} AlphaEarth tiles")
                    if tiles:
                        with rasterio.open(tiles[0]) as src:
                            print(f"Tile: {src.count} bands, {src.dtypes[0]}, {src.width}*{src.height}, {src.res}")
                            # Expected: Tile: 64 bands, int8, 512x512, (0.5, 0.5)


            ################## RENAME AND CREATE LISTS ##############

            # rename tiles to preserve associations between input tiles and sat/solar angle tiles
            print('Renaming tiles')
            utils.renameTiles(inf_data_path)
            print(f'Renamed input tiles, saved in {inf_data_path}')

            # Create lists to feed to models
            print('Creating model lists')
            utils.makeInfList(inf_data_path, site)
            print(f'Created model lists, saved in {inf_data_path}')

            ################ RUN INFERENCE #######################

            print('Running inference')
            run_inference(
                    data_path=inf_data_path,
                    site=site,
                    NORM_CONST=NORM_CONST,
                    model_loc=pathToWeights,
                    epsg_code=epsg,
                    channel=channel  # Pass channel extracted from hparams
                )
            print(f'Ran inference, tiles saved to {os.path.join(inf_data_path, "chm_preds")}')

            ############### MERGE CHM TILES ######################

            print('Merging chm tiles')
            utils.merge_chm_tiles(
                input_folder=os.path.join(inf_data_path, 'chm_preds'),
                output_tif=outputRasterPath,
                b=FEATHER_CONST,
            )
            print(f'Merged chm tiles, saved merged raster to {outputRasterPath}')

            ###### CROP CHM RASTER BACK TO ORIGINAL SHAPE ########
            print('Cropping predicted CHM tif back to original shape')
            utils.cropTif(inputTif=outputRasterPath, shp=inferenceShpPath, outputTif=croppedOutputRasterPath, epsg=epsg)
            print(f'Cropped CHM tif, saved to {croppedOutputRasterPath}')

            ####### SCALE RASTER WITH RESPECT TO TRAINING DATA #######

            print(f'Scaling predicted CHM tif with respect to training data CHM (method: {SCALING_METHOD})')
            training_chm_path = os.path.join(project_parent, f'{site}_data', 'chm')
            utils.scale_tif(geojson_path=inf_shp_output_path, input_lidar_tifs_folder_path=training_chm_path, pred_lidar_tif_path=croppedOutputRasterPath, output_path=scaledOutputRasterPath, option=SCALING_METHOD)
            print(f'Scaled CHM tif, saved to {scaledOutputRasterPath}')

        
        ############### EVALUATE AND GENERATE PLOTS ######################
        
        print('\n' + '='*60)
        print(f'EVALUATION AND PLOTTING FOR SITE: {site}')
        print('='*60 + '\n')
        
        # Check if predicted CHM exists
        if not os.path.exists(scaledOutputRasterPath):
            raise FileNotFoundError(
                f"Predicted CHM not found: {scaledOutputRasterPath}\n"
                f"Please run inference first without --eval-only flag."
            )
        
        # Merge observed CHM if needed
        observed_raster = os.path.join(project_parent, f'{site}_data', f'{site}_observed_CHM_merged.tif')
        obs_chm_folder = os.path.join(project_parent, f'{site}_data', 'chm')
        
        if not os.path.exists(observed_raster):
            if not os.path.isdir(obs_chm_folder):
                raise FileNotFoundError(
                    f"Observed CHM tiles not found: {obs_chm_folder}\n"
                    f"Ground truth data is required for evaluation."
                )
            print(f"Merging observed CHM tiles from: {obs_chm_folder}")
            utils.merge_chm_tiles(
                input_folder=obs_chm_folder,
                output_tif=observed_raster,
                b=FEATHER_CONST
            )
            print(f"Merged observed CHM to: {observed_raster}")
        else:
            print(f"Using existing observed CHM: {observed_raster}")
        
        # Generate full evaluation plot
        plot_output_path = os.path.join(inf_data_path, f'{site}_site_plot.png')
        print(f"\nGenerating evaluation plot...")
        plot_site_inference(
            predicted_raster=scaledOutputRasterPath,
            observed_raster=observed_raster,
            imagery_raster=None,
            output_path=plot_output_path,
            title=f"Site: {site}",
            downsample_factor=4
        )
        
        # Generate clean prediction PNG
        clean_png_path = os.path.join(inf_data_path, f'{site}_prediction_only_plot.png')
        print(f"\nGenerating clean prediction image...")
        save_clean_prediction_png(
            predicted_raster=scaledOutputRasterPath,
            output_path=clean_png_path,
            downsample_factor=4
        )
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE FOR SITE: {site}")
        print(f"  - Full plot: {plot_output_path}")
        print(f"  - Clean PNG: {clean_png_path}")
        print(f"{'='*60}\n")

        ############# GENERATE TREELIST ####################

        # print('Generating treelist with cloud2trees. This will take a while...')
        # if rdsPath != None:
        #     utils.genTreelist(tifPath=croppedOutputRasterPath, projectPath=project_parent, rdsPath=rdsPath, epsg=epsg)
        # else:
        #     utils.genTreelist(tifPath=croppedOutputRasterPath, projectPath=project_parent, epsg=epsg)
        # print(f'Generated treelist, saved to {os.path.join(os.path.dirname(croppedOutputRasterPath), "treelist.csv")}')


# Standard multiprocessing guard
if __name__ == "__main__":
    main()