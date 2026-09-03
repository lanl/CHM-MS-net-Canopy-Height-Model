"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

# SatCHM/infer/main.py
from pathlib import Path
import sys

# Add the project root (the folder that contains `prepTrainInputs/`) to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepTrainInputs.utils as utils
from ms_net.infer import run_inference
import shutil 
import os
import re
import argparse
# import geopandas as gpd
from dotenv import load_dotenv
import time
from prepTrainInputs.site_config import load_site_config

################### SETUP #################
# Parse command line arguments FIRST
parser = argparse.ArgumentParser(description='Run SatCHM inference for a specific site')
parser.add_argument('--site', type=str, required=False,
                   help='Site code (e.g., ws, lm, qm). Overrides .env file if provided.')
parser.add_argument('--scale', type=str, default='mean', choices=['mean', 'max'],
                   help='Scaling method for output CHM: "mean" or "max" (default: mean)')
parser.add_argument('--norm-const', type=float, default=46,
                   help='Normalization constant for CHM data (default: 46)')
parser.add_argument('--feather-const', type=int, default=40,
                   help='Feathering constant for tile merging (default: 40)')
args = parser.parse_args()

# Load env variables
load_dotenv()

# Get site from command line or fall back to .env
if args.site:
    site = args.site
    print(f"✓ Using site from command line: {site}")
    # load vars from config file
    config = load_site_config(site)
    epsg = config['epsg']
    openTopoAPIkey = config['openTopoAPIkey']
    inferenceShpPath = config['inferenceShpPath']
else:
    site = os.getenv('site')
    if not site:
        raise ValueError("Site must be specified via --site flag or in .env file")
    print(f"✓ Using site from .env file: {site}")
    epsg = int(os.getenv('epsg'))
    openTopoAPIkey = os.getenv('openTopoAPIkey')
    inferenceShpPath = os.getenv('inferenceShpPath')

# Load other env variables
chmPath = os.getenv('chmPath')
chmReducedPath = os.getenv('chmReducedPath')
shpPath = os.getenv('shpPath')
customTrainShpPath = os.getenv('customTrainShpPath')
fp_path = os.getenv('fp_path')
maxarAPIkey = os.getenv('maxarAPIkey')
customLidarTifPath = os.getenv('customLidarTifPath')
numTrainImages = 1000

# Use command line args or defaults
SCALING_METHOD = args.scale
NORM_CONST = args.norm_const
FEATHER_CONST = args.feather_const

print(f"✓ Using scaling method: {SCALING_METHOD}")

############### PATH DEFINITIONS ###############
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
inf_data_path = os.path.join(project_path, f'{site}_INF_data')
pathToLidarResources = os.path.join(project_path, 'resources.geojson')
root, _ = os.path.splitext(inferenceShpPath) # Find inf shape as defined during training input prep
inf_shp_output_path = root + "_utm.geojson"
infAnchorsPath = os.path.join(project_path, 'downloads', site, 'infShape', f'{site}_infAnchors.csv')
lidarTilesPath = os.path.join(inf_data_path, 'chm')
dem_download_path = os.path.join(project_path, 'downloads', site, 'dem', f'{site}_dem_INF_download.tif')
dem_UTM_path = os.path.join(project_path, 'downloads', site, 'dem', f'{site}_dem_INF_UTM.tif')
dem_prenorm_tiles_path = os.path.join(inf_data_path, 'dem_prenorm')
dem_tiles_path = os.path.join(inf_data_path, 'dem')
pathToWvimg = os.path.join(project_path, 'downloads', site, 'wvimgInf')
prewvimgPath = os.path.join(inf_data_path, 'prewvimg')
metadataPath = os.path.join(project_path, 'downloads', site, 'metadata', 'DGTilesMetadata.json')
outputRasterPath = os.path.join(inf_data_path, 'INF_chm_pred_merged.tif')
croppedOutputRasterPath = os.path.join(inf_data_path, f'{site}_merged_CHM_inf.tif')

try:
    # SITE-SPECIFIC weight selection - THIS IS THE KEY CHANGE!
    # Looks for models in: lightning_logs/{site}_model/version_X/checkpoints/
    weightsRoot = os.path.join(project_path, 'SatCHM', 'ms_net', 'lightning_logs')
    site_model_dir = os.path.join(weightsRoot, f"{site}_model")
    
    if not os.path.exists(site_model_dir):
        raise FileNotFoundError(
            f"No model directory found for site '{site}' at: {site_model_dir}\n"
            f"Train a model for this site first using: python ms_net/train.py --site {site}"
        )
    
    # Find the latest version for THIS site
    weightsVersion = max(
        (
            os.path.join(site_model_dir, d)
            for d in os.listdir(site_model_dir)
            if d.startswith("version_")
        ),
        key=lambda p: int(p.split("_")[-1])
    )
    
    ckpt_dir = Path(weightsVersion) / "checkpoints"
    epoch_re = re.compile(r"epoch=(\d+)\.ckpt$")
    matches = []
    for p in ckpt_dir.glob("*.ckpt"):
        m = epoch_re.search(p.name)
        if m:
            matches.append((int(m.group(1)), p))
    if not matches:
        raise FileNotFoundError(f"No weights files found in {ckpt_dir}")
    
    # Get the checkpoint with highest epoch number
    checkpoint_epoch, pathToWeights = max(matches, key=lambda t: t[0])
    print(f"✓ Automatically selected weights for site '{site}': {pathToWeights}")
    
    # Create output filename with scaling method and checkpoint info
    scaledOutputRasterPath = os.path.join(inf_data_path, f'{site}_infer_chm_{SCALING_METHOD}_epoch{checkpoint_epoch}.tif')
    
except Exception as e:
    print(f"⚠️  Failed to automatically select weights: {e}")
    print("Using fallback weights path...")
    pathToWeights = ''
    checkpoint_epoch = 'unknown'
    scaledOutputRasterPath = os.path.join(inf_data_path, f'{site}_infer_chm_{SCALING_METHOD}_epoch{checkpoint_epoch}.tif')

# OPTIONAL: Uncomment to manually override
# pathToWeights = ''

# Optional, only needed if trying to compute CBH in treelist
rdsPath = None

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
        epsg_code=epsg
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
training_chm_path = os.path.join(project_path, f'{site}_data', 'chm')
utils.scale_tif(geojson_path=inf_shp_output_path, input_lidar_tifs_folder_path=training_chm_path, pred_lidar_tif_path=croppedOutputRasterPath, output_path=scaledOutputRasterPath, option=SCALING_METHOD)
print(f'Scaled CHM tif, saved to {scaledOutputRasterPath}')

############# GENERATE TREELIST ####################

# print('Generating treelist with cloud2trees. This will take a while...')
# if rdsPath != None:
#     utils.genTreelist(tifPath=croppedOutputRasterPath, projectPath=project_path, rdsPath=rdsPath, epsg=epsg)
# else:
#     utils.genTreelist(tifPath=croppedOutputRasterPath, projectPath=project_path, epsg=epsg)
# print(f'Generated treelist, saved to {os.path.join(os.path.dirname(croppedOutputRasterPath), "treelist.csv")}')
