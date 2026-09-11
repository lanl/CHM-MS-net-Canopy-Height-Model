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
from site_config import load_site_config

# Parse command line arguments FIRST
parser = argparse.ArgumentParser(description='Prepare training data (step 2) for a specific site')
parser.add_argument('--site', type=str, required=False,
                   help='Site code (e.g., ws, lm, qm). Overrides .env file if provided.')
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

# Path definitions
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
site_data_path = os.path.join(project_path, f'{site}_data')

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
