"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

import os
import time
import shutil
import geopandas as gpd
from dotenv import load_dotenv
import utils


def main():
    ################### SETUP #################

    # Load env variables
    load_dotenv()
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    chmPath = os.getenv('chmPath')
    chmReducedPath = os.getenv('chmReducedPath')
    shpPath = os.getenv('shpPath')
    epsg = int(os.getenv('epsg'))
    site = os.getenv('site')
    inferenceShpPath = os.getenv('inferenceShpPath')
    customTrainShpPath = os.getenv('customTrainShpPath')
    fp_path = os.getenv('fp_path')
    openTopoAPIkey = os.getenv('openTopoAPIkey')
    maxarAPIkey = os.getenv('maxarAPIkey')
    customLidarTifPath = os.getenv('customLidarTifPath')
    numTrainImages = 1000

    # Path definitions
    site_data_path = os.path.join(project_path, f'{site}_data')
    lidar_tiles_path = os.path.join(site_data_path, 'chm')
    pathToLidarResources = os.path.join(os.path.dirname(__file__), "resources.geojson")

    # Folder creations
    os.makedirs(site_data_path, exist_ok=True)
    os.makedirs(os.path.join(site_data_path, 'wvimg'), exist_ok=True)
    print(f'CREATING FOLDER: {os.path.join(project_path, 'downloads', site, 'wvimgTrain')}')
    print(f'CREATING FOLDER: {os.path.join(project_path, 'downloads', site, 'wvimgInf')}')
    os.makedirs(os.path.join(project_path, 'downloads', site, 'wvimgTrain'), exist_ok=True)
    os.makedirs(os.path.join(project_path, 'downloads', site, 'wvimgInf'), exist_ok=True)


    ############ PREPARE SHAPES AND ANCHORS #############

    # Create inference shape
    print('Creating inference shape')
    print(f'inferenceShpPath: {inferenceShpPath}')
    root, _ = os.path.splitext(inferenceShpPath)
    inf_shp_output_path = root + "_utm.geojson"
    utils.saveInfShape(
        input_path=inferenceShpPath,
        output_path=inf_shp_output_path,
        buffer_meters=512,
        epsg=epsg,
    )
    print(f'Saved inference shape to: {inf_shp_output_path}')

    trainShpPath = os.path.join(
        project_path,
        'downloads',
        site,
        'trainShp',
        f'{site}_trainShp_utm.geojson',
    )

    # If we have custom lidar, crop to its shape
    if customLidarTifPath and os.path.isfile(customLidarTifPath):
        print(f'customLidarTifPath: {customLidarTifPath}')
        print('Reprojecting lidar to UTM, this will take a while...')
        lidarBaseName, _ = os.path.splitext(customLidarTifPath)
        lidar_UTM_path = f'{lidarBaseName}_utm.tif'
        utils.saveRasterToUTM(customLidarTifPath, epsg, lidar_UTM_path)
        print(f'Saved reprojected lidar tif to: {lidar_UTM_path}')

        print('Creating lidar shape')
        lidarShpPath = f'{lidarBaseName}.geojson'
        lidarShp = utils.getRasterShape(lidar_UTM_path)
        lidarShp.to_file(lidarShpPath)
        trainShpPath = lidarShpPath
        print(f'Saved inference shape to: {lidarShpPath}')

    # Else, use a custom train shape if provided
    elif customTrainShpPath and os.path.isfile(customTrainShpPath):
        print('Cropping and cleaning trainShp')
        print(
            f'inference_path={inf_shp_output_path}, '
            f'rough_train_path={customTrainShpPath}'
        )
        utils.place_train_rect_within_mask(
            inference_path=inf_shp_output_path,
            rough_train_path=customTrainShpPath,
            output_path=trainShpPath,
            n_tiles=numTrainImages,
            epsg=epsg,
        )
        print(f'Cleaned trainShp and saved to {trainShpPath}')

    # Else, create a square around inference region
    else:
        print('Creating train shape')
        utils.build_train_square_near_inference(
            inf_shp_output_path, trainShpPath, epsg=epsg
        )
        print(f'Saved train shape to {trainShpPath}')

    # Create tile anchors
    print('Creating set of tile anchors')
    trainAnchorsPath = os.path.join(
        project_path,
        'downloads',
        site,
        'trainShape',
        f'{site}_trainAnchors.csv',
    )
    if customLidarTifPath and os.path.isfile(customLidarTifPath):
        tileAnchors = utils.genTileAnchors(
            shp_path=trainShpPath, out_path=trainAnchorsPath, margin=512
        )
    else:
        tileAnchors = utils.genTileAnchors(
            shp_path=trainShpPath, out_path=trainAnchorsPath
        )
    print(f'Saved tile anchors to {trainAnchorsPath}')

    ############# PROCESS LIDAR DATA ############################

    if customLidarTifPath and os.path.isfile(customLidarTifPath):
        print('Tiling lidar data for model partition')
        lidarBaseName, _ = os.path.splitext(customLidarTifPath)
        lidar_UTM_path = f'{lidarBaseName}_utm.tif'
        print(f'lidar_UTM_path: {lidar_UTM_path}')
        utils.tileRaster(
            pathToRaster=lidar_UTM_path,
            outputPath=lidar_tiles_path,
            dataType='chm',
            anchors_csv=trainAnchorsPath,
        )
        print(f'Saved lidar tiles to: {lidar_tiles_path}')

    else:
        print('Downloading lidar data from USGS 3DEP')
        alignmentYear = utils.createLidarData(
            trainShpPath,
            pathToLidarResources,
            epsg,
            lidarTilesPath=lidar_tiles_path,
            anchors_csv=trainAnchorsPath,
            projectPath=project_path,
        )
        utils.remove_rasters_with_nodata(lidar_tiles_path, dry_run=False)
        shutil.rmtree(os.path.join(lidar_tiles_path, '_laz_tmp'), ignore_errors=True)
        print(f'Saved lidar data to: {lidar_tiles_path}')

    
    ################## PROCESS DEM DATA ###########################

    print('Requesting DEM data from OpenTopography')
    DEM_download_dir = os.path.join(
        project_path,
        'downloads',
        site,
        'dem'
    )
    DEM_download_path = os.path.join(DEM_download_dir, f'{site}_dem_download.tif')
    os.makedirs(DEM_download_dir, exist_ok=True)
    utils.fetch_DEM(
        geojson_path=trainShpPath,
        save_path=DEM_download_path,
        api_key=openTopoAPIkey,
        buffer_m=512,
    )
    print(f'Saved DEM data to: {DEM_download_path}')

    print('Reprojecting DEM data. This will take a while...')
    dem_UTM_path = os.path.join(
        project_path,
        'downloads',
        site,
        'dem',
        f'{site}_dem_UTM.tif',
    )
    utils.saveRasterToUTM(
        rasterPath=DEM_download_path, epsg=epsg, savePath=dem_UTM_path
    )
    print(f'Saved reprojected DEM data to: {dem_UTM_path}')

    print('Tiling DEM data')
    DEM_prenorm_tiles_path = os.path.join(site_data_path, 'dem_prenorm')
    os.makedirs(DEM_prenorm_tiles_path, exist_ok=True)
    utils.tileRaster(
        pathToRaster=dem_UTM_path,
        outputPath=DEM_prenorm_tiles_path,
        dataType='DEM',
        anchors_csv=trainAnchorsPath,
    )
    print(f'Saved DEM tiles to: {DEM_prenorm_tiles_path}')

    print(f'Normalizing DEM data from {DEM_prenorm_tiles_path}')
    DEM_tiles_path = os.path.join(site_data_path, 'dem')
    os.makedirs(DEM_tiles_path, exist_ok=True)
    utils.normDEMs(src_path=DEM_prenorm_tiles_path, dst_path=DEM_tiles_path)
    shutil.rmtree(DEM_prenorm_tiles_path, ignore_errors=True)
    print(f'Saved normalized DEM tiles to {DEM_tiles_path}')

    print('==========================================================')
    print(f'Align satellite imagery to year: {alignmentYear}')
    print(f'Train shape saved at: {trainShpPath}')
    print(f'Inference shape saved at: {inf_shp_output_path}')
    print('==========================================================')

# ✅ Standard multiprocessing guard
if __name__ == "__main__":
    main()
