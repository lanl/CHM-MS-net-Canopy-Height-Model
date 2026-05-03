"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


"""
Main script for SatCHM. Written by Mia Mitchell.
"""
import os
import shutil
import pandas as pd
import time
from dotenv import load_dotenv, find_dotenv

# Pre processing 
import setup.start_project as start
import digitalglobe.reprojectPolygon as reprojpoly
import digitalglobe.generateWKT as generateforDG
import process_imagery.manageDirectories as manageDirectories
import preparing_forestdata.reprojectCrop as reprojectCrop
import process_imagery.processLidarDEM as processLidarDEM
import preparing_forestdata.PrepareCRSForestData as prepareCRSForestData
import process_imagery.projectImagery as projectSatelliteImagery
import aux_inputs.GenerateSolarAndSensorLayers as createAnglearrays
import preparing_forestdata.match_keys as matchkeys
import preparing_forestdata.train_val_list as train_val_list

# Neural Network : Ms-net
from ms_net_architecture.train import train_main


# Extra tools
from tools.styles import style
from tools.calculate_bounds import calculating_wvimg_bounds
from tools.tif_qaqc import unzip_and_remove, delete_non_tif, tif_qaqc, has_tif_files

def main():
    print("\n" + style.FOREST + "SatCHM... 🌳" + style.RESET + "\n") 
    PATHS = start.start()

    # Terminal stylings
    w, h = shutil.get_terminal_size()
    # Obtaining information (site, utm, dem_path, lidar_path, satellite_imagery_download_directory, angle_metadata and site_shapefile_path) from the .env file
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    # If .env variable has has wrong extension or is blank, then RuntimeError is raised
    site=os.getenv('site')
    site = site.strip()
    if site== '':
        raise RuntimeError(".env file must contain a value for \"site\"") 

    dem_path = os.getenv("dem_path")
    dem_path = dem_path.strip()
    if dem_path== '':
        raise RuntimeError(f".env file must contain a value for {dem_path}") 
    if not dem_path.lower().endswith((".tif", ".tiff")):
        raise RuntimeError(f"dem path must end with .tif: {dem_path}")
    PATHS["dem_raw_path"] = dem_path

    lidar_path = os.getenv("lidar_path")
    lidar_path = lidar_path.strip()
    if lidar_path== '':
        raise RuntimeError(f".env file must contain a value for {lidar_path}") 
    if not lidar_path.lower().endswith((".tif", ".tiff")):
        raise RuntimeError(f"lidar path must end with .tif: {lidar_path}")
    PATHS["lidar_raw_path"] =  lidar_path
    
    satellite_imagery_download_directory = os.getenv('satellite_download_dir')
    satellite_imagery_download_directory = satellite_imagery_download_directory.strip()
    if satellite_imagery_download_directory== '':
        raise RuntimeError(f".env file must contain a value for {satellite_download_dir}") 

    angle_metadata = os.getenv("angle_metadata")
    angle_metadata = angle_metadata.strip()
    if angle_metadata== '':
        raise RuntimeError(f".env file must contain a value for {angle_metadata}") 
    if not angle_metadata.lower().endswith((".csv")):
        raise RuntimeError(f"angle_metadata path must end with .csv: {angle_metadata}")
    PATHS["angle_metadata"] = angle_metadata
    
    print("-" * w)
    print(style.BOLD + "\n-----SHAPEFILE SELECTION-----\n" + style.RESET)

    # If site_shapefile_path is blank, then a RuntimeError is raised 
    site_shapefile_path=os.getenv('site_shapefile_dir')
    if site_shapefile_path == '':
        raise RuntimeError(f".env file must contain a value for {site_shapefile_path}") 
    
    # Searching for .shp file in the site_shapefile_path provided in .env
    shapefile_name = None
    for filename in os.listdir(site_shapefile_path):
        if filename.endswith('.shp'):
            shapefile_name = filename
            shapefile_path = os.path.join(site_shapefile_path, shapefile_name)
            print(f"Using shapefile \"{filename}\" in {site_shapefile_path}\n")
    if shapefile_name is None:
        raise RuntimeError(f"No shapefile (.shp extension) found in {site_shapefile_path}")

    # This script (reprojectPolygon.py) reprojects the shp file into UTM and saves the metadata about the site's bounding box into a csv                  
    output_shp_path, output_csv_path = reprojpoly.reproject_shapefile(shapefile_path)
    PATHS["site-metadata"] = output_csv_path
    PATHS["projected-shapefile"] = output_shp_path

    ######################################### SATELLITE DATA ########################################
    # Path creation for satellite data outputs
    PATHS["satellite_copied_directory"] = os.path.join(PATHS["new_project"], 'satellite-data')
    PATHS["inputs_wvimg"] = os.path.join(PATHS["inputs"], "wvimg")
    print("-" * w)
    print(style.BOLD + "\n-----PROCESSING INPUT DATA-----\n" + style.RESET)
    # Creating a new satellite data directory in ms-data
    PATHS["satellite_directory"] = os.path.join(PATHS["new_project"], 'satellite-data')
    os.makedirs(PATHS["satellite_directory"], exist_ok=True)
    
    if os.path.isdir(os.path.join(PATHS["inputs"], "wvimg")) and has_tif_files(os.path.join(PATHS["inputs"], "wvimg")):
        print("Satellite data detected in inputs folder... ✅ \n")
    else:
        
        print(style.FOREST + "Processing the satellite imagery..." + style.RESET + "\n")
        # Creating a new satellite data directory in ms-data
        PATHS["satellite_directory"] = os.path.join(PATHS["new_project"], 'satellite-data')
        os.makedirs(PATHS["satellite_directory"], exist_ok=True)

        # If angle_metadata has INCORRECT header, has no header or it is blank, then a RuntimeError is raised    
        satellite_metadata = pd.read_csv(angle_metadata)
        satellite_metadata_no_header = pd.read_csv((angle_metadata), header=0)
        if satellite_metadata_no_header.shape[0] == 0: 
            raise RuntimeError(f"Metadata is not complete. Please see that all fields are complete. Rerun program afterwards.") 
        required_columns = ["site", "date", "id", "sensor", "targetazimuth", "offnadir", "solarazimuth", "solarelevation"]
        satellite_metadata_header = satellite_metadata.columns.tolist()
        if not all(item in satellite_metadata_header for item in required_columns):
            raise RuntimeError(f"Metadata is not complete. Please see that all fields are complete. Rerun program afterwards.")
        if satellite_metadata.isnull().values.any(): 
                raise RuntimeError(f"Metadata is not complete. Please see that all fields are complete. Rerun program afterwards.") 
 
        ## Checking if GeoTIFF data is correctly placed in the satellite_imagery_download_directory
        print("\nMaking directories based on metadata...\n")
        manageDirectories.setup_dirs(raw_directory= PATHS["satellite_directory"], csv_file=angle_metadata, site=site)
        folders_in_satellite = [f for f in os.listdir(satellite_imagery_download_directory) if f != ".DS_Store"]
        print(style.BOLD + "\n-----PROCESSING BEGINS (THIS TAKES A WHILE)----\n" + style.RESET)

        # Checks if GeoTIFF files exist in the folders
        tif_found = False
        for root, _, files in os.walk(satellite_imagery_download_directory):
            if any(file.lower().endswith(('.tif', '.tiff')) for file in files):
                tif_found = True
                break # closes loop
        if not tif_found:
            raise RuntimeError(f"GeoTIFF files are not found in the {satellite_imagery_download_directory}. Please unpack your data here and rerun the program.")
            
        ## Checking if GeoTIFF data is correctly placed in the satellite_imagery_download_directory
        print("\nMaking directories based on metadata...\n")
        manageDirectories.setup_dirs(raw_directory= PATHS["satellite_directory"], csv_file=angle_metadata, site=site)
        folders_in_nonDG = [f for f in os.listdir(satellite_imagery_download_directory) if f != ".DS_Store"]

        print(style.BOLD + "\n-----PROCESSING BEGINS (THIS TAKES A WHILE)----\n" + style.RESET)
        for folder in folders_in_nonDG: # folder (e.g. caldor_2012-03-19_wv02_05090939090)
            parts = folder.split("_") 
            main_folder = parts[0] # main_folder = caldor

            src_path = os.path.join(satellite_imagery_download_directory, folder) # source path from satellite_imagery_download_directory
            dst_main_folder = os.path.join(PATHS["satellite_directory"], main_folder) 
            # main folder is being created
            os.makedirs(dst_main_folder, exist_ok=True)
    
            # If .zip files and GeoTIFF files were found in those unzipped folders, then proceed to process the data 
            if tif_found==True and zip_found==True:
                    if os.path.isdir(src_path) and not os.path.exists(dst_subfolder):
                        shutil.copytree(src_path, dst_subfolder)

            # If .zip files weren't found but GeoTIFF files, then proceed to process the data 
            if tif_found==True and zip_found==False:
                for subfolder in os.listdir(src_path): # go over each subfolder inside
                    src_subfolder = os.path.join(src_path, subfolder)
                    dst_subfolder = os.path.join(dst_main_folder, subfolder)
                    if os.path.isdir(src_subfolder) and not os.path.exists(dst_subfolder):
                        shutil.copytree(src_subfolder, dst_subfolder)
            else:
                if os.path.isdir(src_path) and not os.path.exists(dst_subfolder):
                        shutil.copytree(src_path, dst_subfolder)

        # Tiling in 2020 x 2020
        print(style.DARKCYAN + "\nNow tiling 2020 x 2020..." + style.RESET)
        projected_data_directory = projectSatelliteImagery.projectSatelliteImagery(raw_directory=PATHS["satellite_copied_directory"], angle_metadata=PATHS["angle_metadata"], site_metadata=PATHS["site-metadata"])
        PATHS["satellite_projected_directory"] = projected_data_directory 

        # Tiling in 512 x 512
        prepareCRSForestData.PrepareForestData(input_directory=PATHS["projected_satellite_data"], output_directory=PATHS["inputs_wvimg"], path_to_shapefile=PATHS["projected-shapefile"])

    ###################################### DEM DATA ##########################################
    # Path creation for DEM data outputs
    PATHS["dem_data"] = os.path.join(PATHS["new_project"], "dem-data")
    os.makedirs(PATHS["dem_data"], exist_ok=True)
    PATHS["projected_dem_data"] = os.path.join(PATHS["new_project"], "dem-data", "projected-data")
    os.makedirs(PATHS["projected_dem_data"], exist_ok=True)
    PATHS["inputs_dem"] = os.path.join(PATHS["inputs"], "dem")
    print("-" * w)

     # If DEM inputs are detected, it will proceed to lidar data, otherwise it will proceed to processing dem data
    if os.path.isdir(PATHS["inputs_dem"]) and has_tif_files(PATHS["inputs_dem"]):
        print("DEM data detected in inputs folder... ✅")
    else:
        print(style.BOLD + "\n-----PROCESSING DEM DATA-----\n" + style.RESET)
        os.makedirs(PATHS["inputs_dem"], exist_ok=True)

        # Reprojecting DEM data
        print("\nReprojecting and cropping dem data...\n") 
        output_dem_tif_reprojected = reprojectCrop.reprojectTif(input_file=PATHS["dem_raw_path"], output_dir=PATHS["dem_data"], site_shapefile_path=PATHS["projected-shapefile"], resolution=30)
        PATHS["dem_reprojected_tif"] = output_dem_tif_reprojected

        # Tiling 2020 x 2020
        print(style.DARKCYAN + "\nNow tiling 2020 x 2020..." + style.RESET)
        processLidarDEM.processAuxTifs(raw_directory=PATHS["satellite_copied_directory"], metadata_path=PATHS["site-metadata"], input_tif=PATHS["dem_reprojected_tif"], output_directory=PATHS["projected_dem_data"])
       
        # Tiling 512 x 512
        print(style.CYAN + "\n\nNOW MAKING THE DEM INPUTS FOR THE NEURAL NETWORK..." + style.RESET)
        prepareCRSForestData.PrepareForestData(input_directory=PATHS["projected_dem_data"], output_directory=PATHS["inputs_dem"], path_to_shapefile=PATHS["projected-shapefile"]) 

    ################################## LIDAR (CHM) DATA #####################################
    # Path creation for lidar data outputs
    PATHS["projected_lidar_data"] = os.path.join(PATHS["new_project"], "lidar-data", "projected-data")
    os.makedirs(PATHS["projected_lidar_data"], exist_ok=True)
    PATHS["chm_data"] = os.path.join(PATHS["new_project"], "chm-data")
    os.makedirs(PATHS["chm_data"], exist_ok=True)
    PATHS["inputs_chm"] = os.path.join(PATHS["inputs"], "chm")
    print("-" * w)

    # If CHM inputs are detected, it will proceed to solar/sensor data, otherwise it will proceed to processing lidar data
    if os.path.isdir(PATHS["inputs_chm"]) and has_tif_files(PATHS["inputs_chm"]):
        print("Lidar (chm) data detected in inputs folder... ✅")
    else:
        print(style.BOLD + "\n-----PROCESSING LIDAR DATA-----\n" + style.RESET)
        os.makedirs(PATHS["inputs_chm"], exist_ok=True)
       
        # Reprojecting CHM data
        print("\nReprojecting and cropping lidar data...") 
        output_lidar_tif_reprojected = reprojectCrop.reprojectTif(lidar_path, PATHS["chm_data"], output_shp_path, resolution = 0.5)
        PATHS["lidar_reprojected_tif"] = output_lidar_tif_reprojected 

        # Tiling 2020 x 2020
        print(style.DARKCYAN + "\nNow tiling 2020 x 2020...\n" + style.RESET)
        processLidarDEM.processAuxTifs(input_tif=PATHS["lidar_reprojected_tif"], raw_directory=PATHS["satellite_copied_directory"], metadata_path=PATHS["site-metadata"], output_directory=PATHS["projected_lidar_data"])
        
        # Tiling 512 x 512
        print(style.CYAN + "\n\nNOW MAKING THE LIDAR INPUTS FOR THE NEURAL NETWORK..." + style.RESET)
        prepareCRSForestData.PrepareForestData(input_directory=PATHS["projected_lidar_data"], output_directory=PATHS["inputs_chm"], path_to_shapefile=PATHS["projected-shapefile"])
        
    #################################### SENSOR & SOLAR DATA ##################################
    # Path creation for sensor and solar inputs
    PATHS["inputs_solar"] = os.path.join(PATHS["inputs"], "solar")
    PATHS["inputs_sensor"] = os.path.join(PATHS["inputs"], "sensor")
    print("-" * w)

    # If sensor/solar inputs are detected, it will proceed to DATA QAQC, otherwise it will create solar and sensor data
    if os.path.isdir(PATHS["inputs_solar"]) and has_tif_files(PATHS["inputs_solar"]) and os.path.isdir(PATHS["inputs_sensor"]) and has_tif_files(PATHS["inputs_sensor"]):
        print("Solar and sensor data detected in inputs folder... ✅")
    else:
        # This processes off-nadir angle, target azimuth, solar elevation, solar azimuth from the angle metadata
        print(style.BOLD + "\n-----PROCESSING SOLAR AND SENSOR DATA-----\n" + style.RESET)
        os.makedirs(PATHS["inputs_sensor"], exist_ok=True)
        os.makedirs(PATHS["inputs_solar"], exist_ok=True)
        solar_input=PATHS["inputs_solar"], sensor_input= PATHS["inputs_sensor"] = createAnglearrays.SensorSolarAngles(output_directory=PATHS["inputs"], path_to_csv=PATHS["angle_metadata"])
    
    #################################### DATA QAQC #############################################
    # Data QAQC happens every time regardless if it has happened before
    print("-" * w)
    print(style.BOLD + "\n-----QAQC-----\n" + style.RESET)
    print(style.FOREST + "Checking if all data inputs are satisfactory..." + style.RESET) 
    
    # Checks if lidar, dem, and satellite data, specifically contains alot of NAN values or 0s
    print("\nQAQC: Satellite Data...")
    tif_qaqc(PATHS["inputs_wvimg"])
    print("\nQAQC: DEM Data...")
    tif_qaqc(PATHS["inputs_dem"])
    print("\nQAQC: Lidar Data...")
    tif_qaqc(PATHS["inputs_chm"])
    
    # Removes files like ._{filename} or *.xml from all directories
    print("\nRemoving possible non-TIF file artifacts in all input folders...\n")
    delete_non_tif(PATHS["inputs_chm"])
    delete_non_tif(PATHS["inputs_solar"])
    delete_non_tif(PATHS["inputs_sensor"])
    delete_non_tif(PATHS["inputs_wvimg"])
    delete_non_tif(PATHS["inputs_dem"])

    #################################### CALCULATE BOUNDS #######################################
    print("-" * w)
    print(style.BOLD + "\n-----PROCESSING INPUT METADATA-----\n" + style.RESET)

    output_bounds_json = os.path.join(PATHS["inputs"], "bounds.json")
    PATHS["output_bounds_json"] = output_bounds_json

    # If bounds are already created, then this will be skipped
    if os.path.exists(output_bounds_json):
        print("Bounds are already saved...✅ \n")
    else:
        # Bounding boxes for each 512 x 512 wvimg tile are saved in the bounds.json file (to be applied to predictions later)
        print(style.FOREST + "Saving the bounds for satellite imagery...\n" + style.RESET)
        os.makedirs(PATHS["output_bounds_json"], exist_ok=True)
        calculating_wvimg_bounds(directory = PATHS["inputs_wvimg"], output_json=PATHS["output_bounds_json"])

    ##################################### MATCHING KEYS #####################################
    # If inputs are already renamed, then this will be skipped. 
    print("-" * w)
    PATHS["dem_json"] = os.path.join(PATHS["inputs"],"dem.json") 
    PATHS["chm_json"] = os.path.join(PATHS["inputs"],"chm.json")

    if os.path.exists(PATHS["dem_json"]) and os.path.exists(PATHS["chm_json"]):
        print("Inputs are already renamed... ✅ \n")
    else:
        # Match keys for dem & chm, makes sure files are the same
        print(style.FOREST + "Renaming and structuring inputs for ms-net...\n" + style.RESET)
        matchkeys.matchKeys(data_type ="dem")
        matchkeys.matchKeys(data_type = "chm")
 
    ##################################### TRAINING, TESTING, & VALIDATION ##################################
    print("-" * w)
    print(style.FOREST + "Rewriting the training, testing, and validation lists..." + style.RESET)
    train_val_list.train_val_test_split()
    ######################################################################################################
    # Neural network is initiated
    print("-" * w)
    print("-" * w)

    print(style.BOLD + "\n-----INITIATING THE NEURAL NETWORK-----\n" + style.RESET)
    
    print("\nTraining is in progress...")
    print("\n\n\nYou can monitor the validation loss and other metrics using Tensorboard.")
    print("\nOpen a new terminal window, cd into your repo, and run the following command according to documentation:")
    print("         tensorboard --logdir=lightning_logs")
    print("\nOnce you are satisfied with training, Press CTRL+C to quit.\n")
    time.sleep(7)
    train_main()

if __name__ == '__main__':
    main()
