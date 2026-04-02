"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


"""
Main script for CHMer. Written by Mia Mitchell.
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
    # This prompts the start.py script (creates angle-metadata.csv, other directories)
    PATHS = start.start()

    # Obtaining information (site, utm, dem_path, lidar_path, satellite_imagery_download_directory and site_shapefile_path) from the .env file
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    site_shapefile_path=os.getenv('site_shapefile_dir')
    site=os.getenv('site')
    dem_path = os.getenv("dem_path")
    w, h = shutil.get_terminal_size()
    
    if not dem_path.endswith(".tif"):
        raise RuntimeError(f"dem path must end with .tif: {dem_path}")

    lidar_path = os.getenv("lidar_path")
    satellite_imagery_download_directory = os.getenv('satellite_download_dir')
    
    # If site_shapefile_path is blank, then a RuntimeError is raised
    print("-" * w)
    #print("\nConfirming shapefile(.shp) is in provided directory...\n")
    print(style.BOLD + "\n-----SHAPEFILE SELECTION-----\n" + style.RESET)
    if site_shapefile_path == '':
        raise RuntimeError(".env file must contain a value for \"site_shapefile_path\"") 
    
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
    # print(f"Reprojecting the polygon shapefile {shapefile_name} to {utm}...\n")                    
    output_shp_path, output_csv_path = reprojpoly.reproject_shapefile(shapefile_path)
    PATHS["site-metadata"] = output_csv_path
    PATHS["projected-shapefile"] = output_shp_path

    ######################################### SATELLITE DATA ########################################
    # If satellite (wvimg) inputs are detected, it will proceed to DEM data
    print("-" * w)
    print(style.BOLD + "\n-----PROCESSING INPUT DATA-----\n" + style.RESET)
    if os.path.isdir(os.path.join(PATHS["inputs"], "wvimg")) and has_tif_files(os.path.join(PATHS["inputs"], "wvimg")):
        print("Satellite data detected in inputs folder... ✅ \n")
    else:
        
        print(style.FOREST + "Processing the satellite imagery..." + style.RESET + "\n")
        # Creating a new satellite data directory in ms-data
        PATHS["satellite_directory"] = os.path.join(PATHS["new_project"], 'satellite-data')
        os.makedirs(PATHS["satellite_directory"], exist_ok=True)
        print(style.RED + style.UNDERLINE + "Open this .CSV file for documenting angle metadata. Please keep this open as you proceed:" + style.RESET + "\n\n\t", PATHS["dg_csv_path"] + "\n")
        
        # If satellite_imagery_download_directory in .env is blank, then a RuntimeError is raised
        if satellite_imagery_download_directory == '':
            raise RuntimeError(".env file must contain a value for satellite_imagery_download_directory") 
    
        # Prompting user if they are using DigitalGlobe 
        response = input("\nAre you using DigitalGlobe? (Y/N):\n")
        
        if response.upper() == "Y":
            # Directions for DigitalGlobe
            print("You selected DigitalGlobe.")
            print("1. Ensure that you have a DigitalGlobe account set up and are able to download data.")
            print("2. Use the WKT bounding box file to download imagery from DigitalGlobe for area of interest (AOI).")
            print("3. See documentation (docs/DIGITALGLOBE.md) for more specific instructions about naming conventions and downloading imagery.\n\n")
            print("Now generating the WKT (Well-Known Text Script) to input into DigitalGlobe...\n") 

            # This script bounding boxes of the shapefiles transforms them into lat/log WKTs.
            generateforDG.creating_geoinfo(output_shp_path, printt=True)
            
            response_DG = input("Has it been 24 hours and/or you received confirmation that your imagery order has been fufilled? (Y/N):")
            if response_DG.upper() == "Y":
                digitalglobe_metadata = pd.read_csv(PATHS["dg_csv_path"])
                digitalglobe_metadata_no_header = pd.read_csv(PATHS["dg_csv_path"], header=0)
                # If metadata is not complete, then it will raise a RuntimeError
                if digitalglobe_metadata_no_header.shape[0] == 0: 
                    raise RuntimeError(f"Metadata is not complete. Please see that all fields are complete. Rerun program afterwards.") 
                  
                # Directions for unpacking your data into your satellite_imagery_download_directory
                print("\nPlease see the documentation for more specific instructions using HTTPs download.\n")

                input("\nUnpack your data into this directory:" +  "\n\n\t" + f"{satellite_imagery_download_directory}" + "\n\nPress ENTER when complete:")
                # ZACH I want to make sure that it can unzip folders and detect GEOtiffs put them in the correct folder, this will have to change if you update to no CELL with DG"
                # Checking if GeoTIFF data is correctly placed in the satellite_imagery_download_directory
                zip_folders = []
                # First checks if files are .zip files
                zip_found = False
                try:
                    for root, _, files in os.walk(satellite_imagery_download_directory):
                            if any(file.lower().endswith('.zip') for file in files):
                                print(".zip files have been found.")
                                zip_found = True
                                break # closes loop
                            for file in files:
                                if file.endswith('.zip'):
                                    zip_folders.append(os.path.join(root, file))
                    for folder in zip_folders:
                        unzip_and_remove(folder)
                except:
                    print("Since .zip files not found, now looking for GeoTIFFS.")
                    pass

                # Second checks if GeoTIFF files exist in the folders
                tif_found = False
                for root, _, files in os.walk(satellite_imagery_download_directory):
                    if any(file.lower().endswith(('.tif', '.tiff')) for file in files):
                        tif_found = True
                        break # closes loop
                if not tif_found:
                    raise RuntimeError(f"GeoTIFF files are not found in the {satellite_imagery_download_directory}. Please unpack your data here and rerun the program.")
                
                # Checking if metadata table is filled out correctly
                digitalglobe_metadata = pd.read_csv(PATHS["dg_csv_path"])
                digitalglobe_metadata_no_header = pd.read_csv(PATHS["dg_csv_path"], header=0)
                if digitalglobe_metadata_no_header.shape[0] == 0: 
                    raise RuntimeError(f"Metadata is not complete. Please see that all fields are complete. Rerun program afterwards.") 
                
                # Now starting to process the non-DigitalGlobe satellite data
                print("Making directories based on metadata...\n")
                #manageDirectories.setup_dirs(raw_directory= PATHS["satellite_directory"], csv_file=PATHS["dg_csv_path"], site=site)  
                
                # List the folders
                folders_in_nonDG = [f for f in os.listdir(satellite_imagery_download_directory) if f != ".DS_Store"]
        
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
                    
                # Tiling in 2020 x 2020
                projected_data_directory = projectSatelliteImagery.projectSatelliteImagery(raw_directory=PATHS["satellite_directory"], angle_metadata=PATHS["dg_csv_path"], site_metadata=PATHS["site-metadata"])
                PATHS["projected_satellite_data"]=projected_data_directory

                print("Now, making the satellite inputs for the neural network...\n")
                PATHS["inputs_wvimg"] = os.path.join(PATHS["inputs"], "wvimg")
                os.makedirs(PATHS["inputs_wvimg"], exist_ok=True)
                # Tiling in 512 x 512
                prepareCRSForestData.PrepareForestData(input_directory=PATHS["projected_satellite_data"], output_directory=PATHS["inputs_wvimg"], path_to_shapefile=PATHS["projected-shapefile"])
    

            elif response_DG.upper() == "N":
                raise RuntimeError("Once you receive email confirmation that your imagery order is fufilled, please rerun the program and download your imagery from DigitalGlobe using HTTPs. See documentation (docs/DIGITALGLOBE.md) for more details.")  
            else:
                raise RuntimeError("Invalid response. Please enter 'Y' or 'N'.")  
        elif response.upper() == "N":
            # Directions for non-DigitalGlobe
            print("\nPlease see the documentation for more specific instructions using your own satellite data.\n")

            input("\nUnpack your data into this directory:" +  "\n\n\t" + satellite_imagery_download_directory + "\n\nPress ENTER when complete:")

            # Now starting to process the non-DigitalGlobe satellite data
            # Checking if GeoTIFF data is correctly placed in the satellite_imagery_download_directory
                 # ZACH I want to make sure that it can unzip folders and detect GEOtiffs put them in the correct folder, this will have to change if you update to no CELL with DG"
            zip_folders = []
            # First checks if files are .zip files
            zip_found = False
            try:
                for root, _, files in os.walk(satellite_imagery_download_directory):
                        if any(file.lower().endswith('.zip') for file in files):
                            print(".zip files have been found.")
                            zip_found = True
                            break # closes loop
                        for file in files:
                            if file.endswith('.zip'):
                                zip_folders.append(os.path.join(root, file))
                for folder in zip_folders:
                    unzip_and_remove(folder)
            except:
                print("Since .zip files not found, now looking for GeoTIFFS.")
                pass

            # Second checks if GeoTIFF files exist in the folders
            tif_found = False
            for root, _, files in os.walk(satellite_imagery_download_directory):
                if any(file.lower().endswith(('.tif', '.tiff')) for file in files):
                    tif_found = True
                    break # closes loop
            if not tif_found:
                raise RuntimeError(f"GeoTIFF files are not found in the {satellite_imagery_download_directory}. Please unpack your data here and rerun the program.")
            
            ## Checking if GeoTIFF data is correctly placed in the satellite_imagery_download_directory
            print("\nMaking directories based on metadata...\n")
            manageDirectories.setup_dirs(raw_directory= PATHS["satellite_directory"], csv_file=PATHS["dg_csv_path"], site=site)
            folders_in_nonDG = [f for f in os.listdir(satellite_imagery_download_directory) if f != ".DS_Store"]
    
            for folder in folders_in_nonDG: # folder (e.g. caldor_2012-03-19_wv02_05090939090)
                parts = folder.split("_") 
                main_folder = parts[0] # main_folder = caldor

                src_path = os.path.join(satellite_imagery_download_directory, folder) # source path from satellite_imagery_download_directory
                dst_main_folder = os.path.join(PATHS["satellite_directory"], main_folder) 
                # main folder is being created
                os.makedirs(dst_main_folder, exist_ok=True)
                # ZACH I want to make sure that it can unzip folders and detect GEOtiffs put them in the correct folder, this will have to change if you update to no CELL with DG"
                # If .zip files and GeoTIFF files were found in those unzipped folders, then proceed to process the data
                if tif_found==True and zip_found==True:
                        if os.path.isdir(src_path) and not os.path.exists(dst_subfolder):
                            shutil.copytree(src_path, dst_subfolder)

                # If .zip files weren't found but GeoTIFF files, then proceed to process the data TODO
                if tif_found==True and zip_found==False:
                    for subfolder in os.listdir(src_path): # go over each subfolder inside
                        src_subfolder = os.path.join(src_path, subfolder)
                        dst_subfolder = os.path.join(dst_main_folder, subfolder)
                        if os.path.isdir(src_subfolder) and not os.path.exists(dst_subfolder):
                            shutil.copytree(src_subfolder, dst_subfolder)
                else:
                    if os.path.isdir(src_path) and not os.path.exists(dst_subfolder):
                            shutil.copytree(src_path, dst_subfolder)


            # Checking if metadata table is filled out correctly
            digitalglobe_metadata = pd.read_csv(PATHS["dg_csv_path"])
            digitalglobe_metadata = digitalglobe_metadata.drop(digitalglobe_metadata.index)
            #digitalglobe_metadata = digitalglobe_metadata.drop(['id', 'sensor'], axis=1)
            if digitalglobe_metadata.isnull().values.any(): 
                raise RuntimeError(f"Metadata is not complete. Please see that all fields are complete. Rerun program afterwards.") 

            # Tiling in 2020 x 2020       
            projected_data_directory = projectSatelliteImagery.projectSatelliteImagery(raw_directory=PATHS["satellite_directory"], angle_metadata=PATHS["dg_csv_path"], site_metadata=PATHS["site-metadata"])
            print("\nNow, making the satellite inputs for the neural network...\n")
            PATHS["inputs_wvimg"] = os.path.join(PATHS["inputs"], "wvimg")
            os.makedirs(PATHS["inputs_wvimg"], exist_ok=True)
            # Tiling in 512 x 512
            prepareCRSForestData.PrepareForestData(input_directory=PATHS["projected_satellite_data"], output_directory=PATHS["inputs_wvimg"], path_to_shapefile=PATHS["projected-shapefile"])
        else:
            raise RuntimeError("Invalid response. Please enter 'Y' or 'N'.")
    ###################################### DEM DATA ##########################################
    # If DEM inputs are detected, it will proceed to CHM data
    if os.path.isdir(os.path.join(PATHS["inputs"], "dem")) and has_tif_files(os.path.join(PATHS["inputs"], "dem")):
        print("DEM data detected in inputs folder... ✅ \n")
    else:
        print("\n" + style.FOREST + "Processing the dem data..." + style.RESET + "\n") 
        # This script bounding boxes of the shapefiles (if bounding boxes need to be divided into smaller cells, not print statements though)
        generateforDG.creating_geoinfo(output_shp_path, printt=False)
        
        # If dem_path in .env is blank, then a RuntimeError is raised
        if dem_path == '':
            raise RuntimeError(".env file must contain a value for dem_path") 
        else:
            # Reprojecting DEM data
            PATHS["dem_data"] = os.path.join(PATHS["new_project"], "dem-data")
            os.makedirs(PATHS["dem_data"], exist_ok=True)
            print("\nReprojecting and cropping DEM data...\n") 
            output_file_reprojected = reprojectCrop.reprojectTif(dem_path, PATHS["dem_data"], output_shp_path, resolution=30)
            PATHS["projected_dem_data"] = os.path.join(PATHS["new_project"], "dem-data", "projected-data")
            # Making a new directory for the projected DEM data
            os.makedirs(PATHS["projected_dem_data"], exist_ok=True)

            # Tiling 2020 x 2020
            print(style.DARKCYAN + "\nNow tiling...\n" + style.RESET)
            processLidarDEM.processAuxTifs(input_tif=output_file_reprojected, output_directory=PATHS["projected_dem_data"])
            print(style.DARKCYAN + "Now, making the DEM inputs for the neural network..." + style.RESET)
            PATHS["inputs_dem"] = os.path.join(PATHS["inputs"], "dem")
            os.makedirs(PATHS["inputs_dem"], exist_ok=True)
            # Tiling 512 x 512
            prepareCRSForestData.PrepareForestData(input_directory=PATHS["projected_dem_data"], output_directory=PATHS["inputs_dem"], path_to_shapefile=output_shp_path) 

    ################################## LIDAR (CHM) DATA #####################################
    # If CHM inputs are detected, it will proceed to solar/sensor data
    if os.path.isdir(os.path.join(PATHS["inputs"], "chm")) and has_tif_files(os.path.join(PATHS["inputs"], "chm")):
        print("Lidar (chm) data detected in inputs folder... ✅ \n")
    else:
        print(style.FOREST + "Processing the lidar data...." + style.RESET)
        # This script bounding boxes of the shapefiles (if bounding boxes need to be divided into smaller cells, not print statements though)
        generateforDG.creating_geoinfo(output_shp_path, printt=False)

        # If lidar_path in .env is blank, then a RuntimeError is raised
        if lidar_path == '':
            raise RuntimeError(".env file must contain a value for \"lidar_path\"") 
        else:
            PATHS["chm_data"] = os.path.join(PATHS["new_project"], "chm-data")
            os.makedirs(PATHS["chm_data"], exist_ok=True)

            # Reprojecting CHM data
            print("\nReprojecting and cropping lidar data...\n") 
            PATHS["chm_data"] = os.path.join(PATHS["new_project"], "chm-data")
            os.makedirs(PATHS["chm_data"], exist_ok=True)
            output_file_reprojected = reprojectCrop.reprojectTif(lidar_path, PATHS["chm_data"], output_shp_path, resolution = 0.5)
            PATHS["projected_lidar_data"] = os.path.join(PATHS["new_project"], "lidar-data", "projected-data")
            os.makedirs(PATHS["projected_lidar_data"], exist_ok=True)

            # Tiling 2020 x 2020
            print(style.DARKCYAN + "\nNow tiling...\n" + style.RESET)
            processLidarDEM.processAuxTifs(input_tif=output_file_reprojected, output_directory=PATHS["projected_lidar_data"])
            print(style.DARKCYAN + "\n\nNow, making the LiDAR inputs for the neural network...\n\n" + style.RESET)
            PATHS["inputs_chm"] = os.path.join(PATHS["inputs"], "chm")
            os.makedirs(PATHS["inputs_chm"], exist_ok=True)
            # Tiling 512 x 512
            prepareCRSForestData.PrepareForestData(input_directory=PATHS["projected_lidar_data"], output_directory=PATHS["inputs_chm"], path_to_shapefile=output_shp_path)
        
    #################################### SENSOR & SOLAR DATA ##################################
    # If sensor/solar inputs are detected, it will proceed to DATA QAQC
    if os.path.isdir(os.path.join(PATHS["inputs"], "solar")) and has_tif_files(os.path.join(PATHS["inputs"], "solar")) and os.path.isdir(os.path.join(PATHS["inputs"], "sensor")) and has_tif_files(os.path.join(PATHS["inputs"], "sensor")) :
        print("Solar and sensor data detected in inputs folder... ✅ \n")
    else:
        # This processes off-nadir angle, target azimuth, solar elevation, solar azimuth from the angle metadata
        print("\n" + style.FOREST + "Processing solar and sensor data...." + style.RESET + "\n")
        solar_input, sensor_input = createAnglearrays.SensorSolarAngles(output_directory=PATHS["inputs"], path_to_csv=PATHS["dg_csv_path"])
        PATHS["inputs_solar"] = solar_input
        PATHS["input_sensor"] = sensor_input
    #################################### DATA QAQC #############################################
    # Data QAQC happens every time regardless if it has happened before
    print("-" * w)
    print(style.BOLD + "\n-----QAQC-----\n" + style.RESET)
    print(style.FOREST + "Checking if all data inputs are satisfactory..." + style.RESET) 
    
    # Checks if CHM data, specifically contains alot of NAN values or 0s
    tif_qaqc(os.path.join(PATHS["inputs"], "chm"))

    # Removes files like ._{filename} or *.xml from all directories
    print("\nRemoving possible non-TIF file artifacts in all input folders...\n")
    delete_non_tif(os.path.join(PATHS["inputs"], "chm"))
    delete_non_tif(os.path.join(PATHS["inputs"], "solar"))
    delete_non_tif(os.path.join(PATHS["inputs"], "sensor"))
    delete_non_tif(os.path.join(PATHS["inputs"], "wvimg"))
    delete_non_tif(os.path.join(PATHS["inputs"], "dem"))

    #################################### CALCULATE BOUNDS #######################################
    # If bounds are already created, then this will be skipped
    print("-" * w)
    print(style.BOLD + "\n-----PROCESSING INPUT METADATA-----\n" + style.RESET)
    output_bounds_json = os.path.join(PATHS["inputs"], "bounds.json")
    if os.path.exists(output_bounds_json):
        print("Bounds are already saved...✅ \n")
    else:
        # Bounding boxes for each 512 x 512 wvimg tile are saved in the bounds.json file (to be applied to predictions later)
        print(style.FOREST + "Saving the bounds for satellite imagery...\n" + style.RESET)
        output_bounds_json = os.path.join(PATHS["inputs"], "bounds.json")
        calculating_wvimg_bounds(directory = os.path.join(PATHS["inputs"], "wvimg"), output_json=output_bounds_json)

    ##################################### MATCHING KEYS #####################################
    # If inputs are already renamed, then this will be skipped. 
    if os.path.exists(os.path.join(PATHS["inputs"],"dem.json")) and os.path.exists(os.path.join(PATHS["inputs"],"chm.json")):
        print("Inputs are already renamed... ✅ \n")
    else:
        # Match keys for dem & chm, makes sure files are the same
        print(style.FOREST + "Renaming and structuring inputs for ms-net...\n" + style.RESET)
        matchkeys.matchKeys('dem')
        matchkeys.matchKeys('chm')
 
    ##################################### TRAINING, TESTING, & VALIDATION ##################################
    # If lists are already named, then this will be skipped. 
    if os.path.exists(os.path.join(PATHS["inputs"], f"{site}_train.txt")) and os.path.exists(os.path.join(PATHS["inputs"], f"{site}_val.txt")) and os.path.exists(os.path.join(PATHS["inputs"], f"{site}_test.txt")):
        print("Training, testing, and validation lists are already made... ✅ \n")
        print("-" * w)
        response_list = input("\nWould you like to make new training, testing, and validation lists? (Y/N):")
        # You may rewrite lists
        if response_list.upper() == "Y":
            print(style.FOREST + "Rewriting the training, testing, and validation lists..." + style.RESET)
            train_val_list.train_val_test_split()
        # You may continue with existing ones
        elif response_list.upper() == "N":
            print("Continuing with existing training, testing, and validation lists...")
        else:
            raise RuntimeError("Invalid response. Please enter 'Y' or 'N'.")
    else:
        print(style.FOREST + "Writing the training, testing, and validation lists..." + style.RESET)
        train_val_list.train_val_test_split()

    ######################################################################################################
    # Neural network is initiated
    print("-" * w)
    print(style.PURPLE + "\nInitiating the neural network...\n" + style.RESET)
    
    print("\nTraining is in progress...")
    print("\n\n\nYou can monitor the validation loss and other metrics using Tensorboard.")
    print("\nOpen a new terminal window, cd into your repo, and run the following command according to documentation:")
    print("         tensorboard --logdir=lightning_logs")
    print("\nOnce you are satisfied with training, Press CTRL+C to quit.\n")
    time.sleep(7)
    train_main()

if __name__ == '__main__':
    main()
