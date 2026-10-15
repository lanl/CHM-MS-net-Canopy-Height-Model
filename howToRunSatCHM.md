# SatCHM Setup and Usage Guide

This document walks through one-time setup, data preparation, model training, and inference for running **SatCHM**.

---

## 1. Account Creation and Setup (One-Time)

Create accounts at the following websites:

- **https://opentopography.org**
  - After creating your account, request an **API key**.  
    You will need this later in step 3.
- **https://pro.gegd.com**
  - This is the source of the Vantor (formerly known as Maxar) imagery. GEGD Pro is only available to US government users 

---

## 2. Clone the Repository

Create a directory for your project (this will be your **project path**), then clone the repository into it (this will be your **repo path**).

### Example


`project_path=/mnt/c/Users/zach/Desktop/canopy`

`repo_path=/mnt/c/Users/zach/Desktop/canopy/SatCHM`


---

## 3. Create a `.env` File

In the root of the repository, create a file named `.env` and paste the following:

```
site=
epsg=
openTopoAPIkey=""
inferenceShpPath=""

# OPTIONAL ARGS
# customTrainShpPath=""
# customLidarTifPath=""
```

Fill in your values for site, epsg, and openTopoAPIkey.

### Notes

- The inferenceShpPath can refer to either a shp or geojson file
- By default, the model will draw a square around your inference area and select surrounding tiles to train on.
- For some edge cases (e.g., coastal areas), a custom training area may be helpful. You may hand draw a custom area to select training data from, and modify customTrainShpPath
- By default, lidar data is sourced from **USGS 3DEP**.
  - To use a different lidar source, provide a CHM raster `.tif` and modify and uncomment customLidarTifPath.

Fill in all fields **except** `inferenceShpPath`, which will be completed in Step 5.

### Example `.env` File

```
site=caldor
epsg=32610
openTopoAPIkey="abcdefg12345"
inferenceShpPath="/mnt/c/Users/zach/Desktop/canopy/SatCHM/downloads/${site}/infShp/caldor_infShp.geojson"

# OPTIONAL ARGS
# customTrainShpPath="/mnt/c/Users/zach/Desktop/canopy/SatCHM/downloads/${site}/trainShp/caldor_trainShp.geojson"
# customLidarTifPath="/mnt/c/Users/zach/Desktop/canopy/SatCHM/downloads/${site}/laz/caldor_laz/NEON_lidar-point-cloud-line/NEON/merged_CHM.tif"
```

---

## 4. Create an Inference Shapefile

Create or use an existing **shapefile or GeoJSON** defining the inference area.

- Save the file path to `inferenceShpPath` in your `.env` file

### Notes on Inference and Train Shape

You will receive an inference shape and train shape throughout this code. Below is an explanation of what these represent

#### Inference Shape

The inference shape represents the area for which you want to run SatCHM at. Although you provide your own inference shape (shown in red), a new one is generated (shown in green) that is reprojected to the correct CRS and includes a 512m buffer on all sides. Temporally align your imagery to the year you want to run inference, which in most cases is the latest available imagery.

![Inference Shape](figs/inferenceShape.png)

#### Train Shape

Train Shape:
By default, a square that contains 1024 tiles (shown in green) will be created around the center of your original inference area (shown in red). This represents the area for which you should collect data to train your model. Although this works for many cases, this may not be ideal in edge cases such as being near a shoreline, open prairie/desert, or disturbance affected area. In this case, you can provide a custom area for which you want to train at. If using a custom area (shown in blue), the code will reproject your area and automatically fit a rectangle within your training area that contains 1024 tiles. This new train area is the area for which you should try to download imagery for training aligned to the year specified at the end of main1. The original custom training area (shown in blue) can be very approximate and does not need to be precise.

![Train Shape](figs/trainShape.png)

---

## 5. Prepare Training Inputs

### 5.1 Build and Activate the Conda Environment

Navigate to:

`
project_path/SatCHM/setup
`

Then run:

```
conda env create -f SatCHMenv.yml
conda activate SatCHMenv
chmod 700 post_install.sh
./post_install.sh
```

This environment is primarily configured to run on linux. Anecdotally, env creation takes ~10-15 min and post_install.R takes ~30 min - 1 hr to run. Fortunately this step only needs to be done once, and not every time for every run of SatCHM

---

### 5.2 Run `main1.py`

From the `prepTrainInputs` directory:

```
python main1.py
```

Notes:
- Runtime: **1–2 hours**
- Lidar download is the main bottleneck
- Outputs will be written to:

`
project_path/{site}_data/
`

Note the difference between `project_path/{site}_data/` and `project_path/downloads/`. `project_path/{site}_data/` is meant to store the cleaned data after processing, ready to be fed into the neural network, while `project_path/downloads/` is meant to be a temporary storage location for raw data before processing. As the CHM data is processed while it is downloaded, it is saved directly to `project_path/{site}_data/` and does not pass through the downloads folder. Do not delete `project_path/downloads/` until the very end after Step 7, as it is needed for inference at the end of the workflow. 

---

### 5.3 Download Satellite Imagery from Vantor

You must download **cloud-free, snow-free Vantor imagery** with:

- Off-nadir angle **< 20°**
- Coverage spanning training and inference areas

For now, contact **rcrumley@lanl.gov** for assistance.

#### Imagery Alignment

- Training imagery → year reported by `main1.py`
- Inference imagery → year of interest

The year reported by the end of the output of `main1.py` is the mean of the years of collection of 3DEP lidar scans that covered your area.

#### Order Parameters

```
Delivery = Orders Manager
Additional Direct Delivery = No Additional Delivery
Production Parameters = ORTHO-READY (STANDARD) OR2A
Output Bands = Pan
Output File Format = GeoTIFF
Bits Per Pixel = 8
DRA = On
Compression = DEFLATE
Projection = UTM
Kernel = MTF
```

Leave the imagery as zip files, do not need to unzip them. The full zips are required for metadata processing, so manually extracting the tifs and providing them will not work.


Move imagery zips to:

- Training imagery:
`
project_path/downloads/wvimgTrain
`

- Inference imagery:
`
project_path/downloads/wvimgInf
`

---

### 5.4 Run `main2.py`

From `prepTrainInputs`:

```
python main2.py
```

Outputs will be available at:

`
project_path/{site}_data/
`

Note that you may see messages warning you that wvimg has no data tiles. This is ok and expected, as some tiles may generate near the edges of your satellite capture, and pixels outside of the edge will be nodata. These tiles are simply discarded. By default 1000 tiles are attempted to be collected, as 1000 is a genereous buffer for the amount of data required. The exact quantity of data required is not hard set, but if it is a few hundred below 1000, it is OK. Ensure that the value printed after "size of set intersection:" in the terminal is greater than 0, and hopefully at least above 500.

---

## 6. Train the Model

Navigate to `ms_net` and run:

```
python train.py
```

### GPU Notes

- GPU will be used after initial dataloading
- Dataloading may be slow on large datasets

Optional GPU test:

```
python testGPU.py
```

Model weights will be saved to:

`
project_path/SatCHM/ms_net/lightning_logs/version_0/checkpoints/
`

Weights are saved every **10 epochs**, and the model will train for **1000 epochs**.  
Subsequent runs will increment `version_1`, `version_2`, etc.  
If the model has trained to completion without issues, your weights should be saved as **epoch-epoch=999.ckpt**.

---

## 7. Run Inference

Navigate to `infer` and run:

```
python main.py
```

- By default, the most recent weights are used
- To use an alternative weights file that isn't the most recent one, uncomment line 75 and specify the path

Inference outputs will be available at:

`
{site}_INF_data
`

The merged predicted CHM raster is:

`
INF_chm_pred_merged
`

NOTE: you may see some edge effects in the merge between tiles. To soften these, increase the `FEATHER_CONST` value on line 39 of infer/main.py

If tree structure looks good but heights look off, consider adjusting the option parameter on line 202 from `mean` to `max`

If you only followed the conda environment build and activation in Step 5.1 without the optional cloud2trees installation, your code will fail on the treelist generation step. This is OK, and you may use INF_chm_pred_merged as your final output. If cloud2trees was installed, the CHM raster will be segmented (may take a while), and your final output will appear at `{site}_INF_data/{site}_treelist.csv`.
