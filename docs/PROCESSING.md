## Preprocessing
1. From the .env, there is the identification of the site polygon  the and the site polygon is reprojected to specified utm coordinate system. The site bounding box in utm are saved in the site-metadata.csv. A template angle_metadata.csv is created for you where you will write angle information (i.e. target azimuth, off-nadir angle, solar elevation, and solar azimuth) about the satellite imagery.
```
CHMer... 🌳

Project Directory: /Users/mia/Documents/Projects/caldor_run

Confirming shapefile(.shp) is in provided directory...

Using shapefile "caldor_wgs84.shp" in /Users/mia/Downloads/wgs84_caldor

Processing the satellite imagery...

Open this .CSV file for documenting angle metadata. Please keep this open as you proceed:

         C:/Users/mia/Documents/caldor_run/metadata/angle-metadata.csv

```
2. Created are WKT files, geojsons, and a **site-metadata.csv**.
```
├── metadata
│   ├── digitalglobe-downloads.csv
│   └── site-metadata.csv
└── site-polygons
    ├── geojson
    ├── with_crs
    └── wkt
```
3. If you are using DigitalGlobe, follow [this documentation](/docs/DIGITALGLOBE.md) and answer appropriately to the questions prompted. You will be able download your satellite imagery once you receive email confirmation that your data is ready. Download your data here at the **satellite_download_dir** specified in your .env file. If you are not using DigitalGlobe, your satellite imagery data will also be pulled from the **satellite_download_dir** but follow [this documentation](/docs/OWNIMAGERY.md) instead, regarding file structure.
 
### Tiling
4. Satellite imagery, the digital elevation models (DEMs), Lidar-produced canopy height models (CHMs) will first be tiled to 2020 x 2020 pixels (this includes a purposeful overlap of 20 pixels) and subsequently tiled into 512 x 512 pixel inputs for the neural network. All data is resampled to 0.5 m -- therefore each final input GeoTIFF spatially translates to 256 x 256 meters. Since there are multiple satellite GeoTIFFs, they are merged before tiling unlike the DEMs and CHMs. 
![Alt text](/pictures/partition.png)

- Site and date of acquitistion (YYYY-MM-DD) are included in the file name. Moreover, the naming convention of each file is based on the easting and northing (rounded to the nearest thousand) in the southwest corner of the first partition. Subsequently, the GeoTIFF is partitioned into sixteenths and named according to the diagram above. 

    <div style="text-align:center">
    site _ YYYY-MM-DD _ easting<sub>0</sub> _ northing<sub>0</sub>_ xx _ xx </div>
    
 ---

<div style="text-align:center"><h3> Satellite Imagery</h3></div>

![Alt text](/pictures/wvimg_caldor2.png)

Input satellite imagery should be cloudless and at a minimum resolution of 0.5-0.6 meters.

 ---

<div style="text-align:center"><h3> Digital Elevation Models</h3></div>

![Alt text](/pictures/dem_caldor.png)

Coarse digital elevation models (at most 30 meters resolution) will be upsampled to 0.5 meters.

 ---

<div style="text-align:center"><h3> LiDAR Produced Canopy Height Models</h3></div>

![Alt text](/pictures/chm_division.png)

LiDAR-produced canopy height models (at minimum 0.5-0.6 meters), used in backpropagation during training, will be upsampled to 0.5 meters.

 ---

<div style="text-align:center"><h3> Sensor and Solar Normals</h3></div>

![Alt text](/pictures/sensor_solar2.png)
Sensor and Solar normals are calculated from sensor and solar metadata of the satellite images, specifically target azimuth and off-nadir angles of the satellite and solar elevation and solar azimuth specific to the date and time of acquisition of the satellite imagery. 

```
# Calculating vector direction

z0 = 10 # arbitrary reference height (in meters)

x0 = np.sqrt(z0**2 / (((np.tan(np.radians(theta)))**2) + (((np.tan(np.radians(theta)))**2) / (np.tan(np.radians(phi)))**2)))

if phi > 180:
    x0 = x0 * -1

y0 = x0 / np.tan(np.radians(phi))

# Normalization
        s = np.sqrt(x0**2 + y0**2 + z0**2)
        xn = x0 / s
        yn = y0 / s
        zn = z0 / s

xg, yg = np.meshgrid(np.arange(-127.75, 128.25, 0.5), np.arange(-127.75, 128.25, 0.5))
yg = np.flipud(yg)
zg = -xn/zn * xg + -yn/zn * yg

normal = zg

```
### Calculating bounds
The bounds in utm of each 515 x 512 satellite image will be saved in **bounds.json**. They will be later applied to your predicted CHMs for georeferencing.

### Training, validation, and testing lists
Training, testing, and validation lists will be created with a Train-validation-test split (70 | 15 | 15).

If training, testing, and validation lists are already made, it will ask you would like to rewrite them, if necessary. 

```
Training, testing, and validation lists are already made... ✅

Would you like to make new training, testing, and validation lists? (Y/N):
```

### project_path
Before instantiating MS-NET, your project directory will look like this.
- Your **inputs** directory will contain .txt files for your training, testing, inference, and validation lists. The inference list contains all the of inputs that do not have corresponding validation data. Moreover, there will be two .json files that contain information about file naming structure. There will be one .json that contains information about the bounds to be applied to predictions later. If there are spatially duplicate satellite images with different dates of acquisition, other inputs will be duplicated and renamed to correspond to the specific satellite image. Lastly, in this directory, there will be the neural network input TIFFs: dem, chm, wvimg (satellite imagery), sensor, and solar.
- Your **metadata** directory will contain two .csv files. One that contains the bounding box(es) for your site. The other contains the angle metadata for your satellite imagery.
- Your **ms-data** directory contains sub-directories with your raw GeoTIFFs and your projected and tiled GeoTIFFs for the lidar-produced CHMs, the DEMs, and the satellite data.
- Your **site-polygons** directory will contain subdirectories with your reprojected polygon (with_crs), geojson of the site bounds in UTM, and Well Known Text (WKT) file of the sites bounds in longitude and latitude.

```
├── inputs
│   ├── {site}_inference.txt
│   ├── {site}_train.txt
│   ├── {site}_val.txt
│   ├── {site}_test.txt
│   ├── bounds.json
│   ├── chm
│   ├── chm.json
│   ├── dem
│   ├── dem.json
│   ├── sensor
│   ├── solar
│   └── wvimg
├── metadata
│   ├── angle-metadata.csv
│   └── site-metadata.csv
├── ms-data
│   ├── chm-data
│   ├── dem-data
│   └── satellite-data
└── site-polygons
    ├── geojson
    ├── with_crs
    └── wkt
```
# MS-NET: Neural Network
This ensemble of neural network was originally designed by Javier Santos to simulate flow through porous materials, but it is applicable for image processing applications involving
large 2D/3D arrays. Therefore, it has been modified to be applicable to predicting CHMS.



**Unique features:**
- Optimizes computational resources as it performs at different resolutions/scales 
- Coarser resolutions focus on high-level and global patterns, requiring more filters
- Finer resolutions focus on local patterns and low level details, requiring few filters

![Alt text](/pictures/msnet.png)


### Tracking and stopping training
If you have already created all your inputs, you can run main again and train as many times. CHMer automatically detects that the inputs have been created. It will only ask you if you want to recreate the the training, testing, and validation lists (seen above.)

```
Satellite data detected in inputs folder... ✅ 

DEM data detected in inputs folder... ✅ 

LiDAR (chm) data detected in inputs folder... ✅ 

Solar and sensor data detected in inputs folder... ✅ 

Checking if all CHM data inputs are complete (QAQC)...

Files removed with significant bad data: 0 files
Processing files: 100%|█████████████████████████████████████████████████████████████████████| 929/929 [00:09<00:00, 101.48files/s] 

Removing possible non-TIF file artifacts in all input folders...

Bounds are already saved...✅ 

Inputs are already renamed... ✅

Training, testing, and validation lists are already made... ✅
```

Afterwards, a new neural network will be initiated. When you prompt it (tensorboard --logdir=lightning_logs), TensorBoard will run as a local web server on your machine and opens an interface through your localhost (typically at http://localhost:6006 by default).You can monitor total (and specific scale) training and validation loss there.

```
Initiating the neural network...


Training is in progress...



You can monitor the validation loss and other metrics using Tensorboard.

Open a new terminal window, cd into your repo, and run the following command according to documentation:
         tensorboard --logdir=lightning_logs

Once you are satisfied with training, Press CTRL+C to quit
```
You can stop training by hitting CTRL + C. Your checkpoints and loss metrics will be saved in your lightning_logs directory as **.ckpt** files.

### Changing hyperparameters
If you would like to modify hyperparameters (i.e. learning rate and batch size), you can modify them in chm-er/ms_net_architecture/ms_parser.py under **TRAINING PARAMETERS**. The default parameters are listed below.

```
    # TRAINING PARAMETERS
    parser.add_argument("--train", default=True, type=str2bool)   
    # Whether to train the network
    parser.add_argument("--LR", default=1e-4, type=float)         
    # Learning rate for gradient descent
    parser.add_argument("--max_epochs", default=10000, type=int)  
    # Maximum number of epochs
    parser.add_argument("--min_epochs", default=10, type=int)     
    # Minimum number of epochs
    parser.add_argument("--steps", default=4, type=int)         
    # Number of steps 
    parser.add_argument("--batch_size", default=32, type=int)     
    # Number of samples per batch
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)  
    # For gradient accumulation
    parser.add_argument("--check_val_every_n_epoch", default=2, type=int)  
    # How often to run validation
    parser.add_argument("--gradient_clip_val", default=0.00, type=float)  
    # Clipping gradients to avoid large updates
```
Currently, Early Stopping is configured with a patience=9999. Therefore, there is effectively no early stopping. This can be modified in chm-er/ms_net_architecture/train.py.

### Infering
If you would like to evaluate your testing data test once you are satisified with training, open a new terminal and run evaluation.py. Enter the model checkpoint path.

```
Are you infering on your test data? (Y/N):Y
Please enter the path to your model checkpoint file that you are satisified with (must end with '.ckpt'): 
```
Your predicted CHMs will be saved in **{project_path}/outputs/predicted_chms**

You may also predict on inputs that do not have corresponding validation data ({site}_inference.py) by answering **N**. 

# Post Processing: Pixelwise comparison and Tree approximate object (TAO) comparison
After predicted CHMs are created, figures that demonstrate visualize the error will be created: Figures that demonstrate pixelwise comparison (target - predicted) and figures that contain MAE & RSME. They will be all be saved in your **{project_path}/outputs**. Here are examples similar to the figures:

#### Pixelwise comparison
![Alt text](/pictures/pixelwise_error.png)

#### TAO comparison
![Alt text](/pictures/taos.png)


