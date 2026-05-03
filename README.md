# SatCHM (Satellite Canopy Height Model)

# Motivation

These are the required inputs to use **SatCHM**:

![Alt text](pictures/inputs.png)

<br>
<div style="text-align:center">
  <table style="margin: 0 auto; border-collapse: collapse;" border="1">
    <tr>
      <th>Inputs</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>Satellite Imagery</td>
      <td>Input satellite imagery must be cloudless, panchromatic GeoTIFFs that have a resolution of 0.5 - 0.6 meters with discernable crowns; target azimuth, off-nadir angle, solar azimuth, and solar elevation metadata must be able to documented for each image. <strong>The off-nadir angle for each angle must be < 20°.</strong>
 </td>
    </tr>
	<tr>
      <td>Digital Elevation Models (DEMs)</td>
      <td>Input DEMs of the target area should at least be 30 meters resolution</td>
	  <tr>
      <td>Solar and Sensor Angles</td>
      <td>Solar and sensor inputs are created with SatCHM using the target azimuth, off-nadir angle, solar azimuth, and solar elevation metadata from the satellite imagery. This information is entered in the angle metadata. </td>
    </tr>
	<tr>
      <td>Lidar-produced CHMs </td>
      <td> Input lidar-produced CHMs should be at 0.5 - 0.6 meters or finer resolutions </td>
    </tr>
  </table>
</div>

</br>

## Dependencies

### Python version
This project is developed and tested with **Python 3.12**. It may work with other versions, but compatibility is not guaranteed.

### Creating your environment
```
pip install -r satchmenv.txt
```
### Install Pytorch
```
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
This specifies CUDA 11.8 (as indicated by cu118). However, you must adjust this version based on:

- The CUDA version installed on your system

- You are using pip or conda

- Your operating system and Python version

Download from [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/). **There, you can select your preferences (OS, package manager, Python version, and CUDA version), and it will generate the exact install command you need.**


## Getting Started

### Tutorial Data

 Throughout the documentation, we will be using the project 'caldor' as an example. Caldor references the Caldor Fire of 2021, south of Lake Tahoe, California. All satellite images were taken before this fire occured. The worldview satellite images were downloaded from Vantor (formerly Maxar Intelligence). Airborne lidar data of south Lake Tahoe was collected in 2010 and was downloaded from [NASA Earthdata](https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-cms-lidar-agb-california-1537-1#toc-product-summary). 

<div align="center">
<img src="pictures/caldor_burn_boundary.png" width="1560" height="1390">
</div>

#### To download tutorial data, following these instructions:

(1) Download [Pelican]("https://docs.pelicanplatform.org/install"). Pelican is an open-source data repository platform.

(2) Run the Pelican download command in terminal. Navigate to your preferred destitation. 

```
(satchmenv) mia /mnt/c/Users/miashell/Documents >  pelican object sync pelican://osg-htc.org/ndp/public/LANLCHM/ ./LANLCHM/
```

### Creating .env file
Create a .env file where the repo is located. Edit the file in a text editor.


|      Rows     | Description |
| ------------- | ------------- |
| project_dir  | the project directory for the products of preprocessing, the neural network & post-processing  |
| site  | the name of the site you are processing |
| utm  | the Universal Transversal Mercator EPSG code (i.e. EPSG:32610)   |
| dem_path | the path to the DEM GeoTIFF of the site (ends in .tif or .tiff) |
| lidar_path | the path to the lidar-produced CHM GeoTIFF within the site (ends in .tif or .tiff)  |
| site_shapefile_dir | the path to the directory to the shapefile of the site   |
| satellite_download_dir| the path to the directory that holds the satellite GeoTIFFs   |
| angle_metadata| the path to the .csv that holds solar and sensor angle metadata   |

```
UW PICO 5.09                                    File: .env                                       

project_path=/mnt/c/Users/mia/Documents/caldor_run
site=caldor
utm=EPSG:32610
dem_path=/mnt/c/Users/mia/Documents/LANLCHM/caldor_dem.tif
lidar_path=/mnt/c/Users/mia/Documents/LANLCHM/caldorchm.tif
site_shapefile_path=/mnt/c/Users/mia/Documents/LANLCHM/wgs84_caldor
satellite_download_dir=/mnt/c/Users/miashell/Documents/LANLCHM/satellite_data
angle_metadata=/mnt/c/Users/miashell/Documents/LANLCHM/angle-metadata.csv

```
Document the appropriate paths and directories in the .env before running the program. A blank template for angle metadata csv is included in this repo. If using the tutorial data, the angle metadata is located in the data repository.

If you are using your own imagery, [see additional details here](./docs/OWNIMAGERY.md) and place your imagery in the **satellite_download_dir** specified in your .env. 


## Running Main


After creating the .env, run main.
```console
(satchmenv) mia /mnt/c/Users/miashell/Documents/CHM-MS-net-Canopy-Height-Model > python main.py


SatCHM... 🌳

------------------------------------------------------------------------------------------------------------------------------------------------

Project Directory: /mnt/c/Users/miashell/Documents/caldor_run

------------------------------------------------------------------------------------------------------------------------------------------------

-----SHAPEFILE SELECTION-----

Using shapefile "caldor_wgs84.shp" in /mnt/c/Users/miashell/Documents/LANLCHM/wgs84_caldor

------------------------------------------------------------------------------------------------------------------------------------------------

-----PREPROCESSING BEGINS (THIS TAKES A WHILE)-----

------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------

-----PROCESSING SATELLITE DATA-----

```
### Making Directories

```
Making directories based on metadata...
```



### Tiling satellite imagery, DEMs, and lidar data
Satellite imagery, the digital elevation models (DEMs), Lidar-produced canopy height models (CHMs) will first be tiled to 2020 x 2020 pixels  and subsequently tiled into 512 x 512 pixel inputs for the neural network. All data is resampled to 0.5 m -- therefore each final input GeoTIFF spatially translates to 256 x 256 meters. Since there are multiple satellite GeoTIFFs, they are merged before tiling unlike the DEMs and CHMs. 

<div align="center">
<img src="pictures/partition.png">
</div>

Site and date of acquitistion (YYYY-MM-DD) are included in the file name. Moreover, the naming convention of each file is based on the easting and northing (rounded to the nearest thousand) in the southwest corner of the first partition. Subsequently, the GeoTIFF is partitioned into sixteenths and named according to the diagram above. 

  <div style="text-align:center">
    site _ YYYY-MM-DD _ easting<sub>0</sub> _ northing<sub>0</sub>_ xx _ xx </div>
  </div>

 ---
<br>

```
Now tiling 2020 x 2020...
                                                                                                                                               

Working on chunk: caldor

Chunk subfolders: ['caldor.2_2015-08-05', 'caldor.2_2017-06-14', 'caldor_2011-09-05', 'caldor_2015-08-05', 'caldor_2016-06-24', 'caldor_2016-08-24', 'caldor_2017-06-14']

Merging caldor.2_2015-08-05.                                                                                                                    
                                                                                                                                                
Working on images from caldor.2_2015-08-05...                                                                                                   

Merging caldor.2_2017-06-14.                                                                                                                    
Processing easting=765966, northing=4308659: 100%|██████████████████████████████████████████████████████| 2457/2457 [1:28:44<00:00,  2.17s/tile]
                                                                                                                                                
Working on images from caldor.2_2017-06-14...                                                                        | 0/2457 [00:00<?, ?tile/s]
                                                                                                                                                
Merging caldor_2011-09-05.                                                                                                                      
Processing easting=765966, northing=4308659: 100%|██████████████████████████████████████████████████████| 2457/2457 [1:57:11<00:00,  2.86s/tile]
                                                                                                                                                
Working on images from caldor_2011-09-05...                                                                                                     

Merging caldor_2015-08-05.                                                                                                                      
Processing easting=765966, northing=4308659: 100%|██████████████████████████████████████████████████████| 2457/2457 [1:57:41<00:00,  2.87s/tile]
                                                                                                                                                
Working on images from caldor_2015-08-05...                                                                          | 0/2457 [00:00<?, ?tile/s]
                                                                                                                                                
Merging caldor_2016-06-24.                                                                                                                      
Processing easting=765966, northing=4308659: 100%|██████████████████████████████████████████████████████| 2457/2457 [2:00:40<00:00,  2.95s/tile]
                                                                                                                                                
Working on images from caldor_2016-06-24...                                                                                                     

Merging caldor_2016-08-24.                                                                                                                      
Processing easting=765966, northing=4308659: 100%|██████████████████████████████████████████████████████| 2457/2457 [2:27:15<00:00,  3.60s/tile]
                                                                                                                                                
Working on images from caldor_2016-08-24...                                                                          | 0/2457 [00:00<?, ?tile/s]
                                                                                                                                                
Merging caldor_2017-06-14.                                                                                                                      
Processing easting=765966, northing=4308659: 100%|██████████████████████████████████████████████████████| 2457/2457 [2:31:42<00:00,  3.70s/tile]
                                                                                                                                                
Working on images from caldor_2017-06-14...                                                                                                     

Finished creating  2020 x 2020 tiles for the satellite imagery...                                                                               

NOW MAKING THE SATELLITE INPUTS FOR THE NEURAL NETWORK...

Processing tiles:  75%|█████████████████████████████████████████████████████████▌                   | 12856/17200 [2:22:22<1:07:31,  1.07tile/s]

```
<br>

### Creating sensor and solar look angles





---


## Authors

- Mia Mitchell **[mitchell.mia01@gmail.com]** (corresponding)
- Chuck Abolt **[chuck.abolt@gmail.com]**
- Zachary Crennen **[zcrennen@lanl.gov]**
- Adam Atchley **[aatchley@lanl.gov]**

## Version History
0.0.0 (June 2025)

## License
O#: O5010

This program is Open-Source under the BSD-3 License.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Citation

Please use the following citations if using our work:

```
@article{abolt2025deep,
  title        = {Deep-learning-based canopy height model generation from sub-meter resolution panchromatic satellite imagery},
  author       = {Abolt, Charles J. and Santos, Javier E. and Atchley, Adam L. and Wells, Lucas and Martin, Daithi and Parsons, Russell A. and Linn, Rodman R.},
  journal      = {Machine Learning: Science and Technology},
  year         = {2025},
  volume       = {6},
  number       = {015013},
  doi          = {10.1088/2632-2153/ada47e},
  url          = {https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_abolt_c001.pdf}
}

@software{CHM-MS-net-Canopy-Height-Model2025,
  title        = {{CHM-MS-net Canopy Height Model} (v0.0.0)},
  author       = {Mitchell, Mia and Abolt, Charles and Crennen, Zachary and Atchley, Adam},
  year         = {2026},
  url          = {https://github.com/lanl/CHM-MS-net-Canopy-Height-Model/tree/v0.0.0},
  note         = {Computer software, version 0.0.0},
}
```

## Sources
Allred, B. W., McCord, S. E., & 	Morford, S. L. (2025). Canopy height model and NAIP imagery pairs across CONUS. Scientific Data, 12(1), 322. https://doi.org/10.1038/s41597-025-04655-z

Dassot, M., Constant, T., & Fournier, M. (2011). The use of terrestrial LiDAR technology in forest science: Application Fields, Benefits and Challenges. Annals of Forest Science, 68, 959-974. https://doi.org/10.1007/s13595-011-0102-2

Linn, R. R., Goodrick, S. L., Brambilla, S., Brown, M. J., Middleton, R. S., O'Brien, J. J., & Hiers, J. K. (2020). QUIC-fire: A fast-running simulation tool for prescribed fire planning. Environmental Modelling & Software, 125, 104616. https://doi.org/10.1016/j.envsoft.2019.104616

Marcozzi, A., Wells, L., Parsons, R., Mueller, E., Linn, R., & Hiers, J. K. (2025). FastFuels: Advancing wildland fire modeling with high-resolution 3D fuel data and data assimilation. Environmental Modelling & Software, 183, 106214. https://doi.org/10.1016/j.envsoft.2024.106214

## Acknowledgments
This research was funded and supported by the Laboratory Directed Research and Development under 'Experiemental Research' at Los Alamos National Laboratory. Thank you to the entire FIRE team and others in the Earth and Environmental Sciences division, including but not limited to Agnese Marcato, Julia Oliveto, and Javier Santos. 
