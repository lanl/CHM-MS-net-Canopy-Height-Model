# CHMer (Canopy Height Model-er)

## Motivation

Canopy Height Models (CHMs) are useful tools for a wide variety of applications: 3D construction of wildland fuels, biomass estimates, land management, tracking deforestation, etc (Linn et al., 2020; Marcozzi et al., 2025). High-resolution CHMs are usually obtained from processing Digital Terrain Models (DTMs) from LiDAR data and subtracting those from Digital Elevation Models (DEMs) (Allred et al., 2025). However, LiDAR data is expensive and is collected less frequently compared to satellite imagery (Dassot et al., 2011). Therefore, this program leverages a system of convolutional neural networks (MS-net) to predict CHMs from satellite imagery.
## Description

These are the required inputs to use CHMer, followed by the output:

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
      <td>Solar and sensor inputs are created with CHMer using the target azimuth, off-nadir angle, solar azimuth, and solar elevation metadata from the satellite imagery </td>
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

Download from [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/)

<br>

**There, you can select your preferences (OS, package manager, Python version, and CUDA version), and it will generate the exact install command you need.**




## Getting Started
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

#### Throughout the documentation, we will be using the project 'caldor' as an example. Caldor references the Caldor Fire of 2021, south of Lake Tahoe, California. Here is the example:
```
UW PICO 5.09                                    File: .env                                       

project_path=/Users/mia/Documents/Projects/caldor_run
site=caldor
utm=EPSG:32610
dem_path=/Users/mia/Documents/caldor_dem.tif
lidar_path=/Users/mia/Documents/caldor_lidar.tif
site_shapefile_path=/Users/mia/Downloads/wgs84_caldor
satellite_download_dir=/Users/mia/Downloads/satellite_data
```
Document the appropriate paths and directories in the <strong><span style="color:#33484D">.env</span></strong> before running the program.

If you are using Maxar's DigitalGlobe to accquire your satellite imagery, then WKT files are generated for you to use in their [online platform](https://evwhs.digitalglobe.com/myDigitalGlobe/login): follow [the documentation here](./docs/DIGITALGLOBE.md). If you are using your own imagery, [see additional details here](./docs/OWNIMAGERY.md) and place your imagery in the **satellite_download_dir** specified in your .env. 


## Running Main

After creating the .env, running the program:

```
CHMer... 🌳

Project Directory: /Users/mia/Documents/Projects/caldor_run

Confirming shapefile(.shp) is in provided directory...

Using shapefile "caldor_wgs84.shp" in /Users/mia/Downloads/wgs84_caldor

Processing the satellite imagery...

Open this .CSV file for documenting angle metadata. Please keep this open as you proceed:

         C:/Users/mia/Documents/caldor_run/metadata/angle-metadata.csv


Are you using DigitalGlobe? (Y/N):


```
<br>

If you are using DigitalGlobe, answer Y and follow the documentation [here](./docs/DIGITALGLOBE.md). If you are using your own imagery, answer N and follow the documentation [here](./docs/OWNIMAGERY.md)

---

Afterwards, proceed [here](./docs/PROCESSING.md) to obtain detailed instruction about preprocessing, utilizing the neural network, and post processing steps.

<br>

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
