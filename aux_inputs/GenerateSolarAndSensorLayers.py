"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

"""
This script processes off-nadir angle, target azimuth, solar elevation, solar azimuth from the angle metadata of the satellite imagery and produces 2D array inputs for the neural network. Originally, written in Matlab by Chuck Abolt, but edited and written by Mia Mitchell in Santa Fe, New Mexico. Summer 2024.

"""
import numpy as np
import pandas as pd
import os
import argparse
import re
from PIL import Image

def SensorSolarAngles(output_directory, path_to_csv):
    """
    Description
    ___________
    The main function for create the generating sensor and solar 2D array inputs

    Parameters
    __________
    output_directory : str
        the path to output directory for the solar and sensor 2D arrays/TIFFs
    path_to_csv : str
        the path to the .csv that contains the off-nadir, target azimuth, solar azimuth, and solar elevation information for the satellite imagery
    
    Returns
    _______

    solar_input : str
        the path to the solar inputs directory
    sensor_input : str
        the path the sensor inputs directory
    """
    
    solar_input = os.path.join(output_directory,'solar')
    sensor_input = os.path.join(output_directory,'sensor')
    os.makedirs(solar_input, exist_ok=True)
    os.makedirs(sensor_input, exist_ok=True)

    def getSolarNormal(site, date, sensor): 
        """
        Description
        ___________
        Function calculates the solar normal vector for a given site and date

        Parameters
        __________
        site: str
            site for the solar normal to be calculated (extracted from metadata)
        date: str
            the date of the solar normal to be calculated (extracted from metadata)
    
        Returns
        _______
        indexed: DataFrame 
            the relevant angle metadata for the specified site and date
        normal: array 
            the solar normal vector for the specified site and date
        """
        angletable = pd.read_csv(path_to_csv)

        indexed = angletable[(angletable['site'] == site) &
        (angletable['date'] == date)]
        indexed = indexed[indexed['sensor'] == sensor]

        theta = indexed['solarelevation'].values[0]
        phi = indexed['solarazimuth'].values[0]
        z0 = 10
        x0 = np.sqrt(z0**2 / (((np.tan(np.radians(theta)))**2) + (((np.tan(np.radians(theta)))**2) / (np.tan(np.radians(phi)))**2)))
        if phi > 180:
            x0 = x0 * -1
        y0 = x0 / np.tan(np.radians(phi))
        s = np.sqrt(x0**2 + y0**2 + z0**2)
        xn = x0 / s
        yn = y0 / s
        zn = z0 / s
        xg, yg = np.meshgrid(np.arange(-127.75, 128.25, 0.5), np.arange(-127.75, 128.25, 0.5))
        yg = np.flipud(yg)
        zg = -xn/zn * xg + -yn/zn * yg
        normal = zg
        return indexed, normal
        
    def getSensorNormal(site, date, sensor):
        """
        Description
        ___________
        Function calculates the sensor normal vector for a given site and date

        Parameters
        __________
        site: str
            site for the solar normal to be calculated (extracted from metadata)
        date: str
            the date of the solar normal to be calculated (extracted from metadata)
    
        Returns
        _______
        indexed: dataframe 
            the relevant angle metadata for the specified site and date
        normal: array 
            the solar normal vector for the specified site and date
        """
        angletable = pd.read_csv(path_to_csv)
        indexed = angletable[(angletable['site'] == site) &
        (angletable['date'] == date)]
        indexed = indexed[indexed['sensor'] == sensor]

        theta = 90 - indexed['offnadir'].values[0]
        phi = (indexed['targetazimuth'].values[0]+180)%360
        z0 = 10
        x0 = np.sqrt(z0**2 / (((np.tan(np.radians(theta)))**2) + (((np.tan(np.radians(theta)))**2) / (np.tan(np.radians(phi)))**2)))
        if phi > 180:
            x0 = x0 * -1
        y0 = x0 / np.tan(np.radians(phi))
        s = np.sqrt(x0**2 + y0**2 + z0**2)
        xn = x0 / s
        yn = y0 / s
        zn = z0 / s
        xg, yg = np.meshgrid(np.arange(-127.75, 128.25, 0.5), np.arange(-127.75, 128.25, 0.5))
        yg = np.flipud(yg)
        zg = -xn/zn * xg + -yn/zn * yg
        normal = zg
        return indexed, normal
    
    wvsummary = pd.read_csv(path_to_csv)
    solar_max_global = float('-inf')
    sensor_max_global = float('-inf')

    num_rows = wvsummary.shape[0]
    for i in range(num_rows):
        indexed, solarNormal = getSolarNormal(wvsummary['site'][i], wvsummary['date'][i], wvsummary['sensor'][i])
        indexed, sensorNormal = getSensorNormal(wvsummary['site'][i], wvsummary['date'][i], wvsummary['sensor'][i])
        solar_max_global = max(solar_max_global, np.max(np.abs(solarNormal)))
        sensor_max_global = max(sensor_max_global, np.max(np.abs(sensorNormal)))
        solar_scale = solar_max_global / 128  
        sensor_scale = 128 / sensor_max_global

        solarNormal = (solarNormal / solar_scale + 128).astype(np.uint8)
        sensorNormal = (sensorNormal * sensor_scale + 128).astype(np.uint8)


        datetxt = pd.to_datetime(wvsummary['date'][i])
        datetxt = datetxt.strftime('%Y-%m-%d')
        sensor =  wvsummary['sensor'][i]
        match = re.fullmatch(r"(\w+)-(\w+)", sensor)
        if match:
            filename = wvsummary['site'][i] + ('.' + match.group(2) if match.group(2) else '') + '_' + datetxt
        else:
            filename = wvsummary['site'][i] + '_' + datetxt


        Image.fromarray(solarNormal).save(os.path.join(output_directory, "solar", filename + '.tif')) 
        Image.fromarray(sensorNormal).save(os.path.join(output_directory, "sensor", filename + '.tif'))
        

        #print('Done', i+1, 'of', str(wvsummary.shape[0]) + ' arrays!')

    return solar_input, sensor_input

def main():
    parser = argparse.ArgumentParser(description='Generating sensor and solar 512 x 512 arrays.')
    parser.add_argument('--output_directory', type=str, help='Path to the output directory for arrays', required=True)
    parser.add_argument('--path_to_csv', type=str, help='Path to the CSV that contains angle metadata', required=True)
    
    args,  = parser.parse_args()
    SensorSolarAngles(args.output_directory, args.path_to_csv)
   
if __name__ == "__main__":
    main()