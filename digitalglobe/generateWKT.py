"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""

"""
This grabs the bounding boxes of the shapefiles transforms them into lat/log WKTs. Written by Mia Mitchell in Santa Fe, New Mexico. Fall 2024.

"""

from shapely.geometry import box, Polygon, mapping
from shapely.ops import transform
from shapely.wkt import loads
import math
from pyproj import Proj, Transformer
import geopandas as gpd
import json, csv
import os
import argparse
from dotenv import load_dotenv, find_dotenv

def creating_geoinfo(projected_shapefile_path, printt = True):
    """
    Description
    ___________
    Main function for creating the WKT and geojsons. 

    Parameters
    __________
        site_shapefile_path : str
            The path to the site shapefile provided in the .env file
        printt : bool (optional)
            To allow certain print statements to pass
    """
     
    # Pulled from the .env 
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    project_directory=os.getenv('project_dir')
    fire_polygons = os.path.join(project_directory, 'site-polygons')
    utm = os.getenv('utm')
    site = os.getenv('site')

    # Functions 
    def utm_to_wgs84(easting, northing, zone_number, northern_hemisphere=True):
        """
        Description
        ___________
        This converts Universal Transverse Mercator (UTM) to Longtitude/Latitude coordinates.

        Parameters
        __________
        easting: int
            east-west distance, measured in meters
        northing: int
            north-south distance, measured in meters
        zone_number: int
            UTM zone number
        northern_hemisphere: bool
            if in the northern hemisphere, it is set to true. EPSG code starts with '326' for the north hemisphere and '327' for the southern hemisphere.
        
        Returns
        _______
        lon : int
            longitude
        lat : int
            latitude
            
        """
        proj_utm = Proj(proj='utm', zone=zone_number, datum='WGS84', south=not northern_hemisphere)
        proj_wgs84 = Proj(proj='latlong', datum='WGS84')
        transformer = Transformer.from_proj(proj_utm, proj_wgs84)
        lon, lat = transformer.transform(easting, northing)
        return lon, lat
   
    def generate_wkt_from_utm(bbox, zone_number, northern_hemisphere=True):
        """
        Description
        ___________
        This function generates a bounding box from the longitude/latitude coordinates for WKT 

        Parameters
        __________
        bbox: list
            bounding box of shapefile
        zone_number: int
            UTM zone number
        northern_hemisphere: bool
            if in the northern hemisphere, it is set to true. EPSG code starts with '326' for the north hemisphere and '327' for the southern hemisphere.

        Returns
        ______
        geo.wkt : str
            WKT string
        """
        min_easting, min_northing, max_easting, max_northing = bbox

        # Converts UTM coordinates to WGS84 (long/lat)
        min_lon, min_lat = utm_to_wgs84(min_easting, min_northing, zone_number, northern_hemisphere)
        max_lon, max_lat = utm_to_wgs84(max_easting, max_northing, zone_number, northern_hemisphere)

        geom = box(min_lon, min_lat, max_lon, max_lat)

        return geom.wkt

    def split_bounding_box(wkt):
        """
        Description
        ___________
        This function generates splits a bounding box into smaller bounding boxes

        Parameters
        __________
        wkt: str
            the well-known text file string

        Returns
        ______
        grid_cells : list
            list of grid cells and their bounds
        """
        bbox = loads(wkt) # Load wkt bbox
        if not isinstance(bbox, Polygon): # Make sure if polygon
            raise ValueError("The input WKT must be polygon.")
        
        # Find the area of the bounds
        minx, miny, maxx, maxy = bbox.bounds
        width = maxx - minx
        height = maxy - miny
        total_area = width * height 

        # Finding the minimum number of cells, thus cols and rows
        num_cells = math.ceil(total_area / 150000) # 150 km^2
        num_cols = math.ceil(math.sqrt(num_cells * width / height))
        num_rows = math.ceil(num_cells / num_cols)
        cell_width = (maxx - minx) / num_cols
        cell_height = (maxy - miny) / num_rows
        
        # Generate the grid cells
        grid_cells = []
        for i in range(num_rows):
            for j in range(num_cols):
                cell_minx = minx + j * cell_width
                cell_maxx = cell_minx + cell_width
                cell_miny = miny + i * cell_height
                cell_maxy = cell_miny + cell_height
                cell = Polygon([
                    (cell_minx, cell_miny),
                    (cell_minx, cell_maxy),
                    (cell_maxx, cell_maxy),
                    (cell_maxx, cell_miny),
                    (cell_minx, cell_miny)
                ])
                grid_cells.append(cell)
        
        return grid_cells

    def compute_bbox(geojson):
        """
        Description
        ___________
        Obtaining coordinates for geojsons

        Parameters
        __________
        geojson: tuple
            bounding box of cell(s)
            
        Returns
        ______
        e0 : int
            min easting
        e1 : int
            max easting
        n0 : int
            min northing
        n1 : int
            max northing
            
        """
        coordinates = geojson["coordinates"][0]  
        e0 = min(coord[0] for coord in coordinates)  
        e1 = max(coord[0] for coord in coordinates)  
        n0 = min(coord[1] for coord in coordinates)  
        n1 = max(coord[1] for coord in coordinates)  
        return e0, n0, e1, n1

    # Create output directories for WKT files and Geojsons
    output_directory = os.path.join(fire_polygons, 'wkt')
    os.makedirs(output_directory, exist_ok = True)
    output_geojson = os.path.join(fire_polygons, 'geojson')
    os.makedirs(output_geojson, exist_ok = True)

    # Read in site-metadata.csv
    csv_file = os.path.join(project_directory, "metadata", "site-metadata.csv")

    # Read site shapefile
    site_shapefile = gpd.read_file(projected_shapefile_path)

    # Obtain bounds and utm zone ; confirm northern or southern hemisphere
    min_x, min_y, max_x, max_y = site_shapefile.total_bounds
    site_shapefile_crs = f'{utm}'
    zone_number = site_shapefile_crs[-2:]

    if "326" in site_shapefile_crs:
        northern_hemisphere = True
    elif "327" in site_shapefile_crs:
        northern_hemisphere = False

    bbox = min_x, min_y, max_x, max_y
    
    wkt = generate_wkt_from_utm(bbox, zone_number, northern_hemisphere)
   
    #print("Do we need to divide it into smaller bounding boxes?")
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) # calculate area of the bounding box in meters

    # If the area is over 300 km squared, one should splitting the bounding box smaller files... 

    if area > 300000:  # 2 GB #ZACH -- Make sure it works with the data that JOSS is working with OR you are downloading new Caldor data.
        #print("Yes! Splitting bounding box into a grid because area of whole bounding box =", area, "square meters")
        grid = split_bounding_box(wkt)
        #print(f"Therefore, there will be {len(grid)} cells...")
        if printt==True:
            print("\033[1mWKT inputs for DigitalGlobe:\033[0m\n") 
        else:
            pass
        for idx, cell in enumerate(grid):
            if printt == True:  
                print(f"Cell {idx}: {cell.wkt}\n")
            else:
                pass
            outpath = os.path.join(output_directory, f"{site}_cell{idx}.txt")
            with open(outpath, 'w') as f:
                f.write(cell.wkt)

            # Now saving the geojsons, converting from WKTs
            geometry = loads(cell.wkt)
            project_to_utm = Transformer.from_crs("EPSG:4326", f'{utm}', always_xy=True).transform
            reprojected_geometry = transform(project_to_utm, geometry)
            geojson_dict = mapping(reprojected_geometry)
            geojson_dict["crs"] = {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::326{zone_number}"}
            }
            outpath_geojson = os.path.join(output_geojson, f"{site}_cell{idx}.geojson")
            with open(outpath_geojson, "w") as f:
                json.dump(geojson_dict, f, indent=4)
            e0, n0, e1, n1 =compute_bbox(geojson_dict)
            site_name = site + f"{idx}"

            # Writing additional information to the site-metadata.csv
            with open(csv_file, mode='a', newline='') as dst:
                writer = csv.writer(dst)
                writer.writerow([site_name, float(e0), float(e1), float(n0), float(n1), utm, 'chunk bounds'])

    else: 
        print("Saving WKT as a single file because bounding box is not greater than 300 km squared...")
        outpath = os.path.join(output_directory, f"{site}.txt")
        with open(outpath, 'w') as f:
            f.write(wkt)
        if printt == True:   
            print(f"{wkt}\n")
        else:
            pass
        print(f"WKT saved to {outpath}")

        # Now saving the geojsons, converting from WKT
        geometry = loads(wkt)
        project_to_utm = Transformer.from_crs("EPSG:4326",f'{utm}', always_xy=True).transform
        reprojected_geometry = transform(project_to_utm, geometry)
        geojson_dict = mapping(reprojected_geometry)
        geojson_dict["crs"] = {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::326{zone_number}"}
            }
        outpath_geojson = os.path.join(output_geojson, f"{site}.geojson")
        with open(outpath_geojson, "w") as f:
            json.dump(geojson_dict, f, indent=4)
        print("Saving the geojsons...")
        e0, n0, e1, n1 = compute_bbox(geojson_dict)
        write_header = not os.path.exists(csv_file)

        # Writing additional information to the site-metadata.csv
        if write_header:
            with open(csv_file, mode='a', newline='') as dst:
                writer = csv.writer(dst)
                writer.writerow(["name", "e0", "e1", "n0", "n1", 'utm_code', 'chunk_or_whole'])
                writer.writerow([site, utm, float(e0), float(e1), float(n0), float(n1),'whole bounds'])
        else:
            print("Bounds of whole border already exists in file. No need to append....")

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Generate WKT Files')
   parser.add_argument('--shapefile_path', type=str, required=True,
                        help='The projected shapefile path')
   parser.add_argument('--print_statements', type=bool, default=True, required=False, help='See print statements')

   args = parser.parse_args()

   creating_geoinfo(args.shapefile_path, args.print_statements) 


