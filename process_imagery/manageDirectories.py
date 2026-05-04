"""
© 2026. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
"""


"""
This is script is used to create folders based on metadata present in the angle-metadata.csv. Written by Mia Mitchell in Santa Fe, New Mexico. Winter 2024. 
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv, find_dotenv


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
utm = os.getenv('utm')
site = os.getenv('site')


# This creates the class Manager
class Manager:
    """
    A class used to manage directories from the angle-metadata.csv and  site_metadata.csv
    """
    def __init__(self, base_path):
    
        """
        Parameters
        __________

        base_path : str
            The location of where the directories will be created

        """
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)
        utm = os.getenv('utm')
        site = os.getenv('site')

    def create_mainfolders(self, csv_file, site):
        """
        Description
        ___________
        This function creates mainfolders for downloads of site imagery that exists in the digitalglobe-downloads.csv 
        
        Parameters
        __________
        csv_file : str
            The digitalglobe-downloads.csv that has information about sites and angle  metadata
        site : str
            The site to filter the data for
        
        Returns
        _______
        main_folder_paths : list
            A list of the folder paths from the angle-metadata.csv
        """
        df = pd.read_csv(csv_file)
        main_folder_paths = {}
        for _, row in df.iterrows():
            site = row['site']
            folder_name = f"{site}"
            folder_path_main = os.path.join(self.base_path, folder_name)
            os.makedirs(folder_path_main, exist_ok=True)
            main_folder_paths[folder_name] = folder_path_main
        return main_folder_paths
    def create_subfolders(self, csv_file, main_folder_paths):
            """
            Description
            ___________
            This function creates subfolders for specific downloads of site imagery that to be placed in the main folder (see description of script) 

            Parameters
            __________
            csv_file : str
                The digitalglobe-downloads.csv that has information about sites and angles
            main_folder_paths : dict
                A dictionary of the folder paths from the angle-metadata.csv
            """
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                site = row["site"]
                # Makes dates into the correct format (if incorrectly formated)
                date = row["date"].replace('/', '-')  
                date = pd.to_datetime(row["date"]).strftime('%Y-%m-%d')
                # If there are sites that have the same name, cell, sensor, and date, but different angle information from digital-globe, this creates unique site IDs
                sensor = row["sensor"]
                # For dealing with any sensor, if num/lettersA-num/lettersB, than site.B_date otherwise it is site_date
                # This is to deal with if you have a site that has the same date, same site, but different metadata, you add a dash to the sensor to differentiate the two collections of imagery (see documentation)
                match = re.fullmatch(r"(\w+)-(\w+)", sensor)
                if match:
                    subfolder_name = site + ('.' + match.group(2) if match.group(2) else '') + '_' + date
                else:
                    subfolder_name = site + '_' + date

                main_folder = f"{site}"
                if main_folder in main_folder_paths:
                    subfolder_path = os.path.join(main_folder_paths[main_folder], subfolder_name)
                    os.makedirs(subfolder_path, exist_ok=True)
                    #print(f"Created folder: {subfolder_path}")
                else:
                    print(f"No main folder found for {main_folder}")
def setup_dirs(raw_directory, csv_file, site):
   manager = Manager(raw_directory)
   digitalglobe_metadata = pd.read_csv(csv_file)
   if digitalglobe_metadata.isnull().values.any(): 
        raise RuntimeError(f"Metadata is not complete. Please see that all fields are complete. Rerun program afterwards.") 
   else:
       pass
   main_folder_paths = manager.create_mainfolders(csv_file, site)
   manager.create_subfolders(csv_file, main_folder_paths)

if __name__ == '__main__': 
    setup_dirs()
