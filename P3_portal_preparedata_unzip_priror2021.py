import os
import shutil
import subprocess

import geopandas as gpd
import logging
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from shapely.geometry import Point
from pathlib import Path
from O7_openeo_backscatter import get_temporalextents_mastertemporalextent

import numpy as np
from osgeo.gdalconst import GDT_Byte

from src.tondor.util.tool import read_raster_info, save_raster_template, raster2array

def get_temporalextents(start, end):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    time_windows= []

    time_window_start = start_date
    while (time_window_start < datetime.strptime("2024-12-23", "%Y-%m-%d")):
        time_window_end = time_window_start + timedelta(24)
        time_windows.append([time_window_start, time_window_end])
        time_window_start = time_window_end
    return time_windows

def get_3months_temporalextents(start, end):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    time_windows= []

    time_window_start = start_date
    while (time_window_start < datetime.strptime("2024-12-23", "%Y-%m-%d")):
        time_window_end = time_window_start + relativedelta(months=3)
        time_windows.append([time_window_start, time_window_end])
        time_window_start = time_window_end
    return time_windows



gpkg_file = "/mnt/hddarchive.nfs/amazonas_dir/s14ama_updatedtiles.gpkg"

zip_basefolder = "/home/eouser/Downloads/sen4ama"
change_detection_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/change_detection")
zip_files = os.listdir(zip_basefolder)
zip_tiles = [tile_item.split(".")[0] for tile_item in zip_files]

for zip_tile_item in sorted(zip_tiles):
    print(f"{zip_tile_item}")
    change_detection_tile = change_detection_folder.joinpath(zip_tile_item)
    os.makedirs(change_detection_tile, exist_ok=True)

    os.chdir(change_detection_tile)
    unzip_cmd = ["unzip", str(Path(zip_basefolder).joinpath(f'{zip_tile_item}.zip'))]
    subprocess.run(unzip_cmd)

