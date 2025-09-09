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

try:
    from src.tondor.util.tool import read_raster_info, save_raster_template, raster2array
except:
    from tondor.util.tool import read_raster_info, save_raster_template, raster2array

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


def main():

    gpkg_file = "/mnt/hddarchive.nfs/amazonas_dir/s14ama_updatedtiles.gpkg"

    changedetection_openeo_processing = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_processing")
    portal_data_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data")
    portal_data_folder_prior_2021_composites = portal_data_folder.joinpath("prior_2021", "composites", "tree_cover_change")
    prior_2021_tiles = os.listdir(portal_data_folder_prior_2021_composites)

    portal_data_folder_prior_2021_mosaics = portal_data_folder.joinpath("prior_2021", "mosaics")
    os.makedirs(portal_data_folder_prior_2021_mosaics, exist_ok=True)

    time_windows_3months = get_3months_temporalextents("2017-01-01","2021-12-31")
    for time_windows_3months_start, time_windows_3months_end in time_windows_3months:

        output_raster = portal_data_folder_prior_2021_mosaics.joinpath(f"DEC_3monthcomposite_mosaic_{time_windows_3months_start.strftime('%Y-%m-%d')}_{time_windows_3months_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.tif")
        if not output_raster.exists():

            mosaic_input_list = []
            for prior_2021_tile_item in prior_2021_tiles:
                portal_data_folder_prior_2021_composites_tile = portal_data_folder_prior_2021_composites.joinpath(prior_2021_tile_item)
                portal_tile_3monthcomposite_tif = portal_data_folder_prior_2021_composites_tile.joinpath(
                    f"DEC_3monthcomposite_{prior_2021_tile_item}_{time_windows_3months_start.strftime('%Y-%m-%d')}_{time_windows_3months_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.tif")
                if portal_tile_3monthcomposite_tif.exists():
                    mosaic_input_list.append(str(portal_tile_3monthcomposite_tif))
            print(f"creating {len(mosaic_input_list)} - {mosaic_input_list}")

            # # Build gdalwarp command
            cmd = [
                      "gdalwarp",
                      "-t_srs", "EPSG:4326",  # Target CRS
                      "-r", "near",  # Resampling method
                      "-ot", "Byte",
                      "-tr", "0.00017966", "0.00017966",
                      "-r", "max",
                       "-multi", # Multi-threading
                      "-co", "COMPRESS=DEFLATE",
                      "-co", "BIGTIFF=YES",
                      "-dstnodata", "0"  # NoData value
                  ] + mosaic_input_list + [str(output_raster)]

            # Run command
            subprocess.run(cmd, check=True)



        print(f"Merged raster saved to {output_raster}")
        print("---------------------------------------------")


if __name__ == "__main__":
    main()