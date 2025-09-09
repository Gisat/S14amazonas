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

changedetection_openeo_processing = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_processing")
portal_data_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data")
os.makedirs(portal_data_folder, exist_ok=True)
portal_data_folder_post_2021 = portal_data_folder.joinpath("post_2021", "tiles")
os.makedirs(portal_data_folder_post_2021, exist_ok=True)
portal_data_folder_post_2021_composites = portal_data_folder.joinpath("post_2021", "composites")
os.makedirs(portal_data_folder_post_2021_composites, exist_ok=True)

portal_data_folder_post_2021_mosaics = portal_data_folder.joinpath("post_2021", "mosaics")
os.makedirs(portal_data_folder_post_2021_mosaics, exist_ok=True)

portal_data_folder_prior_2021 = portal_data_folder.joinpath("prior_2021", "tiles")
os.makedirs(portal_data_folder_prior_2021, exist_ok=True)

# gdf = gpd.read_file(gpkg_file)
# print(gdf.head())

temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent("2020-11-02",
                                                                                    "2025-03-29")

temporal_extents_24days = get_temporalextents("2021-01-13","2025-03-29")

# post_2021_tiles = []
# for row_index, row_data in gdf.iterrows():
#     print(f"tilename : {row_data.Name} - {row_data.prior_2021} {row_data.post_2021}")
#     if row_data.post_2021 == 1:
#         post_2021_tiles.append(row_data.Name)

post_2021_tiles = ['22LCL', '22LDL', '22NDK', '23MLQ', '18LZR', '18MUE', '18MYS', '18MZS', '18NUF', '18NVF', '18NWF', '18NWG', '18NXG', '18NXH', '18NYH', '18NZH', '19LBL', '19LCL', '19LDK', '19LDL', '19LEJ', '19LEK', '19LEL', '19LFJ', '19LFK', '19LFL', '19MCM', '20LLQ', '20LLR', '20LMQ', '20LMR', '20LNQ', '20LNR', '20LPP', '20LPQ', '20LPR', '20LQP', '20LQR', '20LRM', '20MND', '20MNE', '20MPB', '20MPS', '20MPT', '20MQA', '20MQB', '20MQC', '20MQE', '20MQS', '20MQT', '20MQU', '20MRA', '20MRB', '20MRC', '20MRS', '20MRT', '20MRU', '20NNF', '20NPF', '20NQF', '20NQG', '20NQJ', '20NRH', '21LTG', '21LTH', '21LUG', '21LUH', '21LVF', '21LWF', '21LWG', '21LXF', '21LXG', '21LXK', '21LXL', '21LYF', '21LYG', '21LYK', '21LYL', '21LZF', '21LZG', '21LZH', '21LZJ', '21LZK', '21MTM', '21MTN', '21MTP', '21MTQ', '21MTR', '21MTS', '21MTT', '21MUM', '21MUN', '21MUP', '21MUQ', '21MUR', '21MUS', '21MWN', '21MWP', '21MWQ', '21MWR', '21MXM', '21MXN', '21MXP', '21MXQ', '21MXR', '21MXU', '21MYM', '21MYN', '21MYP', '21MYQ', '21MYR', '21MYS', '21MYU', '21MZN', '21MZP', '21MZR', '21MZS', '22LBM', '22LBN', '22LBP', '22LCM', '22LCN', '22LCP', '22LDM', '22MBA', '22MBB', '22MBT', '22MBU', '22MCA', '22MCB', '22MCD', '22MCE', '22MCT', '22MCU', '22MDA', '22MDB', '22MDE', '22MDU', '22MEA', '22MEB', '22MFT', '22MGS', '22MGT', '22MGU', '22MGV', '22MHA', '22NCF', '22NCG', '22NCJ', '22NCK', '22NDF', '22NDG', '22NDH', '22NDJ', '22NEF', '22NEG', '22NEH', '23MKQ', '23MKR', '23MKS', '23MLS', '23MLT', '23MLU', '23MMS', '23MMT']
print(f"{len(post_2021_tiles)} - {post_2021_tiles}")



time_windows_3months = get_3months_temporalextents("2021-01-01","2025-03-01")
for time_windows_3months_start, time_windows_3months_end in time_windows_3months:

    output_raster = portal_data_folder_post_2021_mosaics.joinpath(f"DEC_3monthcomposite_mosaic_{time_windows_3months_start.strftime('%Y-%m-%d')}_{time_windows_3months_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.tif")
    if not output_raster.exists():

        mosaic_input_list = []
        for post_2021_tile_item in post_2021_tiles:
            portal_data_folder_post_2021_composites_tile = portal_data_folder_post_2021_composites.joinpath(post_2021_tile_item)
            portal_tile_3monthcomposite_tif = portal_data_folder_post_2021_composites_tile.joinpath(
                f"DEC_3monthcomposite_{post_2021_tile_item}_{time_windows_3months_start.strftime('%Y-%m-%d')}_{time_windows_3months_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.tif")
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
                  "-co", "NUM_THREADS=1", # Multi-threading
                  "-co", "COMPRESS=DEFLATE",
                  "-co", "BIGTIFF=YES",
                  "-dstnodata", "0"  # NoData value
              ] + mosaic_input_list + [str(output_raster)]

        # Run command
        subprocess.run(cmd, check=True)

        ##########################
        # output_raster_vrt = portal_data_folder_post_2021_mosaics.joinpath(f"DEC_3monthcomposite_mosaic_{time_windows_3months_start.strftime('%Y-%m-%d')}_{time_windows_3months_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.vrt")
        #
        # # Step 1: Build VRT from all rasters
        # build_vrt_cmd = ["gdalbuildvrt", str(output_raster_vrt)] + mosaic_input_list
        # subprocess.run(build_vrt_cmd, check=True)
        #
        # # Step 2: Warp the VRT to EPSG:4326 and save as GeoTIFF
        # warp_cmd = [
        #     "gdalwarp",
        #     "-t_srs", "EPSG:4326",  # Target CRS
        #     "-r", "near",  # Resampling method
        #     "-multi",  # Multi-threading
        #     "-dstnodata", "0",  # NoData value
        #     str(output_raster_vrt),
        #     str(output_raster)
        # ]
        # subprocess.run(warp_cmd, check=True)


    print(f"Merged raster saved to {output_raster}")
    print("---------------------------------------------")

    # changedetection_openeo_processing_tile = changedetection_openeo_processing.joinpath(f"{row_data.Name}")
        # if not changedetection_openeo_processing_tile.exists():
        #     raise Exception(f"{row_data.Name} is missing")
        # portal_data_folder_post_2021_tile = portal_data_folder_post_2021.joinpath(f"{row_data.Name}")
        # os.makedirs(portal_data_folder_post_2021_tile, exist_ok=True)
        #
        # treecover_change_layer = changedetection_openeo_processing.joinpath(row_data.Name, "deforestation_layers")
        # for temp_window_24days_start, temp_window_24days_end in temporal_extents_24days:
        #     temp_window_list, master_temp_window_list = get_temporalextents_mastertemporalextent(temp_window_24days_start.strftime("%Y-%m-%d"),
        #                                                                 temp_window_24days_end.strftime("%Y-%m-%d"))
        #     portal_tile_24day_tif = portal_data_folder_post_2021_tile.joinpath(f"DEC_{row_data.Name}_{temp_window_24days_start.strftime('%Y-%m-%d')}_{temp_window_24days_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.tif")
        #
        #     treecover_change_list= []
        #     if len(temp_window_list) > 1:
        #         for temp_window_start, temp_window_end in temp_window_list:
        #             treecover_change_layer_item = treecover_change_layer.joinpath(f"DEC_{row_data.Name}_{temp_window_start}_{temp_window_end}_TREECOVERCHANGE.tif")
        #             if treecover_change_layer_item.exists():
        #                 treecover_change_list.append(treecover_change_layer_item)
        #         print(f"aggregating {treecover_change_list}")
        #
        #         if len(treecover_change_list) > 0:
        #             if not portal_tile_24day_tif.exists():
        #                 tree_coverchange_array_list= []
        #                 for treecover_change_list_item in treecover_change_list:
        #                     tree_coverchange_array = raster2array(treecover_change_list_item)
        #                     tree_coverchange_array[np.isnan(tree_coverchange_array)] = 0
        #                     tree_coverchange_array_list.append(tree_coverchange_array)
        #                 tree_coverchange_array_stack = np.stack(tree_coverchange_array_list, axis=0)
        #                 tree_coverchange_sum = np.sum(tree_coverchange_array_stack, axis=0)
        #                 save_raster_template(treecover_change_list_item, portal_tile_24day_tif, tree_coverchange_sum, GDT_Byte, 0)
        #             print(f"created {portal_tile_24day_tif}")
        #     else:
        #         treecover_change_layer_item = treecover_change_layer.joinpath(
        #             f"DEC_{row_data.Name}_{temp_window_24days_start.strftime('%Y-%m-%d')}_{temp_window_24days_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.tif")
        #         if treecover_change_layer_item.exists():
        #             if not portal_tile_24day_tif.exists():
        #                 shutil.copy(treecover_change_layer_item ,portal_tile_24day_tif)
        #             print(f"{treecover_change_layer_item} --> {portal_tile_24day_tif}")
        #
        # ### composites
        # portal_data_folder_post_2021_composites_tile = portal_data_folder_post_2021_composites.joinpath(row_data.Name)
        # os.makedirs(portal_data_folder_post_2021_composites_tile, exist_ok=True)
        #
        # treecover_change_layer = changedetection_openeo_processing.joinpath(row_data.Name, "deforestation_layers")
        # treecover_change_layer_tifs_list = os.listdir(changedetection_openeo_processing.joinpath(row_data.Name, "deforestation_layers"))
        #
        # time_windows_3months = get_3months_temporalextents("2021-01-01","2025-03-01")
        # for time_windows_3months_start, time_windows_3months_end in time_windows_3months:
        #     temp_window_list, master_temp_window_list = get_temporalextents_mastertemporalextent(time_windows_3months_start.strftime("%Y-%m-%d"),
        #                                                                 time_windows_3months_end.strftime("%Y-%m-%d"))
        #
        #     portal_tile_3monthcomposite_tif = portal_data_folder_post_2021_composites_tile.joinpath(f"DEC_3monthcomposite_{row_data.Name}_{time_windows_3months_start.strftime('%Y-%m-%d')}_{time_windows_3months_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.tif")
        #
        #     if not portal_tile_3monthcomposite_tif.exists():
        #         treecover_change_list = []
        #         for treecover_change_layer_tifs_list_item in sorted(treecover_change_layer_tifs_list):
        #             treecover_change_layer_tifs_list_item_time = datetime.strptime(treecover_change_layer_tifs_list_item.split("_")[2], "%Y-%m-%d")
        #             if  time_windows_3months_start < treecover_change_layer_tifs_list_item_time <= time_windows_3months_end:
        #                     treecover_change_list.append(changedetection_openeo_processing.joinpath(row_data.Name,  "deforestation_layers", treecover_change_layer_tifs_list_item))
        #         print(f"compositing {treecover_change_list}")
        #
        #         tree_coverchange_array_list = []
        #         for treecover_change_list_item in treecover_change_list:
        #             tree_coverchange_array = raster2array(treecover_change_list_item)
        #             tree_coverchange_array[np.isnan(tree_coverchange_array)] = 0
        #             tree_coverchange_array_list.append(tree_coverchange_array)
        #
        #
        #         if len(treecover_change_list) > 0:
        #             tree_coverchange_array_stack = np.stack(tree_coverchange_array_list, axis=0)
        #             tree_coverchange_sum = np.sum(tree_coverchange_array_stack, axis=0)
        #             save_raster_template(treecover_change_list_item, portal_tile_3monthcomposite_tif, tree_coverchange_sum, GDT_Byte, 0)
        #
        #     print(f"created compositing{portal_tile_3monthcomposite_tif}")
