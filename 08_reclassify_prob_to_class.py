from pathlib import Path
import os, requests
import subprocess
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte, GDT_Float32
import numpy as np
from datetime import datetime, timedelta, date
import math
from tondortools.tool import read_raster_info, save_raster, mosaic_tifs, save_raster_template
import tensorflow as tf

tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
cutoff_prob = 0.2
model_version = 'best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion = detection_folder.joinpath(f'{model_version}')
########################################################################################################################
support_data = amazonas_root_folder.joinpath("support_data")
merged_sar_ref_folder = support_data.joinpath(f"ref_worldcover_sarmosaic")
predicted_baselc_folder = support_data.joinpath(f"base_worldcover_prediction")
########################################################################################################################


for tile_item in tiles:
    # / mnt / hddarchive.nfs / amazonas_dir / support_data / base_worldcover_prediction / 18L
    # VQ_BASELC_2017_CLASS.tif
    # base_lc_tile_filepath = predicted_baselc_folder.joinpath(f"{tile_item}_BASELC_2017_CLASS.tif")
    # if not base_lc_tile_filepath.exists(): raise Exception(f"{base_lc_tile_filepath} does not exist")
    # dataset = gdal.Open(str(base_lc_tile_filepath))
    # chunk = dataset.GetRasterBand(1).ReadAsArray()

    detection_folder_aiversion_tile = detection_folder.joinpath(f'{model_version}', tile_item)
    folder_files_list = os.listdir(detection_folder_aiversion_tile)

    orbit_directions = os.listdir(detection_folder_aiversion_tile)
    for orbit_direction_item in orbit_directions:

        if orbit_direction_item not in ['ascending', 'descending']:continue

        detection_folder_aiversion_tile_orbit = detection_folder_aiversion_tile.joinpath(orbit_direction_item)
        mosaic_files = os.listdir(detection_folder_aiversion_tile_orbit)
        for mosaic_file_item in sorted(mosaic_files):

            if not "_PROB" in mosaic_file_item: continue

            probablity_filepath = detection_folder_aiversion_tile_orbit.joinpath(mosaic_file_item)
            print(f"reclassifying {probablity_filepath}")

            dataset = gdal.Open(str(probablity_filepath))
            chunk = dataset.GetRasterBand(1).ReadAsArray()

            mosaic_detection_class_path = probablity_filepath.parent.joinpath(probablity_filepath.name.replace("_PROB", "_CLASS"))
            prediction_class = (chunk < cutoff_prob).astype(np.byte)
            save_raster_template(str(probablity_filepath), str(mosaic_detection_class_path),
                                 prediction_class, GDT_Byte, nodata_value=None)
            print(f"creating {mosaic_detection_class_path}")