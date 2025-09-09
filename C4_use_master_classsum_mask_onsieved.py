import shutil
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
from src.tondor.util.tool import reproject_multibandraster_toextent

def raster2array(rasterfn):
    raster = gdal.Open(str(rasterfn))
    band = raster.GetRasterBand(1).ReadAsArray().astype('float')
    band[np.isnan(band)] = np.nan
    return band

def create_large_sieved(file1_path, file1_path_sieved, size_threshold=10000):
    file1_path_sieved_changevalidity = file1_path_sieved.parent.joinpath(f"{file1_path_sieved.stem}_tmp.tif")
    # Construct the gdal_sieve.py command
    command = [
        'gdal_sieve.py',
        '-st', str(size_threshold),
        '-8',  # 8-connected neighborhood
        '-nomask',  # No mask applied
        '-of', 'GTiff',  # Output format as GeoTIFF
        str(file1_path),  # Input raster file
        str(file1_path_sieved_changevalidity)  # Output raster file
    ]

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Sieve operation completed successfully. Output saved to '{file1_path_sieved}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error during sieve operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    change_validity= raster2array(str(file1_path_sieved_changevalidity))
    actual_detection = raster2array(str(file1_path))

    actual_detection[change_validity==1] = 0
    save_raster_template(file1_path, file1_path_sieved, actual_detection, GDT_Byte)
    file1_path_sieved_changevalidity.unlink(missing_ok=True)


def create_masked_file(orbit_class_largesieve_filepath, tile_deforestation_mask, orbit_class_masked_filepath):
    array_to_mask = raster2array(orbit_class_largesieve_filepath)
    mask = raster2array(tile_deforestation_mask)

    array_to_mask[mask == 0] = 0
    save_raster_template(orbit_class_largesieve_filepath, orbit_class_masked_filepath, array_to_mask, GDT_Byte)


def create_sieved(result_filepath_for_sieve, result_filepath_sieved, sieve=10):
    sieve_args = ["gdal_sieve.py", #"-mask", str(mask_tif),
                  "-4", "-nomask",
                  "-st", str(sieve),
                  str(result_filepath_for_sieve), str(result_filepath_sieved)]
    cmd_output = subprocess.run(sieve_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("exit code {} --> {}".format(cmd_output.returncode, sieve_args))

    intermediate_path = result_filepath_sieved.parent.joinpath(result_filepath_sieved.name.replace(".tif", "_inter.tif"))
    shutil.copy(result_filepath_sieved, intermediate_path)
    result_filepath_sieved.unlink()

    raster_ds = gdal.Open(str(intermediate_path))
    raster_array = raster_ds.GetRasterBand(1).ReadAsArray()
    save_raster_template(intermediate_path, result_filepath_sieved, raster_array, data_type=GDT_Byte)
    intermediate_path.unlink()


#'18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP','18LVR', '20LLQ',
tiles = [ '18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP','18LVR', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']

# '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYH', '22MBT', '22MGB']
print(len(tiles))

model_version = 'best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score'
base_version = "jaxa"
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir", "reclassification")
os.makedirs(work_dir, exist_ok=True)
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion_parent = detection_folder.joinpath(f'{model_version}')
########################################################################################################################
support_data = amazonas_root_folder.joinpath("support_data")
merged_sar_ref_folder = support_data.joinpath(f"ref_worldcover_sarmosaic")
predicted_baselc_folder = support_data.joinpath(f"base_worldcover_prediction")
baselc_folder = support_data.joinpath(f"base_worldcover")

prob_var_mask_folder = support_data.joinpath("tile_prob_pixel_info_training_years")
var_threshold_name_suffix = "pt01"
prob_var_mask_prediction_folder = support_data.joinpath("tile_prob_pixel_info")


deforesation_mask_folder = support_data.joinpath("deforestation_mask")
os.makedirs(deforesation_mask_folder, exist_ok=True)
########################################################################################################################

for tile_item in tiles:

    tile_deforestation_mask = deforesation_mask_folder.joinpath(f"master_classsum_mask_{tile_item}.tif")
    tile_mask_array = raster2array(tile_deforestation_mask)
    tile_mask_array = (1 - tile_mask_array).astype(bool)

    detection_folder_aiversion_tile_zscore_check = detection_folder.joinpath(f'{model_version}', tile_item, "zscore_checked")

    detection_folder_aiversion_tile_zscore_check_filtered = detection_folder.joinpath(f'{model_version}', tile_item,
                                                                             "zscore_checked_filtered")
    os.makedirs(detection_folder_aiversion_tile_zscore_check_filtered, exist_ok=True)

    folder_path_files = os.listdir(detection_folder_aiversion_tile_zscore_check)
    orbit_directions = []
    for folder_path_file_item in folder_path_files:
        if folder_path_file_item in ["ascending", "descending"]:
            orbit_directions.append(folder_path_file_item)


    for orbit_direction_item in orbit_directions:

        detection_folder_aiversion_tile_zscore_check_filtered_orbit = detection_folder_aiversion_tile_zscore_check_filtered.joinpath(orbit_direction_item)
        os.makedirs(detection_folder_aiversion_tile_zscore_check_filtered_orbit, exist_ok=True)

        detection_folder_aiversion_tile_zscore_check_orbit = detection_folder_aiversion_tile_zscore_check.joinpath(orbit_direction_item)
        detection_folder_aiversion_tile_zscore_check_orbit_files = os.listdir(detection_folder_aiversion_tile_zscore_check_orbit)

        for detection_folder_aiversion_tile_zscore_check_orbit_file_item in detection_folder_aiversion_tile_zscore_check_orbit_files:
            detection_folder_aiversion_tile_zscore_check_orbit_fileitem_path = detection_folder_aiversion_tile_zscore_check_orbit.joinpath(detection_folder_aiversion_tile_zscore_check_orbit_file_item)

            detection_array = raster2array(detection_folder_aiversion_tile_zscore_check_orbit_fileitem_path).astype(bool)

            filtered_detection_array =  ((detection_array) & (~tile_mask_array)).astype(np.int32)

            detection_folder_aiversion_tile_zscore_check_filtered_orbit_filepath = detection_folder_aiversion_tile_zscore_check_filtered_orbit.joinpath(detection_folder_aiversion_tile_zscore_check_orbit_file_item)
            save_raster_template(detection_folder_aiversion_tile_zscore_check_orbit_fileitem_path, detection_folder_aiversion_tile_zscore_check_filtered_orbit_filepath, filtered_detection_array, GDT_Byte)
