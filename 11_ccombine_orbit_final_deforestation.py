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

# Function to extract the date from the filename
def extract_date(filename):
    return filename.split('_')[2]  # Assuming the date is the 3rd substring


def create_merged_filepath(output_path, raster1, raster2):
    args = ["gdal_merge.py",
            "-separate",
            "-o", str(output_path),
            "-of", "GTiff",
            "-ot", "Float32",
            "-co", "compress=DEFLATE"]
    args = args + [str(raster1)] + [str(raster2)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")


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
#'18LVQ', '18LVR', '18LWR', '18NXG', '18NXH',
tiles = [ '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']

model_version = 'best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score'
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
########################################################################################################################

detection_folder_aiversion_reclassified = detection_folder_aiversion_parent.joinpath("reclassified")
detection_folder_aiversion_reclassified_sieved = detection_folder_aiversion_parent.joinpath("sieved")
detection_folder_aiversion_final_deforestation = detection_folder_aiversion_parent.joinpath("final_deforestation")
os.makedirs(detection_folder_aiversion_reclassified_sieved, exist_ok=True)
os.makedirs(detection_folder_aiversion_final_deforestation, exist_ok=True)

for tile_item in tiles:

    detection_folder_aiversion_reclassified_sieved_tile = detection_folder_aiversion_reclassified_sieved.joinpath(tile_item)

    detection_folder_aiversion_final_deforestation_tile = detection_folder_aiversion_final_deforestation.joinpath(tile_item)
    os.makedirs(detection_folder_aiversion_final_deforestation_tile, exist_ok=True)

    orbit_directions = os.listdir(detection_folder_aiversion_reclassified_sieved_tile)
    if len(orbit_directions) == 1:
        detection_folder_aiversion_reclassified_sieved_tile_orbit = detection_folder_aiversion_reclassified_sieved_tile.joinpath(orbit_directions[0])
        raster_files = os.listdir(detection_folder_aiversion_reclassified_sieved_tile_orbit)
        for raster_file_item in raster_files:
            raster_file_item_path = detection_folder_aiversion_reclassified_sieved_tile_orbit.joinpath(raster_file_item)
            time_point = raster_file_item_path.name.split('_')[2]
            detection_folder_aiversion_final_deforestation_tile_raster = detection_folder_aiversion_final_deforestation_tile.joinpath(f"{tile_item}_{time_point}_CLASS.tif")
            shutil.copy(raster_file_item_path, detection_folder_aiversion_final_deforestation_tile_raster)

    else:
        detection_folder_aiversion_reclassified_sieved_tile_orbit0 = detection_folder_aiversion_reclassified_sieved_tile.joinpath(
            orbit_directions[0])
        detection_folder_aiversion_reclassified_sieved_tile_orbit1 = detection_folder_aiversion_reclassified_sieved_tile.joinpath(
            orbit_directions[1])


        # Get the list of files in each folder
        files1 = [f for f in os.listdir(detection_folder_aiversion_reclassified_sieved_tile_orbit0) if f.endswith('.tif')]  # Assuming raster files are .tif
        files2 = [f for f in os.listdir(detection_folder_aiversion_reclassified_sieved_tile_orbit1) if f.endswith('.tif')]

        # Create a dictionary to store files by date from folder1
        files_dict1 = {extract_date(f): f for f in files1}
        files_dict2 = {extract_date(f): f for f in files2}

        # Create a list of all unique dates from both dictionaries
        all_dates = sorted(set(files_dict1.keys()).union(set(files_dict2.keys())))

        # Create a list of tuples with matching files or None if not present
        matched_tuples = [(files_dict1.get(date), files_dict2.get(date)) for date in all_dates]

        raster_template = None
        # Iterate through the matched tuples
        for f1, f2 in matched_tuples:
            if f1 is not None:
                file1_path = Path(os.path.join(detection_folder_aiversion_reclassified_sieved_tile_orbit0, f1))
                raster_template = file1_path
                time_point = f1.split('_')[2]
            else:
                file1_path = None

            if f2 is not None:
                file2_path = Path(os.path.join(detection_folder_aiversion_reclassified_sieved_tile_orbit1, f2))
                raster_template = file2_path
                time_point = f2.split('_')[2]
            else:
                file2_path = None



            # Open the first file and read as an array (if present)
            if file1_path is not None:
                dataset1 = gdal.Open(str(file1_path))
                dataset1_array = dataset1.GetRasterBand(1).ReadAsArray()
            else:
                dataset1_array = None

            # Open the second file and read as an array (if present)
            if file2_path is not None:
                dataset2 = gdal.Open(str(file2_path))
                dataset2_array = dataset2.GetRasterBand(1).ReadAsArray()
            else:
                dataset2_array = None

            # Perform logical OR operation between the arrays
            if dataset1_array is not None and dataset2_array is not None:
                combined_array = np.logical_or(dataset1_array, dataset2_array)
            elif dataset1_array is not None:  # If only dataset1 is available
                combined_array = dataset1_array
            elif dataset2_array is not None:  # If only dataset2 is available
                combined_array = dataset2_array
            else:
                combined_array = None  # Both arrays are None

            print(f"File from folder1: {file1_path}, File from folder2: {file2_path}")

            detection_folder_aiversion_final_deforestation_tile_raster = detection_folder_aiversion_final_deforestation_tile.joinpath(f"{tile_item}_{time_point}_CLASS.tif")
            save_raster_template(raster_template, detection_folder_aiversion_final_deforestation_tile_raster,  combined_array,
                                 GDT_Byte
                                 )