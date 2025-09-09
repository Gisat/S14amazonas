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
def create_merged_filepath(output_path, raster1, raster2, bounds, pixel_width, width, height):
    output_inter_path = output_path.parent.joinpath(output_path.name.replace(".tif", "_inter.tif"))

    args = ["gdal_merge.py",
            "-separate",
            "-o", str(output_inter_path),
            "-ps", str(pixel_width), str(pixel_width),
            "-ul_lr", str(bounds[0]), str(bounds[3]), str(bounds[2]), str(bounds[1]),
            "-of", "GTiff",
            "-ot", "Float32",
            "-co", "compress=DEFLATE"]
    args = args + [str(raster1)] + [str(raster2)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")

    args = ["gdalwarp",
            "-ts", str(width), str(height),
            "-of", "GTiff",
            "-co", "compress=DEFLATE"]
    args = args + [str(output_inter_path)] + [str(output_path)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")

def create_mask(tile_deforestation_mask, base_lc_tile_filepath, tile_prob_mask):
    dataset = gdal.Open(str(base_lc_tile_filepath))
    jaxa_mask_array = dataset.GetRasterBand(1).ReadAsArray()
    # Inverting 0 and 1 in the numpy array
    inverted_array = np.where(jaxa_mask_array == 0, 1, 0)

    prob_dataset = gdal.Open(str(tile_prob_mask))
    prob_mask_array = prob_dataset.GetRasterBand(1).ReadAsArray()

    final_binary_array = np.logical_or(inverted_array, prob_mask_array).astype(np.uint8)

    inverted_array = np.where(final_binary_array == 0, 1, 0)

    save_raster_template(base_lc_tile_filepath, tile_deforestation_mask, inverted_array, GDT_Byte)


def create_large_sieved(file1_path, file1_path_sieved, size_threshold=5000):
    # Construct the gdal_sieve.py command
    command = [
        'gdal_sieve.py',
        '-st', str(size_threshold),
        '-8',  # 8-connected neighborhood
        '-nomask',  # No mask applied
        '-of', 'GTiff',  # Output format as GeoTIFF
        str(file1_path),  # Input raster file
        str(file1_path_sieved)  # Output raster file
    ]

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Sieve operation completed successfully. Output saved to '{file1_path_sieved}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error during sieve operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_filtered_detection(file1_path, file1_path_sieved, file1_filtered_path):
    dataset1 = gdal.Open(str(file1_path))
    dataset1_array = dataset1.GetRasterBand(1).ReadAsArray()

    dataset2 = gdal.Open(str(file1_path_sieved))
    dataset2_array = dataset2.GetRasterBand(1).ReadAsArray()

    # Create third array with 1s where array2 is 1 and array1 is 0
    array3 = np.where((dataset1_array == 1) & (dataset2_array == 0), 1, 0)

    save_raster_template(file1_path, file1_filtered_path, array3, GDT_Byte)



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

    baselc_folder = support_data.joinpath(f"base_jaxa_worldcover")
    base_lc_tile_filepath = baselc_folder.joinpath(f"{tile_item}_MASK.tif")

    tile_prob_mask = prob_var_mask_folder.joinpath(f"prob_{tile_item}_mask_{var_threshold_name_suffix}_reprojected.tif")
    if not tile_prob_mask.exists(): raise Exception(f"{tile_prob_mask} doesnt exists")

    tile_prob_prediction_mask = prob_var_mask_prediction_folder.joinpath(f"prob_{tile_item}_mask_{var_threshold_name_suffix}_reprojected.tif")
    if not tile_prob_prediction_mask.exists(): raise Exception(f"{tile_prob_prediction_mask} doesnt exists")




    jaxa_tile_ds = gdal.Open(str(base_lc_tile_filepath))
    jaxa_tile_array = jaxa_tile_ds.GetRasterBand(1).ReadAsArray()
    jaxa_tile_array_inverted = 1 - jaxa_tile_array

    prob_mask_ds = gdal.Open(str(tile_prob_mask))
    prob_mask_array = prob_mask_ds.GetRasterBand(1).ReadAsArray()

    prob_mask_prediction_ds = gdal.Open(str(tile_prob_prediction_mask))
    prob_mask_prediction_array = prob_mask_prediction_ds.GetRasterBand(1).ReadAsArray()



    final_mask =  np.logical_or(jaxa_tile_array_inverted, prob_mask_array).astype(int)
    output_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model_version}/deforestation/{tile_item}_mask.tif'
    save_raster_template(tile_prob_prediction_mask, output_path,
                         final_mask, data_type=GDT_Byte)


    prob_mask_prediction_array[final_mask == 1] = 0

    prob_mask_prediction_array[prob_mask_prediction_array == 1] = 2

    output_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model_version}/deforestation/{tile_item}_all_CONFIDENT.tif'
    save_raster_template(tile_prob_prediction_mask, output_path,
                         prob_mask_prediction_array, data_type=GDT_Byte)



