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
from tondortools.tool import read_raster_info, mosaic_tifs, save_raster_template
from osgeo import gdal, gdal_array


def save_raster(output_path, array, geotransform, projection, datatype = gdal.GDT_Float32):
    driver = gdal.GetDriverByName('GTiff')
    y_size, x_size = array.shape
    dataset = driver.Create(str(output_path), x_size, y_size, 1, datatype)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()

def sum_rasters(raster_files):
    # Open the first raster to get the dimensions and georeference information
    raster = gdal.Open(str(raster_files[0]))
    geotransform = raster.GetGeoTransform()
    projection = raster.GetProjection()
    band = raster.GetRasterBand(1)
    data_type = band.DataType
    x_size = raster.RasterXSize
    y_size = raster.RasterYSize

    # Initialize an array to hold the sum of all rasters
    sum_array = np.zeros((y_size, x_size), dtype=gdal_array.GDALTypeCodeToNumericTypeCode(data_type))

    # Sum all rasters
    for raster_file in raster_files:
        print(f"reading {raster_file}")
        raster = gdal.Open(str(raster_file))
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        sum_array += array
        raster = None
        band = None

    return sum_array, geotransform, projection


def std_rasters(raster_files):
    # Open the first raster to get the dimensions and georeference information
    raster = gdal.Open(str(raster_files[0]))
    geotransform = raster.GetGeoTransform()
    projection = raster.GetProjection()
    band = raster.GetRasterBand(1)
    data_type = band.DataType
    x_size = raster.RasterXSize
    y_size = raster.RasterYSize

    # Initialize arrays to hold the sum and sum of squares
    sum_array = np.zeros((y_size, x_size), dtype=gdal_array.GDALTypeCodeToNumericTypeCode(data_type))
    sum_squares_array = np.zeros((y_size, x_size), dtype=gdal_array.GDALTypeCodeToNumericTypeCode(data_type))

    # Sum all rasters and sum of squares for each pixel
    for raster_file in raster_files:
        print(f"reading {raster_file}")
        raster = gdal.Open(str(raster_file))
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()

        # Accumulate the sum and sum of squares
        sum_array += array
        sum_squares_array += array**2

        raster = None
        band = None

    # Calculate the mean and standard deviation for each pixel
    num_rasters = len(raster_files)
    mean_array = sum_array / num_rasters
    variance_array = (sum_squares_array / num_rasters) - (mean_array**2)
    std_array = np.sqrt(variance_array)

    return std_array, variance_array, geotransform, projection

tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
var_cutoff = 0.01
filename_suffix = "pt01"
model_version = 'best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion = detection_folder.joinpath(f'{model_version}')
########################################################################################################################
support_data = amazonas_root_folder.joinpath("support_data")
prob_pixel_info = support_data.joinpath(f"tile_prob_pixel_info")
os.makedirs(prob_pixel_info, exist_ok=True)
########################################################################################################################


for tile_item in tiles:

    detection_folder_aiversion_tile = detection_folder.joinpath(f'{model_version}', 'training_years_prediction', tile_item)

    orbit_directions = os.listdir(detection_folder_aiversion_tile)
    for orbit_direction_item in orbit_directions:

        orbit_detections_list = []

        if orbit_direction_item not in ['ascending', 'descending']:continue

        detection_folder_aiversion_tile_orbit = detection_folder_aiversion_tile.joinpath(orbit_direction_item)

        output_mean_path = prob_pixel_info.joinpath("prob_std_mean", f"{tile_item}_{orbit_direction_item}_mean.tif")
        output_std_path = prob_pixel_info.joinpath("prob_std_mean", f"{tile_item}_{orbit_direction_item}_std.tif")
        output_var_path = prob_pixel_info.joinpath("prob_std_mean", f"{tile_item}_{orbit_direction_item}_var.tif")

        if not output_mean_path.exists() or not output_std_path.exists() or not output_var_path.exists():
            folder_files_list = os.listdir(detection_folder_aiversion_tile_orbit)
            for folder_files_list_item in folder_files_list:
                if not "_PROB" in folder_files_list_item: continue

                orbit_ditection_item = detection_folder_aiversion_tile_orbit.joinpath(folder_files_list_item)
                orbit_detections_list.append(orbit_ditection_item)


            sum_array, geotransform, projection = sum_rasters(orbit_detections_list)

            number_files = len(orbit_detections_list)
            mean_array = np.divide(sum_array, number_files)
            save_raster(output_mean_path, mean_array, geotransform, projection)

            std_array, variance_array, geotransform, projection = std_rasters(orbit_detections_list)
            save_raster(output_std_path, std_array, geotransform, projection)
            save_raster(output_var_path, variance_array, geotransform, projection)

            print(orbit_detections_list)

    # Initialize the variable that will store the result of the binary array
    final_binary_array = None
    for orbit_direction_item in orbit_directions:
        if orbit_direction_item not in ['ascending', 'descending']: continue

        output_var_path = prob_pixel_info.joinpath("prob_std_mean", f"{tile_item}_{orbit_direction_item}_var.tif")

        # Open the raster file
        raster = gdal.Open(str(output_var_path))

        geotransform = raster.GetGeoTransform()
        projection = raster.GetProjection()

        # Get the first band (assuming you want to work with the first band)
        band = raster.GetRasterBand(1)

        # Read the band as an array
        array = band.ReadAsArray()

        # Create a binary array where pixels with a value above 0.15 are marked as 1, and others as 0
        binary_array = np.where(array > var_cutoff, 1, 0)

        # If it's the first iteration, initialize the final binary array
        if final_binary_array is None:
            final_binary_array = binary_array
        else:
            # Perform a logical OR operation to keep track of any pixel that was above 0.15 in at least one file
            final_binary_array = np.logical_or(final_binary_array, binary_array).astype(np.uint8)

    output_mask_path = prob_pixel_info.joinpath(f"prob_{tile_item}_mask_{filename_suffix}.tif")
    save_raster(output_mask_path, final_binary_array, geotransform, projection, gdal.GDT_Byte)





