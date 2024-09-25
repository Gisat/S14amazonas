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
from tondortools.tool import read_raster_info, save_raster, mosaic_tifs
import tensorflow as tf


tile = "21LYG"
year_from = 2018
year_to = 2021
acq_freq = 12
number_bands = 3
time_window = 60
block_size = 256
model_version = 'ver7_Segmod'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
model_folder = amazonas_root_folder.joinpath("model")
model_filepath = model_folder.joinpath(f"model_{model_version}.h5")
loaded_model = tf.keras.models.load_model(model_filepath)
########################################################################################################################
#/mnt/hddarchive.nfs/amazonas_dir/support_data/base_worldcover_prediction/18LVQ_BASELC_2017_CLASS.tif
base_worldcover_folder = amazonas_root_folder.joinpath("support_data", "base_worldcover_prediction")
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
output_folder_multiband_mosaic = output_folder.joinpath('Multiband_mosaic')
output_folder_multiband_mosaic_tile = output_folder_multiband_mosaic.joinpath(tile)
output_folder_multiband_mosaic_tile_orbit = output_folder_multiband_mosaic_tile.joinpath('descending')
work_dir = amazonas_root_folder.joinpath('work_dir')
########################################################################################################################
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion = detection_folder.joinpath(f'{model_version}')
os.makedirs(detection_folder_aiversion, exist_ok=True)
########################################################################################################################

mosaic_files = os.listdir(output_folder_multiband_mosaic_tile_orbit)
for mosaic_file in sorted(mosaic_files):
    raster_list = []
    if not f"_{year}" in mosaic_file: continue
    mosaic_file_date_str = mosaic_file.split('_')[-1].split('.')[0]
    mosaic_file_date = datetime.strptime(mosaic_file_date_str, "%Y%m%d")
    for mosaic_file_item in sorted(mosaic_files):
        mosaic_file_date_item_str = mosaic_file_item.split('_')[-1].split('.')[0]
        mosaic_file_item_date = datetime.strptime(mosaic_file_date_item_str, "%Y%m%d")

        if (mosaic_file_item_date <= mosaic_file_date) and (mosaic_file_item_date > mosaic_file_date - timedelta(days=time_window)):
            raster_list.append(Path(output_folder_multiband_mosaic_tile_orbit).joinpath(mosaic_file_item))

    print(f"{mosaic_file} -- > {sorted(raster_list)}")

    mosaic_workdir = work_dir.joinpath(mosaic_file)
    os.makedirs(mosaic_workdir, exist_ok=True)

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands,
     imagery_extent_box) = read_raster_info(raster_list[0])
    result_block_list = []
    for x_block in range(0, RasterXSize+1, block_size):
        for y_block in range(0, RasterYSize+1, block_size):

            chunk_list = []

            save_np = True

            if x_block + block_size < RasterXSize:
                cols = block_size
            else:
                cols = RasterXSize - x_block
                save_np = False

            if y_block + block_size < RasterYSize:
                rows = block_size
            else:
                rows = RasterYSize - y_block
                save_np = False

            for raster_list_item in raster_list:
                dataset = gdal.Open(str(raster_list_item))
                chunk = dataset.ReadAsArray(x_block, y_block, cols, rows)

                # Transpose the data
                transposed_chunk = chunk.transpose(1, 2, 0)
                chunk_list.append(transposed_chunk)

            if not save_np: continue
            chunk_list_array = np.array(chunk_list)
            stacked_data = np.transpose(chunk_list_array, (1, 2, 0, 3)).reshape(256, 256, 15)
            reshaped_data = np.expand_dims(stacked_data, axis=0)
            result = loaded_model.predict(reshaped_data)
            result = result[0]
            result_transpose = result.transpose(2, 0, 1)
            #predicted_single_band = np.argmax(result, axis=-1)
            #segmentation_2D = predicted_single_band[0]
            seg_result_block = mosaic_workdir.joinpath(f"{model_version}_{x_block}_{y_block}.tif")

            # tile_ulx and tile_uly are the absolute coordinates of the upper left corner of the tile.
            tile_ulx = xmin + x_block*pixel_width
            tile_uly = ymax - y_block*pixel_width

            save_raster(result_transpose, seg_result_block, 'GTiff', epsg,
                        tile_ulx, tile_uly,
                        pixel_width, GDT_Float32)
            result_block_list.append(seg_result_block)

    mosaic_detection =  detection_folder_aiversion.joinpath(mosaic_file)
    mosaic_tifs(result_block_list, mosaic_detection)


    print("here")

