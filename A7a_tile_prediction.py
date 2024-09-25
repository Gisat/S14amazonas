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

tiles = ['18LVQ', '18LVR', '20LLQ', '18LWR', '18NYH', '18NXH', '20LLP', '18NXG', '20LMP', '20LMQ', '20NQG', '21LYG',
         '20NQF', '20NRG', '21LYH', '22MBT', '22MGB']
year = 2017
number_bands = 3
block_size = 256
inner_buffer = 16
cutoff_prob = 0.5
model_version = "build_vgg16_segmentation_batchingestion_thirdrun"
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
work_dir = amazonas_root_folder.joinpath('work_dir')
########################################################################################################################
model_folder = amazonas_root_folder.joinpath("ref_model")
model_filepath = model_folder.joinpath(f"model_{model_version}.h5")
loaded_model = tf.keras.models.load_model(model_filepath)

########################################################################################################################
support_data = amazonas_root_folder.joinpath("support_data")
merged_sar_ref_folder = support_data.joinpath(f"ref_worldcover_sarmosaic")
########################################################################################################################
predicted_baselc_folder = support_data.joinpath(f"base_worldcover_prediction")
os.makedirs(predicted_baselc_folder, exist_ok=True)


########################################################################################################################
def process_window(chunk, loaded_model):
    transposed_chunk = chunk.transpose(1, 2, 0)
    reshaped_data = np.expand_dims(transposed_chunk, axis=0)
    result = loaded_model.predict(reshaped_data)
    result = result[0]
    return np.squeeze(result)


moving_window_size = block_size - (inner_buffer * 2)

for tile_item in tiles:

    predicted_baselc_tile_prob_path = predicted_baselc_folder.joinpath(f"{tile_item}_BASELC_{year}_PROB_wooverlap.tif")
    predicted_baselc_tile_class_path = predicted_baselc_folder.joinpath(f"{tile_item}_BASELC_{year}_CLASS.tif")

    tile_prediction_year_mosaic_tif = merged_sar_ref_folder.joinpath(f"{tile_item}_BAC_MERGED_{year}.tif")
    dataset = gdal.Open(str(tile_prediction_year_mosaic_tif))
    cols, rows = dataset.RasterXSize, dataset.RasterYSize

    prediction_class = np.zeros((rows, cols), dtype=np.byte)
    prediction_prob = np.zeros((rows, cols), dtype=np.float16)

    if not tile_prediction_year_mosaic_tif.exists(): continue

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands,
     imagery_extent_box) = read_raster_info(str(tile_prediction_year_mosaic_tif))

    window_size = block_size
    mosaic_workdir = work_dir.joinpath(tile_prediction_year_mosaic_tif.stem)
    os.makedirs(mosaic_workdir, exist_ok=True)

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

            dataset = gdal.Open(str(tile_prediction_year_mosaic_tif))
            chunk = dataset.ReadAsArray(x_block, y_block, cols, rows)

            # Transpose the data
            transposed_chunk = chunk.transpose(1, 2, 0)

            if not save_np: continue
            reshaped_data = np.expand_dims(transposed_chunk, axis=0)
            result = loaded_model.predict(reshaped_data)
            result = result[0]
            result_transpose = result.transpose(2, 0, 1)


            seg_result_block = mosaic_workdir.joinpath(f"{model_version}_{x_block}_{y_block}.tif")

            # tile_ulx and tile_uly are the absolute coordinates of the upper left corner of the tile.
            tile_ulx = xmin + x_block*pixel_width
            tile_uly = ymax - y_block*pixel_width

            save_raster(result_transpose, seg_result_block, 'GTiff', epsg,
                        tile_ulx, tile_uly,
                        pixel_width, GDT_Float32)
            result_block_list.append(seg_result_block)

    mosaic_tifs(result_block_list, predicted_baselc_tile_prob_path)

