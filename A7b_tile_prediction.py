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


tiles = ['18LVQ',  '18LVR',  '20LLQ', '18LWR', '18NYH', '18NXH', '20LLP', '18NXG', '20LMP', '20LMQ', '20NQG', '21LYG', '20NQF', '20NRG', '21LYH', '22MBT',  '22MGB']
year = 2017
number_bands = 3
block_size = 256
inner_buffer = 16
cutoff_prob = 0.5
model_version = "build_vgg16_segmentation_batchingestion_thirdrun"
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
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


moving_window_size = block_size - (inner_buffer*2)

for tile_item in tiles:

    predicted_baselc_tile_prob_path = predicted_baselc_folder.joinpath(f"{tile_item}_BASELC_{year}_PROB.tif")
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
    # first lets do the four corners
    window = dataset.ReadAsArray(0, 0, window_size, window_size)
    processed_window = process_window(window, loaded_model)
    prediction_prob[0:window_size, 0: window_size] = processed_window

    window = dataset.ReadAsArray((cols - window_size), 0, window_size, window_size)
    processed_window = process_window(window, loaded_model)
    prediction_prob[0:window_size, (cols - window_size): cols] = processed_window

    window = dataset.ReadAsArray(0, (rows - window_size), window_size, window_size)
    processed_window = process_window(window, loaded_model)
    prediction_prob[(rows - window_size): rows, 0: window_size] = processed_window

    window = dataset.ReadAsArray((cols - window_size), (rows - window_size), window_size, window_size)
    processed_window = process_window(window, loaded_model)
    prediction_prob[(rows - window_size): rows, (cols - window_size): cols] = processed_window

    inner_buffer_window = window_size - inner_buffer
    # Process edges
    for edge in ['top', 'bottom', 'left', 'right']:
        if edge in ['top', 'bottom']:
            for start_col in range(inner_buffer, cols, moving_window_size):

                left_limit = start_col - inner_buffer
                right_limit = left_limit + window_size
                if right_limit > cols: continue

                if edge == 'top':
                    window = dataset.ReadAsArray(left_limit, 0, window_size, window_size)
                else:
                    window = dataset.ReadAsArray(left_limit, rows - window_size, window_size, window_size)

                processed_window = process_window(window, loaded_model)

                if edge == 'top':
                    prediction_prob[0:window_size, start_col:start_col + moving_window_size] = processed_window[:,
                                                                                               inner_buffer:-inner_buffer]
                else:
                    prediction_prob[rows - window_size:rows,
                    start_col:start_col + moving_window_size] = processed_window[:,
                                                                inner_buffer:-inner_buffer]

        else:  # 'left' or 'right'
            for start_row in range(inner_buffer, rows, moving_window_size):

                upper_limit = start_row - inner_buffer
                lower_limit = upper_limit + window_size
                if lower_limit > rows: continue

                if edge == 'left':
                    window = dataset.ReadAsArray(0, upper_limit, window_size, window_size)
                else:  # right
                    window = dataset.ReadAsArray(cols - window_size, upper_limit, window_size, window_size)

                processed_window = process_window(window, loaded_model)

                if edge == 'left':
                    prediction_prob[start_row:start_row + moving_window_size, 0:window_size] = processed_window[
                                                                                               inner_buffer:-inner_buffer,
                                                                                               :]
                else:
                    prediction_prob[start_row:start_row + moving_window_size,
                    cols - window_size:cols] = processed_window[inner_buffer:-inner_buffer, :]

    for x_block in range(inner_buffer, RasterXSize + 1, moving_window_size):
        for y_block in range(inner_buffer, RasterYSize + 1, moving_window_size):

            x_block_prediction_window = x_block - inner_buffer
            y_block_prediction_window = y_block - inner_buffer

            do_pred = True
            if x_block_prediction_window + block_size < RasterXSize:
                cols = block_size
            else:
                do_pred = False

            if y_block_prediction_window + block_size < RasterYSize:
                rows = block_size
            else:
                do_pred = False

            if do_pred:
                chunk = dataset.ReadAsArray(x_block_prediction_window, y_block_prediction_window, block_size,
                                            block_size)

                transposed_chunk = chunk.transpose(1, 2, 0)
                reshaped_data = np.expand_dims(transposed_chunk, axis=0)
                result = loaded_model.predict(reshaped_data)
                result = result[0]
                result_sq = np.squeeze(result)
                inner_box = result_sq[inner_buffer:-inner_buffer, inner_buffer:-inner_buffer]

                prediction_prob[y_block: (y_block + moving_window_size),
                x_block: (x_block + moving_window_size)] = inner_box


    save_raster_template(str(tile_prediction_year_mosaic_tif), str(predicted_baselc_tile_prob_path), prediction_prob, GDT_Float32, nodata_value=None)

    prediction_class = (prediction_prob > cutoff_prob).astype(np.byte)
    save_raster_template(str(tile_prediction_year_mosaic_tif), str(predicted_baselc_tile_class_path), prediction_class, GDT_Byte, nodata_value=None)

