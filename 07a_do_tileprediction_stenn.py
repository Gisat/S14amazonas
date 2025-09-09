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

import multiprocessing
from multiprocessing import Pool
CPU_COUNT = int(np.floor(multiprocessing.cpu_count()*.80))

tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
tiles = ['21LYG']

year_from = 2017
year_to = 2018
acq_freq = 12
number_bands = 3
time_window = 60
block_size = 256
inner_buffer = 32
cutoff_prob = 0.1
model_version = 'best_build_stenn_segmentation_batchingestion_smalldataset_weighted_f1score_morelayers'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")

number_of_files = int(np.floor(time_window/acq_freq))
########################################################################################################################
model_folder = amazonas_root_folder.joinpath("model")
model_filepath = model_folder.joinpath(f"model_{model_version}.h5")
# loaded_model = tf.keras.models.load_model(model_filepath)
########################################################################################################################
#/mnt/hddarchive.nfs/amazonas_dir/support_data/base_worldcover_prediction/18LVQ_BASELC_2017_CLASS.tif
base_worldcover_folder = amazonas_root_folder.joinpath("support_data", "base_worldcover_prediction")
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
work_dir = amazonas_root_folder.joinpath('work_dir')


########################################################################################################################
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion_parent = detection_folder.joinpath(f'{model_version}', "training_years_prediction")
os.makedirs(detection_folder_aiversion_parent, exist_ok=True)
########################################################################################################################
def create_window(raster_list, xoff, yoff, x_window_size, y_window_size):
    chunk_list = []
    for raster_list_item in raster_list:
        dataset = gdal.Open(str(raster_list_item))
        chunk = dataset.ReadAsArray(xoff, yoff, x_window_size, y_window_size)

        # Transpose the data
        transposed_chunk = chunk.transpose(1, 2, 0)
        chunk_list.append(transposed_chunk)
    return np.array(chunk_list)

def process_window(chunk, loaded_model):
    stacked_data = np.transpose(chunk, (1, 2, 0, 3)).reshape(256, 256, 15)
    reshaped_data = np.expand_dims(chunk, axis=0)
    result = loaded_model.predict(reshaped_data)
    result = result[0]
    return np.squeeze(result[:, :, 0])


def do_prediction(input_data):
    class_year = input_data[0]
    mosaic_detection_class_path = Path(input_data[1])
    mosaic_detection_prob_path = Path(input_data[2])
    raster_list = input_data[3]
    model_filepath = input_data[4]
    block_size = int(input_data[5])
    inner_buffer = int(input_data[6])
    cutoff_prob = float(input_data[7])

    print(f"{class_year} -- {mosaic_detection_class_path} -- > {sorted(raster_list)}")
    loaded_model = tf.keras.models.load_model(model_filepath)

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands,
     imagery_extent_box) = read_raster_info(raster_list[0])
    cols, rows = RasterXSize, RasterYSize

    prediction_class = np.zeros((rows, cols), dtype=np.byte)
    prediction_prob = np.zeros((rows, cols), dtype=np.float16)

    window_size = block_size
    # first lets do the four corners
    window = create_window(raster_list, 0, 0, window_size, window_size)
    processed_window = process_window(window, loaded_model)
    prediction_prob[0:window_size, 0: window_size] = processed_window

    window = create_window(raster_list, (cols - window_size), 0, window_size, window_size)
    processed_window = process_window(window, loaded_model)
    prediction_prob[0:window_size, (cols - window_size): cols] = processed_window

    window = create_window(raster_list, 0, (rows - window_size), window_size, window_size)
    processed_window = process_window(window, loaded_model)
    prediction_prob[(rows - window_size): rows, 0: window_size] = processed_window

    window = create_window(raster_list, (cols - window_size), (rows - window_size), window_size, window_size)
    processed_window = process_window(window, loaded_model)
    prediction_prob[(rows - window_size): rows, (cols - window_size): cols] = processed_window

    moving_window_size = block_size - (inner_buffer * 2)

    inner_buffer_window = window_size - inner_buffer
    # Process edges
    for edge in ['top', 'bottom', 'left', 'right']:
        if edge in ['top', 'bottom']:
            for start_col in range(inner_buffer, cols, moving_window_size):

                left_limit = start_col - inner_buffer
                right_limit = left_limit + window_size
                if right_limit > cols: continue

                if edge == 'top':
                    window = create_window(raster_list, left_limit, 0, window_size, window_size)
                else:
                    window = create_window(raster_list, left_limit, rows - window_size, window_size, window_size)

                processed_window = process_window(window, loaded_model)

                if edge == 'top':
                    prediction_prob[0:window_size, start_col:start_col + moving_window_size] = processed_window[
                                                                                               :,
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
                    window = create_window(raster_list, 0, upper_limit, window_size, window_size)
                else:  # right
                    window = create_window(raster_list, cols - window_size, upper_limit, window_size, window_size)

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
                window = create_window(raster_list, x_block_prediction_window, y_block_prediction_window,
                                       block_size, block_size)
                processed_window = process_window(window, loaded_model)
                inner_box = processed_window[inner_buffer:-inner_buffer, inner_buffer:-inner_buffer]

                prediction_prob[y_block: (y_block + moving_window_size),
                x_block: (x_block + moving_window_size)] = inner_box

    save_raster_template(str(raster_list[0]), str(mosaic_detection_prob_path), prediction_prob,
                         GDT_Float32, nodata_value=None)

    prediction_class = (prediction_prob > cutoff_prob).astype(np.byte)
    save_raster_template(str(raster_list[0]), str(mosaic_detection_class_path),
                         prediction_class, GDT_Byte, nodata_value=None)


year_list = range(year_from, year_to+1)

for tile_item in tiles:

    detection_folder_aiversion = detection_folder_aiversion_parent.joinpath(tile_item)
    os.makedirs(detection_folder_aiversion, exist_ok=True)

    output_folder_multiband_mosaic = output_folder.joinpath('Multiband_mosaic')
    output_folder_multiband_mosaic_from_2020 = output_folder.joinpath('Multiband_mosaic_from_2020')

    output_folder_multiband_mosaic_tile = output_folder_multiband_mosaic.joinpath(tile_item)
    output_folder_multiband_mosaic_from_2020_tile = output_folder_multiband_mosaic_from_2020.joinpath(tile_item)

    mosaic_filepaths = []
    orbit_directions = os.listdir(output_folder_multiband_mosaic_tile)
    for orbit_direction_item in orbit_directions:
        if orbit_direction_item not in ['ascending', 'descending']:continue

        output_folder_multiband_mosaic_tile_orbit = output_folder_multiband_mosaic_tile.joinpath(orbit_direction_item)
        mosaic_files = os.listdir(output_folder_multiband_mosaic_tile_orbit)
        for mosaic_file_item in sorted(mosaic_files):
            mosaic_filepaths.append(Path(output_folder_multiband_mosaic_tile_orbit).joinpath(mosaic_file_item))

        output_folder_multiband_mosaic_tile_orbit = output_folder_multiband_mosaic_from_2020_tile.joinpath(orbit_direction_item)
        mosaic_files = os.listdir(output_folder_multiband_mosaic_tile_orbit)
        for mosaic_file_item in sorted(mosaic_files):
            mosaic_filepaths.append(Path(output_folder_multiband_mosaic_tile_orbit).joinpath(mosaic_file_item))

        detection_folder_aiversion_orbit = detection_folder_aiversion.joinpath(orbit_direction_item)
        os.makedirs(detection_folder_aiversion_orbit, exist_ok=True)

        prediction_subtasks = []

        for year in year_list:

            for mosaic_index, mosaic_filepath in enumerate(sorted(mosaic_filepaths)):
                mosaic_file = mosaic_filepath.name

                if not mosaic_index > number_of_files: continue

                mosaic_file_date_item_str = mosaic_file.split('_')[-1].split('.')[0]
                mosaic_file_item_date = datetime.strptime(mosaic_file_date_item_str, "%Y%m%d")
                mosaic_detection_prob_path = detection_folder_aiversion_orbit.joinpath(f"{tile_item}_{orbit_direction_item}_{mosaic_file_date_item_str}_PROB.tif")
                mosaic_detection_class_path = detection_folder_aiversion_orbit.joinpath(f"{tile_item}_{orbit_direction_item}_{mosaic_file_date_item_str}_CLASS.tif")

                class_year = mosaic_file_date_item_str[0:4]
                if not str(year) == class_year: continue

                raster_list = mosaic_filepaths[mosaic_index - number_of_files + 1:mosaic_index + 1]
                if not mosaic_detection_class_path.exists() and not mosaic_detection_prob_path.exists():

                    prediction_subtasks_args = [class_year, mosaic_detection_class_path, mosaic_detection_prob_path,
                                                    raster_list, model_filepath, block_size, inner_buffer, cutoff_prob]
                    prediction_subtasks.append(prediction_subtasks_args)

        p = Pool(CPU_COUNT)
        p.map(do_prediction, tuple(prediction_subtasks), chunksize=1)