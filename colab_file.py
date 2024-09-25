from pathlib import Path
import os, requests
import subprocess
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte
import numpy as np
from datetime import datetime, timedelta, date
import math
from tondortools.tool import read_raster_info
import copy

tile_list = ['21LYG', '18LVQ', '18LVR', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '21LYH', '22MBT', '22MGB']
year = [2019, 2020]
time_window = 60
acq_freq = 12
number_bands = 3
count_threshold = 1000
block_size = 256
min_labelpixel = 25
use_mcd = True
stack_training_in_one = True

amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
output_folder_multiband_mosaic = output_folder.joinpath('Multiband_mosaic')
########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir")
support_data = amazonas_root_folder.joinpath("support_data")
########################################################################################################################
########################################################################################################################
training_folder = amazonas_root_folder.joinpath('training')
os.makedirs(training_folder, exist_ok=True)
training_folder_label = training_folder.joinpath('label')
os.makedirs(training_folder_label, exist_ok=True)
training_folder_data = training_folder.joinpath('data')
os.makedirs(training_folder_data, exist_ok=True)
detection_mosaic_pair = dict()

for tile_item in tile_list:
    print(f"---- {tile_item} ----")
    output_folder_multiband_mosaic_tile = output_folder_multiband_mosaic.joinpath(tile_item)
    orbit_directions = os.listdir(output_folder_multiband_mosaic_tile)
    for orbit_direction in orbit_directions:
        output_folder_multiband_mosaic_tile_orbit = output_folder_multiband_mosaic_tile.joinpath(orbit_direction)
        if not output_folder_multiband_mosaic_tile_orbit.is_dir() or '.txt' in orbit_direction: continue

        ########################################################################################################################
        if not use_mcd:
            gfw_folder_root = amazonas_root_folder.joinpath("Detections", "GFW")
            gfw_data = work_dir.joinpath("gfw_data")

            gfw_folder_tile_folder = gfw_folder_root.joinpath(tile_item)
            gfw_folder_tile_raster = gfw_folder_tile_folder.joinpath(f"{tile_item}.tif")
            gfw_folder_tile_folder_label = gfw_folder_tile_folder.joinpath("label")
            gfw_folder_tile_folder_renamed_label = gfw_folder_tile_folder.joinpath("label_renamed")
            detection_files = os.listdir(gfw_folder_tile_folder_renamed_label)
            detection_folder = gfw_folder_tile_folder_renamed_label
        else:
            mcd_folder_tile_folder = amazonas_root_folder.joinpath("output", "mcd_detection", f"{tile_item}")
            detection_files = os.listdir(mcd_folder_tile_folder)
            detection_folder = mcd_folder_tile_folder
        ########################################################################################################################
        mosaic_files = os.listdir(output_folder_multiband_mosaic_tile_orbit)

        for year_item in year:
            print(f"-- {year_item} -- {tile_item} -- {orbit_direction}")

            for detection_file in sorted(detection_files):
                detection_filepath = detection_folder.joinpath(detection_file)

                if not use_mcd:
                    gfw_file_count = int(detection_file.split('_')[-1].split('.')[0])
                    if not f"_{year_item}-" in detection_file or not gfw_file_count > count_threshold: continue
                    gfw_file_date_str = detection_file.split('_')[1]
                    acq_date = datetime.strptime(gfw_file_date_str, "%Y-%m-%d")
                else:
                    if not "_s6-8LT_INT_C" in detection_file or not f"_{year_item}" in detection_file: continue
                    mcd_file_date_str = detection_file.split('_')[2]
                    acq_date = datetime.strptime(mcd_file_date_str, "%Y%m%d")

                mosaic_file_list = []

                for mosaic_file in mosaic_files:
                    if not Path(mosaic_file).name.endswith('.tif'): continue
                    mosaic_date_str = mosaic_file.split('_')[-1].split('.')[0]
                    mosaic_date = datetime.strptime(mosaic_date_str, "%Y%m%d")
                    if (mosaic_date <= acq_date) and (mosaic_date > acq_date - timedelta(days=time_window)):
                        mosaic_file_list.append(output_folder_multiband_mosaic_tile_orbit.joinpath(mosaic_file))
                if len(mosaic_file_list) == math.floor(time_window/acq_freq):
                    detection_mosaic_pair[detection_filepath] = sorted(mosaic_file_list)

    print(f"detection mosaic pair count {len(detection_mosaic_pair)}")

data_list = []
label_list = []

pair_count = 0
count = 0
for detection_file, raster_list in detection_mosaic_pair.items():

    pair_count += 1
    if pair_count > 2:
        continue
    print(f"pair count {pair_count}")

    print(f"pair raster len {len(raster_list)}")

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands,
     imagery_extent_box) = read_raster_info(raster_list[0])

    for x_block in range(0, RasterXSize + 1, block_size):
        for y_block in range(0, RasterYSize + 1, block_size):


            chunk_list = []

            array_bands = np.zeros((block_size, block_size))
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

            label_dataset = gdal.Open(str(detection_file))
            label_chunk = label_dataset.ReadAsArray(x_block, y_block, cols, rows)

            if stack_training_in_one:
                chunk_list_array = np.array(chunk_list)
                print(chunk_list_array.shape)
                stacked_data = np.transpose(chunk_list_array, (1, 2, 0, 3)).reshape(256, 256, 15)
            else:
                chunk_list_array = np.array(chunk_list)

            num_non_zero = np.count_nonzero(label_chunk)
            if not num_non_zero > min_labelpixel:
                save_np = False

            label_mask = copy.copy(label_chunk)
            label_mask[label_mask > 0] = 1

            if save_np:
                print(f"shape {chunk_list_array.shape}")
                if stack_training_in_one:
                    print("chunk")
                    data_list.append(stacked_data)
                else:
                    data_list.append(chunk_list_array)

                label_list.append(label_mask)

            count += 1
print(f" number of chunks {len(data_list)} {len(label_list)}--")
