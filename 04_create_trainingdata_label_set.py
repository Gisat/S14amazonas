import shutil
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

tile_list = ['18LVQ',  '18LVR',  '20LLQ',  '18LWR',  '18NYH',  '18NXH',  '20LLP',  '18NXG',  '20LMP',  '20LMQ',  '20NQG',  '21LYG',  '20NQF',  '20NRG',  '21LYH',  '22MBT',  '22MGB']
tile_list = ['18LVQ']
year = [2021, 2023]
time_window = 60
acq_freq = 12
number_bands = 3
count_threshold = 1000
block_size = 256
min_labelpixel = 25
use_mcd = True
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
output_folder_multiband_mosaic = output_folder.joinpath('Multiband_mosaic_from_2020')
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
                    if (mosaic_date < acq_date + timedelta(days=time_window)) and (mosaic_date >= acq_date - timedelta(days=time_window)):
                        mosaic_file_list.append(output_folder_multiband_mosaic_tile_orbit.joinpath(mosaic_file))
                if len(mosaic_file_list) == math.floor(2*time_window/acq_freq):
                    detection_mosaic_pair[detection_filepath] = {"raster_list":sorted(mosaic_file_list),
                                                                 "tile": tile_item}
        print(f"-- done collecting MCD mosaic set {tile_item}")
    print(f"detection mosaic pair count {len(detection_mosaic_pair)}")

data_list = []
label_list = []


count = 1
for detection_file, tile_raster_list_dict in detection_mosaic_pair.items():

    raster_list = tile_raster_list_dict["raster_list"]
    tile_name = tile_raster_list_dict["tile"]

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(raster_list[0])

    for x_block in range(0, RasterXSize+1, block_size):
        for y_block in range(0, RasterYSize+1, block_size):

            chunk_list = []

            array_bands = np.zeros((block_size, block_size))
            save_np = True

            if x_block + block_size < RasterXSize:
                cols = block_size
            else:
                x_block = RasterXSize - block_size
                save_np = True

            if y_block + block_size < RasterYSize:
                rows = block_size
            else:
                y_block = RasterYSize - block_size
                save_np = True

            for raster_list_item in raster_list:
                dataset = gdal.Open(str(raster_list_item))
                chunk = dataset.ReadAsArray(x_block, y_block, block_size, block_size)

                # Transpose the data
                transposed_chunk = chunk.transpose(1, 2, 0)
                chunk_list.append(transposed_chunk)

            label_dataset = gdal.Open(str(detection_file))
            label_chunk = label_dataset.ReadAsArray(x_block, y_block, block_size, block_size)
            chunk_list_array = np.array(chunk_list)

            num_non_zero = np.count_nonzero(label_chunk)
            label_mask = copy.copy(label_chunk)


            if save_np:
                training_folder_data_npy = training_folder_data.joinpath(f"data_{count}.npy")
                np.save(training_folder_data_npy, chunk_list_array)

                training_folder_label_npy = training_folder_label.joinpath(f"label_{count}.npy")
                np.save(training_folder_label_npy, label_mask)

                if training_folder_data_npy.exists() and training_folder_label_npy.exists():
                    if count % 100 == 0:
                        print(f"{tile_name} -- {count} with {num_non_zero} {Path(detection_file).name}--")
                else:
                    raise Exception(f"something is wrong")

                count += 1
