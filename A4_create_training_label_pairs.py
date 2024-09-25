from pathlib import Path
import os, requests
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte
import numpy as np
import datetime
import subprocess
import copy
from tondortools.tool import reproject_multibandraster_toextent, read_raster_info, save_raster_template, mosaic_tifs

#######
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")

ref_year = 2021
block_size = 256

support_data = amazonas_root_folder.joinpath("support_data")
merged_sar_ref_folder = support_data.joinpath(f"merged_sar_ref_worldcover")
ref_sar_data_mosaic_folder = support_data.joinpath("ref_worldcover_sarmosaic")
ref_lc_data_mosaic_folder = support_data.joinpath("2021_ref_worldcover_mosaic")

#/mnt/hddarchive.nfs/amazonas_dir/support_data/2021_ref_worldcover_mosaic
if not ref_sar_data_mosaic_folder.exists(): raise Exception(f"{ref_sar_data_mosaic_folder} doesnt exists")
if not ref_lc_data_mosaic_folder.exists(): raise Exception(f"{ref_lc_data_mosaic_folder} doesnt exists")
#######
ref_training_folder = amazonas_root_folder.joinpath("ref_training")
os.makedirs(ref_training_folder, exist_ok=True)

ref_training_data_folder = ref_training_folder.joinpath("data")
ref_training_label_folder = ref_training_folder.joinpath("label")
os.makedirs(ref_training_data_folder, exist_ok=True)
os.makedirs(ref_training_label_folder, exist_ok=True)
#######
work_dir = amazonas_root_folder.joinpath("work_dir", "tmp")
os.makedirs(work_dir, exist_ok=True)

#######
tile_list = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
print(f"len of tiles {len(tile_list)}")

count = 0
for tile_item in tile_list:
    print(f"tile {tile_item}")


    work_dir_tile = work_dir.joinpath(f"data_label_{tile_item}")
    os.makedirs(work_dir_tile, exist_ok=True)

    multiband_path = merged_sar_ref_folder.joinpath(f"{tile_item}_BAC_LC_MERGED_{ref_year}.tif")
    if not multiband_path.exists(): raise Exception(f"{multiband_path} doesnt exists")

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(multiband_path)

    for x_block in range(0, RasterXSize, block_size):
        for y_block in range(0, RasterYSize, block_size):

            count += 1

            array_bands = np.zeros((block_size, block_size))
            save_np = True

            if x_block + block_size < RasterXSize:
                starting_x = x_block
            else:
                starting_x = RasterXSize - block_size - 1

            if y_block + block_size < RasterYSize:
                starting_y = y_block
            else:
                starting_y = RasterYSize - block_size - 1

            dataset = gdal.Open(str(multiband_path))
            chunk = dataset.ReadAsArray(starting_x, starting_y, block_size, block_size)

            sar_array = chunk[:3, :, :]
            label_chunk = chunk[3:, :, :]

            # Transpose the data
            transposed_chunk = sar_array.transpose(1, 2, 0)

            try:
                label_mask = copy.copy(label_chunk)
                label_mask[label_chunk != 10] = 0
                label_mask[label_chunk == 10] = 1
            except Exception as e:
                print(f"{label_mask}")

            if save_np:
                training_folder_data_npy = ref_training_data_folder.joinpath(f"data_{count}.npy")
                np.save(training_folder_data_npy, transposed_chunk)

                training_folder_label_npy = ref_training_label_folder.joinpath(f"label_{count}.npy")
                np.save(training_folder_label_npy, label_mask)

    print(f"---- count {count} ----")