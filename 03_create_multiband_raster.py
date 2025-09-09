import os
import shutil

from osgeo import gdal
from collections import defaultdict
from tondortools.tool import save_raster_template
from gdalconst import GDT_Float32, GDT_Int16, GDT_Byte, GDT_UInt16
from pathlib import Path
import numpy as np
from multiprocessing import Pool
MIN_VALUE = 0
MAX_VALUE = 1

def cutoff_minmax_scale(numpy_array):
    lower_cutoff = -30
    upper_cutoff = 0
    # Step 1: Set all values less than -30 to 0
    numpy_array[numpy_array <= lower_cutoff] = lower_cutoff
    numpy_array[numpy_array > upper_cutoff] = upper_cutoff

    # Step 2: Use min-max scaling with min as -30 and max as 0
    min_val = lower_cutoff
    max_val = upper_cutoff
    scaled_arr = (((numpy_array - min_val) * (MAX_VALUE - MIN_VALUE))/(max_val - min_val))  + MIN_VALUE
    return scaled_arr

def do_merging(input_data):
    tile_item = input_data[0]

    archive_mosaic_tile_folder = archive_mosaic_folder.joinpath(tile_item)
    archive_multiband_mosaic_tile_folder = archive_multiband_mosaic_folder.joinpath(tile_item)
    os.makedirs(archive_multiband_mosaic_tile_folder, exist_ok=True)

    orbit_directions = os.listdir(archive_mosaic_tile_folder)
    for orbit_direction in orbit_directions:
        archive_multiband_mosaic_tile_orbit_folder = archive_multiband_mosaic_tile_folder.joinpath(orbit_direction)
        os.makedirs(archive_multiband_mosaic_tile_orbit_folder, exist_ok=True)

        # Folder path
        folder_path = archive_mosaic_tile_folder.joinpath(orbit_direction)
        if not folder_path.is_dir(): continue

        # Group files by date
        file_groups = defaultdict(dict)

        for file_name in os.listdir(folder_path):
            if not file_name.endswith('filled.tif'):
                continue

            parts = file_name.split('_')
            date = parts[3]
            polarization = parts[2]

            file_groups[date][polarization] = os.path.join(folder_path, file_name)

        # Process each group
        for date, files in file_groups.items():
            if not str(date).startswith('2019'): continue
            if "VV" in files and "VH" in files:
                vh_file = files["VH"]
                vv_file = files["VV"]

                merged_file_archive_path = os.path.join(archive_multiband_mosaic_tile_orbit_folder,
                                                        f"{tile_item}_BAC_MERGED_{date}.tif")
                if Path(merged_file_archive_path).exists():
                    Path(merged_file_archive_path).unlink()
                # Calculate difference
                vh_raster = gdal.Open(vh_file, gdal.GA_ReadOnly)
                vv_raster = gdal.Open(vv_file, gdal.GA_ReadOnly)

                vh_band = vh_raster.GetRasterBand(1).ReadAsArray()
                vv_band = vv_raster.GetRasterBand(1).ReadAsArray()

                difference = vh_band - vv_band

                vh_band_scaled = cutoff_minmax_scale(vh_band)
                vv_band_scaled = cutoff_minmax_scale(vv_band)
                diff_scaled = cutoff_minmax_scale(difference)

                # Merge to a three-band raster
                work_dir_file = Path(work_dir).joinpath(f"{tile_item}_BAC_MERGED_{date}")
                os.makedirs(work_dir_file, exist_ok=True)
                merged_file_local_path = work_dir_file.joinpath(f"{tile_item}_BAC_MERGED_{date}.tif")

                three_band_array = np.stack((vh_band_scaled, vv_band_scaled, diff_scaled))
                three_band_array = three_band_array
                save_raster_template(vh_file, merged_file_local_path, three_band_array, GDT_Float32, 0)
                shutil.copy(merged_file_local_path, merged_file_archive_path)
                shutil.rmtree(work_dir_file)
                print(f"{tile_item} -- {work_dir_file.name}")


######################################################
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
#tile_list = ['18LVQ', '18LVR', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '21LYH', '22MBT', '22MGB']
tile_list = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
process_cpu_count = 5
######################################################
archive_folder = amazonas_root_folder.joinpath('output')
archive_mosaic_folder = archive_folder.joinpath("Mosaic")

archive_multiband_mosaic_folder = archive_folder.joinpath("Multiband_mosaic")
os.makedirs(archive_multiband_mosaic_folder, exist_ok=True)
work_dir = Path("/home/eouser/userdoc/sar_workdir")
######################################################

subtasks = []
for tile_item in tile_list:
    args = [tile_item]
    subtasks.append(args)

if len(subtasks) > 0:
    print(f"Doing interpolating for {len(subtasks)}")
    p = Pool(process_cpu_count)
    p.map(do_merging, tuple(subtasks))
    p.close()  # Close the pool
    p.join()
