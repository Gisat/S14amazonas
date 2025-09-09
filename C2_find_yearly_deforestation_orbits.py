import os
import glob
import numpy as np
from osgeo import gdal, gdal_array
from tondortools.tool import save_raster_template
import shutil
import subprocess
from pathlib import Path
from gdalconst import GDT_Byte, GDT_Float32

def get_raster_files_by_year(folder_path, tile_name, year, orbit_directions):
    files_list = []
    for orbit_direction_item in orbit_directions:
        # Create a pattern to match files for the given year
        pattern = os.path.join(folder_path, orbit_direction_item, f'{tile_name}_*{year}*sieved.tif')
        # Use glob to get all matching files
        files_list.extend(glob.glob(pattern))

    return files_list

def sum_rasters(raster_files):
    # Open the first raster to get the dimensions and georeference information
    raster = gdal.Open(raster_files[0])
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
        raster = gdal.Open(raster_file)
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        sum_array += array
    sum_array[sum_array > 0] = 1
    return sum_array, geotransform, projection

def create_sieved(result_filepath_for_sieve, result_filepath_sieved, sieve=10):
    sieve_args = ["gdal_sieve.py", #"-mask", str(mask_tif),
                  "-4", "-nomask",
                  "-st", str(sieve),
                  str(result_filepath_for_sieve), str(result_filepath_sieved)]
    cmd_output = subprocess.run(sieve_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("exit code {} --> {}".format(cmd_output.returncode, sieve_args))

    intermediate_path = result_filepath_sieved.parent.joinpath(result_filepath_sieved.name.replace(".tif", "_inter.tif"))
    shutil.copy(result_filepath_sieved, intermediate_path)
    result_filepath_sieved.unlink()

    raster_ds = gdal.Open(str(intermediate_path))
    raster_array = raster_ds.GetRasterBand(1).ReadAsArray()
    save_raster_template(intermediate_path, result_filepath_sieved, raster_array, data_type=GDT_Byte)
    intermediate_path.unlink()


def save_raster(output_path, array, geotransform, projection):
    driver = gdal.GetDriverByName('GTiff')
    y_size, x_size = array.shape
    dataset = driver.Create(output_path, x_size, y_size, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()

def main(folder_path, tile_name, year, output_path, orbit_directions):

    raster_files = get_raster_files_by_year(folder_path,tile_name, year, orbit_directions)
    if not raster_files:
        print(f"No raster files found for the year {year}")
        return

    sum_array, geotransform, projection = sum_rasters(raster_files)
    save_raster(output_path, sum_array, geotransform, projection)
    print(f"Summed raster saved to {output_path}")

def sum_all(folder_path, tile_name, year_list, output_path, orbit_directions):
    raster_files_master  =[]
    for year in year_list:
        raster_files = get_raster_files_by_year(folder_path,tile_name, year, orbit_directions)
        raster_files_master.extend(raster_files)
    if not raster_files:
        print(f"No raster files found for the year {year}")
        return

    sum_array, geotransform, projection = sum_rasters(raster_files_master)
    save_raster(output_path, sum_array, geotransform, projection)
    print(f"Summed raster saved to {output_path}")

if __name__ == "__main__":
    tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '18LVR', '20LLQ', '20LMP', '20LMQ', '20NQF',
             '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
    model = "best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score"
    years = ['2019','2020','2021']


    for tile_item in tiles:
        folder_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/{tile_item}/zscore_checked'

        folder_path_files = os.listdir(folder_path)
        orbit_directions = []
        for folder_path_file_item in folder_path_files:
            if folder_path_file_item in ["ascending", "descending"]:
                orbit_directions.append(folder_path_file_item)


        for year in years:

            output_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/{tile_item}/zscore_checked/{tile_item}_{year}.tif'
            if not Path(output_path).exists():
                main(folder_path, tile_item, year, output_path, orbit_directions)

        output_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/{tile_item}/zscore_checked/{tile_item}_all.tif'
        if not Path(output_path).exists():
            sum_all(folder_path, tile_item, years, output_path, orbit_directions)

        output_sieved_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/{tile_item}/zscore_checked/{tile_item}_all_largesieve.tif'
        Path(output_sieved_path).unlink(missing_ok=True)
        create_sieved(Path(output_path), Path(output_sieved_path), sieve=50000)