import os
import glob
import numpy as np
from osgeo import gdal, gdal_array


def get_raster_files_by_year(folder_path, tile_name, year, orbit_directions):
    files_list = []
    for orbit_direction_item in orbit_directions:
        # Create a pattern to match files for the given year
        pattern = os.path.join(folder_path, orbit_direction_item, f'{tile_name}_*{year}*CLASS.tif')
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
        inverted_array  = np.where(array == 0, 1, 0).astype(gdal_array.GDALTypeCodeToNumericTypeCode(data_type))
        sum_array += inverted_array
    sum_array[sum_array >0] = 1
    return sum_array, geotransform, projection

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
    years = ['2017','2018']


    for tile_item in tiles:
        folder_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/training_years_prediction/{tile_item}'

        folder_path_files = os.listdir(folder_path)
        orbit_directions = []
        for folder_path_file_item in folder_path_files:
            if folder_path_file_item in ["ascending", "descending"]:
                orbit_directions.append(folder_path_file_item)


        for year in years:

            output_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/training_years_prediction/{tile_item}/trainingyears_classsummask_{tile_item}_{year}.tif'
            main(folder_path, tile_item, year, output_path, orbit_directions)

        output_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/training_years_prediction/{tile_item}/trainingyears_classsummask_{tile_item}_all.tif'
        sum_all(folder_path, tile_item, years, output_path, orbit_directions)

