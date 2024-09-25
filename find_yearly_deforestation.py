import os
import glob
import numpy as np
from osgeo import gdal, gdal_array

def get_raster_files_by_year(folder_path, tile_name, year):
    # Create a pattern to match files for the given year
    pattern = os.path.join(folder_path, f'{tile_name}_{year}*.tif')
    # Use glob to get all matching files
    return glob.glob(pattern)

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

    return sum_array, geotransform, projection

def save_raster(output_path, array, geotransform, projection):
    driver = gdal.GetDriverByName('GTiff')
    y_size, x_size = array.shape
    dataset = driver.Create(output_path, x_size, y_size, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()

def main(folder_path, tile_name, year, output_path):
    raster_files = get_raster_files_by_year(folder_path,tile_name, year)
    if not raster_files:
        print(f"No raster files found for the year {year}")
        return

    sum_array, geotransform, projection = sum_rasters(raster_files)
    save_raster(output_path, sum_array, geotransform, projection)
    print(f"Summed raster saved to {output_path}")

def sum_all(folder_path, tile_name, year_list, output_path):
    raster_files_master  =[]
    for year in year_list:
        raster_files = get_raster_files_by_year(folder_path,tile_name, year)
        raster_files_master.extend(raster_files)
    if not raster_files:
        print(f"No raster files found for the year {year}")
        return

    sum_array, geotransform, projection = sum_rasters(raster_files_master)
    save_raster(output_path, sum_array, geotransform, projection)
    print(f"Summed raster saved to {output_path}")

if __name__ == "__main__":
    tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG',
             '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
    model = "best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score"
    years = ['2019','2020','2021']


    for tile_item in tiles:
        folder_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/deforestation/{tile_item}'
        for year in years:

            output_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/deforestation/{tile_item}_{year}.tif'
            main(folder_path, tile_item, year, output_path)

        output_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/deforestation/{tile_item}_all.tif'
        sum_all(folder_path, tile_item, years, output_path)

