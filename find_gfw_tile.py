
from pathlib import Path
import os, requests
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte
import numpy as np
import datetime

from tondortools.tool import reproject_multibandraster_toextent, read_raster_info, save_raster_template


def decode_alert(alert_value_list):
    date_dict = {}
    for alert_value_list_item in alert_value_list:
        if alert_value_list_item == 0:
            continue

        days_since_base_date = int(str(alert_value_list_item)[1:])

        # Calculate the actual date
        base_date = "2014-12-31"
        date = datetime.datetime.strptime(base_date, "%Y-%m-%d") + datetime.timedelta(days=days_since_base_date)
        if not date in date_dict.keys():
            date_dict[date] = [alert_value_list_item]
        else:
            date_dict[date].append(alert_value_list_item)

    return date_dict

def find_template_raster(root_folder):

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.tif'):
                return Path(foldername).joinpath(filename)

amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
gfw_extents_filename = "gftw_tileextents.gpkg"
s2_extents = "S2_tiles.gpkg"
api_url = "https://data-api.globalforestwatch.org/dataset/gfw_integrated_alerts/latest/download/geotiff?grid=10/100000&tile_id={}&pixel_meaning=date_conf&x-api-key=2d60cd88-8348-4c0f-a6d5-bd9adb585a8c"

tile_list = ["22MBT", "21LYH", "21LYG", "22MGB", "20NRG", "20NQG", "20NQF", "20LMQ", "20LLQ","18NXH",
                 "18NXG", "18LWR", "18LVR", "18LVQ"]

########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir")
os.makedirs(work_dir, exist_ok=True)

support_data = amazonas_root_folder.joinpath("support_data")
os.makedirs(support_data, exist_ok=True)

gfw_folder_root = amazonas_root_folder.joinpath('Detections', 'GFW')

gfw_data = work_dir.joinpath('gfw_data')
os.makedirs(gfw_data, exist_ok=True)
##
gfw_extents = support_data.joinpath(gfw_extents_filename)
s2_extents = support_data.joinpath(s2_extents)

tile_map_file = support_data.joinpath('tile_map.json')
tile_map_file_open = open(tile_map_file)
tile_map = json.load(tile_map_file_open)
update_tilemap_json = False
FIND_GFWTILE = False

archive_folder = amazonas_root_folder.joinpath("output")
archive_mosaic_folder = archive_folder.joinpath("Mosaic")


for tile in tile_list:
    print(f"{tile}")
    archive_mosaic_tile_folder = archive_mosaic_folder.joinpath(tile)

    tile_template_raster = find_template_raster(archive_mosaic_tile_folder)
    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, bbox) = read_raster_info(tile_template_raster)
    ymin = bbox.bounds[1]
    xmax = bbox.bounds[2]


    if not str(tile) in tile_map.keys():
        gfw_gdf = gpd.read_file(gfw_extents)
        s2_gdf = gpd.read_file(s2_extents)

        s2_tile_gdf = s2_gdf[s2_gdf.Name == tile]
        s2_tile_gdf_bounds = s2_tile_gdf.bounds

        overlap = gpd.overlay(s2_tile_gdf, gfw_gdf, how='intersection')
        gfw_tile = overlap.tile_id
        gfw_tile = gfw_tile.values[0]
        tile_map[tile] = gfw_tile
        update_tilemap_json = True
    else:
        gfw_tile = tile_map[tile]

    gfw_tile = gfw_tile[0]
    # Create the complete API URL by formatting the tile_id
    complete_api_url = api_url.format(gfw_tile)
    gfw_filepath = gfw_data.joinpath(f"{gfw_tile}.tif")
    # Extract the filename from the URL
    if not gfw_filepath.exists():

        # Send an HTTP GET request to the URL
        response = requests.get(complete_api_url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open a file for binary writing
            with open(gfw_filepath, 'wb') as file:
                # Write the content from the response to the file
                file.write(response.content)
            print(f"Downloaded {gfw_filepath} to {gfw_data}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

    gfw_folder_tile_folder = gfw_folder_root.joinpath(tile)
    os.makedirs(gfw_folder_tile_folder, exist_ok=True)


    gfw_folder_tile_raster = gfw_folder_tile_folder.joinpath(f"{tile}.tif")
    gfw_folder_tile_folder_label = gfw_folder_tile_folder.joinpath("label")
    os.makedirs(gfw_folder_tile_folder_label, exist_ok=True)

    if not gfw_folder_tile_raster.exists():
        reproject_multibandraster_toextent(gfw_filepath, gfw_folder_tile_raster, epsg, pixel_width, xmin, xmax, ymin, ymax, work_dir, method ='near')

    gfw_folder_tile_raster_ds = gdal.Open(str(gfw_folder_tile_raster))
    gfw_folder_tile_rasterdata = gfw_folder_tile_raster_ds.GetRasterBand(1).ReadAsArray()
    unique_data = np.unique(gfw_folder_tile_rasterdata)



    date_dict = decode_alert(unique_data)

    for date_dict_keys, values in date_dict.items():

        # Date to compare against
        comparison_date = datetime.datetime(2020, 1, 1, 0, 0)
        if date_dict_keys > comparison_date: continue

        mask = np.zeros_like(gfw_folder_tile_rasterdata)
        value_count_dict = {2: 0, 3: 0, 4: 0}
        count = 0
        for value_item in values:
            label_value = int(str(value_item)[0])
            value_mask = gfw_folder_tile_rasterdata == value_item
            value_mask_count = np.count_nonzero(value_mask)
            mask[gfw_folder_tile_rasterdata == value_item] = label_value
            value_count_dict[label_value] = value_mask_count
            count += value_mask_count

        conf_str = ""
        for value_count_key, value_count in value_count_dict.items():
            conf_str = f"{conf_str}_{value_count_key}-{value_count}"

        data_str = date_dict_keys.strftime('%Y-%m-%d')
        gfw_folder_tile_folder_label_filepath = gfw_folder_tile_folder_label.joinpath(f"{tile}_{data_str}{conf_str}_{count}.tif")
        if not gfw_folder_tile_folder_label_filepath.exists():
            save_raster_template(tile_template_raster, gfw_folder_tile_folder_label_filepath, mask, GDT_Byte,0)
