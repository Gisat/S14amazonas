from pathlib import Path
import os, requests
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte
import numpy as np
import datetime

from tondortools.tool import reproject_multibandraster_toextent, read_raster_info, save_raster_template

amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
gfw_extents_filename = "gftw_tileextents.gpkg"
s2_extents = "S2_tiles.gpkg"
api_url = "https://data-api.globalforestwatch.org/dataset/gfw_integrated_alerts/latest/download/geotiff?grid=10/100000&tile_id={}&pixel_meaning=date_conf&x-api-key=2d60cd88-8348-4c0f-a6d5-bd9adb585a8c"


tiles = ['18LVQ',
 '18LVR',
 '20LLQ',
 '18LWR',
 '18NYH',
 '18NXH',
 '20LLP',
 '18NXG',
 '20LMP',
 '20LMQ',
 '20NQG',
 '21LYG',
 '20NQF',
 '20NRG',
 '21LYH',
 '22MBT',
 '22MGB']

update_tilemap_json = False

support_data = amazonas_root_folder.joinpath("support_data")

##
gfw_extents = support_data.joinpath(gfw_extents_filename)
s2_extents = support_data.joinpath(s2_extents)

tile_map_file = support_data.joinpath('tile_map.json')
tile_map_file_open = open(tile_map_file)
tile_map = json.load(tile_map_file_open)

for tile_item in tiles:
    print(f"looking for tile {tile_item}")
    if not str(tile_item) in tile_map.keys() or True:
        gfw_gdf = gpd.read_file(gfw_extents)
        s2_gdf = gpd.read_file(s2_extents)

        s2_tile_gdf = s2_gdf[s2_gdf.Name == tile_item]
        s2_tile_gdf_bounds = s2_tile_gdf.bounds

        overlap = gpd.overlay(s2_tile_gdf, gfw_gdf, how='intersection')
        gfw_tile = overlap.tile_id
        gfw_tile = gfw_tile.values
        tile_map[tile_item] = list(gfw_tile)
        update_tilemap_json = True
    else:
        gfw_tile = tile_map[tile_item]

if update_tilemap_json:
    # Serializing json
    json_object = json.dumps(tile_map, indent=4)

    # Writing to sample.json
    with open(tile_map_file, "w") as outfile:
        outfile.write(json_object)

