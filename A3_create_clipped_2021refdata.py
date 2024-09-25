import subprocess
from pathlib import Path
import os, requests
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte
import numpy as np
import datetime

from tondortools.tool import reproject_multibandraster_toextent, read_raster_info, save_raster_template, mosaic_tifs

#######
def find_template_raster(root_folder):

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.tif'):
                return Path(foldername).joinpath(filename)

def merge_tifs(lc_mosaic_filepath, sar_mosaic, multiband_path, work_dir):


    vrt_filepath_list= []
    for band_count in range(3):
        vrt_filepath = work_dir.joinpath(f"{lc_mosaic_filepath.stem}_{sar_mosaic.stem}_{band_count+1}.vrt")
        vrt_cmd = ["gdalbuildvrt", "-b", f"{band_count + 1}", str(vrt_filepath), str(sar_mosaic)]
        subprocess.run(vrt_cmd)
        vrt_filepath_list.append(str(vrt_filepath))



    vrt_filepath = work_dir.joinpath(f"lc_{lc_mosaic_filepath.stem}_{sar_mosaic.stem}.vrt")
    vrt_cmd = ["gdalbuildvrt", "-b", "1", str(vrt_filepath), str(lc_mosaic_filepath)]
    subprocess.run(vrt_cmd)

    ###
    tif_float_path = str(vrt_filepath.parent.joinpath(vrt_filepath.name.replace(".vrt", ".tif")))
    gdal_translate_cmd = ["gdal_translate", "-ot", "Float32", str(vrt_filepath), tif_float_path]
    subprocess.run(gdal_translate_cmd)

    vrt_filepath = work_dir.joinpath(f"lc_float_{lc_mosaic_filepath.stem}_{sar_mosaic.stem}.vrt")
    vrt_cmd = ["gdalbuildvrt", "-b", "1", str(vrt_filepath), str(tif_float_path)]
    subprocess.run(vrt_cmd)
    vrt_filepath_list.append(str(vrt_filepath))
    ###

    big_vrt_cmd = ["gdalbuildvrt", "-separate"]
    big_vrt_cmd.append(f"{lc_mosaic_filepath.stem}_{sar_mosaic.stem}.vrt")
    big_vrt_cmd.extend(vrt_filepath_list)
    subprocess.run(big_vrt_cmd)




    gdal_translate_cmd = ["gdal_translate", f"{lc_mosaic_filepath.stem}_{sar_mosaic.stem}.vrt", str(multiband_path)]
    subprocess.run(gdal_translate_cmd)


#######
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
gfw_extents_filename = "gftw_tileextents.gpkg"
s2_extents = "S2_tiles.gpkg"
ref_year = 2021
###
support_data = amazonas_root_folder.joinpath("support_data")
if not support_data.exists(): raise Exception(f"{support_data} doesnt exists.")

ref_data_folder = support_data.joinpath("ref_worldcover")
if not ref_data_folder.exists(): raise Exception(f"{ref_data_folder} doesnt exists.")

ref_data_mosaic_folder = support_data.joinpath(f"{ref_year}_ref_worldcover_mosaic")
os.makedirs(ref_data_mosaic_folder, exist_ok=True)

merged_sar_ref_folder = support_data.joinpath(f"merged_sar_ref_worldcover")
os.makedirs(merged_sar_ref_folder, exist_ok=True)

ref_sar_mosaic_folder = support_data.joinpath("ref_worldcover_sarmosaic")


work_dir = amazonas_root_folder.joinpath("work_dir", "tmp")
os.makedirs(work_dir, exist_ok=True)



archive_folder = amazonas_root_folder.joinpath("output")
archive_mosaic_folder = archive_folder.joinpath("Mosaic")

###
tiles = ['18LVQ',  '18LVR',  '20LLQ',  '18LWR',  '18NYH',  '18NXH',  '20LLP',  '18NXG',  '20LMP',  '20LMQ',  '20NQG',  '21LYG',  '20NQF',  '20NRG',  '21LYH',  '22MBT',  '22MGB']


tile_map_file = support_data.joinpath('tile_map.json')
tile_map_file_open = open(tile_map_file)
tile_map = json.load(tile_map_file_open)


for tile_item, big_tiles_list in tile_map.items():
    print(f"Tile: {tile_item} - {big_tiles_list}")

    work_dir_tile = work_dir.joinpath(f"{tile_item}")
    os.makedirs(work_dir_tile, exist_ok=True)


    big_tile_rasters = []
    for big_tiles_list_item in big_tiles_list:
        big_tile_raster_path = ref_data_folder.joinpath(f"WC_{big_tiles_list_item}_C.tif")
        if big_tile_raster_path.exists():
            big_tile_rasters.append(str(big_tile_raster_path))
    print(f" big raster {big_tile_rasters}")

    archive_mosaic_tile_folder = archive_mosaic_folder.joinpath(tile_item)
    tile_template_raster = find_template_raster(archive_mosaic_tile_folder)
    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, bbox) = read_raster_info(tile_template_raster)
    ymin = bbox.bounds[1]
    xmax = bbox.bounds[2]

    warped_paths = []
    for big_tile_raster_item in big_tile_rasters:
        tile_raster = work_dir_tile.joinpath(f"{tile_item}_{Path(big_tile_raster_item).name}.tif")
        if not tile_raster.exists():
            reproject_multibandraster_toextent(big_tile_raster_item, tile_raster, epsg, pixel_width, xmin, xmax, ymin,
                                               ymax, work_dir, method='near')
        warped_paths.append(tile_raster)

    lc_mosaic_filepath = ref_data_mosaic_folder.joinpath(f"ref_raster_{tile_item}.tif")
    if not lc_mosaic_filepath.exists():
        mosaic_tifs(warped_paths, lc_mosaic_filepath, no_data=0)

    sar_mosaic = ref_sar_mosaic_folder.joinpath(f"{tile_item}_BAC_MERGED_{ref_year}.tif")

    multiband_path = merged_sar_ref_folder.joinpath(f"{tile_item}_BAC_LC_MERGED_{ref_year}.tif")
    merge_tifs(lc_mosaic_filepath, sar_mosaic, multiband_path, work_dir_tile)

    print(f"merging {warped_paths}")
    print("----")




