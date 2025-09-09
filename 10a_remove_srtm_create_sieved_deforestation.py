import shutil
from pathlib import Path
import os, requests
import subprocess
import geopandas as gpd
import json

import pandas as pd
from osgeo import gdal
from gdalconst import GDT_Byte, GDT_Float32
import numpy as np
from datetime import datetime, timedelta, date
import math
from tondortools.tool import read_raster_info, save_raster, mosaic_tifs, save_raster_template
from src.tondor.util.tool import reproject_multibandraster_toextent

def create_merged_filepath(output_path, raster1, raster2):
    args = ["gdal_merge.py",
            "-separate",
            "-o", str(output_path),
            "-of", "GTiff",
            "-ot", "Float32",
            "-co", "compress=DEFLATE"]
    args = args + [str(raster1)] + [str(raster2)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")


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

tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
tile_srtm_csvpath = "/mnt/hddarchive.nfs/amazonas_dir/aux_data/geom_data/S2_grid_AmazonBasin_detections_thresholds.csv"
model_version = 'best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
srtm_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/aux_data/SRTM")
########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir", "reclassification")
os.makedirs(work_dir, exist_ok=True)
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion_parent = detection_folder.joinpath(f'{model_version}')

########################################################################################################################
support_data = amazonas_root_folder.joinpath("support_data")
merged_sar_ref_folder = support_data.joinpath(f"ref_worldcover_sarmosaic")
predicted_baselc_folder = support_data.joinpath(f"base_worldcover_prediction")
srtm_mask_folder = support_data.joinpath("tile_srtm")
os.makedirs(srtm_mask_folder, exist_ok=True)
########################################################################################################################


detection_folder_aiversion_reclassified_sieved = detection_folder_aiversion_parent.joinpath("sieved")
os.makedirs(detection_folder_aiversion_reclassified_sieved, exist_ok=True)

for tile_item in tiles:

    detection_folder_aiversion_reclassified = detection_folder_aiversion_parent.joinpath(tile_item, "reclassified")
    detection_folder_aiversion_sieved_tile = detection_folder_aiversion_parent.joinpath("deforestation",
                                                                                        tile_item)
    os.makedirs(detection_folder_aiversion_sieved_tile, exist_ok=True)

    template_raster = None
    reclassified_tifs = os.listdir(detection_folder_aiversion_reclassified)
    for folder_file_list_item in sorted(reclassified_tifs):
        detection_path = detection_folder_aiversion_reclassified.joinpath(folder_file_list_item)
        if not ".tif" in folder_file_list_item: continue
        template_raster = detection_path
        if template_raster.exists(): break
    print(f"using template rasters: {template_raster}")
    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands,
     imagery_extent_box) = read_raster_info(template_raster)


    tile_srtm_df = pd.read_csv(tile_srtm_csvpath)
    srtm_name = tile_srtm_df.loc[tile_srtm_df["Name"] == tile_item, "SRTM"].iloc[0]
    srtm_filepath = srtm_folder.joinpath(srtm_name)
    if not srtm_filepath.exists(): raise Exception(f"{srtm_filepath} not found")


    tile_srtm_path = srtm_mask_folder.joinpath(f"srtm_{tile_item}.tif")
    if not tile_srtm_path.exists():
        reproject_multibandraster_toextent( srtm_filepath,tile_srtm_path, epsg, pixel_width, xmin,
                                           imagery_extent_box.bounds[2], imagery_extent_box.bounds[1], ymax, work_dir=None,
                                           method='near')





    for raster_file_item in sorted(reclassified_tifs):
        if raster_file_item.endswith("CLASS.tif"):
            result_filepath = detection_folder_aiversion_reclassified.joinpath(raster_file_item)
            detection_folder_aiversion_reclassified_srtm = detection_folder_aiversion_reclassified.joinpath(f"{result_filepath.stem}_srtm.tif")



            srtm_ds = gdal.Open(str(tile_srtm_path))
            base_srtm = srtm_ds.GetRasterBand(1).ReadAsArray()
            base_srtm_mask = (base_srtm<1600).astype(int)

            result_ds = gdal.Open(str(result_filepath))
            result_array = result_ds.GetRasterBand(1).ReadAsArray()

            result_array[base_srtm_mask == 0] = 0
            save_raster_template(template_raster, detection_folder_aiversion_reclassified_srtm, result_array, GDT_Byte)


            result_filepath_sieved = detection_folder_aiversion_sieved_tile.joinpath(raster_file_item)
            create_sieved(detection_folder_aiversion_reclassified_srtm, result_filepath_sieved)

