from pathlib import Path
import os, requests
import subprocess
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte
import numpy as np
from datetime import datetime, timedelta, date

tile = "21LYG"
amazonas_root_folder = Path("/home/yantra/gisat/amazonas")

########################################################################################################################
########################################################################################################################
########################################################################################################################
### CREATE A GRID-LIST OF DATES EVERY 6/12 DAYS (for 12-day acquisition frequency) ###
acq_frequency = 12

startdateyear  = 2015
startdatemonth = 5
startdatedate  = 1

enddateyear  = 2022
enddatemonth = 6
enddatedate  = 30

start_date = datetime(startdateyear, startdatemonth, startdatedate)
end_date = datetime(enddateyear, enddatemonth, enddatedate)
days_interval = np.ndarray.tolist(np.arange(start_date,end_date,timedelta(days=acq_frequency),dtype = "datetime64[D]"))
########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir")
support_data = amazonas_root_folder.joinpath("support_data")
gfw_folder_root = amazonas_root_folder.joinpath("Detections", "GFW")
gfw_data = work_dir.joinpath("gfw_data")
########################################################################################################################
gfw_folder_tile_folder = gfw_folder_root.joinpath(tile)
gfw_folder_tile_raster = gfw_folder_tile_folder.joinpath(f"{tile}.tif")
gfw_folder_tile_folder_label = gfw_folder_tile_folder.joinpath("label")
########################################################################################################################
gfw_folder_tile_folder_renamed_label = gfw_folder_tile_folder.joinpath("label_renamed")
os.makedirs(gfw_folder_tile_folder_renamed_label, exist_ok=True)
gfw_files = os.listdir(gfw_folder_tile_folder_label)
gfw_raster_day_bin = dict()

for date_item in days_interval:
    week_date = datetime.strptime(str(date_item), "%Y-%m-%d")

    raster_list = []
    for gfw_file in gfw_files:
        if not gfw_file.endswith(".tif"): continue
        date_str = gfw_file.split("_")[1]
        acq_date = datetime.strptime(date_str, "%Y-%m-%d")
        if (acq_date >= week_date - timedelta(days=acq_frequency)) and (acq_date < week_date):
            raster_list.append(gfw_file)

    gfw_raster_day_bin[date_item.strftime("%Y-%m-%d")] = raster_list

for date_item, raster_filename_list in gfw_raster_day_bin.items():
    if len(raster_filename_list) == 0: continue
    raster_list = [str(Path(gfw_folder_tile_folder_label).joinpath(raster_list_item)) for raster_list_item in raster_filename_list]
    output_filepath = str(Path(gfw_folder_tile_folder_renamed_label).joinpath(f"{tile}_{date_item}.tif"))
    args = ["gdalwarp",
            "-of", "GTiff",
            "-r", "max",
            "-co", "BIGTIFF=IF_NEEDED",
            "-co", "TILED=YES",
            "-co", "COMPRESS=LZW"] + raster_list + [output_filepath]
    pr = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("exit code {:d} --> {:s}.".format(pr.returncode, repr(args)))

    value_count_dict = {2: 0, 3: 0, 4: 0}
    values = [2, 3, 4]
    count = 0
    raster_ds = gdal.Open(str(output_filepath))
    raster_array = raster_ds.ReadAsArray().astype(int)
    for value_item in values:
        raster_array_mask = raster_array == value_item
        value_mask_count = np.count_nonzero(raster_array_mask)
        value_count_dict[int(value_item)] = value_mask_count
        count += value_mask_count

    conf_str = ""
    for value_count_key, value_count in value_count_dict.items():
        conf_str = f"{conf_str}_{value_count_key}-{value_count}"

    output_filepath_new = str(Path(gfw_folder_tile_folder_renamed_label).joinpath(f"{tile}_{date_item}{conf_str}_{count}.tif"))
    os.rename(str(output_filepath), str(output_filepath_new))
    print('----')


print(os.listdir(gfw_folder_tile_folder_label))