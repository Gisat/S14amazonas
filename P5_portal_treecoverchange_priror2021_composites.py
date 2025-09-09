import os
import shutil
from pathlib import Path
import numpy as np
from tondortools.tool import save_raster_template
from osgeo.gdalconst import GDT_Byte
from src.tondor.util.tool import raster2array
import multiprocessing
from multiprocessing import Pool
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def get_temporalextents(start, end):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    time_windows= []

    time_window_start = start_date
    while (time_window_start < datetime.strptime("2024-12-23", "%Y-%m-%d")):
        time_window_end = time_window_start + timedelta(24)
        time_windows.append([time_window_start, time_window_end])
        time_window_start = time_window_end
    return time_windows

def get_3months_temporalextents(start, end):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    time_windows= []

    time_window_start = start_date
    while (time_window_start < datetime.strptime("2024-12-23", "%Y-%m-%d")):
        time_window_end = time_window_start + relativedelta(months=3)
        time_windows.append([time_window_start, time_window_end])
        time_window_start = time_window_end
    return time_windows





tree_cover_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/tree_cover_change")
tree_cover_prior_composite = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/composites/tree_cover_change")
os.makedirs(tree_cover_prior_composite, exist_ok=True)
tiles_list = os.listdir(tree_cover_folder)

for tiles_list_item in sorted(tiles_list):

    tree_cover_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/tree_cover_change")
    tree_cover_folder_tile = tree_cover_folder.joinpath(tiles_list_item)
    nodata_txt_filepath = tree_cover_folder_tile.joinpath(f"{tiles_list_item}_status.txt")

    portal_data_folder_prior_2021_composites_tile = tree_cover_prior_composite.joinpath(tiles_list_item)
    os.makedirs(portal_data_folder_prior_2021_composites_tile, exist_ok=True)

    if nodata_txt_filepath.exists():
        tree_coverchange_tifs = os.listdir(tree_cover_folder_tile)
        tree_coverchange_tifs = [tree_coverchange_tif_item for tree_coverchange_tif_item in tree_coverchange_tifs
                                 if tree_coverchange_tif_item.endswith("CHANGE.tif")]

        time_windows_3months = get_3months_temporalextents("2017-01-01", "2022-01-01")
        for time_windows_3months_start, time_windows_3months_end in time_windows_3months:
            treecover_change_list = []
            portal_tile_3monthcomposite_tif = portal_data_folder_prior_2021_composites_tile.joinpath(f"DEC_3monthcomposite_{tiles_list_item}_{time_windows_3months_start.strftime('%Y-%m-%d')}_{time_windows_3months_end.strftime('%Y-%m-%d')}_TREECOVERCHANGE.tif")

            if not portal_tile_3monthcomposite_tif.exists():
                for treecover_change_layer_tifs_list_item in sorted(tree_coverchange_tifs):
                    treecover_change_layer_tifs_list_item_time = datetime.strptime(treecover_change_layer_tifs_list_item.split("_")[1], "%Y%m%d")
                    if  time_windows_3months_start < treecover_change_layer_tifs_list_item_time <= time_windows_3months_end:
                            treecover_change_list.append(tree_cover_folder_tile.joinpath(treecover_change_layer_tifs_list_item))
                print(f"compositing {treecover_change_list}")

                tree_coverchange_array_list = []
                for treecover_change_list_item in treecover_change_list:
                    tree_coverchange_array = raster2array(treecover_change_list_item)
                    tree_coverchange_array[np.isnan(tree_coverchange_array)] = 0
                    tree_coverchange_array_list.append(tree_coverchange_array)


                if len(treecover_change_list) > 0:
                    tree_coverchange_array_stack = np.stack(tree_coverchange_array_list, axis=0)
                    tree_coverchange_sum = np.sum(tree_coverchange_array_stack, axis=0)
                    tree_coverchange_sum[tree_coverchange_sum >0] = 1
                    save_raster_template(treecover_change_list_item, portal_tile_3monthcomposite_tif, tree_coverchange_sum, GDT_Byte, 0)

            print(f"created compositing{portal_tile_3monthcomposite_tif}")


