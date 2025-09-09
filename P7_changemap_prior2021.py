import os
from datetime import datetime, timedelta
from pathlib import Path
from tondortools.tool import raster2array
import json
import numpy as np

def get_24day_temporalextents(start, end):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    time_windows= []

    time_window_start = start_date
    while (time_window_start <= end_date):
        time_window_end = time_window_start + timedelta(days=24)
        time_windows.append([time_window_start, time_window_end])
        time_window_start = time_window_end
    return time_windows




tree_cover_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/tree_cover_change")
tiles_list = os.listdir(tree_cover_folder)

prior_2021_changemap = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/change_map")
os.makedirs(prior_2021_changemap, exist_ok=True)


for tiles_list_item in sorted(tiles_list):

    prior_2021_changemap_count_tile = prior_2021_changemap.joinpath(f"{tiles_list_item}_changemap_count.json")
    prior_2021_changemap_ratio_tile = prior_2021_changemap.joinpath(f"{tiles_list_item}_changemap_ratio.json")
    prior_2021_changemap_total_count_tile = prior_2021_changemap.joinpath(f"{tiles_list_item}_total_count.txt")

    if not prior_2021_changemap_count_tile.exists():
        tile_time_count_dict = {}
        tile_item_ratio_dict = {}
        print(f"collecting data for {tiles_list_item}")
        tree_cover_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/tree_cover_change")
        tree_cover_folder_tile = tree_cover_folder.joinpath(tiles_list_item)
        tree_coverchange_tifs = os.listdir(tree_cover_folder_tile)
        tree_coverchange_tifs = [tree_coverchange_tif_item for tree_coverchange_tif_item in tree_coverchange_tifs
                                 if tree_coverchange_tif_item.endswith("CHANGE.tif")]

        tree_cover_files = os.listdir(tree_cover_folder_tile)
        time_windows_24days = get_24day_temporalextents("2017-01-01", "2022-01-01")
        for time_windows_24days_start, time_windows_24days_end in time_windows_24days:
            treecover_change_list = []

            for treecover_change_layer_tifs_list_item in sorted(tree_coverchange_tifs):
                treecover_change_layer_tifs_list_item_time = datetime.strptime(
                    treecover_change_layer_tifs_list_item.split("_")[1], "%Y%m%d")
                if time_windows_24days_start <= treecover_change_layer_tifs_list_item_time <= time_windows_24days_end:
                    treecover_change_list.append(tree_cover_folder_tile.joinpath(treecover_change_layer_tifs_list_item))
            print(f"- collecting {treecover_change_list}")

            change_pixels_count = 0
            if len(treecover_change_list) > 0:
                for treecover_change_list_item in treecover_change_list:
                    rasterarray = raster2array(treecover_change_list_item)
                    change_pixels_count += np.sum(rasterarray == 1)
                total = rasterarray.size
                tile_item_ratio_dict[time_windows_24days_end] = (change_pixels_count/total)*100
                tile_time_count_dict[time_windows_24days_end] = change_pixels_count


        # Convert to required format
        tile_time_count_json = {
            dt.strftime("%Y%m%d"): {
                "Date": dt.strftime("%Y%m%d"),
                "Count": str(count)
            }
            for dt, count in tile_time_count_dict.items()
        }

        # Save to JSON file
        with open(str(prior_2021_changemap_count_tile), "w") as f:
            json.dump(tile_time_count_json, f, indent=4)

        ########
        # Convert to required format
        tile_time_ratio_json = {
            dt.strftime("%Y%m%d"): {
                "Date": dt.strftime("%Y%m%d"),
                "Count": str(count)
            }
            for dt, count in tile_item_ratio_dict.items()
        }

        # Save to JSON file
        with open(str(prior_2021_changemap_ratio_tile), "w") as f:
            json.dump(tile_time_ratio_json, f, indent=4)

        print(f"total {total}")
        prior_2021_changemap_total_count_tile.write_text(str(total))

        print("------------------------------------------------")
