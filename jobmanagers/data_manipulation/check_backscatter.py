import os
from pathlib import Path
import geopandas as gpd
from Oa_openeo_utils import get_temporalextents_mastertemporalextent
from datetime import datetime
from sentinel1_query import query_sentinel1
import time
import urllib.error

def safe_query_catalog(xmin, ymin, xmax, ymax, start_date, end_date, retries=3, delay=10, **kwargs):
    for attempt in range(retries):
        try:
            return query_sentinel1(xmin, ymin, xmax, ymax, start_date, end_date)
        except urllib.error.HTTPError as e:
            print(f"Attempt {attempt + 1}: HTTPError {e.code} - {e.reason}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            print(f"Attempt {attempt + 1}: Other error - {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent("2020-11-02",
                                                                                    "2025-03-29")

input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI.gpkg")
input_df = gpd.read_file(input_df_path)

sarbackscatter_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter")

tile_list = os.listdir(sarbackscatter_folder)

missing_files = []
for tile_list_item in tile_list:
    if not tile_list_item == "21LYG": continue
    print(f" -- {tile_list_item} --")

    check_ok_file = sarbackscatter_folder.joinpath(tile_list_item, "tile.check.ok")
    if check_ok_file.exists():
        # print(f"Skipping {tile_list_item} as check.ok exists.")
        # continue
        pass

    sarbackscatter_tile_folder = sarbackscatter_folder.joinpath(tile_list_item)
    sarbackscatter_tile_files = os.listdir(sarbackscatter_tile_folder)
    tile_has_missing_files = False

    for temporal_extent_item in temporal_extents:
        temporal_extent_startdate = temporal_extent_item[0]
        temporal_extent_enddate = temporal_extent_item[1]
        end_date = datetime.strptime(temporal_extent_enddate, "%Y-%m-%d")
        start_date = datetime.strptime(temporal_extent_startdate, "%Y-%m-%d")

        print(f"{temporal_extent_item}")
        expected_file = f"SARBAC_{tile_list_item}_{temporal_extent_item[0]}_{temporal_extent_item[1]}.tif"
        if expected_file not in sarbackscatter_tile_files:
            row = input_df[input_df['Name'] == tile_list_item].iloc[0]
            scene_infos = safe_query_catalog(row.xmin, row.ymin, row.xmax, row.ymax, start_date, end_date)
            if len(scene_infos) > 0:
                missing_files.append((tile_list_item, start_date, end_date))
                tile_has_missing_files = True

    # After all date periods are checked for this tile
    if not tile_has_missing_files:
        # Create the check.ok file
        with open(check_ok_file, 'w') as f:
            f.write('Checked and complete.\n')
print("----")
print(missing_files)