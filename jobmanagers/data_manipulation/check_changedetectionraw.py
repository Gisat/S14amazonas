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

temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent("2021-01-01",
                                                                                    "2024-12-23")

input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI.gpkg")
input_df = gpd.read_file(input_df_path)

changedetection_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_raw")

tile_list = os.listdir(changedetection_folder)

missing_files = []
for tile_list_item in tile_list:
    if tile_list_item == "stac_dir": continue
    print(f" -- {tile_list_item} --")


    cd_tile_folder = changedetection_folder.joinpath(tile_list_item)
    cd_tile_files = os.listdir(cd_tile_folder)
    tile_has_missing_files = False

    for temporal_extent_item in temporal_extents:
        temporal_extent_startdate = temporal_extent_item[0]
        temporal_extent_enddate = temporal_extent_item[1]
        end_date = datetime.strptime(temporal_extent_enddate, "%Y-%m-%d")
        start_date = datetime.strptime(temporal_extent_startdate, "%Y-%m-%d")

        print(f"{temporal_extent_item}")
        expected_file = f"DEC_{tile_list_item}_{temporal_extent_item[0]}_{temporal_extent_item[1]}_MCD.tif"
        if expected_file not in cd_tile_files:
            missing_files.append((tile_list_item, temporal_extent_startdate, temporal_extent_enddate))


    # After all date periods are checked for this tile

print("----")
print(missing_files)