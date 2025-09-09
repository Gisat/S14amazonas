import subprocess
import json
from datetime import datetime
from Oa_openeo_utils import get_temporalextents_mastertemporalextent
from jobmanagers.data_manipulation.sentinel1_query import query_sentinel1
import time
import urllib.error
from pathlib import Path
import geopandas as gpd


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

def list_s3_folders(bucket_path, config_path):
    cmd = [
        "rclone",
        "--config", config_path,
        "lsf",
        bucket_path,
        "--dirs-only"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error listing folders:\n{result.stderr}")
    return [line.strip("/\n") for line in result.stdout.splitlines() if line.strip()]


def list_tif_files_in_folder(bucket_path, config_path):
    cmd = [
        "rclone",
        "--config", config_path,
        "lsjson",
        bucket_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Warning] Could not access {bucket_path}\n{result.stderr.strip()}")
        return []

    try:
        files = json.loads(result.stdout)
        return [f["Name"] for f in files if f["Name"].endswith(".tif")]
    except json.JSONDecodeError:
        print(f"[Warning] JSON decoding failed for {bucket_path}")
        return []


temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent("2020-11-02",
                                                                                    "2025-03-29")
input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI.gpkg")
input_df = gpd.read_file(input_df_path)

# Settings
bucket_name = "deforestation"
base_s3_path = "sarbackscatter"
config_path = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"
full_bucket_path = f"s14amazonas:{bucket_name}/{base_s3_path}"

# Get folder list
print("\n--- Listing Folders and TIF Files ---\n")
folders = list_s3_folders(full_bucket_path, config_path)

all_tif_files = {}

missing_files = []
for folder in sorted(folders):
    if folder == "stac_dir": continue
    tile_list_item = folder
    folder_path = f"{full_bucket_path}/{folder}"
    tif_files = list_tif_files_in_folder(folder_path, config_path)
    if tif_files:
        all_tif_files[folder] = tif_files
        print(f"{folder}: {len(tif_files)} TIF files")
    else:
        print(f"{folder}: No TIF files found")

    for temporal_extent_item in temporal_extents:
        temporal_extent_startdate = temporal_extent_item[0]
        temporal_extent_enddate = temporal_extent_item[1]
        end_date = datetime.strptime(temporal_extent_enddate, "%Y-%m-%d")
        start_date = datetime.strptime(temporal_extent_startdate, "%Y-%m-%d")

        print(f"{temporal_extent_item}")
        expected_file = f"SARBAC_{tile_list_item}_{temporal_extent_item[0]}_{temporal_extent_item[1]}.tif"
        if expected_file not in tif_files:
            row = input_df[input_df['Name'] == tile_list_item].iloc[0]
            scene_infos = safe_query_catalog(row.xmin, row.ymin, row.xmax, row.ymax, start_date, end_date)
            if len(scene_infos) > 0:
                missing_files.append((tile_list_item, start_date, end_date))
                tile_has_missing_files = True


print(missing_files)
