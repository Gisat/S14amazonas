import os
import re
import shutil
from datetime import datetime
import pandas as pd
from pathlib import Path
from O7_openeo_backscatter import get_temporalextents_mastertemporalextent

reference_date = datetime(2021, 12, 16)

BAND_LIST = ["_MCD_t", "_MCDthreshold_t", "VVpmin_t", "VHpmin_t", "_AIMCD_t"]


BAND_LIST_DSTNAME = {"_MCD_t": "MCD",
                     "_MCDthreshold_t": "THRESHOLD",
                     "VVpmin_t": "VVPMIN",
                     "VHpmin_t":"VHPMIN",
                     "_AIMCD_t": "AIMCD"}

# Define all intervals from your project
intervals_str = [
    "2021-01-01_2021-01-13", "2021-01-13_2021-01-25", "2021-01-25_2021-02-06",
    "2021-02-06_2021-02-18", "2021-02-18_2021-03-02", "2021-03-02_2021-03-14",
    "2021-03-14_2021-03-26", "2021-03-26_2021-04-07", "2021-04-07_2021-04-19",
    "2021-04-19_2021-05-01", "2021-05-01_2021-05-13", "2021-05-13_2021-05-25",
    "2021-05-25_2021-06-06", "2021-06-06_2021-06-18", "2021-06-18_2021-06-30",
    "2021-06-30_2021-07-12", "2021-07-12_2021-07-24", "2021-07-24_2021-08-05",
    "2021-08-05_2021-08-17", "2021-08-17_2021-08-29", "2021-08-29_2021-09-10",
    "2021-09-10_2021-09-22", "2021-09-22_2021-10-04", "2021-10-04_2021-10-16",
    "2021-10-16_2021-10-28", "2021-10-28_2021-11-09", "2021-11-09_2021-11-21",
    "2021-11-21_2021-12-03", "2021-12-03_2021-12-15", "2021-12-15_2022-01-08",
    "2022-01-08_2022-02-01", "2022-02-01_2022-02-25", "2022-02-25_2022-03-21",
    "2022-03-21_2022-04-14", "2022-04-14_2022-05-08", "2022-05-08_2022-06-01",
    "2022-06-01_2022-06-25", "2022-06-25_2022-07-19", "2022-07-19_2022-08-12",
    "2022-08-12_2022-09-05", "2022-09-05_2022-09-29", "2022-09-29_2022-10-23",
    "2022-10-23_2022-11-16", "2022-11-16_2022-12-10", "2022-12-10_2023-01-03",
    "2023-01-03_2023-01-27", "2023-01-27_2023-02-20", "2023-02-20_2023-03-16",
    "2023-03-16_2023-04-09", "2023-04-09_2023-05-03", "2023-05-03_2023-05-27",
    "2023-05-27_2023-06-20", "2023-06-20_2023-07-14", "2023-07-14_2023-08-07",
    "2023-08-07_2023-08-31", "2023-08-31_2023-09-24", "2023-09-24_2023-10-18",
    "2023-10-18_2023-11-11", "2023-11-11_2023-12-05", "2023-12-05_2023-12-29",
    "2023-12-29_2024-01-22", "2024-01-22_2024-02-15", "2024-02-15_2024-03-10",
    "2024-03-10_2024-04-03", "2024-04-03_2024-04-27", "2024-04-27_2024-05-21",
    "2024-05-21_2024-06-14", "2024-06-14_2024-07-08", "2024-07-08_2024-08-01",
    "2024-08-01_2024-08-25", "2024-08-25_2024-09-18", "2024-09-18_2024-10-12",
    "2024-10-12_2024-11-05", "2024-11-05_2024-11-29", "2024-11-29_2024-12-23"
]




# Match pattern for date in filenames: YYYYMMDD
date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

def extract_dates_from_name(name):
    matches = date_pattern.findall(name)
    return [datetime.strptime(d, "%Y-%m-%d") for d in matches]

def process_directory(root_dir, tile_name):
    # Create a set to store unique intervals found
    found_intervals = dict()

    for intervals_str_time in intervals_str: found_intervals[intervals_str_time] = {}

    # Walk through the directory to find matching files
    for root, _, files in os.walk(root_dir):
        for file in files:
            # Check if file is a .tif and matches the tile and band of interest
            for band in BAND_LIST:
                if file.endswith(".tif") and tile_name in file and band in file:

                    # Extract the start and end dates from the file name
                    start_date, end_date = extract_dates_from_name(file)

                    # Get all intervals and the master extent for the file's date range
                    temporal_extents, master_extent = get_temporalextents_mastertemporalextent(
                        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                    )

                    # Loop over each interval and match it with the filename
                    for index, interval in enumerate(temporal_extents):
                        formatted_interval = f"{interval[0]}_{interval[1]}"
                        expected_filename = f"{index + 1}.tif"

                        if expected_filename in file:
                            found_intervals[formatted_interval][band] = {
                                "filepath": Path(root) / file,
                                "time_index": index,
                                "time_steps": len(temporal_extents)
                            }

    # Sort the intervals
    incomplete_intervals = [interval for interval, bands_dict in found_intervals.items() if len(bands_dict) != 5]

    result = True
    if len(incomplete_intervals) > 0: result = False
    return found_intervals, result




# Load the CSV file
file_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_jobmanagement_missingchangedetection256/job_database-missingchangedetection256.csv")
log_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/log_folder")
detection_jobmanagement_folder = file_path.parent
changedetection_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_raw")
os.makedirs(changedetection_folder, exist_ok=True)

def main():
    df = pd.read_csv(file_path)

    log_filepath = log_folder.joinpath(f"changedetectionlog_{file_path.stem}.txt")
    jobmanager_folder = file_path.parent

    # Display the unique values in the "Name" column
    tile_list = df["Name"].unique()

    missing_files = []
    for tile_item in tile_list:

        changedetection_tile_folder = changedetection_folder.joinpath(tile_item)
        os.makedirs(changedetection_tile_folder, exist_ok=True)

        temporal_extent_filepath_dict, extent_check = process_directory(detection_jobmanagement_folder, tile_item)
        # unique_filepaths = set(entry["filepath"] for entry in temporal_extent_filepath_dict.values())

        for temporal_index, temporal_band_dict in temporal_extent_filepath_dict.items():
            for band_item, band_filepath in temporal_band_dict.items():
                dst_path = changedetection_tile_folder.joinpath(f"DEC_{tile_item}_{temporal_index}_{BAND_LIST_DSTNAME[band_item]}.tif")

                log_message = f"{datetime.now().isoformat()} | {Path(band_filepath['filepath']).name} --> {dst_path}\n"
                with open(log_filepath, 'a') as log_file:
                    log_file.write(log_message)

                # if dst_path.exists():
                #     continue
                if Path(band_filepath['filepath']).exists():
                    shutil.move(Path(band_filepath['filepath']), dst_path)
                    print(f"{Path(band_filepath['filepath']).name} --> {dst_path}")
                else:
                    missing_files.append(Path(band_filepath['filepath']))
    print("------------------------")
    print(missing_files)



if __name__ == "__main__":
    main()
