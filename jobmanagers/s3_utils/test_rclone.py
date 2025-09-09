import os
from pathlib import Path
import subprocess

######################################################
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
#tile_list = ['18LVQ', '18LVR', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '21LYH', '22MBT', '22MGB']
tile_list = ['18LWR', '18NXG', '20LMQ', '20NQF', '20NQG', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG']

detection_set = "Detections_set6"
######################################################
archive_folder = amazonas_root_folder.joinpath('output')
archive_detection_folder = archive_folder.joinpath("mcd_detection")


def download_from_s3(bucket_name, s3_path, local_path, config_path):
    cmd = ['rclone',
           '--config', config_path,
        'ls',
        f'gisat:{bucket_name}']

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing rclone: {result.stderr}")
    else:
        print("Download completed!")

for tile_item in tile_list:
    # Example usage:
    bucket_name = "gisat-archive"
    s3_path = f"backscatter"

    archive_detection_folder_tile = archive_detection_folder.joinpath(f"{tile_item}")
    os.makedirs(archive_detection_folder_tile, exist_ok=True)
    local_path = f"{archive_detection_folder_tile}"
    config_path = "/home/eouser/userdoc/rclone.conf"

    download_from_s3(bucket_name, s3_path, local_path, config_path)