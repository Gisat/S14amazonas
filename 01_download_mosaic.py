import os
from pathlib import Path
import subprocess

######################################################
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
tile = "21LYH"
tile_list = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']


######################################################
archive_folder = amazonas_root_folder.joinpath('output')
archive_mosaic_folder = archive_folder.joinpath("Mosaic")


def download_from_s3(bucket_name, s3_path, local_path, config_path):
    cmd = [
        'rclone',
        'copy',
        '--config', config_path,
        "--log-level=INFO",
        "--no-gzip-encoding",
        f'gisat:{bucket_name}/{s3_path}',
        local_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing rclone: {result.stderr}")
    else:
        print("Download completed!")

# Example usage:
for tile_item in tile_list:
    bucket_name = "amazonas-archive"
    s3_path = f"output/Mosaics/{tile_item}"

    local_path =  archive_mosaic_folder.joinpath(f'{tile_item}')
    os.makedirs(local_path, exist_ok=True)
    config_path = "/home/eouser/userdoc/rclone.conf"

    download_from_s3(bucket_name, s3_path, local_path, config_path)