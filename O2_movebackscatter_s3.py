import os
from pathlib import Path
import subprocess

######################################################



def copy_to_s3(bucket_name, s3_path, local_path, config_path):
    cmd = ['rclone',
           '--config', config_path,
        'copy',
        '--ignore-existing',
        '-v',
        f'{local_path}',
        f's14amazonas:{bucket_name}/{s3_path}']

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing rclone: {result.stderr}")
    else:
        print("Download completed!")


# Example usage:
bucket_name = "deforestation"
s3_path = f"changedetection_raw"

local_path = "/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_raw/"
config_path = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"

tile_list = os.listdir(local_path)
for tile_item in tile_list:
    tile_path = Path(local_path).joinpath(tile_item)
    if tile_path.is_dir():
        print(f"Processing tile: {tile_item}")
        s3_tile_path = f"{s3_path}/{tile_item}"
        print(f"Copying to S3 path: {s3_tile_path}")
        copy_to_s3(bucket_name, s3_tile_path, tile_path, config_path)