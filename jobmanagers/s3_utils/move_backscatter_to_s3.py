import os
from pathlib import Path
import subprocess

def copy_to_s3(bucket_name, s3_path, local_path, config_path):
    cmd = [
        'rclone',
        '--config', config_path,
        'copy',
        '--ignore-existing',
        '-v',
        str(local_path),
        f's14amazonas:{bucket_name}/{s3_path}'
    ]

    print(f"Starting upload: {local_path} -> s14amazonas:{bucket_name}/{s3_path}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error executing rclone for {local_path}")
    else:
        print(f"Upload completed for {local_path}")

# Setup paths


data_to_backup = "sarbackscatter" #"changedetection_raw" #"sarbackscatter"
bucket_name = "deforestation"



base_local_path = Path(f"/mnt/hddarchive.nfs/amazonas_dir/work_dir/{data_to_backup}")
stac_base_path = base_local_path / "stac_dir"
config_path = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"
s3_base_path = f"{data_to_backup}"

s3_stac_path = "sarbackscatter/stac_dir"
local_path = "/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter/stac_dir"
copy_to_s3(bucket_name, s3_stac_path, local_path, config_path)
exit(0)

# tile_list = ['15QYU', '15QYV', '20LLQ', '20LLR', '20MQA', '20MQB', '20NPH', '20NPJ', '21LWF', '21LWG', '21MWN', '21MWP', '21MXN', '21MXP', '22KHB', '22KHC', '22LBP', '22LEP', '22MBB', '22MBE', '22MCB', '22MCU']
# Loop through tiles (ignoring 'stac_dir' folder itself)
for tile_folder in base_local_path.iterdir():
    # tile_folder = base_local_path.joinpath(tile_folder_item)
    if tile_folder.is_dir() and tile_folder.name != "stac_dir":
        tile_name = tile_folder.name

        # Upload the backscatter folder
        backscatter_tile_path = tile_folder
        s3_backscatter_path = f"{s3_base_path}/{tile_name}"
        copy_to_s3(bucket_name, s3_backscatter_path, backscatter_tile_path, config_path)

        # Upload the corresponding stac_dir subfolder
        stac_tile_path = stac_base_path / tile_name
        if stac_tile_path.exists():
            s3_stac_path = f"{s3_base_path}/stac_dir/{tile_name}"
            copy_to_s3(bucket_name, s3_stac_path, stac_tile_path, config_path)
        else:
            print(f"STAC folder not found for tile: {tile_name}")