import os
from pathlib import Path
import subprocess

def copy_to_s3(bucket_name, s3_path, config_path):
    cmd = [
        'rclone',
        '--config', config_path,
        'purge',
        # '--dry-run',
        # '--ignore-existing',
        '-v',
        f's14amazonas:{bucket_name}/{s3_path}'
    ]

    print(f"Starting upload: -> s14amazonas:{bucket_name}/{s3_path}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error executing rclone for ")
    else:
        print(f"Upload completed for")

# Setup paths
bucket_name = "deforestation"
config_path = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"
s3_path = "sarbackscatter/stac_dir"

copy_to_s3(bucket_name, s3_path, config_path)