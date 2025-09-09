import os
from pathlib import Path
import subprocess

def copy_to_s3(bucket_name, s3_path, local_path, config_path):
    cmd = [
        'rclone',
        '--config', config_path,
        'move',
        # '--dry-run',
        # '--ignore-existing',
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
bucket_name = "deforestationbackup"
base_local_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/detection_jobmanagement_batch3")
config_path = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"
s3_base_path = base_local_path.name #"sarbackscatter"

copy_to_s3(bucket_name, s3_base_path, base_local_path, config_path)

