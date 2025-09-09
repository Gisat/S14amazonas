import os
import subprocess
import shutil
from pathlib import Path

def run_rclone_check(local_path, remote_path, config_path):
    cmd = [
        'rclone',
        '--config', config_path,
        'check',
        '--one-way',
        '--size-only',  # Remove this if you want full checksum comparison
        str(local_path),
        f's14amazonas:{remote_path}'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True  # Files match
    else:
        print(f"[Mismatch] {local_path} ↔ {remote_path}\n{result.stderr.strip()}\n")
        return False

def remove_local_path(path):
    if Path(path).exists():
        print(f"🗑️  Removing local path: {path}")
        shutil.rmtree(path)
    else:
        print(f"[Skip] Path does not exist: {path}")

# Setup paths
bucket_name = "deforestation"
base_s3_path = "sarbackscatter"
config_path = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"

local_backscatter_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter")
local_stac_path = local_backscatter_path / "stac_dir"

tile_list = os.listdir(local_backscatter_path)

print("\n--- Checksum Verification and Cleanup Report ---\n")

for tile in sorted(tile_list):
    if tile == "stac_dir": continue
    print(f"🔍 Checking tile: {tile}")

    # Define paths
    local_tile_path = local_backscatter_path / tile
    remote_tile_path = f"{bucket_name}/{base_s3_path}/{tile}"

    local_stac_tile_path = local_stac_path / tile
    remote_stac_tile_path = f"{bucket_name}/{base_s3_path}/stac_dir/{tile}"

    # Run checks
    backscatter_ok = run_rclone_check(local_tile_path, remote_tile_path, config_path)

    stac_ok = False
    if local_stac_tile_path.exists():
        stac_ok = run_rclone_check(local_stac_tile_path, remote_stac_tile_path, config_path)
    else:
        print(f"[Missing] Local stac_dir tile not found for {tile}")

    # Decision
    if backscatter_ok:
        print(f"✓ {tile}: All files verified in S3.")
        #remove_local_path(local_tile_path)
        # remove_local_path(local_stac_tile_path)
    else:
        print(f"✗ {tile}: Verification failed or incomplete, skipping deletion.\n")