import subprocess
from pathlib import Path

def run_rclone_check(local_path, remote_path, config_path):
    cmd = [
        'rclone',
        '--config', config_path,
        'check',
        '--one-way',
        '--size-only',  # remove this if you want full checksum (may be slower)
        local_path,
        f's14amazonas:{remote_path}'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True  # files match
    else:
        print(f"[Mismatch] {local_path} ↔ {remote_path}\n{result.stderr.strip()}\n")
        return False

# Setup paths
bucket_name = "deforestation"
base_s3_path = "sarbackscatter"
config_path = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"

local_backscatter_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter")
local_stac_path = local_backscatter_path / "stac_dir"

# Get local tile names (excluding stac_dir itself)
# local_tiles = [p.name for p in local_backscatter_path.iterdir() if p.is_dir() and p.name != "stac_dir"]
# local_tiles = ["18MZS", "19PEK", "22MFS", "18NWF"]
print("\n--- Checksum Verification Report ---\n")

tile_list = [
    "16PEU", "16PGA", "16PGC", "16PGT", "20LNQ", "20LPR", "20LQR",
    "20MPB", "20MPE", "20MQC", "20MQD", "20MQE", "20MQU", "20NPG",
    "20NPK", "20NQJ", "20NRH", "20NRJ", "21LTF", "21LTG", "21LXG",
    "21LYK", "21MTP", "22MBT", "22MCA"
]

for tile in sorted(tile_list):
    print(f"Checking tile: {tile}")

    # Define local and remote paths
    local_tile_path = str(local_backscatter_path / tile)
    remote_tile_path = f"{bucket_name}/{base_s3_path}/{tile}"

    local_stac_tile_path = str(local_stac_path / tile)
    remote_stac_tile_path = f"{bucket_name}/{base_s3_path}/stac_dir/{tile}"

    # Check backscatter tile
    backscatter_ok = run_rclone_check(local_tile_path, remote_tile_path, config_path)

    # Check stac_dir tile
    stac_ok = False
    if Path(local_stac_tile_path).exists():
        stac_ok = run_rclone_check(local_stac_tile_path, remote_stac_tile_path, config_path)
    else:
        print(f"[Missing] Local stac_dir tile not found for {tile}")

    if backscatter_ok and stac_ok:
        print(f"✓ {tile}: All files verified.\n")
    else:
        print(f"✗ {tile}: Some files are missing or mismatched.\n")