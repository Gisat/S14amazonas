import os
import re
from datetime import datetime
from O7_openeo_backscatter import get_temporalextents_mastertemporalextent
from pathlib import Path
from jobmanagers.data_manipulation.arrange_changedetection import process_directory
from jobmanagers.data_manipulation.arrange_changedetection import extract_dates_from_name
import subprocess
import json
from tondortools.tool import read_raster_info
from Oa_openeo_utils import extract_band
from osgeo.gdal import GDT_Int16
BAND_LIST = ["MCD", "THRESHOLD", "VVPMIN", "VHPMIN", "AIMCD"]
CHECK_SPATIAL_CONSISTENCY = False


def get_extent_and_crs(filepath):
    """Uses gdalinfo to get the extent and CRS from a GeoTIFF."""
    cmd = ["gdalinfo", "-json", filepath]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalinfo failed for {filepath}:\n{result.stderr}")
    info = json.loads(result.stdout)

    crs = info.get("coordinateSystem", {}).get("wkt", "UNKNOWN")
    extent = tuple(
        info["cornerCoordinates"]["upperLeft"] + info["cornerCoordinates"]["lowerRight"])  # (ulx, uly, lrx, lry)
    return extent, crs

# Example usage:
change_detection_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection")
root_directory = "/mnt/hddarchive.nfs/amazonas_dir/work_dir/detection_jobmanagement_poc"
root_work_dir = "/mnt/hddarchive.nfs/amazonas_dir/work_dir/tmp"

tile_list = ["20LNR"]

for tile_item in tile_list:
    change_detection_tile_folder = change_detection_folder.joinpath(tile_item)
    temporal_extent_filepath_dict, extent_check = process_directory(change_detection_tile_folder, tile_item)

    work_dir = Path(root_work_dir).joinpath("change_detection_raw", tile_item)
    os.makedirs(work_dir, exist_ok=True)


    # Collect the extents and CRSes
    if CHECK_SPATIAL_CONSISTENCY:
        reference_extent = None
        reference_crs = None
        inconsistent_files = []
        unique_filepaths = set(entry["filepath"] for entry in temporal_extent_filepath_dict.values())
        for unique_filepath_item in unique_filepaths:
            try:
                extent, crs = get_extent_and_crs(unique_filepath_item)
                if reference_extent is None:
                    reference_extent = extent
                    reference_crs = crs
                else:
                    if extent != reference_extent or crs != reference_crs:
                        inconsistent_files.append((unique_filepath_item, extent, crs))
            except Exception as e:
                print(f"Error processing {unique_filepath_item}: {e}")

        # Output results
        if inconsistent_files:
            print("Inconsistent extent or CRS found in the following files:")
            for entry in inconsistent_files:
                print(f"File: {entry[1]}")
                print(f"Extent: {entry[2]}")
                print(f"CRS: {entry[3]}")
        else:
            print("All files have the same extent and CRS.")

    for temporal_extent, master_detection_path_timeindex_dict in temporal_extent_filepath_dict.items():
        time_index = master_detection_path_timeindex_dict["time_index"]
        master_detection_path = master_detection_path_timeindex_dict["filepath"]
        time_steps = master_detection_path_timeindex_dict["time_steps"]
        print(f"{tile_item} {temporal_extent} {time_index} {time_steps} {master_detection_path.name}")

        for band_index, band_item in enumerate(BAND_LIST):
            tif_band_index = band_index*time_steps + time_index + 1
            dst_path = work_dir.joinpath(f"DEC_{tile_item}_{temporal_extent}_{band_item}.tif")
            print(f"extract {band_index} - {band_item} = {tif_band_index}, {dst_path.name}")
            if not dst_path.exists():
                extract_band(master_detection_path, dst_path ,tif_band_index, datatype="UInt16")

        print("---------------------------")