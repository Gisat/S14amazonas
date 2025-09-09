import os
import shutil
from datetime import datetime
import pandas as pd
from pathlib import Path
from O7_openeo_backscatter import get_temporalextents_mastertemporalextent
from sentinel1_query import query_sentinel1




reference_date = datetime(2021, 12, 14)
reference_endate = datetime(2025, 3, 29)

CHECK_MISSING_BAC = True

# Load the CSV file
file_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/backscatter_jobmanagement_missingsarbackscatter256_2/job_database-missingsarbackscatter256_2.csv")
sarbackscatter_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter")

df = pd.read_csv(file_path)

jobmanager_folder = file_path.parent

# Display the unique values in the "Name" column
unique_names = df["Name"].unique()

missing_tifs = []
skipped_tifs = []

for index, row in df.iterrows():
    job_id = row.id
    start_date = row.startdate
    end_date = row.enddate
    tile_name = row.Name

    sarbackscatter_tile_folder = sarbackscatter_folder.joinpath(tile_name)
    os.makedirs(sarbackscatter_tile_folder, exist_ok=True)

    temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent(start_date, end_date)

    print(f"job {job_id} tile {tile_name} start {start_date} end {end_date}\n temporal ext {temporal_extents} \n master temp ext {master_temporal_extent}\n -------")

    for temporal_extent_item in temporal_extents:
        temporal_extent_startdate = temporal_extent_item[0]
        temporal_extent_enddate = temporal_extent_item[1]
        end_date = datetime.strptime(temporal_extent_enddate, "%Y-%m-%d")
        start_date = datetime.strptime(temporal_extent_startdate, "%Y-%m-%d")
        # if start_date > reference_date: continue

        sarbackscatter_filename = f"openEO_{temporal_extent_startdate}Z.tif"
        sarbackscatter_filepath = jobmanager_folder.joinpath(f"job_{job_id}", sarbackscatter_filename)
        if not sarbackscatter_filepath.exists():
            if CHECK_MISSING_BAC:
                scene_infos = query_sentinel1(row.west, row.south, row.east, row.north, start_date, end_date)
                print(f"{sarbackscatter_filepath} doesnt exist")
                if len(scene_infos) >0:
                    print(f"Check {temporal_extent_startdate} - {temporal_extent_enddate} - {tile_name} - {job_id}")
                    missing_tifs.append((temporal_extent_startdate,temporal_extent_enddate, tile_name ))
                    continue
                else:
                    skipped_tifs.append((temporal_extent_startdate, temporal_extent_enddate, tile_name))
                    continue
            else:
                skipped_tifs.append((temporal_extent_startdate, temporal_extent_enddate, tile_name))
                continue
        sarbackscatter_tile_mosaicfilename = f"SARBAC_{tile_name}_{temporal_extent_startdate}_{temporal_extent_item[1]}.tif"
        sarbackscatter_tile_mosaicfilpath = sarbackscatter_tile_folder.joinpath(sarbackscatter_tile_mosaicfilename)
        if not sarbackscatter_tile_mosaicfilpath.exists():
            # shutil.copy(sarbackscatter_filepath, sarbackscatter_tile_mosaicfilpath)
            shutil.move(sarbackscatter_filepath, sarbackscatter_tile_mosaicfilpath)
        print(f"{sarbackscatter_filepath} --> {sarbackscatter_tile_mosaicfilpath}")

print("--- whats here ---")
print("------")
print(missing_tifs)
print("------")
print(skipped_tifs)


