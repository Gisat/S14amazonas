import os
from pathlib import Path
import geopandas as gpd
from Oa_openeo_utils import get_temporalextents_mastertemporalextent
from datetime import datetime
from jobmanagers.data_manipulation.sentinel1_query import query_sentinel1

temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent("2020-11-02",
                                                                                    "2025-03-29")

input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI.gpkg")
input_df = gpd.read_file(input_df_path)

sarbackscatter_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter")

# Display the unique values in the "Name" column
tile_list = input_df["Name"].unique()

missing_files = []
for tile_list_item in tile_list:
    if not tile_list_item == "21LYG": continue
    temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent("2020-11-02",
                                                                                        "2025-03-29")
    print(f" -- {tile_list_item} --")

    for temporal_extent_item in temporal_extents:
        temporal_extent_startdate = temporal_extent_item[0]
        temporal_extent_enddate = temporal_extent_item[1]
        end_date = datetime.strptime(temporal_extent_enddate, "%Y-%m-%d")
        start_date = datetime.strptime(temporal_extent_startdate, "%Y-%m-%d")

        row = input_df[input_df['Name'] == tile_list_item].iloc[0]
        scene_infos = query_sentinel1(row.xmin, row.ymin, row.xmax, row.ymax, start_date, end_date)
        if len(scene_infos) == 0:
            missing_files.append((tile_list_item, start_date, end_date))
print(missing_files)





