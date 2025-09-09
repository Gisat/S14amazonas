import geopandas as gpd
import pandas as pd
import datetime
from pathlib import Path
import os

import os
from pathlib import Path

import openeo
from openeo.extra.job_management import MultiBackendJobManager
from openeo.extra.job_management import create_job_db

import pandas as pd
import geopandas as gpd

from O7_openeo_backscatter import sarbackscatter_jm_wrapper, get_monthyear_periods_joblist

NUM_PROCESS = 10
# Assuming the provided list is stored in a variable
date_ranges = [('19LFJ', datetime.datetime(2024, 4, 3, 0, 0), datetime.datetime(2024, 4, 27, 0, 0)), ('21MUM', datetime.datetime(2021, 7, 24, 0, 0), datetime.datetime(2021, 8, 5, 0, 0)), ('21MUN', datetime.datetime(2024, 5, 21, 0, 0), datetime.datetime(2024, 6, 14, 0, 0)), ('22KBE', datetime.datetime(2024, 2, 15, 0, 0), datetime.datetime(2024, 3, 10, 0, 0)), ('22LBM', datetime.datetime(2021, 6, 30, 0, 0), datetime.datetime(2021, 7, 12, 0, 0)), ('22LBM', datetime.datetime(2022, 8, 12, 0, 0), datetime.datetime(2022, 9, 5, 0, 0)), ('22LBM', datetime.datetime(2023, 5, 27, 0, 0), datetime.datetime(2023, 6, 20, 0, 0)), ('22LBM', datetime.datetime(2024, 5, 21, 0, 0), datetime.datetime(2024, 6, 14, 0, 0)), ('22MHT', datetime.datetime(2021, 1, 13, 0, 0), datetime.datetime(2021, 1, 25, 0, 0)), ('22MHT', datetime.datetime(2021, 11, 9, 0, 0), datetime.datetime(2021, 11, 21, 0, 0))]
RUN_NAME = "missingsarbackscatter256_3"
# Define the input file and work directory paths
input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI.gpkg")
work_dir = Path(f"/mnt/hddarchive.nfs/amazonas_dir/work_dir/backscatter_jobmanagement_{RUN_NAME}")
os.makedirs(work_dir, exist_ok=True)

# Load the input data
assert input_df_path.exists(), f"Input file {input_df_path} does not exist"
input_df = gpd.read_file(input_df_path)

# Prepare the list for the dataframe
job_list = []

# Iterate over the date_ranges and get the bounding box for each tile
for  tile_name, startdate, enddate in date_ranges:
    # Query the GPKG for the tile's bounding box
    tile_data = input_df[input_df["Name"] == tile_name]

    if not tile_data.empty:
        # Extract xmin, xmax, ymin, ymax
        xmin = tile_data["xmin"].values[0]
        xmax = tile_data["xmax"].values[0]
        ymin = tile_data["ymin"].values[0]
        ymax = tile_data["ymax"].values[0]

        # Convert startdate and enddate to the format %Y-%m-%d
        startdate_str = startdate.strftime('%Y-%m-%d')
        enddate_str = enddate.strftime('%Y-%m-%d')

        # Append the result to the job list
        job_list.append({
            "Name": tile_name,
            "startdate": startdate_str,
            "enddate": enddate_str,
            "west": xmin,
            "east": xmax,
            "south": ymin,
            "north": ymax,
            "epsg": 4326
        })

# Create a DataFrame from the job list
job_df = pd.DataFrame(job_list)


job_df.rename(mapper={"xmin": "west", "xmax": "east", "ymin": "south", "ymax": "north"}, axis=1, inplace=True)
job_df["epsg"] = 4326

column_list= ["west", "east", "south", "north", "epsg", "startdate", "enddate",  "Name"]
job_df = job_df[column_list]

connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

if RUN_NAME is not None:
    runstr = RUN_NAME
else:
    runstr = datetime.now().strftime("%Y%m%d-%Hh%M")
print(f"runstr: {runstr}")



job_database = create_job_db(
    path= work_dir / f"job_database-{runstr}.csv",
    df=job_df,
    on_exists="skip"
)

job_manager = MultiBackendJobManager(
    poll_sleep=10,
    root_dir=work_dir
)

job_manager.add_backend(
    "cdse",
    connection=connection,
    parallel_jobs=NUM_PROCESS
)

job_manager.run_jobs(
    df=job_df,
    start_job=sarbackscatter_jm_wrapper,
    job_db=job_database
)


