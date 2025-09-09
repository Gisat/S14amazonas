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
from O4_openeo_deforestation_detection import changedetection_jm_wrapper

import pandas as pd
import geopandas as gpd

from O7_openeo_backscatter import sarbackscatter_jm_wrapper, get_monthyear_periods_joblist

NUM_PROCESS = 10
# Assuming the provided list is stored in a variable
date_ranges = [('19NEJ', '2022-09-05', '2022-09-29'), ('19NEJ', '2022-09-29', '2022-10-23'), ('19NEJ', '2022-10-23', '2022-11-16'), ('19NEJ', '2022-11-16', '2022-12-10'), ('19NEJ', '2022-12-10', '2023-01-03'), ('19NEJ', '2023-01-03', '2023-01-27'), ('19NEJ', '2023-01-27', '2023-02-20'), ('19NEJ', '2023-02-20', '2023-03-16'), ('19NEJ', '2023-03-16', '2023-04-09'), ('19NEJ', '2023-04-09', '2023-05-03'), ('19NEJ', '2023-05-03', '2023-05-27'), ('19NEJ', '2023-05-27', '2023-06-20'), ('19NEJ', '2023-06-20', '2023-07-14'), ('19NEJ', '2023-07-14', '2023-08-07'), ('19NEJ', '2023-08-07', '2023-08-31'), ('19NEJ', '2023-08-31', '2023-09-24'), ('19NEJ', '2023-09-24', '2023-10-18'), ('19NEJ', '2023-10-18', '2023-11-11'), ('19NEJ', '2023-11-11', '2023-12-05'), ('19NEJ', '2023-12-05', '2023-12-29'), ('22LFP', '2021-08-29', '2021-09-10'), ('22LFP', '2021-09-10', '2021-09-22'), ('22LFP', '2021-09-22', '2021-10-04'), ('22LFP', '2021-10-04', '2021-10-16'), ('22LFP', '2021-10-16', '2021-10-28'), ('22LFP', '2021-10-28', '2021-11-09'), ('22LFP', '2021-11-09', '2021-11-21'), ('22LFP', '2021-11-21', '2021-12-03'), ('22LFP', '2021-12-03', '2021-12-15'), ('22LFP', '2021-12-15', '2022-01-08'), ('22LFP', '2022-01-08', '2022-02-01'), ('22LFP', '2022-02-01', '2022-02-25'), ('22LFP', '2022-02-25', '2022-03-21'), ('22LFP', '2022-03-21', '2022-04-14'), ('22LFP', '2022-04-14', '2022-05-08'), ('22LFP', '2022-05-08', '2022-06-01'), ('22LFP', '2022-06-01', '2022-06-25'), ('22LFP', '2022-06-25', '2022-07-19'), ('22LFP', '2022-07-19', '2022-08-12'), ('22LFP', '2022-08-12', '2022-09-05'), ('19NFJ', '2022-09-05', '2022-09-29'), ('19NFJ', '2022-09-29', '2022-10-23'), ('19NFJ', '2022-10-23', '2022-11-16'), ('19NFJ', '2022-11-16', '2022-12-10'), ('19NFJ', '2022-12-10', '2023-01-03'), ('19NFJ', '2023-01-03', '2023-01-27'), ('19NFJ', '2023-01-27', '2023-02-20'), ('19NFJ', '2023-02-20', '2023-03-16'), ('19NFJ', '2023-03-16', '2023-04-09'), ('19NFJ', '2023-04-09', '2023-05-03'), ('19NFJ', '2023-05-03', '2023-05-27'), ('19NFJ', '2023-05-27', '2023-06-20'), ('19NFJ', '2023-06-20', '2023-07-14'), ('19NFJ', '2023-07-14', '2023-08-07'), ('19NFJ', '2023-08-07', '2023-08-31'), ('19NFJ', '2023-08-31', '2023-09-24'), ('19NFJ', '2023-09-24', '2023-10-18'), ('19NFJ', '2023-10-18', '2023-11-11'), ('19NFJ', '2023-11-11', '2023-12-05'), ('19NFJ', '2023-12-05', '2023-12-29'), ('21LUF', '2023-05-03', '2023-05-27'), ('21LUF', '2023-05-27', '2023-06-20'), ('21LUF', '2023-06-20', '2023-07-14'), ('21LUF', '2023-07-14', '2023-08-07'), ('21LUF', '2023-08-07', '2023-08-31'), ('21LUF', '2023-08-31', '2023-09-24'), ('21LUF', '2023-09-24', '2023-10-18'), ('21LUF', '2023-10-18', '2023-11-11'), ('21LUF', '2023-11-11', '2023-12-05'), ('21LUF', '2023-12-05', '2023-12-29')]
RUN_NAME = "missingchangedetection256"
# Define the input file and work directory paths
input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI.gpkg")
work_dir = Path(f"/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_jobmanagement_{RUN_NAME}")
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
        startdate_str = startdate
        enddate_str = enddate

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
    start_job=changedetection_jm_wrapper,
    job_db=job_database
)


