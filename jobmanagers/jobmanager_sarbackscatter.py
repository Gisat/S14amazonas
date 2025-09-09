import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import openeo  # For OpenEO API interactions
from openeo.extra.job_management import MultiBackendJobManager
import pandas as pd
import geopandas as gpd
from openeo.extra.job_management import create_job_db
from jobmanagers.jobmanager_utils.jobmanager_utils import ClassificationJobManager
from O7_openeo_backscatter import sarbackscatter_jm_wrapper, get_monthyear_periods_joblist

## INPUTS
NUM_PROCESS = 10
past_runs = 0
RUN_NAME = "priorisar256"
priority = 256
input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI_sartoprocess.gpkg")
work_dir = Path(f"/mnt/hddarchive.nfs/amazonas_dir/work_dir/backscatter_jobmanagement_{RUN_NAME}")
os.makedirs(work_dir, exist_ok=True)

####
assert input_df_path.exists(), f"Input file {input_df_path} does not exist"
input_df = gpd.read_file(input_df_path)

# if priority is not None:
input_df = input_df[(input_df["Priority"] == priority)]
# input_df = input_df[input_df["Name"] == RUN_NAME]

month_years = get_monthyear_periods_joblist(20201102, 20250329, 15)
# months_years = months_years[::-1]
job_df = pd.concat([input_df.assign(startdate=months_year_item[0], enddate = months_year_item[1]) for months_year_item in month_years], ignore_index=True)

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







