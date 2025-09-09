import os
from pathlib import Path
from datetime import datetime, timedelta

import openeo
from openeo.extra.job_management import MultiBackendJobManager
from openeo.extra.job_management import create_job_db

import pandas as pd
import geopandas as gpd

from Oa_openeo_utils import get_temporalextents_mastertemporalextent, get_extended_temporalextents_with_padding, get_monthyear_periods_joblist
from O4_openeo_deforestation_detection import changedetection_jm_wrapper

tile_list = None

# tile_list= ['16PFA', '18MYS', '22NDH', '19LCL', '19NEJ', '23MLS', '18PUQ', '17PPL', '22NEF', '22NCJ', '16PHT', '22NDK', '18NXH', '19LDL', '22MGT', '19LEK', '22NDJ', '16QBE', '18LZR', '22NCG', '19LEL', '22MDE', '19NFJ', '22NCK', '19LDK', '22MDU', '22KCG', '19LBL']
# RUN_NAME = "batch3"


# tile_list =list(set(['22LBP', '22LEP', '22MBB', '22MBE', '22MCB', '22MCU','19LFJ', '22MEB', '22MFS', '22MFT', '22MGS', '22MHT',
#                      '17PPL', '18NWF', '16QBE', '19PEK', '19PFK']))
# RUN_NAME = "batchprocess8"

# tile_list =list(set(['19PEK', '19PFK']))
# RUN_NAME = "batchprocess19pek19pfk"


# tile_list = [
#     "16PEU", "16PGA", "16PGC", "16PGT", "20LNQ", "20LPR", "20LQR",
#     "20MPB", "20MPE", "20MQC", "20MQD", "20MQE", "20MQU", "20NPG",
#     "20NPK", "20NQJ", "20NRH", "20NRJ", "21LTF", "21LTG", "21LXG",
#     "21LYK", "21MTP", "22MBT", "22MCA"
# ]
# RUN_NAME = "batchpirori12"


# tile_list = ['20MRA', '20LMQ', '20LPQ', '20MQS', '21MZN', '21MUM', '22LFQ', '22MBU', '21LXF', '21MUS', '16PGS', '22KCE', '20MPS', '21NTA', '20MPD', '21MWR', '21MXQ', '20MND', '22MBA', '19MBM', 'stac_dir', '21MXU', '16PHA', '20NQH', '20MRE', '20NPF', '22LEM', '21MYS', '21MVN', '20MPT', '16PFB']
# RUN_NAME = "batchpriori11"


# tile_list =  ['15QYU', '15QYV', '20LLQ', '20LLR', '20MQA', '20MQB', '20NPH', '20NPJ', '21LWF', '21LWG', '21MWN', '21MWP', '21MXN', '21MXP', '22KHB', '22KHC', '22LBP', '22LEP', '22MBB', '22MBE', '22MCB', '22MCU']
# RUN_NAME = "batchpriori7"

# tile_list = [ '22KCF', '16PHS', '19LFL', '22MGV', '23MMS', '22NEG', '23MMT', '18PWS', '23MLU', '18PWR', '22LCN', '22NDF', '18NUF', '22MHA', '18NXG', '22LGL', '22LGM', '19LDJ', '22NDG', '19LEJ']
# RUN_NAME = "batch2"

# tile_list = ['18MZS', '19PEK', '18NWF', '23MKQ', '22NCH', '22NCF', '18NWG', '17PNK', '22MHS', '22MGU', '19PFK', '22NEH', '23MLT', '22MEA']
# RUN_NAME = "batch1"

tile_list = None
RUN_NAME = "batch256"


# RUN_NAME = "batchpriori14"
## INPUTS
NUM_PROCESS = 10
past_runs = 0
# RUN_NAME =  "detectbatch1" #"poc20230503-20240825"
priority = 256
input_df_path = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/S2_Tiles_MCD_AI_cdtoprocess.gpkg")
work_dir = Path(f"/mnt/hddarchive.nfs/amazonas_dir/work_dir/detection_jobmanagement_{RUN_NAME}")
os.makedirs(work_dir, exist_ok=True)

####
assert input_df_path.exists(), f"Input file {input_df_path} does not exist"
input_df = gpd.read_file(input_df_path)



if priority is not None:
    input_df = input_df[input_df["Priority"] == priority]
if tile_list is not None:
    input_df = input_df[input_df['Name'].isin(tile_list)]
# input_df = input_df[input_df["Name"] == "20LNR"]
print(input_df)

# temporal_extents = get_monthyear_periods_joblist("20230503", "20240825", 1)
temporal_extents = get_monthyear_periods_joblist("20210101", "20241223", 10)


# Creating a new DataFrame with the assigned startdate and enddate for each month-year item
job_df = pd.concat([input_df.assign(startdate=months_year_item[0], enddate=months_year_item[1])
                    for months_year_item in temporal_extents], ignore_index=True)
# Sorting the job_df DataFrame by the 'Name' column (or another column you prefer)
job_df = job_df.sort_values(by='Name', ascending=False).reset_index(drop=True)

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


