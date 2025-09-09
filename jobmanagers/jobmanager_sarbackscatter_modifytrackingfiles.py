import pandas as pd
from pathlib import Path
from O7_openeo_backscatter import get_temporalextents_mastertemporalextent

file_path = '/mnt/hddarchive.nfs/amazonas_dir/work_dir/backscatter_jobmanagement_priori14/job_database-priori14.csv'
BACKSCATTER = False
jobmanager_folder = Path(file_path).parent

# Load the CSV file
df = pd.read_csv(file_path)

# Identify jobs with status error/start_failed or missing start time
initial_mask = (
    df['status'].isin(['error', 'start_failed']) |
    df['running_start_time'].isna() |
    (df['running_start_time'].astype(str).str.strip() == '')
)

# Collect job IDs needing update
jobs_to_reset = set(df[initial_mask]['id'])

if BACKSCATTER:
    # Check for missing files and add those job IDs too
    for index, row in df.iterrows():
        job_id = row.id
        start_date = row.startdate
        end_date = row.enddate
        tile_name = row.Name

        try:
            temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent(start_date, end_date)
        except Exception as e:
            print(f"Error with job {job_id}: {e}")
            jobs_to_reset.add(job_id)
            continue

        print(f"job {job_id} tile {tile_name} start {start_date} end {end_date}\n temporal ext {temporal_extents} \n master temp ext {master_temporal_extent}\n -------")


        for temporal_extent_item in temporal_extents:
            temporal_extent_startdate = temporal_extent_item[0]
            sarbackscatter_filename = f"openEO_{temporal_extent_startdate}Z.tif"
            sarbackscatter_filepath = jobmanager_folder.joinpath(f"job_{job_id}", sarbackscatter_filename)
            if not sarbackscatter_filepath.exists():
                print(f"{sarbackscatter_filepath} doesn't exist")
                jobs_to_reset.add(job_id)
        print("--------------")

# Update rows in df
df.loc[df['id'].isin(jobs_to_reset), 'status'] = 'not_started'

# Clear columns for those jobs
columns_to_remove = ['start_time', 'running_start_time', 'cpu', 'memory', 'duration', 'costs']
df.loc[df['id'].isin(jobs_to_reset), columns_to_remove] = None

# Optional: save the updated DataFrame
df.to_csv(file_path, index=False)