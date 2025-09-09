from io import BytesIO
from datetime import datetime, timedelta
import os, json
from pathlib import Path
import tempfile
from typing import Optional, Union
import openeo
from openeo.extra.job_management import MultiBackendJobManager, JobDatabaseInterface
import pandas as pd
import boto3
import fnmatch
from botocore.exceptions import NoCredentialsError, ClientError

import shapely.geometry
import geopandas as gpd


def buffer_aoi(aoi: dict, buffer: int) -> dict:
    # Create a polygon from AOI bounds
    polygon = shapely.geometry.box(aoi["west"], aoi["south"], aoi["east"], aoi["north"])
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=aoi["crs"])

    # Reproject to UTM for buffering in meters
    utm = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm)

    # Apply buffer with square caps
    gdf["geometry"] = gdf["geometry"].buffer(buffer, cap_style="square")

    # Reproject back to original CRS
    gdf = gdf.to_crs(aoi["crs"])

    # Get new bounds
    buffered_polygon = gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = buffered_polygon.bounds

    # Return updated AOI
    return {
        "west": minx,
        "east": maxx,
        "south": miny,
        "north": maxy,
        "crs": aoi["crs"]
    }


def rename_job_database(folder_path, run_name):
    # Construct the original file name
    original_filename = f'job_database-{run_name}.csv'
    original_filepath = os.path.join(folder_path, original_filename)

    # Get the current time in a suitable format (e.g., YYYYMMDD_HHMMSS)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Construct the new file name with current time
    new_filename = f'past_run_{current_time}_job_database-{run_name}.csv'
    new_filepath = os.path.join(folder_path, new_filename)

    # Check if the original file exists
    if os.path.exists(original_filepath):
        # Rename the file
        os.rename(original_filepath, new_filepath)
        print(f"File renamed to {new_filename}")
    else:
        print(f"File {original_filename} not found in {folder_path}")


def filter_jobdf(folder_path, jobdf, run_name, column_list):
    # Step 1: Check for the files
    pattern_1 = f'job_database-{run_name}.csv'
    pattern_2 = f'past_run_*_job_database-{run_name}.csv'

    # Find files matching the patterns
    files_to_check = []
    files_to_check.extend(fnmatch.filter(os.listdir(folder_path), pattern_1))
    files_to_check.extend(fnmatch.filter(os.listdir(folder_path), pattern_2))

    if len(files_to_check) == 0:
        return jobdf

    # Step 2: Read the CSV files into DataFrames
    dfs = []
    for file in files_to_check:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Combine all the DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Step 3: Handle priority for rows with the same block_id and year
    # First, sort by status to prioritize 'finished'
    combined_df['priority'] = combined_df['status'].apply(lambda x: 1 if x == 'finished' else 0)
    combined_df.sort_values(by=['block_id', 'year', 'priority'], ascending=[True, True, False], inplace=True)

    # Drop duplicates, keeping the first (which will be the 'finished' ones due to sorting)
    condensed_df = combined_df.drop_duplicates(subset=['block_id', 'year'], keep='first')
    condensed_df = condensed_df[condensed_df['priority'] == 1]

    # Step 4: Filter out rows from jobdf that are in condensed_df # Assuming jobdf is in a CSV file
    # Ensure both DataFrames have the same data type for 'year'
    jobdf['year'] = pd.to_numeric(jobdf['year'], errors='coerce')
    condensed_df['year'] = pd.to_numeric(condensed_df['year'], errors='coerce')

    filtered_df = jobdf.merge(
        condensed_df[['block_id', 'year']],
        on=['block_id', 'year'],
        how='left',
        indicator=True
    ).query('_merge == "left_only"').drop(columns=['_merge'])

    rename_job_database(folder_path, run_name)
    filtered_df = filtered_df[column_list]
    return filtered_df


class ClassificationJobManager(MultiBackendJobManager):
    def __init__(self, poll_sleep: int, root_dir: Optional[Union[str, Path]] = None):
        super().__init__(poll_sleep=poll_sleep, root_dir=root_dir)
        
    def on_job_done(self, job: openeo.BatchJob, row):
        """
        Handles jobs that have finished. Can be overridden to provide custom behaviour.

        Default implementation downloads the results into a folder containing the title.

        :param job: The job that has finished.
        :param row: DataFrame row containing the job's metadata.
        """
        # Add a prefix to the job ID or job directory path
        prefix = row.Name  # Specify your desired prefix
        job_metadata = job.describe()

        # Use the prefix in the job directory path
        prefixed_job_dir = self._root_dir / f"{prefix}-{job.job_id}"
        metadata_path = self._root_dir / f"{prefix}-{job.job_id}"/ f"job_{job.job_id}.json"

        if not prefixed_job_dir.exists():
            prefixed_job_dir.mkdir(parents=True)

        # Download the results with the prefixed directory
        job.get_results().download_files(target=prefixed_job_dir)

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(job_metadata, f, ensure_ascii=False)

class S3JobDatabase(JobDatabaseInterface):
    def __init__(self, bucket_name: str, object_key: str):
        self.bucket_name = bucket_name
        self.object_key = object_key

        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        endpoint_url = "https://s3.waw3-1.cloudferro.com"
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError("AWS credentials are not set in the os environment") 

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url
        )

    def exists(self) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=self.object_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise RuntimeError(f"Failed to check if file exists in S3: {e}")

    def read(self) -> pd.DataFrame:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.object_key)
            file_stream = BytesIO(response['Body'].read())
            return pd.read_parquet(file_stream)
        except ClientError as e:
            raise RuntimeError(f"Failed to read file from S3: {e}")

    def persist(self, df: pd.DataFrame):
        with tempfile.NamedTemporaryFile() as tmpfile:
            tmpfile_name = tmpfile.name
            df.to_parquet(tmpfile_name)

            try:
                with open(tmpfile_name, 'rb') as f:
                    self.s3_client.upload_fileobj(f, self.bucket_name, self.object_key)
            except (NoCredentialsError, ClientError) as e:
                raise RuntimeError(f"Failed to upload file to S3: {e}")



def get_detection_temporalextents_mastertemporalextent(start_date, end_date):

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Define the specific date for comparison (2021-12-27)
    reference_date = datetime(2021, 12, 28)

    temporal_extents = []

    start_date_window = start_date
    for i in range(5):
        if start_date_window < reference_date:
            start_date_window -= timedelta(days=12)
        else:
            start_date_window -= timedelta(days=24)

    end_date_window = end_date
    for i in range(5):
        if end_date_window + timedelta(days=12) < reference_date:
            end_date_window += timedelta(days=12)
        else:
            end_date_window += timedelta(days=24)

    current_date = start_date_window
    end_date = end_date_window


    while current_date < end_date:
        # Append the current year-month to the periods list
        temporal_extent = [current_date.strftime('%Y-%m-%d')]
        if current_date < reference_date:
            current_date += timedelta(days=12)
            used_timedelta = 12
        else:
            current_date += timedelta(days=24)
            used_timedelta = 24
        temporal_extent.append(current_date.strftime('%Y-%m-%d'))
        temporal_extents.append(temporal_extent)

    # Temporal extent from first to last date
    master_temporal_extent = [temporal_extents[0][0], temporal_extents[-1][1]]
    return temporal_extents, master_temporal_extent