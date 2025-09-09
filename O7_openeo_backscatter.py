
import openeo
from datetime import datetime, timedelta
from pathlib import Path
import geopandas as gpd
import numpy as np
from Oa_openeo_utils import get_temporalextents_mastertemporalextent, get_monthyear_periods_joblist
import json

#######
def sarbackscatter_jm_wrapper(
        row: gpd.GeoDataFrame,
        connection: openeo.Connection,
        *args,
        **kwargs,
) -> openeo.BatchJob:
    """
    This function is a wrapper around the LC_prediction_inference function. It is used to pass the correct
    parameters to the LC_prediction_inference function when running this function in a job manager.
    """

    bbox = {
        "west": float(row.west),
        "east": float(row.east),
        "north": float(row.north),
        "south": float(row.south),
        "crs": f"EPSG:{row.epsg}"
    }
    tile_name = row.Name

    #############################################
    temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent(row.startdate, row.enddate)

    load_collection_options = {
        "polarisation": lambda pol: pol == "VV&VH"
    }
    s1_collection = (connection.load_collection('SENTINEL1_GRD', bands=['VV', 'VH'], properties=load_collection_options, spatial_extent=bbox)
     .filter_temporal(extent=master_temporal_extent)).sar_backscatter(
        coefficient="sigma0-ellipsoid",
        elevation_model="COPERNICUS_30",
        local_incidence_angle=False
    )
    s1_collection = s1_collection.resample_spatial(resolution=20, method="near")
    # Apply temporal aggregation over intervals
    s1grd = s1_collection.aggregate_temporal(
        intervals=temporal_extents,
        reducer="mean"  # Or any other reducer like 'median', 'min', 'max'
    )
    s1grd = s1grd.filter_bbox(bbox)

    job_options = {
        "executor-memory": "3G",
        "executor-memoryOverhead": "500m",
        "driver-memory": "2G",
        "python-memory": "2500m",
        "driver-memoryOverhead": "2G",
        "soft-errors": True,
        "max_executors": 10
    }
    return s1grd.create_job(title=f"SAR - {tile_name} - {master_temporal_extent[0]} - {master_temporal_extent[1]}",
                                    job_options=job_options)

def main():
    acq_frequency = 12
    month_year = "3-2024"
    tile_name = "21LYGsatpatch"
    west, south, east, north = (-54.8225145935671563, -12.1968997799477616, -54.6903701720671549, -12.0434982956194023)
    bbox = {
        "west": west,
        "east": east,
        "north": north,
        "south": south,
        "crs": "EPSG:4326"
    }
    crs_epsg = 4326
    # connection = openeo.connect("openeo.dataspace.copernicus.eu")
    # connection.authenticate_oidc()
    download_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/openeojobmanagement/sarbackscatter")
    download_folder = download_root_folder.joinpath(tile_name)

    month_years = get_monthyear_periods_joblist(20210101, 20250329, 10)
    total_extents = 0
    for month_year_item in month_years:
        print(f"{month_year_item[0]} - {month_year_item[1]}")
        temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent(month_year_item[0], month_year_item[1])
        print(temporal_extents)
        print(master_temporal_extent)
        total_extents += len(temporal_extents)
        print("--------------")
        print("-X---------X-")
    print(total_extents)
    exit(0)
    #############################################

    # Temporal extent from first to last date
    master_temporal_extent = [datetime.strptime(temporal_extents[0][0], "%Y-%m-%d"), datetime.strptime(temporal_extents[-1][1], "%Y-%m-%d")]

    load_collection_options = {
        "polarisation": lambda pol: pol == "VV&VH"
    }
    s1grd = (connection.load_collection('SENTINEL1_GRD', bands=['VV', 'VH'],
                                        properties=load_collection_options, spatial_extent=bbox)
     .filter_temporal(extent=master_temporal_extent)).sar_backscatter(
        coefficient="sigma0-ellipsoid",
        elevation_model="COPERNICUS_30",
        local_incidence_angle=False
    )
    # Apply temporal aggregation over intervals
    s1_collection = s1grd.aggregate_temporal(
        intervals=temporal_extents,
        reducer="mean"  # Or any other reducer like 'median', 'min', 'max'
    )
    job = s1_collection.filter_bbox(bbox).execute_batch(out_format="GTiff")
    # results = job.get_results()
    # results.download_file(f"openeo_backscatter_21LYG.tiff")
    job.get_results().download_files(str(download_folder))



if __name__ == "__main__":
    main()