from datetime import datetime, timedelta
import openeo.processes as eop
import openeo
from pathlib import Path
from openeo.api.process import Parameter

DEBUG= False

# Connect and authenticate to openEO back-end
connection = openeo.connect("openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()

# forest
# -72.9740036300407411,1.9102176002040647 : -72.7452957453280220,2.1086008564781480

# marsh
# -72.8869927591712923,2.4539797025543422 : -72.7443260537405507,2.5777300735605664

# small road path
# west, south, east, north = -72.4702925867019445,1.7073639256367503, -72.2053094178994712,1.9365211118118424


# marsh
west, south, east, north = -72.8869927591712923,2.4539797025543422, -72.7443260537405507,2.5777300735605664

resample_method = "near"
tile_name = "18NYH"
extra_name = "marsh"

def process_time_period(time_period_count):
    # Input: Detection time
    detection_time = "2019-01-04"  # Example input
    detection_time = datetime.strptime(detection_time, "%Y-%m-%d") + timedelta(days=12)*time_period_count
    detection_time = detection_time.strftime("%Y-%m-%d")

    # Convert to datetime
    detection_date = datetime.strptime(detection_time, "%Y-%m-%d")

    # Create intervals
    intervals = []

    # Calculate 5 intervals before the detection date
    current_date = detection_date - timedelta(days=12 * 5)

    for _ in range(10):
        end_date = current_date + timedelta(days=12)
        intervals.append([current_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")])
        current_date = end_date

    # Temporal extent from first to last date
    temporal_extent = [datetime.strptime(intervals[0][0], "%Y-%m-%d"), datetime.strptime(intervals[-1][1], "%Y-%m-%d")]

    print("----------------------------------------------------------------")
    print(f"--- Detection Time: {detection_time} ---")
    print(f"Intervals: {intervals}")
    print(f"Temporal Extent: {temporal_extent[0]} to {temporal_extent[1]}")



    # (-54.79845,-12.10473
    #  -54.80161,-12.23477
    #  -54.66517,-12.23423
    #  -54.66531,-12.10863)
    s1grd = (connection.load_collection('SENTINEL1_GRD', bands=['VV', 'VH'])
     .filter_bbox(west=west, east=east, north=north, south=south)
     .filter_temporal(extent=temporal_extent)).sar_backscatter(
        coefficient="sigma0-ellipsoid",
        elevation_model="COPERNICUS_30",
        local_incidence_angle=False
    )
    datacube = s1grd.ard_normalized_radar_backscatter().aggregate_temporal(intervals=intervals, reducer="mean")
    # Convert SAR values to dB
    datacube = datacube.resample_spatial(resolution=20, method="near")
    datacube = datacube.apply(lambda x: 10 * eop.log(x, 10))


    job_options = {
        "executor-memory": "1G"
    }


    # Reduce to a single array by collapsing the time dimension into a single stack
    datacube_time_as_bands = datacube.apply_dimension(
        dimension='t',
        target_dimension='bands',
        process=lambda d: eop.array_create(data=d)
    )
    band_names = [band + "_t" + str(i+1).zfill(2) for band in ["VV", "VH"] for i in range(10)]
    print(f"band names {band_names}")
    datacube_time_as_bands = datacube_time_as_bands.rename_labels('bands', band_names)

    if DEBUG:
        job = datacube.create_job(job_options=job_options)

        job.start_and_wait()
        results = job.get_results()

        for asset in results.get_assets():
            asset.download()


    # Load the UDF from a file.
    udf = openeo.UDF.from_file(Path(__file__).parent.resolve() / "O5_udf_deforestation_detection.py")

    # Apply the UDF to the data cube.
    datacube_udf = datacube_time_as_bands.apply_neighborhood(
        process=udf,
        size=[
            {"dimension": "x", "value": 400, "unit": "px"},
            {"dimension": "y", "value": 400, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": 0, "unit": "px"},
            {"dimension": "y", "value": 0, "unit": "px"},
        ])


    # Optionally, define additional operations on the datacube, like rescaling or masking
    # For now, we just download the DEM data as is

    job_options = {
        "executor-memory": "1500m",
        "executor-memoryOverhead": "2G",
        "driver-memory": "2G",
        "driver-memoryOverhead": "2G",
        "soft-errors": True,
        "max_executors": 15
    }

    # Save the result as a GeoTIFF file
    job = datacube_udf.create_job(out_format="GTiff") #, job_options=job_options)
    job.start_and_wait()
    results = job.get_results()
    results.download_file(f"openeo_detection_{tile_name}{extra_name}_{detection_time}_{resample_method}.tiff")


if __name__ == "__main__":
    num_processes = 1  # Use all available cores
    for time_instance in range(87):
        retries = 5
        for attempt in range(retries):
            try:
                process_time_period(time_instance)
                break  # Exit retry loop if successful
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed to process time instance {time_instance} after {retries} attempts.")
                continue
