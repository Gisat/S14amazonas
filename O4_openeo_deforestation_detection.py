import openeo
from pathlib import Path
import openeo.processes as eop
import geopandas as gpd
from openeo.metadata import CollectionMetadata, BandDimension, TemporalDimension, SpatialDimension
from O7_openeo_backscatter import get_temporalextents_mastertemporalextent
from datetime import timedelta, datetime
from Oa_openeo_utils import get_extended_temporalextents_with_padding

# Connect to the openEO backend (e.g., VITO backend)
# You can replace this URL with the specific backend you're using
# Setup the connection
connection = openeo.connect("openeo.dataspace.copernicus.eu")
# connection.authenticate_oidc().authenticate_basic("grasslandwatch@gmail.com", "Productionenvironment1234!")
# connection.authenticate_oidc_device()
# Authenticate (if needed, this could be skipped if not required)
connection.authenticate_oidc()


DEGUG_NC = False
DEBUG = False

if DEBUG:
    job_prefix = "DEBUG_SARCD"
else:
    job_prefix = "SARCD"

bands = ["VV", "VH"]
resample_method = "near"

# full 21LYG extent
# north = -11.7536266487057866
# south = -12.7372447835033018
# east = -54.1584601180090601
# west = -55.1654233290762974

# small buffer
# north = -11.8536
# south = -12.6372
# east = -54.258460
# west = -55.06542

# datacube_udf = datacube.apply_dimension(
#     process=udf,
#     size=[
#         {"dimension": "x", "value": 400, "unit": "px"},
#         {"dimension": "y", "value": 400, "unit": "px"},
#     ],
#     overlap=[
#         {"dimension": "x", "value": 0, "unit": "px"},
#         {"dimension": "y", "value": 0, "unit": "px"},
#     ])

# Optionally, define additional operations on the datacube, like rescaling or masking
# For now, we just download the DEM data as is

# job_options = {
#     "executor-memory": "4G",
#     "executor-memoryOverhead": "3G",
#     "driver-memory": "3G",
#     "driver-memoryOverhead": "3G",
#     "soft-errors": True,
#     # "max_executors": 15,
#     "udf-dependency-archives": [
#     f"{DEPENDENCY_URL}#onnx_deps",
#     f"{MODEL_URL}#onnx_models"]
# }


def changedetection_backscatter(west, east, north, south, crs_epsg, tile_name, start_time, end_time):
    master_aoi = {
        "west": west,
        "east": east,
        "north": north,
        "south": south,
        "crs": f"EPSG:{crs_epsg}"
    }

    aoi = master_aoi #buffer_aoi(master_aoi, buffer=-5)


    url = f"https://s3.waw3-1.cloudferro.com/swift/v1/deforestation/sarbackscatter/stac_dir/{tile_name}/{tile_name}_backscatter_catalog.json"
    # url = "https://s3.waw3-1.cloudferro.com/swift/v1/gisat-archive/SAR/21LYG/21LYG_catalog.json"
    temporal_extent = [start_time, end_time]

    temporal_extents, master_temporal_extent = get_temporalextents_mastertemporalextent(temporal_extent[0],
                                                                                        temporal_extent[1])
    number_of_timewindows = len(temporal_extents) - 10 + 1
    print(f"temporal extent: {temporal_extents}, master: {master_temporal_extent}, number of windows {number_of_timewindows}")

    model_relative_path = f"amazonas/ml_models/amazonas_ai_cnn.zip"
    MODEL_URL = f"https://s3.waw3-1.cloudferro.com/swift/v1/{model_relative_path}"

    # Add the onnx dependencies to the job options. You can reuse this existing dependencies archive
    DEPENDENCY_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/openeo/onnx_dependencies_1.16.3.zip"

    # Define the spatial dimensions based on the bounding box and CRS
    x_extent = [aoi["west"], aoi["east"]]
    y_extent = [aoi["south"], aoi["north"]]

    stac_item = connection.load_stac(
        url=url,
        spatial_extent=aoi,
        temporal_extent=master_temporal_extent,
        bands=bands
    )

    stac_item.metadata = CollectionMetadata(
        metadata={},
        dimensions=[
            BandDimension(
                name="bands",
                bands=[openeo.metadata.Band(name=band) for band in bands]
            ),
            TemporalDimension(
                name="t",
                extent=temporal_extents
            ),
            SpatialDimension(
                name="x",
                extent=x_extent,
                crs=f"EPSG:{crs_epsg}"
            ),
            SpatialDimension(
                name="y",
                extent=y_extent,
                crs=f"EPSG:{crs_epsg}"
            )
        ]
    )

    datacube = stac_item.resample_spatial(resolution=20, method=resample_method)
    datacube = datacube.aggregate_temporal(intervals=temporal_extents, reducer="mean")
    datacube = datacube.apply_dimension(dimension="t", process="array_interpolate_linear")
    datacube = datacube.apply(lambda x: 10 * eop.log(x, 10))
    # datacube = datacube.resample_spatial(projection=4326)

    # Reduce to a single array by collapsing the time dimension into a single stack
    datacube_time_as_bands = datacube.apply_dimension(
        dimension='t',
        target_dimension='bands',
        process=lambda d: eop.array_create(data=d)
    )
    band_names = [band + "_t" + str(i+1).zfill(2) for band in ["VV", "VH"] for i in range(len(temporal_extents))]
    print(f"band names {band_names}")
    datacube_time_as_bands = datacube_time_as_bands.rename_labels('bands', band_names)

    if DEGUG_NC:
        job = datacube_time_as_bands.create_job(out_format="GTiff")  # , job_options=job_options)
        job.start_and_wait()
        results = job.get_results()
        results.download_file(f"result_{tile_name}_bac_udf.tiff")

    # Load the UDF from a file.
    udf = openeo.UDF.from_file(Path(__file__).parent.resolve() / "O5_udf_deforestation_detection.py")
    # Apply the UDF to the data cube.
    datacube_udf = datacube_time_as_bands.apply_dimension(
        process=udf,
        dimension="bands"
    )
    target = [band + "_t" + str(i+1).zfill(2) for band in ["MCD", "MCDthreshold", "VVpmin", "VHpmin"] for i in range(number_of_timewindows)]
    datacube_udf = datacube_udf.rename_labels(dimension="bands", target=target)

    udf_ai = openeo.UDF.from_file(Path(__file__).parent.resolve() / "O6_udf_amazonas_ai.py")
    # Apply the UDF to the data cube.
    datacube_ai_udf = datacube_time_as_bands.apply_neighborhood(
        process=udf_ai,
        size=[
            {"dimension": "x", "value": 192, "unit": "px"},
            {"dimension": "y", "value": 192, "unit": "px"},
        ],
        overlap=[
            {"dimension": "x", "value": 32, "unit": "px"},
            {"dimension": "y", "value": 32, "unit": "px"},
        ])
    # target = ["AIMCD"]
    target = [band + "_t" + str(i + 1).zfill(2) for band in ["AIMCD"] for i in
              range(number_of_timewindows)]
    datacube_ai_udf = datacube_ai_udf.rename_labels(dimension="bands", target=target)

    change_detection = datacube_udf.merge_cubes(datacube_ai_udf)

    # job_options = {
    #     "executor-memory": "2500m",
    #     "executor-memoryOverhead": "2G",
    #     "driver-memory": "2G",
    #     "driver-memoryOverhead": "2G",
    #     "soft-errors": True,
    #     "max_executors": 10,
    #     "udf-dependency-archives": [
    #     f"{DEPENDENCY_URL}#onnx_deps",
    #     f"{MODEL_URL}#onnx_models"]
    # }


    job_options = {
        "executor-memory": "5G",
        "python-memory": "3500m", # check this
        "soft-errors": True,
        "max-executors": 10,
        "udf-dependency-archives": [
        f"{DEPENDENCY_URL}#onnx_deps",
        f"{MODEL_URL}#onnx_models"]
    }

    change_detection = change_detection.apply(process=eop.int)
    detection_first_date = temporal_extents[5][0]
    detection_last_date = temporal_extents[number_of_timewindows + 5][0]
    save_options = {"filename_prefix": f"DEC_{tile_name}_{detection_first_date}_{detection_last_date}", "dtype": "int16", "separate_asset_per_band": True}
    change_detection = change_detection.save_result(options=save_options)
    return change_detection.create_job(title=f"{job_prefix} - {tile_name} - {master_temporal_extent[0]} - {master_temporal_extent[1]}",
                                    job_options=job_options)


def changedetection_jm_wrapper(
        row: gpd.GeoDataFrame,
        connection: openeo.Connection,
        *args,
        **kwargs,
) -> openeo.BatchJob:



    west, east, north, south = row.west, row.east, row.north, row.south
    crs = row.epsg
    tile_name = row.Name

    start_time = row.startdate
    end_time = row.enddate
    #############################################
    temporal_extents, padded_start_time, padded_end_time = get_extended_temporalextents_with_padding(start_time, end_time)
    changedetection_backscatter_job = changedetection_backscatter(west, east, north, south, crs, tile_name, padded_start_time, padded_end_time)

    return changedetection_backscatter_job

def main():

    # small buffer
    # north = -11.8536
    # south = -12.6372
    # east = -54.258460
    # west = -55.06542
    # tile_name = "21LYGsmallbuffer"

    # satellite patch 21LYG
    west, south, east, north = (-54.8225145935671563, -12.1968997799477616, -54.6903701720671549, -12.0434982956194023)
    tile_name = "21LYGsatpatch"
    crs = 4326

    # west, south, east, north = (-55.1638792009850079,-12.4940553106659173, -54.2122532979983021,-11.7558354090227191)
    # tile_name = "21LYGside"
    # crs = 4326

    # west, south, east, north =  (499960.0000000000000000,8990040.0000000000000000, 610080.0000000000000000,9100140.0000000000000000)
    # tile_name = "20LNR"
    # crs = 32720

    # case 1
    # start_time_master = datetime.datetime.strptime("2021-01-04", "%Y-%m-%d")
    # end_time_master = datetime.datetime.strptime("2021-05-04", "%Y-%m-%d")
    #
    # for i in range (8):
    #     start_time = start_time_master.strftime('%Y-%m-%d')
    #     end_time = end_time_master.strftime('%Y-%m-%d')
    #     print(f" start time {start_time} end time {end_time}")
    #
    #     job = changedetection_backscatter(west, east, north, south, crs, tile_name, start_time, end_time)
    #     # Save the result as a GeoTIFF file
    #     job.start_and_wait()
    #     results = job.get_results()
    #     results.download_file(f"MCD_{start_time}_{end_time}.tiff")
    #
    #     start_time_master += timedelta(days=12)
    #     end_time_master += timedelta(days=12)

    # case 2
    start_time = "2021-05-03"
    end_time =  "2021-05-15"

    temporal_extents, padded_start_time, padded_end_time = get_extended_temporalextents_with_padding(start_time, end_time)
    job = changedetection_backscatter(west, east, north, south, crs, tile_name, padded_start_time, padded_end_time)
    job.start_and_wait()
    results = job.get_results()
    # results.download_file(f"DEC_{tile_name}_{start_time}_{end_time}.tiff")
    results.download_files("/mnt/hddarchive.nfs/amazonas_dir/work_dir/tmp")

if __name__ == "__main__":
    main()