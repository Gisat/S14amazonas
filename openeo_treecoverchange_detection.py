import openeo
from pathlib import Path
import openeo.processes as eop
import geopandas as gpd
from openeo.metadata import CollectionMetadata, BandDimension, TemporalDimension, SpatialDimension
from openeo_utils import get_extended_temporalextents_with_padding, get_temporalextents_mastertemporalextent

# Connect to the openEO backend (e.g., VITO backend)
# Setup the connection
connection = openeo.connect("openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()

DEGUG_NC = False
DEBUG = False

if DEBUG:
    job_prefix = "DEBUG_SARCD"
else:
    job_prefix = "SARCD"

bands = ["VV", "VH"]
resample_method = "near"

def changedetection_backscatter(west, east, north, south, crs_epsg, tile_name, start_time, end_time):
    master_aoi = {
        "west": west,
        "east": east,
        "north": north,
        "south": south,
        "crs": f"EPSG:{crs_epsg}"
    }

    aoi = master_aoi

    url = f"https://s3.waw3-1.cloudferro.com/swift/v1/deforestation/sarbackscatter/stac_dir/{tile_name}/{tile_name}_backscatter_catalog.json"
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
    udf = openeo.UDF.from_file(Path(__file__).parent.resolve() / "udf_mcd_deforestation_detection.py")
    # Apply the UDF to the data cube.
    datacube_udf = datacube_time_as_bands.apply_dimension(
        process=udf,
        dimension="bands"
    )
    target = [band + "_t" + str(i+1).zfill(2) for band in ["MCD", "MCDthreshold", "VVpmin", "VHpmin"] for i in range(number_of_timewindows)]
    datacube_udf = datacube_udf.rename_labels(dimension="bands", target=target)

    udf_ai = openeo.UDF.from_file(Path(__file__).parent.resolve() / "udf_ai_deforestation_detection.py")
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

    # Run this script for testing
    # satellite patch 21LYG
    west, south, east, north = (-54.8225145935671563, -12.1968997799477616, -54.6903701720671549, -12.0434982956194023)
    tile_name = "21LYGsatpatch"
    crs = 4326

    # case 2
    start_time = "2021-05-03"
    end_time =  "2021-05-15"

    temporal_extents, padded_start_time, padded_end_time = get_extended_temporalextents_with_padding(start_time, end_time)
    job = changedetection_backscatter(west, east, north, south, crs, tile_name, padded_start_time, padded_end_time)
    job.start_and_wait()
    results = job.get_results()
    results.download_files()

if __name__ == "__main__":
    main()