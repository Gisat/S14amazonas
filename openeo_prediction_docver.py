
import openeo
from pathlib import Path

# Test parameters
south = -12.73714945
north = -11.753685105
west = -55.165335994
east = -54.158521354
bbox = {"west": round(west, 2), "south": round(south, 2), "east": round(east, 2), "north": round(north, 2), 'crs': 4326}
base_output_path = Path("s1datacube")
base_output_path.mkdir(exist_ok=True)


c=openeo.connect("openeo.dataspace.copernicus.eu")
c.authenticate_oidc()


S1_collection = "SENTINEL1_GRD"
if "SENTINEL1_GRD_SIGMA0" in c.list_collection_ids():
    S1_collection = "SENTINEL1_GRD_SIGMA0"


sentinel1 = c.load_collection(
    S1_collection,
    temporal_extent = ["2022-06-04", "2022-08-04"],
    bands = ["VV","VH"]
)

if S1_collection == "SENTINEL1_GRD":
    sentinel1 = sentinel1.sar_backscatter(
        coefficient='sigma0-ellipsoid',
        local_incidence_angle=False,
        elevation_model='COPERNICUS_30')

sentinel1 = sentinel1.aggregate_temporal_period("dekad",reducer="median")\
    .apply_dimension(dimension="t", process="array_interpolate_linear")



dependencies_url = "https://artifactory.vgt.vito.be:443/auxdata-public/openeo/onnx_dependencies.zip"
models_url = "https://artifactory.vgt.vito.be:443/artifactory/auxdata-public/openeo/parcelDelination/BelgiumCropMap_unet_3BandsGenerator_Models.zip"
job_options = {
    "udf-dependency-archives": [
        f"{dependencies_url}#onnx_deps",
        f"{models_url}#onnx_models",
    ]
}




my_udf = openeo.UDF("""
from openeo.udf import XarrayDataCube
from openeo.udf.debug import inspect

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    array = cube.get_array()
    inspect(array,level="ERROR",message="inspecting input cube")
    array.values = 0.0001 * array.values
    return cube
""")

fused = sentinel1.apply_neighborhood(
    process=openeo.UDF.from_file("udf_segmentation.py"),
    size=[
        {"dimension": "x", "value": 224, "unit": "px"},
        {"dimension": "y", "value": 224, "unit": "px"},
    ],
    overlap=[
        {"dimension": "x", "value": 16, "unit": "px"},
        {"dimension": "y", "value": 16, "unit": "px"},
    ],
)


segmentation_job = fused.create_job(
    title="segmentation_onnx_job",
    out_format="NetCDF",
    job_options=job_options
)
segmentation_job.start_and_wait()
segmentation_job.download_result(base_output_path / "delineation.nc")



#      apply_neighborhood(my_udf, size=[
#     {'dimension': 'x', 'value': 224, 'unit': 'px'},
#     {'dimension': 'y', 'value': 224, 'unit': 'px'}
# ], overlap=[
#     {'dimension': 'x', 'value': 16, 'unit': 'px'},
#     {'dimension': 'y', 'value': 16, 'unit': 'px'}
# ])

job=fused.filter_bbox(bbox).execute_batch("result.nc", title="Sentinel composite", filename_prefix="merged_cube")