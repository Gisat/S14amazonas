# https://s3.waw3-1.cloudferro.com/swift/v1/stac_test/stac_dir/ndvi/ndvi_catalog.json

import openeo
import json
from pathlib import Path
import openeo.processes as eop
from openeo.processes import array_element, subtract, array_create, rename_labels


# Connect to the openEO backend (e.g., VITO backend)
# You can replace this URL with the specific backend you're using
# Setup the connection
connection = openeo.connect("openeo.dataspace.copernicus.eu")
# connection.authenticate_oidc().authenticate_basic("grasslandwatch@gmail.com", "Productionenvironment1234!")
connection.authenticate_oidc_device()
# Authenticate (if needed, this could be skipped if not required)
# connection.authenticate_oidc()

# Define the area of interest (AOI) using a bounding box (or GeoJSON if available)
# Example bounding box for a region (replace with your desired area)

west, south, east, north = (-55.3030463987837777,-4.5142345136488746, -54.1789278464915895,-3.8562537197872184)
crs_epsg = 4326


aoi = {
    "west": west,
    "east": east,
    "north": north,
    "south": south,
    "crs": f"EPSG:{crs_epsg}"
}
# -55.059959007,-54.881706667,-12.654733931,-12.447807500
# aoi = {
#     "west": -55.059959007,
#     "east": -54.881706667,
#     "north": -12.447807500,
#     "south": -12.654733931,
#     "crs": "EPSG:4326"
# }

# -55.16530653244335,
# -12.737140243568525,
# -54.15849189244335,
# -11.753681403568525

# Load the DEM collection from the STAC catalog (e.g., "DEM_aspec_30m")
# You can browse collections at https://radiantearth.github.io/stac-browser/#/external/stac.openeo.vito.be/
# collection_id = "DEM_slope_30m"  # Change this to the appropriate collection if needed
# datacube = connection.load_stac(
#     url="https://stac.openeo.vito.be/collections/DEM_slope_10m",
#     spatial_extent=aoi,
#     temporal_extent=None,  # Set this if you want data for a specific time range
#     bands=["SLP10"]  # Specify the band(s) if necessary
# )

# url = "https://s3.waw3-1.cloudferro.com/swift/v1/supportivedata/mask/stac_dir/forestelevationmask/forestelevationmask/collection.json"
url = "https://s3.waw3-1.cloudferro.com/swift/v1/supportivedata/mask/stac_dir/forestmask/forest_mask_2020/forest_mask_2020.json"
bands = ["forestmask"]
resample_method = "near"

temporal_extent = [f"2020-01-01", f"2020-03-31"]
# temporal_extent = [f"2021-05-04", f"2021-08-22"]
# temporal_extent = [f"2021-01-04", f"2021-01-06"]

stac_item = connection.load_stac(
    url=url,
    spatial_extent=aoi,
    temporal_extent=temporal_extent,
    bands=bands
)

stac_datacube = stac_item.reduce_temporal('last')
# Apply nearest-neighbor resampling to avoid resolution changes
datacube = stac_datacube.resample_spatial(projection=32620, method=resample_method)
datacube = datacube.resample_spatial(resolution=10)

job_options = {
    "executor-memory": "1000m",
    "executor-memoryOverhead": "1G",
    "driver-memory": "1G",
    "driver-memoryOverhead": "1G",
    "soft-errors": True,
    "max_executors": 15
}

# Save the result as a GeoTIFF file
job = datacube.create_job(out_format="GTiff") #, job_options=job_options)
job.start_and_wait()
results = job.get_results()
results.download_file(f"forest_{resample_method}_neg.tiff")
