import logging
import openeo
import openeo.processes as eop
import datetime
from pathlib import Path
import json

from openeo_gfmap.manager import _log
from openeo_gfmap import TemporalContext, Backend, BackendContext, FetchType
from openeo_gfmap.manager.job_splitters import split_job_hex, split_job_s2grid
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager import _log
from openeo_gfmap.backend import cdse_connection, vito_connection
from openeo_gfmap.fetching import build_sentinel2_l2a_extractor, build_sentinel1_grd_extractor


class MyLoggerFilter(logging.Filter):
    def __init__(self, log_name):
        super().__init__()
        self.log_name = log_name

    def filter(self, record):
        return record.name == self.log_name

# Initialize logging
_log = logging.getLogger(__name__)
_log.addFilter(MyLoggerFilter(_log.name))

def timesteps_as_bands(base_features):
    band_names = [band + "_m" + str(i+1) for band in base_features.metadata.band_names for i in range(12)]
    result =  base_features.apply_dimension(
        dimension='t',
        target_dimension='bands',
        process=lambda d: eop.array_create(data=d)
    )
    return result.rename_labels('bands', band_names)


def get_s1_features(
        s1_datacube
) -> openeo.DataCube:
    # Aggregate the datacube by dekad (10-day periods)
    s1_dekad = s1_datacube.aggregate_temporal_period(period="dekad", reducer="mean")

    # Interpolate the values linearly to fill any gaps
    s1_dekad = s1_dekad.apply_dimension(dimension="t", process="array_interpolate_linear")

    # Select three dekads to create a 3-band raster
    s1_features = timesteps_as_bands(s1_dekad)
    return s1_features

# Test parameters
south = -12.73714945
north = -11.753685105
west = -55.165335994
east = -54.158521354
bbox = {"west": round(west, 2), "south": round(south, 2), "east": round(east, 2), "north": round(north, 2), 'crs': 4326}
year_item = 2020

# Create a polygon from the bounding box coordinates
polygon = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [bbox["west"], bbox["south"]],
                        [bbox["east"], bbox["south"]],
                        [bbox["east"], bbox["north"]],
                        [bbox["west"], bbox["north"]],
                        [bbox["west"], bbox["south"]]
                    ]
                ]
            }
        }
    ]
}
# Convert to JSON string for display
polygon_json = json.dumps(polygon, indent=4)
print(polygon_json)

base_output_path = Path("s1datacube")
base_output_path.mkdir(exist_ok=True)


# Setup the connection
connection = openeo.connect("openeo.dataspace.copernicus.eu")
connection.authenticate_oidc()

temporal_extent = (f"{year_item}-01-01", f"{year_item}-02-11")

# Load S1 collection
s1 = connection.load_collection(
    "SENTINEL1_GRD",
    spatial_extent=bbox,
    temporal_extent=temporal_extent,
    bands=["VH", "VV"],
).sar_backscatter(  # GFMap performed this step in the extraction
    coefficient="sigma0-ellipsoid"
)

# Get the S1 features
s1_features = get_s1_features(s1)

# Define the timesteps_as_bands function
def timesteps_as_bands(datacube):
    # Assuming we want the first three dekads as separate bands
    bands = []
    for i in range(3):
        bands.append(datacube.slice(dimension="t", start=i, end=i+1))
    return openeo.DataCube.merge_cubes(*bands)

# Save the result or proceed with further processing
# Assuming we have a save_result or similar function
s1_features.save_result(format="GTiff")

# Print the result for debugging
print(s1_features.execute())

########################################################################################################################
timenow = datetime.datetime.now()
timestr = timenow.strftime("%Y%m%d-%Hh%M")
print(f"Timestr: {timestr}")
tracking_file = base_output_path / f"tracking_{timestr}.csv"

manager = GFMAPJobManager(
    output_dir=base_output_path / timestr,
    output_path_generator=generate_output_path,
    poll_sleep=60,
    n_threads=1,
    collection_id="LC_feature_extraction",
    )

manager.add_backend(Backend.CDSE, cdse_connection, parallel_jobs=2)

manager.run_jobs(
    job_df,
    load_lc_features,
    tracking_file
)