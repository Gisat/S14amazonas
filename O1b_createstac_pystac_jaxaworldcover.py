import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import rasterio
import pystac
from shapely.geometry import shape
from pystac.extensions.eo import Band, EOExtension
import rasterio
from rasterio import warp as rasterio_warp

##############################################
##############################################
##############################################
#### INPUTS ###

# Configuration and paths for input data and output directory
# The root directory where the input data resides
# input_root_dir = Path("/mnt/hddarchive.nfs/EUGW/output")
input_root_dir = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/forest_elevation_mask/mask")
# Name for the base output directory in S3 storage
s3_base_output_dirname = "mask"
# Directory where STAC (SpatioTemporal Asset Catalog) files will be stored
stac_dir_root = input_root_dir.joinpath("stac_dir")
# Ensure that the directory exists, or create it
os.makedirs(stac_dir_root, exist_ok=True)

# bucket_name = "eugw.hst8hx13lamhh1wukzpfru15xjiimknmf0jjzjjaossezr5mx9z7sqmspd"
bucket_name = "supportivedata"

##


component_for_stac_creation = "forestelevationmask"
collection_name = "forestelevationmask"
number_bands = 1
bands= ["forestelevationmask"]


# component_for_stac_creation = "forestmask"
# collection_name = "forestmask"
# number_bands = 1
# bands= ["forestmask"]

##############################################
#### PREP FROM INPUTS ###

# Create the output directory specific to the STAC collection
stac_dir = stac_dir_root.joinpath(collection_name)
os.makedirs(stac_dir_root, exist_ok=True)

# Base URL for accessing files from the S3 bucket
s3_base_url = f"https://s3.waw3-1.cloudferro.com/swift/v1/{bucket_name}"


##############################################
##############################################
##############################################
def group_forestelevationmask_rasters_by_year(input_dir_list, year= None):
    raster_groups = defaultdict(list)
    for input_dir_list_item in input_dir_list:
        for root, dirs, files in os.walk(input_dir_list_item):
            for filename in files:
                # Check if the file is a .tif file
                if filename== "forest_elevation_mask_2020.tif":
                    # Construct the full path to the file
                    file_path = os.path.join(root, filename)
                    # Do something with the .tif file
                    print(f"Processing file: {file_path}")
                    # You can add your file processing logic here
                else:
                    continue
                start_date = datetime.strptime(year, "%Y")
                raster_groups[start_date.strftime("%Y")].append(file_path)
    return raster_groups

def group_forestmask_rasters_by_year(input_dir_list, year= None):
    raster_groups = defaultdict(list)
    for input_dir_list_item in input_dir_list:
        for root, dirs, files in os.walk(input_dir_list_item):
            for filename in files:
                # Check if the file is a .tif file
                if filename == "forest_mask_2020.tif":
                    # Construct the full path to the file
                    file_path = os.path.join(root, filename)
                    # Do something with the .tif file
                    print(f"Processing file: {file_path}")
                    # You can add your file processing logic here
                else:
                    continue
                start_date = datetime.strptime(year, "%Y")
                raster_groups[start_date.strftime("%Y")].append(file_path)
    return raster_groups

##############################################
##############################################
##############################################
##############################################
##############################################
##############################################





# Initialize an empty dictionary to hold raster groups
raster_groups = defaultdict(list)

# get raster dirs
input_rasters_dirs_list = [input_root_dir]
# Group rasters by month


# Choose which function to use based on the selected component
if component_for_stac_creation == "forestelevationmask":
    raster_groups = group_forestelevationmask_rasters_by_year(input_rasters_dirs_list, year= "2020")
elif component_for_stac_creation == "forestmask":
    raster_groups = group_forestmask_rasters_by_year(input_rasters_dirs_list, year="2020")
else:
    raise Exception("Implement find raster groups")



# Function to extract bbox and footprint
def get_raster_metadata(filepath):
    with rasterio.open(filepath) as src:
        # Get the bounds of the raster in the source coordinate system
        proj_bounds = list(src.bounds)
        # Reproject the bounds into EPSG:4326 (WGS84 - lat/lon)
        left, bottom, right, top = rasterio_warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        # Prepare bbox and footprint based on reprojected coordinates
        bbox = [left, bottom, right, top]
        footprint = [[
                [left, bottom],
                [right, bottom],
                [right, top],
                [left, top],
                [left, bottom]
            ]]

        # Metadata properties related to the raster file
        properties = {
            "proj:epsg": src.crs.to_epsg(),  # EPSG code of the projection
            "proj:shape": src.shape,  # Raster shape (height, width)
            "proj:bbox": proj_bounds,  # Original bounds in the source projection
        }

        return bbox, footprint, properties




stac_items = []
# Add items to catalog
for month, files in raster_groups.items():
    for file_path in files:
        bbox, footprint, properties = get_raster_metadata(file_path)

        # Calculate spatial extents
        spatial_extent_x = [bbox[0], bbox[2]]
        spatial_extent_y = [bbox[1], bbox[3]]

        item_id = Path(file_path).stem
        item = pystac.Item(
            id=item_id,
            geometry={"type": "Polygon", "coordinates": footprint},
            bbox=bbox,
            datetime=datetime.strptime(month, "%Y"),
            properties=properties
        )
        stac_extensions=[
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
        ],
        # Add polarization band information
        pystac_bands = [
            Band.create(
                name=band_name,
                description=f"{band_name} band",
                common_name="raster bands"
            ) for band_name in bands  # Ensure 'polarizations' is defined
        ]

        # Create an asset for the file and add to the STAC item
        asset_key = Path(file_path).stem
        tif_url = f"{s3_base_url}/{s3_base_output_dirname}/{Path(file_path).relative_to(input_root_dir)}"
        asset = pystac.Asset(
            href=tif_url,
            media_type=pystac.MediaType.GEOTIFF,
            roles=['data']
        )

        # Apply EO extension to the asset and link bands
        eo_asset = EOExtension.ext(asset, add_if_missing=False)
        eo_asset.apply(bands=pystac_bands)  # Ensure 'pystac_bands' is defined


        item.add_asset(
            key=asset_key,
            asset=asset
        )

        # try:
        #     item.validate()
        #     print(f"Item {item.id} is valid.")
        # except pystac.STACValidationError as e:
        #     print(f"Validation error for item {item.id}: {e}")

        stac_items.append(item)


# Generate Collection BBox and Footprint
unioned_footprint = shape(stac_items[0].geometry)
for item in stac_items[1:]:
    unioned_footprint = unioned_footprint.union(shape(item.geometry))

collection_bbox = list(unioned_footprint.bounds)
spatial_extent = pystac.SpatialExtent(bboxes=[collection_bbox])

collection_interval = sorted([item.datetime for item in stac_items])
temporal_extent = pystac.TemporalExtent(intervals=[[collection_interval[0], collection_interval[-1]]])

collection_extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)

# Create STAC Collection
collection = pystac.Collection(
    id=collection_name,
    description='Raster images from public S3 bucket',
    extent=collection_extent,
    license='CC-BY-SA-4.0',
    href=f'{s3_base_url}/{s3_base_output_dirname}/{str(stac_dir.relative_to(input_root_dir))}/collection.json'
)

collection.add_items(stac_items)
print(f"creating collection")

# Create STAC catalog
catalog = pystac.Catalog(id=f"{component_for_stac_creation} rasters_catalog",
                         description="Raster catalog grouped by EPSG and date",
                         href=f'{s3_base_url}/{s3_base_output_dirname}/{str(stac_dir.relative_to(input_root_dir))}/{collection_name}_catalog.json'
                         )

catalog.add_child(collection)

# Save catalog to disk
# catalog.normalize_and_save(root_href=str(stac_dir), catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
catalog.save(dest_href=str(stac_dir), catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

