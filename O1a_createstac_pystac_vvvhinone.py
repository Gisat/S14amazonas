import os

import pystac
from pystac.extensions.eo import Band, EOExtension
from shapely.geometry import Polygon, mapping
from datetime import datetime
from pathlib import Path
import rasterio
from shapely.geometry import shape

def file_datetime_list(folder_path):
    rasterfile_list = os.listdir(folder_path)
    datetime_list = []
    for rasterfile_list_item in rasterfile_list:
        if not rasterfile_list_item.endswith("tif"):
            continue
        datetime_str = rasterfile_list_item.split("_")[3]
        datetime_list.append(datetime_str)
    return datetime_list

# Function to Extract BBox and Footprint from S3 Asset
def get_bbox_and_footprint(s3_url_item):
    with rasterio.open(s3_url_item) as r:
        bounds = r.bounds
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        footprint = Polygon([
            [bounds.left, bounds.bottom],
            [bounds.left, bounds.top],
            [bounds.right, bounds.top],
            [bounds.right, bounds.bottom]
        ])
        return bbox, mapping(footprint)

# S3 Base URL and Datetime List
tile_name = "21LYG"

pol_list = ["VV", "VH"]
# datetime_list = ["20211020", "20211101", "20211113", "20211125", "20211207", "20211219"]
base_url = "https://s3.waw3-1.cloudferro.com/swift/v1/gisat-archive/SAR"
root_path = Path("/mnt/hddarchive.nfs/amazonas_dir/openEO/SAR")


datetime_list = sorted(list(set(file_datetime_list(root_path.joinpath(tile_name, "backscatter")))))

stac_items = []

for datetime_list_item in datetime_list:
    bbox, footprint = None, None  # Initialize bbox and footprint

    collection_item = pystac.Item(
        id=f'{tile_name}-{datetime_list_item}',
        geometry=footprint,
        bbox=bbox,
        datetime=datetime.strptime(datetime_list_item, "%Y%m%d"),
        properties={},
        href=f"{base_url}/{tile_name}/backscatter/21LYG_BAC_{datetime_list_item}_filled.json"  # S3 path reference
    )

    collection_item.common_metadata.gsd = 0.3
    collection_item.common_metadata.platform = 'Gisat'
    collection_item.common_metadata.instruments = ['ForestView']


    for pol in pol_list:

        tif_dateitem_pol_url = f"{base_url}/{tile_name}/backscatter/21LYG_BAC_{pol}_{datetime_list_item}_filled.tif"
        # Band Information for VV Polarisation
        pol_bands = [Band.create(name=pol, description=f'{pol} polarisation', common_name='pol')]

        bbox, footprint = get_bbox_and_footprint(tif_dateitem_pol_url)

        asset = pystac.Asset(href=tif_dateitem_pol_url, media_type=pystac.MediaType.GEOTIFF, roles=['data'])

        ####
        # collection_item.add_asset("image", asset)
        # EOExtension.ext(asset, add_if_missing=True).apply(vv_bands)

        eo_asset = EOExtension.ext(asset, add_if_missing=False)
        eo_asset.apply(pol_bands)
        collection_item.add_asset(pol, asset)


    # Update item geometry and bbox after processing assets
    collection_item.bbox = bbox
    collection_item.geometry = footprint

    stac_items.append(collection_item)



print(f"done create stac items")
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
    id='backscatter',
    description='Raster images from public S3 bucket',
    extent=collection_extent,
    license='CC-BY-SA-4.0',
    href=f'{base_url}/{tile_name}/backscatter/collection.json'
)

collection.add_items(stac_items)
print(f"creating collection")

# Create Catalog
catalog = pystac.Catalog(
    id='catalog-with-collection',
    description='STAC Catalog with Collection for S3-hosted SAR data',
    href=f'{base_url}/{tile_name}/{tile_name}_catalog.json'  # S3 path reference
)
catalog.add_child(collection)

print(f"creating catalog")
catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED, dest_href='/mnt/hddarchive.nfs/amazonas_dir/openEO/SAR/21LYG')


