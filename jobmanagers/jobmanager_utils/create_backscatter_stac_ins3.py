import os
import json
import subprocess
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.session import AWSSession
from rasterio.env import Env
from rasterio import warp as rasterio_warp
import pystac
from shapely.geometry import shape
from pystac.extensions.eo import EOExtension, Band

# ---- Config ----
bucket_name = "deforestation"
base_s3_path = "sarbackscatter"
s3_url_base = f"https://s3.waw3-1.cloudferro.com/swift/v1/{bucket_name}"
rclone_config = "/mnt/ssdarchive.nfs/userdoc/rclone.conf"
local_stac_dir = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter/stac_dir")
os.makedirs(local_stac_dir, exist_ok=True)

bands = ["VV", "VH"]
component = "backscatter"

# ---- Rclone utilities ----
def list_s3_folders(bucket_path, config_path):
    cmd = ["rclone", "--config", config_path, "lsf", "--dirs-only", bucket_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip().splitlines() if result.returncode == 0 else []

def list_tif_files_in_folder(bucket_path, config_path):
    cmd = ["rclone", "--config", config_path, "lsjson", bucket_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Warning] Could not access {bucket_path}")
        return []
    try:
        files = json.loads(result.stdout)
        return [f["Name"] for f in files if f["Name"].endswith(".tif")]
    except json.JSONDecodeError:
        return []

# ---- Metadata from remote GeoTIFF ----
def get_raster_metadata(remote_url):
    with rasterio.open(remote_url) as src:
        bounds = list(src.bounds)
        left, bottom, right, top = rasterio_warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        bbox = [left, bottom, right, top]
        footprint = [[
            [left, bottom], [right, bottom], [right, top],
            [left, top], [left, bottom]
        ]]
        props = {
            "proj:epsg": src.crs.to_epsg(),
            "proj:shape": src.shape,
            "proj:bbox": bounds
        }
    return bbox, footprint, props

# ---- Loop through folders and create STAC ----
folders = list_s3_folders(f"s14amazonas:{bucket_name}/{base_s3_path}", rclone_config)

for tile in sorted(folders):
    if tile == "stac_dir":
        continue
    tile_name = tile.split("/")[0]

    print(f"\nProcessing tile: {tile}")
    stac_dir = local_stac_dir / tile
    stac_dir.mkdir(exist_ok=True)
    collection_id = f"{tile_name}_{component}"
    item_list = []

    folder_path = f"s14amazonas:{bucket_name}/{base_s3_path}/{tile}"
    tif_files = list_tif_files_in_folder(folder_path, rclone_config)

    for tif_file in tif_files:
        file_url = f"{s3_url_base}/{base_s3_path}/{tile}{tif_file}"
        try:
            tif_date = tif_file.split("_")[2]
            dt = datetime.strptime(tif_date, "%Y-%m-%d")
        except Exception as e:
            print(f"Skipping {tif_file} due to date parsing issue: {e}")
            continue

        bbox, footprint, props = get_raster_metadata(file_url)

        # Calculate spatial extents
        spatial_extent_x = [bbox[0], bbox[2]]
        spatial_extent_y = [bbox[1], bbox[3]]

        item_id = Path(tif_file).stem

        item = pystac.Item(
            id=item_id,
            geometry={"type": "Polygon", "coordinates": footprint},
            bbox=bbox,
            datetime=dt,
            properties=props
        )
        item.stac_extensions = [
            "https://stac-extensions.github.io/eo/v1.1.0/schema.json",
            "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
        ]
        # Add polarization band information
        pystac_bands = [
            Band.create(
                name=band_name,
                description=f"{band_name} band",
                common_name="raster bands"
            ) for band_name in bands  # Ensure 'polarizations' is defined
        ]

        asset_key = item_id



        asset = pystac.Asset(
            href=file_url,
            media_type=pystac.MediaType.GEOTIFF,
            roles=["data"]
        )
        eo = EOExtension.ext(asset, add_if_missing=False)
        eo.apply(bands=pystac_bands)

        item.add_asset(item_id, asset)
        item_list.append(item)

    if not item_list:
        print(f"No valid items for {tile_name}")
        continue

    # Create collection
    union_geom = shape(item_list[0].geometry)
    for it in item_list[1:]:
        union_geom = union_geom.union(shape(it.geometry))

    bbox = list(union_geom.bounds)
    dates = sorted([it.datetime for it in item_list])
    spatial = pystac.SpatialExtent([bbox])
    temporal = pystac.TemporalExtent([[dates[0], dates[-1]]])
    extent = pystac.Extent(spatial=spatial, temporal=temporal)

    collection = pystac.Collection(
        id=collection_id,
        description=f"SAR backscatter collection for tile {tile_name}",
        extent=extent,
        license="CC-BY-SA-4.0",
        href=f'{s3_url_base}/{base_s3_path}/stac_dir/{tile_name}/{tile_name}_backscatter/collection.json'
    )
    collection.add_items(item_list)

    catalog = pystac.Catalog(
        id=f"{component}_catalog",
        description="SAR STAC Catalog",
        href=f'{s3_url_base}/{base_s3_path}/stac_dir/{tile_name}/{tile_name}_backscatter_catalog.json'
    )
    catalog.add_child(collection)

    print(f"Saving STAC catalog for {tile_name}")
    catalog.save(dest_href=str(stac_dir), catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
