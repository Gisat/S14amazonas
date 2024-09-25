
from pathlib import Path
import subprocess
from osgeo import gdal

def clip_and_reproject(big_raster_path, small_raster_path, output_path):
    # Open the small raster to get its properties
    small_raster = gdal.Open(small_raster_path)
    small_geo_transform = small_raster.GetGeoTransform()
    small_projection = small_raster.GetProjection()
    small_width = small_raster.RasterXSize
    small_height = small_raster.RasterYSize

    # Get the extent of the small raster
    minx = small_geo_transform[0]
    maxx = minx + small_geo_transform[1] * small_width
    miny = small_geo_transform[3] + small_geo_transform[5] * small_height
    maxy = small_geo_transform[3]

    # Close the small raster
    small_raster = None

    # Use gdalwarp to clip and reproject the big raster
    gdalwarp_command = [
        'gdalwarp',
        '-t_srs', small_projection,
        '-te', str(minx), str(miny), str(maxx), str(maxy),
        '-ts', str(small_width), str(small_height),
        big_raster_path, output_path
    ]

    subprocess.run(gdalwarp_command)



tile_list = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']



sar_base_lc_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/support_data/base_worldcover")

jax_mosaic_path = Path("/mnt/hddarchive.nfs/amazonas_dir/support_data/jaxa_worldcover/reclassified/mosaic.tif")
jax_base_lc_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/support_data/jaxa_worldcover")
work_dir = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir")

for tile_list_item in tile_list:

    sar_base_lc_path = sar_base_lc_folder.joinpath(f"{tile_list_item}_MASK.tif")
    if not sar_base_lc_path.exists(): raise Exception(f"Here")

    jax_base_lc_path = work_dir.joinpath(f"jax_{tile_list_item}_MASK.tif")

    clip_and_reproject(str(jax_mosaic_path), str(sar_base_lc_path), str(jax_base_lc_path))


