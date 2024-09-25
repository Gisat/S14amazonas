import os
from pathlib import Path
from osgeo import gdal
import numpy as np
import subprocess
import shutil
from csv import reader
import dask.array as da


from tondortools.tool import read_raster_info, save_raster_template
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32, GDT_Byte

def create_sieved(result_filepath_for_sieve, result_filepath_sieved, sieve=10):
    sieve_args = ["gdal_sieve.py", #"-mask", str(mask_tif),
                  "-4", "-nomask",
                  "-st", str(sieve),
                  str(result_filepath_for_sieve), str(result_filepath_sieved)]
    cmd_output = subprocess.run(sieve_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("exit code {} --> {}".format(cmd_output.returncode, sieve_args))




    result_filepath_sieved_inverted = result_filepath_sieved.parent.joinpath(result_filepath_sieved.name.replace(".tif", "_inverted.tif"))
    raster_ds = gdal.Open(str(result_filepath_sieved))
    raster_array = raster_ds.GetRasterBand(1).ReadAsArray()
    raster_array_inverted = np.logical_not(raster_array).astype(int)
    save_raster_template(result_filepath_sieved, result_filepath_sieved_inverted, raster_array_inverted, data_type=GDT_Byte)

    result_filepath_large_sieved = result_filepath_sieved.parent.joinpath(result_filepath_sieved.name.replace(".tif", "_largesive.tif"))
    cmd_sieve = 'gdal_sieve.py -st %s -8 -of GTiff %s %s' % (
    '50000', str(result_filepath_sieved), str(result_filepath_large_sieved))
    os.system(cmd_sieve)



    intermediate_path = result_filepath_sieved.parent.joinpath(result_filepath_sieved.name.replace(".tif", "_inter.tif"))
    shutil.copy(result_filepath_sieved, intermediate_path)
    result_filepath_sieved.unlink()

    raster_ds = gdal.Open(str(intermediate_path))
    raster_array = raster_ds.GetRasterBand(1).ReadAsArray()
    save_raster_template(intermediate_path, result_filepath_sieved, raster_array, data_type=GDT_Byte)
    intermediate_path.unlink()

def raster2array(rasterfn):
    raster = gdal.Open(str(rasterfn))
    band = raster.GetRasterBand(1).ReadAsArray().astype('float')
    return band



GFW_path = Path("/mnt/hddarchive.nfs/amazonas_dir/aux_data/CCI/ESACCI-LC-L4-LCCS-Map-300m-2015_Amazon_only30to90_20m.tif")
SRTM_path = Path("/mnt/hddarchive.nfs/amazonas_dir/aux_data/SRTM")
mosaic_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/output/Mosaic")
ref_sar_mosaic_folder = Path("/ mnt/hddarchive.nfs/amazonas_dir/support_data/ref_worldcover_sarmosaic")

result_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/support_data/base_worldcover")

work_dir = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir")


tile_list = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
tile_list = ['20LLQ']

VH_THR = -15
VV_THR = -9
pol = ["VV", "VH"]

THR_SRTM_elevation_mountains = 1800.0  # Areas aboe this heigh are mountains and are masked out.
THR_SRTM_elevation = 40.0

def find_template_tif(directory_path):
    # List to store the paths of .tif files
    tif_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file ends with .tif
            if file.endswith('.tif'):
                # Append the full path of the file to the list
                tif_files.append(os.path.join(root, file))

    # Variable to store the template, if .tif files are found
    template_tif = tif_files[0] if tif_files else None
    return  template_tif

def create_zeros_array(template_tif_path):
    dataset = gdal.Open(template_tif_path, gdal.GA_ReadOnly)

    # Get raster dimensions
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # Create a numpy array of zeros with the same dimensions
    zeros_array = np.ones((height, width))
    return zeros_array

def getRasterExtent(rasterfn):
    raster = gdal.Open(rasterfn)
    ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
    lrx = ulx + (raster.RasterXSize * xres)
    lry = uly + (raster.RasterYSize * yres)
    return [ulx,uly,lrx,lry]

print(len(tile_list))

for tile_list_item in tile_list:

    mask_raster = result_folder.joinpath(f"{tile_list_item}_MASK.tif")
    if not mask_raster.exists():

        print(f"creating base lc for {tile_list_item}")

        tile_mosaic_folder = mosaic_folder.joinpath(tile_list_item)
        tempalte_tif = find_template_tif(tile_mosaic_folder)
        (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(tempalte_tif)
        extent = getRasterExtent(str(tempalte_tif))

        base_mask_array = create_zeros_array(tempalte_tif)

        tile_mosaic_orbits = os.listdir(tile_mosaic_folder)



        fmin_raster = work_dir.joinpath(f"{tile_list_item}_fmin.tif")
        fmin_raster_sieved = work_dir.joinpath(f"{tile_list_item}_fmin_seived.tif")
        if not fmin_raster_sieved.exists():
            if not fmin_raster.exists():

                for pol_item in pol:
                    array_stack = []
                    orbit_count = 0
                    for orbit_direction in tile_mosaic_orbits:

                        if orbit_direction not in ["ascending", "descending"]: continue
                        orbit_count += 1
                        if orbit_count >1: continue
                        tile_mosaic_orbit_filename_list = os.listdir(tile_mosaic_folder.joinpath(orbit_direction))

                        for tile_mosaic_orbit_filename_list_item in sorted(tile_mosaic_orbit_filename_list):
                            if "_filled" in tile_mosaic_orbit_filename_list_item: continue
                            year_file = int(tile_mosaic_orbit_filename_list_item.split("_")[3][0:4])
                            if not year_file <= 2017: continue
                            pol_file = tile_mosaic_orbit_filename_list_item.split("_")[2]
                            if not pol_file == pol_item: continue
                            print(f" {tile_mosaic_orbit_filename_list_item} pol {pol_file}")

                            tile_mosaic_orbit_filepath = tile_mosaic_folder.joinpath(orbit_direction, tile_mosaic_orbit_filename_list_item)
                            raster_ds = gdal.Open(str(tile_mosaic_orbit_filepath), gdal.GA_ReadOnly)
                            raster_nodata = raster_ds.GetRasterBand(1).GetNoDataValue()

                            raster_array = raster_ds.GetRasterBand(1).ReadAsArray()
                            raster_array[raster_array == raster_nodata] = np.nan

                            raster_ds = None

                            array_stack.append(raster_array)

                    dask_array = da.from_array(array_stack, chunks='auto')  # 'auto' lets Dask decide the chunk size
                    if pol_item == "VH":
                        vh_array_mean = da.nanmedian(dask_array, axis=0).compute()
                    if pol_item == "VV":
                        vv_array_mean = da.nanmedian(dask_array, axis=0).compute()
                    array_stack = None

                base_mask_array[(vh_array_mean < VH_THR) | (vv_array_mean < VV_THR)] = 0
                save_raster_template(tempalte_tif, fmin_raster, base_mask_array, GDT_Byte)

            create_sieved(fmin_raster, fmin_raster_sieved, sieve=1000)
        vv_array_stack = None
        vh_array_stack = None

        gfw_tile = work_dir.joinpath(f"{tile_list_item}_GFW.tif")
        if not gfw_tile.exists():
            cmd_GFWTC_tile = 'gdal_translate -ot Int32 -r nearest -co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=IF_NEEDED -projwin %s %s %s %s -outsize %s %s %s %s' % (
            extent[0], extent[1], extent[2], extent[3], str(RasterXSize), str(RasterYSize), str(GFW_path),
            str(gfw_tile))
            print("{}".format(cmd_GFWTC_tile))
            os.system(cmd_GFWTC_tile)

        srtm_tile = work_dir.joinpath(f"{tile_list_item}_SRTM.tif")
        srtm_tiled = work_dir.joinpath(f"{tile_list_item}_SRTM_tiled.tif")
        if not srtm_tile.exists():

            PATHS_LIST = '/mnt/hddarchive.nfs/amazonas_dir/aux_data/geom_data/S2_grid_AmazonBasin_detections_paths.csv'
            with open(PATHS_LIST, 'r') as read_paths:
                txt_reader = reader(read_paths)
                for each_Ama_tile in txt_reader:
                    Ama_tile = str(each_Ama_tile[0])
                    if Ama_tile == str(tile_list_item):
                        Ama_path = str(each_Ama_tile[1])
                        SRTM_layer = Path(SRTM_path).joinpath(str(each_Ama_tile[3]))
                        print("USE SRTM layer: {}".format(SRTM_layer))

            if not Path(srtm_tiled).exists():
                cmd_SRTM_tile = 'gdal_translate -co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=IF_NEEDED -r average -projwin %s %s %s %s -outsize %s %s %s %s' % (
                extent[0], extent[1], extent[2], extent[3], str(RasterXSize), str(RasterYSize), SRTM_layer,
                srtm_tiled)
                os.system(cmd_SRTM_tile)

            SRTM_GZ = raster2array(str(srtm_tiled))
            SRTM_GZ[SRTM_GZ > THR_SRTM_elevation_mountains] = 1
            SRTM_GZ[SRTM_GZ <= THR_SRTM_elevation] = 1
            SRTM_GZ[SRTM_GZ > THR_SRTM_elevation] = 0
            SRTM_GZ_inverted = 1 - SRTM_GZ
            save_raster_template(tempalte_tif, srtm_tile, SRTM_GZ_inverted, GDT_Byte)


        fmin = raster2array(fmin_raster_sieved)
        gfw = raster2array(gfw_tile)
        srtm = raster2array(srtm_tile)

        mask = np.array((fmin ==1) & (srtm ==1) & (gfw ==0)).astype(int)
        save_raster_template(tempalte_tif,mask_raster, mask, GDT_Byte)

        fmin, gfw, srtm, mask = None, None, None, None




