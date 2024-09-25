from pathlib import Path
import os, requests
import subprocess
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte, GDT_Float32
import numpy as np
from datetime import datetime, timedelta, date
import math
from tondortools.tool import read_raster_info, save_raster, mosaic_tifs, save_raster_template
from src.tondor.util.tool import reproject_multibandraster_toextent
def create_merged_filepath(output_path, raster1, raster2, bounds, pixel_width, width, height):
    output_inter_path = output_path.parent.joinpath(output_path.name.replace(".tif", "_inter.tif"))

    args = ["gdal_merge.py",
            "-separate",
            "-o", str(output_inter_path),
            "-ps", str(pixel_width), str(pixel_width),
            "-ul_lr", str(bounds[0]), str(bounds[3]), str(bounds[2]), str(bounds[1]),
            "-of", "GTiff",
            "-ot", "Float32",
            "-co", "compress=DEFLATE"]
    args = args + [str(raster1)] + [str(raster2)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")

    args = ["gdalwarp",
            "-ts", str(width), str(height),
            "-of", "GTiff",
            "-co", "compress=DEFLATE"]
    args = args + [str(output_inter_path)] + [str(output_path)]
    cmd_output = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"exit code {cmd_output.returncode}  --> {args}")

def create_mask(tile_deforestation_mask, base_lc_tile_filepath, tile_prob_mask):
    dataset = gdal.Open(str(base_lc_tile_filepath))
    jaxa_mask_array = dataset.GetRasterBand(1).ReadAsArray()
    # Inverting 0 and 1 in the numpy array
    inverted_array = np.where(jaxa_mask_array == 0, 1, 0)

    prob_dataset = gdal.Open(str(tile_prob_mask))
    prob_mask_array = prob_dataset.GetRasterBand(1).ReadAsArray()

    final_binary_array = np.logical_or(inverted_array, prob_mask_array).astype(np.uint8)

    inverted_array = np.where(final_binary_array == 0, 1, 0)

    save_raster_template(base_lc_tile_filepath, tile_deforestation_mask, inverted_array, GDT_Byte)


def create_large_sieved(file1_path, file1_path_sieved, size_threshold=5000):
    # Construct the gdal_sieve.py command
    command = [
        'gdal_sieve.py',
        '-st', str(size_threshold),
        '-8',  # 8-connected neighborhood
        '-nomask',  # No mask applied
        '-of', 'GTiff',  # Output format as GeoTIFF
        str(file1_path),  # Input raster file
        str(file1_path_sieved)  # Output raster file
    ]

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Sieve operation completed successfully. Output saved to '{file1_path_sieved}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error during sieve operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_filtered_detection(file1_path, file1_path_sieved, file1_filtered_path):
    dataset1 = gdal.Open(str(file1_path))
    dataset1_array = dataset1.GetRasterBand(1).ReadAsArray()

    dataset2 = gdal.Open(str(file1_path_sieved))
    dataset2_array = dataset2.GetRasterBand(1).ReadAsArray()

    # Create third array with 1s where array2 is 1 and array1 is 0
    array3 = np.where((dataset1_array == 1) & (dataset2_array == 0), 1, 0)

    save_raster_template(file1_path, file1_filtered_path, array3, GDT_Byte)



#'18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP','18LVR', '20LLQ',
tiles = [ '18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP','18LVR', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
# tiles = ['18NYH']
# '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYH', '22MBT', '22MGB']
print(len(tiles))

model_version = 'best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score'
base_version = "jaxa"
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir", "reclassification")
os.makedirs(work_dir, exist_ok=True)
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion_parent = detection_folder.joinpath(f'{model_version}')
########################################################################################################################
support_data = amazonas_root_folder.joinpath("support_data")
merged_sar_ref_folder = support_data.joinpath(f"ref_worldcover_sarmosaic")
predicted_baselc_folder = support_data.joinpath(f"base_worldcover_prediction")
baselc_folder = support_data.joinpath(f"base_worldcover")

prob_var_mask_folder = support_data.joinpath("tile_prob_pixel_info")
var_threshold_name_suffix = "pt01"

deforesation_mask_folder = support_data.joinpath("deforestation_mask")
os.makedirs(deforesation_mask_folder, exist_ok=True)
########################################################################################################################

for tile_item in tiles:
    detection_folder_aiversion = detection_folder_aiversion_parent.joinpath(tile_item)
    detection_folder_aiversion_reclassified = detection_folder_aiversion.joinpath("reclassified")
    os.makedirs(detection_folder_aiversion_reclassified, exist_ok=True)
    # / mnt / hddarchive.nfs / amazonas_dir / support_data / base_worldcover_prediction / 18L
    # VQ_BASELC_2017_CLASS.tif

    template_raster = None
    orbit_directions = os.listdir(detection_folder_aiversion)
    for orbit_direction_item in sorted(orbit_directions):
        if not orbit_direction_item in ['ascending', 'descending']: continue

        folder_files_list = os.listdir(detection_folder_aiversion.joinpath(orbit_direction_item))
        for folder_file_list_item in sorted(folder_files_list):
            detection_path = detection_folder_aiversion.joinpath(orbit_direction_item, folder_file_list_item)
            if not "CLASS.tif" in folder_file_list_item: continue
            template_raster = detection_path
            if template_raster.exists(): break
    print(f"using template rasters: {template_raster}")

    if base_version == "mcd":
        base_lc_tile_filepath = predicted_baselc_folder.joinpath(f"{tile_item}_BASELC_2017_CLASS.tif")
        if not base_lc_tile_filepath.exists(): raise Exception(f"{base_lc_tile_filepath} does not exist")
    elif base_version == "jaxa":
        (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(template_raster)
        jaxa_mosaic = "/mnt/hddarchive.nfs/amazonas_dir/support_data/jaxa_worldcover/reclassified/mosaic.tif"
        baselc_folder = support_data.joinpath(f"base_jaxa_worldcover")
        os.makedirs(baselc_folder, exist_ok=True)
        base_lc_tile_filepath = baselc_folder.joinpath(f"{tile_item}_MASK.tif")
        if not base_lc_tile_filepath.exists():
            reproject_multibandraster_toextent(jaxa_mosaic, base_lc_tile_filepath, epsg, pixel_width, xmin, imagery_extent_box.bounds[2], imagery_extent_box.bounds[1], ymax, work_dir= None, method ='near')
    else:
        base_lc_tile_filepath = baselc_folder.joinpath(f"{tile_item}_MASK.tif")
        if not base_lc_tile_filepath.exists(): raise Exception(f"{base_lc_tile_filepath} does not exist")


    tile_prob_mask = prob_var_mask_folder.joinpath(f"prob_{tile_item}_mask_{var_threshold_name_suffix}.tif")
    if not tile_prob_mask.exists(): raise Exception(f"{tile_prob_mask} doesnt exists")

    tile_prob_mask_reprojected = prob_var_mask_folder.joinpath(f"prob_{tile_item}_mask_{var_threshold_name_suffix}_reprojected.tif")
    reproject_multibandraster_toextent(tile_prob_mask, tile_prob_mask_reprojected, epsg, pixel_width, xmin,
                                       imagery_extent_box.bounds[2], imagery_extent_box.bounds[1], ymax, work_dir=None,
                                       method='near')

    tile_deforestation_mask = deforesation_mask_folder.joinpath(f"mask_{tile_item}.tif")
    create_mask(tile_deforestation_mask, base_lc_tile_filepath, tile_prob_mask_reprojected)

    (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(base_lc_tile_filepath)

    dataset = gdal.Open(str(tile_deforestation_mask))
    base_lc = dataset.GetRasterBand(1).ReadAsArray()
    base_deforestation = 1 - base_lc

    orbit_directions = os.listdir(detection_folder_aiversion)
    if len(orbit_directions) == 3:

        reclassified_orbit_files_dict = {}

        for orbit_direction_item in sorted(orbit_directions):

            if not orbit_direction_item in ['ascending', 'descending']: continue

            reclassified_orbit_files_dict[orbit_direction_item] = {}

            detection_folder_aiversion_orbit = detection_folder_aiversion.joinpath(
                orbit_direction_item)
            folder_files_list = os.listdir(detection_folder_aiversion_orbit)

            for folder_file_list_item in sorted(folder_files_list):
                if not folder_file_list_item.endswith("_CLASS.tif"): continue
                timepoint = folder_file_list_item.split('_')[2]
                reclassified_orbit_files_dict[orbit_direction_item][timepoint] = detection_folder_aiversion_orbit.joinpath(folder_file_list_item)
                pass

        print(reclassified_orbit_files_dict)

        all_dates = sorted(set(reclassified_orbit_files_dict['ascending'].keys()).union(set(reclassified_orbit_files_dict['descending'].keys())))

        # Create a list of tuples with matching files or None if not present
        matched_tuples = [(reclassified_orbit_files_dict['ascending'].get(date), reclassified_orbit_files_dict['descending'].get(date)) for date in all_dates]

        raster_template = None
        # Iterate through the matched tuples
        for f1, f2 in matched_tuples:
            if f1 is not None:
                file1_path = f1
                file1_path_sieved = work_dir.joinpath(f"{Path(file1_path).stem}_sieved.tif")
                file1_filtered_path = work_dir.joinpath(f"{Path(file1_path).stem}_filtered.tif")
                file1_path_sieved.unlink(missing_ok=True)
                file1_filtered_path.unlink(missing_ok=True)


                create_large_sieved(file1_path, file1_path_sieved)
                create_filtered_detection(file1_path, file1_path_sieved, file1_filtered_path)
                raster_template = file1_filtered_path
                time_point = file1_filtered_path.name.split('_')[2]
                file1_path = file1_filtered_path
            else:
                file1_path = None

            if f2 is not None:
                file2_path = f2
                file2_path_sieved = work_dir.joinpath(f"{Path(file2_path).stem}_sieved.tif")
                file2_filtered_path = work_dir.joinpath(f"{Path(file2_path).stem}_filtered.tif")
                file2_path_sieved.unlink(missing_ok=True)
                file2_filtered_path.unlink(missing_ok=True)


                create_large_sieved(file2_path, file2_path_sieved)
                create_filtered_detection(file2_path, file2_path_sieved, file2_filtered_path)
                raster_template = file2_path
                time_point = file2_filtered_path.name.split('_')[2]
                file2_path = file2_filtered_path
            else:
                file2_path = None





            # Open the first file and read as an array (if present)
            if file1_path is not None:
                dataset1 = gdal.Open(str(file1_path))
                dataset1_array = dataset1.GetRasterBand(1).ReadAsArray()
            else:
                dataset1_array = None

            # Open the second file and read as an array (if present)
            if file2_path is not None:
                dataset2 = gdal.Open(str(file2_path))
                dataset2_array = dataset2.GetRasterBand(1).ReadAsArray()
            else:
                dataset2_array = None

            # Perform logical OR operation between the arrays
            if dataset1_array is not None and dataset2_array is not None:
                combined_array = np.logical_or(dataset1_array, dataset2_array).astype(int)
            elif dataset1_array is not None:  # If only dataset1 is available
                combined_array = dataset1_array
            elif dataset2_array is not None:  # If only dataset2 is available
                combined_array = dataset2_array
            else:
                combined_array = None  # Both arrays are None

            detected_change_baselc_removed = (combined_array.astype(np.int32)) & (~base_deforestation.astype(np.int32))
            base_deforestation = base_deforestation + detected_change_baselc_removed

            detection_folder_aiversion_reclassified_path = detection_folder_aiversion_reclassified.joinpath(f"{tile_item}_{time_point}_CLASS.tif")
            save_raster_template(raster_template, detection_folder_aiversion_reclassified_path, detected_change_baselc_removed, data_type=GDT_Byte)
            print(detection_folder_aiversion_reclassified_path)





    else:

        for orbit_direction_item in sorted(orbit_directions):

            if not orbit_direction_item in ['ascending', 'descending']: continue


            detection_folder_aiversion_orbit = detection_folder_aiversion.joinpath(
                orbit_direction_item)
            folder_files_list = os.listdir(detection_folder_aiversion_orbit)

            for folder_file_list_item in sorted(folder_files_list):
                if not folder_file_list_item.endswith("_CLASS.tif"): continue
                raster_filepath = detection_folder_aiversion_orbit.joinpath(folder_file_list_item)
                time_point = folder_file_list_item.split('_')[2]



                file1_path_sieved = work_dir.joinpath(f"{Path(raster_filepath).stem}_sieved.tif")
                file1_filtered_path = work_dir.joinpath(f"{Path(raster_filepath).stem}_filtered.tif")
                file1_path_sieved.unlink(missing_ok=True)
                file1_filtered_path.unlink(missing_ok=True)

                create_large_sieved(raster_filepath, file1_path_sieved)
                create_filtered_detection(raster_filepath, file1_path_sieved, file1_filtered_path)


                dataset2 = gdal.Open(str(file1_filtered_path))
                combined_array = dataset2.GetRasterBand(1).ReadAsArray()


                detected_change_baselc_removed = (combined_array.astype(np.int32)) & (~base_deforestation.astype(np.int32))
                base_deforestation = base_deforestation + detected_change_baselc_removed

                detection_folder_aiversion_reclassified_path = detection_folder_aiversion_reclassified.joinpath(f"{tile_item}_{time_point}_CLASS.tif")
                save_raster_template(str(raster_filepath), detection_folder_aiversion_reclassified_path, detected_change_baselc_removed, data_type=GDT_Byte)
                print(detection_folder_aiversion_reclassified_path)




