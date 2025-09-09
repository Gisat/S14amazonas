import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import rasterio
from osgeo.gdalconst import GDT_Int16
from scipy.ndimage import binary_dilation
from tondortools.tool import read_raster_info, reproject_multibandraster_toextent, raster2array, save_raster_template
import numpy as np
from osgeo.gdal import GDT_Byte
from scipy.ndimage import binary_erosion, binary_dilation, label, binary_opening
from osgeo import gdal
import subprocess

ELEVATION_MAX = 1800
ELEVATION_MIN = 40
LARGE_SIEVE = 50000

forest_tifpath = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/forest_elevation_mask/mask/forest_mask_2020.tif")
elevation_tifpath = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/srtmlayer.tif")
change_detection_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_raw")
root_work_dir = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_processing")
output_dir = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/output")


# tile_list = sorted(os.listdir(change_detection_root_folder))
tile_list = ["22NEG", "21LXF", "22MCT", "21LXG"]


def find_ai_files_in_time_window(ai_warpeddir_date_dict, datetime_list_item, time_window = 24):
    datestr_datetime_up = datetime_list_item + timedelta(days=time_window)
    datestr_datetime_down = datetime_list_item - timedelta(days=time_window)

    files_list = []
    for date_index, ai_warpeddir_item_path in ai_warpeddir_date_dict.items():
        if date_index <= datestr_datetime_up and date_index >= datestr_datetime_down:
            files_list.append(ai_warpeddir_item_path["AIMCD"])

    return files_list


def set_colormap(input_path):
    # Define a colormap dictionary {value: (R, G, B, A)}
    colormap = {
        1: (255, 0, 0, 255),  # Red
    }

    # Open the raster file in read/write mode
    with rasterio.open(str(input_path), 'r+') as src:
        src.write_colormap(1, colormap)  # Apply colormap to the first band


def find_cummulative_aidetection(ai_files_list):
    rasterarray_list = []
    for ai_files_list_item in ai_files_list:
        if not ai_files_list_item.exists(): continue
        print(f"ai detection {ai_files_list_item}")
        rasterarray = raster2array(ai_files_list_item)
        rasterarray[rasterarray < 50] = 1
        rasterarray[rasterarray >= 50] = 0
        rasterarray_list.append(rasterarray)
    stack_array_sum = None
    if len(rasterarray_list) >0:
        stack_array = np.stack(rasterarray_list, axis=0)
        stack_array_sum = np.nansum(stack_array, axis=0)
        stack_array_sum[stack_array_sum>0] = 1
    return stack_array_sum

def create_ai_mcd_merged_detection(raster_array_1, raster_array_2):
    # Step 2: Create binary masks for non-zero pixels
    mask1 = raster_array_1 > 0
    mask2 = raster_array_2 > 0

    raster_array_1[np.isnan(raster_array_1)] = 0
    raster_array_2[np.isnan(raster_array_2)] = 0

    # Step 3: Perform dilation with a larger kernel (5 or 10 pixels)
    dilated_mask1 = binary_dilation(mask1, iterations=10)  # Adjust iterations for larger dilation

    # Step 4: Identify pixels that are adjacent but do not overlap
    adjacent_mask = np.logical_and(dilated_mask1, np.logical_not(mask1))
    adjacent_touching_mask = np.logical_and(adjacent_mask, mask2)

    # Step 5: Extract pixels from raster2 that overlap with the adjacent_touching_mask
    overlapping_pixels = np.where(adjacent_touching_mask, raster_array_2, 0)

    # Step 6: Add the overlapping pixels to raster1
    final_raster = raster_array_1 + overlapping_pixels

    return final_raster

# Apply GDAL Sieve filter to remove patches smaller than 10 pixels
def apply_sieve(input_path, output_path, threshold=10, connectivity=8):
    output_path_small_sieve = str(Path(output_path).with_suffix('.small_sieve.tif'))
    src_ds = gdal.Open(str(input_path), gdal.GA_ReadOnly)
    drv = gdal.GetDriverByName('GTiff')

    creation_options = [
        "TILED=YES",         # Enables internal tiling (good for performance)
        "COMPRESS=LZW",      # LZW compression reduces file size
        "PREDICTOR=2"        # Improves compression for integer data
    ]
    out_ds = drv.CreateCopy(
        str(output_path_small_sieve),
        src_ds,
        strict=0,
        options=creation_options
    )
    # GDAL SieveFilter modifies in-place
    band = out_ds.GetRasterBand(1)
    gdal.SieveFilter(band, None, band, threshold, connectivity)

    out_ds.FlushCache()
    out_ds = None

    output_path_large_sieve = str(Path(output_path).with_suffix('.large_sieve.tif'))
    cmd_sieve = ["gdal_sieve.py", "-st", "{}".format(50000),
                 "-8", "-nomask", "-of", "GTiff",
                 "{}".format(str(output_path_small_sieve)),
                 "{}".format(output_path_large_sieve)]
    cmd_output = subprocess.run(cmd_sieve, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("exit code {} --> {}".format(cmd_output.returncode, cmd_sieve))

    CHANGE_VALIDITY = raster2array(output_path_small_sieve)
    CHANGE_VALIDITY_largesieve = raster2array(output_path_large_sieve)
    CHANGE_VALIDITY[CHANGE_VALIDITY_largesieve == 1] = 0
    save_raster_template(input_path, output_path, CHANGE_VALIDITY, GDT_Byte, 0)
    if os.path.exists(output_path_large_sieve):
        os.remove(output_path_large_sieve)
    if os.path.exists(output_path_small_sieve):
        os.remove(output_path_small_sieve)




def main():

    for tile_list_item in tile_list:
        print(f"-------------   {tile_list_item}   ---------------------------")

        output_dir_tile = output_dir.joinpath(tile_list_item)
        os.makedirs(output_dir_tile, exist_ok=True)

        print(f"tile name {tile_list_item}")
        work_dir = root_work_dir.joinpath(tile_list_item)
        os.makedirs(work_dir, exist_ok=True)

        #####
        change_detection_tile_folder = change_detection_root_folder.joinpath(tile_list_item)
        if not change_detection_tile_folder.exists(): raise Exception(f"{change_detection_tile_folder} doesnt exist")

        change_detection_tile_files = os.listdir(change_detection_tile_folder)
        mcd_files = [Path(change_detection_tile_folder).joinpath(f)  for f in change_detection_tile_files if f.endswith("_MCD.tif")]
        # Create a sorted dictionary with datetime as the key and file path as the value
        mcd_files_dict = {
            datetime.strptime(x.name.split('_')[2], "%Y-%m-%d"): {
                "MCD": x,
                "AIMCD": x.with_name(x.name.replace("MCD", "AIMCD"))
            }
            for x in mcd_files
        }

        # Sort the dictionary by keys (datetime)
        mcd_files_sorted = dict(sorted(mcd_files_dict.items(), key=lambda item: item[0]))
        # print(mcd_files_sorted)
        #####
        # Get the first file in the sorted dictionary as the template
        template_file = list(mcd_files_sorted.values())[0]["MCD"]
        (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box) = read_raster_info(template_file)
        with rasterio.open(template_file) as src:
            epsg = src.crs.to_epsg()
        #####
        tile_forest_elevation_mask_path = work_dir.joinpath(f"{tile_list_item}_forest_dilatedelevation_mask.tif")

        if not tile_forest_elevation_mask_path.exists():
            tile_dilatedelevation_path = work_dir.joinpath(f"{tile_list_item}_elevation_dilation.tif")
            if not tile_dilatedelevation_path.exists():
                tile_elevation_path = work_dir.joinpath(f"{tile_list_item}_elevation.tif")
                if not tile_elevation_path.exists():
                    reproject_multibandraster_toextent(elevation_tifpath, tile_elevation_path, epsg, pixel_width,
                                                       xmin, imagery_extent_box.bounds[2], imagery_extent_box.bounds[1], ymax, work_dir= work_dir, method ='near')
                elevation_array = raster2array(tile_elevation_path)
                if int(tile_list_item[0:2]) < 20:
                    ELEVATION_MIN = 180
                else:
                    ELEVATION_MIN = 40
                elevation_array[elevation_array < ELEVATION_MIN] = 0
                elevation_array[elevation_array > ELEVATION_MAX] = 0
                elevation_array[elevation_array != 0] = 1

                # Invert the mask so that 0s become 1s and vice versa
                inverted = np.logical_not(elevation_array)
                # Erode the inverted mask to grow the original 0 regions
                expanded_zeros = binary_erosion(inverted, structure=np.ones((3, 3)))
                # Invert back to get the final mask
                result = np.logical_not(expanded_zeros).astype(np.uint8)
                save_raster_template(tile_elevation_path, tile_dilatedelevation_path, result, GDT_Byte)

            tile_forest_mask_path = work_dir.joinpath(f"{tile_list_item}_forestmask.tif")
            if not tile_forest_mask_path.exists():
                reproject_multibandraster_toextent(forest_tifpath, tile_forest_mask_path, epsg, pixel_width,
                                                   xmin, imagery_extent_box.bounds[2], imagery_extent_box.bounds[1], ymax, work_dir= work_dir, method ='near')

            elevationmask = raster2array(tile_dilatedelevation_path).astype(bool)
            forestmask = raster2array(tile_forest_mask_path).astype(bool)
            forest_elevationmask = elevationmask & forestmask
            save_raster_template(template_file, tile_forest_elevation_mask_path, forest_elevationmask, GDT_Byte)

        #######################
        ### Z score removal ###
        mcd_zscoreerror = work_dir.joinpath(f"{tile_list_item}_mcd_zscoreerror")
        zscore_mcd_dict = {}

        zscore_ok_file = mcd_zscoreerror.joinpath("zscore.ok")
        if not zscore_ok_file.exists():
            shutil.rmtree(mcd_zscoreerror, ignore_errors=True)
            os.makedirs(mcd_zscoreerror, exist_ok=True)

            FILES_VALUES = []
            for datetime_index, mcd_ai_dict_path in mcd_files_sorted.items():
                mcd_ai_array = raster2array(mcd_ai_dict_path["MCD"])
                mcd_ai_array[np.isnan(mcd_ai_array)] = 0
                FILE_SUM = np.sum(mcd_ai_array)
                FILES_VALUES.append(FILE_SUM)


            FILES_MEAN = np.mean(FILES_VALUES)
            FILES_STD = np.std(FILES_VALUES)
            for datetime_index, mcd_ai_dict_path in mcd_files_sorted.items():
                mcd_path = mcd_ai_dict_path["MCD"]
                mcd_ai_array = raster2array(mcd_path)
                mcd_ai_array[np.isnan(mcd_ai_array)] = 0
                FILE_SUM = np.sum(mcd_ai_array)
                FILE_Z = (FILE_SUM - FILES_MEAN) / FILES_STD
                print(f"zscore = {FILE_Z} - {mcd_path.name}")
                if FILE_Z > 2.5:
                    error_filepath = mcd_zscoreerror.joinpath(mcd_path.name.replace(".tif", "_error.tif"))
                    shutil.copy(mcd_path, error_filepath)
                else:
                    zscore_mcd_path = mcd_zscoreerror.joinpath(mcd_path.name)
                    shutil.copy(mcd_path, zscore_mcd_path)
                    zscore_mcd_dict[datetime_index] = zscore_mcd_path


            with open(zscore_ok_file, 'w') as f:
                f.write('Checked and complete.\n')
        else:
            zscore_mcd_files = os.listdir(mcd_zscoreerror)
            for zscore_mcd_file_item in zscore_mcd_files:
                if zscore_mcd_file_item.endswith("error.tif") or zscore_mcd_file_item.endswith(".ok"): continue
                zscore_mcd_start_time = datetime.strptime(zscore_mcd_file_item.split("_")[2], "%Y-%m-%d")
                zscore_mcd_dict[zscore_mcd_start_time] = mcd_zscoreerror.joinpath(zscore_mcd_file_item)
        #####
        zscore_mcd_ai_dict = {
            datetime.strptime(x.name.split('_')[2], "%Y-%m-%d"): {
                "MCD": x,
                "AIMCD": change_detection_tile_folder.joinpath(x.name.replace("MCD", "AIMCD"))
            }
            for x in list(zscore_mcd_dict.values())
        }

        work_dir_aimcd_combined = work_dir.joinpath("ai_mcd_combined")
        os.makedirs(work_dir_aimcd_combined, exist_ok=True)
        mcd_ai_combined_dit = {}
        for datetime_index, zscore_mcd_ai_path in zscore_mcd_ai_dict.items():
            print(f"{datetime_index}")
            mcd_detection = zscore_mcd_ai_path["MCD"]
            ai_mcd_detection_path = work_dir_aimcd_combined.joinpath(mcd_detection.name.replace("MCD", "MCDAICOMBINED"))

            if not ai_mcd_detection_path.exists():
                mcd_array = raster2array(mcd_detection)
                mcd_array[np.isnan(mcd_array)] = 0
                ai_files_list = find_ai_files_in_time_window(zscore_mcd_ai_dict, datetime_index)
                ai_detection_timewindow = find_cummulative_aidetection(ai_files_list)
                if ai_detection_timewindow is not None:
                    ai_mcd_detection = create_ai_mcd_merged_detection(mcd_array, ai_detection_timewindow)
                else:
                    ai_mcd_detection = mcd_array
                save_raster_template(mcd_detection, ai_mcd_detection_path, ai_mcd_detection, GDT_Byte, 0)

            mcd_ai_combined_dit[datetime_index] = ai_mcd_detection_path

        #####
        ## cummulative
        #####
        master_detection_nomask_path = work_dir.joinpath(f"{tile_list_item}_nomask_cummulative_detection.tif")
        master_detection_masked_path = work_dir.joinpath(f"{tile_list_item}_masked_combined_detection.tif")
        archive_files = [master_detection_nomask_path, master_detection_masked_path]

        if not master_detection_nomask_path.exists() or not master_detection_masked_path.exists():
            tile_forest_elevation_mask = raster2array(tile_forest_elevation_mask_path)
            tile_forest_elevation_mask[np.isnan(tile_forest_elevation_mask)] = 0

            master_detection_nomask_array = np.zeros_like(tile_forest_elevation_mask)

            for datetime_index, mcd_ai_path in mcd_ai_combined_dit.items():
                mcd_ai_array = raster2array(mcd_ai_path)
                mcd_ai_array[np.isnan(mcd_ai_array)] = 0
                master_detection_nomask_array += mcd_ai_array


            save_raster_template(tile_forest_elevation_mask_path, master_detection_nomask_path, master_detection_nomask_array, GDT_Int16, 0)

            master_detection_nomask_array[tile_forest_elevation_mask == 0] = 0
            master_detection_nomask_array[master_detection_nomask_array == 1] = 0
            master_detection_nomask_array[master_detection_nomask_array > 1] = 1

            save_raster_template(tile_forest_elevation_mask_path, master_detection_masked_path,
                                 master_detection_nomask_array, GDT_Int16, 0)

        master_detection_filtered_masked_path = work_dir.joinpath(f"{tile_list_item}_masked_combined_filtered_detection.tif")
        if not master_detection_filtered_masked_path.exists():
            tile_forest_elevation_mask = raster2array(tile_forest_elevation_mask_path)
            tile_forest_elevation_mask[np.isnan(tile_forest_elevation_mask)] = 0
            master_detection_mask_array = raster2array(master_detection_masked_path)
            master_detection_mask_array[np.isnan(master_detection_mask_array)] = 0

            # Step 1: Define forest boundary (where dilation != erosion)
            forest_dil = binary_dilation(tile_forest_elevation_mask, structure=np.ones((3, 3)))
            forest_ero = binary_erosion(tile_forest_elevation_mask, structure=np.ones((3, 3)))
            forest_boundary = ((forest_dil != forest_ero) & (tile_forest_elevation_mask == 1)).astype(np.uint8)
            # Step 2: Label all change patches
            labeled_change, num = label(master_detection_mask_array)
            # Step 3: Retain patches that touch the forest boundary
            boundary_touching_labels = np.unique(labeled_change[forest_boundary == 1])
            boundary_touching_labels = boundary_touching_labels[boundary_touching_labels != 0]
            edge_changes = np.isin(labeled_change, boundary_touching_labels).astype(np.uint8)

            inner_changes = master_detection_mask_array * master_detection_mask_array
            # Apply binary opening to inner changes
            opened_inner = binary_opening(inner_changes, structure=np.ones((2, 2))).astype(np.uint8)
            # Label original inner changes
            labeled_inner, num_labels = label(inner_changes)

            # Find overlapping labels between original and opened result
            overlapping_labels = np.unique(labeled_inner[opened_inner == 1])
            overlapping_labels = overlapping_labels[overlapping_labels != 0]

            # Retain only overlapping original patches
            final_inner = np.isin(labeled_inner, overlapping_labels).astype(np.uint8)

            # Combine edge patches and retained inner patches
            final_changes = (edge_changes | final_inner).astype(np.uint8)

            save_raster_template(tile_forest_elevation_mask_path, master_detection_filtered_masked_path,
                                 final_changes, GDT_Byte, 0)


        master_detection_filtered_sieved_masked_path = master_detection_filtered_masked_path.parent.joinpath(f"{master_detection_filtered_masked_path.stem}_sieved.tif")
        master_detection_filtered_sieved_masked_path.unlink(missing_ok =True)
        if not master_detection_filtered_sieved_masked_path.exists():
            apply_sieve(master_detection_filtered_masked_path, master_detection_filtered_sieved_masked_path, threshold=100)

        archive_files.append(master_detection_filtered_sieved_masked_path)
        #####
        #####
        # APPLY MASK
        #####
        #####
        valid_changes = raster2array(master_detection_filtered_sieved_masked_path)
        valid_changes[np.isnan(valid_changes)] = 0
        masked_mcd_ai_combined_dict = {}

        work_dir_final_deforestation = work_dir.joinpath("changedetection_layers")
        os.makedirs(work_dir_final_deforestation, exist_ok=True)
        for datetime_index, mcd_ai_path in mcd_ai_combined_dit.items():

            final_deforestation_tile_layer_path = work_dir_final_deforestation.joinpath(mcd_ai_path.name.replace("MCDAICOMBINED", "CHANGEDETECTION"))
            final_deforestation_tile_layer_path.unlink(missing_ok=True)
            if not final_deforestation_tile_layer_path.exists():
                mcd_ai_array = raster2array(mcd_ai_path)
                mcd_ai_array[np.isnan(mcd_ai_array)] = 0
                mcd_ai_array[valid_changes == 0] = 0

                save_raster_template(mcd_ai_path, final_deforestation_tile_layer_path, mcd_ai_array,
                                     GDT_Byte, 0)
                set_colormap(final_deforestation_tile_layer_path)
            archive_files.append(final_deforestation_tile_layer_path)

        #####
        #####
        # CREATE PROGRESSIVE FORESTCHANGE
        #####
        #####
        # Sort the dictionary by datetime keys in ascending order
        sorted_dict = dict(sorted(mcd_ai_combined_dit.items()))

        work_dir_final_deforestation = work_dir.joinpath("deforestation_layers")
        os.makedirs(work_dir_final_deforestation, exist_ok=True)
        for datetime_index, mcd_ai_path in sorted_dict.items():
            final_deforestation_tile_layer_path = work_dir_final_deforestation.joinpath(
                mcd_ai_path.name.replace("MCDAICOMBINED", "TREECOVERCHANGE"))
            mcd_ai_array = raster2array(mcd_ai_path)
            mcd_ai_array[np.isnan(mcd_ai_array)] = 0

            mcd_ai_current_deforestation_removed = (mcd_ai_array.astype(int)) & (valid_changes.astype(int))
            valid_changes = mcd_ai_current_deforestation_removed + valid_changes
            save_raster_template(mcd_ai_path, final_deforestation_tile_layer_path, mcd_ai_current_deforestation_removed,
                                 GDT_Byte, 0)
            set_colormap(final_deforestation_tile_layer_path)
            archive_files.append(final_deforestation_tile_layer_path)

        output_dir_tile_changedetection = output_dir_tile.joinpath("changedetection")
        os.makedirs(output_dir_tile_changedetection, exist_ok=True)
        for archive_file_item in archive_files:
            if "CHANGEDETECTION" in archive_file_item.name:
                output_tif_path = output_dir_tile_changedetection.joinpath(archive_file_item.name)
                shutil.copy(archive_file_item, output_tif_path)
            else:
                output_tif_path = output_dir_tile.joinpath(archive_file_item.name)
                shutil.copy(archive_file_item, output_tif_path)


if __name__ == "__main__":
    main()