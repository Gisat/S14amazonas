import os
import glob
import numpy as np
from osgeo import gdal, gdal_array
from tondortools.tool import save_raster_template, raster2array
import shutil
import subprocess
from pathlib import Path
from gdalconst import GDT_Byte, GDT_Float32

amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
support_data = amazonas_root_folder.joinpath("support_data")
deforesation_mask_folder = support_data.joinpath("deforestation_mask")
os.makedirs(deforesation_mask_folder, exist_ok=True)

if __name__ == "__main__":
    tiles = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '18LVR', '20LLQ', '20LMP', '20LMQ', '20NQF',
             '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']
    model = "best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score"

    for tile_item in tiles:
        predictionyear_sum_sieved_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/{tile_item}/zscore_checked/{tile_item}_all_largesieve.tif'

        trainingyear_sum_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/training_years_prediction/{tile_item}/trainingyears_classsummask_{tile_item}_all.tif'

        predictionyear_sum_sieved_array = raster2array(predictionyear_sum_sieved_path)
        trainingyear_sum_array = raster2array(trainingyear_sum_path)

        combined_array_sum = predictionyear_sum_sieved_array  + trainingyear_sum_array
        combined_array_sum[combined_array_sum >0] = 1





        combined_array_sum_path = deforesation_mask_folder.joinpath(f"classsum_mask_{tile_item}.tif")
        save_raster_template(predictionyear_sum_sieved_path, combined_array_sum_path, combined_array_sum, GDT_Byte)

        tile_master_mask_path = deforesation_mask_folder.joinpath(f"master_mask_{tile_item}.tif")
        master_mask_array = raster2array(tile_master_mask_path)
        master_mask_inverted_array = np.where(master_mask_array == 0, 1, 0)

        combined_array_sum_master = combined_array_sum + master_mask_inverted_array
        combined_array_sum_master[combined_array_sum_master >0] = 1




        combined_array_sum_master_path = deforesation_mask_folder.joinpath(f"master_classsum_mask_{tile_item}.tif")
        combined_array_sum_master_inverted = np.where(combined_array_sum_master == 0, 1, 0)
        save_raster_template(predictionyear_sum_sieved_path, combined_array_sum_master_path, combined_array_sum_master_inverted, GDT_Byte)




        print(f"{tile_item}")