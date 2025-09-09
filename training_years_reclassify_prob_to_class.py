







import os
from pathlib import Path
import numpy as np
from tondortools.tool import raster2array, save_raster_template
from gdalconst import GDT_Byte, GDT_Float32

cutoff_prob = 0.2

def main():
    tiles = ['21LYG']
    model = "best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score"
    years = ['2017','2018']


    for tile_item in tiles:
        folder_path = f'/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/{model}/training_years_prediction/{tile_item}'

        folder_path_files = os.listdir(folder_path)
        orbit_directions = []
        for folder_path_file_item in folder_path_files:
            if folder_path_file_item in ["ascending", "descending"]:
                orbit_directions.append(folder_path_file_item)

        for orbit_direction_item in orbit_directions:
            folder_path_orbit = Path(folder_path).joinpath(orbit_direction_item)
            files_in_folder = os.listdir(folder_path_orbit)

            prob_files = [file_item for file_item in files_in_folder if file_item.endswith("PROB.tif")]

            for prob_file_item in prob_files:
                prob_file_item_path = folder_path_orbit.joinpath(prob_file_item)
                prediction_prob = raster2array(prob_file_item_path)

                class_file_item_path = folder_path_orbit.joinpath(prob_file_item.replace("PROB.tif", "CLASS.tif"))
                prediction_class = (prediction_prob > cutoff_prob).astype(np.byte)
                save_raster_template(str(prob_file_item_path), str(class_file_item_path),
                                     prediction_class, GDT_Byte, nodata_value=None)


if __name__ == "__main__":
    main()
