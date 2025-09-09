import os
import shutil
from pathlib import Path
import numpy as np
from tondortools.tool import save_raster_template
from osgeo.gdalconst import GDT_Byte
from src.tondor.util.tool import raster2array
import multiprocessing
from multiprocessing import Pool

change_detection_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/change_detection")
tree_cover_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/tree_cover_change")
os.makedirs(tree_cover_folder, exist_ok=True)

tiles_list = os.listdir(change_detection_folder)
CPU_COUNT = 6


def create_treecover_change_layers(input_data):
    tiles_list_item = input_data[0]

    change_detection_folder = Path(
        "/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/change_detection")
    tree_cover_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/portal_data/prior_2021/tiles/tree_cover_change")

    tree_cover_folder_tile = tree_cover_folder.joinpath(tiles_list_item)
    os.makedirs(tree_cover_folder_tile, exist_ok=True)
    nodata_txt_filepath = tree_cover_folder_tile.joinpath(f"{tiles_list_item}_status.txt")
    if not nodata_txt_filepath.exists():
        change_detection_tile = change_detection_folder.joinpath(tiles_list_item)

        change_detection_tile_tif_list = os.listdir(change_detection_tile)
        change_detection_tile_tif_filepath_list = [Path(change_detection_tile).joinpath(change_detection_tile_tif_filepath_list_item)
                                                   for change_detection_tile_tif_filepath_list_item in sorted(change_detection_tile_tif_list)
                                                   if "_CHANGE.tif" in change_detection_tile_tif_filepath_list_item]
        template_raster_array = raster2array(change_detection_tile_tif_filepath_list[0])
        deforestation_layer = np.zeros_like(template_raster_array)

        for change_detection_tile_tif_filepath_list_item in change_detection_tile_tif_filepath_list:
            print(f"{tiles_list_item} -- {change_detection_tile_tif_filepath_list_item.name}")
            change_detection_array = raster2array(change_detection_tile_tif_filepath_list_item)
            change_detection_array[np.isnan(change_detection_array)] = 0
            change_detection_array[deforestation_layer == 1] = 0

            deforestation_layer = deforestation_layer + change_detection_array
            deforestation_layer[deforestation_layer > 0] = 1
            tree_cover_layer_path = tree_cover_folder_tile.joinpath(change_detection_tile_tif_filepath_list_item.name)
            save_raster_template(change_detection_tile_tif_filepath_list_item, tree_cover_layer_path, change_detection_array, GDT_Byte, 0)

        cum_change_mask = change_detection_tile.joinpath(f"{tiles_list_item}_CHANGE_CONF.tif")
        cum_change_treecover = tree_cover_folder_tile.joinpath(cum_change_mask.name)
        shutil.copy(cum_change_mask, cum_change_treecover)

        cum_conf_mask = change_detection_tile.joinpath(f"{tiles_list_item}_CHANGE_MASK.tif")
        cum_change_treecover = tree_cover_folder_tile.joinpath(cum_conf_mask.name)
        shutil.copy(cum_conf_mask, cum_change_treecover)

        nodata_txt_filepath.write_text("Tile done")


args = []
for tiles_list_item in tiles_list:
    args.append([tiles_list_item])

p = Pool(CPU_COUNT)
p.map(create_treecover_change_layers, tuple(args), chunksize=1)



