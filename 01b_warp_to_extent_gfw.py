import geopandas as gpd
import requests
import os
import json
import subprocess
from pathlib import Path



def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_value(key, data):
    return data.get(key, "Key not found")

# Main function to execute the process
def main(s2_gpkg_path, gfw_gpkg_path, s2_gfw_json_path, sentinel_tile_id, output_dir="gfw_data"):

    data = load_json_file(s2_gfw_json_path)

    if sentinel_tile_id in data.keys():
        # Replace with the key you want to search for
        value = get_value(sentinel_tile_id, data)
        value = value[0]

    print(f"value {value}")


if __name__ == "__main__":
    s2_gpkg_path = "/mnt/hddarchive.nfs/amazonas_dir/support_data/S2_tiles.gpkg"
    gfw_gpkg_path = "/mnt/hddarchive.nfs/amazonas_dir/support_data/gftw_tileextents.gpkg"
    s2_gfw_json_path = "/mnt/hddarchive.nfs/amazonas_dir/support_data/tile_map.json"
    sentinel_tile_id_list = ["22MBT", "21LYH", "21LYG", "22MGB", "20NRG", "20NQG", "20NQF", "20LMQ", "20LLQ", "18NYH", "18NXH", "18NXG", "18LWR", "18LVR", "18LVQ"]


    for sentinel_tile_id_item in sentinel_tile_id_list:
        main(s2_gpkg_path, gfw_gpkg_path, s2_gfw_json_path, sentinel_tile_id_item)
