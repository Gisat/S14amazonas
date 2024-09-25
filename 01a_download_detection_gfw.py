import geopandas as gpd
import requests
import os
import json
import subprocess
from pathlib import Path

def download_file(url, output_path):
    try:
        # Run the wget command with the specified URL and output path
        result = subprocess.run(['wget', '-O', output_path, url], check=True)
        print(f"File downloaded successfully to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

# Function to load the JSON file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to get the value for a specific key
def get_value(key, data):
    return data.get(key, "Key not found")

# Function to load the GeoPackage files
def load_gpkg(filepath, layer_name=None):
    return gpd.read_file(filepath, layer=layer_name)


# Function to identify overlapping GFW extents for a given Sentinel-2 tile
def identify_gfw_extent(s2_tile, gfw_gdf):
    # Ensure both are in the same coordinate reference system (CRS)
    s2_tile = s2_tile.to_crs(gfw_gdf.crs)

    # Perform spatial join to identify overlapping GFW extents
    overlapping_gfw_extents = gpd.sjoin(gfw_gdf, s2_tile, how="inner", op="intersects")
    return overlapping_gfw_extents


# Function to download data from GFW API
def download_gfw_data(gfw_extent, tile_name, output_dir="/mnt/hddarchive.nfs/amazonas_dir/Detections/gfw_raw"):


    gfw_api_url = f"https://data-api.globalforestwatch.org/dataset/gfw_integrated_alerts/latest/download/geotiff?grid=10/100000&tile_id={gfw_extent}&pixel_meaning=date_conf&x-api-key=2d60cd88-8348-4c0f-a6d5-bd9adb585a8c"
    output_path = Path(output_dir).joinpath(f"{gfw_extent}.tif")
    if not output_path.exists():
        download_file(gfw_api_url, output_path)


# Main function to execute the process
def main(s2_gpkg_path, gfw_gpkg_path, s2_gfw_json_path, sentinel_tile_id, output_dir="gfw_data"):

    data = load_json_file(s2_gfw_json_path)

    if sentinel_tile_id in data.keys():
        # Replace with the key you want to search for
        value = get_value(sentinel_tile_id, data)
        value = value[0]
    else:
        # Load the GPKG files
        s2_gdf = load_gpkg(s2_gpkg_path)
        gfw_gdf = load_gpkg(gfw_gpkg_path)

        # Identify the Sentinel-2 tile
        s2_tile = s2_gdf[s2_gdf['tile_id'] == sentinel_tile_id]  # replace 'tile_id' with actual column name

        if s2_tile.empty:
            print(f"No Sentinel-2 tile found for ID: {sentinel_tile_id}")
            return

        # Identify the overlapping GFW extents
        overlapping_gfw_extents = identify_gfw_extent(s2_tile, gfw_gdf)

        if overlapping_gfw_extents.empty:
            print("No overlapping GFW extents found.")
            return

    # Download data for the identified GFW extents
    download_gfw_data(value,output_dir)


# Example usage
if __name__ == "__main__":
    s2_gpkg_path = "/mnt/hddarchive.nfs/amazonas_dir/support_data/S2_tiles.gpkg"
    gfw_gpkg_path = "/mnt/hddarchive.nfs/amazonas_dir/support_data/gftw_tileextents.gpkg"
    s2_gfw_json_path = "/mnt/hddarchive.nfs/amazonas_dir/support_data/tile_map.json"
    sentinel_tile_id_list = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB'] # replace with actual tile ID
    sentinel_list = ["22MBT", "21LYH", "21LYG", "22MGB", "20NRG", "20NQG", "20NQF", "20LMQ", "20LLQ", "18NYH", "18NXH", "18NXG", "18LWR", "18LVR", "18LVQ"]


    for sentinel_tile_id_item in sentinel_tile_id_list:
        main(s2_gpkg_path, gfw_gpkg_path, s2_gfw_json_path, sentinel_tile_id_item)
