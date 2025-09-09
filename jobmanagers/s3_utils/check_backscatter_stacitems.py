import os
import json
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path

def is_remote(path):
    return urlparse(path).scheme in ("http", "https")

def load_json(path):
    if is_remote(path):
        response = requests.get(path)
        response.raise_for_status()
        return response.json()
    else:
        with open(path) as f:
            return json.load(f)

def count_stac_items(start_path):
    visited = set()
    item_count = 0

    def walk_stac(file_path):
        nonlocal item_count

        if file_path in visited:
            return
        visited.add(file_path)

        try:
            content = load_json(file_path)
        except Exception:
            return

        base_path = os.path.dirname(file_path) if not is_remote(file_path) else file_path.rsplit('/', 1)[0]

        for link in content.get("links", []):
            href = link.get("href")
            if not href:
                continue
            next_path = urljoin(base_path + "/", href) if is_remote(base_path) else os.path.normpath(os.path.join(base_path, href))
            if link.get("rel") == "item":
                item_count += 1
            elif link.get("rel") in ["child"]:
                walk_stac(next_path)

    walk_stac(start_path)
    return item_count

# Example usage
tile_list = os.listdir("/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter/stac_dir")
backscatter_folder = Path("/mnt/hddarchive.nfs/amazonas_dir/work_dir/sarbackscatter")
ignore_list = ['22MEB', '22MFS', '22MFT', '22MGS', '22MHT']
missing_tiles = []
for tile_name in tile_list:
    if tile_name in ignore_list: continue
    catalog_url = f"https://s3.waw3-1.cloudferro.com/swift/v1/deforestation/sarbackscatter/stac_dir/{tile_name}/{tile_name}_backscatter_catalog.json"
    catalog_count = count_stac_items(catalog_url)

    backscatter_tile_folder = backscatter_folder.joinpath(tile_name)
    backscatter_count = len(os.listdir(backscatter_tile_folder))
    print(f"📦 Number of STAC items: {catalog_count} = {backscatter_count} backscatter count")
    if catalog_count != backscatter_count-1:
        missing_tiles.append((tile_name, backscatter_count-1, catalog_count))
print(missing_tiles)

