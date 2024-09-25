import os
import gdal
from pathlib import Path
import numpy as np

######################################################
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
tile_list = ['18LVQ', '18LVR', '18LWR', '18NXG', '18NXH', '18NYH', '20LLP', '20LLQ', '20LMP', '20LMQ', '20NQF', '20NQG', '20NRG', '21LYG', '21LYH', '22MBT', '22MGB']

######################################################
archive_folder = amazonas_root_folder.joinpath('output')
archive_mosaic_folder = archive_folder.joinpath("Mosaic")
######################################################
POL = ['VV', 'VH']



def get_date_from_filename(filename):
    """Extract date from filename and return as integer."""
    return int(filename.split('_')[-1].split('.')[0])

def fill_min(src_ds):
    src_band = src_ds.GetRasterBand(1)

    nodata = src_band.GetNoDataValue()
    src_data = src_band.ReadAsArray()

    nodata_indices = src_data == nodata
    src_data[nodata_indices] = -25
    return src_data




def fill_nodata_from_previous(src_ds, prev_ds):
    """Fill NoData values in src_ds with values from prev_ds and return the filled array."""
    src_band = src_ds.GetRasterBand(1)
    prev_band = prev_ds.GetRasterBand(1)

    nodata = src_band.GetNoDataValue()
    src_data = src_band.ReadAsArray()
    prev_data = prev_band.ReadAsArray()

    if src_data.ndim != 2 or prev_data.ndim != 2:
        raise ValueError("Unexpected data dimensions. Both src_data and prev_data should be 2D arrays.")

    # Get the indices where src_data is nodata
    nodata_indices = src_data == nodata

    # Replace the nodata values in src_data with values from prev_data
    src_data[nodata_indices] = prev_data[nodata_indices]

    count_nodata = np.sum(nodata_indices == True)
    return src_data, count_nodata

def main():


    for tile_item in tile_list:
        archive_mosaic_tile_folder = archive_mosaic_folder.joinpath(tile_item)
        orbit_directions = os.listdir(archive_mosaic_tile_folder)
        for orbit_direction in orbit_directions:
            folder_path = archive_mosaic_tile_folder.joinpath(orbit_direction)
            if not folder_path.is_dir(): continue

            for polarization in POL:

                # Get list of .tif files in folder
                files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and f"_{polarization}_" in f and 'filled' not in f]

                # Sort files by date in descending order
                sorted_files = sorted(files, key=get_date_from_filename, reverse=True)

                driver = gdal.GetDriverByName("GTiff")

                for i, filename in enumerate(sorted_files):
                    src_file = os.path.join(folder_path, filename)
                    print(f"doing filling for {src_file}")
                    src_ds = gdal.Open(src_file, gdal.GA_ReadOnly)  # Open in update mode to write changes

                    filled_file = os.path.join(folder_path, filename.replace('.tif', '_filled.tif'))
                    if Path(filled_file).exists(): continue

                    filled_ds = driver.CreateCopy(filled_file, src_ds, 0)

                    count_nodata = 0
                    # Iterate over previous files to fill nodata values
                    for j in range(i + 1, len(sorted_files)):
                        prev_file = os.path.join(folder_path, sorted_files[j])
                        prev_ds = gdal.Open(prev_file, gdal.GA_ReadOnly)

                        filled_data, count_nodata = fill_nodata_from_previous(filled_ds, prev_ds)
                        filled_band = filled_ds.GetRasterBand(1)
                        filled_band.WriteArray(filled_data)

                        # Close the previous dataset
                        prev_ds = None
                        if count_nodata == 0:
                            break

                    if count_nodata > 0:
                        filled_data = fill_min(filled_ds)
                        filled_band = filled_ds.GetRasterBand(1)
                        filled_band.WriteArray(filled_data)

                    # Close the source and filled datasets
                    src_ds = None
                    filled_ds = None

if __name__ == "__main__":
    main()
