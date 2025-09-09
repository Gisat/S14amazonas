import os
import subprocess
import numpy as np
from osgeo import gdal
from pathlib import Path

def convert_to_uint16(input_path):
    input_path = Path(input_path)

    # Open the input .tif file
    dataset = gdal.Open(str(input_path))
    if dataset is None:
        print(f"Failed to open {input_path}")
        return

    # Read the data into a NumPy array
    band = dataset.GetRasterBand(1)  # Assuming single-band raster
    data = band.ReadAsArray()

    # Convert the array to UInt16
    data_uint16 = data.astype(np.uint16)

    # Get the output path (overwrite original)
    output_path = input_path

    # Create a driver to write the data back to disk
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        print(f"Failed to get GTiff driver for {input_path}")
        return

    # Create a new dataset with the same shape and type
    out_dataset = driver.Create(
        str(output_path),
        dataset.RasterXSize,
        dataset.RasterYSize,
        1,  # Single band
        gdal.GDT_UInt16,  # Output datatype
        options=["COMPRESS=LZW"]  # Compression options
    )

    # Write the converted data to the output dataset
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(data_uint16)

    # Copy the georeferencing information and projection
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())

    # Close datasets
    dataset = None
    out_dataset = None

    print(f"Converted and replaced: {input_path}")

def process_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".tif"):
                full_path = os.path.join(dirpath, filename)
                convert_to_uint16(full_path)

# Set your root directory path here
root_directory = "/mnt/hddarchive.nfs/amazonas_dir/work_dir/changedetection_raw"
process_directory(root_directory)
