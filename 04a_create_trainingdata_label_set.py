import os
import numpy as np
import h5py
from pathlib import Path

amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
########################################################################################################################
output_folder = amazonas_root_folder.joinpath('output')
output_folder_multiband_mosaic = output_folder.joinpath('Multiband_mosaic')
########################################################################################################################
work_dir = amazonas_root_folder.joinpath("work_dir")
support_data = amazonas_root_folder.joinpath("support_data")
########################################################################################################################
########################################################################################################################
training_folder = amazonas_root_folder.joinpath('training')
training_folder_label = training_folder.joinpath('label')
training_folder_data = training_folder.joinpath('data')
training_folder_hdf5 = training_folder.joinpath('hdf5_folder')
os.makedirs(training_folder_hdf5, exist_ok=True)



# Paths to your folders
data_folder = training_folder_data
label_folder = training_folder_label
output_hdf5_file = training_folder_hdf5.joinpath('combined_dataset.hdf5')

# Initialize lists to hold file paths
data_files = [os.path.join(data_folder, f) for f in sorted(os.listdir(data_folder)) if f.startswith('data_')]
label_files = [os.path.join(label_folder, f) for f in sorted(os.listdir(label_folder)) if f.startswith('label_')]

# Determine the shape of your data for preallocation (assuming all files have the same shape)
sample_data_shape = np.load(data_files[0]).shape
sample_label_shape = np.load(label_files[0]).shape

# Number of samples
num_samples = len(data_files)

# Create or open the HDF5 file
with h5py.File(output_hdf5_file, 'w') as h5f:
    # Create datasets for data and labels
    # Adjust dtype and maxshape as necessary
    data_set = h5f.create_dataset("data", shape=(num_samples,) + sample_data_shape, dtype=np.float32)
    label_set = h5f.create_dataset("label", shape=(num_samples,) + sample_label_shape, dtype=np.float32)

    # Iterate and store each npy file into the datasets
    for i in range(num_samples):
        # Load the data and label
        data_file = data_folder.joinpath(f"data_{i+1}.npy")
        label_file = label_folder.joinpath(f"label_{i+1}.npy")
        if data_file.exists() and label_file.exists():
            data = np.load(data_file)
            label = np.load(label_file)

            # Write the data and label to the respective dataset in the HDF5 file
            data_set[i, ...] = data
            label_set[i, ...] = label
        else:
            print(f"couldnt find {i}")

        if i % 1000 == 0:
            print(f"{i} -- count --")
print("Data and labels have been successfully saved to HDF5.")
