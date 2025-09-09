import h5py
import numpy as np
from pathlib import Path

# Paths to your HDF5 files
input_hdf5_file = Path("/mnt/hddarchive.nfs/amazonas_dir/training/hdf5_folder/combined_dataset.hdf5")
output_hdf5_file = Path("/mnt/hddarchive.nfs/amazonas_dir/training/hdf5_folder/filtered_dataset.hdf5")
chunk_size = 5000

# Open the input HDF5 file for reading
with h5py.File(input_hdf5_file, 'r') as input_h5:
    # Access datasets
    data = input_h5["data"]  # Assuming "data" is the name of the dataset
    label = input_h5["label"]  # Assuming "label" is the name of the dataset

    num_samples = data.shape[0]
    sample_shape = data.shape[1:]  # Shape of a single data sample
    label_shape = label.shape[1:]  # Shape of a single label

    # Create temporary lists to store filtered indices
    filtered_indices = []

    # Process data in chunks
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        print(f"Processing samples {start_idx} to {end_idx}...")

        # Load a chunk of data
        data_chunk = label[start_idx:end_idx]
        mask = np.sum(data_chunk == 1, axis=tuple(range(1, label.ndim))) > 400

        # Get the indices within the chunk where the condition is met
        chunk_filtered_indices = np.where(mask)[0] + start_idx
        filtered_indices.extend(chunk_filtered_indices)
        print(f"Number of filtered samples: {len(filtered_indices)}")

    # Total number of filtered samples
    num_filtered = len(filtered_indices)
    print(f"Number of filtered samples: {num_filtered}")

    # Open the output HDF5 file for writing
    with h5py.File(output_hdf5_file, 'w') as output_h5:
        # Create datasets in the new file
        filtered_data_set = output_h5.create_dataset(
            "data", shape=(num_filtered,) + sample_shape, dtype=data.dtype
        )
        filtered_label_set = output_h5.create_dataset(
            "label", shape=(num_filtered,) + label_shape, dtype=label.dtype
        )

        # Write the filtered data and labels in chunks
        write_idx = 0
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            print(f"Saving filtered samples for chunk {start_idx} to {end_idx}...")

            # Load a chunk of data
            data_chunk = data[start_idx:end_idx]
            label_chunk = label[start_idx:end_idx]

            # Apply the same mask as before
            mask = np.sum(data_chunk == 1, axis=tuple(range(1, data.ndim))) > 400
            chunk_filtered_indices = np.where(mask)[0]

            # Write the filtered samples to the output datasets
            for i in chunk_filtered_indices:
                filtered_data_set[write_idx] = data_chunk[i]
                filtered_label_set[write_idx] = label_chunk[i]
                write_idx += 1

print("Filtered data and labels have been successfully saved to the new HDF5 file.")

