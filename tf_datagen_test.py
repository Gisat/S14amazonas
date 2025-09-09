import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from pathlib import Path
from cnn_architectures import *
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import ProgbarLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Generator function to read data from HDF5 file
def data_generator(h5_file_path, indices):
    with h5py.File(str(h5_file_path), 'r') as f:

        for idx in indices:
            # Read data and labels from HDF5 file
            data = f['data'][idx]  # Adjust 'data' to your actual dataset name in the HDF5 file
            label = f['label'][idx]

            data_stacked = data.reshape(256, 256, 15)
            label = np.expand_dims(label, axis=-1)  # Add channel dimension to labels
            yield data_stacked.astype(np.float32), label.astype(np.float32)

# Path to your HDF5 file
h5_file_path = '/mnt/hddarchive.nfs/amazonas_dir/training/hdf5_folder/combined_dataset_small.hdf5'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("model")
os.makedirs(model_folder, exist_ok=True)
model_version = "build_vgg16_segmentation_batchingestion_test"

learning_rate = 0.001
loss = 'binary_crossentropy'

# Read the total number of samples in the HDF5 file
with h5py.File(h5_file_path, 'r') as f:
    total_samples = len(f['data'])

# Split the indices into training, validation, and test sets
indices = np.arange(total_samples)
print(f"Total size {total_samples}")
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Example indices for training and validation
train_indices = list(range(800))  # Replace with actual train indices
val_indices = list(range(800, 900))  # Replace with actual validation indices


# Define the output types and shapes
output_types = (tf.float32, tf.float32)
output_shapes = ((256, 256, 15), (256, 256, 1))

# Create the datasets for training and validation
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(h5_file_path, train_indices),
    output_types=output_types,
    output_shapes=output_shapes
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(h5_file_path, val_indices),
    output_types=output_types,
    output_shapes=output_shapes
)

# Shuffle, batch, and prefetch the datasets
# train_dataset = train_dataset.shuffle(buffer_size=len(train_indices)).batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)
# val_dataset = val_dataset.batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)


# Define the input shape based on your data
input_shape = (256, 256, 15)
model = build_vgg16_segmentation_bn(input_shape)

# Compile the model
optimizer = Adam(learning_rate=1e-4, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=2, restore_best_weights=True)

# Set up TensorBoard and ProgbarLogger callbacks
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
progbar_logger = ProgbarLogger(count_mode='steps')

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)


callbacks = [reduce_lr, early_stop, checkpoint]

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10,
                    callbacks=callbacks,verbose=2)  # Adjust epochs as needed



# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

# Save the plot to a file
fig_filepath = model_folder.joinpath(f"training_validation_accuracy_{model_version}.png")
plt.savefig(str(fig_filepath))


model_filepath = model_folder.joinpath(f"model_{model_version}.h5")
model.save(str(model_filepath))
