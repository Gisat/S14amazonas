import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Input, Conv2D, MaxPooling2D, LSTM, Reshape, Conv2DTranspose, TimeDistributed, Flatten, Dense, UpSampling2D
from tensorflow.keras.models import Model
import json
from sklearn.model_selection import train_test_split
import os
import numpy as np
import h5py

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D, BatchNormalization, Dropout
# from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from pathlib import Path
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from osgeo import gdal, osr
from copy import copy

from cnn_architectures import *

def data_generator(hdf5_file, indices, batch_size):
    while True:
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            batch_indices.sort()
            batch_data = hdf5_file['data'][batch_indices]
            batch_data_stacked = batch_data.reshape(batch_size, 256, 256, 15)
            batch_labels = hdf5_file['label'][batch_indices]
            yield batch_data_stacked, batch_labels

model_version = 'ver7_Segmod_hdf5_batchtraining_labelmorethan120'
batch_size = 100
learning_rate = 0.001
class_weight = {0: 0.5035927282687647, 1: 100.08500095136921}
print(f"Class weights: {class_weight}")
loss = 'binary_crossentropy'

amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("model")

# Load the HDF5 file
file_path = '/mnt/hddarchive.nfs/amazonas_dir/training/hdf5_folder/combined_dataset_labelmorethan120.hdf5'
hdf5_file = h5py.File(file_path, 'r')

# Get the size of the datasets
data_size = hdf5_file['data'].shape
label_size = hdf5_file['label'].shape

# Split indices into train, validation, and test sets
indices = np.arange(data_size[0])
train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.33, random_state=42)

train_generator = data_generator(hdf5_file, train_indices, batch_size)
val_generator = data_generator(hdf5_file, val_indices, batch_size)
test_gen = data_generator(hdf5_file, test_indices, batch_size)



# Create the model
model = build_vgg16_segmentation_bn((256, 256, 15))

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
callbacks = [reduce_lr, early_stop]

history = model.fit_generator(train_generator, validation_data=val_generator, epochs=10, callbacks=callbacks,
                                class_weight=class_weight,
                    use_multiprocessing=True,
                    workers=2
                    )

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)


# Save the plot to a file
fig_filepath = model_folder.joinpath(f"training_validation_accuracy_{model_version}.png")
plt.savefig(str(fig_filepath))


model_filepath = model_folder.joinpath(f"model_{model_version}.h5")
model.save(str(model_filepath))
