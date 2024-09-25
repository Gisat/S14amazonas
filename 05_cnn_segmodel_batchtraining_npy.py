import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Input, Conv2D, MaxPooling2D, LSTM, Reshape, Conv2DTranspose, TimeDistributed, Flatten, Dense, UpSampling2D
from tensorflow.keras.models import Model
import json
from sklearn.model_selection import train_test_split
import os
import numpy as np

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


# Path to folders
cnn_training_folder = Path('/mnt/hddarchive.nfs/amazonas_dir/training/data')
cnn_label_folder =Path( '/mnt/hddarchive.nfs/amazonas_dir/training/label')

learning_rate = 0.001
loss = 'binary_crossentropy'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("model")
os.makedirs(model_folder, exist_ok=True)



year_list = [2018, 2019]
timeperiod = 60
mosaic_gap = 12
block_size = 256
batch_size = 100
min_labelpixel = 25
number_of_mosaic = int(timeperiod/mosaic_gap)

model_version = "build_vgg16_segmentation_batchingestion"
stack_training_in_one = True




class SegmentationDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=8, target_size=(256, 256), shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        batch_dataset = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(batch_dataset)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.dataset)

    def __data_generation(self, detection_mosaic_pair):
        data_list, label_list = [], []
        for detection_mosaic_pair_item in detection_mosaic_pair:

            label_path = detection_mosaic_pair_item["label"]
            data_path = detection_mosaic_pair_item["training"]

            data_array = np.load(data_path)
            stacked_data = np.transpose(data_array, (1, 2, 0, 3)).reshape(256, 256, 15)

            label_array = np.load(label_path)

            data_list.append(stacked_data)
            label_list.append(label_array)

        print(f" number of chunks {len(data_list)} {len(label_list)}--")
        return np.array(data_list), np.array(label_list)


cnn_label_tiles = os.listdir(cnn_label_folder)
cnn_training_tiles = os.listdir(cnn_training_folder)



label_training_pairs = []
data_files = sorted([f for f in os.listdir(cnn_training_folder) if f.startswith('data_') and f.endswith('.npy')])
label_files = sorted([f for f in os.listdir(cnn_label_folder) if f.startswith('label_') and f.endswith('.npy')])
for data_file in data_files:
    data_file_index = data_file.split('_')[1].split('.')[0]
    label_file_name = f"label_{data_file_index}.npy"
    if cnn_label_folder.joinpath(label_file_name).exists():
        label_training_pair_dict = {"label": str(cnn_label_folder.joinpath(label_file_name)),
                                    "training": str(cnn_training_folder.joinpath(data_file))}
        label_training_pairs.append(label_training_pair_dict)
print(f"label_training_pairs: {len(label_training_pairs)}")

label_training_pairs = label_training_pairs[1 :1000]
# Split dataset into training + validation (80%) and test (20%)
train_val_dataset, test_dataset = train_test_split(label_training_pairs, test_size=0.1, random_state=42)

# Split training + validation into actual training (64%) and validation (16%) sets
train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, random_state=42)
print("----------------------")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create data generators for training and validation sets
train_generator = SegmentationDataGenerator(train_dataset, batch_size=batch_size, target_size=(256, 256))
val_generator = SegmentationDataGenerator(val_dataset, batch_size=batch_size, target_size=(256, 256))
print("----------------------")
print("--- train steps, shape, val steps ---")
train_steps = train_generator.__len__()
print(train_steps)
X,y = train_generator.__getitem__(1)
print(X.shape)
print(y.shape)
val_steps = val_generator.__len__()
print(val_steps)
print("----------------------")
class_weight = {0: 0.5035927282687647, 1: 70.08500095136921}

# Create the model
model = build_vgg16_segmentation_bn((256, 256, 15))

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
callbacks = [reduce_lr, early_stop]

history = model.fit(train_generator, validation_data=val_generator, epochs=5, callbacks=callbacks,
                              use_multiprocessing=True, class_weight=class_weight,
                              workers=6
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
