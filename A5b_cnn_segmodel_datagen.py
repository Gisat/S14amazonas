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
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from osgeo import gdal, osr
from copy import copy

from cnn_architectures import *

try:
    from ClassificationValidation.classification_validation import ComputeCmStatistics
except:
    from internal.ClassificationValidation.classification_validation import ComputeCmStatistics

amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
support_data = amazonas_root_folder.joinpath("support_data")
merged_sar_ref_folder = support_data.joinpath(f"merged_sar_ref_worldcover")
model_folder = amazonas_root_folder.joinpath("ref_model")



ref_training_folder = amazonas_root_folder.joinpath("ref_training")
cnn_training_folder = ref_training_folder.joinpath("data")
cnn_label_folder = ref_training_folder.joinpath("label")
training_folder_hdf5 = ref_training_folder.joinpath('hdf5_folder')
hdf5_file = training_folder_hdf5.joinpath('combined_dataset.hdf5')


def train_val_split(hdf5_file, val_split=0.2):
    """Split the data into training and validation sets."""
    with h5py.File(hdf5_file, 'r') as f:
        num_samples = f['data'].shape[0]
    indices = np.arange(num_samples)

    # Split into training and temp (temp combines validation and test)
    train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)

    # Split the temp set into validation and test
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)

    return train_indices, val_indices, test_indices


def data_generator(hdf5_file, indices, batch_size, augment = True):
    """A generator that yields batches of data and labels."""
    with h5py.File(hdf5_file, 'r') as f:
        while True:
            # Shuffle indices each epoch
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                end = min(start + batch_size, len(indices))
                batch_indices = indices[start:end]

                # Sort the batch indices to avoid the TypeError
                sorted_batch_indices = np.sort(batch_indices)

                # Fetch the batch of data and labels using sorted indices
                data_batch = f['data'][sorted_batch_indices]
                label_batch = f['label'][sorted_batch_indices]

                # Transpose label_batch if it has 3 dimensions
                if label_batch.ndim == 3:
                    label_batch = np.transpose(label_batch, (1, 2, 0))
                elif label_batch.ndim > 3:
                    # Adjust the transpose dimensions based on your actual data shape
                    # This is just an example for 3D; adapt it for your specific case
                    label_batch = np.transpose(label_batch, (0, 2, 3, 1))

                data_aug_list = []
                label_aug_list = []
                if augment:
                    for i in range(data_batch.shape[0]):
                        # Flip horizontally
                        data_batch_flip_hori = np.flip(data_batch[1], axis=1)
                        label_batch_flip_hori = np.flip(label_batch[i], axis=1)
                        data_aug_list.append(data_batch_flip_hori)
                        label_aug_list.append(label_batch_flip_hori)

                        # Flip vertically
                        data_batch_flip_vert = np.flip(data_batch[i], axis=0)
                        label_batch_flip_vert = np.flip(label_batch[i], axis=0)
                        data_aug_list.append(data_batch_flip_vert)
                        label_aug_list.append(label_batch_flip_vert)
                    data_batch_full = np.concatenate((data_batch, np.array(data_aug_list)), axis=0)
                    label_batch_full = np.concatenate((label_batch, np.array(label_aug_list)), axis=0)
                else:
                    data_batch_full = data_batch
                    label_batch_full = label_batch

                yield data_batch_full, label_batch_full


# File path
batch_size = 50
model_version = "build_vgg16_segmentation_hdf5ingestion"
learning_rate = 0.001
cutoff_prob = 0.5
loss = 'binary_crossentropy'
TRAIN_NEW_MODEL = False

# Split the data
train_indices, val_indices, test_indices = train_val_split(hdf5_file, val_split=0.1)

# Create the generators
train_gen = data_generator(hdf5_file, train_indices, batch_size, augment=True)
val_gen = data_generator(hdf5_file, val_indices, batch_size)


# Extract the first batch
first_batch_data, first_batch_labels = next(train_gen)


# Create the model
model = build_vgg16_segmentation_bn((256, 256, 3))

model_filepath = model_folder.joinpath(f"model_{model_version}.h5")

if not model_filepath.exists() or TRAIN_NEW_MODEL:

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    callbacks = [reduce_lr, early_stop]

    history = model.fit(train_gen, validation_data=val_gen, epochs=15, batch_size=32, callbacks=callbacks, use_multiprocessing=True,
                                  workers=6)


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


    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    # Save the plot to a file
    fig_filepath = model_folder.joinpath(f"training_validation_loss_{model_version}.png")
    plt.savefig(str(fig_filepath))


    model.save(str(model_filepath))
    loaded_model = model
else:
    loaded_model = tf.keras.models.load_model(model_filepath)
#####################################
## REPORT ##
log_filepath = model_folder.joinpath(f"training_validation_accuracy_{model_version}.log")
log_filepath.unlink(missing_ok=True)
logfile = open(log_filepath, "a")

## validation ##
logfile.write("------ \n")
logfile.write("--- Validation dataset --- \n")
logfile.write("------ \n")

X_val, y_val = load_data_from_pairs(val_dataset)
yval_pred = loaded_model.predict(X_val)
yval_pred_class = (yval_pred > cutoff_prob).astype(int)

test_target_flatten = y_val.flatten()
model_output_classif_flatten = yval_pred_class.flatten()

accuracy_confusion_matrix = classification_report(test_target_flatten, model_output_classif_flatten)
logfile.write(
    "classification_report from validation dataset \n")
logfile.write("{} \n".format(accuracy_confusion_matrix))
logfile.write("------ \n")

cmstat = ComputeCmStatistics(model_output_classif_flatten, test_target_flatten, weight=None)
logfile.write("{} \n".format(repr(cmstat)))
logfile.write("------ \n")

cm = confusion_matrix(test_target_flatten, model_output_classif_flatten) #true - rows, predict - cols
logfile.write("{} \n".format(repr(cm)))
logfile.write("------ \n")

## test ##
logfile.write("------ \n")
logfile.write("--- Test dataset --- \n")
logfile.write("------ \n")

X_test, y_test = load_data_from_pairs(test_dataset)
y_pred = loaded_model.predict(X_test)
y_pred_class = (y_pred > cutoff_prob).astype(int)

test_target_flatten = y_test.flatten()
model_output_classif_flatten = y_pred_class.flatten()

accuracy_confusion_matrix = classification_report(test_target_flatten, model_output_classif_flatten)
logfile.write(
    "classification_report from test dataset \n")
logfile.write("{} \n".format(accuracy_confusion_matrix))
logfile.write("------ \n")
print(accuracy_confusion_matrix)

cmstat = ComputeCmStatistics(model_output_classif_flatten, test_target_flatten, weight=None)
logfile.write("{} \n".format(repr(cmstat)))
logfile.write("------ \n")
print(cmstat)

cm = confusion_matrix(test_target_flatten, model_output_classif_flatten) #true - rows, predict - cols
logfile.write("{} \n".format(repr(cm)))
logfile.close()
print(cm)
print("-------------")
