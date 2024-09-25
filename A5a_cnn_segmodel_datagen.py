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




ref_training_folder = amazonas_root_folder.joinpath("ref_training")
cnn_training_folder = ref_training_folder.joinpath("data")
cnn_label_folder = ref_training_folder.joinpath("label")

cnn_label_tiles = os.listdir(cnn_label_folder)
cnn_training_tiles = os.listdir(cnn_training_folder)





learning_rate = 0.01
cutoff_prob = 0.5
batch_size = 30
loss = 'binary_crossentropy'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("ref_model")
os.makedirs(model_folder, exist_ok=True)






model_version = "build_vgg16_segmentation_batchingestion_fourthrun"
stack_training_in_one = True
TRAIN_NEW_MODEL = False

def load_data_from_pairs(label_training_pairs):
    data_list, label_list = [], []
    for label_training_pair_item in label_training_pairs:
        label_path = label_training_pair_item["label"]
        data_path = label_training_pair_item["training"]

        data_array = np.load(data_path).astype(np.uint16)
        label_array = np.load(label_path).astype(np.uint16)
        transposed_label_array = np.transpose(label_array, (1, 2, 0))

        data_list.append(data_array)
        label_list.append(transposed_label_array)

    return np.array(data_list), np.array(label_list)




class SegmentationDataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=8, target_size=(256, 256), shuffle=True, flip_horizontal=False,
                     flip_vertical=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
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

            data_array = np.load(data_path).astype(np.uint16)
            label_array = np.load(label_path).astype(np.uint16)
            transposed_label_array = np.transpose(label_array, (1, 2, 0))

            data_list.append(data_array)
            label_list.append(transposed_label_array)

            # Apply horizontal flip
            if self.flip_horizontal:
                stacked_data_flip_hori = np.flip(data_array, axis=1)
                label_array_flip_hori = np.flip(transposed_label_array, axis=1)

                data_list.append(stacked_data_flip_hori)
                label_list.append(label_array_flip_hori)

            # Apply vertical flip
            if self.flip_vertical:
                stacked_data_flip_vert = np.flip(data_array, axis=0)
                label_array_flip_vert = np.flip(transposed_label_array, axis=0)

                data_list.append(stacked_data_flip_vert)
                label_list.append(label_array_flip_vert)

        return np.array(data_list), np.array(label_list)


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

# Split dataset into training + validation (80%) and test (20%)
train_val_dataset, test_dataset = train_test_split(label_training_pairs, test_size=0.1, random_state=4)

# Split training + validation into actual training (64%) and validation (16%) sets
train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, random_state=4)
print("----------------------")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create data generators for training and validation sets
train_generator = SegmentationDataGenerator(train_dataset, batch_size=batch_size, target_size=(256, 256),
                                            flip_horizontal=True, flip_vertical=True)
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

    history = model.fit(train_generator, validation_data=val_generator, epochs=25, batch_size=batch_size, callbacks=callbacks, use_multiprocessing=True,
                                  workers=5)


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
