from pathlib import Path
import os, requests
import subprocess
import geopandas as gpd
import json
from osgeo import gdal
from gdalconst import GDT_Byte, GDT_Float32
import numpy as np
from datetime import datetime, timedelta, date
import math
from tondortools.tool import read_raster_info, save_raster, mosaic_tifs
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_data_from_pairs(label_training_pairs):
    data_list, label_list = [], []
    for label_training_pair_item in label_training_pairs:
        label_path = label_training_pair_item["label"]
        data_path = label_training_pair_item["training"]

        data_array = np.load(data_path)
        label_array = np.load(label_path)
        transposed_label_array = np.transpose(label_array, (1, 2, 0))

        data_list.append(data_array)
        label_list.append(transposed_label_array)

    return np.array(data_list), np.array(label_list)


########################################################################################################################
model_version = 'build_vgg16_segmentation_batchingestion'
cutoff_prob = 0.5


amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("ref_model")

model_filepath = model_folder.joinpath(f"model_{model_version}.h5")
loaded_model = tf.keras.models.load_model(model_filepath)

########################################################################################################################
ref_training_folder = amazonas_root_folder.joinpath("ref_training")
cnn_training_folder = ref_training_folder.joinpath("data")
cnn_label_folder = ref_training_folder.joinpath("label")

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
########################################################################################################################
# Split dataset into training + validation (80%) and test (20%)
train_val_dataset, test_dataset = train_test_split(label_training_pairs, test_size=0.1, random_state=42)

# Split training + validation into actual training (64%) and validation (16%) sets
train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, random_state=42)
print("----------------------")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

test_dataset = test_dataset
X_test, y_test = load_data_from_pairs(test_dataset)

y_pred = loaded_model.predict(X_test)
y_pred_class = (y_pred > cutoff_prob).astype(int)



test_target_flatten = y_test.flatten()
model_output_classif_flatten = y_pred_class.flatten()
cm = confusion_matrix(test_target_flatten, model_output_classif_flatten) #true - rows, predict - cols
print(cm)
print()

cr=classification_report(test_target_flatten, model_output_classif_flatten)
print(cr)

