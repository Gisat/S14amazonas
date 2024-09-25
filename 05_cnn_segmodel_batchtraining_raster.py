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


model_version = 'model_build_vgg16_segmentation_batchingestion_200000dataset'

# Path to folders
cnn_training_folder = Path('/mnt/hddarchive.nfs/amazonas_dir/output/Multiband_mosaic')
cnn_label_folder =Path( '/mnt/hddarchive.nfs/amazonas_dir/output/mcd_detection')

learning_rate = 0.001
loss = 'binary_crossentropy'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("model")
os.makedirs(model_folder, exist_ok=True)



year_list = [2018, 2019]
timeperiod = 60
mosaic_gap = 12
block_size = 256
batch_size = 1
min_labelpixel = 25
number_of_mosaic = int(timeperiod/mosaic_gap)

stack_training_in_one = True



def read_raster_info(raster_filepath):
    ds = gdal.Open(str(raster_filepath))

    RasterXSize = ds.RasterXSize
    RasterYSize = ds.RasterYSize
    gt = ds.GetGeoTransform()
    ulx_raster = gt[0]
    uly_raster = gt[3]
    lrx_raster = gt[0] + gt[1] * ds.RasterXSize + gt[2] * ds.RasterYSize
    lry_raster = gt[3] + gt[4] * ds.RasterXSize + gt[5] * ds.RasterYSize
    imagery_extent_box = None

    xmin = gt[0]
    ymax = gt[3]
    pixel_width = gt[1]
    yres = gt[5]

    projection = ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(projection)
    epsg = int(srs.GetAttrValue('AUTHORITY', 1))

    datatype = ds.GetRasterBand(1).DataType
    n_bands = ds.RasterCount
    ds = None
    return (xmin, ymax, RasterXSize, RasterYSize, pixel_width, projection, epsg, datatype, n_bands, imagery_extent_box)


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

            detection_file = detection_mosaic_pair_item["label"]
            raster_list = detection_mosaic_pair_item["training"]

            (_, _, RasterXSize, RasterYSize, _, _, _, _, _,
             _) = read_raster_info(raster_list[0])

            for x_block in range(0, RasterXSize + 1, block_size):
                for y_block in range(0, RasterYSize + 1, block_size):

                    chunk_list = []
                    save_np = True

                    if x_block + block_size < RasterXSize:
                        cols = block_size
                    else:
                        cols = RasterXSize - x_block
                        save_np = False

                    if y_block + block_size < RasterYSize:
                        rows = block_size
                    else:
                        rows = RasterYSize - y_block
                        save_np = False

                    for raster_list_item in raster_list:
                        dataset = gdal.Open(str(raster_list_item))
                        chunk = dataset.ReadAsArray(x_block, y_block, cols, rows)

                        # Transpose the data
                        transposed_chunk = chunk.transpose(1, 2, 0)

                        chunk_list.append(transposed_chunk)
                        del dataset, chunk, transposed_chunk

                    label_dataset = gdal.Open(str(detection_file))
                    label_chunk = label_dataset.ReadAsArray(x_block, y_block, cols, rows)
                    del label_dataset


                    # Convert chunk_list to a NumPy array for further processing.
                    chunk_list_array = np.array(chunk_list)

                    # If stacking is enabled and saving is required, stack and reshape data.
                    if stack_training_in_one and save_np:
                        stacked_data = np.transpose(chunk_list_array, (1, 2, 0, 3)).reshape(256, 256, 15)

                    # Create a binary mask from label_chunk, marking all non-zero values as 1.
                    label_mask = copy(label_chunk)
                    label_mask[label_mask > 0] = 1

                    # Append processed data and labels to their respective lists if saving is enabled.
                    if save_np:
                        # Choose between stacked_data and chunk_list_array based on stacking setting.
                        data_to_append = stacked_data if stack_training_in_one else chunk_list_array
                        data_list.append(data_to_append)
                        label_list.append(label_mask)

                        # Clean up by deleting variables no longer needed to free memory.
                        del data_to_append
                        if stack_training_in_one:
                            del stacked_data
                    del chunk_list, chunk_list_array, label_mask, label_chunk

        print(f" number of chunks {len(data_list)} {len(label_list)}--")
        return np.array(data_list), np.array(label_list)


cnn_label_tiles = os.listdir(cnn_label_folder)
cnn_training_tiles = os.listdir(cnn_training_folder)



label_training_pairs = []


for cnn_label_tile_item in cnn_label_tiles:
  #if not cnn_label_tile_item == '21LYG': continue
  cnn_label_tile_folder = cnn_label_folder.joinpath(cnn_label_tile_item)
  cnn_label_tile_files = os.listdir(cnn_label_tile_folder)

  found_training_tile = False
  for cnn_training_tile_item in cnn_training_tiles:
    if not cnn_training_tile_item == cnn_label_tile_item:
      continue
    else:
      found_training_tile = True
      print(f"found training tile: {cnn_training_tile_item}")

  if not found_training_tile:
    continue

  for cnn_label_tile_file_item in sorted(cnn_label_tile_files):


    cnn_label_tile_file_path = cnn_label_tile_folder.joinpath(cnn_label_tile_file_item)

    if not cnn_label_tile_file_item.endswith("LT_INT_C.tif"): continue

    label_time = cnn_label_tile_file_item.split('_')[2]
    label_date_year = int(label_time[0:4])
    if not label_date_year in year_list: continue
    label_datetime = datetime.datetime.strptime(label_time, "%Y%m%d")
    label_datetime_begin_window = label_datetime - timedelta(days = timeperiod - mosaic_gap)

    #print(f"label file {cnn_label_tile_file_item} -- {label_datetime_begin_window} <-> {label_datetime} ")


    cnn_training_tile_folder = cnn_training_folder.joinpath(cnn_label_tile_item)
    tile_orbits = os.listdir(cnn_training_tile_folder)

    for tile_orbit_item in tile_orbits:

      label_training_pair_item = {"label": None, "training": []}
      label_training_pair_item["label"] = cnn_label_tile_file_path

      cnn_training_tile_orb_folder = cnn_training_tile_folder.joinpath(tile_orbit_item)
      if not os.path.isdir(str(cnn_training_tile_orb_folder)): continue
      cnn_training_tile_orb_files = os.listdir(cnn_training_tile_orb_folder)

      for cnn_training_tile_orb_file_item in sorted(cnn_training_tile_orb_files):
        cnn_training_tile_orb_file_path = cnn_training_tile_orb_folder.joinpath(cnn_training_tile_orb_file_item)

        training_time = cnn_training_tile_orb_file_item.split("_")[3].split(".")[0]
        training_datetime = datetime.datetime.strptime(training_time, "%Y%m%d")

        if training_datetime >= label_datetime_begin_window and training_datetime <= label_datetime:
          label_training_pair_item["training"].append(cnn_training_tile_orb_file_path)


      if len(label_training_pair_item["training"]) == number_of_mosaic:
        #print(f"{label_training_pair_item}")
        label_training_pairs.append(label_training_pair_item)

print(f"label_training_pairs: {len(label_training_pairs)}")

label_training_pairs = label_training_pairs[1 :100]
# Split dataset into training + validation (80%) and test (20%)
train_val_dataset, test_dataset = train_test_split(label_training_pairs, test_size=0.1, random_state=42)

# Split training + validation into actual training (64%) and validation (16%) sets
train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create data generators for training and validation sets
train_generator = SegmentationDataGenerator(train_dataset, batch_size=batch_size, target_size=(256, 256))
val_generator = SegmentationDataGenerator(val_dataset, batch_size=batch_size, target_size=(256, 256))

print("--- train steps, shape, val steps ---")
train_steps = train_generator.__len__()
print(train_steps)
X,y = train_generator.__getitem__(1)
print(y.shape)
val_steps = val_generator.__len__()
print(val_steps)


# Create the model
model = build_vgg16_segmentation_bn((256, 256, 15))

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
callbacks = [reduce_lr, early_stop]

history = model.fit_generator(train_generator, validation_data=val_generator, epochs=25, callbacks=callbacks,
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
