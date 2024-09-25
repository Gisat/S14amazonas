import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
import json
import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split

from cnn_architectures import build_vgg16_segmentation_bn

def collapse_to_index(x):
    """Collapse probabilities to index."""
    return tf.expand_dims(tf.argmax(x, axis=-1, output_type=tf.int32), axis=-1)

def load_data_from_hdf5(h5_file_path, batch_size=32, count=None):
    """Load data from HDF5 file in batches."""
    try:
        with h5py.File(h5_file_path, 'r') as f:
            total_samples = len(f['data']) if count is None else count

            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                data_batch = f['data'][start:end]
                labels_batch = f['label'][start:end]

                data_list, label_list = [], []
                for data_array, label_array in zip(data_batch, labels_batch):
                    stacked_data = np.transpose(data_array, (1, 2, 0, 3)).reshape(256, 256, 15)
                    # Normalize labels to be binary (0 or 1)
                    label_array = np.where(label_array > 0, 1, 0).astype(np.int32)

                    data_list.append(stacked_data)
                    label_list.append(label_array)

                data_array = np.array(data_list)
                label_array = np.array(label_list)

                # Debugging: print unique values in labels and check data shapes
                unique_values = np.unique(label_array)
                print(f"Batch shape: {data_array.shape}, Label shape: {label_array.shape}, Unique values in labels: {unique_values}")
                assert np.all(np.isin(unique_values, [0, 1])), f"Unexpected label values: {unique_values}"

                yield data_array, label_array
    except Exception as e:
        print(f"Error loading data from HDF5: {e}")
        raise

def data_generator(h5_file_path, batch_size=32, count=None):
    """Generator function to load data in batches."""
    while True:
        for data_batch, label_batch in load_data_from_hdf5(h5_file_path, batch_size=batch_size, count=count):
            yield data_batch, label_batch

model_version = 'ver7_Segmod_loadcount5000'
h5_file_path = '/mnt/hddarchive.nfs/amazonas_dir/training/hdf5_folder/combined_dataset_labelmorethan60.hdf5'
learning_rate = 0.001
loss = 'binary_crossentropy'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("model")
os.makedirs(model_folder, exist_ok=True)

batch_size = 32
total_samples = 16000
steps_per_epoch = total_samples // batch_size
validation_steps = (total_samples // 5) // batch_size

model = build_vgg16_segmentation_bn((256, 256, 15))
class_weight = {0: 0.5035927282687647, 1: 70.08500095136921}

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
callbacks = [reduce_lr, early_stop]

history = model.fit(
    data_generator(h5_file_path, batch_size=batch_size, count=total_samples),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(h5_file_path, batch_size=batch_size, count=total_samples // 5),
    validation_steps=validation_steps,
    class_weight=class_weight,
    epochs=25,
    callbacks=callbacks
)

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

fig_filepath = model_folder.joinpath(f"training_validation_accuracy_{model_version}.png")
plt.savefig(str(fig_filepath))

model_filepath = model_folder.joinpath(f"model_{model_version}.h5")
model.save(str(model_filepath))

params = {
    "model_version": model_version,
    "h5_file_path": h5_file_path,
    "model_folder": str(model_folder),
    "model_filepath": str(model_filepath),
    "input_shape": list((256, 256, 15)),
    "num_classes": 1,
    "optimizer": {
        "type": "Adam",
        "learning_rate": learning_rate
    },
    "loss": loss,
    "metrics": ["accuracy"],
    "reduce_lr": {
        "monitor": "val_loss",
        "factor": 0.2,
        "patience": 5,
        "min_lr": 0.0001,
        "verbose": 1
    },
    "early_stop": {
        "monitor": "val_loss",
        "patience": 5,
        "verbose": 1,
        "restore_best_weights": True
    },
    "learning_rate": learning_rate,
    "test_size": 0.2,
    "random_state": 42,
    "epochs": 25,
    "batch_size": batch_size
}

json_filepath = model_folder.joinpath(f"params_{model_version}.json")
with open(json_filepath, 'w') as json_file:
    json.dump(params, json_file, indent=4)

print("Training completed and parameters saved.")
