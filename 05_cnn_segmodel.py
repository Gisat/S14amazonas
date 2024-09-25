import tensorflow as tf
from keras.layers import Concatenate, Input, Conv2D, MaxPooling2D, LSTM, Reshape, Conv2DTranspose, TimeDistributed, Flatten, Dense, UpSampling2D
from keras.models import Model
import json
from sklearn.model_selection import train_test_split
import os
import numpy as np

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical

from pathlib import Path
import matplotlib.pyplot as plt

from cnn_architectures import *

def collapse_to_index(x):
    return tf.expand_dims(tf.argmax(x, axis=-1, output_type=tf.int32), axis=-1)

# Load data and labels from folders
def load_data_from_folder(data_folder, label_folder, count = None):
    data_files = sorted([f for f in os.listdir(data_folder) if f.startswith('data_') and f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.startswith('label_') and f.endswith('.npy')])

    if count is not None:
        data_files = data_files[:count]
        label_files = label_files[:count]

    data_list, label_list = [], []

    for data_file, label_file in zip(data_files, label_files):
        data_path = os.path.join(data_folder, data_file)
        label_path = os.path.join(label_folder, label_file)

        data_array = np.load(data_path)
        label_array = np.load(label_path)

        stacked_data = np.transpose(data_array, (1, 2, 0, 3)).reshape(256, 256, 15)

        data_list.append(stacked_data)
        label_list.append(label_array)

    return np.array(data_list), np.array(label_list)




model_version = 'ver7_Segmod'

# Path to folders
data_folder = '/mnt/hddarchive.nfs/amazonas_dir/training/data'
label_folder = '/mnt/hddarchive.nfs/amazonas_dir/training/label'

learning_rate = 0.001
loss = 'binary_crossentropy'
amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
model_folder = amazonas_root_folder.joinpath("model")
os.makedirs(model_folder, exist_ok=True)

# Load data
X, y = load_data_from_folder(data_folder, label_folder, count=5000)


# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(x_train.shape)  # should be (num_samples, 5, 256, 256, 3)
print(y_train.shape)  # should be (num_samples, 256, 256, 1)


# Create the model
model = build_vgg16_segmentation_bn((256, 256, 15))



# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
callbacks = [reduce_lr, early_stop]

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=25, batch_size=32, callbacks=callbacks)


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


# Save parameters to a JSON file
params = {
    "model_version": model_version,
    "data_folder": data_folder,
    "label_folder": label_folder,
    "model_folder": str(model_folder),
    "model_filepath": str(model_filepath),
    "input_shape": list(x_train.shape[1:]),
    "num_classes": y_train.shape[-1],
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
    "batch_size": 32
}

json_filepath = model_folder.joinpath(f"params_{model_version}.json")
with open(json_filepath, 'w') as json_file:
    json.dump(params, json_file, indent=4)
print("Here")
