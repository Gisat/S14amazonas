import onnx
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx
from cnn_architectures import build_vgg16_segmentation_bn
import zipfile
from pathlib import Path
import subprocess

def copy_to_s3(project_name, local_path, config_path, s3_path=""):
    cmd = [
        'rclone',
        'copy',
        '--config', config_path,
        "--log-level=INFO",
        "--no-gzip-encoding",
        str(local_path),
        f'{project_name}:{s3_path}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error executing rclone: {result.stderr}")
    else:
        print(f"Upload {local_path} completed!")

# Load the HDF5 model
h5_file_path = '/mnt/hddarchive.nfs/amazonas_dir/model/model_best_build_vgg16_segmentation_batchingestion_labelmorethan120dataset_weighted_f1score.h5'  # Replace with your HDF5 model path
rclone_config_path = "/home/eouser/userdoc/rclone.conf"
# Step 2: Rebuild the model architecture
model = build_vgg16_segmentation_bn((256, 256, 15))
model.load_weights(h5_file_path)

# Convert the Keras model to ONNX
spec = (tf.TensorSpec((None,) + model.input.shape[1:], tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model
onnx_model_path = '/mnt/hddarchive.nfs/amazonas_dir/onnxmodel/amazonas_ai_cnn.onnx'  # Replace with your desired ONNX file path
# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

onnx_zip_local_path = '/mnt/hddarchive.nfs/amazonas_dir/onnxmodel/amazonas_ai_cnn.zip'
with zipfile.ZipFile(onnx_zip_local_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(onnx_model_path, "ml_model.onnx")

s3_path = Path("amazonas").joinpath("ml_models")
copy_to_s3("gisat", onnx_zip_local_path, rclone_config_path, s3_path=s3_path)





print(f"Model successfully converted to ONNX and saved at {onnx_model_path}")
