import cv2
import matplotlib.pyplot as plt
import rasterio
import os
import numpy as np
from pathlib import Path




# Directory containing the raster images
model_version = 'ver7_Segmod'



amazonas_root_folder = Path("/mnt/hddarchive.nfs/amazonas_dir")
output_folder = amazonas_root_folder.joinpath('output')
detection_folder = output_folder.joinpath('ai_detection')
detection_folder_aiversion = detection_folder.joinpath(f'{model_version}')



raster_dir = '/mnt/hddarchive.nfs/amazonas_dir/output/ai_detection/ver7_Segmod/reclassified/sieved'
input_rasters = []
raster_files = os.listdir(raster_dir)
for file in sorted(raster_files):
    if "_CLASS" in file:
        input_rasters.append(Path(raster_dir).joinpath(file))

# Video output settings
video_name = 'output_video5_reclassified.avi'
fps = 1  # Frames per second

# Initialize video writer with H.264 codec
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = None
for file in input_rasters:
    # Extract date from filename
    date = file.name.split('_')[1]

    # Read the raster file
    with rasterio.open(os.path.join(raster_dir, file)) as src:
        raster = src.read(1)

    # Normalize and scale the raster data for visualization
    raster = raster.astype('float32')
    raster -= raster.min()
    raster /= raster.max()
    raster *= 255.0
    raster = raster.astype('uint8')

    # Convert to a format compatible with OpenCV
    raster_rgb = cv2.cvtColor(raster, cv2.COLOR_GRAY2RGB)

    # Create a matplotlib plot with adjusted layout
    fig, ax = plt.subplots()
    ax.imshow(raster_rgb)
    ax.set_title(date, fontsize=10)  # Adjust font size as needed
    ax.axis('off')

    # Adjust the layout to remove white space
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Save the plot to a temporary file
    frame_filename = 'temp_frame.png'
    plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Read the image using OpenCV
    frame = cv2.imread(frame_filename)

    # Initialize video writer with the frame size from the first image
    if video is None:
        frame_size = (frame.shape[1], frame.shape[0])
        video = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

    video.write(frame)

    # Remove the temporary file
    os.remove(frame_filename)

# Release the video writer
video.release()