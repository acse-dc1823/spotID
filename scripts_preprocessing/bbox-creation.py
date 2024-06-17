"""
This script iterates through the given directory in base_input_dir,
through all the subdirectories,  and creates bounding boxes using
PytorchWildlife (based on YOLO) around the animal. Both the cropped
image and the full image with the bounding box around the animal will
be saved in different directories in the same root.
"""

import logging
import time
import numpy as np
import os
from PIL import Image
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife import utils as pw_utils

logging.basicConfig(filename='../data/processing_log.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)

# Base directories for datasets and outputs
base_input_dir = "../data/joined_dataset"
base_full_output_dir = "../data/full_output"
base_crop_output_dir = "../data/crop_output"

# Ensure output directories exist
os.makedirs(base_full_output_dir, exist_ok=True)
os.makedirs(base_crop_output_dir, exist_ok=True)

# Image transformation
transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                               stride=detection_model.STRIDE)


# Start timing for the entire processing
start_total_time = time.time()

# Function to process a single image
def process_image(img_path, full_output_path, crop_output_path):
    start_time = time.time()  # Start time for processing this image
    
    img = np.array(Image.open(img_path).convert("RGB"))
    results = detection_model.single_image_detection(transform(img), img.shape,
                                                     img_path, conf_thres=0.6)
    pw_utils.save_detection_images(results, full_output_path)
    results_list = [results]
    pw_utils.save_crop_images(results_list, crop_output_path)

    # Calculate time taken and log it
    end_time = time.time()
    return end_time - start_time  # Return processing time for this image

image_times = []  # Store times for each image to calculate average
dir_times = []
# Traverse the dataset directory
for root, dirs, files in os.walk(base_input_dir):
    dir_start_time = time.time()  # Time tracking for this directory
    
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, base_input_dir)
            full_output_path = os.path.join(base_full_output_dir, relative_path)
            crop_output_path = os.path.join(base_crop_output_dir, relative_path)
            
            # Ensure output subdirectories exist
            os.makedirs(full_output_path, exist_ok=True)
            os.makedirs(crop_output_path, exist_ok=True)

            # Process the image and log time
            image_time = process_image(img_path, full_output_path, crop_output_path)
            image_times.append(image_time)
    # Log directory processing time
    dir_end_time = time.time()
    dir_times.append(dir_end_time-dir_start_time)

# Calculate average time per image and log it
if image_times:
    average_time_per_image = sum(image_times) / len(image_times)
    logging.info(f"Average time per image in whole directory of {len(image_times)} images: {average_time_per_image:.2f} seconds")

if dir_times:
    average_time_per_dir = sum(dir_times) / len(dir_times)
    logging.info(f"Average time per flank in whole directory of {len(dir_times)} flanks: {average_time_per_dir:.2f} seconds")


end_total_time = time.time()
total_processing_time = end_total_time - start_total_time
logging.info(f"Total processing time for all images: {total_processing_time:.2f} seconds")
print("All images have been processed.")
