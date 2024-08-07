"""
This script processes images in the specified base_input_dir to create bounding boxes 
around animals using PytorchWildlife (based on YOLO). It saves both cropped images 
and full images with bounding boxes in separate directories.

Steps:

1. **Setup:**
   - Configures logging and device settings (CUDA if available).
   - Initializes the MegaDetectorV5 model from PytorchWildlife.
   - Defines base directories for input and output data, creating them if they don't exist.
   - Sets up image transformation parameters.

2. **Functions:**
   - `process_image(img_path, full_output_path, crop_output_path)`:
   Processes a single image to detect animals, save the full image with bounding boxes,
   and save cropped images of detected animals. Logs processing time for each image.

3. **Main Loop:**
   - Iterates through all subdirectories and images in base_input_dir.
   - Processes each image, saving results in the appropriate output directories.
   - Logs the processing time for each directory and calculates average times for images and directories.

Finally, the script logs the total processing time and prints a confirmation message upon completion.
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


# Function to process a single image
def process_image(img_path, full_output_path, crop_output_path,
                  transform, detection_model, store_full_images=False):
    start_time = time.time()  # Start time for processing this image

    img = np.array(Image.open(img_path).convert("RGB"))
    results = detection_model.single_image_detection(transform(img), img.shape,
                                                     img_path, conf_thres=0.6)
    if store_full_images:
        pw_utils.save_detection_images(results, full_output_path)
    results_list = [results]
    pw_utils.save_crop_images(results_list, crop_output_path)

    # Calculate time taken and log it
    end_time = time.time()
    return end_time - start_time  # Return processing time for this image
def is_image_processed(filename, output_dir):
    for root, _, files in os.walk(output_dir):
        if any(filename in f for f in files):
            return True
    return False

def get_unprocessed_images(input_dir, output_dir):
    unprocessed = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jpg"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                if not is_image_processed(file, os.path.join(output_dir, relative_path)):
                    print(f"Processing image {os.path.join(output_dir, relative_path, file)}...")
                    unprocessed.append((input_path, relative_path))
                else:
                    print(f"Image {os.path.join(output_dir, relative_path, file)} has already been processed for background removal, skipping...")
    return unprocessed

def crop_images_folder(base_input_dir,
                       base_crop_output_dir, base_full_output_dir="",
                       store_full_images=False):
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)

    # Ensure output directories exist
    if store_full_images:
        os.makedirs(base_full_output_dir, exist_ok=True)
    os.makedirs(base_crop_output_dir, exist_ok=True)

    # Image transformation
    transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                   stride=detection_model.STRIDE)

    # Start timing for the entire processing
    start_total_time = time.time()
    image_times = []  # Store times for each image to calculate average
    dir_times = {}

    # Get list of unprocessed images
    unprocessed_images = get_unprocessed_images(base_input_dir, base_crop_output_dir)
    
    logging.info(f"Found {len(unprocessed_images)} unprocessed images.")

    # Process unprocessed images
    for img_path, relative_path in unprocessed_images:
        full_output_path = os.path.join(base_full_output_dir, relative_path)
        crop_output_path = os.path.join(base_crop_output_dir, relative_path)

        # Ensure output subdirectories exist
        if store_full_images:
            os.makedirs(full_output_path, exist_ok=True)
        os.makedirs(crop_output_path, exist_ok=True)

        # Process the image and log time
        image_time = process_image(img_path, full_output_path, crop_output_path,
                                   transform, detection_model, store_full_images)
        image_times.append(image_time)

        # Update directory times
        if relative_path not in dir_times:
            dir_times[relative_path] = 0
        dir_times[relative_path] += image_time

    # Log statistics
    if image_times:
        average_time_per_image = sum(image_times) / len(image_times)
        logging.info(f"Average time per image: {average_time_per_image:.2f} seconds")

    for dir_path, dir_time in dir_times.items():
        logging.info(f"Total time for directory {dir_path}: {dir_time:.2f} seconds")

    end_total_time = time.time()
    total_processing_time = end_total_time - start_total_time
    logging.info(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Processed {len(unprocessed_images)} new images.")

# base_input_dir = "../data/inference_images"
# base_full_output_dir = "../data/full_output_test_2"
# base_crop_output_dir = "../data/crop_output_test_2"
# crop_images_folder(base_input_dir, base_crop_output_dir,
#                    store_full_images=False)
