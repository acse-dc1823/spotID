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
                if store_full_images:
                    os.makedirs(full_output_path, exist_ok=True)
                os.makedirs(crop_output_path, exist_ok=True)

                # Process the image and log time
                image_time = process_image(img_path, full_output_path, crop_output_path,
                                           transform, detection_model, store_full_images)
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


# base_input_dir = "../data/inference_images"
# base_full_output_dir = "../data/full_output_test_2"
# base_crop_output_dir = "../data/crop_output_test_2"
# crop_images_folder(base_input_dir, base_crop_output_dir,
#                    store_full_images=False)
