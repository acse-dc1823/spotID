"""
This script iterates through the given directory in base_input_dir,
through all the subdirectories,  and creates bounding boxes using
PytorchWildlife (based on YOLO) around the animal. Both the cropped
image and the full image with the bounding box around the animal will
be saved in different directories in the same root.
"""

import numpy as np
import os
from PIL import Image
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife import utils as pw_utils

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)

# Base directories for datasets and outputs
base_input_dir = "Data/minimum_usable_dataset"
base_full_output_dir = "Data/full_output"
base_crop_output_dir = "Data/crop_output"

# Ensure output directories exist
os.makedirs(base_full_output_dir, exist_ok=True)
os.makedirs(base_crop_output_dir, exist_ok=True)

# Image transformation
transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                               stride=detection_model.STRIDE)


# Function to process a single image
def process_image(img_path, full_output_path, crop_output_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    results = detection_model.single_image_detection(transform(img), img.shape,
                                                     img_path, conf_thres=0.6)
    pw_utils.save_detection_images(results, full_output_path)
    results_list = [results]
    pw_utils.save_crop_images(results_list, crop_output_path)


# Traverse the dataset directory
for root, dirs, files in os.walk(base_input_dir):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, base_input_dir)
            full_output_path = os.path.join(base_full_output_dir, relative_path)
            crop_output_path = os.path.join(base_crop_output_dir, relative_path)
            
            # Ensure output subdirectories exist
            os.makedirs(full_output_path, exist_ok=True)
            os.makedirs(crop_output_path, exist_ok=True)

            # Process the image
            process_image(img_path, full_output_path, crop_output_path)
            print(f"Processed {img_path}")

print("All images have been processed.")
