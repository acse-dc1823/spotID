"""
Image Processing Script for Contour Detection and Edge Highlighting

This script processes a subset of images from a specified directory, applies contour detection and edge highlighting,
and saves the processed images into designated output directories. The script performs the following tasks:

1. Sets up logging to record processing details and errors.
2. Ensures that the output directories for binary edge-detected images and merged contour-highlighted images exist.
3. Defines a function `process_image` that:
   - Loads an image from the specified path.
   - Converts the image to grayscale and applies Gaussian blur to reduce noise.
   - Uses Canny edge detection to highlight the edges in the image.
   - Finds contours in the edge-detected image and draws them on the original image.
   - Saves the edge-detected binary image and the merged image with highlighted contours to the respective output paths.
4. Traverses the input dataset directory to process each image file:
   - For each image, determines the output subdirectories and ensures their existence.
   - Logs the start time, processes the image using the `process_image` function, and records the elapsed time for each image.
5. Logs the average and total processing times for all images.

Directories:
- Input Directory: ../data/background_removed
- Binary Output Directory: ../data/binary_output
- Merged Output Directory: ../data/merged_output

Requirements:
- OpenCV (cv2)
- os and time modules for file handling and performance measurement
- Logging module for recording processing details

Usage:
- Run the script to process all .jpg images in the input directory.
- The processed images will be saved in the respective output directories with binary edge detection and merged contour highlights.
"""

import logging
import time
import os
from rembg import remove

from skimage import io
from skimage.exposure import match_histograms
import cv2

# Setup logging
logging.basicConfig(filename='../data/processing_log.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Base directories for datasets and outputs
base_input_dir = "../data/crop_output"
base_output_dir = "../data/background_removed_histogram_matching"
reference_path_for_matching = "../data/crop_output/BGL01_left/0_0_BG-01A-2019-02-05_05-10-44.jpg"
reference_image = io.imread(reference_path_for_matching)
if reference_image is None:
    logging.error(f"Failed to load reference image at {reference_image}, exiting.")
    exit()

# Ensure input directory is valid
if not os.path.isdir(base_input_dir):
    logging.error(f"Invalid input directory: {base_input_dir}")
    exit()
else:
    logging.info(f"Input directory: {base_input_dir}")

# Ensure output directory exists
os.makedirs(base_output_dir, exist_ok=True)

# Start timing for the entire processing
start_total_time = time.time()


# Function to remove background from a single image
def remove_background(img_path, output_path):
    start_time = time.time()  # Start time for processing this image

    # Read the image
    target_image = io.imread(img_path)
    if target_image is None:
        logging.error(f"Failed to load image at {img_path}")
        return
    matched_image = match_histograms(target_image, reference_image,
                                     channel_axis=-1)

    matched_image = cv2.cvtColor(matched_image, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", matched_image)
    if not is_success:
        logging.error(f"Failed to encode image at {img_path}")
        return

    # Remove the background
    output_image = remove(buffer.tobytes())

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the result
    with open(output_path, 'wb') as out:
        out.write(output_image)

    # Calculate time taken and log it
    end_time = time.time()
    return end_time - start_time  # Return processing time for this image


image_times = []  # Store times for each image to calculate average

# Traverse the dataset directory
for root, dirs, files in os.walk(base_input_dir):
    for file in files:
        if file.endswith(".jpg"):
            img_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, base_input_dir)
            output_dir = os.path.join(base_output_dir, relative_path)

            # Ensure output subdirectories exist
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, file)

            # Process the image and log time
            image_time = remove_background(img_path, output_path)
            image_times.append(image_time)

# Calculate average time per image and log it
if image_times:
    average_time_per_image = sum(image_times) / len(image_times)
    logging.info(f"Average time per image for {len(image_times)} images: {average_time_per_image:.2f} seconds")

end_total_time = time.time()
total_processing_time = end_total_time - start_total_time
logging.info(f"Total processing time for all images: {total_processing_time:.2f} seconds")
logging.info("All images have been processed.")
