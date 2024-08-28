"""
Image Processing Script for Contour Detection and Edge Highlighting

This script processes a subset of images from a specified directory, applies contour detection and
edge highlighting, and saves the processed images into designated output directories. The script
performs the following tasks:

1. Sets up logging to record processing details and errors.
2. Ensures that the output directories for binary edge-detected images and merged
   contour-highlighted images exist.
3. Defines a function `process_image` that:
   - Loads an image from the specified path.
   - Converts the image to grayscale and applies Gaussian blur to reduce noise.
   - Uses Canny edge detection to highlight the edges in the image.
   - Finds contours in the edge-detected image and draws them on the original image.
   - Saves the edge-detected binary image and the merged image with highlighted contours to the
     respective output paths.
4. Traverses the input dataset directory to process each image file:
   - For each image, determines the output subdirectories and ensures their existence.
   - Logs the start time, processes the image using the `process_image` function, and records the
     elapsed time for each image.
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
- The processed images will be saved in the respective output directories with binary edge
  detection and merged contour highlights.
"""

import logging
import time
import os
from rembg import remove
from skimage import io
from skimage.exposure import match_histograms
import cv2

# Setup logging
logging.basicConfig(
    filename="../data/processing_log.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


# Function to remove background from a single image
def remove_background(img_path, output_path, reference_image):
    start_time = time.time()  # Start time for processing this image

    # Read the image
    target_image = io.imread(img_path)
    if target_image is None:
        logging.error(f"Failed to load image at {img_path}")
        return None
    matched_image = match_histograms(target_image, reference_image, channel_axis=-1)

    matched_image = cv2.cvtColor(matched_image, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", matched_image)
    if not is_success:
        logging.error(f"Failed to encode image at {img_path}")
        return None

    # Remove the background
    output_image = remove(buffer.tobytes())

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the result
    with open(output_path, "wb") as out:
        out.write(output_image)

    # Calculate time taken and log it
    end_time = time.time()
    return end_time - start_time  # Return processing time for this image


def remove_background_processor(base_input_dir, base_output_dir, reference_path_for_matching=None):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Base directories for datasets and outputs
    if reference_path_for_matching is None:
        reference_path_for_matching = os.path.join(
            script_dir, "../", "data", "histogram_matching", "0_0_BG-01A-2019-02-05_05-10-44.jpg"
        )

    reference_image = io.imread(reference_path_for_matching)
    if reference_image is None:
        logging.error(f"Failed to load reference image at {reference_path_for_matching}, exiting.")
        return

    # Ensure input directory is valid
    if not os.path.isdir(base_input_dir):
        logging.error(f"Invalid input directory: {base_input_dir}")
        return
    else:
        logging.info(f"Input directory: {base_input_dir}")

    # Ensure output directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    # Get list of already processed images
    processed_images = set()
    for root, _, files in os.walk(base_output_dir):
        for file in files:
            if file.endswith(".jpg"):
                relative_path = os.path.relpath(root, base_output_dir)
                processed_images.add(os.path.join(relative_path, file))

    # Start timing for the entire processing
    start_total_time = time.time()
    image_times = []  # Store times for each image to calculate average
    new_images_processed = 0

    # Traverse the dataset directory
    for root, _, files in os.walk(base_input_dir):
        for file in files:
            if file.endswith(".jpg"):
                relative_path = os.path.relpath(root, base_input_dir)
                img_relative_path = os.path.join(relative_path, file)

                # Check if the image has already been processed
                if img_relative_path in processed_images:
                    print(f"Image {img_relative_path} has already been processed, skipping...")
                    continue
                else:
                    print(f"Processing image {img_relative_path}...")

                new_images_processed += 1

                img_path = os.path.join(root, file)
                output_dir = os.path.join(base_output_dir, relative_path)
                output_path = os.path.join(output_dir, file)

                # Ensure output subdirectories exist
                os.makedirs(output_dir, exist_ok=True)

                # Process the image and log time
                image_time = remove_background(img_path, output_path, reference_image)
                if image_time is not None:
                    image_times.append(image_time)

    # Calculate average time per image and log it
    if image_times:
        average_time_per_image = sum(image_times) / len(image_times)
        logging.info(
            f"Average time per new image for background removal: {average_time_per_image:.2f} seconds"
        )

    end_total_time = time.time()
    total_processing_time = end_total_time - start_total_time
    logging.info(
        f"Total processing time for new images (background removal): {total_processing_time:.2f} seconds"
    )
    logging.info(f"Number of new images processed for background removal: {new_images_processed}")
    print(
        f"All new images have been processed for background removal. Total new images: {new_images_processed}"
    )


# base_input_dir = "../data/crop_output_small"
# base_output_dir = "../data/bg_rem_2"
# remove_background_processor(base_input_dir, base_output_dir)
