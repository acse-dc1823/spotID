import os
import cv2
import time
import logging
import numpy as np

# Setup logging
logging.basicConfig(filename='data/processing_log.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def process_image(image_path, binary_output_path, merged_output_path):
    # Load the image for edge detection
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image at {image_path}")
        return

    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 160)

    # Find contours and draw on a copy of the original image for visualization
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_image = image.copy()
    cv2.drawContours(merged_image, contours, -1, (0, 255, 0), 2)

    # Save the processed images
    cv2.imwrite(binary_output_path, edges)
    cv2.imwrite(merged_output_path, merged_image)

# Base directories for dataset and outputs
input_dir = "data/leopard_ids_background_removed_histogram_matching"
binary_output_dir = "data/leopard_ids_binary_output"
merged_output_dir = "data/leopard_ids_merged_output"

# Ensure output directories exist
os.makedirs(binary_output_dir, exist_ok=True)
os.makedirs(merged_output_dir, exist_ok=True)

# Traverse the dataset directory
logging.info("Processing images...")
image_times = []
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".jpg"):
            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dir)
            binary_output_path = os.path.join(binary_output_dir, relative_path, file)
            merged_output_path = os.path.join(merged_output_dir, relative_path, file)

            # Ensure output subdirectories exist
            os.makedirs(os.path.dirname(binary_output_path), exist_ok=True)
            os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)

            # Log and process the image
            start_time = time.time()
            process_image(image_path, binary_output_path, merged_output_path)
            elapsed_time = time.time() - start_time
            image_times.append(elapsed_time)

# Log performance
avg_time = sum(image_times) / len(image_times) if image_times else 0
logging.info(f"Average time taken per image for {len(image_times)} total images: {avg_time:.3f} seconds")
total_time = sum(image_times)
logging.info(f"Total time taken: {total_time:.3f} seconds")
logging.info("All images have been processed.")
print("All images have been processed.")
