import os
import cv2
import time
import logging
import numpy as np

# Setup logging
logging.basicConfig(filename='data/processing_log.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to process a single image pair
def process_image(hist_img_path, orig_img_path, binary_output_path, merged_output_path, rgba_output_path):
    # Load the histogram-matched image for edge detection
    hist_image = cv2.imread(hist_img_path)
    orig_image = cv2.imread(orig_img_path)
    if hist_image is None or orig_image is None:
        logging.error(f"Failed to load image at {hist_img_path} or {orig_img_path}")
        return

    # Edge detection on histogram-matched image
    gray = cv2.cvtColor(hist_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 160)
    
    # Find contours and draw on the original image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_image = orig_image.copy()
    cv2.drawContours(merged_image, contours, -1, (0, 255, 0), 2)

    # Process edge data for RGBA and binary output
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_normalized = (edges > 0).astype(np.uint8) * 255
    rgba_image = cv2.merge((orig_image[:, :, 0], orig_image[:, :, 1], orig_image[:, :, 2], edges_normalized))

    # Save the processed images
    cv2.imwrite(os.path.join(binary_output_path, os.path.basename(hist_img_path)), edges_colored)
    cv2.imwrite(os.path.join(merged_output_path, os.path.basename(orig_img_path)), merged_image)
    cv2.imwrite(os.path.join(rgba_output_path, os.path.splitext(os.path.basename(orig_img_path))[0] + '.png'), rgba_image)

# Base directories for datasets and outputs
base_hist_input_dir = "data/background_removed_histogram_matching"
base_orig_input_dir = "data/background_removed"
base_binary_output_dir = "data/binary_output_test"
base_merged_output_dir = "data/merged_output_test"
base_rgba_output_dir = "data/rgba_output_test"

# Ensure output directories exist
os.makedirs(base_binary_output_dir, exist_ok=True)
os.makedirs(base_merged_output_dir, exist_ok=True)
os.makedirs(base_rgba_output_dir, exist_ok=True)

# Traverse the histogram-matched dataset directory
logging.info("Processing images...")
image_times = []
for root, dirs, files in os.walk(base_hist_input_dir):
    for file in files:
        if file.lower().endswith(".jpg"):
            hist_img_path = os.path.join(root, file)
            orig_img_path = os.path.join(base_orig_input_dir, os.path.relpath(hist_img_path, base_hist_input_dir))  # Corresponding original image
            relative_path = os.path.relpath(root, base_hist_input_dir)
            binary_output_path = os.path.join(base_binary_output_dir, relative_path)
            merged_output_path = os.path.join(base_merged_output_dir, relative_path)
            rgba_output_path = os.path.join(base_rgba_output_dir, relative_path)

            # Ensure output subdirectories exist
            os.makedirs(binary_output_path, exist_ok=True)
            os.makedirs(merged_output_path, exist_ok=True)
            os.makedirs(rgba_output_path, exist_ok=True)

            # Log and process the image pair
            start_time = time.time()
            process_image(hist_img_path, orig_img_path, binary_output_path, merged_output_path, rgba_output_path)
            elapsed_time = time.time() - start_time
            image_times.append(elapsed_time)

# Log performance
avg_time = sum(image_times) / len(image_times) if image_times else 0
logging.info("Average time taken per image for {} total images: {:.3f} seconds".format(len(image_times), avg_time))
total_time = sum(image_times)
logging.info("Total time taken: {:.3f} seconds".format(total_time))
logging.info("All images have been processed.")
print("All images have been processed.")
