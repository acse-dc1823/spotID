# author: David Colomer Matachana
# GitHub username: acse-dc1823

import os
import cv2
import time
import logging

# Setup logging
logging.basicConfig(
    filename="../data/processing_log.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def process_image(image_path, binary_output_path, merged_output_path=None):
    # Load the image for edge detection
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image at {image_path}")
        return None

    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 55, 160)

    # Find contours and draw on a copy of the original image for visualization
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if merged_output_path is not None:
        merged_image = image.copy()
        cv2.drawContours(merged_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(merged_output_path, merged_image)

    # Save the processed images
    cv2.imwrite(binary_output_path, edges)

    return True


def edge_detection(input_dir, binary_output_dir, merged_output_dir=None):
    print("start edge detection")
    # Ensure output directories exist
    os.makedirs(binary_output_dir, exist_ok=True)
    if merged_output_dir is not None:
        os.makedirs(merged_output_dir, exist_ok=True)

    # Get list of already processed images
    processed_images = set()
    for root, _, files in os.walk(binary_output_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                relative_path = os.path.relpath(root, binary_output_dir)
                processed_images.add(os.path.join(relative_path, file))

    # Traverse the dataset directory
    logging.info("Processing new images for edge detection...")
    image_times = []
    new_images_processed = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                relative_path = os.path.relpath(root, input_dir)
                img_relative_path = os.path.join(relative_path, file)

                # Check if the image has already been processed
                if img_relative_path in processed_images:
                    print(f"Image {img_relative_path} has already been processed, skipping...")
                    continue
                else:
                    print(f"Processing image {img_relative_path}...")

                new_images_processed += 1

                image_path = os.path.join(root, file)
                binary_output_path = os.path.join(binary_output_dir, relative_path, file)
                if merged_output_dir is not None:
                    merged_output_path = os.path.join(merged_output_dir, relative_path, file)
                    os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)
                else:
                    merged_output_path = None

                # Ensure output subdirectories exist
                os.makedirs(os.path.dirname(binary_output_path), exist_ok=True)

                # Log and process the image
                start_time = time.time()
                result = process_image(image_path, binary_output_path, merged_output_path)
                if result:
                    elapsed_time = time.time() - start_time
                    image_times.append(elapsed_time)

    # Log performance
    if image_times:
        avg_time = sum(image_times) / len(image_times)
        logging.info(f"Average time taken per new image for edge detection: {avg_time:.3f} seconds")
        total_time = sum(image_times)
        logging.info(f"Total time taken for new images (edge detection): {total_time:.3f} seconds")

    logging.info(f"Number of new images processed for edge detection: {new_images_processed}")
    print(
        f"All new images have been processed for edge detection. Total new images: {new_images_processed}"
    )


# Base directories for dataset and outputs
# input_dir = "../data/bg_rem_2"
# binary_output_dir = "../data/edge_detection_output"

# edge_detection(input_dir, binary_output_dir)
