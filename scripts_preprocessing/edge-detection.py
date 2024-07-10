import os
import cv2
import time
import logging

# Setup logging
logging.basicConfig(filename='../data/processing_log.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Base directories for datasets and outputs
base_input_dir = "../data/background_removed"
base_binary_output_dir = "../data/binary_output"
base_merged_output_dir = "../data/merged_output"

# Ensure output directories exist
os.makedirs(base_binary_output_dir, exist_ok=True)
os.makedirs(base_merged_output_dir, exist_ok=True)


# Function to process a single image
def process_image(img_path, binary_output_path, merged_output_path):
    # Load the image
    output_image = cv2.imread(img_path)
    if output_image is None:
        logging.error(f"Failed to load image at {img_path}")
        return

    # Convert the entire image to grayscale
    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to highlight edges
    edges = cv2.Canny(blurred, 55, 160)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image to highlight features
    merged_image = output_image.copy()
    cv2.drawContours(merged_image, contours, -1, (0, 255, 0), 2)

    # Convert edges to a 3-channel image for saving
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Save the edge-detected binary image and merged image
    cv2.imwrite(os.path.join(binary_output_path, os.path.basename(img_path)), edges_colored)
    cv2.imwrite(os.path.join(merged_output_path, os.path.basename(img_path)), merged_image)


# Traverse the dataset directory
logging.info("Processing images...")
image_times = [] 
for root, dirs, files in os.walk(base_input_dir):
    for file in files:
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, base_input_dir)
            binary_output_path = os.path.join(base_binary_output_dir, relative_path)
            merged_output_path = os.path.join(base_merged_output_dir, relative_path)
            # Ensure output subdirectories exist
            os.makedirs(binary_output_path, exist_ok=True)
            os.makedirs(merged_output_path, exist_ok=True)

            # Log and process the image
            start_time = time.time()
            process_image(img_path, binary_output_path, merged_output_path)
            elapsed_time = time.time() - start_time
            image_times.append(elapsed_time)
logging.info("Average time taken per image for {} total images: {:.3f} seconds".format(len(image_times), sum(image_times) / len(image_times)))
logging.info("Total time taken: {:.3f} seconds".format(sum(image_times)))
logging.info("All images have been processed.")
print("All images have been processed.")
