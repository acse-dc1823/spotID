# author: David Colomer Matachana
# GitHub username: acse-dc1823

"""
Script to run all of the preprocessing steps for the image dataset to prepare
the dataset for training. This script will take your directory of raw
leopard images, and will crop, remove the background, and perform edge
detection on them. For training, remember that each individual leopard should
be in their own subdirectory. Hence, the input directory should contain
subdirectories for each individual leopard, and each subdirectory should
contain images of that leopard. Use the config_preprocessing.json file to
specify the unprocessed image folder, crop output folder, background removed
output folder, and binary output folder.

For training, you will need the crop output folder and the binary output
folder, you can delete the bg_rem if you want to save space.
"""


import os
import json
import sys
from bbox_creation import crop_images_folder
from background_removal import remove_background_processor
from edge_detection import edge_detection

project_root = os.path.dirname(os.path.abspath(__file__))

def run_preprocessing(config_path):
    config_path = os.path.abspath(config_path)

    with open(config_path, "r") as file:
        config = json.load(file)

    # Convert relative paths to absolute paths based on script location
    base_input_dir = os.path.abspath(
        os.path.join(project_root, config["unprocessed_image_folder"])
    )
    base_crop_output_dir = os.path.abspath(
        os.path.join(project_root, config["crop_output_folder"])
    )
    base_bg_removed_output_dir = os.path.abspath(
        os.path.join(project_root, config["bg_removed_output_folder"])
    )
    base_binary_output_dir = os.path.abspath(
        os.path.join(project_root, config["base_binary_output_folder"])
    )

    # Create output folders if they don't exist
    os.makedirs(base_crop_output_dir, exist_ok=True)
    os.makedirs(base_bg_removed_output_dir, exist_ok=True)
    os.makedirs(base_binary_output_dir, exist_ok=True)

    if config["preprocess"]:
        crop_images_folder(
            base_input_dir, base_crop_output_dir, store_full_images=False
        )
        remove_background_processor(
            base_crop_output_dir, base_bg_removed_output_dir
        )
        edge_detection(base_bg_removed_output_dir, base_binary_output_dir)


if __name__ == "__main__":
    config_path = os.path.join(project_root, "../config_inference.json")
    run_preprocessing(config_path)
