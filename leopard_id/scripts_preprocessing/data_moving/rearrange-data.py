# author: David Colomer Matachana
# GitHub username: acse-dc1823

"""
This script is only relevant for the NCF, ignore it otherwise.
processes a dataset from the NCF, which is organized by location, year, flank, etc.,
and consolidates it into a simpler format.
It combines IDs from different years into a unified ID format (e.g., converting `MML2016-01M` to
`MML01_left` or `MML01_right`).

The script performs the following steps:

1. Defines the source directory as the current working directory and creates a destination
   directory called `joined_dataset` within the source directory.
2. Contains utility functions:
   - `valid_directory(path)`: Checks if the given path is a valid directory and
      not a system file like `.DS_Store`.
   - `extract_id(id_dir)`: Extracts and normalizes the ID in the desired format by
      taking the first three characters as the prefix and the digits after the first hyphen.
3. Iterates through the directories and subdirectories in the source directory:
   - Skips the destination directory `joined_dataset`.
   - For each location and year, processes the `All_flanks_uncropped` subdirectory.
   - For each ID directory, normalizes the ID and processes the left and right flank subdirectories.
   - Copies `.jpg` images from the original directories to the new destination directories,
     organized by the normalized ID and flank side.

Finally, the script prints a confirmation message once all images have been reorganized.
"""

import os
import shutil

src_dir = os.path.join(
    os.getcwd(), "Data"
)  # This sets the source directory to the current working directory
dest_dir = os.path.join(src_dir, "joined_dataset")  # Path to the destination directory

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)


# Function to check if the path is a directory and not a system file like .DS_Store
def valid_directory(path):
    return os.path.isdir(path) and not path.endswith(".DS_Store")


# Function to extract ID in the desired format
def extract_id(id_dir):
    prefix = id_dir[:3]  # First three characters
    number = "".join(filter(str.isdigit, id_dir.split("-")[1]))  # Digits after the first hyphen
    return (
        prefix + number
    )  # Concatenate prefix and number without any characters after the underscore


# Iterate through the directories and subdirectories
for location in os.listdir(src_dir):
    print(f"Processing {location}...")
    location_path = os.path.join(src_dir, location)
    if (
        valid_directory(location_path) and location != "joined_dataset"
    ):  # Skip the destination directory
        for year in os.listdir(location_path):
            year_path = os.path.join(location_path, year)
            all_flanks_path = os.path.join(year_path, "All_flanks_uncropped")
            if valid_directory(all_flanks_path):
                for id_dir in os.listdir(all_flanks_path):
                    if valid_directory(os.path.join(all_flanks_path, id_dir)):
                        normalized_id = extract_id(id_dir)
                        id_path = os.path.join(all_flanks_path, id_dir)
                        for side in os.listdir(id_path):
                            side_path = os.path.join(id_path, side)
                            if valid_directory(side_path):
                                for image in os.listdir(side_path):
                                    if image.lower().endswith(".jpg"):  # Process only .jpg images
                                        source_image_path = os.path.join(side_path, image)
                                        dest_folder = f"{normalized_id}_{side}"
                                        dest_folder_path = os.path.join(dest_dir, dest_folder)
                                        if not os.path.exists(dest_folder_path):
                                            os.makedirs(dest_folder_path)
                                        # Copy the image to the new directory
                                        dest_image_path = os.path.join(dest_folder_path, image)
                                        shutil.copy2(source_image_path, dest_image_path)
            else:
                if ".DS_Store" not in all_flanks_path:
                    print(f"invalid path: {all_flanks_path}")

print("All images have been reorganized.")
