import os
import shutil

# Define the paths
first_dir = '../data/test_dataset_2'
second_dir = '../data/crop_output'
output_dir = '../data/crop_output_subset'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List subdirectories in the first directory
subdirs_in_first = {name for name in os.listdir(first_dir) if os.path.isdir(os.path.join(first_dir, name))}

# Filter and copy subdirectories from the second directory
for subdir in os.listdir(second_dir):
    if subdir in subdirs_in_first:
        full_subdir_path = os.path.join(second_dir, subdir)
        if os.path.isdir(full_subdir_path):
            shutil.copytree(full_subdir_path, os.path.join(output_dir, subdir))

print("Copying complete.")
