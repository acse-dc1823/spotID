# author: David Colomer Matachana
# GitHub username: acse-dc1823

"""
This script splits a dataset into training and testing sets based on a specified test percentage.
The split is done by directories (e.g., by leopard), so the image proportions may vary.
There is a second functionality in which the user specifies the test directories
to be used for the test set as a set, i.e. test_directories_set = {'CUL179_right', 'MML152_Left',
'MML78_Right'} and then split_data(src_directory, test_dirs_set=test_directories_set).

Steps:

1. **Function Definitions:**
   - `count_images(directory)`: Counts all `.jpg` images in a directory and its subdirectories.
   - `copy_directory(src, dest)`: Copies a directory from the source path to the destination path.
   - `split_data(src_directory, test_split=0.2, seed=12)`: Splits the directories into training and
   testing datasets based on the specified test split fraction and random seed.

2. **Main Processing:**
   - Sets the random seed for reproducibility.
   - Defines and creates the `train_dataset` and `test_dataset` directories.
   - Lists, shuffles, and splits the subdirectories from the source directory.
   - Copies the directories to the respective train and test directories.
   - Counts and prints the number of images in each dataset and the percentage in the training set.

Usage:
- Set the source directory path and call `split_data(src_directory, test_split=0.1)` with
the desired test split percentage.
"""

import os
import shutil
import random


def count_images(directory):
    """
    Count all jpg images in a directory and its subdirectories.

    Parameters:
    - directory: The directory to count images in.

    Returns:
    - The total count of jpg images.
    """
    total_images = 0
    for root, dirs, files in os.walk(directory):
        total_images += sum(1 for file in files if file.lower().endswith(".jpg"))
    return total_images


def copy_directory(src, dest):
    """
    Copies a directory from src to dest.

    Parameters:
    - src: Source directory path.
    - dest: Destination directory path.
    """
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == shutil.errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print("Directory not copied. Error: %s" % e)


def split_data(src_directory, test_split=0.2, seed=12, test_dirs_set=None):
    """
    Splits the directories at the source into training and testing datasets by copying them.

    Parameters:
    - src_directory: The source directory containing subdirectories to split.
    - test_split: The fraction of directories to use as the test set.
    - seed: Random seed for reproducibility.
    """
    random.seed(seed)  # Set the seed for random operations
    # Set the paths for the training and testing directories at the parent level of the src_directory
    base_dir = os.path.dirname(src_directory)
    train_dir = os.path.join(base_dir, "train_dataset_rgba")
    test_dir = os.path.join(base_dir, "test_dataset_rgba")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    all_dirs = [
        d for d in os.listdir(src_directory) if os.path.isdir(os.path.join(src_directory, d))
    ]

    if test_dirs_set is not None:
        test_dirs = [d for d in all_dirs if d in test_dirs_set]
        train_dirs = [d for d in all_dirs if d not in test_dirs_set]
    else:
        random.shuffle(all_dirs)
        num_test = int(len(all_dirs) * test_split)
        test_dirs = all_dirs[:num_test]
        train_dirs = all_dirs[num_test:]

    for d in test_dirs:
        copy_directory(os.path.join(src_directory, d), os.path.join(test_dir, d))

    for d in train_dirs:
        copy_directory(os.path.join(src_directory, d), os.path.join(train_dir, d))

    test_images_count = count_images(test_dir)
    train_images_count = count_images(train_dir)

    print(
        f"Copied {len(test_dirs)} directories to the test dataset with a total of {test_images_count} images."
    )
    print(
        f"Copied {len(train_dirs)} directories to the train dataset with a total of {train_images_count} images."
    )
    print(
        f"Percentage of images in training set: {100 * train_images_count / (train_images_count + test_images_count):.2f}%"
    )


# Example usage:
src_directory = os.path.join(os.getcwd(), "data/rgba_output_test")
split_data(src_directory, test_split=0.1)
