import os
import itertools
import string

def rename_files_in_subdirs(root_dir):
    # Get all subdirectories in the specified root directory
    for subdir, dirs, files in os.walk(root_dir):
        if dirs:
            continue  # Skip processing if the current directory has subdirectories

        # Generate two-letter combinations for renaming
        two_letter_combinations = itertools.product(string.ascii_lowercase, repeat=2)
        for file, suffix in zip(files, two_letter_combinations):
            # Generate new file name
            new_name = f"{os.path.basename(subdir)}_{suffix[0]}{suffix[1]}{os.path.splitext(file)[1]}"

            # Get current file path and new file path
            current_file_path = os.path.join(subdir, file)
            new_file_path = os.path.join(subdir, new_name)

            # Rename the file
            os.rename(current_file_path, new_file_path)
            print(f"Renamed {current_file_path} to {new_file_path}")

# Usage example: Specify the path to the directory you want to process
if __name__ == "__main__":
    directory_path = "data/test"
    rename_files_in_subdirs(directory_path)
