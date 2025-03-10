#!/bin/bash

# Define paths
base_folder="/scratch/projects/fouheylab/shared_datasets/epic-kitchens/og/frames_rgb_flow/rgb"  # Source folder path
destination_folder="/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/preprocessed_data/"  # Replace with the actual path to preprocessed_data
file_list="/scratch/projects/fouheylab/dma9300/OSNOM/video_ids.txt"  # Replace with the path to the text file that contains folder names

# Ensure the destination folder exists
mkdir -p "$destination_folder"

# Loop through the list of folders in the text file
while IFS= read -r folder; do
    # Extract the PXX part from the folder name
    subfolder=$(echo "$folder" | cut -d'_' -f1)

    # Construct the paths to check in 'train' and 'test' subfolders
    train_path="$base_folder/train/$subfolder/$folder.tar"
    test_path="$base_folder/test/$subfolder/$folder.tar"

    # Check if the .tar file exists in either 'train' or 'test'
    if [[ -f "$train_path" ]]; then
        # Create an absolute symbolic link in the destination folder
        ln -s "$train_path" "$destination_folder/$folder.tar"
        echo "Created symlink for $train_path in $destination_folder"
    elif [[ -f "$test_path" ]]; then
        # Create an absolute symbolic link in the destination folder
        ln -s "$test_path" "$destination_folder/$folder.tar"
        echo "Created symlink for $test_path in $destination_folder"
    else
        echo "File for $folder not found in either 'train' or 'test' subfolders"
    fi
done < "$file_list"
