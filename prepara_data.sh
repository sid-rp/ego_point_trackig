#!/bin/bash

# Define paths
source_folder="/scratch/projects/fouheylab/shared_datasets/epic-kitchens/og/"     # Replace with the actual path to rgb_frames
destination_folder="/scratch/projects/fouheylab/shared_datasets/epic_tracking_dataset/data/preprocessed_data/"  # Replace with the actual path to preprocessed_data
file_list="/scratch/projects/fouheylab/dma9300/OSNOM/video_ids.txt"     # Replace with the path to the text file that contains folder names

# Ensure the destination folder exists
mkdir -p "$destination_folder"

# Read folder names from the .txt file and process each
while IFS= read -r folder; do
    # Extract the PXX part from the folder name
    subfolder=$(echo "$folder" | cut -d'_' -f1)

    # Construct the path to the .tar file
    tar_file="$source_folder/$subfolder/rgb_frames/$folder.tar"

    # Check if the .tar file exists
    if [[ -f "$tar_file" ]]; then
        # Create an absolute symbolic link in the destination folder
        ln -s "$tar_file" "$destination_folder/$folder.tar"
        echo "Created symlink for $tar_file in $destination_folder"
    else
        echo "File $tar_file does not exist"
    fi
done < "$file_list"