import os
import shutil
import glob

# Base directories
sparse_dir = "/scratch/projects/fouheylab/shared_datasets/epic-fields"
og_frames_base_dir = "/scratch/projects/fouheylab/shared_datasets/epic-kitchens/og/frames_rgb_flow/rgb"
new_frames_dir = "/mnt/sda/epic-kitchens/new"

# Function to check existence of frames and copy them
def copy_frames(folder_name, images_folder):
    # Check both 'train' and 'test' directories in 'og' path
    for sub_dir in ['train', 'test']:
        og_path = os.path.join(og_frames_base_dir, sub_dir, folder_name[:3], folder_name, "frame_*")
        og_files = glob.glob(og_path)

        # If 'og' files exist, use that path
        if og_files:
            for file in og_files:
                target_file = os.path.join(images_folder, os.path.basename(file))
                if not os.path.exists(target_file):
                    shutil.copy(file, images_folder)
            return

    # Try the 'new' path format
    new_path = os.path.join(new_frames_dir, folder_name[:3], "rgb_frames", folder_name, "frame_*")
    new_files = glob.glob(new_path)

    # If 'new' files exist, use that path
    if new_files:
        for file in new_files:
            target_file = os.path.join(images_folder, os.path.basename(file))
            if not os.path.exists(target_file):
                shutil.copy(file, images_folder)

# Enumerate and process folders
for idx, folder in enumerate(os.listdir(sparse_dir)):
    print(f'{idx} / {len(os.listdir(sparse_dir))}... {folder}')
    folder_path = os.path.join(sparse_dir, folder)
    
    # Skip if not a directory
    if not os.path.isdir(folder_path):
        continue

    # Create 'images' subfolder
    images_folder = os.path.join(folder_path, "images")
    os.makedirs(images_folder, exist_ok=True)

    # Copy frames to 'images' folder
    copy_frames(folder, images_folder)

print("Copying of frames complete.")

