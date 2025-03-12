import sys
sys.path.append('./code/tracking_code')
import torch.nn as nn
from util import *
import pickle
import torch
from tqdm import tqdm
import argparse
import json
import os
import cv2
from PIL import Image
import tarfile
import cv2
import numpy as np
from io import BytesIO


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
rescale_scores = json.load(open(os.path.join(os.path.dirname(__file__), '../../../data/rescale_scores.json')))
class PHALP(nn.Module):
    """
    PHALP: A class for object tracking and appearance feature extraction using DINOv2.

    Attributes:
        output_dir (str): Path to save extracted features.
        data_path (str): Path to the dataset containing poses and masks.
        frames_path (str): Path to the folder containing video frames.
        kitchen (str): Identifier for the specific kitchen or dataset instance.
    """
    def __init__(self, output_dir, data_path, frames_path, kitchen):
        super(PHALP, self).__init__()

        # Initialize class attributes
        self.RGB_tuples = get_colors()  # Predefined RGB tuples for visualization
        self.path_to_save = output_dir
        self.kitchen = kitchen
        self.output_dir_name = f"saved_feat_2D/{self.kitchen}"
        self.path_to_save = os.path.join(self.path_to_save, self.output_dir_name)

        # Create the output directory if it does not exist
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)

        self.data_path = data_path
        self.frames_path = frames_path

        # Load poses from JSON file
        with open(os.path.join(self.data_path, 'poses.json'), 'r') as f:
            self.poses = json.load(f)

        # Read data including masks, camera poses, and frames
        self.masks, _, self.camera_poses, self.frames, _ = read_data_all(
            self.data_path, kitchen, True
        )

        # Extract object bounding boxes from annotations
        self.bbs_dict = get_object_bbs_new(self.masks['video_annotations'])

        # Select the appropriate device (GPU or CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Device: ', self.device)

        # Load the DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        self.model = self.model.to(self.device)

        # Define image transformation pipeline
        self.transform = T.Compose([
            T.Resize((224, 224)),  # Resize to fixed size
            T.ToTensor(),  # Convert to tensor
            T.Normalize(  # Normalize using ImageNet stats
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Move the entire model to the selected device
        self.to(self.device)

        # Set the model to evaluation mode
        self.eval()

    def read_image_from_tar(self, tar_file_path, file_name):
        """
        Reads an image file from a .tar archive and converts it to OpenCV format (NumPy array).
        
        Parameters:
            tar_file_path (str): Path to the .tar archive.
            file_name (str): Name of the image file inside the tar archive to read.
        
        Returns:
            numpy.ndarray: The image as a NumPy array in OpenCV format.
        """
        ### gimage
            #  .tar -> images 
            #  .tar -> folder->images

        # with tarfile.open(tar_file_path, "r") as tar:
        #     file_names = tar.getnames()  # List all files inside the tar
        #     breakpoint()

        # breakpoint()
        try:
            # Open the tar file in read mode
            with tarfile.open(tar_file_path, "r") as tar:
                # Try to extract the specific image file
                file = tar.extractfile(f"./{file_name}")
             
                
                if file:
                    # Read the image content as bytes
                    image_bytes = file.read()
                    
                    # Convert the bytes to a NumPy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    
                    # Decode the NumPy array into an OpenCV image
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Check if the image was successfully decoded
                    if img is None:
                        print(f"Failed to decode the image {file_name}.")
                        return None
                    return img
                else:
                    print(f"File {file_name} not found inside the tar archive.")
                    return None
        except KeyError:
            print(f"The file {file_name} does not exist inside the tar archive.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def track(self):
        """
        Track objects across frames and extract 2D appearance features.

        Saves features and corresponding metadata into a pickle file.
        """
        save_dict_2D = {}

        # Set model to evaluation mode
        self.eval()

        file_path = os.path.join(self.path_to_save, f"2D_feat_{self.kitchen}.pkl")
    
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            return 
       


        # Initialize temporary storage for batch processing
        batched_bbs = []  # Batched bounding boxes
        frames_bbs = []  # Corresponding image frames
        frame_names = []  # Frame names
        batched_objs = []  # Associated object identifiers


        # Iterate over sorted frame names
        for t_, frame_name in tqdm(enumerate(sorted(self.frames)), total=len(self.frames)):
            # Get bounding boxes and object IDs for the frame
            bbs, objs = self.bbs_dict[f"{self.kitchen}_{frame_name}"]

            # Skip frames with no detected objects
            if len(bbs) == 0:
                continue

            # Read and preprocess the image frame
            image_frame = self.read_image_from_tar(self.frames_path, f"{frame_name}.jpg")
            # breakpoint()
            # cv2.imread(os.path.join(self.frames_path, f"{frame_name}.jpg"))
            image_frame = cv2.resize(image_frame, (854, 480))
            image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
            if not isinstance(image_frame, Image.Image):
                image_frame = Image.fromarray(image_frame)

            # Accumulate data for batch processing
            if len(bbs) > 0:
                batched_bbs.extend(bbs)
                batched_objs.extend(objs)
                frames_bbs.extend([image_frame] * len(bbs))
                frame_names.extend([frame_name] * len(bbs))

            # Process batches of bounding boxes
            if len(batched_bbs) >= 100:
                appe_features = extract_dino_features_batch(
                    frames_bbs, batched_bbs, self.model, self.device
                )
                j = 0
                unique_elements_ordered = []

                # Retain unique frame names in order of appearance
                for item in frame_names:
                    if item not in unique_elements_ordered:
                        unique_elements_ordered.append(item)
                
                # Map features to frames
                for i, s in enumerate(unique_elements_ordered):
                    if frame_names.count(s) == len(self.bbs_dict[f"{self.kitchen}_{s}"][0]):
                        save_dict_2D[s] = appe_features[j:j + frame_names.count(s)].cpu().numpy()
                        j += frame_names.count(s)

                # Clear the temporary storage
                batched_bbs = []
                batched_objs = []
                frames_bbs = []
                frame_names = []

        # Handle any remaining bounding boxes in the last batch
        if len(batched_bbs) > 0:
            appe_features = extract_dino_features_batch(
                frames_bbs, batched_bbs, self.model, self.device
            )
            j = 0
            unique_elements_ordered = []

            # Retain unique frame names in order of appearance
            for item in frame_names:
                if item not in unique_elements_ordered:
                    unique_elements_ordered.append(item)

            # Map features to frames
            for i, s in enumerate(unique_elements_ordered):
                if frame_names.count(s) == len(self.bbs_dict[f"{self.kitchen}_{s}"][0]):
                    save_dict_2D[s] = appe_features[j:j + frame_names.count(s)].cpu().numpy()
                    j += frame_names.count(s)

        # Save the extracted features and metadata to a file
        with open(os.path.join(self.path_to_save, f"2D_feat_{self.kitchen}.pkl"), 'wb') as f:
            pickle.dump(save_dict_2D, f)


def main():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--output_dir", required=True, help="Output directory path")
    parser.add_argument("--data_path", required=True, help="Data path")
    parser.add_argument("--frames_path", required=True, help="Frames path")
    parser.add_argument("--kitchen", required=True, help="Frames path")

    args = parser.parse_args()

    # Initialize your class with the configuration file argument
    phalp_instance = PHALP(args.output_dir, args.data_path, args.frames_path,
                           args.kitchen)
    phalp_instance.track()


if __name__ == "__main__":
    main()
